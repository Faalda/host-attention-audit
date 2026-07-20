"""
run_slot_cap.py — Slot Attention plans (Idea 3)

Motivation: run_cap_v2.py (value-augmented A₀@V plans) scored 8.5% mean
target-noun rate — KILL. Near-semantic pairs (dog→cat: 18.2%) responded but
cross-category pairs (pizza→sandwich: 0%) didn't. Content is present but
unstructured — no single vector owns a concept. Slot Attention fixes this by
having K learnable slots that compete to bind one object each.

Architecture:
  X [B, N=256, D=256]  →  SlotAttention (K=8, 3 iters)  →  S [B, K=8, D=256]
  S  →  plan (the latent variable)
  [X; S]  →  SlotDecoder  →  caption

Phases:
  1. SlotAttention + SlotDecoder + ClassHead trained jointly (concept-supervised)
  2. SlotDenoiser trained on masked oracle slots (MSE)
  3. SlotDecoder fine-tuned on mixed oracle/denoised slots (curriculum)
  4. Slot-level centroid arithmetic evaluation

Kill criterion: mean target-noun rate < 0.15
Pass criterion: mean target-noun rate > 0.30
Baseline ca_out: 0.085, baseline A₀: 0.038

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 \\
        run_slot_cap.py > results/slot_cap.log 2>&1 &
"""

import os, sys, re, json, random
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(__file__))
from run_summarization import (
    Tokenizer, DecoderLayer,
    DEVICE, D_MODEL, N_HEADS, N_LAYERS, L_SUM as L_CAP, BATCH, LR, N_MP,
)
from run_captioning import PatchProjector, N_CAP_PER
from run_cap_improved import (
    ENCODER, N_PATCHES, ENCODER_DIM,
    N_IMAGES, VAL_IMAGES, TEST_IMAGES,
    EPOCHS_FREE, EPOCHS_DEN, EPOCHS_GUIDED,
    RESULTS_DIR, CACHE_PATH,
    extract_patch_features,
)

K_SLOTS        = 8
SLOT_ITERS     = 3
CONCEPT_LAMBDA = 0.1
TOP_N_NOUNS    = 20
CFG_DROP_P     = 0.15
CFG_SCALE      = 2.0
SEED           = 42
EVAL_PAIRS     = [
    ("train", "car"), ("dog", "cat"), ("chair", "couch"),
    ("pizza", "sandwich"), ("horse", "cow"),
]
SAVE_DIR = os.path.join(RESULTS_DIR, "saved_plans_slot")

COCO_NOUNS = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis",
    "snowboard","kite","bat","glove","skateboard","surfboard","bottle","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli",
    "carrot","pizza","donut","cake","chair","couch","bed","toilet","monitor",
    "laptop","mouse","keyboard","phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy","toothbrush",
    "cellphone","cell","racket","racquet","hotdog","hot dog",
}

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_OK = True
except Exception:
    SPACY_OK = False


def extract_dominant_noun(caption):
    cap_lower = caption.lower()
    if SPACY_OK:
        doc = nlp(cap_lower)
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN") and token.lemma_ in COCO_NOUNS:
                return token.lemma_
    for word in cap_lower.split():
        w = re.sub(r"[^a-z]", "", word)
        if w in COCO_NOUNS:
            return w
    return None


def noun_in_caption(noun, caption):
    words = [re.sub(r"[^a-z]", "", w) for w in caption.lower().split()]
    return noun in words


# ── SlotAttention ────────────────────────────────────────────────────────────

class SlotAttention(nn.Module):
    """
    Locatello et al. 2020 — competitive slot attention.
    Normalises over slots (dim=1) so each patch is explained by at most one slot.
    """
    def __init__(self, K, D, n_iters=SLOT_ITERS):
        super().__init__()
        self.K = K
        self.D = D
        self.n_iters = n_iters

        self.slot_mu        = nn.Parameter(torch.randn(1, K, D))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, K, D))

        self.norm_input = nn.LayerNorm(D)
        self.norm_slots = nn.LayerNorm(D)
        self.norm_ff    = nn.LayerNorm(D)

        self.W_q = nn.Linear(D, D, bias=False)
        self.W_k = nn.Linear(D, D, bias=False)
        self.W_v = nn.Linear(D, D, bias=False)

        self.gru = nn.GRUCell(D, D)
        self.ff  = nn.Sequential(nn.Linear(D, D * 4), nn.GELU(), nn.Linear(D * 4, D))

    def forward(self, X):
        """X: [B, N, D]  →  slots: [B, K, D]"""
        B, N, D = X.shape

        sigma = self.slot_log_sigma.exp().expand(B, -1, -1)
        slots = self.slot_mu.expand(B, -1, -1) + sigma * torch.randn_like(sigma)

        X_norm = self.norm_input(X)
        k_proj = self.W_k(X_norm)   # [B, N, D]
        v_proj = self.W_v(X_norm)   # [B, N, D]

        for _ in range(self.n_iters):
            prev  = slots
            q     = self.W_q(self.norm_slots(slots))          # [B, K, D]
            # [B, K, N] attention logits
            attn  = torch.einsum('bkd,bnd->bkn', q, k_proj) * (D ** -0.5)
            # CRITICAL: softmax over K — slots compete for each patch
            attn  = attn.softmax(dim=1)                        # [B, K, N], sum_k = 1
            # Normalise over N for weighted mean
            attn_w = attn / (attn.sum(dim=2, keepdim=True) + 1e-8)   # [B, K, N]
            updates = torch.einsum('bkn,bnd->bkd', attn_w, v_proj)    # [B, K, D]

            # GRU update per slot
            slots = self.gru(
                updates.reshape(B * self.K, D),
                prev.reshape(B * self.K, D)
            ).reshape(B, self.K, D)

            slots = slots + self.ff(self.norm_ff(slots))

        return slots   # [B, K, D]


# ── SlotDecoder ──────────────────────────────────────────────────────────────

class SlotDecoder(nn.Module):
    """
    Transformer decoder that attends over mem = cat([X, S], dim=1).
    Works for both Phase 1 (oracle slots) and Phase 3 (denoised slots).
    """
    def __init__(self, vocab_size, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS):
        super().__init__()
        self.emb    = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos    = nn.Embedding(L_CAP + 2, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads) for _ in range(n_layers)])
        self.ln     = nn.LayerNorm(d_model)
        self.head   = nn.Linear(d_model, vocab_size)

    def forward(self, X, slots, tgt_ids):
        """
        X:       [B, N, D]
        slots:   [B, K, D]
        tgt_ids: [B, L]   (BOS-prefixed caption tokens)
        returns: logits [B, L, vocab]
        """
        mem = torch.cat([X, slots], dim=1)         # [B, N+K, D]
        B, L = tgt_ids.shape
        pos_idx = torch.arange(L, device=tgt_ids.device)
        x = self.emb(tgt_ids) + self.pos(pos_idx)
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        for layer in self.layers:
            x = layer(x, mem, tgt_mask=causal_mask)
        return self.head(self.ln(x))               # [B, L, vocab]

    @torch.no_grad()
    def generate(self, X, slots, max_len=L_CAP):
        """Greedy decode. Returns [B, max_len] token ids."""
        B = X.size(0)
        ids = torch.full((B, 1), Tokenizer.BOS, device=X.device, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool, device=X.device)
        for _ in range(max_len - 1):
            logits = self.forward(X, slots, ids)   # [B, cur_len, V]
            next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)  # [B, 1]
            ids = torch.cat([ids, next_tok], dim=1)
            done |= (next_tok.squeeze(1) == Tokenizer.EOS)
            if done.all():
                break
        return ids[:, 1:]   # strip BOS


# ── ClassHead ────────────────────────────────────────────────────────────────

class ClassHead(nn.Module):
    """Pool over slots → noun class logits. Used for concept-supervised loss."""
    def __init__(self, d_model, n_classes):
        super().__init__()
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, slots):
        """slots: [B, K, D]  →  [B, n_classes]"""
        return self.fc(slots.mean(dim=1))

    def per_slot_logits(self, slots):
        """slots: [B, K, D]  →  [B, K, n_classes]"""
        return self.fc(slots)


# ── SlotDenoiser ─────────────────────────────────────────────────────────────

class SlotDenoiser(nn.Module):
    """
    Self-attention over slots + cross-attention to image features.
    Predicts masked slot vectors (MSE regression).
    Plan is [B, K, D] — much smaller than prior plans [B, L, N].
    """
    def __init__(self, K=K_SLOTS, D=D_MODEL, n_heads=N_HEADS, n_mp=N_MP):
        super().__init__()
        self.null_X  = nn.Parameter(torch.zeros(1, N_PATCHES, D))
        self.proj_in = nn.Linear(D, D)
        self.blocks  = nn.ModuleList()
        for _ in range(n_mp):
            self.blocks.append(nn.ModuleList([
                nn.MultiheadAttention(D, n_heads, batch_first=True),   # self-attn
                nn.MultiheadAttention(D, n_heads, batch_first=True),   # cross-attn
                nn.Sequential(nn.Linear(D, D * 4), nn.GELU(), nn.Linear(D * 4, D)),
                nn.LayerNorm(D), nn.LayerNorm(D), nn.LayerNorm(D),
            ]))
        self.head = nn.Linear(D, D)

    def forward(self, S_noisy, X_cond, drop_X=False):
        """
        S_noisy: [B, K, D]  (masked slots zeroed out)
        X_cond:  [B, N, D]
        drop_X:  CFG null — replace X with learned null
        """
        B = S_noisy.size(0)
        mem = self.null_X.expand(B, -1, -1) if drop_X else X_cond
        h = self.proj_in(S_noisy)
        for sa, ca, ff, ln1, ln2, ln3 in self.blocks:
            h2, _ = sa(ln1(h), ln1(h), ln1(h))
            h = h + h2
            h2, _ = ca(ln2(h), mem, mem)
            h = h + h2
            h = h + ff(ln3(h))
        return self.head(h)   # [B, K, D]

    @torch.no_grad()
    def cfg_predict(self, X, S_noisy, mask):
        """CFG-blended prediction for a batch."""
        pred_cond  = self.forward(S_noisy, X, drop_X=False)
        pred_uncond = self.forward(S_noisy, X, drop_X=True)
        pred = pred_uncond + CFG_SCALE * (pred_cond - pred_uncond)
        # Fill only masked slots
        out = S_noisy.clone()
        out[mask] = pred[mask]
        return out


# ── Datasets ─────────────────────────────────────────────────────────────────

class CocoCapDatasetSlot(Dataset):
    def __init__(self, patch_feats, captions_per_img, tok):
        self.patch_feats = patch_feats
        self.pairs = []
        for img_idx, caps in enumerate(captions_per_img):
            for cap in caps:
                ids  = tok.enc(cap, L_CAP, bos=True, eos=True)
                noun = extract_dominant_noun(cap)
                self.pairs.append((img_idx, torch.tensor(ids, dtype=torch.long), cap, noun))

    def __len__(self): return len(self.pairs)

    def __getitem__(self, i):
        img_idx, ids, cap, noun = self.pairs[i]
        return {"patches": self.patch_feats[img_idx], "cap_ids": ids, "ref": cap, "noun": noun}


class SlotPlanDataset(Dataset):
    """Stores (slot_plan [K,D], X [N,D], cap_ids [L], noun_label_idx)."""
    def __init__(self, slots, X, cap_ids, noun_label_ids):
        self.slots = slots
        self.X = X
        self.cap_ids = cap_ids
        self.noun_label_ids = noun_label_ids

    def __len__(self): return len(self.slots)

    def __getitem__(self, i):
        return self.slots[i], self.X[i], self.cap_ids[i], self.noun_label_ids[i]


def make_collate_slot(noun_to_idx):
    def _collate(batch):
        patches  = torch.stack([b["patches"] for b in batch])
        cap_ids  = torch.stack([b["cap_ids"]  for b in batch])
        refs     = [b["ref"]  for b in batch]
        noun_idx = torch.tensor(
            [noun_to_idx.get(b["noun"], -1) for b in batch], dtype=torch.long
        )
        return {"patches": patches, "cap_ids": cap_ids, "ref": refs, "noun_idx": noun_idx}
    return _collate


# ── Phase 1: joint training ───────────────────────────────────────────────────

def train_slot_phase1(proj, slot_attn, decoder, class_head, loader, n_epochs, device):
    """Train SlotAttention + SlotDecoder + ClassHead jointly with concept loss."""
    params = (list(proj.parameters()) + list(slot_attn.parameters()) +
              list(decoder.parameters()) + list(class_head.parameters()))
    opt = torch.optim.Adam(params, lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)

    for ep in range(1, n_epochs + 1):
        proj.train(); slot_attn.train(); decoder.train(); class_head.train()
        tot_cap = tot_cls = 0.0
        n = 0
        for batch in loader:
            patches  = batch["patches"].to(device)
            ids      = batch["cap_ids"].to(device)
            noun_idx = batch["noun_idx"].to(device)

            X = proj(patches)               # [B, N, D]
            S = slot_attn(X)                # [B, K, D]

            logits = decoder(X, S, ids[:, :-1])   # [B, L-1, V]
            loss_cap = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                ids[:, 1:].reshape(-1), ignore_index=0
            )

            labeled = (noun_idx >= 0)
            loss_cls = torch.tensor(0.0, device=device)
            if labeled.any():
                cls_logits = class_head(S[labeled])   # [M, n_classes]
                loss_cls   = F.cross_entropy(cls_logits, noun_idx[labeled])
                tot_cls   += loss_cls.item()

            loss = loss_cap + CONCEPT_LAMBDA * loss_cls
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot_cap += loss_cap.item(); n += 1

        sch.step()
        if ep % 10 == 0:
            cls_str = f"  cls_loss={tot_cls/n:.4f}" if tot_cls > 0 else ""
            print(f"    ep {ep:3d}  cap_loss={tot_cap/n:.4f}{cls_str}", flush=True)


# ── Extract oracle slots ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_slots(proj, slot_attn, loader, device):
    """Extract oracle slot plans S [N_total, K, D] for a dataset."""
    proj.eval(); slot_attn.eval()
    all_S, all_X, all_refs, all_ids = [], [], [], []
    for batch in loader:
        patches = batch["patches"].to(device)
        ids     = batch["cap_ids"].to(device)
        X = proj(patches)
        S = slot_attn(X)
        all_S.append(S.cpu())
        all_X.append(X.cpu())
        all_refs += batch["ref"]
        all_ids.append(ids.cpu())
    return torch.cat(all_S), torch.cat(all_X), all_refs, torch.cat(all_ids)


# ── Phase 2: train SlotDenoiser ──────────────────────────────────────────────

def train_slot_denoiser(denoiser, slot_ds, n_epochs, device):
    """MSE regression on randomly masked slot vectors."""
    loader = DataLoader(slot_ds, BATCH, shuffle=True)
    opt = torch.optim.Adam(denoiser.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)

    for ep in range(1, n_epochs + 1):
        denoiser.train()
        tot = 0.0; n = 0
        for S, X, _, _ in loader:
            S = S.to(device); X = X.to(device)
            B = S.size(0)

            # Mask random fraction of the K slots
            mask_rate = random.uniform(0.1, 0.95)
            mask = torch.rand(B, K_SLOTS, device=device) < mask_rate   # [B, K] bool
            if not mask.any():
                continue

            S_noisy = S.clone()
            S_noisy[mask] = 0.0

            drop_X = random.random() < CFG_DROP_P
            pred   = denoiser(S_noisy, X, drop_X=drop_X)

            loss = F.mse_loss(pred[mask], S[mask])
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1

        sch.step()
        if ep % 10 == 0:
            s = f"{tot/n:.4f}" if n > 0 else "nan"
            print(f"    ep {ep:3d}  slot_mse={s}", flush=True)


# ── Phase 3: guided decoder fine-tune ────────────────────────────────────────

def train_slot_guided(decoder, slot_ds, denoiser, n_epochs, device, oracle_frac=0.5):
    """Fine-tune SlotDecoder with mixed oracle/denoised slots."""
    loader = DataLoader(slot_ds, BATCH, shuffle=True)
    opt = torch.optim.Adam(decoder.parameters(), lr=LR * 0.3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)

    for ep in range(1, n_epochs + 1):
        decoder.train(); denoiser.eval()
        tot = 0.0; n = 0
        for S, X, ids, _ in loader:
            S = S.to(device); X = X.to(device); ids = ids.to(device)

            if random.random() > oracle_frac:
                with torch.no_grad():
                    mask_rate = random.uniform(0.3, 0.9)
                    B = S.size(0)
                    mask = torch.rand(B, K_SLOTS, device=device) < mask_rate
                    S_noisy = S.clone(); S_noisy[mask] = 0.0
                    S_used = denoiser.cfg_predict(X, S_noisy, mask)
            else:
                S_used = S

            logits = decoder(X, S_used, ids[:, :-1])
            loss   = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                ids[:, 1:].reshape(-1), ignore_index=0
            )
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1

        sch.step()
        if ep % 10 == 0:
            print(f"    ep {ep:3d}  guided_slot_loss={tot/n:.4f}", flush=True)


# ── MaskGIT-style slot generation ────────────────────────────────────────────

@torch.no_grad()
def generate_slots_denoised(denoiser, X, n_steps=4, device=None):
    """
    Iteratively unmask slots: start all-masked, commit most-confident each step.
    Returns S_hat [B, K, D].
    """
    if device is None:
        device = X.device
    B = X.size(0)
    S = torch.zeros(B, K_SLOTS, D_MODEL, device=device)
    unmasked = torch.zeros(B, K_SLOTS, dtype=torch.bool, device=device)

    n_per_step = max(1, K_SLOTS // n_steps)

    for step in range(n_steps):
        pred = denoiser.cfg_predict(X, S, ~unmasked)

        # Confidence = predicted vector norm (larger = more confident)
        confidence = pred.norm(dim=-1)   # [B, K]
        confidence[unmasked] = -1e9      # don't re-unmask

        # Unmask top n_per_step slots per batch element
        n_unmask = n_per_step if step < n_steps - 1 else K_SLOTS  # last step: all remaining
        topk = confidence.topk(min(n_unmask, (~unmasked).sum(dim=1).min().item() + 1),
                                dim=1).indices
        for b in range(B):
            unmasked[b, topk[b]] = True
            S[b, topk[b]] = pred[b, topk[b]]

    return S   # [B, K, D]


# ── Generate captions ─────────────────────────────────────────────────────────

@torch.no_grad()
def generate_slot_captions(proj, slot_attn, denoiser, decoder, tok, feats,
                            batch_size=32, device=DEVICE):
    proj.eval(); slot_attn.eval(); denoiser.eval(); decoder.eval()
    all_caps = []
    for i in range(0, len(feats), batch_size):
        X_raw = feats[i:i + batch_size].to(device)
        X     = proj(X_raw)
        S     = generate_slots_denoised(denoiser, X, n_steps=4, device=device)
        toks  = decoder.generate(X, S)
        all_caps += [tok.dec(t.tolist()) for t in toks.cpu()]
        if (i // batch_size) % 20 == 0:
            print(f"    {i + X_raw.size(0)}/{len(feats)}", flush=True)
    return all_caps


# ── Arithmetic evaluation ──────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_slot_arithmetic(proj, slot_attn, denoiser, decoder, class_head, tok,
                              tv_feats, tv_noun_labels, tv_refs,
                              test_feats, test_caps_per_img,
                              n_classes, noun_to_idx, eval_pairs, device):
    """
    1. Extract oracle slots for all train+val images (one slot set per image).
    2. For each noun c, find k*(c) = the slot index most often highest-scoring for c
       according to ClassHead across training images containing c.
    3. Centroid[c] = mean S[:, k*(c)] over tv images with noun c.
    4. For each test image with src_noun:
         S_edit[:, k*(src)] = S_free[:, k*(src)] - centroid[src] + centroid[tgt]
         caption = decoder.generate(X, S_edit)
         check tgt_noun in caption
    """
    proj.eval(); slot_attn.eval(); denoiser.eval(); decoder.eval(); class_head.eval()

    # ── Extract one oracle slot set per tv image ──────────────────────────────
    print("  Extracting oracle slot plans for train+val images...", flush=True)
    all_S_tv = []
    bs = BATCH
    n_tv = len(tv_feats)
    for i in range(0, n_tv, bs):
        X = proj(tv_feats[i:i+bs].to(device))
        S = slot_attn(X)
        all_S_tv.append(S.cpu())
    S_tv = torch.cat(all_S_tv)   # [n_tv, K, D]

    # ── For each noun, find k*(c) = dominant slot index ───────────────────────
    # Per-slot class logits for each tv image: [n_tv, K, n_classes]
    per_slot_logits_list = []
    for i in range(0, n_tv, bs):
        S_batch = S_tv[i:i+bs].to(device)
        per_slot_logits_list.append(class_head.per_slot_logits(S_batch).cpu())
    per_slot_logits = torch.cat(per_slot_logits_list)   # [n_tv, K, n_classes]

    idx_to_noun = {v: k for k, v in noun_to_idx.items()}
    noun_kstar = {}   # noun → dominant slot index k*

    for noun, cls_idx in noun_to_idx.items():
        img_mask = [tv_noun_labels[i] == noun for i in range(n_tv)]
        img_indices = [i for i, m in enumerate(img_mask) if m]
        if len(img_indices) < 5:
            continue
        # For each image with this noun, which slot scores highest on this class?
        scores = per_slot_logits[img_indices, :, cls_idx]   # [n, K]
        best_slot_per_img = scores.argmax(dim=1)            # [n]
        counts = Counter(best_slot_per_img.tolist())
        noun_kstar[noun] = counts.most_common(1)[0][0]

    # ── Build centroids ───────────────────────────────────────────────────────
    centroids = {}
    for noun, k_star in noun_kstar.items():
        img_mask = [tv_noun_labels[i] == noun for i in range(n_tv)]
        img_indices = [i for i, m in enumerate(img_mask) if m]
        centroids[noun] = S_tv[img_indices, k_star].mean(0)  # [D]

    print(f"  Centroids built for {len(centroids)} nouns", flush=True)
    print(f"  Slot assignments: " +
          ", ".join(f"{n}→k{k}" for n, k in sorted(noun_kstar.items())[:8]), flush=True)

    # ── Extract oracle slots for test images ──────────────────────────────────
    n_test = len(test_feats)
    all_S_test = []
    all_X_test = []
    for i in range(0, n_test, bs):
        X = proj(test_feats[i:i+bs].to(device))
        S = slot_attn(X)
        all_S_test.append(S.cpu())
        all_X_test.append(X.cpu())
    S_test = torch.cat(all_S_test)   # [n_test, K, D]
    X_test = torch.cat(all_X_test)   # [n_test, N, D]

    # ── Evaluate arithmetic ───────────────────────────────────────────────────
    results = {}
    for src_noun, tgt_noun in eval_pairs:
        if src_noun not in centroids or tgt_noun not in centroids:
            print(f"  {src_noun}→{tgt_noun}: missing centroid, skip", flush=True)
            results[f"{src_noun}→{tgt_noun}"] = {"skip": True}
            continue

        c_src  = centroids[src_noun].to(device)
        c_tgt  = centroids[tgt_noun].to(device)
        k_src  = noun_kstar[src_noun]

        # Find test images whose first caption's dominant noun is src_noun
        test_indices = []
        for i, caps in enumerate(test_caps_per_img):
            if caps and extract_dominant_noun(caps[0]) == src_noun:
                test_indices.append(i)

        if len(test_indices) == 0:
            print(f"  {src_noun}→{tgt_noun}: no test images, skip", flush=True)
            results[f"{src_noun}→{tgt_noun}"] = {"skip": True}
            continue

        tgt_appeared = 0
        for idx in test_indices:
            S_i = S_test[idx].unsqueeze(0).to(device)   # [1, K, D]
            X_i = X_test[idx].unsqueeze(0).to(device)   # [1, N, D]

            # Slot-level arithmetic: edit only k_src slot
            S_edit = S_i.clone()
            S_edit[0, k_src] = S_edit[0, k_src] - c_src + c_tgt

            # Decode
            toks = decoder.generate(X_i, S_edit)
            cap  = tok.dec(toks[0].tolist())

            if noun_in_caption(tgt_noun, cap):
                tgt_appeared += 1

        rate = tgt_appeared / len(test_indices)
        print(f"  {src_noun}→{tgt_noun}: n={len(test_indices)}  target={tgt_appeared}  rate={rate:.3f}",
              flush=True)
        results[f"{src_noun}→{tgt_noun}"] = {
            "n_test": len(test_indices),
            "target_appeared": tgt_appeared,
            "rate": rate,
        }

    valid = [v["rate"] for v in results.values() if "rate" in v]
    mean_rate = sum(valid) / len(valid) if valid else 0.0

    if mean_rate < 0.15:
        verdict = "KILL"
    elif mean_rate > 0.30:
        verdict = "PASS"
    else:
        verdict = "MARGINAL"

    return results, mean_rate, verdict


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Load patch features (cached) ─────────────────────────────────────────
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached features from {CACHE_PATH}...", flush=True)
        patch_feats = torch.load(CACHE_PATH, weights_only=True)
    else:
        print("Extracting patch features (DINOv2)...", flush=True)
        from datasets import load_dataset as _lds
        ds    = _lds("clip-benchmark/wds_mscoco_captions", split="train")
        total = N_IMAGES + VAL_IMAGES + TEST_IMAGES
        imgs  = []
        for item in ds.select(range(total)):
            img = item["jpg"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            imgs.append(img)
        patch_feats = extract_patch_features(imgs, encoder=ENCODER, device=DEVICE)
        torch.save(patch_feats, CACHE_PATH)
    print(f"  features: {patch_feats.shape}", flush=True)

    # ── Load captions ─────────────────────────────────────────────────────────
    total = N_IMAGES + VAL_IMAGES + TEST_IMAGES
    print("Loading MSCOCO captions...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("clip-benchmark/wds_mscoco_captions", split="train")
    caps_per_img = []
    for item in ds.select(range(total)):
        caps = [c.strip() for c in item["txt"].strip().split("\n") if c.strip()]
        caps_per_img.append(caps[:N_CAP_PER])

    tok = Tokenizer()
    tok.build([c for caps in caps_per_img for c in caps], max_vocab=6000)
    print(f"  Vocab: {len(tok.vocab)}", flush=True)

    # ── Build noun index ─────────────────────────────────────────────────────
    n_tv      = N_IMAGES + VAL_IMAGES
    tv_flat   = [c for caps in caps_per_img[:n_tv] for c in caps]
    tv_nouns  = [extract_dominant_noun(c) for c in tv_flat]
    noun_cnt  = Counter(n for n in tv_nouns if n)
    top_nouns = [n for n, _ in noun_cnt.most_common(TOP_N_NOUNS)]
    noun_to_idx = {n: i for i, n in enumerate(top_nouns)}
    n_classes   = len(top_nouns)
    print(f"  Top-{n_classes} nouns: {top_nouns}", flush=True)

    # Per-image noun label for tv (dominant noun from first caption)
    tv_noun_labels = []
    for caps in caps_per_img[:n_tv]:
        noun = None
        for cap in caps:
            noun = extract_dominant_noun(cap)
            if noun:
                break
        tv_noun_labels.append(noun)

    # ── Feature splits ────────────────────────────────────────────────────────
    train_feats = patch_feats[:N_IMAGES]
    tv_feats    = patch_feats[:n_tv]
    test_feats  = patch_feats[n_tv: n_tv + TEST_IMAGES]
    train_caps  = caps_per_img[:N_IMAGES]
    tv_caps     = caps_per_img[:n_tv]
    test_caps   = caps_per_img[n_tv: n_tv + TEST_IMAGES]

    # ── Build models ──────────────────────────────────────────────────────────
    proj       = PatchProjector(ENCODER_DIM, D_MODEL).to(DEVICE)
    slot_attn  = SlotAttention(K_SLOTS, D_MODEL, SLOT_ITERS).to(DEVICE)
    decoder    = SlotDecoder(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)
    class_head = ClassHead(D_MODEL, n_classes).to(DEVICE)
    denoiser   = SlotDenoiser(K_SLOTS, D_MODEL, N_HEADS, N_MP).to(DEVICE)

    print(f"  proj={sum(p.numel() for p in proj.parameters()):,}"
          f"  slot_attn={sum(p.numel() for p in slot_attn.parameters()):,}"
          f"  decoder={sum(p.numel() for p in decoder.parameters()):,}"
          f"  denoiser={sum(p.numel() for p in denoiser.parameters()):,}",
          flush=True)

    # ── Phase 1: joint slot + decoder + concept loss ──────────────────────────
    print(f"\nPhase 1: SlotAttention + SlotDecoder + ClassHead ({N_IMAGES} images)...",
          flush=True)
    collate_fn   = make_collate_slot(noun_to_idx)
    train_ds     = CocoCapDatasetSlot(train_feats, train_caps, tok)
    train_loader = DataLoader(train_ds, BATCH, shuffle=True, collate_fn=collate_fn)
    train_slot_phase1(proj, slot_attn, decoder, class_head, train_loader, EPOCHS_FREE, DEVICE)

    # ── Extract oracle slots for all train+val captions ───────────────────────
    print(f"\nExtracting oracle slots for train+val ({n_tv} images × {N_CAP_PER} caps)...",
          flush=True)
    tv_ds     = CocoCapDatasetSlot(tv_feats, tv_caps, tok)
    tv_loader = DataLoader(tv_ds, BATCH, shuffle=False, collate_fn=collate_fn)
    S_tv, X_tv, refs_tv, ids_tv = extract_slots(proj, slot_attn, tv_loader, DEVICE)
    print(f"  slots: {S_tv.shape}   X: {X_tv.shape}", flush=True)

    noun_label_ids_tv = torch.tensor(
        [noun_to_idx.get(extract_dominant_noun(c), -1) for c in refs_tv],
        dtype=torch.long
    )
    slot_ds = SlotPlanDataset(S_tv, X_tv, ids_tv, noun_label_ids_tv)

    # ── Phase 2: train SlotDenoiser ───────────────────────────────────────────
    print(f"\nPhase 2: SlotDenoiser (MSE on masked slots)...", flush=True)
    train_slot_denoiser(denoiser, slot_ds, EPOCHS_DEN, DEVICE)

    # ── Phase 3: guided decoder fine-tune ────────────────────────────────────
    print(f"\nPhase 3: SlotDecoder fine-tune on mixed oracle/denoised slots...",
          flush=True)
    train_slot_guided(decoder, slot_ds, denoiser, EPOCHS_GUIDED, DEVICE)

    # ── Save ─────────────────────────────────────────────────────────────────
    print(f"\nSaving to {SAVE_DIR}/...", flush=True)
    torch.save(S_tv,                       os.path.join(SAVE_DIR, "slots_tv.pt"))
    torch.save(proj.state_dict(),          os.path.join(SAVE_DIR, "model_proj.pt"))
    torch.save(slot_attn.state_dict(),     os.path.join(SAVE_DIR, "model_slot_attn.pt"))
    torch.save(decoder.state_dict(),       os.path.join(SAVE_DIR, "model_decoder.pt"))
    torch.save(class_head.state_dict(),    os.path.join(SAVE_DIR, "model_class_head.pt"))
    torch.save(denoiser.state_dict(),      os.path.join(SAVE_DIR, "model_denoiser.pt"))
    with open(os.path.join(SAVE_DIR, "tok_vocab.json"), "w") as f:
        json.dump(tok.vocab, f)
    with open(os.path.join(SAVE_DIR, "noun_labels_tv.json"), "w") as f:
        json.dump(tv_noun_labels, f)
    meta = {
        "plan_type": "slot_attention",
        "K_slots": K_SLOTS, "slot_iters": SLOT_ITERS,
        "plan_shape": [n_tv * N_CAP_PER, K_SLOTS, D_MODEL],
        "n_classes": n_classes, "top_nouns": top_nouns,
        "encoder": ENCODER, "n_patches": N_PATCHES,
        "d_model": D_MODEL, "l_cap": L_CAP,
    }
    with open(os.path.join(SAVE_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("  Saved.", flush=True)

    # ── Arithmetic evaluation ─────────────────────────────────────────────────
    print("\nSlot-level arithmetic evaluation...", flush=True)
    results, mean_rate, verdict = evaluate_slot_arithmetic(
        proj, slot_attn, denoiser, decoder, class_head, tok,
        tv_feats, tv_noun_labels, refs_tv,
        test_feats, test_caps,
        n_classes, noun_to_idx,
        EVAL_PAIRS, DEVICE,
    )

    # ── Generate test captions ────────────────────────────────────────────────
    print("\nGenerating test captions (denoised slots, first 200 images)...", flush=True)
    caps_gen = generate_slot_captions(
        proj, slot_attn, denoiser, decoder, tok,
        test_feats[:200], device=DEVICE
    )
    noun_dist = Counter(n for c in caps_gen for n in [extract_dominant_noun(c)] if n)
    print("Top nouns in generated captions:", flush=True)
    for n, c in noun_dist.most_common(10):
        print(f"  {n:15s}: {c}", flush=True)

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = {
        "arithmetic": results,
        "mean_target_noun_rate": mean_rate,
        "verdict": verdict,
        "noun_dist_test_200": dict(noun_dist.most_common(20)),
        "baseline_ca_out_rate": 0.085,
        "baseline_A0_rate": 0.038,
        "kill_criterion": 0.15,
        "pass_criterion": 0.30,
    }
    out_path = os.path.join(RESULTS_DIR, "slot_arithmetic.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"FINAL VERDICT: {verdict}", flush=True)
    print(f"Mean target-noun rate (slot arithmetic):  {mean_rate:.3f}", flush=True)
    print(f"Baseline ca_out rate:                     0.085", flush=True)
    print(f"Baseline A₀ rate:                         0.038", flush=True)
    print(f"Kill < 0.15 | Marginal 0.15-0.30 | Pass > 0.30", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"\nResults → {out_path}", flush=True)


if __name__ == "__main__":
    main()
