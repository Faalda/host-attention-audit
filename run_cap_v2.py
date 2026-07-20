"""
run_cap_v2.py — Value-augmented plans (Idea 1) + concept-supervised loss (Idea 2)

Changes vs the original pipeline:

  Idea 1 — Value-augmented plans:
    Plan = ca_out = A₀@V  [B, L_cap, D_model]  instead of A₀ [B, L_cap, N_patches]
    ca_out = cross-attention *output*, already computed in DecoderLayer but never stored.
    It carries "what" (value-weighted content) not just "where" (routing weights).
    Lucky coincidence: D_model = N_patches = 256 → denoiser row dimension unchanged.

  Idea 2 — Concept-supervised auxiliary loss (Phase 1):
    L_total = L_caption + 0.1 * CE(ClassHead(ca_out.mean(dim=1)), noun_label)
    Pushes the plan to be linearly decodable to the dominant COCO noun.

  Guided decoder: src_ext = cat([X_patches, plan], dim=1) — standard cross-attention
    over extended source (N_patches + L_CAP tokens).  No soft-gate needed.

  Denoiser: CFGDenoiserV2 — self-attn + cross-attn(plan→X), L2 regression on masked rows.
    MaskGIT-style inference: iterative unmasking by predicted-vector norm.

  Arithmetic test:
    P_edit = P_src − centroid(src_noun) + centroid(tgt_noun)   in ca_out space
    Kill criterion: mean target-noun rate < 0.10 → KILL  (baseline A₀ rate = 0.038)
    Pass criterion: mean target-noun rate > 0.20 → PASS

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 \\
        run_cap_v2.py > results/cap_v2.log 2>&1
"""

import os, sys, re, json, random
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from run_summarization import (
    Tokenizer, DEVICE, D_MODEL, N_HEADS, N_LAYERS, L_SUM as L_CAP, BATCH, LR, N_MP,
)
from run_captioning import (
    PatchProjector, N_CAP_PER,
)
from run_cap_improved import (
    ENCODER, N_PATCHES, ENCODER_DIM,
    N_IMAGES, VAL_IMAGES, TEST_IMAGES,
    EPOCHS_FREE, EPOCHS_DEN, EPOCHS_GUIDED,
    RESULTS_DIR, CACHE_PATH, extract_patch_features,
)

SAVE_DIR       = os.path.join(RESULTS_DIR, "saved_plans_v2")
SEED           = 42
CFG_SCALE      = 2.0
CFG_DROP_P     = 0.15
CONCEPT_LAMBDA = 0.1
TOP_N_NOUNS    = 20
EVAL_PAIRS     = [
    ("train", "car"), ("dog", "cat"), ("chair", "couch"),
    ("pizza", "sandwich"), ("horse", "cow"),
]

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


# ── V2 decoder layer: returns ca_out = A₀@V ─────────────────────────────────
class DecoderLayerV2(nn.Module):
    def __init__(self, D, H, dropout=0.1):
        super().__init__()
        self.sa   = nn.MultiheadAttention(D, H, dropout=dropout, batch_first=True)
        self.ca   = nn.MultiheadAttention(D, H, dropout=dropout, batch_first=True)
        self.ff   = nn.Sequential(nn.Linear(D, D * 4), nn.GELU(), nn.Linear(D * 4, D))
        self.ln1  = nn.LayerNorm(D)
        self.ln2  = nn.LayerNorm(D)
        self.ln3  = nn.LayerNorm(D)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mem, tgt_mask=None, tgt_pad=None, ret_ca_out=False):
        xn = self.ln1(x)
        sa_out, _ = self.sa(xn, xn, xn, attn_mask=tgt_mask, key_padding_mask=tgt_pad)
        x = x + self.drop(sa_out)
        xn = self.ln2(x)
        # ca_out = A₀ @ V — value-weighted attended content [B, L, D]
        ca_out, _ = self.ca(xn, mem, mem, need_weights=False)
        x = x + self.drop(ca_out)
        x = x + self.drop(self.ff(self.ln3(x)))
        return (x, ca_out) if ret_ca_out else x


class CaptionDecoderV2(nn.Module):
    """Free decoder extracting ca_out plan from last layer."""
    def __init__(self, V, D, H, NL, dropout=0.1):
        super().__init__()
        self.embed  = nn.Embedding(V, D)
        self.pos    = nn.Embedding(L_CAP + 2, D)
        self.layers = nn.ModuleList([DecoderLayerV2(D, H, dropout) for _ in range(NL)])
        self.head   = nn.Linear(D, V)
        self.NL     = NL

    def forward(self, src, tgt, ret_last_ca_out=False):
        B, L = tgt.shape
        x       = self.embed(tgt) + self.pos(torch.arange(L, device=tgt.device))
        causal  = nn.Transformer.generate_square_subsequent_mask(L, device=tgt.device)
        tgt_pad = (tgt == 0)
        ca_out  = None
        for i, layer in enumerate(self.layers):
            last = ret_last_ca_out and (i == self.NL - 1)
            out  = layer(x, src, tgt_mask=causal, tgt_pad=tgt_pad, ret_ca_out=last)
            if last:
                x, ca_out = out
            else:
                x = out
        logits = self.head(x)
        return (logits, ca_out) if ret_last_ca_out else logits

    @torch.no_grad()
    def generate(self, src, max_len=L_CAP, bos=1, eos=2):
        B = src.shape[0]
        toks = torch.full((B, 1), bos, device=src.device, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len - 1):
            logits = self.forward(src, toks)
            next_t = logits[:, -1].argmax(-1)
            toks   = torch.cat([toks, next_t.unsqueeze(1)], dim=1)
            done  |= (next_t == eos)
            if done.all(): break
        return toks


# ── Concept-supervised class head (Idea 2) ───────────────────────────────────
class ClassHead(nn.Module):
    def __init__(self, D, n_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, n_classes))

    def forward(self, x): return self.net(x)   # [B, D] → [B, n_classes]


# ── CFG denoiser for continuous plans ────────────────────────────────────────
class CFGDenoiserV2(nn.Module):
    """
    Denoiser for ca_out plans [B, L, D].
    plan_t → self-attn → cross-attn(plan→X) → FF → out_proj → predicted plan [B, L, D]
    Training loss: MSE on masked rows.
    D_model = N_patches = 256 so the architecture is dimension-compatible.
    """
    def __init__(self, D, n_src=N_PATCHES, n_tgt=L_CAP):
        super().__init__()
        self.plan_proj  = nn.Linear(D, D)
        self.mask_emb   = nn.Embedding(2, D)
        self.self_attn  = nn.MultiheadAttention(D, 4, batch_first=True)
        self.self_norm  = nn.LayerNorm(D)
        self.cross_attn = nn.MultiheadAttention(D, 4, batch_first=True)
        self.cross_norm = nn.LayerNorm(D)
        self.ff         = nn.Sequential(nn.Linear(D, D * 2), nn.GELU(), nn.Linear(D * 2, D))
        self.ff_norm    = nn.LayerNorm(D)
        self.out_proj   = nn.Linear(D, D)
        # Null image token for CFG unconditional path
        self.null_X     = nn.Parameter(torch.zeros(1, n_src, D))

    def _forward_once(self, X_src, plan_t, mask):
        h = F.gelu(self.plan_proj(plan_t)) + self.mask_emb(mask.long())
        h2, _ = self.self_attn(h, h, h, need_weights=False)
        h = self.self_norm(h + h2)
        h2, _ = self.cross_attn(h, X_src, X_src, need_weights=False)
        h = self.cross_norm(h + h2)
        h = self.ff_norm(h + self.ff(h))
        return self.out_proj(h)   # [B, L, D]

    def forward(self, X_src, plan_t, mask, drop_X=False):
        if drop_X:
            return self._forward_once(
                self.null_X.expand(X_src.shape[0], -1, -1), plan_t, mask
            )
        return self._forward_once(X_src, plan_t, mask)

    @torch.no_grad()
    def cfg_predict(self, X_src, plan_t, mask, scale=CFG_SCALE):
        cond   = self._forward_once(X_src, plan_t, mask)
        uncond = self._forward_once(
            self.null_X.expand(X_src.shape[0], -1, -1), plan_t, mask
        )
        return uncond + scale * (cond - uncond)


def mask_plan(plan, mask_rate):
    """Zero-mask rows of a continuous plan tensor."""
    B, L, D = plan.shape
    mask   = (torch.rand(B, L, device=plan.device) < mask_rate).float()
    plan_t = plan.clone()
    plan_t[mask.bool()] = 0.0
    return plan_t, mask


@torch.no_grad()
def generate_plan_v2(denoiser, X_src, n_steps=4, device=None):
    """MaskGIT-style iterative denoising for continuous ca_out plans."""
    if device is None: device = X_src.device
    B = X_src.shape[0]
    plan = torch.zeros(B, L_CAP, D_MODEL, device=device)
    mask = torch.ones(B, L_CAP, device=device)
    n_per_step = max(1, L_CAP // n_steps)

    for _ in range(n_steps):
        pred = denoiser.cfg_predict(X_src, plan, mask)
        # Unmask rows with highest predicted-vector norm as confidence proxy
        conf = pred.norm(dim=-1).masked_fill(~mask.bool(), -1e9)
        n_unmask = min(n_per_step, int(mask.sum(-1).min().item()))
        if n_unmask == 0: break
        to_unmask = conf.topk(n_unmask, dim=-1).indices
        mask_new  = mask.clone()
        mask_new.scatter_(1, to_unmask, 0.0)
        # Write predicted values into newly unmasked positions
        still_masked = mask_new.bool()
        new_unmasked = mask.bool() & ~still_masked
        plan = plan.clone()
        plan[new_unmasked] = pred[new_unmasked]
        mask = mask_new
        if not mask.any(): break

    # Final pass for any remaining masked rows
    if mask.any():
        final = denoiser.cfg_predict(X_src, plan, mask)
        plan  = plan.clone()
        plan[mask.bool()] = final[mask.bool()]
    return plan   # [B, L_CAP, D_MODEL]


# ── Guided decoder: extended source (patches + plan) ─────────────────────────
class GuidedDecoderV2(nn.Module):
    """
    Standard transformer decoder over src_ext = cat([X_patches, plan], dim=1).
    No soft-gate; the decoder learns to attend to plan tokens directly.
    """
    def __init__(self, V, D, H, NL, dropout=0.1):
        super().__init__()
        self.embed  = nn.Embedding(V, D)
        self.pos    = nn.Embedding(L_CAP + 2, D)
        self.layers = nn.ModuleList([DecoderLayerV2(D, H, dropout) for _ in range(NL)])
        self.head   = nn.Linear(D, V)
        self.NL     = NL

    def forward(self, src_ext, tgt):
        B, L = tgt.shape
        x       = self.embed(tgt) + self.pos(torch.arange(L, device=tgt.device))
        causal  = nn.Transformer.generate_square_subsequent_mask(L, device=tgt.device)
        tgt_pad = (tgt == 0)
        for layer in self.layers:
            x = layer(x, src_ext, tgt_mask=causal, tgt_pad=tgt_pad)
        return self.head(x)

    @torch.no_grad()
    def generate(self, src_ext, max_len=L_CAP, bos=1, eos=2):
        B = src_ext.shape[0]
        toks = torch.full((B, 1), bos, device=src_ext.device, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool, device=src_ext.device)
        for _ in range(max_len - 1):
            logits = self.forward(src_ext, toks)
            next_t = logits[:, -1].argmax(-1)
            toks   = torch.cat([toks, next_t.unsqueeze(1)], dim=1)
            done  |= (next_t == eos)
            if done.all(): break
        return toks


# ── Datasets ─────────────────────────────────────────────────────────────────
class CocoCapDatasetV2(Dataset):
    """(img_idx, cap_ids, ref, noun) per caption."""
    def __init__(self, patch_feats, captions_per_img, tok):
        self.patch_feats = patch_feats
        self.tok         = tok
        self.pairs       = []
        for img_idx, caps in enumerate(captions_per_img):
            for cap in caps:
                ids  = tok.enc(cap, L_CAP, bos=True, eos=True)
                noun = extract_dominant_noun(cap)
                self.pairs.append((img_idx, torch.tensor(ids, dtype=torch.long), cap, noun))

    def __len__(self): return len(self.pairs)

    def __getitem__(self, i):
        img_idx, ids, cap, noun = self.pairs[i]
        return {"patches": self.patch_feats[img_idx], "cap_ids": ids, "ref": cap, "noun": noun}


class PlanDatasetV2(Dataset):
    def __init__(self, plans, X, cap_ids, noun_label_ids):
        self.plans = plans; self.X = X
        self.cap_ids = cap_ids; self.noun_label_ids = noun_label_ids

    def __len__(self): return len(self.plans)

    def __getitem__(self, i):
        return self.plans[i], self.X[i], self.cap_ids[i], self.noun_label_ids[i]


def make_collate_with_noun(noun_to_idx):
    def _collate(batch):
        patches  = torch.stack([b["patches"] for b in batch])
        cap_ids  = torch.stack([b["cap_ids"]  for b in batch])
        refs     = [b["ref"]  for b in batch]
        noun_idx = torch.tensor(
            [noun_to_idx.get(b["noun"], -1) for b in batch], dtype=torch.long
        )
        return {"patches": patches, "cap_ids": cap_ids, "ref": refs, "noun_idx": noun_idx}
    return _collate


# ── Phase 1: free decoder + concept-supervised loss ──────────────────────────
def train_free_v2(proj, dec, class_head, loader, n_epochs, device):
    params = (list(proj.parameters()) + list(dec.parameters()) +
              list(class_head.parameters()))
    opt = torch.optim.Adam(params, lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)

    for ep in range(1, n_epochs + 1):
        proj.train(); dec.train(); class_head.train()
        tot_cap = tot_cls = 0.0; n = 0
        for batch in loader:
            patches  = batch["patches"].to(device)
            ids      = batch["cap_ids"].to(device)
            noun_idx = batch["noun_idx"].to(device)

            X = proj(patches)
            logits, ca_out = dec(X, ids[:, :-1], ret_last_ca_out=True)

            loss_cap = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                ids[:, 1:].reshape(-1), ignore_index=0
            )
            loss = loss_cap

            if ca_out is not None:
                labeled = (noun_idx >= 0)
                if labeled.any():
                    plan_mean  = ca_out[labeled].mean(dim=1)       # [M, D]
                    cls_logits = class_head(plan_mean)              # [M, n_classes]
                    loss_cls   = F.cross_entropy(cls_logits, noun_idx[labeled])
                    loss       = loss_cap + CONCEPT_LAMBDA * loss_cls
                    tot_cls   += loss_cls.item()

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot_cap += loss_cap.item(); n += 1

        sch.step()
        if ep % 10 == 0:
            cls_str = f"  cls_loss={tot_cls/n:.4f}" if tot_cls > 0 else ""
            print(f"    ep {ep:3d}  cap_loss={tot_cap/n:.4f}{cls_str}")


@torch.no_grad()
def extract_ca_out(proj, dec, loader, device):
    """Extract padded ca_out plans [N, L_CAP, D] from last decoder layer."""
    proj.eval(); dec.eval()
    plans, Xs, refs, ids_all = [], [], [], []
    for batch in loader:
        patches = batch["patches"].to(device)
        ids     = batch["cap_ids"].to(device)
        X = proj(patches)
        _, ca_out = dec(X, ids[:, :-1], ret_last_ca_out=True)
        B, L1, D = ca_out.shape
        pad  = torch.zeros(B, L_CAP - L1, D, device=device)
        plan = torch.cat([ca_out, pad], dim=1)   # [B, L_CAP, D]
        plans.append(plan.cpu())
        Xs.append(X.cpu())
        refs     += batch["ref"]
        ids_all.append(ids.cpu())
    return torch.cat(plans), torch.cat(Xs), refs, torch.cat(ids_all)


# ── Phase 2: train CFG denoiser on continuous plans ──────────────────────────
def train_den_v2(denoiser, plan_ds, n_epochs, device):
    loader = DataLoader(plan_ds, BATCH, shuffle=True)
    opt = torch.optim.Adam(denoiser.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)

    for ep in range(1, n_epochs + 1):
        denoiser.train()
        tot = 0.0; n = 0
        for plans, X, _, _ in loader:
            plans = plans.to(device); X = X.to(device)
            mr     = random.uniform(0.1, 0.95)
            plan_t, mask = mask_plan(plans, mr)
            drop   = (random.random() < CFG_DROP_P)
            pred   = denoiser(X, plan_t, mask, drop_X=drop)
            m = mask.bool()
            if not m.any(): continue
            loss = F.mse_loss(pred[m], plans[m])
            if torch.isnan(loss) or torch.isinf(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1
        sch.step()
        if ep % 10 == 0:
            s = f"{tot/n:.4f}" if n > 0 else "nan"
            print(f"    ep {ep:3d}  den_mse_loss={s}")


# ── Phase 3: train guided decoder ────────────────────────────────────────────
def train_guided_v2(guided_dec, plan_ds, denoiser, n_epochs, device, oracle_frac=0.5):
    """Mix oracle plans with denoiser-corrupted plans to prevent distribution shift."""
    loader = DataLoader(plan_ds, BATCH, shuffle=True)
    opt = torch.optim.Adam(guided_dec.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)

    for ep in range(1, n_epochs + 1):
        guided_dec.train()
        tot = 0.0; n = 0
        for plans, X, ids, _ in loader:
            plans = plans.to(device); X = X.to(device); ids = ids.to(device)

            if random.random() > oracle_frac:
                with torch.no_grad():
                    plan_t, mask = mask_plan(plans, random.uniform(0.3, 0.9))
                    plan_used    = denoiser.cfg_predict(X, plan_t, mask)
            else:
                plan_used = plans

            src_ext = torch.cat([X, plan_used], dim=1)   # [B, N+L, D]
            logits  = guided_dec(src_ext, ids[:, :-1])
            loss    = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                ids[:, 1:].reshape(-1), ignore_index=0
            )
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(guided_dec.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1
        sch.step()
        if ep % 10 == 0:
            print(f"    ep {ep:3d}  guided_v2_loss={tot/n:.4f}")


# ── Generate captions using denoised plan ────────────────────────────────────
@torch.no_grad()
def generate_captions_v2(proj, free_dec, guided_dec, denoiser, feats, tok,
                          batch_size=32, device=DEVICE):
    proj.eval(); free_dec.eval(); guided_dec.eval(); denoiser.eval()
    all_caps = []
    for i in range(0, len(feats), batch_size):
        X_raw = feats[i:i + batch_size].to(device)
        X     = proj(X_raw)
        plan  = generate_plan_v2(denoiser, X, n_steps=4, device=device)
        src_ext = torch.cat([X, plan], dim=1)
        toks  = guided_dec.generate(src_ext)
        all_caps += [tok.dec(t.tolist()) for t in toks.cpu()]
        if (i // batch_size) % 20 == 0:
            print(f"    {i + len(X_raw)}/{len(feats)}")
    return all_caps


# ── Arithmetic evaluation ─────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_arithmetic_v2(proj, free_dec, guided_dec, denoiser,
                            tv_feats, tv_noun_labels, tv_caps_flat, tv_cap_ids_flat,
                            test_feats, test_refs_per_img, tok, device):
    """
    Build centroids from oracle ca_out plans over train+val images.
    For each test image with src_noun in first caption:
      plan_free = denoised plan from denoiser(X)
      plan_edit = plan_free - centroid(src_noun) + centroid(tgt_noun)
      cap = guided_dec.generate(cat([X, plan_edit]))
      check if tgt_noun in cap
    """
    proj.eval(); free_dec.eval(); guided_dec.eval(); denoiser.eval()

    # ── Build centroids from oracle ca_out (first caption per tv image) ──────
    # tv_noun_labels: list of nouns per tv image (one per image, not per caption)
    # tv_feats: [N_tv, N_patches, D_raw] raw features
    n_tv = len(tv_noun_labels)
    batch_size = 64
    noun_plans = defaultdict(list)

    print("  Building concept centroids from train+val oracle plans...")
    for i in range(0, n_tv, batch_size):
        X_raw = tv_feats[i:i + batch_size].to(device)
        X     = proj(X_raw)
        # Use first caption per image — tv_cap_ids_flat has 5 caps per image,
        # but we only need one plan per image for the centroid
        # Index: image i → flat index i*N_CAP_PER (first caption)
        ids_batch = tv_cap_ids_flat[i * N_CAP_PER:(i + batch_size) * N_CAP_PER:N_CAP_PER]
        ids_batch = ids_batch.to(device)
        _, ca_out = free_dec(X[:len(ids_batch)], ids_batch[:, :-1], ret_last_ca_out=True)
        B, L1, D = ca_out.shape
        pad  = torch.zeros(B, L_CAP - L1, D, device=device)
        plan = torch.cat([ca_out, pad], dim=1).cpu()   # [B, L_CAP, D]
        for j, noun in enumerate(tv_noun_labels[i:i + len(ids_batch)]):
            if noun is not None:
                noun_plans[noun].append(plan[j])

    centroids = {}
    for noun, plan_list in noun_plans.items():
        if len(plan_list) >= 5:
            centroids[noun] = torch.stack(plan_list).mean(0).to(device)   # [L_CAP, D]
    print(f"  Centroids available: {sorted(centroids.keys())[:10]} ...")

    # ── Evaluate arithmetic per pair ─────────────────────────────────────────
    results = {}
    for src_noun, tgt_noun in EVAL_PAIRS:
        if src_noun not in centroids or tgt_noun not in centroids:
            print(f"  Skipping {src_noun}→{tgt_noun}: centroid missing")
            continue
        c_src = centroids[src_noun]   # [L_CAP, D]
        c_tgt = centroids[tgt_noun]   # [L_CAP, D]

        # Find test images whose first reference caption mentions src_noun
        test_idxs = [i for i, caps in enumerate(test_refs_per_img)
                     if caps and noun_in_caption(src_noun, caps[0])]
        n = min(len(test_idxs), 60)
        if n == 0:
            print(f"  Skipping {src_noun}→{tgt_noun}: no test images with src_noun")
            continue
        test_idxs = test_idxs[:n]

        target_appeared = 0
        for idx in test_idxs:
            X_raw    = test_feats[idx:idx + 1].to(device)   # [1, N_patches, D_raw]
            X        = proj(X_raw)                           # [1, N_patches, D]
            plan_den = generate_plan_v2(denoiser, X, n_steps=4, device=device)  # [1, L, D]
            # Arithmetic in ca_out space
            plan_edit = plan_den - c_src.unsqueeze(0) + c_tgt.unsqueeze(0)
            src_ext   = torch.cat([X, plan_edit], dim=1)
            toks      = guided_dec.generate(src_ext)
            cap_gen   = tok.dec(toks[0].tolist())
            if noun_in_caption(tgt_noun, cap_gen):
                target_appeared += 1

        rate = target_appeared / n
        results[f"{src_noun}→{tgt_noun}"] = {
            "n": n, "target_appeared": target_appeared, "rate": rate
        }
        print(f"  {src_noun}→{tgt_noun}: n={n}  target={target_appeared}  rate={rate:.3f}")

    mean_rate = sum(r["rate"] for r in results.values()) / max(1, len(results))
    print(f"\nMean target-noun rate: {mean_rate:.3f}")
    if mean_rate >= 0.20:
        verdict = "PASS"
    elif mean_rate < 0.10:
        verdict = "KILL"
    else:
        verdict = "MARGINAL"
    print(f"Verdict: {verdict}  (kill<0.10, pass>0.20, baseline A₀=0.038)")
    return results, mean_rate, verdict


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Load features ──────────────────────────────────────────────────────────
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached features from {CACHE_PATH}...")
        patch_feats = torch.load(CACHE_PATH, weights_only=True)
    else:
        print("Extracting patch features (DINOv2)...")
        from datasets import load_dataset as _lds
        ds    = _lds("clip-benchmark/wds_mscoco_captions", split="train")
        total = N_IMAGES + VAL_IMAGES + TEST_IMAGES
        imgs  = []
        for item in ds.select(range(total)):
            img = item["jpg"]
            if img.mode != "RGB": img = img.convert("RGB")
            imgs.append(img)
        patch_feats = extract_patch_features(imgs, encoder=ENCODER, device=DEVICE)
        torch.save(patch_feats, CACHE_PATH)
    print(f"  features: {patch_feats.shape}")

    # ── Load captions ──────────────────────────────────────────────────────────
    total = N_IMAGES + VAL_IMAGES + TEST_IMAGES
    print("Loading MSCOCO captions...")
    from datasets import load_dataset
    ds = load_dataset("clip-benchmark/wds_mscoco_captions", split="train")
    caps_per_img = []
    for item in ds.select(range(total)):
        caps = [c.strip() for c in item["txt"].strip().split("\n") if c.strip()]
        caps_per_img.append(caps[:N_CAP_PER])

    tok = Tokenizer()
    tok.build([c for caps in caps_per_img for c in caps], max_vocab=6000)
    print(f"  Vocab: {len(tok.vocab)}")

    # ── Top-N noun index for class head ───────────────────────────────────────
    n_tv       = N_IMAGES + VAL_IMAGES
    tv_flat    = [c for caps in caps_per_img[:n_tv] for c in caps]
    tv_nouns   = [extract_dominant_noun(c) for c in tv_flat]
    noun_cnt   = Counter(n for n in tv_nouns if n)
    top_nouns  = [n for n, _ in noun_cnt.most_common(TOP_N_NOUNS)]
    noun_to_idx = {n: i for i, n in enumerate(top_nouns)}
    n_classes  = len(top_nouns)
    print(f"  Top-{n_classes} nouns: {top_nouns}")

    # Per-image noun label for tv set (first non-None noun per image)
    tv_img_noun_labels = []
    for caps in caps_per_img[:n_tv]:
        noun = None
        for cap in caps:
            noun = extract_dominant_noun(cap)
            if noun: break
        tv_img_noun_labels.append(noun)

    # ── Feature splits ────────────────────────────────────────────────────────
    train_feats = patch_feats[:N_IMAGES]
    tv_feats    = patch_feats[:n_tv]
    test_feats  = patch_feats[n_tv: n_tv + TEST_IMAGES]
    train_caps  = caps_per_img[:N_IMAGES]
    tv_caps     = caps_per_img[:n_tv]
    test_caps   = caps_per_img[n_tv: n_tv + TEST_IMAGES]

    # ── Build models ──────────────────────────────────────────────────────────
    proj        = PatchProjector(ENCODER_DIM, D_MODEL).to(DEVICE)
    free_dec    = CaptionDecoderV2(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)
    class_head  = ClassHead(D_MODEL, n_classes).to(DEVICE)
    denoiser    = CFGDenoiserV2(D_MODEL, N_PATCHES, L_CAP).to(DEVICE)
    guided_dec  = GuidedDecoderV2(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)

    print(f"  proj={sum(p.numel() for p in proj.parameters()):,}"
          f"  free_dec={sum(p.numel() for p in free_dec.parameters()):,}"
          f"  denoiser={sum(p.numel() for p in denoiser.parameters()):,}"
          f"  guided_dec={sum(p.numel() for p in guided_dec.parameters()):,}")

    # ── Phase 1: free decoder + concept-supervised loss ───────────────────────
    print(f"\nPhase 1: free decoder + concept loss on train split ({N_IMAGES} images)...")
    train_ds    = CocoCapDatasetV2(train_feats, train_caps, tok)
    collate_fn  = make_collate_with_noun(noun_to_idx)
    train_loader = DataLoader(train_ds, BATCH, shuffle=True, collate_fn=collate_fn)
    train_free_v2(proj, free_dec, class_head, train_loader, EPOCHS_FREE, DEVICE)

    # ── Extract ca_out plans for all train+val captions ───────────────────────
    print(f"\nExtracting ca_out plans for train+val ({n_tv} images × {N_CAP_PER} captions)...")
    tv_ds     = CocoCapDatasetV2(tv_feats, tv_caps, tok)
    tv_loader = DataLoader(tv_ds, BATCH, shuffle=False, collate_fn=collate_fn)
    plans_tv, X_tv, refs_tv, ids_tv = extract_ca_out(proj, free_dec, tv_loader, DEVICE)
    print(f"  ca_out plans: {plans_tv.shape}")  # [N_tv*N_CAP, L_CAP, D]

    # Noun label per flattened caption entry
    noun_label_ids_tv = torch.tensor(
        [noun_to_idx.get(extract_dominant_noun(c), -1) for c in refs_tv],
        dtype=torch.long
    )
    plan_ds = PlanDatasetV2(plans_tv, X_tv, ids_tv, noun_label_ids_tv)

    # ── Phase 2: denoiser ──────────────────────────────────────────────────────
    print(f"\nPhase 2: CFGDenoiserV2 on ca_out plans (MSE regression)...")
    train_den_v2(denoiser, plan_ds, EPOCHS_DEN, DEVICE)

    # ── Phase 3: guided decoder ───────────────────────────────────────────────
    print(f"\nPhase 3: GuidedDecoderV2 (extended source: patches + plan)...")
    train_guided_v2(guided_dec, plan_ds, denoiser, EPOCHS_GUIDED, DEVICE)

    # ── Save models ───────────────────────────────────────────────────────────
    print(f"\nSaving to {SAVE_DIR}/...")
    torch.save(plans_tv,                  os.path.join(SAVE_DIR, "plans_tv.pt"))
    torch.save(proj.state_dict(),         os.path.join(SAVE_DIR, "model_proj.pt"))
    torch.save(free_dec.state_dict(),     os.path.join(SAVE_DIR, "model_free.pt"))
    torch.save(denoiser.state_dict(),     os.path.join(SAVE_DIR, "model_denoiser.pt"))
    torch.save(guided_dec.state_dict(),   os.path.join(SAVE_DIR, "model_guided.pt"))
    torch.save(class_head.state_dict(),   os.path.join(SAVE_DIR, "model_class_head.pt"))
    with open(os.path.join(SAVE_DIR, "tok_vocab.json"), "w") as f:
        json.dump(tok.vocab, f)
    with open(os.path.join(SAVE_DIR, "noun_labels_tv.json"), "w") as f:
        json.dump(tv_img_noun_labels, f)
    meta = {
        "plan_type": "ca_out_value_augmented",
        "plan_shape": list(plans_tv.shape),
        "n_classes": n_classes, "top_nouns": top_nouns,
        "encoder": ENCODER, "n_patches": N_PATCHES,
        "d_model": D_MODEL, "l_cap": L_CAP,
    }
    with open(os.path.join(SAVE_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("  Saved.")

    # ── Arithmetic evaluation ─────────────────────────────────────────────────
    print("\nArithmetic evaluation in ca_out space...")
    results, mean_rate, verdict = evaluate_arithmetic_v2(
        proj, free_dec, guided_dec, denoiser,
        tv_feats, tv_img_noun_labels, refs_tv, ids_tv,
        test_feats, test_caps, tok, DEVICE,
    )

    # ── Generate test captions for noun distribution check ────────────────────
    print("\nGenerating test captions (denoised plan, first 200 images)...")
    caps_gen = generate_captions_v2(
        proj, free_dec, guided_dec, denoiser,
        test_feats[:200], tok, device=DEVICE
    )
    noun_dist = Counter(n for c in caps_gen for n in [extract_dominant_noun(c)] if n)
    print("Top nouns in generated captions:")
    for n, c in noun_dist.most_common(10):
        print(f"  {n:15s}: {c}")

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = {
        "arithmetic": results,
        "mean_target_noun_rate": mean_rate,
        "verdict": verdict,
        "noun_dist_test_200": dict(noun_dist.most_common(20)),
        "baseline_A0_rate": 0.038,
    }
    out_path = os.path.join(RESULTS_DIR, "cap_v2_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"FINAL VERDICT: {verdict}")
    print(f"Mean target-noun rate (ca_out arithmetic): {mean_rate:.3f}")
    print(f"Baseline A₀ arithmetic rate:               0.038")
    print(f"Kill criterion: < 0.10 | Pass criterion: > 0.20")
    print(f"{'='*60}")
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
