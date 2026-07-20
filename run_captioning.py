"""
Attention diffusion for image captioning — MSCOCO, 5 captions per image.

Fix 2: genuine non-determinism from multiple captions per image.
  Same image X → 5 different A_0 (one per caption)
  → denoiser cannot predict A_0 from X alone
  → must use A_t to know which caption to generate

Source tokens: CLIP ViT-B/32 patch features [49, 768] projected to [49, D]
Target: caption tokens (up to L_CAP words)
A_0: cross-attention [L_CAP, 49]  (which patch drove which word)

The key experiment:
  recovery at α=1 should be ~1/5 = 20%  (5 valid captions → can't distinguish without A_t)
  recovery at α=0.5 should be >> 20%   (A_t reveals which caption)
"""

import os, sys, json, re, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from run_summarization import (
    Tokenizer, DecoderLayer, GuidedDecoderLayer,
    DEVICE, D_MODEL, N_HEADS, N_LAYERS, L_SUM as L_CAP,
    BATCH, LR, EPOCHS_FREE, EPOCHS_GUIDED, N_MP,
    eval_rouge,
)
from run_sum_masked import (
    MaskedGNNDenoiser, mask_attn, maskgit_denoise,
    recovery_test, EPOCHS_MASK_DEN, MASK_RATES,
)

N_PATCHES   = 49       # CLIP ViT-B/32: 7×7
CLIP_DIM    = 768      # CLIP patch hidden dim (before projection)
IMG_EMBED   = 512      # CLIP output dim (unused — we use patch features)
N_CAP_PER   = 5        # captions per image
N_IMAGES    = 2000     # images to use (→ 10k caption pairs)
VAL_IMAGES  = 300
TEST_IMAGES = 300
EPOCHS_FREE_CAP = 40
K_LIST      = [1, 2, 4, 8]
SEEDS       = [42, 7, 13]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CACHE_PATH  = os.path.join(RESULTS_DIR, "clip_patch_features.pt")

# ── CLIP patch feature extraction ──────────────────────────────────────────
def extract_clip_patches(images, batch_size=64, device=DEVICE):
    """
    Extract CLIP ViT-B/32 patch features for a list of PIL images.
    Returns [N, 49, 768] tensor.
    """
    from transformers import CLIPProcessor, CLIPVisionModel
    print("  Loading CLIP ViT-B/32...")
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    all_feats = []
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i+batch_size]
        inputs = proc(images=batch_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        # last_hidden_state: [B, 50, 768] — skip CLS token (index 0)
        patches = out.last_hidden_state[:, 1:, :]  # [B, 49, 768]
        all_feats.append(patches.cpu().float())
        if (i // batch_size) % 5 == 0:
            print(f"    {i+len(batch_imgs)}/{len(images)} images processed")
    return torch.cat(all_feats, dim=0)  # [N, 49, 768]

# ── Data ───────────────────────────────────────────────────────────────────
class CocoCapDataset(Dataset):
    """Each item: one (image_idx, caption). Patch features pre-extracted."""
    def __init__(self, patch_feats, captions_per_img, tok, patch_proj):
        """
        patch_feats:       [N_img, 49, 768]
        captions_per_img:  list of length N_img, each is list of 5 str captions
        patch_proj:        nn.Linear(768 → D_MODEL) — project patches
        """
        self.patch_feats = patch_feats
        self.tok = tok
        # Flatten: (img_idx, cap_text)
        self.pairs = []
        for img_idx, caps in enumerate(captions_per_img):
            for cap in caps:
                ids = tok.enc(cap, L_CAP, bos=True, eos=True)
                self.pairs.append((img_idx, torch.tensor(ids, dtype=torch.long), cap))

    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        img_idx, ids, cap = self.pairs[i]
        return {
            "patches": self.patch_feats[img_idx],  # [49, 768]
            "cap_ids": ids,                          # [L_CAP]
            "ref": cap,
        }

def collate_cap(batch):
    return {
        "patches": torch.stack([b["patches"] for b in batch]),  # [B,49,768]
        "cap_ids": torch.stack([b["cap_ids"] for b in batch]),  # [B,L]
        "ref": [b["ref"] for b in batch],
    }

# ── Patch encoder projector ────────────────────────────────────────────────
class PatchProjector(nn.Module):
    """Project CLIP patch features [B,49,768] → [B,49,D_MODEL]."""
    def __init__(self, clip_dim=CLIP_DIM, d_model=D_MODEL):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(clip_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
    def forward(self, patches): return self.proj(patches)

# ── Caption decoder (same architecture as summarization) ───────────────────
class CaptionDecoder(nn.Module):
    def __init__(self, V, D, H, NL, n_src=N_PATCHES, dropout=0.1):
        super().__init__()
        self.embed  = nn.Embedding(V, D)
        self.pos    = nn.Embedding(L_CAP + 2, D)
        self.layers = nn.ModuleList([DecoderLayer(D, H, dropout) for _ in range(NL)])
        self.head   = nn.Linear(D, V)
        self.NL = NL

    def forward(self, src, tgt, ret_last_attn=False):
        B, L = tgt.shape
        x = self.embed(tgt) + self.pos(torch.arange(L, device=tgt.device))
        causal  = nn.Transformer.generate_square_subsequent_mask(L, device=tgt.device)
        tgt_pad = (tgt == 0)
        attn_w  = None
        for i, layer in enumerate(self.layers):
            last = (i == self.NL - 1) and ret_last_attn
            out  = layer(x, src, tgt_mask=causal, tgt_pad=tgt_pad, ret_attn=last)
            if last: x, attn_w = out
            else:    x = out
        logits = self.head(x)
        return (logits, attn_w) if ret_last_attn else logits

    @torch.no_grad()
    def generate(self, src, max_len=L_CAP, bos=1, eos=2):
        B = src.shape[0]
        toks = torch.full((B, 1), bos, device=src.device, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len - 1):
            logits = self.forward(src, toks)
            next_t = logits[:, -1].argmax(-1)
            toks = torch.cat([toks, next_t.unsqueeze(1)], dim=1)
            done |= (next_t == eos)
            if done.all(): break
        return toks

class GuidedCaptionDecoder(nn.Module):
    def __init__(self, V, D, H, NL, dropout=0.1):
        super().__init__()
        self.embed  = nn.Embedding(V, D)
        self.pos    = nn.Embedding(L_CAP + 2, D)
        self.layers = nn.ModuleList([GuidedDecoderLayer(D, H, dropout) for _ in range(NL)])
        self.head   = nn.Linear(D, V)

    def forward(self, src, tgt, attn_plan):
        B, L = tgt.shape
        x = self.embed(tgt) + self.pos(torch.arange(L, device=tgt.device))
        causal  = nn.Transformer.generate_square_subsequent_mask(L, device=tgt.device)
        tgt_pad = (tgt == 0)
        for layer in self.layers:
            x = layer(x, src, attn_plan[:, :L, :], tgt_mask=causal, tgt_pad=tgt_pad)
        return self.head(x)

    @torch.no_grad()
    def generate(self, src, attn_plan, max_len=L_CAP, bos=1, eos=2):
        B = src.shape[0]
        toks = torch.full((B, 1), bos, device=src.device, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for step in range(max_len - 1):
            logits = self.forward(src, toks, attn_plan)
            next_t = logits[:, -1].argmax(-1)
            toks = torch.cat([toks, next_t.unsqueeze(1)], dim=1)
            done |= (next_t == eos)
            if done.all(): break
        return toks

# ── Training ─────────────────────────────────────────────────────────────────
def train_free_cap(proj, dec, loader, n_epochs, device):
    params = list(proj.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    for ep in range(1, n_epochs + 1):
        proj.train(); dec.train()
        tot = 0; n = 0
        for batch in loader:
            patches = batch["patches"].to(device)   # [B,49,768]
            ids     = batch["cap_ids"].to(device)   # [B,L]
            X = proj(patches)                        # [B,49,D]
            logits = dec(X, ids[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   ids[:, 1:].reshape(-1), ignore_index=0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += loss.item(); n += 1
        sch.step()
        if ep % 10 == 0:
            print(f"    ep {ep:3d}  cap_loss={tot/n:.4f}")

@torch.no_grad()
def extract_A0_cap(proj, dec, loader, device):
    proj.eval(); dec.eval()
    A0s, Xs, refs, id_list = [], [], [], []
    for batch in loader:
        patches = batch["patches"].to(device)
        ids     = batch["cap_ids"].to(device)
        X = proj(patches)
        _, attn_w = dec(X, ids[:, :-1], ret_last_attn=True)  # [B,L-1,49]
        B, L1, NP = attn_w.shape
        pad = torch.zeros(B, L_CAP - L1, NP, device=device)
        A0_full = torch.cat([attn_w, pad], dim=1)   # [B,L_CAP,49]
        A0s.append(A0_full.cpu())
        Xs.append(X.cpu())
        refs += batch["ref"]
        id_list.append(ids.cpu())
    return torch.cat(A0s), torch.cat(Xs), refs, torch.cat(id_list)

class A0CapDataset(Dataset):
    def __init__(self, A0, X, cap_ids): self.A0=A0; self.X=X; self.cap_ids=cap_ids
    def __len__(self): return len(self.A0)
    def __getitem__(self, i): return self.A0[i], self.X[i], self.cap_ids[i]

def train_guided_cap(guided_dec, A0_ds, n_epochs, device):
    loader = DataLoader(A0_ds, BATCH, shuffle=True)
    opt = torch.optim.Adam(guided_dec.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    for ep in range(1, n_epochs + 1):
        guided_dec.train()
        tot = 0; cnt = 0
        for A0, X, ids in loader:
            A0 = A0.to(device); X = X.to(device); ids = ids.to(device)
            plan = F.softmax(A0 * 10, dim=-1)
            logits = guided_dec(X, ids[:, :-1], plan)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   ids[:, 1:].reshape(-1), ignore_index=0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(guided_dec.parameters(), 1.0)
            opt.step()
            tot += loss.item(); cnt += 1
        sch.step()
        if ep % 10 == 0:
            print(f"    ep {ep:3d}  guided_cap_loss={tot/cnt:.4f}")

def train_masked_den(denoiser, A0_ds, n_epochs, device):
    loader = DataLoader(A0_ds, BATCH, shuffle=True)
    opt = torch.optim.Adam(denoiser.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    for ep in range(1, n_epochs + 1):
        denoiser.train()
        tot = 0; n = 0
        for A0, X, _ in loader:
            A0 = A0.to(device); X = X.to(device)
            mr = random.uniform(0.1, 0.95)
            A_t, mask = mask_attn(A0, mr)
            logits = denoiser(X, A_t, mask)
            tgt = A0.argmax(-1)
            m = mask.bool()
            if not m.any(): continue
            loss = F.cross_entropy(logits[m], tgt[m])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1
        sch.step()
        if ep % 10 == 0:
            print(f"    ep {ep:3d}  mask_den_loss={tot/n:.4f}")

# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_cap(proj, free_dec, guided_dec, denoiser, test_loader, tok, device):
    proj.eval(); free_dec.eval(); guided_dec.eval(); denoiser.eval()
    free_preds, oracle_preds, refs = [], [], []
    k_preds = {k: [] for k in K_LIST}

    for batch in test_loader:
        patches = batch["patches"].to(device)
        ids     = batch["cap_ids"].to(device)
        X = proj(patches)
        B = X.shape[0]

        toks = free_dec.generate(X)
        free_preds += [tok.dec(r) for r in toks.cpu().tolist()]

        _, attn_w = free_dec(X, ids[:, :-1], ret_last_attn=True)
        pad = torch.zeros(B, L_CAP - attn_w.shape[1], N_PATCHES, device=device)
        A_oracle = torch.cat([attn_w, pad], dim=1).detach()
        plan_or  = F.softmax(A_oracle * 10, dim=-1)
        oracle_preds += [tok.dec(r) for r in guided_dec.generate(X, plan_or).cpu().tolist()]
        refs += batch["ref"]

        for k in K_LIST:
            A_mg = maskgit_denoise(denoiser, X, n_steps=k, device=device)
            plan = F.softmax(A_mg * 10, dim=-1)
            k_preds[k] += [tok.dec(r) for r in guided_dec.generate(X, plan).cpu().tolist()]

    return {
        "free":   eval_rouge(free_preds,   refs),
        "oracle": eval_rouge(oracle_preds, refs),
        **{f"K{k}": eval_rouge(k_preds[k], refs) for k in K_LIST},
    }

@torch.no_grad()
def recovery_test_cap(denoiser, A0_all, X_all, n_samples=512, device=DEVICE):
    """
    For captioning: at mask_rate=1 (all masked), x-only accuracy should be
    ~20% (5 valid captions per image). With mask_rate=0.5, accuracy should
    be >> 20% because unmasked rows reveal which of the 5 captions is active.
    """
    idx = torch.randperm(len(A0_all))[:n_samples]
    A0  = A0_all[idx].to(device)
    X   = X_all[idx].to(device)
    results = {}
    for mr in MASK_RATES:
        A_t, mask = mask_attn(A0, mr)
        logits = denoiser(X, A_t, mask)
        pred = logits.argmax(-1)
        tgt  = A0.argmax(-1)
        m = mask.bool()
        if not m.any(): results[mr] = float('nan'); continue
        results[mr] = (pred == tgt)[m].float().mean().item()
    return results

# ── Main ──────────────────────────────────────────────────────────────────────
def run_seed(seed, patch_feats, caps_per_img_all, tok, n_train, n_val, n_test):
    print(f"\n{'='*60}")
    print(f"Seed {seed}")
    print(f"{'='*60}")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    n_imgs = n_train + n_val + n_test
    train_feats = patch_feats[:n_train]
    val_feats   = patch_feats[n_train:n_train+n_val]
    test_feats  = patch_feats[n_train+n_val:n_imgs]

    train_caps = caps_per_img_all[:n_train]
    val_caps   = caps_per_img_all[n_train:n_train+n_val]
    test_caps  = caps_per_img_all[n_train+n_val:n_imgs]

    proj        = PatchProjector().to(DEVICE)
    free_dec    = CaptionDecoder(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS, N_PATCHES).to(DEVICE)
    mask_den    = MaskedGNNDenoiser(D_MODEL, N_PATCHES, L_CAP, N_MP).to(DEVICE)
    guided_dec  = GuidedCaptionDecoder(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)

    print(f"  proj={sum(p.numel() for p in proj.parameters()):,} "
          f"dec={sum(p.numel() for p in free_dec.parameters()):,} "
          f"den={sum(p.numel() for p in mask_den.parameters()):,}")

    # All 5 captions per image for training/val denoiser
    tv_feats = torch.cat([train_feats, val_feats])
    tv_caps  = train_caps + val_caps

    train_ds = CocoCapDataset(train_feats, train_caps, tok, proj)
    tv_ds    = CocoCapDataset(tv_feats,   tv_caps,    tok, proj)
    test_ds  = CocoCapDataset(test_feats,  test_caps,  tok, proj)

    train_loader = DataLoader(train_ds, BATCH, shuffle=True,  collate_fn=collate_cap)
    test_loader  = DataLoader(test_ds,  BATCH, shuffle=False, collate_fn=collate_cap)

    # Phase 1: train free decoder
    print("  Phase 1: training free decoder...")
    train_free_cap(proj, free_dec, train_loader, EPOCHS_FREE_CAP, DEVICE)

    # Extract A_0 for train+val (all 5 captions per image)
    print("  Extracting A_0 (all captions)...")
    tv_loader = DataLoader(tv_ds, BATCH, shuffle=False, collate_fn=collate_cap)
    A0_all, X_all, _, cap_ids_all = extract_A0_cap(proj, free_dec, tv_loader, DEVICE)
    A0_cap_ds = A0CapDataset(A0_all, X_all, cap_ids_all)

    # Phase 2: train masked denoiser
    print("  Phase 2: training masked denoiser...")
    train_masked_den(mask_den, A0_cap_ds, EPOCHS_MASK_DEN, DEVICE)

    # Phase 3: train guided decoder
    print("  Phase 3: training guided decoder...")
    train_guided_cap(guided_dec, A0_cap_ds, EPOCHS_GUIDED, DEVICE)

    # Recovery test — key diagnostic for Fix 2
    A0_test, X_test, _, _ = extract_A0_cap(proj, free_dec,
        DataLoader(test_ds, BATCH, shuffle=False, collate_fn=collate_cap), DEVICE)
    rec = recovery_test_cap(mask_den, A0_test, X_test)

    # ROUGE
    print("  Evaluating...")
    scores = evaluate_cap(proj, free_dec, guided_dec, mask_den, test_loader, tok, DEVICE)

    print("\n  ROUGE-1 F1:")
    for k, v in scores.items():
        delta = v - scores["free"]
        print(f"    {k:20s}: {v:.4f}  Δfree={delta:+.4f}")

    # Expected: at mask_rate=1.0, acc ≈ 20% (1/5 captions)
    # If > 20% at mask_rate=0.5 → denoiser uses context from unmasked rows
    print("\n  Recovery on masked rows (expected random=~20% from 5 captions):")
    for mr, acc in sorted(rec.items()):
        print(f"   mask_rate={mr:.2f}  acc={acc:.3f}  vs_random={acc-0.20:+.3f}")

    return scores, rec

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load MSCOCO
    print("Loading MSCOCO...")
    from datasets import load_dataset
    ds = load_dataset("clip-benchmark/wds_mscoco_captions", split="test")
    total = N_IMAGES + VAL_IMAGES + TEST_IMAGES

    images, caps_per_img = [], []
    for item in ds.select(range(total)):
        img = item["jpg"]
        if img.mode != "RGB": img = img.convert("RGB")
        images.append(img)
        caps = [c.strip() for c in item["txt"].strip().split("\n") if c.strip()]
        caps_per_img.append(caps[:N_CAP_PER])

    print(f"  {len(images)} images, {N_CAP_PER} captions each")

    # Extract or load CLIP patch features
    if os.path.exists(CACHE_PATH):
        print(f"  Loading cached patch features from {CACHE_PATH}")
        patch_feats = torch.load(CACHE_PATH, weights_only=True)
    else:
        print("  Extracting CLIP patch features (one-time)...")
        patch_feats = extract_clip_patches(images, device=DEVICE)
        torch.save(patch_feats, CACHE_PATH)
        print(f"  Saved to {CACHE_PATH}")

    print(f"  Patch features: {patch_feats.shape}")  # [total, 49, 768]

    # Build vocabulary from all captions
    print("Building vocabulary...")
    tok = Tokenizer()
    all_texts = [cap for caps in caps_per_img for cap in caps]
    tok.build(all_texts, max_vocab=6000)
    print(f"  Vocab size: {len(tok.vocab)}")

    all_scores, all_recs = [], []
    for seed in SEEDS:
        scores, rec = run_seed(seed, patch_feats, caps_per_img, tok,
                                N_IMAGES, VAL_IMAGES, TEST_IMAGES)
        all_scores.append(scores); all_recs.append(rec)

    keys = ["free", "oracle"] + [f"K{k}" for k in K_LIST]
    free_mean = float(np.mean([s["free"] for s in all_scores]))
    print(f"\n{'='*60}\nSUMMARY — mean over {len(SEEDS)} seeds\n{'='*60}")
    print("\n  ROUGE-1 F1:")
    for key in keys:
        m = float(np.mean([s[key] for s in all_scores]))
        std = float(np.std([s[key] for s in all_scores]))
        print(f"    {key:20s}: {m:.4f} ± {std:.4f}  Δfree={m-free_mean:+.4f}")

    print("\n  Recovery on masked rows (expected random≈20%  = 1 out of 5 captions):")
    for mr in MASK_RATES:
        vals = [r[mr] for r in all_recs if not math.isnan(r[mr])]
        m = float(np.mean(vals)) if vals else float('nan')
        print(f"   mask_rate={mr:.2f}  acc={m:.3f}  vs_random={m-0.20:+.3f}")

    out = {
        "config": {"task":"captioning","n_patches":N_PATCHES,"n_cap_per":N_CAP_PER},
        "scores": all_scores,
        "recovery": [{str(k):v for k,v in r.items()} for r in all_recs],
    }
    path = os.path.join(RESULTS_DIR, "captioning_attn_diff.json")
    with open(path, "w") as f: json.dump(out, f, indent=2)
    print(f"\nFull results → {path}")

if __name__ == "__main__":
    main()
