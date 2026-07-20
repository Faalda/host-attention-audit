"""
Improved captioning with attention diffusion — stronger cross-attention denoiser.

Upgrade over run_cap_improved.py:
  CrossAttnCFGDenoiser replaces CFGDenoiser with:
  • Direct cross-attention from plan tokens → DINOv2 patch features
    Each plan position can directly look up which patches are visually
    relevant, independent of the current (possibly random) A_t.
    This is the primary fix for colour hallucination: the denoiser sees
    patch content (colour, texture) while deciding where to route attention.
  • 2D spatial positional encoding on patches (row + col embeddings)
    The 256 patches form a 16×16 grid; spatial proximity is now explicit.
  • Two full transformer blocks (self-attn + cross-attn + FF) for
    greater expressive capacity.

All other components (guided decoder, CFG, salience loss, 15K data) unchanged.
"""

import os, sys, json, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
from run_summarization import (
    Tokenizer,
    DEVICE, D_MODEL, N_HEADS, N_LAYERS, L_SUM as L_CAP,
    BATCH, LR, N_MP, eval_rouge,
)
from run_captioning import (
    PatchProjector, CaptionDecoder, GuidedCaptionDecoder,
    CocoCapDataset, collate_cap, A0CapDataset,
    train_free_cap, extract_A0_cap,
    N_PATCHES as N_PATCHES_CLIP, CLIP_DIM, N_CAP_PER,
)
from run_sum_masked import (
    mask_attn, MASK_RATES,
)
# Import shared components from run_cap_improved
from run_cap_improved import (
    ENCODER, _ENCODER_CFG, N_PATCHES, ENCODER_DIM,
    N_IMAGES, VAL_IMAGES, TEST_IMAGES,
    EPOCHS_FREE, EPOCHS_DEN, EPOCHS_GUIDED,
    K_LIST, SEEDS, CFG_SCALE, CFG_DROP_P, KL_LAMBDA, SAL_LAMBDA,
    RESULTS_DIR, CACHE_PATH,
    extract_patch_features,
    SoftGuidedDecoderLayer, SoftGuidedCaptionDecoder,
    maskgit_cfg,
    patch_salience,
    train_cfg_denoiser,
    train_guided_mixed,
    diversity_score,
    recovery_test,
    evaluate,
    plan_editing_demo,
)

# ── Stronger denoiser: cross-attention to patch features ──────────────────
class CrossAttnCFGDenoiser(nn.Module):
    """
    Denoiser with direct cross-attention from plan tokens to DINOv2 patch features.

    Architecture per forward pass:
      1. Encode each attention row (A_t[b,l,:]) + mask status  →  h [B,L,D]
      2. Message passing: weighted-sum of patch features via A_t  (as before)
      3. Self-attention over L plan positions                     (block 1)
      4. Cross-attention: plan tokens → patch features            (NEW)
         Each plan position directly attends to all 256 patches.
         Spatial PE on patches makes adjacent-patch structure explicit.
      5. Feed-forward                                             (block 1 FF)
      6. Self-attention                                           (block 2)
      7. Feed-forward                                             (block 2 FF)
      8. Output: Q=plan, K=patches → logits [B,L,N]
    """

    def __init__(self, D, n_src=N_PATCHES, n_tgt=L_CAP, n_mp=N_MP):
        super().__init__()
        self.n_src = n_src
        self.D     = D
        grid       = int(round(n_src ** 0.5))   # 16 for DINOv2 256 patches

        # ── Input encoding ────────────────────────────────────────────────
        self.row_proj  = nn.Linear(n_src, D)
        self.mask_emb  = nn.Embedding(2, D)

        # ── 2D spatial PE on patches (row + col learnable embeddings) ─────
        self.row_emb   = nn.Embedding(grid, D // 2)
        self.col_emb   = nn.Embedding(grid, D // 2)
        rows = torch.arange(grid).repeat_interleave(grid)  # [n_src]
        cols = torch.arange(grid).repeat(grid)             # [n_src]
        self.register_buffer('patch_rows', rows)
        self.register_buffer('patch_cols', cols)

        # ── Message passing (as before: plan-weighted feature aggregation) ─
        self.msg_projs = nn.ModuleList([nn.Linear(D, D) for _ in range(n_mp)])
        self.mp_norms  = nn.ModuleList([nn.LayerNorm(D) for _ in range(n_mp)])

        # ── Block 1: self-attention over plan positions ───────────────────
        self.sa1      = nn.MultiheadAttention(D, 4, batch_first=True)
        self.sa1_norm = nn.LayerNorm(D)

        # ── Block 1: cross-attention — plan tokens query patch features ───
        self.ca       = nn.MultiheadAttention(D, 4, batch_first=True)
        self.ca_norm  = nn.LayerNorm(D)
        self.ca_ff    = nn.Sequential(
            nn.Linear(D, D * 4), nn.GELU(), nn.Linear(D * 4, D)
        )
        self.ca_ff_norm = nn.LayerNorm(D)

        # ── Block 2: self-attention + FF ──────────────────────────────────
        self.sa2      = nn.MultiheadAttention(D, 4, batch_first=True)
        self.sa2_norm = nn.LayerNorm(D)
        self.sa2_ff   = nn.Sequential(
            nn.Linear(D, D * 4), nn.GELU(), nn.Linear(D * 4, D)
        )
        self.sa2_ff_norm = nn.LayerNorm(D)

        # ── Output projection ─────────────────────────────────────────────
        self.q_proj = nn.Linear(D, D)
        self.k_proj = nn.Linear(D, D)

        # Null token for CFG unconditional path
        self.null_X = nn.Parameter(torch.zeros(1, n_src, D))
        self.scale  = D ** -0.5

    def _spatial_patches(self, X_src):
        """Add 2D spatial positional encoding to patch features. [B, N, D]"""
        pos = torch.cat([
            self.row_emb(self.patch_rows),  # [N, D//2]
            self.col_emb(self.patch_cols),  # [N, D//2]
        ], dim=-1)                           # [N, D]
        return X_src + pos.unsqueeze(0)      # [B, N, D]  (broadcast over batch)

    def _forward_once(self, X_src, A_t, mask):
        B, L, N = A_t.shape

        # Patch features with spatial encoding
        X_pos = self._spatial_patches(X_src)   # [B, N, D]

        # Step 1 — encode each attention row + mask token
        h = F.gelu(self.row_proj(A_t)) + self.mask_emb(mask.long())  # [B, L, D]

        # Step 2 — message passing: plan-weighted patch aggregation
        for mp, ln in zip(self.msg_projs, self.mp_norms):
            msgs = torch.bmm(A_t, mp(X_pos))   # [B, L, D]
            h    = ln(h + msgs)

        # Block 1: self-attention over plan positions
        h2, _ = self.sa1(h, h, h, need_weights=False)
        h = self.sa1_norm(h + h2)

        # Block 1: cross-attention — each plan token directly queries all patches
        # Q=plan positions, K=V=spatially-encoded patch features
        h_ca, _ = self.ca(h, X_pos, X_pos, need_weights=False)   # [B, L, D]
        h = self.ca_norm(h + h_ca)
        h = self.ca_ff_norm(h + self.ca_ff(h))

        # Block 2: self-attention + FF
        h2, _ = self.sa2(h, h, h, need_weights=False)
        h = self.sa2_norm(h + h2)
        h = self.sa2_ff_norm(h + self.sa2_ff(h))

        # Output: compute match scores between refined plan and patch features
        Q = F.normalize(self.q_proj(h), dim=-1)
        K = self.k_proj(X_pos)
        return torch.bmm(Q, K.transpose(-1, -2)) * self.scale   # [B, L, N]

    def forward(self, X_src, A_t, mask, drop_X=False):
        if drop_X:
            X_null = self.null_X.expand(X_src.shape[0], -1, -1)
            return self._forward_once(X_null, A_t, mask)
        return self._forward_once(X_src, A_t, mask)

    @torch.no_grad()
    def cfg_predict(self, X_src, A_t, mask, scale=CFG_SCALE):
        cond   = self._forward_once(X_src, A_t, mask)
        uncond = self._forward_once(
            self.null_X.expand(X_src.shape[0], -1, -1), A_t, mask
        )
        return uncond + scale * (cond - uncond)


# ── Main ──────────────────────────────────────────────────────────────────
def run_seed(seed, patch_feats, caps_per_img, tok):
    print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    n_train = N_IMAGES; n_val = VAL_IMAGES

    train_feats = patch_feats[:n_train]
    tv_feats    = patch_feats[:n_train + n_val]
    test_feats  = patch_feats[n_train + n_val:n_train + n_val + TEST_IMAGES]

    train_caps = caps_per_img[:n_train]
    tv_caps    = caps_per_img[:n_train + n_val]
    test_caps  = caps_per_img[n_train + n_val:n_train + n_val + TEST_IMAGES]

    proj       = PatchProjector(ENCODER_DIM, D_MODEL).to(DEVICE)
    free_dec   = CaptionDecoder(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS,
                                n_src=N_PATCHES).to(DEVICE)
    denoiser   = CrossAttnCFGDenoiser(D_MODEL, N_PATCHES, L_CAP, N_MP).to(DEVICE)
    guided_dec = SoftGuidedCaptionDecoder(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)

    n_den = sum(p.numel() for p in denoiser.parameters())
    print(f"  Encoder={ENCODER}  N_PATCHES={N_PATCHES}  "
          f"CrossAttnDenoiser={n_den:,} params  KL_LAMBDA={KL_LAMBDA}")

    # Phase 1: train image-captioning model
    train_ds     = CocoCapDataset(train_feats, train_caps, tok, proj)
    train_loader = DataLoader(train_ds, BATCH, shuffle=True, collate_fn=collate_cap)
    print("  Phase 1: training captioning model...")
    train_free_cap(proj, free_dec, train_loader, EPOCHS_FREE, DEVICE)

    # Phase 2: extract A_0, train cross-attention denoiser
    print("  Extracting A_0 (5 captions × image)...")
    tv_ds     = CocoCapDataset(tv_feats, tv_caps, tok, proj)
    tv_loader = DataLoader(tv_ds, BATCH, shuffle=False, collate_fn=collate_cap)
    A0_all, X_all, _, cap_ids_all = extract_A0_cap(proj, free_dec, tv_loader, DEVICE)
    A0_ds = A0CapDataset(A0_all, X_all, cap_ids_all)
    print(f"  A_0 dataset: {len(A0_ds)} pairs ({len(tv_caps)} images × 5 captions)")

    print("  Phase 2: training cross-attention denoiser (masked + CFG dropout)...")
    train_cfg_denoiser(denoiser, A0_ds, EPOCHS_DEN, DEVICE)

    # Phase 3: train guided decoder with mixed plans
    proj.cpu(); free_dec.cpu()
    del train_loader, tv_loader
    torch.cuda.empty_cache()
    print("  Phase 3: training guided decoder (50% oracle / 50% denoiser)...")
    train_guided_mixed(guided_dec, A0_ds, denoiser, EPOCHS_GUIDED, DEVICE)
    proj.to(DEVICE); free_dec.to(DEVICE)

    # Recovery test
    A0_test, X_test, _, _ = extract_A0_cap(proj, free_dec,
        DataLoader(CocoCapDataset(test_feats, test_caps, tok, proj),
                   BATCH, shuffle=False, collate_fn=collate_cap), DEVICE)
    rec = recovery_test(denoiser, A0_test, X_test)

    # Evaluate
    test_ds     = CocoCapDataset(test_feats, test_caps, tok, proj)
    test_loader = DataLoader(test_ds, BATCH, shuffle=False, collate_fn=collate_cap)
    print("  Evaluating...")
    scores = evaluate(proj, free_dec, guided_dec, denoiser, test_loader, tok, DEVICE)

    print("\n  ROUGE-1 F1:")
    for k, v in scores.items():
        if k == "diversity":
            print(f"    {'diversity':20s}: {v:.4f}  (distinct-1)")
            continue
        delta = v - scores["free"]
        tag = f"  Δfree={delta:+.4f}" if k != "free" else "  ← baseline"
        print(f"    {k:20s}: {v:.4f}{tag}")

    if hasattr(guided_dec, 'gate_values'):
        gates = guided_dec.gate_values()
        print(f"\n  Learned plan gates (per layer):")
        for i, g in enumerate(gates):
            bar = "█" * (int(g * 20) if math.isfinite(g) else 0)
            print(f"    layer {i}: {g:.3f}  {bar}")

    print(f"\n  Recovery (random = 1/{N_PATCHES} = {1/N_PATCHES:.3f}):")
    for mr, acc in sorted(rec.items()):
        print(f"   mask={mr:.2f}  acc={acc:.3f}  vs_random={acc - 1/N_PATCHES:+.3f}")

    return scores, rec


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    from datasets import load_dataset
    print("Loading MSCOCO train split (15K images)...")
    ds = load_dataset("clip-benchmark/wds_mscoco_captions", split="train")
    total = N_IMAGES + VAL_IMAGES + TEST_IMAGES

    images, caps_per_img = [], []
    for item in ds.select(range(total)):
        img = item["jpg"]
        if img.mode != "RGB": img = img.convert("RGB")
        images.append(img)
        caps = [c.strip() for c in item["txt"].strip().split("\n") if c.strip()]
        caps_per_img.append(caps[:N_CAP_PER])

    # Load cached patch features
    if os.path.exists(CACHE_PATH):
        print(f"  Loading cached {ENCODER} patch features from {CACHE_PATH}...")
        patch_feats = torch.load(CACHE_PATH, weights_only=True)
    else:
        print(f"  Extracting {ENCODER} patch features...")
        patch_feats = extract_patch_features(images, encoder=ENCODER, device=DEVICE)
        torch.save(patch_feats, CACHE_PATH)
    print(f"  Patch features: {patch_feats.shape}")

    tok = Tokenizer()
    tok.build([cap for caps in caps_per_img for cap in caps], max_vocab=6000)
    print(f"  Vocab: {len(tok.vocab)}")

    all_scores, all_recs = [], []
    for seed in SEEDS:
        scores, rec = run_seed(seed, patch_feats, caps_per_img, tok)
        all_scores.append(scores)
        all_recs.append(rec)

    keys = ["free", "oracle_free", "oracle_self"] + [f"K{k}" for k in K_LIST]
    free_mean = float(np.mean([s["free"] for s in all_scores]))
    print(f"\n{'='*60}\nSUMMARY — {len(SEEDS)} seeds\n{'='*60}")
    for key in keys:
        m   = float(np.mean([s[key] for s in all_scores]))
        std = float(np.std([s[key]  for s in all_scores]))
        tag = "  ← baseline" if key == "free" else f"  Δfree={m-free_mean:+.4f}"
        print(f"  {key:20s}: {m:.4f} ± {std:.4f}{tag}")
    div_m = float(np.mean([s["diversity"] for s in all_scores]))
    print(f"  {'diversity':20s}: {div_m:.4f}")

    print(f"\n  Recovery (random = 1/{N_PATCHES} = {1/N_PATCHES:.4f}):")
    for mr in MASK_RATES:
        vals = [r[mr] for r in all_recs if not math.isnan(r[mr])]
        m = float(np.mean(vals)) if vals else float('nan')
        print(f"   mask={mr:.2f}  acc={m:.3f}  vs_random={m - 1/N_PATCHES:+.3f}")

    out = {
        "config": {
            "encoder": ENCODER, "denoiser": "CrossAttnCFGDenoiser",
            "CFG_SCALE": CFG_SCALE, "oracle_frac": 0.5,
            "n_patches": N_PATCHES, "n_cap_per": N_CAP_PER,
            "n_train": N_IMAGES, "sal_lambda": SAL_LAMBDA,
        },
        "scores": all_scores,
        "recovery": [{str(k): v for k, v in r.items()} for r in all_recs],
    }
    path = os.path.join(RESULTS_DIR, "cap_xattn_dinov2_15k.json")
    with open(path, "w") as f: json.dump(out, f, indent=2)
    print(f"\nFull results → {path}")


if __name__ == "__main__":
    main()
