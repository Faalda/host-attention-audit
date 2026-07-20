"""
run_recovery_top5.py — Add Top-5 recovery to the existing recovery table.

Loads the saved models (proj, free_dec, denoiser), re-extracts oracle A₀
from the free decoder using MSCOCO reference captions, then measures both
Top-1 and Top-5 recovery for every mask rate.

This matches what was done at training time (in run_save_plans.py), so
results are directly comparable to the paper's Tab. 3 numbers.

Usage:
    python3 run_recovery_top5.py
"""

import os, sys, json, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from run_summarization import Tokenizer, DEVICE, D_MODEL, N_HEADS, N_LAYERS, L_SUM as L_CAP, N_MP
from run_captioning import PatchProjector, CaptionDecoder, CocoCapDataset, collate_cap, N_CAP_PER, extract_A0_cap
from run_cap_improved import (
    ENCODER, N_PATCHES, ENCODER_DIM,
    N_IMAGES, VAL_IMAGES, TEST_IMAGES,
    RESULTS_DIR, CACHE_PATH,
    CFGDenoiser,
)
from run_sum_masked import mask_attn, MASK_RATES

SAVE_DIR = os.path.join(RESULTS_DIR, "saved_plans")
BATCH    = 64


@torch.no_grad()
def recovery_top1_top5(denoiser, A0_all, X_all, n_samples=512):
    denoiser.eval()
    idx = torch.randperm(len(A0_all))[:n_samples]
    A0  = A0_all[idx].to(DEVICE)
    X   = X_all[idx].to(DEVICE)

    results = {}
    for mr in MASK_RATES:
        A_t, mask_flag = mask_attn(A0, mr)
        logits = denoiser(X, A_t, mask_flag)          # [B, L, N_PATCHES]
        tgt    = A0.argmax(-1)                         # [B, L]
        m      = mask_flag.bool()

        if not m.any():
            results[mr] = (float('nan'), float('nan'))
            continue

        # Top-1
        pred1 = logits.argmax(-1)
        top1  = (pred1 == tgt)[m].float().mean().item()

        # Top-5
        top5_idx = logits.topk(5, dim=-1).indices     # [B, L, 5]
        tgt_exp  = tgt.unsqueeze(-1).expand_as(top5_idx)
        top5     = (top5_idx == tgt_exp).any(-1)[m].float().mean().item()

        results[mr] = (top1, top5)
    return results


def main():
    # ── Tokenizer ─────────────────────────────────────────────────────────────
    with open(os.path.join(SAVE_DIR, "tok_vocab.json")) as f:
        vocab = json.load(f)
    tok = Tokenizer()
    tok.vocab = vocab
    tok.w2i = {w: i for i, w in enumerate(vocab)}
    print(f"Vocab size: {len(tok.vocab)}")

    # ── Load saved models ─────────────────────────────────────────────────────
    proj = PatchProjector(ENCODER_DIM, D_MODEL).to(DEVICE)
    proj.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_proj.pt"),
                                    map_location=DEVICE, weights_only=True))
    proj.eval()

    free_dec = CaptionDecoder(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS,
                               n_src=N_PATCHES).to(DEVICE)
    free_dec.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_free.pt"),
                                         map_location=DEVICE, weights_only=True))
    free_dec.eval()

    denoiser = CFGDenoiser(D_MODEL, N_PATCHES, L_CAP, N_MP).to(DEVICE)
    denoiser.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_denoiser.pt"),
                                         map_location=DEVICE, weights_only=True))
    denoiser.eval()
    print("Models loaded")

    # ── Patch features ────────────────────────────────────────────────────────
    print("Loading cached patch features...")
    patch_feats = torch.load(CACHE_PATH, weights_only=True)
    test_start  = N_IMAGES + VAL_IMAGES
    test_feats  = patch_feats[test_start: test_start + TEST_IMAGES]
    print(f"  test_feats: {test_feats.shape}")

    # ── MSCOCO reference captions for test split ──────────────────────────────
    print("Loading MSCOCO test captions from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("clip-benchmark/wds_mscoco_captions", split="train")
    test_caps = []
    for item in ds.select(range(test_start, test_start + TEST_IMAGES)):
        caps = [c.strip() for c in item["txt"].strip().split("\n") if c.strip()]
        test_caps.append(caps[:N_CAP_PER])
    print(f"  Loaded {len(test_caps)} test images × up to {N_CAP_PER} captions")

    # ── Extract oracle A₀ from free decoder ───────────────────────────────────
    print("Extracting oracle A₀ from free decoder...")
    test_ds     = CocoCapDataset(test_feats, test_caps, tok, proj)
    test_loader = DataLoader(test_ds, BATCH, shuffle=False, collate_fn=collate_cap)
    A0_test, X_test, _, _ = extract_A0_cap(proj, free_dec, test_loader, DEVICE)
    print(f"  A0_test: {A0_test.shape},  X_test: {X_test.shape}")

    # ── Recovery: Top-1 and Top-5 ─────────────────────────────────────────────
    print("\nComputing Top-1 and Top-5 recovery (n_samples=512)...")
    results = recovery_top1_top5(denoiser, A0_test, X_test, n_samples=512)

    rand1 = 1.0 / N_PATCHES
    rand5 = 5.0 / N_PATCHES

    print(f"\nRandom Top-1 = 1/{N_PATCHES} = {rand1:.4f}")
    print(f"Random Top-5 = 5/{N_PATCHES} = {rand5:.4f}")
    print()
    print(f"{'Mask':>6}  {'Top-1':>8}  {'×rand':>8}  {'Top-5':>8}  {'×rand5':>8}")
    print("-" * 50)
    for mr in MASK_RATES:
        t1, t5 = results[mr]
        print(f"{mr:6.2f}  {t1:8.3f}  {t1/rand1:8.1f}x  {t5:8.3f}  {t5/rand5:8.1f}x")

    print("\n--- LaTeX rows for tab:recovery ---")
    for mr in MASK_RATES:
        t1, t5 = results[mr]
        print(f"{mr:.2f} & {t1:.3f} & {t1-rand1:+.3f} & {t5:.3f} \\\\")


if __name__ == "__main__":
    main()
