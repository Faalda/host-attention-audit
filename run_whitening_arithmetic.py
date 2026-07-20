"""
Idea 5 — Whitening diagnostic for plan-space arithmetic.
ZCA-whiten the plan manifold (anisotropy=22.4) before centroid arithmetic.
Tests whether failure is geometry (fixable) vs missing content (not fixable).
Proxy metric: nearest-neighbour noun match in original vs whitened space.

Usage:
    python3 run_whitening_arithmetic.py > results/whitening.log 2>&1
Prerequisite: run_save_plans.py must have completed.
CPU-only. ~5 minutes.
"""

import os, sys, json
import numpy as np
import torch
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(__file__))
from run_cap_improved import RESULTS_DIR, N_PATCHES

SAVE_DIR   = os.path.join(RESULTS_DIR, "saved_plans")
OUT_PATH   = os.path.join(RESULTS_DIR, "whitening_arithmetic.json")
NOUN_PAIRS = [("dog","cat"), ("chair","couch"), ("train","car"),
              ("pizza","sandwich"), ("horse","cow")]
K_NN       = 3    # neighbours for proxy noun vote
N_TEST     = 30   # max test images per pair


# ── Load plans ────────────────────────────────────────────────────────────────

def load_plans():
    tv   = torch.load(os.path.join(SAVE_DIR, "plans_tv.pt"),   weights_only=True)
    test = torch.load(os.path.join(SAVE_DIR, "plans_test.pt"), weights_only=True)
    with open(os.path.join(SAVE_DIR, "noun_labels_tv.json"))   as f: lbl_tv   = json.load(f)
    with open(os.path.join(SAVE_DIR, "noun_labels_test.json")) as f: lbl_test = json.load(f)
    # mean over L_cap → [N, N_patches]
    X_tv   = tv.mean(dim=1).numpy().astype(np.float32)
    X_test = test.mean(dim=1).numpy().astype(np.float32)
    return X_tv, lbl_tv, X_test, lbl_test


# ── ZCA whitening ─────────────────────────────────────────────────────────────

def fit_zca(X, eps=1e-5):
    mu   = X.mean(axis=0)
    X_c  = X - mu
    C    = (X_c.T @ X_c) / (len(X_c) - 1)
    eigvals, eigvecs = np.linalg.eigh(C)          # sorted ascending
    D_inv = np.diag(1.0 / np.sqrt(eigvals + eps))
    D_sq  = np.diag(np.sqrt(eigvals + eps))
    W_zca = eigvecs @ D_inv @ eigvecs.T           # whitening
    W_inv = eigvecs @ D_sq  @ eigvecs.T           # un-whitening
    return mu, W_zca, W_inv

def whiten(X, mu, W_zca):
    return (X - mu) @ W_zca.T

def unwhiten(X_w, mu, W_inv):
    return X_w @ W_inv.T + mu

def renorm(x):
    x = np.clip(x, 0, None)
    s = x.sum(-1, keepdims=True)
    s = np.where(s < 1e-8, 1.0, s)
    return x / s


# ── NN-based noun proxy ───────────────────────────────────────────────────────

def nn_noun(query, gallery, gallery_labels, k=K_NN):
    """Majority noun among k nearest gallery plans (L2). Returns noun or None."""
    dists = cdist(query[None], gallery, metric="euclidean")[0]
    idx   = np.argpartition(dists, k)[:k]
    nouns = [gallery_labels[i] for i in idx if gallery_labels[i] is not None]
    if not nouns:
        return None
    return max(set(nouns), key=nouns.count)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading plans...")
    X_tv, lbl_tv, X_test, lbl_test = load_plans()
    print(f"  train+val: {X_tv.shape}   test: {X_test.shape}")

    print("Fitting ZCA whitening on 14K train+val plans...")
    mu, W_zca, W_inv = fit_zca(X_tv)
    X_tv_w   = whiten(X_tv,   mu, W_zca)
    X_test_w = whiten(X_test, mu, W_zca)
    print("  Done.")

    results = {}

    for src_noun, tgt_noun in NOUN_PAIRS:
        src_idx_tv = [i for i, l in enumerate(lbl_tv) if l == src_noun]
        tgt_idx_tv = [i for i, l in enumerate(lbl_tv) if l == tgt_noun]
        test_idx   = [i for i, l in enumerate(lbl_test) if l == src_noun][:N_TEST]

        if not src_idx_tv or not tgt_idx_tv or not test_idx:
            print(f"  {src_noun}→{tgt_noun}: skipped (missing noun group)")
            continue

        # Centroids in original space
        c_src = X_tv[src_idx_tv].mean(axis=0)
        c_tgt = X_tv[tgt_idx_tv].mean(axis=0)

        # Centroids in whitened space
        c_src_w = whiten(c_src[None], mu, W_zca)[0]
        c_tgt_w = whiten(c_tgt[None], mu, W_zca)[0]

        match_orig = 0
        match_white = 0

        for ti in test_idx:
            x      = X_test[ti]
            x_w    = X_test_w[ti]

            # Original-space arithmetic
            x_edit_orig = renorm((x - c_src + c_tgt)[None])[0]

            # Whitened-space arithmetic → unwhiten
            x_edit_w    = x_w - c_src_w + c_tgt_w
            x_edit_uw   = unwhiten(x_edit_w[None], mu, W_inv)[0]
            x_edit_uw   = renorm(x_edit_uw[None])[0]

            # NN proxy in original space for original edit
            pred_orig  = nn_noun(x_edit_orig,  X_tv, lbl_tv)
            # NN proxy in whitened space for whitened edit
            pred_white = nn_noun(x_edit_w,     X_tv_w, lbl_tv)

            if pred_orig  == tgt_noun: match_orig  += 1
            if pred_white == tgt_noun: match_white += 1

        n  = len(test_idx)
        r_orig  = match_orig  / n
        r_white = match_white / n
        delta   = r_white - r_orig
        print(f"  {src_noun}→{tgt_noun}: n={n}  orig={r_orig:.3f}  white={r_white:.3f}  Δ={delta:+.3f}")

        results[f"{src_noun}→{tgt_noun}"] = {
            "n_test":                  n,
            "nn_noun_match_original":  round(r_orig,  4),
            "nn_noun_match_whitened":  round(r_white, 4),
            "delta":                   round(delta,   4),
        }

    if not results:
        print("No pairs evaluated — check noun labels in saved_plans/.")
        return

    mean_orig  = float(np.mean([v["nn_noun_match_original"] for v in results.values()]))
    mean_white = float(np.mean([v["nn_noun_match_whitened"] for v in results.values()]))
    mean_delta = mean_white - mean_orig

    if mean_delta > 0.05:
        interp = (f"whitening raises NN noun match by {mean_delta:.3f} → "
                  f"geometry (anisotropy) is a real contributor to arithmetic failure.")
    elif mean_delta > 0.01:
        interp = (f"whitening gives a small improvement ({mean_delta:.3f}) → "
                  f"geometry is a minor contributor; missing semantic content dominates.")
    else:
        interp = (f"whitening has no effect (Δ={mean_delta:.3f}) → "
                  f"arithmetic failure is NOT a geometry problem; missing content is the true bottleneck.")

    print(f"\n{'='*60}")
    print(f"WHITENING SUMMARY")
    print(f"  Original  mean NN noun match: {mean_orig:.4f}")
    print(f"  Whitened  mean NN noun match: {mean_white:.4f}")
    print(f"  Delta:                        {mean_delta:+.4f}")
    print(f"  Interpretation: {interp}")
    print(f"{'='*60}")

    out = {
        "whitening":  "ZCA",
        "k_nn":       K_NN,
        "noun_pairs": results,
        "summary": {
            "mean_original":  round(mean_orig,  4),
            "mean_whitened":  round(mean_white, 4),
            "mean_delta":     round(mean_delta, 4),
            "interpretation": interp,
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
