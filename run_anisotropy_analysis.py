"""
Covariance anisotropy analysis of per-noun A₀ plan distributions.
Tests whether high anisotropy predicts arithmetic failure.

Usage:
    python3 run_anisotropy_analysis.py > results/anisotropy.log 2>&1
    Prerequisite: run_save_plans.py must have completed.
    CPU-only (uses numpy eigvalsh). Takes ~5 minutes for 14K plans.
"""

import os, sys, json
import numpy as np
import torch
from collections import defaultdict
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(__file__))
from run_cap_improved import RESULTS_DIR, N_PATCHES

SAVE_DIR = os.path.join(RESULTS_DIR, "saved_plans")
ARITH_PATH = os.path.join(RESULTS_DIR, "plan_arithmetic.json")
OUT_PATH = os.path.join(RESULTS_DIR, "anisotropy_analysis.json")

# 16x16 grid -> normalized patch coordinates [0,1]
patch_xy = np.array([[col / 15.0, row / 15.0]
                     for row in range(16) for col in range(16)],
                    dtype=np.float32)  # [256, 2]


def compute_anisotropy(X):
    """
    X: [n_samples, N_patches] float32 numpy array (already mean-pooled over L_cap)
    Returns dict with anisotropy, condition_number, effective_rank, eigenvalues_top3
    """
    n = X.shape[0]
    X_c = X - X.mean(axis=0)
    C = (X_c.T @ X_c) / (n - 1)  # [N_patches, N_patches]
    C += 1e-8 * np.eye(N_PATCHES)
    eigvals = np.linalg.eigvalsh(C)  # sorted ascending
    eigvals = np.clip(eigvals, 0, None)

    lam_max = eigvals[-1]
    lam_mean = eigvals.mean()
    lam_min = max(eigvals[0], 1e-12)

    anisotropy = float(lam_max / lam_mean) if lam_mean > 0 else 0.0
    condition_number = float(lam_max / lam_min)

    # effective rank = exp(entropy of normalized eigenvalue spectrum)
    total = eigvals.sum()
    if total > 0:
        p = eigvals / total
        p = p[p > 0]
        eff_rank = float(np.exp(-np.sum(p * np.log(p))))
    else:
        eff_rank = 1.0

    top3 = eigvals[-3:][::-1].tolist()
    return {
        "anisotropy": round(anisotropy, 4),
        "condition_number": round(condition_number, 4),
        "effective_rank": round(eff_rank, 4),
        "eigenvalues_top3": [round(v, 6) for v in top3],
    }


def main():
    print("Loading plans...")
    plans_tv = torch.load(os.path.join(SAVE_DIR, "plans_tv.pt"),
                          weights_only=True)  # [14000, L_cap, N_patches]
    print(f"  plans_tv: {plans_tv.shape}")

    with open(os.path.join(SAVE_DIR, "noun_labels_tv.json")) as f:
        noun_labels = json.load(f)

    # Mean over L_cap -> [14000, N_patches]
    print("Computing per-image plan means...")
    X_all = plans_tv.mean(dim=1).numpy()  # [14000, N_patches]
    print(f"  X_all: {X_all.shape}")

    # Group by noun
    noun_to_indices = defaultdict(list)
    for i, noun in enumerate(noun_labels):
        if noun is not None:
            noun_to_indices[noun].append(i)

    print(f"\nFound {len(noun_to_indices)} unique nouns.")

    # Compute anisotropy per noun (>=20 samples)
    per_noun = {}
    for noun, idxs in sorted(noun_to_indices.items(), key=lambda x: -len(x[1])):
        if len(idxs) < 20:
            continue
        X_noun = X_all[idxs]  # [n, N_patches]
        stats = compute_anisotropy(X_noun)
        # spatial centroid: mean plan -> weighted centroid
        mean_plan = X_noun.mean(axis=0)  # [N_patches]
        centroid = (mean_plan[:, None] * patch_xy).sum(axis=0)  # [2]
        stats["n_samples"] = len(idxs)
        stats["centroid_xy"] = [round(float(centroid[0]), 4), round(float(centroid[1]), 4)]
        per_noun[noun] = stats
        print(f"  {noun:20s} n={len(idxs):5d}  anisotropy={stats['anisotropy']:.2f}  "
              f"eff_rank={stats['effective_rank']:.1f}  "
              f"centroid=({centroid[0]:.2f},{centroid[1]:.2f})")

    print(f"\nComputed anisotropy for {len(per_noun)} nouns with >=20 samples.")

    # Correlation with arithmetic results
    arith_pairs = []
    if os.path.exists(ARITH_PATH):
        with open(ARITH_PATH) as f:
            arith = json.load(f)

        # Handle both list and dict formats
        if isinstance(arith, list):
            pair_list = arith
        elif "results" in arith:
            pair_list = list(arith["results"].values())
        elif "pairs" in arith:
            pair_list = arith["pairs"]
        else:
            pair_list = list(arith.values()) if isinstance(arith, dict) else []

        for entry in pair_list:
            src = entry.get("src_noun") or entry.get("source_noun") or entry.get("src")
            tgt = entry.get("tgt_noun") or entry.get("target_noun") or entry.get("tgt")
            hit_tgt = entry.get("hit_tgt_rate") or entry.get("target_rate")
            if hit_tgt is None:
                n_hit = entry.get("n_hit_tgt", 0)
                n_total = entry.get("n_total", entry.get("n_test", 0))
                hit_tgt = n_hit / n_total if n_total > 0 else 0.0
            if src and tgt and src in per_noun and tgt in per_noun:
                mean_aniso = (per_noun[src]["anisotropy"] + per_noun[tgt]["anisotropy"]) / 2.0
                arith_pairs.append({
                    "src": src,
                    "tgt": tgt,
                    "mean_anisotropy": round(float(mean_aniso), 4),
                    "arithmetic_success": round(float(hit_tgt), 4),
                })
    else:
        print(f"  Warning: {ARITH_PATH} not found; skipping arithmetic correlation.")

    if len(arith_pairs) >= 2:
        x = np.array([p["mean_anisotropy"] for p in arith_pairs])
        y = np.array([p["arithmetic_success"] for p in arith_pairs])
        r, pval = pearsonr(x, y)
        interp = (f"higher anisotropy predicts {'lower' if r < 0 else 'higher'} "
                  f"arithmetic success (r={r:.3f}, p={pval:.3f})")
        print(f"\nArithmetic correlation: r={r:.3f}, p={pval:.3f}  [{interp}]")
        arith_corr = {
            "pairs": arith_pairs,
            "pearson_r": round(float(r), 4),
            "p_value": round(float(pval), 4),
            "n_pairs": len(arith_pairs),
            "interpretation": interp,
        }
    else:
        arith_corr = {
            "pairs": arith_pairs,
            "pearson_r": None,
            "n_pairs": len(arith_pairs),
            "interpretation": "Insufficient pairs for correlation",
        }

    # Summary stats
    all_aniso = [v["anisotropy"] for v in per_noun.values()]
    most_aniso = max(per_noun, key=lambda n: per_noun[n]["anisotropy"]) if per_noun else ""
    least_aniso = min(per_noun, key=lambda n: per_noun[n]["anisotropy"]) if per_noun else ""

    summary = {
        "n_nouns_analyzed": len(per_noun),
        "mean_anisotropy_all_nouns": round(float(np.mean(all_aniso)), 4) if all_aniso else 0.0,
        "most_anisotropic": most_aniso,
        "most_anisotropic_score": round(per_noun[most_aniso]["anisotropy"], 4) if most_aniso else 0.0,
        "least_anisotropic": least_aniso,
        "least_anisotropic_score": round(per_noun[least_aniso]["anisotropy"], 4) if least_aniso else 0.0,
        "interpretation": (
            "High anisotropy indicates plan-space curvature; additive centroid "
            "arithmetic is expected to fail in elongated (non-isotropic) manifolds."
        ),
    }

    print(f"\nSummary:")
    print(f"  Mean anisotropy: {summary['mean_anisotropy_all_nouns']:.2f}")
    print(f"  Most anisotropic: {most_aniso} ({summary['most_anisotropic_score']:.2f})")
    print(f"  Least anisotropic: {least_aniso} ({summary['least_anisotropic_score']:.2f})")

    result = {
        "per_noun": per_noun,
        "arithmetic_correlation": arith_corr,
        "summary": summary,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
