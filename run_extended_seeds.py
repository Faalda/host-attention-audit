"""
run_extended_seeds.py — Extend the GNN vs cross-attn seed comparison to n=10.

Existing results (seeds 42, 7, 13) are in:
  results/cap_improved_dinov2_15k_sal.json   (GNN)
  results/cap_xattn_dinov2_15k.json          (cross-attn)

This script adds seeds [0, 1, 2, 3, 4, 5, 6] for both architectures, saving to:
  results/cap_improved_extended.json         (GNN new seeds)
  results/cap_xattn_extended.json            (cross-attn new seeds)
Seeds 0-4 are already cached from the earlier n=8 run; only 5 and 6 will
actually run (the resume logic skips indices already present in the file).

After both finish, combine_seeds() prints the full n=10 t-test result.

Usage:
    python3 run_extended_seeds.py
    python3 run_extended_seeds.py --arch gnn
    python3 run_extended_seeds.py --arch xattn
    python3 run_extended_seeds.py --combine-only
"""

import os, sys, json, argparse
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))

NEW_SEEDS = [0, 1, 2, 3, 4, 5, 6]
RESULTS_DIR_BASE = os.path.join(os.path.dirname(__file__), "results")
GNN_OLD   = os.path.join(RESULTS_DIR_BASE, "cap_improved_dinov2_15k_sal.json")
XATTN_OLD = os.path.join(RESULTS_DIR_BASE, "cap_xattn_dinov2_15k.json")
GNN_NEW   = os.path.join(RESULTS_DIR_BASE, "cap_improved_extended.json")
XATTN_NEW = os.path.join(RESULTS_DIR_BASE, "cap_xattn_extended.json")


def combine_and_report():
    """Combine old + new seeds and run paired t-test."""
    with open(GNN_OLD) as f:   gnn_old   = json.load(f)["scores"]
    with open(XATTN_OLD) as f: xattn_old = json.load(f)["scores"]

    gnn_deltas   = [s["K4"] - s["free"] for s in gnn_old]
    xattn_deltas = [s["K4"] - s["free"] for s in xattn_old]

    if os.path.exists(GNN_NEW):
        with open(GNN_NEW) as f:
            gnn_new = json.load(f)["scores"]
        gnn_deltas += [s["K4"] - s["free"] for s in gnn_new]

    if os.path.exists(XATTN_NEW):
        with open(XATTN_NEW) as f:
            xattn_new = json.load(f)["scores"]
        xattn_deltas += [s["K4"] - s["free"] for s in xattn_new]

    n = min(len(gnn_deltas), len(xattn_deltas))
    gnn_arr   = np.array(gnn_deltas[:n])
    xattn_arr = np.array(xattn_deltas[:n])

    print(f"\n{'='*60}")
    print(f"Combined results: n={n} seeds each")
    print(f"  GNN   deltas: {[f'{d:+.4f}' for d in gnn_arr]}")
    print(f"  XAttn deltas: {[f'{d:+.4f}' for d in xattn_arr]}")
    print(f"  GNN   mean ± std: {gnn_arr.mean():+.4f} ± {gnn_arr.std():.4f}")
    print(f"  XAttn mean ± std: {xattn_arr.mean():+.4f} ± {xattn_arr.std():.4f}")

    if n >= 3:
        t, p = stats.ttest_rel(gnn_arr, xattn_arr, alternative='greater')
        print(f"  Paired one-sided t-test: t={t:.3f}, p={p:.4f} (H1: GNN > xattn)")
        if p < 0.05:
            print(f"  *** SIGNIFICANT at p<0.05 ***")
        elif p < 0.1:
            print(f"  ~ Marginal (p<0.10)")
        else:
            print(f"  Not significant (p={p:.4f})")

    print(f"\n--- LaTeX rows (Table 4 update) ---")
    print(f"GNN-style (659K)   & "
          f"${gnn_arr.mean():+.4f}\\!\\pm\\!{gnn_arr.std():.4f}$ \\\\")
    print(f"Cross-attn (2.24M) & "
          f"${xattn_arr.mean():+.4f}\\!\\pm\\!{xattn_arr.std():.4f}$ \\\\")
    print(f"n={n} seeds, p={p:.4f}")


def run_gnn_seeds(patch_feats, caps_per_img, tok):
    from run_cap_improved import run_seed
    results = {"scores": [], "recovery": []}
    existing = {}
    if os.path.exists(GNN_NEW):
        with open(GNN_NEW) as f:
            existing = json.load(f)
        results = existing
        done = len(results["scores"])
        print(f"  Resuming GNN from seed index {done}/{len(NEW_SEEDS)}")
    else:
        done = 0

    for i, seed in enumerate(NEW_SEEDS):
        if i < done:
            print(f"  GNN seed {seed}: already done, skipping")
            continue
        print(f"\n  GNN seed {seed} ({i+1}/{len(NEW_SEEDS)})...")
        scores, rec = run_seed(seed, patch_feats, caps_per_img, tok)
        results["scores"].append(scores)
        results["recovery"].append(rec)
        with open(GNN_NEW, "w") as f:
            json.dump(results, f, indent=2)
        delta = scores["K4"] - scores["free"]
        print(f"  → ROUGE delta = {delta:+.4f}")


def run_xattn_seeds(patch_feats, caps_per_img, tok):
    from run_cap_xattn import run_seed
    results = {"scores": [], "recovery": []}
    existing = {}
    if os.path.exists(XATTN_NEW):
        with open(XATTN_NEW) as f:
            existing = json.load(f)
        results = existing
        done = len(results["scores"])
        print(f"  Resuming XAttn from seed index {done}/{len(NEW_SEEDS)}")
    else:
        done = 0

    for i, seed in enumerate(NEW_SEEDS):
        if i < done:
            print(f"  XAttn seed {seed}: already done, skipping")
            continue
        print(f"\n  XAttn seed {seed} ({i+1}/{len(NEW_SEEDS)})...")
        scores, rec = run_seed(seed, patch_feats, caps_per_img, tok)
        results["scores"].append(scores)
        results["recovery"].append(rec)
        with open(XATTN_NEW, "w") as f:
            json.dump(results, f, indent=2)
        delta = scores["K4"] - scores["free"]
        print(f"  → ROUGE delta = {delta:+.4f}")


def load_data():
    import torch
    from datasets import load_dataset
    from run_summarization import Tokenizer
    from run_cap_improved import N_IMAGES, VAL_IMAGES, CACHE_PATH
    from run_cap_xattn import TEST_IMAGES
    from run_captioning import N_CAP_PER

    print("Loading MSCOCO captions...")
    ds = load_dataset("clip-benchmark/wds_mscoco_captions", split="train")
    total = N_IMAGES + VAL_IMAGES + TEST_IMAGES
    caps_per_img = []
    for item in ds.select(range(total)):
        caps = [c.strip() for c in item["txt"].strip().split("\n") if c.strip()]
        caps_per_img.append(caps[:N_CAP_PER])

    print(f"Loading patch features from {CACHE_PATH}...")
    patch_feats = torch.load(CACHE_PATH, weights_only=True)
    print(f"  patch_feats: {patch_feats.shape}")

    tok = Tokenizer()
    tok.build([cap for caps in caps_per_img for cap in caps], max_vocab=6000)
    print(f"  vocab: {len(tok.vocab)}")

    return patch_feats, caps_per_img, tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["gnn", "xattn", "both"],
                        default="both", help="Which architecture to run")
    parser.add_argument("--combine-only", action="store_true",
                        help="Skip training, just print combined t-test")
    args = parser.parse_args()

    if args.combine_only:
        combine_and_report()
        return

    patch_feats, caps_per_img, tok = load_data()

    if args.arch in ("gnn", "both"):
        print("\n" + "="*60)
        print("Running GNN seeds:", NEW_SEEDS)
        print("="*60)
        run_gnn_seeds(patch_feats, caps_per_img, tok)

    if args.arch in ("xattn", "both"):
        print("\n" + "="*60)
        print("Running Cross-Attn seeds:", NEW_SEEDS)
        print("="*60)
        run_xattn_seeds(patch_feats, caps_per_img, tok)

    combine_and_report()


if __name__ == "__main__":
    main()
