"""
Phase I Experiment 2: nearest-neighbor retrieval in plan space.

For each test plan A₀, retrieve the 5 nearest training plans by L2 distance.
Measure whether retrieved neighbors share location (quadrant) or noun class.
If location precision >> chance and noun precision ≈ chance → plan encodes WHERE not WHAT.

Usage:
    python3 run_plan_retrieval.py > results/retrieval.log 2>&1
Prerequisite: run_save_plans.py must have completed.
CPU-only, no GPU needed. Takes ~2 minutes.
"""

import os, sys, json
import numpy as np
import torch
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(__file__))
from run_cap_improved import RESULTS_DIR, N_PATCHES

SAVE_DIR   = os.path.join(RESULTS_DIR, "saved_plans")
OUT_PATH   = os.path.join(RESULTS_DIR, "plan_retrieval.json")
N_QUERIES  = 200
K          = 5
GRID_SIZE  = 16
SEED       = 42

rng = np.random.default_rng(SEED)


def build_patch_xy(grid=GRID_SIZE):
    return np.array([[col / (grid - 1), row / (grid - 1)]
                     for row in range(grid) for col in range(grid)],
                    dtype=np.float32)  # [256, 2]


def centroid_to_quadrant(xy):
    """xy: [N, 2] → [N] int in {0,1,2,3} (TL, TR, BL, BR)."""
    x, y = xy[:, 0], xy[:, 1]
    return (y >= 0.5).astype(int) * 2 + (x >= 0.5).astype(int)


def precision_at_k(match_matrix, k):
    """match_matrix: [N_queries, N_gallery] bool. Returns mean P@k."""
    top_k = match_matrix[:, :k]
    return float(top_k.sum(axis=1).mean() / k)


def main():
    # ── Load plans ────────────────────────────────────────────────────────────
    print("Loading plans...")
    plans_tv   = torch.load(os.path.join(SAVE_DIR, "plans_tv.pt"),   weights_only=True)
    plans_test = torch.load(os.path.join(SAVE_DIR, "plans_test.pt"), weights_only=True)
    print(f"  gallery: {plans_tv.shape}  queries: {plans_test.shape}")

    with open(os.path.join(SAVE_DIR, "noun_labels_tv.json"))   as f: nouns_tv   = json.load(f)
    with open(os.path.join(SAVE_DIR, "noun_labels_test.json")) as f: nouns_test = json.load(f)

    # ── Mean over L_cap → [N, N_patches] ─────────────────────────────────────
    gal = plans_tv.mean(dim=1).numpy().astype(np.float32)    # [14000, 256]
    qry_all = plans_test.mean(dim=1).numpy().astype(np.float32)  # [1000, 256]

    # ── Compute centroids + quadrants ─────────────────────────────────────────
    patch_xy = build_patch_xy()                          # [256, 2]
    gal_ctr  = gal @ patch_xy                            # [14000, 2]
    qry_ctr  = qry_all @ patch_xy                        # [1000, 2]

    gal_quad = centroid_to_quadrant(gal_ctr)             # [14000]
    qry_quad = centroid_to_quadrant(qry_ctr)             # [1000]

    nouns_tv_arr   = np.array([n if n else "" for n in nouns_tv])
    nouns_test_arr = np.array([n if n else "" for n in nouns_test])

    unique_nouns = sorted(set(nouns_tv_arr) - {""})
    n_unique     = len(unique_nouns)
    chance_noun  = 1.0 / n_unique if n_unique else 0.0
    print(f"  unique nouns in gallery: {n_unique}")
    print(f"  chance_noun={chance_noun:.4f}  chance_location=0.25")

    # ── Select query subset ───────────────────────────────────────────────────
    idx_q = rng.choice(len(qry_all), size=min(N_QUERIES, len(qry_all)), replace=False)
    qry   = qry_all[idx_q]
    q_quad = qry_quad[idx_q]
    q_noun = nouns_test_arr[idx_q]

    # ── Compute full distance matrix ──────────────────────────────────────────
    print(f"Computing L2 distances ({len(idx_q)} queries × {len(gal)} gallery)...")
    dist_mat = cdist(qry, gal, metric="euclidean")       # [N_q, 14000]
    nn_idx   = np.argsort(dist_mat, axis=1)              # [N_q, 14000] sorted

    # ── Nearest-neighbor metrics ──────────────────────────────────────────────
    results_nn = {}
    for k_val in (1, K):
        top_k_idx = nn_idx[:, :k_val]                    # [N_q, k_val]

        # location precision
        gal_quad_top = gal_quad[top_k_idx]               # [N_q, k_val]
        loc_match    = gal_quad_top == q_quad[:, None]   # [N_q, k_val]
        loc_prec     = float(loc_match.mean())

        # noun precision (only queries with non-empty noun)
        q_has_noun   = q_noun != ""
        gal_noun_top = nouns_tv_arr[top_k_idx]           # [N_q, k_val]
        noun_match   = gal_noun_top == q_noun[:, None]   # [N_q, k_val]
        if q_has_noun.sum() > 0:
            noun_prec = float(noun_match[q_has_noun].mean())
        else:
            noun_prec = 0.0

        results_nn[f"noun_precision_at_{k_val}"]     = round(noun_prec, 4)
        results_nn[f"location_precision_at_{k_val}"] = round(loc_prec,  4)

        print(f"  K={k_val}: loc_prec={loc_prec:.4f}  noun_prec={noun_prec:.4f}")

    # ── Random baseline ───────────────────────────────────────────────────────
    print("Computing random baseline...")
    rand_nn   = {}
    n_rand    = min(100, len(idx_q))
    idx_rand  = idx_q[:n_rand]
    q_rand    = qry_all[idx_rand]
    q_rand_quad = qry_quad[idx_rand]
    q_rand_noun = nouns_test_arr[idx_rand]
    q_rand_has_noun = q_rand_noun != ""

    # random K indices from gallery (not nearest)
    rand_gal_idx = rng.integers(0, len(gal), size=(n_rand, K))
    rand_quad    = gal_quad[rand_gal_idx]
    rand_noun    = nouns_tv_arr[rand_gal_idx]

    rand_loc_prec  = float((rand_quad == q_rand_quad[:, None]).mean())
    if q_rand_has_noun.sum() > 0:
        rand_noun_prec = float((rand_noun == q_rand_noun[:, None])[q_rand_has_noun].mean())
    else:
        rand_noun_prec = 0.0

    rand_nn["noun_precision_at_5"]     = round(rand_noun_prec, 4)
    rand_nn["location_precision_at_5"] = round(rand_loc_prec,  4)
    print(f"  random: loc={rand_loc_prec:.4f}  noun={rand_noun_prec:.4f}")

    # ── Interpretation ────────────────────────────────────────────────────────
    loc_at5  = results_nn["location_precision_at_5"]
    noun_at5 = results_nn["noun_precision_at_5"]
    loc_lift  = loc_at5 - 0.25
    noun_lift = noun_at5 - chance_noun

    if loc_lift > noun_lift and loc_at5 > 0.30:
        interp = (f"NN location precision ({loc_at5:.2f}) >> chance (0.25); "
                  f"noun precision ({noun_at5:.2f}) ≈ chance ({chance_noun:.2f}): "
                  f"plan encodes WHERE not WHAT")
    elif noun_lift > loc_lift and noun_at5 > chance_noun * 1.5:
        interp = (f"NN noun precision ({noun_at5:.2f}) >> chance ({chance_noun:.2f}); "
                  f"location precision ({loc_at5:.2f}) near chance (0.25): "
                  f"plan encodes WHAT not WHERE")
    elif loc_lift > 0.05 and noun_lift > chance_noun * 0.5:
        interp = (f"Plan encodes both location ({loc_at5:.2f} vs 0.25) and "
                  f"noun ({noun_at5:.2f} vs {chance_noun:.2f})")
    else:
        interp = (f"Weak signal: loc={loc_at5:.2f} (chance 0.25), "
                  f"noun={noun_at5:.2f} (chance {chance_noun:.2f})")

    print(f"\n{interp}")

    out = {
        "k": K,
        "n_queries": int(len(idx_q)),
        "n_gallery": int(len(gal)),
        "n_unique_nouns": n_unique,
        "nearest_neighbor": {
            **results_nn,
            "chance_noun":     round(chance_noun, 4),
            "chance_location": 0.25,
        },
        "random_baseline": rand_nn,
        "interpretation": interp,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
