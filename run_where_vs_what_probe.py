"""
"Where vs. What" linear probe: does A₀ encode spatial location or noun class?

Train two probes from per-image plan distributions:
  1. Location probe (LinearRegression): [N_patches] → attention centroid [2]   → R²
  2. Class probe   (LogisticRegression): [N_patches] → dominant COCO noun      → accuracy

Expected result: location R² high (>0.7), class accuracy ≈ chance (1/n_classes).

Usage:
    python3 run_where_vs_what_probe.py > results/probe.log 2>&1
    Prerequisite: run_save_plans.py must have completed.
    CPU-only, no GPU needed.
"""

import os, sys, json
import numpy as np
import torch
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from run_cap_improved import RESULTS_DIR, N_PATCHES

SAVE_DIR = os.path.join(RESULTS_DIR, "saved_plans")
OUT_PATH = os.path.join(RESULTS_DIR, "where_vs_what_probe.json")

GRID_SIZE = 16      # 16x16 = 256 patches for DINOv2
TOP_N_NOUNS = 20    # keep only top-20 noun classes for class probe


def build_patch_xy(grid_size=GRID_SIZE):
    """Returns [N_patches, 2] normalized patch coordinates in [0,1]."""
    coords = []
    for row in range(grid_size):
        for col in range(grid_size):
            coords.append([col / (grid_size - 1), row / (grid_size - 1)])
    return np.array(coords, dtype=np.float32)


def load_plans(save_dir):
    print("Loading saved plans...")
    plans_tv   = torch.load(os.path.join(save_dir, "plans_tv.pt"),   weights_only=True)
    plans_test = torch.load(os.path.join(save_dir, "plans_test.pt"), weights_only=True)
    print(f"  plans_tv:   {plans_tv.shape}")
    print(f"  plans_test: {plans_test.shape}")
    return plans_tv, plans_test


def compute_features(plans_tensor):
    """
    Compute per-image plan features.

    plans_tensor: [N, L_cap, N_patches]
    Returns:
      X:        [N, N_patches] -- mean attention distribution over caption positions
      centroid: [N, 2]         -- attention-weighted spatial centroid
    """
    patch_xy = build_patch_xy()              # [256, 2]
    X = plans_tensor.mean(dim=1).numpy()     # [N, 256]
    centroid = X @ patch_xy                  # [N, 2]
    return X.astype(np.float32), centroid.astype(np.float32)


def run_location_probe(X_tr, ctr_tr, X_te, ctr_te):
    from sklearn.linear_model import LinearRegression

    print("\nFitting location probe (LinearRegression) ...")
    reg = LinearRegression()
    reg.fit(X_tr, ctr_tr)
    r2_train = float(reg.score(X_tr, ctr_tr))
    r2_test  = float(reg.score(X_te, ctr_te))
    print(f"  R2 train: {r2_train:.4f}")
    print(f"  R2 test:  {r2_test:.4f}")
    return {"r2_train": r2_train, "r2_test": r2_test,
            "n_train": int(len(X_tr)), "n_test": int(len(X_te))}


def run_class_probe(X_tr, labels_tr, X_te, labels_te, top_n=TOP_N_NOUNS):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    tr_mask = [i for i, l in enumerate(labels_tr) if l is not None]
    te_mask = [i for i, l in enumerate(labels_te) if l is not None]
    print(f"\nLabeled: {len(tr_mask)} train, {len(te_mask)} test")

    if len(tr_mask) < 50:
        print("  Too few labeled examples -- skipping class probe")
        return {"error": "too few labeled examples"}

    noun_counts = Counter(labels_tr[i] for i in tr_mask)
    top_nouns   = {n for n, _ in noun_counts.most_common(top_n)}
    print(f"  Top-{top_n} nouns: {sorted(top_nouns)}")

    tr_final = [i for i in tr_mask if labels_tr[i] in top_nouns]
    te_final = [i for i in te_mask if labels_te[i] in top_nouns]
    print(f"  After filtering to top-{top_n}: {len(tr_final)} train, {len(te_final)} test")

    if len(te_final) < 10:
        print("  Too few test examples -- skipping class probe")
        return {"error": "too few test examples after noun filtering"}

    X_tr_cls = X_tr[tr_final]
    X_te_cls = X_te[te_final]
    y_tr_raw = [labels_tr[i] for i in tr_final]
    y_te_raw = [labels_te[i] for i in te_final]

    le = LabelEncoder()
    le.fit(y_tr_raw)
    y_tr = le.transform(y_tr_raw)
    # handle unseen test labels gracefully
    y_te = le.transform([l if l in set(le.classes_) else le.classes_[0] for l in y_te_raw])

    n_classes = len(le.classes_)
    chance    = 1.0 / n_classes

    print(f"  Fitting LogisticRegression ({n_classes} classes, chance={chance:.3f}) ...")
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(X_tr_cls, y_tr)
    acc_train = float(clf.score(X_tr_cls, y_tr))
    acc_test  = float(clf.score(X_te_cls, y_te))
    print(f"  Accuracy train: {acc_train:.4f}")
    print(f"  Accuracy test:  {acc_test:.4f}  (chance: {chance:.4f})")

    return {
        "test_accuracy":   acc_test,
        "train_accuracy":  acc_train,
        "chance_level":    chance,
        "n_classes":       n_classes,
        "n_train_labeled": len(tr_final),
        "n_test_labeled":  len(te_final),
        "top_classes":     list(le.classes_),
    }


def main():
    plans_tv, plans_test = load_plans(SAVE_DIR)

    with open(os.path.join(SAVE_DIR, "noun_labels_tv.json"))   as f:
        labels_tv   = json.load(f)
    with open(os.path.join(SAVE_DIR, "noun_labels_test.json")) as f:
        labels_test = json.load(f)

    print(f"  noun_labels_tv:   {len(labels_tv)} entries")
    print(f"  noun_labels_test: {len(labels_test)} entries")

    print("\nComputing features (mean plan distribution + centroid) ...")
    X_tv,   ctr_tv   = compute_features(plans_tv)
    X_test, ctr_test = compute_features(plans_test)
    print(f"  X_tv:   {X_tv.shape},  ctr_tv:   {ctr_tv.shape}")
    print(f"  X_test: {X_test.shape}, ctr_test: {ctr_test.shape}")

    loc_results = run_location_probe(X_tv, ctr_tv, X_test, ctr_test)
    cls_results = run_class_probe(X_tv, labels_tv, X_test, labels_test)

    r2  = loc_results.get("r2_test", 0.0)
    acc = cls_results.get("test_accuracy", 0.0)
    chn = cls_results.get("chance_level", 0.05)
    interpretation = (
        f"location R2={r2:.2f} ({'high' if r2 > 0.7 else 'moderate'}), "
        f"class accuracy={acc:.2f} (chance={chn:.2f}): "
        f"plans encode WHERE {'strongly' if r2 > 0.7 else 'moderately'}, "
        f"class {'at chance' if acc < chn * 1.5 else 'above chance'}"
    )
    print(f"\n{interpretation}")

    results = {
        "location":       loc_results,
        "class":          cls_results,
        "interpretation": interpretation,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
