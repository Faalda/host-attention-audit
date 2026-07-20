"""
Generate prediction-gap distribution figure (Figure 3 replacement).
Overlapping histograms of per-image ROUGE-1 for denoised vs oracle plans.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCORES_PATH = os.path.join(os.path.dirname(__file__),
                           "results", "saved_plans", "inversion_scores_n1000.json")
OUT_PDF = os.path.join(os.path.dirname(__file__),
                       "figures", "prediction_gap_hist.pdf")
OUT_PNG = os.path.join(os.path.dirname(__file__),
                       "figures", "prediction_gap_hist.png")

with open(SCORES_PATH) as f:
    d = json.load(f)

base = np.array(d["baseline_scores"])
ora  = np.array(d["oracle_scores"])
gaps = ora - base

print(f"n={len(base)}")
print(f"Baseline  mean={base.mean():.4f}  std={base.std():.4f}")
print(f"Oracle    mean={ora.mean():.4f}  std={ora.std():.4f}")
print(f"Gap       mean={gaps.mean():.4f}  std={gaps.std():.4f}")
print(f"Gap > 0:  {(gaps > 0).mean():.1%}")
print(f"Gap > 0.1:{(gaps > 0.1).mean():.1%}")

# ── Figure ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
fig.subplots_adjust(wspace=0.35)

BLUE  = "#2980b9"
GREEN = "#27ae60"
ALPHA = 0.62
BINS  = 28

# Left panel: overlapping ROUGE distributions
ax = axes[0]
bins = np.linspace(0.0, 1.0, BINS + 1)
ax.hist(base, bins=bins, color=BLUE,  alpha=ALPHA, label="Denoised plan",  density=True)
ax.hist(ora,  bins=bins, color=GREEN, alpha=ALPHA, label="Oracle inverted", density=True)
ax.axvline(base.mean(), color=BLUE,  lw=1.5, ls="--")
ax.axvline(ora.mean(),  color=GREEN, lw=1.5, ls="--")
ax.set_xlabel("ROUGE-1", fontsize=9)
ax.set_ylabel("Density", fontsize=9)
ax.set_title("Per-image ROUGE-1\n(denoised vs. oracle inverted)", fontsize=9)
ax.legend(fontsize=7.5, framealpha=0.8)
ax.tick_params(labelsize=8)

# Right panel: gap distribution
ax2 = axes[1]
gap_bins = np.linspace(-0.5, 0.8, BINS + 1)
ax2.hist(gaps, bins=gap_bins, color="#8e44ad", alpha=0.75, density=True)
ax2.axvline(0,           color="black", lw=1.0, ls=":", label="No gain")
ax2.axvline(gaps.mean(), color="#8e44ad", lw=1.5, ls="--",
            label=f"Mean $+{gaps.mean():.3f}$")
ax2.set_xlabel("ROUGE-1 gap (oracle $-$ denoised)", fontsize=9)
ax2.set_ylabel("Density", fontsize=9)
ax2.set_title(f"Prediction gap distribution\n"
              f"({(gaps>0).mean():.0%} of images benefit, $n{{=}}1000$)",
              fontsize=9)
ax2.legend(fontsize=7.5, framealpha=0.8)
ax2.tick_params(labelsize=8)

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
fig.savefig(OUT_PNG, bbox_inches="tight", dpi=200)
print(f"\nSaved → {OUT_PDF}")
print(f"Saved → {OUT_PNG}")
