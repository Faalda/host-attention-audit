"""Plot the constructed positive control from immutable JSONs, on CPU only.

From the project root:
    CUDA_VISIBLE_DEVICES='' ../.venv/bin/python -B analysis/make_visual_recovery_figure.py

Forward captions are scored against all five recipient references (max R1),
not the held-out protocol. Stored CIs condition on three fixed models and a
fixed donor permutation, averaging seeds before resampling recipient images.
Fault-aware repair uses the known preprocessing mask, not image-only correction.
No inference, new bootstrap draws, or source modifications are performed.
Source SHA-256 receipts are printed and embedded in both output files.
"""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ROOT / "results/visual_headroom_provenance/results.json",
    ROOT / "results/visual_headroom_control/results.json",
)
OUTPUT = ROOT / "analysis/figures/fig_visual_recovery"
PROTOCOL = "each_inversion_all_refs_avg"
CONTRASTS = (
    "own_repair_minus_fault",
    "donor_repair_minus_fault",
    "null_repair_minus_fault",
    "interaction",
)
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
    "text.color": "#252525",
    "axes.labelcolor": "#252525",
    "xtick.color": "#454545",
    "ytick.color": "#252525",
}


def validate_sources(provenance, control):
    """Reject wrong protocols, malformed summaries, and inconsistent image pairs."""
    def require(ok, message):
        if not ok:
            raise ValueError(message)

    def close(actual, expected, message):
        require(np.allclose(actual, expected, rtol=0, atol=1e-12), message)

    require(provenance["status"] == control["status"] == "complete", "Incomplete sources")
    require(provenance["all_original_files_and_inputs_unchanged"] is True
            and control["all_consumed_inputs_unchanged"] is True, "Source integrity check failed")
    seeds = tuple(provenance["per_seed_results"])
    require(set(seeds) == set(control["per_seed_image_scores"]) == {"42", "7", "13"},
            "Expected the same three frozen models")
    require(provenance["uncertainty"] == (
        "10000 paired recipient-image bootstrap draws, seed0, after averaging three fixed seeds; "
        "fixed donor permutation"), "Unexpected provenance interval scope")
    require(control["interval_scope"] == (
        "paired images, five slots then three fixed seeds averaged; not training uncertainty"),
        "Unexpected control interval scope")

    rows = {key: provenance["comparisons"][key] for key in CONTRASTS}
    n = rows[CONTRASTS[0]]["n_images"]
    require(n == 1000, "Expected the full 1000-image cohort")
    pairs = {}
    for key, row in rows.items():
        require(row["n_images"] == n, f"Image count mismatch: {key}")
        summary = np.asarray([row["inv"], row["base"], row["delta"]], dtype=float)
        ci = np.asarray(row["ci95"], dtype=float)
        require(ci.shape == (2,) and np.isfinite(ci).all()
                and np.isfinite(summary).all(), f"Invalid summary/CI: {key}")
        require(ci[0] <= row["delta"] <= ci[1], f"Unordered CI: {key}")
        close(row["inv"] - row["base"], row["delta"], f"Delta arithmetic: {key}")
        pair = np.asarray([row["per_image"][side] for side in ("inv", "base")], dtype=float)
        require(pair.shape == (2, n) and np.isfinite(pair).all(), f"Invalid image pairs: {key}")
        close(pair.mean(axis=1), summary[:2], f"Image means: {key}")
        close((pair[0] - pair[1]).mean(), row["delta"], f"Image delta: {key}")
        pairs[key] = pair

    own, donor, null, interaction = (rows[key] for key in CONTRASTS)
    for i, key in enumerate(CONTRASTS[:2]):
        close(pairs["interaction"][i], pairs[key][0] - pairs[key][1],
              "Interaction must use the same paired, seed-averaged image contrasts")
    close(interaction["delta"], own["delta"] - donor["delta"], "Interaction arithmetic")
    require(provenance["all_mean_null_invariance_passed"] is True,
            "Identical-caption all-mean null was not verified")
    require(np.array_equal(*pairs["null_repair_minus_fault"])
            and null["delta"] == 0 and null["ci95"] == [0, 0], "Null is not deterministic")

    # This forward all-five-reference protocol is deliberately NOT 'heldout'.
    pooled = control["pooled"]["repair"][PROTOCOL]
    require(pooled["n_images"] == n, "Control image count mismatch")
    for key in ("inv", "base", "delta", "ci95"):
        close(pooled[key], own[key], f"Cross-source forward summary mismatch: {key}")
    seed_pairs = np.asarray([
        control["per_seed_image_scores"][seed]["repair"][PROTOCOL] for seed in seeds
    ], dtype=float)
    require(seed_pairs.shape == (len(seeds), n, 2) and np.isfinite(seed_pairs).all(),
            "Invalid per-seed forward score shape/values")
    close(seed_pairs.mean(axis=0).T, pairs["own_repair_minus_fault"],
          "Cross-source seed-averaged forward image scores differ")
    return rows, n, len(seeds)


def make_figure(rows):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(3.25, 1.8))
        fig.subplots_adjust(left=0.34, right=0.96, bottom=0.26, top=0.78)
        ax.axvline(0, color="#a5a5a5", linewidth=0.7, zorder=0)
        for y, key, color in ((2, CONTRASTS[0], "#176b80"),
                              (1, CONTRASTS[1], "#9a543c")):
            row = rows[key]
            delta = row["delta"]
            low, high = row["ci95"]
            ax.hlines(y, 0, delta, color=color, linewidth=4, alpha=0.18, zorder=1)
            ax.errorbar(delta, y, xerr=[[delta - low], [high - delta]],
                        fmt="o", color=color, markersize=4.5,
                        elinewidth=1.4, capsize=3, capthick=1, zorder=3)
            ax.annotate(f"{delta:+.3f}", (delta, y), xytext=(0, 7),
                         textcoords="offset points", ha="center", va="bottom", color=color)
        # A hollow marker denotes exact caption invariance, not a sampling CI.
        ax.plot(rows[CONTRASTS[2]]["delta"], 0, "o", markersize=5,
                markerfacecolor="none", markeredgecolor="#5d5d5d", markeredgewidth=1.1)
        ax.annotate("0", (0, 0), xytext=(8, 0),
                     textcoords="offset points", va="center", color="#555555")
        ax.set_yticks([2, 1, 0], ["Correct image", "Different image", "No image signal"])
        ax.set_ylim(-0.45, 2.7)
        ax.set_xlim(-0.025, 0.085)
        ax.set_xticks([-0.02, 0, 0.04, 0.08], ["-0.02", "0", "+0.04", "+0.08"])
        ax.tick_params(axis="y", length=0, pad=5)
        ax.tick_params(axis="x", length=2.5, width=0.6, pad=3)
        for side in ("top", "left", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#999999")
        ax.spines["bottom"].set_linewidth(0.6)
        ax.set_xlabel("ROUGE-1 change", fontsize=8.5, labelpad=4)
        fig.text(0.03, 0.96, "Recovery needs the correct image", fontsize=9,
                 weight="bold", va="top")
    return fig


def main():
    raw = [path.read_bytes() for path in SOURCES]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(data).hexdigest()
              for path, data in zip(SOURCES, raw)}
    rows, n_images, n_models = validate_sources(*(json.loads(data) for data in raw))
    fig = make_figure(rows)
    receipt = json.dumps({"source_sha256": hashes, "protocol": PROTOCOL,
                          "n_images": n_images, "n_models": n_models,
                          "uncertainty": "paired recipient images; three fixed models averaged first",
                          "repair": "known preprocessing mask; not general image-only correction"},
                         sort_keys=True)
    # No tight bounding-box crop: preserve the exact single-column physical size.
    with plt.rc_context(STYLE):
        fig.savefig(OUTPUT.with_suffix(".pdf"), transparent=True,
                    metadata={"Title": "Constructed visual recovery control", "Subject": receipt,
                              "CreationDate": None, "ModDate": None})
        fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, transparent=True,
                    metadata={"Description": receipt})
    plt.close(fig)
    for path, data in zip(SOURCES, raw):
        if path.read_bytes() != data:
            raise RuntimeError(f"Source changed during figure build: {path}")
    print(receipt)
    for key, row in rows.items():
        print(f"{key}: {row['delta']:+.5f}; stored CI {row['ci95']}")
    print(f"Wrote {OUTPUT}.pdf and .png (3.25 x 1.8 in; 300 dpi PNG)")


if __name__ == "__main__":
    main()
