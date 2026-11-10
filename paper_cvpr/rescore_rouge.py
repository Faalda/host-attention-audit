"""CPU-only, offline rescoring of saved captions; never imports model/training code.

Run with the project virtualenv: python -B paper_cvpr/rescore_rouge.py
Only standard_rouge_results.json and standard_rouge_tables.tex beside this file
are written. Source dumps and paper sources are never modified. No downloads.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys
from functools import lru_cache

import numpy as np
from rouge_score import rouge_scorer


ROOT = Path(__file__).resolve().parents[1]
SEED = 0
RESAMPLES = 10000
SCORER = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)


def norm(text):
    if not isinstance(text, str):
        raise ValueError("A caption must be a string (empty strings are allowed)")
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


@lru_cache(maxsize=250000)
def score(prediction, reference):
    return SCORER.score(norm(reference), norm(prediction))["rouge1"].fmeasure


def paired(inv, base, *, resamples=RESAMPLES, seed=SEED, batch=100):
    """Already image-aggregated scores; bounded-memory, paired percentile bootstrap."""
    a, b = np.asarray(inv, dtype=float), np.asarray(base, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not len(a):
        raise ValueError("Expected equally sized, nonempty image-score vectors")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite image scores")
    if resamples < 1 or batch < 1:
        raise ValueError("Bootstrap sizes must be positive")
    rng = np.random.default_rng(seed)
    boot = np.empty((resamples, 3))
    delta = a - b
    # Chunking preserves the RNG draw order of the original full index matrix.
    for start in range(0, resamples, batch):
        end = min(start + batch, resamples)
        indices = rng.integers(0, len(a), (end - start, len(a)))
        boot[start:end, 0] = a[indices].mean(1)
        boot[start:end, 1] = b[indices].mean(1)
        boot[start:end, 2] = delta[indices].mean(1)
    ci = np.percentile(boot, [2.5, 97.5], axis=0)
    return {
        "n_images": len(a),
        "inv": float(a.mean()), "base": float(b.mean()),
        "delta": float(delta.mean()), "ci95": ci[:, 2].tolist(),
        "inv_ci95": ci[:, 0].tolist(), "base_ci95": ci[:, 1].tolist(),
        "frac_images_pos": float((delta > 0).mean()),
        "frac_images_neg": float((delta < 0).mean()),
        "per_image": {"inv": a.tolist(), "base": b.tolist()},
    }


def image_scores(record, variant):
    """Preserve reference slots, including duplicate strings and missing jobs."""
    refs = record["refs"]
    if len(refs) < 2 or not all(isinstance(r, str) for r in refs):
        raise ValueError("Need at least two reference slots")
    data = record["variants"][variant]
    if len(data["true"]) != len(refs):
        raise ValueError("True captions must be indexed by reference slot")
    slots = [k for k, caption in enumerate(data["true"]) if caption is not None]
    if not slots:
        raise ValueError("No true-target jobs for image/variant")
    bs = np.array([score(record["baseline"], r) for r in refs])
    mat = np.array([[score(data["true"][k], r) for r in refs] for k in slots])
    own = np.array([mat[row, k] for row, k in enumerate(slots)])
    held = [np.delete(mat[row], k).max() for row, k in enumerate(slots)]
    held_base = [np.delete(bs, k).max() for k in slots]
    human = [max(score(refs[k], r) for j, r in enumerate(refs) if j != k)
             for k in slots]
    result = {
        "single_ref": (float(own.mean()), float(bs[slots].mean())),
        "heldout": (float(np.mean(held)), float(np.mean(held_base))),
        "human_heldout": (float(np.mean(human)), float(np.mean(held_base))),
        "each_inversion_all_refs_avg": (float(mat.max(1).mean()), float(bs.max())),
    }
    # A partial two-target experiment must never be called "best5".
    if len(refs) == 5 and len(slots) == 5:
        result["best5_unclipped"] = (float(own.max()), float(bs.max()))
        result["best5_clipped"] = (float(max(own.max(), bs.max())), float(bs.max()))
    if data.get("mismatch") is not None:
        pred, target = data["mismatch"], record["mismatch_target"]
        result["mismatch_target"] = (score(pred, target), score(record["baseline"], target))
        result["mismatch_grounding"] = (max(score(pred, r) for r in refs), float(bs.max()))
    return result, slots


def protocol(data, *, require_five=False, include_mismatch=True):
    records = sorted(data["records"], key=lambda r: r["image"])
    ids = [r["image"] for r in records]
    if data.get("partial") or len(ids) != data["n_images"] or len(set(ids)) != len(ids):
        raise ValueError("Incomplete dump or duplicate image IDs")
    result = {"n_images": len(ids), "image_ids": ids, "variants": {}}
    for variant in data["variants"]:
        rows = [image_scores(r, variant) for r in records]
        if require_five and any(len(slots) != 5 or len(r["refs"]) != 5
                                for r, (_, slots) in zip(records, rows)):
            raise ValueError("Core decomposition requires all five target slots")
        keys = set(rows[0][0])
        if any(set(values) != keys for values, _ in rows):
            raise ValueError("Inconsistent evaluated jobs across images")
        summaries = {key: paired([values[key][0] for values, _ in rows],
                                 [values[key][1] for values, _ in rows])
                     for key in sorted(keys)
                     if include_mismatch or not key.startswith("mismatch_")}
        result["variants"][variant] = {
            "target_slots_by_image": [slots for _, slots in rows],
            "protocol": {k: summaries[k] for k in
                         ("single_ref", "heldout", "mismatch_target", "mismatch_grounding")
                         if k in summaries},
            "decomposition": {k: summaries[k] for k in
                              ("best5_clipped", "best5_unclipped", "each_inversion_all_refs_avg", "heldout")
                              if k in summaries},
            "human_heldout": summaries["human_heldout"],
        }
    return result


def full_coco(captions, references):
    refs, ids = references["caps"], references["keys"]
    if len(refs) != 5000 or len(ids) != 5000 or len(set(ids)) != 5000:
        raise ValueError("Expected the complete 5000-image COCO test set")
    if any(len(r) != 5 for r in refs):
        raise ValueError("Expected five COCO reference slots")
    values = {}
    for name in ("free", "guided_denoised", "guided_uniform"):
        if len(captions[name]) != len(refs):
            raise ValueError("Caption/reference length mismatch")
        values[name] = [max(score(p, r) for r in rs) for p, rs in zip(captions[name], refs)]
    systems = {}
    for name, vals in values.items():
        summary = paired(vals, np.zeros(len(vals)))
        systems[name] = {"mean": summary["inv"], "ci95": summary["inv_ci95"], "per_image": vals}
    return {
        "n_images": len(refs), "image_ids": ids, "systems": systems,
        "paired": {
            "denoised_minus_uniform": paired(values["guided_denoised"], values["guided_uniform"]),
            "denoised_minus_free": paired(values["guided_denoised"], values["free"]),
        },
    }


def file_info(path):
    content = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content)}


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def tables(results):
    lines = ["% Generated by rescore_rouge.py. Standard rouge_score ROUGE-1 F1; no stemming.",
             "% Paired image bootstrap, 10000 resamples, seed 0. Requires booktabs.",
             "% All scores in [0,1]. Human rows use the same target slots as each variant."]

    def table(title, rows):
        lines.extend([r"\begin{table*}[t]\centering", r"\caption{" + title + "}",
                      r"\small\begin{tabular}{llrrr}", r"\toprule",
                      r"Variant & Comparison & Score & Baseline & $\Delta$ [95\% CI] \\", r"\midrule"])
        for variant, label, m in rows:
            variant, label = variant.replace("_", r"\_"), label.replace("_", r"\_")
            lines.append(f"{variant} & {label} & {m['inv']:.4f} & {m['base']:.4f} & "
                         f"${m['delta']:+.4f}$ [{m['ci95'][0]:+.4f}, {m['ci95'][1]:+.4f}] " + r"\\")
        lines.extend([r"\bottomrule\end{tabular}\end{table*}", ""])

    for name, dataset in results["datasets"].items():
        if "variants" not in dataset:
            continue
        display = name.replace("_", " ")
        table(f"Standard ROUGE-1 protocol: {display}, $n={dataset['n_images']}$.",
              [(v, p, m) for v, vr in dataset["variants"].items()
               for p, m in list(vr["protocol"].items()) + [("human held-out", vr["human_heldout"])]])
        if name in ("small", "fullcoco_protocol"):
            table(f"Standard ROUGE-1 decomposition: {display}.",
                  [(v, p, m) for v, vr in dataset["variants"].items() for p, m in vr["decomposition"].items()])
    table("Full-COCO, 5000 images: standard ROUGE-1 paired forward comparisons.",
          [("guided", k, v) for k, v in results["datasets"]["fullcoco_uniform"]["paired"].items()])
    return "\n".join(lines) + "\n"


def main():
    sources = {}

    def load(relative):
        path = ROOT / relative
        raw = path.read_bytes()
        sources[relative] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                             "size_bytes": len(raw)}
        return json.loads(raw)

    small = load("results/revision/inversion_protocol_n1000_s0.json")
    full = load("results/full_coco/inversion_protocol_n1000_s0.json")
    captions = load("results/full_coco/captions_test.json")
    refs = load("results/full_coco/caps_test.json")
    datasets = {}
    for name, data in (("small", small), ("fullcoco_protocol", full)):
        print("Scoring", name, flush=True)
        datasets[name] = protocol(data, require_five=True)
    print("Scoring fullcoco_uniform", flush=True)
    datasets["fullcoco_uniform"] = full_coco(captions, refs)
    optional = {
        "blip2": "blip2_protocol_n500_s0.json",
        "blip2_forward": "blip2_forward_n500_r8_s0.json",
        "instructblip": "instructblip_protocol_n300_s0.json",
        "qwen": "qwen_protocol_n150_s0.json",
    }
    skipped = []
    for name, filename in optional.items():
        relative = "results/revision/" + filename
        if not (ROOT / relative).is_file():
            skipped.append(relative)
            continue
        print("Scoring", name, flush=True)
        datasets[name] = protocol(load(relative), include_mismatch=name != "blip2_forward")
    saved = load("results/full_coco/captioning_eval.json")
    cider = {
        "status": "Copied from saved evaluation; NOT recomputed in this ROUGE-only run",
        "source": "results/full_coco/captioning_eval.json",
        "systems": {k: {"mean": v["cider"], "ci95": v["cider_ci95"]}
                    for k, v in saved["systems"].items()},
        "paired": {k: v["cider"] for k, v in saved["paired"].items()},
    }
    checks = {}
    for name, relative in (("small", "results/revision/inversion_protocol_n1000_s0_metrics.json"),
                           ("fullcoco_protocol", "results/full_coco/inversion_protocol_n1000_s0_metrics.json")):
        old = load(relative)
        differences = []
        for v, vr in datasets[name]["variants"].items():
            for p, row in vr["protocol"].items():
                expected = old[v][p]["rouge1"]
                for k in ("inv", "base", "delta"):
                    differences.append(abs(row[k] - expected[k]))
                differences.extend(abs(a - b) for a, b in zip(row["ci95"], expected["ci95"]))
        error = max(differences)
        if error > 0.0000500001:
            raise ValueError(f"Saved standard metrics disagree for {name}: {error}")
        checks[name + "_saved_standard_metrics"] = {"passed": True, "max_abs_error": error,
                                                    "tolerance": 0.0000500001}
    mismatch = [r["image"] for r in full["records"]
                if r["baseline"] != captions["guided_denoised"][r["image"]]]
    if not all(r["refs"] == refs["caps"][r["image"]] for r in full["records"]):
        raise ValueError("Full COCO protocol references are not aligned")
    checks["fullcoco_alignment"] = {
        "references_match": True, "different_baseline_image_ids": mismatch,
        "policy": "Protocol uses its own baseline; 5K comparison uses captions_test.json only",
    }
    # Verify inputs still match the bytes that were actually scored.
    for relative, info in sources.items():
        if file_info(ROOT / relative) != info:
            raise RuntimeError(f"Source changed during scoring: {relative}")
    result = {
        "schema_version": 1,
        "metadata": {
            "metric": "rouge_score rouge1 fmeasure", "use_stemmer": False,
            "normalization": "lowercase; replace [^a-z0-9 ] by space; collapse whitespace",
            "device": "cpu", "model_inference": False, "downloads": False,
            "bootstrap": {"seed": SEED, "resamples": RESAMPLES, "unit": "image",
                          "method": "paired percentile [2.5,97.5]", "rng": "numpy.default_rng (PCG64)",
                          "batch": 100, "seed_reset_per_comparison": True},
            "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "versions": {k: version(k) for k in ("numpy", "rouge-score", "nltk", "absl-py", "six")},
            "implementation": file_info(Path(__file__).resolve()),
            "synthetic_tests": file_info(Path(__file__).with_name("test_rescore_rouge.py")),
            "sources": sources,
            "dataset_sources": {
                "small": ["results/revision/inversion_protocol_n1000_s0.json"],
                "fullcoco_protocol": ["results/full_coco/inversion_protocol_n1000_s0.json"],
                "fullcoco_uniform": ["results/full_coco/captions_test.json", "results/full_coco/caps_test.json"],
                **{k: ["results/revision/" + v] for k, v in optional.items() if k in datasets},
            },
            "definitions": {
                "best5_unclipped": "max_k score(y_k,r_k); NOT max over all 25 pairs",
                "best5_clipped": "max(best5_unclipped, max_j score(baseline,r_j)) per image",
                "each_inversion_all_refs_avg": "mean_k max_j score(y_k,r_j)",
                "heldout": "mean_k max_{j != k} score(y_k,r_j); same excluded slot for baseline",
                "single_ref": "mean_k score(y_k,r_k); baseline mean_k score(baseline,r_k)",
                "human_heldout": "mean over evaluated k of max_{j != k} score(r_k,r_j)",
                "mismatch_target": "score(shuffled_output, mismatch_target); same target for baseline",
                "mismatch_grounding": "max over own-image refs for shuffled output and baseline",
                "fullcoco_uniform": "max over five refs for each single saved caption, then mean over images",
                "per_image": "Arrays follow dataset.image_ids; null target jobs excluded, not zero-filled",
            },
            "notes": [
                "No legacy metric is computed or included in primary tables.",
                "BLIP2 forward dump has dummy mismatch fields; these are deliberately not evaluated.",
                "Uniform external plan cancels its constant log bias; native learned QK routing remains active.",
                "Semantic metrics are not computed. No cache access or model downloads attempted.",
                "Other scalar-only evidence, gate sweeps, interventions and remedies are outside this run.",
            ],
            "optional_sources_missing": skipped,
            "semantic_tool_versions": {k: version(k) for k in
                                       ("bert-score", "sentence-transformers", "open-clip-torch", "pycocoevalcap")},
            "semantic_weights": "Not inspected: cache access was not approved; availability unverified",
        },
        "datasets": datasets, "fullcoco_saved_cider": cider, "verification": checks,
    }
    out = Path(__file__).with_name("standard_rouge_results.json")
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tex = Path(__file__).with_name("standard_rouge_tables.tex")
    tex.write_text(tables(result), encoding="utf-8")
    print("Wrote", out, "and", tex, flush=True)
    print("Small free decomposition:")
    for name, row in datasets["small"]["variants"]["free"]["decomposition"].items():
        print(name, {k: row[k] for k in ("inv", "base", "delta", "ci95")})
    print("Full COCO denoised minus uniform:",
          {k: datasets["fullcoco_uniform"]["paired"]["denoised_minus_uniform"][k]
           for k in ("inv", "base", "delta", "ci95")})


if __name__ == "__main__":
    main()
