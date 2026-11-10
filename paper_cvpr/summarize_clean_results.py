"""Verify and summarize the completed stable-KL runs, on CPU, without inference.

From the project root (use the environment recorded by the pipeline):
  CUDA_VISIBLE_DEVICES='' ../.venv/bin/python3 -B paper_cvpr/summarize_clean_results.py
Only the requested summary is written, exclusively; --check compares an existing
summary byte-for-byte. Original receipts, models, captions and papers are read-only.
"""

import argparse
import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from pycocoevalcap.cider.cider import Cider
import torch

import run_clean_pipeline as pipeline
from paper_cvpr.rescore_rouge import image_scores, norm, paired, score
from run_clean_semantics import validate_inputs

SEEDS = (42, 7, 13)
VARIANTS = ("free", "onehot", "hidden")
PROTOCOLS = ("single_ref", "heldout", "mismatch_target", "mismatch_grounding")
DECOMP = ("best5_clipped", "best5_unclipped", "each_inversion_all_refs_avg", "heldout")
FRACTIONS = ("0", "0.25", "0.5", "1")
REPORT_SOURCES = ("paper_cvpr/main.tex", "paper_cvpr/supp.tex", "paper_cvpr/references.bib",
                  "paper_cvpr/RESPONSE_TO_REVIEWS.md", "paper_cvpr/summarize_clean_results.py",
                  "paper_cvpr/test_summarize_clean_results.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_json(value):
    if isinstance(value, float):
        require(math.isfinite(value), "Nonfinite JSON value")
    elif isinstance(value, dict):
        for item in value.values():
            finite_json(item)
    elif isinstance(value, list):
        for item in value:
            finite_json(item)


def verify_provenance_refresh(previous, current):
    """Refresh report snapshots only after exact result and experiment-hash checks."""
    require({k: v for k, v in previous.items() if k != "source_sha256"}
            == {k: v for k, v in current.items() if k != "source_sha256"},
            "Scientific results changed; refusing provenance-only refresh")
    require({k: v for k, v in previous["source_sha256"].items() if k not in REPORT_SOURCES}
            == {k: v for k, v in current["source_sha256"].items() if k not in REPORT_SOURCES},
            "Experiment source hashes changed; refusing provenance-only refresh")


def compact(row):
    return {key: value for key, value in row.items() if key != "per_image"}


def aggregate(rows):
    """Pair images across models first; never treat 3,000 model/images as IID."""
    require(len(rows) == 3, "Exactly three trained seeds required")
    a = np.array([r["per_image"]["inv"] for r in rows], dtype=float)
    b = np.array([r["per_image"]["base"] for r in rows], dtype=float)
    require(a.ndim == 2 and a.shape == b.shape and a.shape[0] == 3,
            "Misaligned per-seed image arrays")
    result = compact(paired(a.mean(0), b.mean(0)))
    result["per_seed"] = {str(seed): compact(row) for seed, row in zip(SEEDS, rows)}
    result["seed_minmax"] = {key: [min(r[key] for r in rows), max(r[key] for r in rows)]
                             for key in ("inv", "base", "delta")}
    return result


def compare_saved(actual, saved, tolerance=1e-12):
    for key in ("inv", "base", "delta", "ci95"):
        require(np.allclose(actual[key], saved[key], atol=tolerance, rtol=0),
                f"Saved metric disagreement: {key}")
    if "per_image" in saved:
        for side in ("inv", "base"):
            require(np.allclose(actual["per_image"][side], saved["per_image"][side],
                                atol=tolerance, rtol=0), "Saved image scores disagree")


def cider(predictions, references):
    # Same corpus-dependent document frequencies and empty-caption convention
    # as run_rev_eval_metrics, without importing its legacy model dependencies.
    gts = {i: [norm(r) for r in refs] for i, refs in enumerate(references)}
    res = {i: [norm(p) or "empty"] for i, p in enumerate(predictions)}
    return np.asarray(Cider().compute_score(gts, res)[1], dtype=float)


def cider_protocol(records, variant, protocol, slots):
    predictions, baselines, references = [], [], []
    for record in records:
        refs = record["refs"]
        for k in slots if protocol in ("single_ref", "heldout", "each_inversion_all_refs_avg") else [-1]:
            predictions.append(record["variants"][variant]["true"][k] if k >= 0
                               else record["variants"][variant]["mismatch"])
            baselines.append(record["baseline"])
            references.append([refs[k]] if protocol == "single_ref" else
                              [r for j, r in enumerate(refs) if j != k] if protocol == "heldout" else
                              [record["mismatch_target"]] if protocol == "mismatch_target" else refs)
    a, b = cider(predictions, references), cider(baselines, references)
    return paired(a.reshape(len(records), -1).mean(1), b.reshape(len(records), -1).mean(1))


def validate_dump(data, original, slots, variants, n=1000):
    require(data.get("partial") is False and data["n_images"] == n,
            "Incomplete protocol dump")
    require(data["evaluated_true_slots"] == list(slots) and data["variants"] == list(variants),
            "Unexpected slots or variants")
    records = data["records"]
    require([r["image"] for r in records] == list(range(n)), "Image alignment mismatch")
    for record, source in zip(records, original):
        for key in ("image", "feature_index", "refs", "mismatch_source", "mismatch_target"):
            require(record[key] == source[key], f"Protocol mismatch: {key}")
        require(set(record["variants"]) == set(variants), "Missing variant")
        for variant in variants:
            row = record["variants"][variant]
            require(len(row["true"]) == 5 and isinstance(row["mismatch"], str), "Missing jobs")
            require([k for k, cap in enumerate(row["true"]) if isinstance(cap, str)] == list(slots),
                    "Incomplete true jobs")
            require(all(cap is None for k, cap in enumerate(row["true"]) if k not in slots),
                    "Unexpected true jobs")
    finite_json(data)


def mask_audit(records, encodings, plans, fraction):
    """Exact loss-bearing teacher-forced rows, not all 40 allocated plan rows."""
    rng = torch.Generator(device="cpu").manual_seed(20261030)
    order = torch.stack([torch.randperm(40, generator=rng) for _ in records])
    replacement = torch.randint(256, (len(records), 40), generator=rng)
    masks = torch.zeros(len(records), 40, dtype=torch.bool)
    masks.scatter_(1, order[:, :round(40 * float(fraction))], True)
    changed = masks & (replacement != plans.argmax(-1))
    for i, record in enumerate(records):
        require(record["corruption_rows"] == masks[i].nonzero().flatten().tolist(), "Corruption mask mismatch")
        require(record["replacement_patch_indices"] == replacement[i].tolist(), "Replacement RNG mismatch")
    result = {}
    for kind in ("true", "mismatch"):
        jobs = [j for j in encodings if j["kind"] == kind and (kind != "true" or j["k"] in (0, 1))]
        valid = torch.tensor([j["target_tokens"][1:] for j in jobs]) != 0
        ids = [j["image"] for j in jobs]
        used_mask = masks[ids, :39] & valid
        used_changed = changed[ids, :39] & valid
        result[kind] = {
            "n_jobs": len(jobs), "loss_bearing_rows": int(valid.sum()),
            "selected_loss_bearing_rows": int(used_mask.sum()),
            "changed_argmax_loss_bearing_rows": int(used_changed.sum()),
            "pooled_selected_fraction": float(used_mask.sum() / valid.sum()),
            "pooled_changed_argmax_fraction": float(used_changed.sum() / valid.sum()),
            "mean_loss_bearing_rows_per_job": float(valid.sum(1).float().mean()),
            "fraction_jobs_with_selected_loss_rows": float(used_mask.any(1).float().mean()),
        }
    result["nominal_selected_rows_per_image"] = round(40 * float(fraction))
    result["generation_rows"] = "Exact generated token IDs were not saved; decoded strings discard PAD/BOS/EOS. No exact generated-row exposure claim."
    result["scope"] = "Exact nonpadding target[:,1:] loss mask; decoder slices plan[:,:L]; row39 is never consumed. Earlier causal self-attention can mediate effects."
    return result


def print_tables(result):
    comparisons = result["comparisons"]
    groups = {
        "Primary ROUGE-1 decomposition": [f"primary/rouge1/{v}/{p}" for v in VARIANTS for p in DECOMP],
        "Primary protocols": [f"primary/{m}/{v}/{p}" for m in ("rouge1", "cider", "bertscore")
                              for v in VARIANTS for p in PROTOCOLS],
        "Forward controls": [k for k in comparisons if k.startswith("forward/")],
        "Strict k01": [k for k in comparisons if k.startswith("strict_k01/")],
        "Corruption": [f"corruption/{f}/rouge1/{p}" for f in FRACTIONS for p in
                       ("free/heldout", "repair/heldout", "free_minus_repair_heldout",
                        "free/mismatch_target", "free/mismatch_grounding")],
    }
    for title, keys in groups.items():
        print("\n" + title)
        print("| Comparison | Oracle/base | Mean delta [95% CI] | Seed42 delta [CI] | Seed7 delta [CI] | Seed13 delta [CI] | Seed min/max |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for key in keys:
            row = comparisons[key]
            cells = [key, f"{row['inv']:.6f}/{row['base']:.6f}"]
            for r in [row, *[row["per_seed"][str(s)] for s in SEEDS]]:
                cells.append(f"{r['delta']:+.6f} [{r['ci95'][0]:+.6f}, {r['ci95'][1]:+.6f}]")
            cells.append("/".join(f"{v:+.6f}" for v in row["seed_minmax"]["delta"]))
            print("| " + " | ".join(cells) + " |")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=ROOT / "results/clean_small_stablekl")
    parser.add_argument("--eval-root", type=Path, default=ROOT / "results/clean_eval_stablekl")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("clean_results_summary.json"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Recompute and compare; do not write")
    mode.add_argument("--refresh-provenance", action="store_true",
                      help="Update report hashes only if recomputed results and experiment hashes are identical")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    pipeline.TRAIN_ROOT = pipeline.checked_path(args.train_root)
    pipeline.EVAL_ROOT = pipeline.checked_path(args.eval_root)
    output = pipeline.checked_path(args.output)
    if not args.check and not args.refresh_provenance:
        require(not output.exists(), "Refusing to overwrite existing summary; use --check")
    inventory = {}

    def track(path):
        path = pipeline.checked_path(path)
        relative = path.relative_to(ROOT).as_posix()
        digest = pipeline.sha256(path)
        require(relative not in inventory or inventory[relative] == digest, "Input changed during summary")
        inventory[relative] = digest
        return path

    def load(path):
        value = pipeline.read_json(track(path))
        finite_json(value)
        return value

    statuses = list(pipeline.TRAIN_ROOT.glob("pipeline_*/status.json"))
    require(len(statuses) == 1, "Expected one authoritative queue status")
    queue = load(statuses[0])
    require(queue["status"] == "complete", "Queue not complete")
    require(queue["semantic_scoring"]["completed_seeds"] == list(SEEDS), "Semantic seeds incomplete")
    stages = pipeline.build_stages(python=queue["stages"][0]["argv"][0])
    require(stages == queue["stages"], "Configured build_stages differs from recorded queue")
    require([s["name"] for s in stages] == [s["name"] for s in queue["completed_stages"]], "Queue stages incomplete")
    actions = {}
    for stage in stages:
        print("Verifying", stage["name"], flush=True)
        actions[stage["name"]] = pipeline.stage_action(stage)
        require(actions[stage["name"]] == "skip_hash_verified", "Unverified pipeline stage")
        require(pipeline.receipt_signature(stage) == queue["stage_signatures"][stage["name"]], "Queue code signature stale")
        folder = Path(stage["folder"])
        if stage["kind"] == "train":
            manifest = load(folder / "manifest.json")
            for name in manifest["hashes"]:
                track(ROOT / name)
            for name in manifest["artifact_hashes"]:
                track(folder / name)
        else:
            receipt = load(folder / f"pipeline_{stage['kind']}_receipt.json")
            for name in receipt["hashes"]:
                track(folder / name)

    for name in REPORT_SOURCES:
        track(ROOT / name)

    rows, audits, matched = {}, {}, {}

    def add(key, seed, row):
        require(seed not in rows.setdefault(key, {}), "Duplicate result")
        rows[key][seed] = row

    reference_identity = None
    for seed in SEEDS:
        print("Summarizing seed", seed, flush=True)
        train = pipeline.TRAIN_ROOT / f"seed{seed}"
        folder = pipeline.EVAL_ROOT / f"seed{seed}"
        manifest = load(train / "manifest.json")
        require(manifest["config"]["numerics"]["salience_kl"] == "stablekl-v1", "Wrong numerics")
        require(manifest["config"]["precision"] == "float32; TF32 disabled", "Wrong precision")
        require(manifest["split_ids"] == {"train": list(range(13000)), "val": list(range(13000, 14000)),
                                           "test": list(range(14000, 15000))}, "Split mismatch")
        history_audit = {}
        for name, budget in (("free", 40), ("denoiser", 60), ("guided", 50)):
            history = manifest["stages"][name]["history"]
            require([r["epoch"] for r in history] == list(range(1, budget + 1)), "Incomplete training budget")
            for row in history:
                require(row["train_batches"] == 2032 and row["val_batches"] == 157, "Training cohort changed")
                require(row["rejected_nonfinite"] == 0 and row["finite_loss_checks"] == 2189
                        and row["finite_gradient_checks"] == 2032 and row["finite_parameter_checks"] == 2032,
                        "Incomplete finite training checks")
            best = min(history, key=lambda r: r["val_loss"])
            require(manifest["stages"][name]["best"] == {"epoch": best["epoch"], "val_loss": best["val_loss"]},
                    "Checkpoint not minimum validation loss")
            history_audit[name] = {"epochs": budget, "best": manifest["stages"][name]["best"],
                                   "finite_loss_checks": sum(r["finite_loss_checks"] for r in history),
                                   "finite_gradient_checks": sum(r["finite_gradient_checks"] for r in history),
                                   "finite_parameter_checks": sum(r["finite_parameter_checks"] for r in history),
                                   "rejected_nonfinite": 0}
        parameter_counts = {}
        for name in ("proj", "free", "denoiser", "guided"):
            state = torch.load(train / f"model_{name}.pt", map_location="cpu", weights_only=True)
            require(all(torch.isfinite(t).all() for t in state.values()), "Nonfinite exported model")
            require(all(t.dtype == torch.float32 for t in state.values() if t.is_floating_point()), "Non-FP32 model")
            parameter_counts[name] = sum(t.numel() for t in state.values())
        plans = torch.load(train / "plans_test.pt", map_location="cpu", weights_only=True)
        require(plans.shape == (1000, 40, 256) and torch.isfinite(plans).all() and (plans >= 0).all()
                and torch.allclose(plans.sum(-1), torch.ones(1000, 40), atol=2e-6, rtol=1e-5), "Invalid test plans")
        cpu = load(folder / "cpu_scores.json")
        semantic_artifact = load(folder / "semantic_scores.json")
        for name in semantic_artifact["metadata"]["code_sha256"]:
            track(ROOT / name)
        semantic = semantic_artifact["datasets"]
        forward = load(folder / "forward.json")
        primary = load(folder / "inversion_protocol_n1000_s0.json")
        records = validate_inputs(primary, forward)
        identity = [{k: r[k] for k in ("image", "refs", "mismatch_source", "mismatch_target")} for r in records]
        if reference_identity is None:
            reference_identity = identity
        require(identity == reference_identity, "Seeds do not share aligned images/references/donors")
        planned = load(folder / "planned_jobs.json")
        for name in planned["metadata"]["input_sha256"]:
            track(ROOT / name)
        encoding = load(folder / "target_encoding.json")
        vocab = load(train / "tok_vocab.json")
        w2i = {w: i for i, w in enumerate(vocab)}
        require(len(encoding["protocol"]) == 6000 and len(encoding["counterfactual"]) == 901,
                "Incomplete target encoding")
        for jobs, expected in ((encoding["protocol"], planned["protocol_jobs"]),
                               (encoding["counterfactual"], planned["swap_jobs"])):
            for job, planned_job in zip(jobs, expected):
                require(all(job[k] == v for k, v in planned_job.items()), "Target job mismatch")
                tokens = ([1] + [w2i.get(w.lower(), 3) for w in job["target"].split()] + [2])[:40]
                tokens += [0] * (40 - len(tokens))
                require(tokens == job["target_tokens"], "Saved target token/mask mismatch")
        require(forward["baseline_logits_finite"] is True, "Uncertified forward logits")
        validate_dump(primary, records, range(5), VARIANTS)
        # Rescore R1 from captions and validate every primary saved image vector.
        for variant in VARIANTS:
            r1_values = [image_scores(r, variant)[0] for r in records]
            for protocol in (*dict.fromkeys((*DECOMP, *PROTOCOLS)), "human_heldout"):
                row = paired([v[protocol][0] for v in r1_values], [v[protocol][1] for v in r1_values])
                saved_variant = cpu["protocols"]["inversion_protocol_n1000_s0.json"]["standard_rouge"]["variants"][variant]
                saved = (saved_variant["human_heldout"] if protocol == "human_heldout" else
                         saved_variant["protocol"].get(protocol, saved_variant["decomposition"].get(protocol)))
                compare_saved(row, saved)
                add(f"primary/rouge1/{variant}/{protocol}", seed, row)
            for protocol in (*PROTOCOLS, "each_inversion_all_refs_avg"):
                row = cider_protocol(records, variant, protocol, range(5))
                if protocol in PROTOCOLS:
                    compare_saved(row, cpu["protocols"]["inversion_protocol_n1000_s0.json"]["metrics"][variant][protocol]["cider"], 5.00001e-5)
                add(f"primary/cider/{variant}/{protocol}", seed, row)
            for protocol, row in semantic["protocol"]["variants"][variant].items():
                add(f"primary/bertscore/{variant}/{protocol}", seed, row)
        # Restrict targets, not references: preserve all five reference slots.
        strict = copy.deepcopy(records)
        for record in strict:
            for variant in VARIANTS:
                record["variants"][variant]["true"][2:] = [None] * 3
        for variant in VARIANTS:
            values = [image_scores(r, variant)[0] for r in strict]
            for protocol in ("single_ref", "heldout", "each_inversion_all_refs_avg"):
                add(f"strict_k01/rouge1/{variant}/{protocol}", seed,
                    paired([v[protocol][0] for v in values], [v[protocol][1] for v in values]))
                add(f"strict_k01/cider/{variant}/{protocol}", seed,
                    cider_protocol(records, variant, protocol, (0, 1)))
        refs = forward["refs"]
        for metric in ("rouge1", "cider", "bertscore"):
            vectors = {}
            for system, captions in forward["caps"].items():
                vectors[system] = ([max(score(c, r) for r in rs) for c, rs in zip(captions, refs)] if metric == "rouge1" else
                                   cider(captions, refs) if metric == "cider" else
                                   semantic["forward"]["systems"][system]["per_image"])
                if metric == "rouge1":
                    require(np.allclose(vectors[system], forward["per_image_rouge1"][system], atol=1e-12, rtol=0), "Forward R1 mismatch")
                add(f"forward/{metric}/{system}", seed, paired(vectors[system], np.zeros(1000)))
            for system in ("free", "guided_uniform"):
                row = paired(vectors["guided_denoised"], vectors[system])
                if metric != "cider":
                    saved = forward if metric == "rouge1" else semantic["forward"]
                    compare_saved(row, saved["paired"][f"denoised_minus_{system}"])
                add(f"forward/{metric}/denoised_minus_{system}", seed, row)
        audit = {"training": history_audit, "export_state_elements_all_finite_fp32": parameter_counts,
                 "test_plans_finite_normalized": True, "checks": {k: v for k, v in manifest["checks"].items() if isinstance(v, bool)},
                 "unknown_rates": manifest["unknown_rates"],
                 "forward_export_mismatches": forward["training_export_caption_mismatches"], "corruption_masks": {}}
        for fraction in FRACTIONS:
            name = f"corruption_{fraction}.json"
            data = load(folder / name)
            validate_dump(data, records, (0, 1), ("free", "repair"))
            require(data["metadata"]["corruption_fraction"] == float(fraction), "Wrong corruption fraction")
            for record in data["records"]:
                clean = records[record["image"]]["baseline"]
                require(record["clean_baseline"] == clean and
                        record["variants"]["repair"]["true"][:2] == [clean, clean]
                        and record["variants"]["repair"]["mismatch"] == clean, "Repair used a different baseline")
                if fraction == "0":
                    require(record["baseline"] == clean, "Zero corruption changed baseline")
            audit["corruption_masks"][fraction] = mask_audit(data["records"], encoding["protocol"], plans, fraction)
            if fraction == "0":
                audit["zero_corruption_vs_primary_k01_caption_mismatches"] = sum(
                    record["variants"]["free"]["true"][k] != records[record["image"]]["variants"]["free"]["true"][k]
                    for record in data["records"] for k in (0, 1))
            for variant in ("free", "repair"):
                values = [image_scores(r, variant)[0] for r in data["records"]]
                for protocol in PROTOCOLS:
                    row = paired([v[protocol][0] for v in values], [v[protocol][1] for v in values])
                    compare_saved(row, cpu["protocols"][name]["standard_rouge"]["variants"][variant]["protocol"][protocol])
                    add(f"corruption/{fraction}/rouge1/{variant}/{protocol}", seed, row)
                    c = cider_protocol(data["records"], variant, protocol, (0, 1))
                    compare_saved(c, cpu["protocols"][name]["metrics"][variant][protocol]["cider"], 5.00001e-5)
                    add(f"corruption/{fraction}/cider/{variant}/{protocol}", seed, c)
            for metric in ("rouge1", "cider"):
                free = rows[f"corruption/{fraction}/{metric}/free/heldout"][seed]
                repair = rows[f"corruption/{fraction}/{metric}/repair/heldout"][seed]
                require(free["per_image"]["base"] == repair["per_image"]["base"], "Nonidentical heldout baselines")
                add(f"corruption/{fraction}/{metric}/free_minus_repair_heldout", seed,
                    paired(free["per_image"]["inv"], repair["per_image"]["inv"]))
                zero = rows[f"corruption/0/{metric}/free/heldout"][seed]
                require(np.allclose(zero["per_image"]["base"], repair["per_image"]["inv"], atol=1e-12, rtol=0),
                        "Reference-free repair did not recover clean heldout baseline")
                add(f"corruption/{fraction}/{metric}/baseline_loss_heldout", seed,
                    paired(zero["per_image"]["base"], free["per_image"]["base"]))
                add(f"corruption/{fraction}/{metric}/free_gain_change_from_zero", seed,
                    paired(np.subtract(free["per_image"]["inv"], free["per_image"]["base"]),
                           np.subtract(zero["per_image"]["inv"], zero["per_image"]["base"])))
            add(f"corruption/{fraction}/rouge1/repair_allrefs_secondary", seed,
                data["cohort_summary"]["repair_gain"])
        bounded = load(folder / "bounded_unbounded_n200.json")
        validate_dump(bounded, records, (0, 1), ("bounded", "unbounded"), 200)
        swaps = load(folder / "swap_injection_n1000_s0.json")
        require(swaps["n"] == 901 and swaps["jobs_dropped"] == 0
                and len({r["image"] for r in swaps["records"]}) == 901, "Incomplete swaps")
        for record, job in zip(swaps["records"], planned["swap_jobs"]):
            require(all(record[k] == v for k, v in job.items()) and
                    all(isinstance(record[v], str) for v in VARIANTS), "Swap jobs mismatch")
        lock = load(folder / "matched_calibration_lock.json")
        match = load(folder / "matched_gain.json")
        validation = load(folder / "matched_validation.json")
        require(lock["calibration_split"] == "validation" and lock["test_selection"] is False
                and lock["n_calibration"] == validation["n"] == 150 and match["n_evaluation"] == 901
                and match["calibration"] == lock and len(match["jobs"]) == 901
                and len(match["free300_captions"]) == 901
                and all(len(c) == 901 for c in match["hidden_captions_by_step"].values())
                and all(13000 <= i < 14000 for i in lock["calibration_feature_indices"]), "Incomplete matched artifacts")
        require(len(validation["jobs"]) == len(validation["free300_captions"]) == 150
                and set(validation["hidden_captions_by_step"]) == {str(s) for s in range(21)}
                and all(len(c) == 150 for c in validation["hidden_captions_by_step"].values()),
                "Incomplete matched calibration grid")
        matched[str(seed)] = {"status": "hash_verified_complete; scientific matching audit delegated, not duplicated",
                              "n_calibration": 150, "n_evaluation": 901,
                              "selected_steps": lock["selected_steps"],
                              "step0_baseline_mismatches": match["step0_baseline_mismatches"]}
        audits[str(seed)] = audit

    print("Bootstrapping seed-averaged paired image scores", flush=True)
    comparisons = {key: aggregate([by_seed[s] for s in SEEDS]) for key, by_seed in rows.items()}
    attenuation = {}
    for label in ("all_seed_mean", *map(str, SEEDS)):
        values = {p: comparisons[f"primary/rouge1/free/{p}"] for p in ("best5_clipped", "single_ref", "heldout")}
        deltas = {p: (r if label == "all_seed_mean" else r["per_seed"][label])["delta"] for p, r in values.items()}
        attenuation[label] = {"best5_to_heldout_ratio": deltas["best5_clipped"] / deltas["heldout"],
                              "single_to_heldout_ratio": deltas["single_ref"] / deltas["heldout"],
                              "heldout_fraction_of_best5": deltas["heldout"] / deltas["best5_clipped"]}
    result = {
        "schema_version": 1, "status": "complete_hash_verified", "device": "cpu",
        "training_or_model_inference_performed": False, "seeds": list(SEEDS),
        "train_root": pipeline.TRAIN_ROOT.relative_to(ROOT).as_posix(),
        "eval_root": pipeline.EVAL_ROOT.relative_to(ROOT).as_posix(),
        "verification": {"stage_actions": actions, "per_seed": audits, "matched_status_only": matched,
                         "all_primary_saved_r1_vectors_recomputed": True,
                         "saved_cider_protocols_recomputed_with_rounding_tolerance": 5.00001e-5,
                         "semantic_source_code_snapshot_package_hashes": "Verified by pipeline.stage_action/verify_result; no semantic inference",
                         "sources_unchanged_after_summary": True},
        "statistical_policy": {
            "primary": "Equal-weight mean of all three fixed trained seeds; seed42 was fixed first, not selected by results (user provenance).",
            "ci": "Paired percentile image bootstrap, 10000 resamples, numpy.default_rng(0), 95%; average aligned scores across seeds BEFORE resampling images.",
            "scope": "Conditional on these three trained models and saved scoring corpora; NOT training-seed uncertainty, not 3000 independent images, no multiplicity correction.",
            "seed_minmax": "Descriptive range across three seeds, not a confidence interval.",
            "image_ids": list(range(1000)),
        },
        "definitions": {
            "primary": "All five target slots; duplicate reference strings remain distinct slots.",
            "best5_unclipped": "max_k R1(y_k,r_k) vs max_j R1(b,r_j), not max over all25 pairs",
            "best5_clipped": "max(best5_unclipped oracle, baseline) per image; positive floor only here",
            "each_inversion_all_refs_avg": "Mean over target jobs; R1/BERTScore max over reference slots; CIDEr uses joint multi-reference scoring, not max",
            "heldout": "Mean_k score(y_k, refs excluding slot k), baseline excludes the identical slot; R1/BERTScore max over remaining slots",
            "single_ref": "Mean_k score(y_k,r_k), same baseline reference",
            "strict_k01": "Only targets k=0,1; five reference slots retained, heldout excludes exactly the targeted slot; NOT best5",
            "corruption": "Free heldout gain and reference-free repair heldout gain share k=0,1 and the corrupted baseline. free_minus_repair_heldout subtracts those paired gains.",
            "baseline_loss_heldout": "Clean baseline minus corrupted baseline on identical heldout slots; equals repair gain exactly",
            "repair_allrefs_secondary": "Archived cohort_summary repair_gain uses all five refs; deliberately NOT compared against free heldout gain",
            "forward": "One caption per image, all five refs. System rows use base=0 only as an absolute-score reporting device; deltas in denoised_minus_* rows are actual controls.",
            "cider": "pycocoevalcap Cider (CIDEr-D implementation), raw scale, lowercase/punctuation normalization, empty->empty; corpus DF refit per protocol, fixed during image bootstrap",
            "bertscore": "Saved canonical roberta-large layer17 raw F1, idf=False, no baseline rescaling; no new inference",
            "not_available": "No saved per-pair BERTScores for k01/corruption or best5; these are not fabricated. Best5 decomposition is ROUGE-1 only; CIDEr corresponding single/heldout/shuffled/allrefs protocols are supplied.",
            "training_masks": "Denoiser CE/salience use sampled mask.bool() over 40 rows (not a nonpadding mask); guided KL/salience use ids[:,1:]!=0; inversion uses target[:,1:]!=0. Corruption selects among all40 rows, but loss-bearing exposure is audited separately.",
        },
        "paper_replacement_requirements": [
            "Replace old small-model provenance (14k later-stage training and test-text vocabulary exposure) with strict13k/1k/1k, train-only vocabulary, validation-selected checkpoints; do not rewrite historical artifacts.",
            "Replace main abstract/decomposition/protocol/semantic-transfer numbers and reduction ratios from primary all-seed results; retain seed-specific CIs and descriptive ranges.",
            "Use standard clean forward metrics instead of substituting legacy small-model or full-COCO numbers; the clean replication does not rerun full-COCO or external VLMs.",
            "Replace queued/not-complete status only after this hash audit; stablekl numerical repair is not itself scientific evidence.",
            "Report signed corruption gains, matching heldout repair, actual loss-mask exposure, and any failure of monotonicity; do not guarantee a positive sensitivity result.",
            "Preserve rejected NaN-checkpoint finding, symbolic-only calibration limits, metric-dependent semantic negatives, uniform-native-QK distinction, effect-matching caveats, unrun reciprocal BLIP2 control, and blocked external numerical reproduction.",
            "Matched effects are separately audited; do not promote dimension matching or approximate global matching to causal equivalence.",
        ],
        "comparisons": comparisons,
        "free_rouge_attenuation": attenuation,
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "rouge-score", "pycocoevalcap", "bert-score")},
        "source_sha256": inventory,
    }
    # Detect changes during scoring, including paper files that this script never writes.
    for name, expected in inventory.items():
        require(pipeline.sha256(ROOT / name) == expected, f"Source changed during summary: {name}")
    require(not torch.cuda.is_initialized(), "Unexpected CUDA initialization")
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    require("/home/" not in text and str(Path.home()) not in text, "Private absolute path in summary")
    if args.check:
        require(output.read_text() == text, "Summary is not byte-for-byte reproducible")
        print("Reproducibility check passed:", output.relative_to(ROOT))
    elif args.refresh_provenance:
        verify_provenance_refresh(pipeline.read_json(output), result)
        pipeline.write_json(output, result, replace=True)
        print("Refreshed report provenance; scientific results unchanged:", output.relative_to(ROOT))
    else:
        pipeline.write_json(output, result)
        print("Wrote", output.relative_to(ROOT))
    print_tables(result)
    return result


if __name__ == "__main__":
    main()
