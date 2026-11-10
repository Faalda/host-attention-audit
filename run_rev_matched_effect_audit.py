"""Held-out, standard-metric audit of the saved901 counterfactual swap jobs.

No training: only per-job hidden input biases are optimized, with frozen models.
Run with CUDA_VISIBLE_DEVICES=0. All outputs are exclusive-create, never replaced.
"""
import os

# Check before imports that might initialize CUDA; this script must never use GPU1.
if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
    raise RuntimeError("Run with CUDA_VISIBLE_DEVICES=0 ONLY")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer

from run_rev_matched_effect_control import load_models
from run_rev_inversion_protocol import (
    CACHE_PATH, DEVICE, L_CAP, OUT_DIR, SAVE_DIR, TEST_START,
    dec_forward, dec_generate, invert,
)
from run_rev_eval_metrics import objects, norm, lr1

STEPS = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20]
CALIB_N = 150
SEED = 0
BOOT_SEED = 20261030
N_BOOT = 10000
LR = 0.05
BATCH = 500
QUANTILES = [0, .05, .25, .5, .75, .95, 1]
SCORER = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def edit_distance(a, b):
    """Token Levenshtein / max(token lengths, 1); rouge_score tokenization."""
    a, b = SCORER._tokenizer.tokenize(a), SCORER._tokenizer.tokenize(b)
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = nxt
    return row[-1] / max(len(a), len(b), 1)


def caption_metrics(caps, jobs, baselines, refs):
    metrics = {k: [] for k in [
        "target_rouge1", "target_rougeL", "target_legacy_rouge1",
        "gain_rouge1", "gain_rougeL", "gain_legacy_rouge1",
        "normalized_token_edit", "caption_changed", "injection",
        "original_ref_rouge1", "gain_original_ref_rouge1",
        "target_exact_normalized",
    ]}
    for cap, job, base, ref in zip(caps, jobs, baselines, refs):
        fit = SCORER.score(job["target"], cap)
        base_fit = SCORER.score(job["target"], base)
        old_fit, old_base = lr1(cap, [job["target"]]), lr1(base, [job["target"]])
        original = SCORER.score(ref, cap)["rouge1"].fmeasure
        canonical = objects(job["fake"])
        assert len(canonical) == 1, job["fake"]
        values = [
            fit["rouge1"].fmeasure, fit["rougeL"].fmeasure, old_fit,
            fit["rouge1"].fmeasure - base_fit["rouge1"].fmeasure,
            fit["rougeL"].fmeasure - base_fit["rougeL"].fmeasure,
            old_fit - old_base, edit_distance(cap, base), float(cap != base),
            float(bool(canonical & objects(cap))), original,
            original - SCORER.score(ref, base)["rouge1"].fmeasure,
            float(norm(cap) == norm(job["target"])),
        ]
        for key, value in zip(metrics, values):
            metrics[key].append(value)
    return metrics


def distribution(values, boot_means):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()), "mean_ci95": np.quantile(boot_means, [.025, .975]).tolist(),
        "std": float(values.std()),
        "quantiles": {str(q): float(v) for q, v in zip(QUANTILES, np.quantile(values, QUANTILES))},
        "n_positive": int((values > 0).sum()), "n_zero": int((values == 0).sum()),
        "n_negative": int((values < 0).sum()),
    }


def main():
    started = time.time()
    root = Path(__file__).resolve().parent
    output = Path(OUT_DIR) / "matched_effect_audit.json"
    checkpoints = Path(OUT_DIR) / "matched_effect_audit_checkpoints"
    if output.exists() or checkpoints.exists():
        raise FileExistsError("Audit output/checkpoints already exist; refusing to overwrite")
    gpu_status = subprocess.check_output([
        "nvidia-smi", "--id=0", "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
        "--format=csv,noheader",
    ], text=True).strip()
    print("GPU0 before audit:", gpu_status, flush=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA device is required")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    swap_path = Path(OUT_DIR) / "swap_injection_n1000_s0.json"
    baseline_path = Path(OUT_DIR) / "inversion_protocol_n1000_s0.json"
    with swap_path.open() as f:
        swap = json.load(f)
    with baseline_path.open() as f:
        previous = json.load(f)
    jobs = swap["records"]
    assert len(jobs) == 901 and len({j["image"] for j in jobs}) == 901
    previous_by_image = {r["image"]: r for r in previous["records"]}
    baselines = [previous_by_image[j["image"]]["baseline"] for j in jobs]
    refs = [previous_by_image[j["image"]]["refs"][j["k"]] for j in jobs]
    assert all(not (objects(j["fake"]) & set().union(*[
        objects(r) for r in previous_by_image[j["image"]]["refs"]
    ])) for j in jobs)
    config = {
        "cohort": "saved901, original saved order, one job per unique image",
        "steps": STEPS, "calibration_positions": [0, CALIB_N],
        "evaluation_positions": [CALIB_N, len(jobs)], "ranges_half_open": True,
        "seed": SEED, "bootstrap_seed": BOOT_SEED, "bootstrap_resamples": N_BOOT,
        "lr": LR, "optimizer": "Adam(default betas/eps), zero hidden bias initialization",
        "loss": "cross_entropy over all non-padding target tokens in batch, as original invert",
        "batch": BATCH, "calibration_batch": CALIB_N,
        "primary": "rouge_score ROUGE-1 F1 gain vs common saved baseline, no stemming",
        "secondary": "ROUGE-L F1; legacy ROUGE-1 is descriptive only",
        "edit_distance": "Levenshtein on rouge_score lowercase alphanumeric tokens / max(lengths,1)",
        "injection": "nonempty intersection of objects(fake) and objects(caption); no raw-word fallback",
        "global_selection": "closest calibration mean ROUGE-1 gain; unrounded; lowest-step tie break",
        "per_image_gain_selection": "closest gain on fixed grid; evaluation-target-aware diagnostic, NOT deployable",
        "per_image_edit_selection": "closest free normalized edit distance on fixed grid; diagnostic, NOT deployable",
        "ties": "lowest step in prespecified increasing grid; no injection-based selection",
        "baseline": str(baseline_path), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "test_start": TEST_START, "model_training": False, "deterministic_algorithms": True,
        "tf32": False, "bootstrap": "paired percentile 95% CIs, resample unique evaluation images; selection fixed",
    }
    sources = [swap_path, baseline_path, Path(CACHE_PATH)]
    sources += [Path(SAVE_DIR) / n for n in ["tok_vocab.json", "model_proj.pt", "model_guided.pt", "plans_test.pt"]]
    sources += [root / n for n in [
        "run_rev_matched_effect_audit.py", "run_rev_matched_effect_control.py",
        "run_rev_inversion_protocol.py", "run_rev_eval_metrics.py", "run_rev_swap_injection.py",
        "run_cap_improved.py", "run_captioning.py", "run_summarization.py", "run_sum_masked.py",
        "paper_cvpr/main.tex", "paper_cvpr/supp.tex", "results/revision/matched_effect_control.json",
    ]]
    print("Hashing existing inputs (including cached features); no downloads", flush=True)
    hashes = {str(p.relative_to(root)): sha256(p) for p in sources}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    tok, proj, dec = load_models()
    assert all(not p.requires_grad for m in [proj, dec] for p in m.parameters())
    # mmap avoids reading the entire feature tensor into RAM just to select 901 rows.
    feats = torch.load(CACHE_PATH, weights_only=True, map_location="cpu", mmap=True)
    plans = torch.load(Path(SAVE_DIR) / "plans_test.pt", weights_only=True, map_location="cpu")
    checkpoints.mkdir()
    captions = {"saved_baseline": baselines}
    captions.update({f"{v}_300": [r[v] for r in jobs] for v in ["free", "onehot", "hidden"]})
    captions.update({f"hidden_{s}": [] for s in STEPS})
    checkpoint_files = []
    trajectory_check = None
    zero_bias_vs_no_bias = []

    def run_chunk(start, end):
        nonlocal trajectory_check
        chunk = jobs[start:end]
        indices = [j["image"] for j in chunk]
        with torch.no_grad():
            X = proj(feats[[TEST_START + i for i in indices]].to(DEVICE))
        A = plans[indices].to(DEVICE)
        target = torch.tensor([tok.enc(j["target"], max_len=L_CAP, bos=True, eos=True) for j in chunk], device=DEVICE)
        delta = torch.nn.Parameter(torch.zeros(len(chunk), L_CAP, X.shape[-1], device=DEVICE))
        opt = torch.optim.Adam([delta], lr=LR)
        valid = target[:, 1:] != 0
        states = {}
        local_caps = {}
        for step in range(max(STEPS) + 1):
            if step in STEPS:
                local_caps[step] = [tok.dec(t) for t in dec_generate(dec, X, A, delta).cpu().tolist()]
                captions[f"hidden_{step}"].extend(local_caps[step])
                states[str(step)] = delta.detach().cpu().clone()
                if step == 0:
                    unbiased = [tok.dec(t) for t in dec_generate(dec, X, A).cpu().tolist()]
                    zero_bias_vs_no_bias.extend([a != b for a, b in zip(local_caps[0], unbiased)])
            if step == max(STEPS):
                break
            pred = dec_forward(dec, X, target[:, :-1], A, delta)
            loss = F.cross_entropy(pred[valid], target[:, 1:][valid])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if start == 0:
            _, reference_delta, reference_tokens = invert(dec, X, A, target, "hidden", 2, LR)
            reference_caps = [tok.dec(t) for t in reference_tokens.cpu().tolist()]
            max_error = float((reference_delta.cpu() - states["2"]).abs().max())
            trajectory_check = {"n": len(chunk), "step": 2, "max_abs_delta_error": max_error,
                                "caption_mismatches": sum(a != b for a, b in zip(reference_caps, local_caps[2]))}
            assert max_error == 0 and trajectory_check["caption_mismatches"] == 0
        path = checkpoints / f"hidden_bias_{start:04d}_{end:04d}.pt"
        with path.open("xb") as f:
            torch.save({"job_positions": list(range(start, end)), "image_ids": indices,
                        "config_sha256": config_hash, "delta_by_step": states,
                        "captions_by_step": local_caps}, f)
        checkpoint_files.append({"path": str(path.relative_to(root)), "sha256": sha256(path)})
        print(f"Saved hidden trajectory {start}:{end}, steps {STEPS}; {time.time()-started:.1f}s", flush=True)

    # Lock the global count before generating any evaluation trajectories.
    run_chunk(0, CALIB_N)
    calibration_caps = {k: v[:CALIB_N] for k, v in captions.items()}
    calibration_metrics = {k: caption_metrics(v, jobs[:CALIB_N], baselines[:CALIB_N], refs[:CALIB_N])
                           for k, v in calibration_caps.items()}
    free_mean = float(np.mean(calibration_metrics["free_300"]["gain_rouge1"]))
    sweep = [{"step": s,
              "mean_gain_rouge1": float(np.mean(calibration_metrics[f"hidden_{s}"]["gain_rouge1"]))}
             for s in STEPS]
    for r in sweep:
        r["signed_mean_gain_residual_vs_free"] = r["mean_gain_rouge1"] - free_mean
    chosen_step = min(sweep, key=lambda r: abs(r["signed_mean_gain_residual_vs_free"]))["step"]
    calibration = {"n": CALIB_N, "free_mean_gain_rouge1": free_mean, "sweep": sweep,
                   "selected_step": chosen_step, "captions": calibration_caps, "metrics": calibration_metrics}
    with (checkpoints / "calibration_lock.json").open("x") as f:
        json.dump({"config": config, "config_sha256": config_hash, "calibration": calibration}, f, indent=2)
    print(f"CALIBRATION LOCK: global step={chosen_step}, free mean gain={free_mean:.9f}", flush=True)
    for start in range(CALIB_N, len(jobs), BATCH):
        run_chunk(start, min(start + BATCH, len(jobs)))

    metrics = {k: caption_metrics(v, jobs, baselines, refs) for k, v in captions.items()}
    mismatch = [i for i, (a, b) in enumerate(zip(baselines, captions["hidden_0"])) if a != b]
    baseline_audit = {
        "identical": not mismatch, "n_mismatched_all901": len(mismatch),
        "n_mismatched_calibration150": sum(i < CALIB_N for i in mismatch),
        "n_mismatched_evaluation751": sum(i >= CALIB_N for i in mismatch),
        "mismatched_job_positions": mismatch,
        "zero_bias_vs_no_bias_mismatch_count": sum(zero_bias_vs_no_bias),
        "policy": "FLAG mismatches, do not replace the common saved baseline or rebase hidden gains",
        "warning": "Step 0 is not an identity relative to saved captions" if mismatch else None,
    }
    print("BASELINE AUDIT:", json.dumps(baseline_audit), flush=True)
    evaluation = slice(CALIB_N, None)
    ev_caps = {k: v[evaluation] for k, v in captions.items()}
    ev_metrics = {k: {m: np.asarray(a[evaluation]) for m, a in v.items()} for k, v in metrics.items()}
    n = len(jobs) - CALIB_N
    selections = {}
    for name, match_metric in [("nearest_gain_diagnostic", "gain_rouge1"),
                               ("nearest_edit_diagnostic", "normalized_token_edit")]:
        grid = np.stack([ev_metrics[f"hidden_{s}"][match_metric] for s in STEPS])
        choice = np.abs(grid - ev_metrics["free_300"][match_metric]).argmin(axis=0)
        selected_steps = np.asarray(STEPS)[choice]
        ev_caps[name] = [ev_caps[f"hidden_{step}"][i] for i, step in enumerate(selected_steps)]
        ev_metrics[name] = {m: np.array([ev_metrics[f"hidden_{step}"][m][i]
                                       for i, step in enumerate(selected_steps)]) for m in metrics["free_300"]}
        selections[name] = {"steps": selected_steps.tolist(), "selection_metric": match_metric,
                            "step_counts": {str(s): int((selected_steps == s).sum()) for s in STEPS},
                            "free_outside_hidden_grid_range_count": int(((ev_metrics["free_300"][match_metric] < grid.min(0)) |
                                                                        (ev_metrics["free_300"][match_metric] > grid.max(0))).sum()),
                            "deployable": False, "uses_evaluation_target_fit_for_selection": match_metric == "gain_rouge1"}
    global_name = "global_calibrated"
    ev_caps[global_name] = ev_caps[f"hidden_{chosen_step}"]
    ev_metrics[global_name] = ev_metrics[f"hidden_{chosen_step}"]

    # The same image-resampling weights are used for every method and contrast.
    weights = np.random.default_rng(BOOT_SEED).multinomial(n, np.full(n, 1 / n), size=N_BOOT).astype(float) / n
    names = list(ev_metrics)
    metric_names = list(metrics["free_300"])
    matrix = np.column_stack([ev_metrics[name][m] for name in names for m in metric_names])
    boot = weights @ matrix
    boot = boot.reshape(N_BOOT, len(names), len(metric_names))
    summaries = {}
    for j, name in enumerate(names):
        summaries[name] = {m: distribution(ev_metrics[name][m], boot[:, j, k]) for k, m in enumerate(metric_names)}
        inj = ev_metrics[name]["injection"]
        inj_boot = boot[:, j, metric_names.index("injection")]
        summaries[name]["injection"].update({"count": int(inj.sum()), "n": n,
                                             "resampled_count_ci95": np.quantile(inj_boot * n, [.025, .975]).tolist()})
    contrasts = {}
    for name in [global_name, "nearest_gain_diagnostic", "nearest_edit_diagnostic", "hidden_0", "hidden_300", "free_300"]:
        contrasts[name] = {}
        for reference in ["free_300", "saved_baseline"]:
            if name == reference:
                continue
            contrasts[name][f"minus_{reference}"] = {
                m: distribution(ev_metrics[name][m] - ev_metrics[reference][m],
                                boot[:, names.index(name), k] - boot[:, names.index(reference), k])
                for k, m in enumerate(metric_names)
            }
            inj, other = ev_metrics[name]["injection"], ev_metrics[reference]["injection"]
            contrasts[name][f"minus_{reference}"]["injection"].update({
                "count_difference": int(inj.sum() - other.sum()),
                "only_method_injects": int(((inj == 1) & (other == 0)).sum()),
                "only_reference_injects": int(((inj == 0) & (other == 1)).sum()),
                "both_inject": int(((inj == 1) & (other == 1)).sum()),
            })
    residuals = {}
    for name in [global_name, "nearest_gain_diagnostic", "nearest_edit_diagnostic"]:
        residuals[name] = {}
        for m in ["gain_rouge1", "gain_rougeL", "normalized_token_edit"]:
            d = ev_metrics[name][m] - ev_metrics["free_300"][m]
            absolute = np.abs(d)
            residuals[name][m] = {
                "signed": contrasts[name]["minus_free_300"][m],
                "absolute": distribution(absolute, weights @ absolute),
                "rmse": float(np.sqrt(np.mean(d * d))),
                "counts_within_absolute_tolerance": {str(t): int((absolute <= t).sum()) for t in [0, .01, .025, .05, .1]},
                "per_image_signed": d.tolist(), "per_image_absolute": absolute.tolist(),
            }
    # Rehash models, old results, and manuscript files to verify they stayed unchanged.
    preserved = [p for p in sources if p != Path(CACHE_PATH)]
    unchanged = all(sha256(p) == hashes[str(p.relative_to(root))] for p in preserved)
    assert unchanged, "An input changed during the audit"
    result = {
        "config": config, "config_sha256": config_hash, "input_sha256": hashes,
        "reproducibility": {
            "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "numpy": np.__version__, "rouge_score": importlib.metadata.version("rouge-score"),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu0_before_start": gpu_status, "argv": sys.argv,
            "models_old_results_and_manuscripts_unchanged": unchanged,
            "trajectory_vs_original_invert": trajectory_check,
            "checkpoint_files": checkpoint_files,
        },
        "n_all": len(jobs), "n_calibration": CALIB_N, "n_evaluation": n,
        "jobs": [{"position": i, "split": "calibration" if i < CALIB_N else "evaluation",
                  "image": j["image"], "k": j["k"], "target": j["target"], "fake": j["fake"],
                  "original_ref": refs[i]} for i, j in enumerate(jobs)],
        "calibration": calibration, "baseline_audit": baseline_audit,
        "all901_captions": captions, "all901_metric_arrays": metrics,
        "evaluation": {
            "image_ids": [j["image"] for j in jobs[CALIB_N:]], "global_step": chosen_step,
            "captions": ev_caps,
            "metric_arrays": {k: {m: a.tolist() for m, a in v.items()} for k, v in ev_metrics.items()},
            "selections": selections, "summaries": summaries, "paired_contrasts": contrasts,
            "matching_residuals": residuals,
        },
        "limitations": [
            "Single saved901 swap cohort/seed/small trained captioner, not full COCO or cross-model replication.",
            "First150 is a fixed ordered calibration subset, not a random or externally independent calibration dataset; final751 excludes it.",
            "Global selection matches only a calibration mean approximately; held-out per-image effects and distributions need not match.",
            "Nearest-gain matching explicitly uses evaluation targets and free outcomes: descriptive target-aware diagnostic, not a deployable stopping map or unbiased causal contrast.",
            "Nearest-edit matching also conditions on realized free outcomes and target-driven hidden trajectories; it is descriptive, not a deployable or causal control.",
            "Matching restricted to the prespecified 0..20 grid, with lowest-step ties; no exact matching or interpolation is assumed.",
            "Bootstrap CIs condition on the fixed selected checkpoints/cohort/models, not calibration selection, training seed or target-generation uncertainty; no multiplicity correction or equivalence claim.",
            "Sparse injection yields discrete, potentially degenerate percentile bootstrap intervals; absence of significance is not equivalence.",
            "Object injection is the existing lexical objects() proxy; absence in references does not prove visual absence.",
            "Saved free/hidden300 captions are reused, not rerun; original swap batch500 differs from calibration150 and evaluation500/251, and Adam epsilon/numerics can make batch size matter.",
            "Step0 mismatches are flagged, not repaired. All methods retain the same saved baseline; nonzero step0 gains/edits therefore include regeneration differences.",
            "Only input biases are checkpointed; optimizer states are not saved for resuming. Models, fixed plans and features are hash-identified inputs.",
        ],
        "elapsed_seconds": time.time() - started,
    }
    with output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print("FINAL HELD-OUT RESULTS", flush=True)
    for name in ["saved_baseline", "free_300", "hidden_0", global_name, "nearest_gain_diagnostic", "nearest_edit_diagnostic", "hidden_300"]:
        s = summaries[name]
        print(json.dumps({"method": name, "target_rouge1": s["target_rouge1"]["mean"],
                          "gain_rouge1": s["gain_rouge1"]["mean"],
                          "edit_distance": s["normalized_token_edit"]["mean"],
                          "injection": s["injection"]}), flush=True)
    print(f"Saved {output}, sha256={sha256(output)}, elapsed={time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
