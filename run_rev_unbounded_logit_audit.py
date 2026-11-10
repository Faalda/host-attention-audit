"""Verify completed own-model artifacts and probe BLIP-2 strictly offline."""
import os

if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
    raise RuntimeError("GPU0 only")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from rouge_score import rouge_scorer

from run_rev_unbounded_logit_control import (
    additive_logits, load_models, model_digest, sha256, write_json,
    CACHE_PATH, DEVICE, L_CAP, SAVE_DIR, TEST_START, dec_generate, make_plan,
)
from run_rev_eval_metrics import norm, halluc


def main():
    started = time.perf_counter()
    root = Path(__file__).resolve().parent
    out = root / "results/revision/unbounded_logit_control_n200_s0"
    with (out / "results.json").open() as f:
        result = json.load(f)
    for name, digest in result["sources"].items():
        path = out / name if name == "preregistered_jobs.json" else root / name
        assert sha256(path) == digest, name
    with (root / "results/revision/inversion_protocol_n1000_s0.json").open() as f:
        source = json.load(f)
    jobs = result["jobs"]
    assert len(jobs) == 600 and len({j["image"] for j in jobs}) == 200
    assert Counter(j["kind"] for j in jobs) == {"true": 400, "mismatch": 200}
    for i, job in enumerate(jobs):
        original = source["records"][job["image"]]
        assert job["job"] == i and job["refs"] == original["refs"]
        assert result["captions"]["saved_baseline"][i] == original["baseline"]
        if job["kind"] == "true":
            assert job["k"] in [0, 1] and job["target"] == original["refs"][job["k"]]
            expected = original["variants"]["free"]["true"][job["k"]]
        else:
            assert job["target"] == original["mismatch_target"]
            assert job["mismatch_source"] != job["image"]
            assert job["target"] in source["records"][job["mismatch_source"]]["refs"]
            expected = original["variants"]["free"]["mismatch"]
        assert result["captions"]["saved_free300"][i] == expected
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)
    for method, caps in result["captions"].items():
        for kind in ["true", "mismatch"]:
            selected = [i for i, j in enumerate(jobs) if j["kind"] == kind]
            expected = result["metrics"]["summary"][kind][method]
            values = [scorer.score(norm(jobs[i]["target"]), norm(caps[i]))["rouge1"].fmeasure for i in selected]
            assert abs(np.mean(values) - expected["target_r1"]) < 1e-12
            proxy = halluc([caps[i] for i in selected], [jobs[i]["refs"] for i in selected])
            assert proxy["chair_s_proxy"] == round(expected["chair_s_proxy"], 4)
            assert proxy["chair_i_proxy"] == round(expected["chair_i_proxy"], 4)

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    tok, proj, dec = load_models()
    assert model_digest([proj, dec]) == result["validation"]["model_state_sha256_before"]
    feats = torch.load(CACHE_PATH, map_location="cpu", weights_only=True, mmap=True)
    plans = torch.load(Path(SAVE_DIR) / "plans_test.pt", map_location="cpu", weights_only=True)
    replay = []
    numerical = {}
    for method in ["bounded300", "unbounded300"]:
        rows = []
        for checkpoint in result["checkpoints"]:
            if not checkpoint["path"].startswith(method):
                continue
            path = out / checkpoint["path"]
            assert sha256(path) == checkpoint["sha256"]
            saved = torch.load(path, weights_only=True, map_location="cpu")
            assert torch.isfinite(saved["parameter"]).all()
            assert saved["parameter"].shape[1:] == (40, 256)
            assert [tok.dec(t) for t in saved["tokens"].tolist()] == [result["captions"][method][i] for i in saved["job_ids"]]
            indices = [jobs[i]["image"] for i in saved["job_ids"]]
            A = plans[indices]
            parameter = saved["parameter"]
            effective = (make_plan(parameter, "free").clamp(min=1e-6).log() if method == "bounded300"
                         else A.clamp(min=1e-6).log() + parameter)
            span = effective.max(-1).values - effective.min(-1).values
            valid = saved["target_tokens"][:, 1:] != 0
            for b, job_id in enumerate(saved["job_ids"]):
                used = span[b, :-1][valid[b]]
                rows.append({"job": job_id, "mean_span": used.mean().item(), "max_span": used.max().item(),
                             "fraction_rows_beyond_native_limit": (used > -np.log(1e-6) + 1e-5).float().mean().item()})
            # Replay first two jobs in every checkpoint, including the mismatch-only chunk.
            idx = indices[:2]
            with torch.no_grad():
                X = proj(feats[[TEST_START + i for i in idx]].to(DEVICE))
                if method == "bounded300":
                    tokens = dec_generate(dec, X, make_plan(parameter[:2].to(DEVICE), "free"))
                else:
                    with additive_logits(dec, parameter[:2].to(DEVICE)):
                        tokens = dec_generate(dec, X, A[:2].to(DEVICE))
            caps = [tok.dec(t) for t in tokens.cpu().tolist()]
            assert caps == [result["captions"][method][i] for i in saved["job_ids"][:2]]
            replay.append({"checkpoint": path.name, "jobs": saved["job_ids"][:2], "captions_match": True})
        numerical[method] = {}
        for kind in ["true", "mismatch"]:
            selected = [row for row in rows if jobs[row["job"]]["kind"] == kind]
            stats = [result["control_stats"][method][row["job"]] for row in selected]
            numerical[method][kind] = {
                "mean_target_ce": float(np.mean([s["final_target_ce"] for s in stats])),
                "mean_pre_gate_span": float(np.mean([r["mean_span"] for r in selected])),
                "max_pre_gate_span": max(r["max_span"] for r in selected),
                "fraction_jobs_exceeding_native_span_limit": float(np.mean([r["fraction_rows_beyond_native_limit"] > 0 for r in selected])),
                "mean_fraction_used_rows_exceeding_native_span_limit": float(np.mean([r["fraction_rows_beyond_native_limit"] for r in selected])),
            }
    assert model_digest([proj, dec]) == result["validation"]["model_state_sha256_after"]

    # Do not attempt downloads or infer availability from old caption dumps.
    model_id = "Salesforce/blip2-opt-2.7b"
    availability = {"model_id": model_id, "cache": HF_HUB_CACHE, "local_files_only": True,
                    "checked_after_own_run": True, "gpu_run_performed": False}
    try:
        snapshot = snapshot_download(model_id, local_files_only=True)
        availability.update({"snapshot_found": True, "snapshot": snapshot,
                             "status": "Local snapshot exists; completeness/loadability still needs verification"})
    except Exception as exc:
        availability.update({"snapshot_found": False, "error_type": type(exc).__name__, "error": str(exc),
                             "status": "BLIP-2 matched-simplex/gain control not run: no locally cached model snapshot"})
    disagreements = {}
    for kind in ["true", "mismatch"]:
        selected = [i for i, j in enumerate(jobs) if j["kind"] == kind]
        disagreements[kind] = sum(result["captions"]["bounded300"][i] != result["captions"]["saved_free300"][i] for i in selected)
    audit = {
        "audit_source_sha256": sha256(__file__), "result_sha256": sha256(out / "results.json"),
        "all_input_and_checkpoint_hashes_reverified": True, "exact_slots_and_saved_captions_verified": True,
        "independent_standard_r1_rescore_matches": True, "repository_chair_proxy_rescore_matches": True,
        "frozen_models_reload_matches_before_and_after": True, "checkpoint_replays": replay,
        "saved_free_vs_rerun_disagreements": disagreements,
        "numerical_scope": numerical, "blip2_availability": availability,
        "paired_image_bootstrap": result["metrics"]["paired_image_bootstrap"],
        "limitations": [
            "200 fixed images, 400 true-reference slots and 200 saved shuffled slots; one initialization, no seed sweep",
            "Target-aware test-time optimization is an oracle diagnostic, not deployable captioning",
            "300 steps does not establish converged or maximum achievable capacity",
            "Gain-matched unbounded control retains each learned gate; not a unit-gain raw additive control",
            "Without the native 1e-6 plan floor, log-softmax differs from unconstrained logits only by a row constant; this isolates the implemented numerical bound, not simplex normalization in general",
            "CHAIR-style reference-category proxy, not official segmentation/object-annotation CHAIR; no evidence of higher proxy hallucination here",
            "Historical free300 batch grouping/environment not fully recorded; fresh bounded run matches this control's batches",
            "CIs resample images, not image-target slots; no multiple-comparison correction",
        ],
        "audit_seconds": time.perf_counter() - started,
    }
    write_json(out / "audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
