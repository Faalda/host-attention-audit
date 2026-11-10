"""CPU artifact verification and trained-oracle reference comparison; no generation."""
import sys
import os
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for clean_pipeline./inversion. imports
from inversion.run_rev_eval_metrics import boot_ci, r1


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--directory", default="results/revision/host_predictor_failure_n200_s0_audit")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    directory = root / args.directory
    artifact_hashes = load(directory / "artifact_sha256.json")
    source_hashes = load(directory / "sources.json")["sha256"]
    for name, expected in artifact_hashes.items():
        assert digest(directory / name) == expected, name
    for name, expected in source_hashes.items():
        assert digest(root / name) == expected, name
    data = load(directory / "results.json")
    source = load(root / "results/revision/inversion_protocol_n1000_s0.json")
    validation = data["validation"]
    assert validation["state_sha256_before"] == validation["state_sha256_after"]
    assert validation["frozen_gates"] == validation["frozen_gates_after"]
    assert validation["trained_caption_vs_saved_disagreements"] == 0
    assert validation["failed_caption_vs_scaling_saved_disagreements"] == 0
    assert len(data["jobs"]) == len(data["rejected_inversion_jobs"]) == 600
    assert "degraded_init_free300" not in data["captions"]
    for job, rejected in zip(data["jobs"], data["rejected_inversion_jobs"]):
        assert job["job"] == rejected["job"]
        assert rejected["fresh_init"] and rejected["initial_target_ce_is_nan"]
        assert rejected["nonfinite_initial_parameters"] == 10240 and rejected["optimizer_steps_executed"] == 0
        rec = source["records"][job["image"]]
        assert job["refs"] == rec["refs"] and job["mismatch_source"] == rec["mismatch_source"]
        assert job["target"] == (rec["refs"][job["slot"]] if job["kind"] == "true" else rec["mismatch_target"])
        expected = rec["variants"]["free"]["true"][job["slot"]] if job["kind"] == "true" else rec["variants"]["free"]["mismatch"]
        assert data["captions"]["saved_trained_oracle_reference"][job["job"]] == expected
        assert data["captions"]["trained_forward"][job["job"]] == rec["baseline"]
    plans = torch.load(directory / "forward_plans.pt", weights_only=True, map_location="cpu")
    assert torch.isnan(plans["degraded_forward"]).all()
    for name in ("trained_forward", "uniform_forward"):
        assert torch.isfinite(plans[name]).all()
        torch.testing.assert_close(plans[name].sum(-1), torch.ones(200, 40))
    assert torch.equal(plans["uniform_forward"], torch.full((200, 40, 256), 1 / 256))
    saved = torch.load(root / "results/saved_plans/plans_test.pt", weights_only=True, map_location="cpu")
    assert torch.equal(plans["trained_forward"], saved[:200])
    failed = torch.load(root / "results/revision/rev_ckpt/scaling_frac1.00_nmp1.pt", weights_only=True, map_location="cpu")
    assert sum(int(torch.isnan(v).sum()) for v in failed.values()) == 527872
    assert all(torch.isnan(v).all() for k, v in failed.items() if k != "null_X")
    assert torch.isfinite(failed["null_X"]).all()
    per_image = {}
    for kind in ("true", "mismatch"):
        per_image[kind] = {}
        for name, captions in data["captions"].items():
            values = {key: [[] for _ in range(200)] for key in ("target_r1", "heldout_r1", "grounding_r1")}
            for job, caption in zip(data["jobs"], captions):
                if job["kind"] != kind:
                    continue
                references = {"target_r1": [job["target"]], "grounding_r1": job["refs"],
                              "heldout_r1": [r for k, r in enumerate(job["refs"]) if kind != "true" or k != job["slot"]]}
                for key, refs in references.items():
                    value = r1(caption, refs)
                    assert value == data["metrics"]["per_job"][name][job["job"]][key]
                    values[key][job["image"]].append(value)
            per_image[kind][name] = {key: np.array([np.mean(v) for v in rows]) for key, rows in values.items()}
            for key, values in per_image[kind][name].items():
                assert values.mean() == data["metrics"]["summary"][kind][name][key]
    reference_comparisons = {}
    for kind, methods in per_image.items():
        reference_comparisons[kind] = {}
        for (a, b) in (("saved_trained_oracle_reference", "trained_forward"), ("uniform_forward", "trained_forward")):
            reference_comparisons[kind][a + "_minus_" + b] = {}
            for key in methods[a]:
                mean, lo, hi = boot_ci(methods[a][key] - methods[b][key], n=10000, seed=0)
                reference_comparisons[kind][a + "_minus_" + b][key] = {"delta": mean, "ci95": [lo, hi]}
                if a == "uniform_forward":
                    previous = data["metrics"]["paired_image_bootstrap"][kind][a + "_minus_" + b][key]
                    assert previous["delta"] == mean and previous["ci95"] == [lo, hi]
        for a in methods:
            if a == "degraded_forward":
                continue
            for key in methods[a]:
                mean, lo, hi = boot_ci(methods[a][key] - methods["degraded_forward"][key], n=10000, seed=0)
                previous = data["metrics"]["paired_image_bootstrap"][kind][a + "_minus_degraded_forward"][key]
                assert previous["delta"] == mean and previous["ci95"] == [lo, hi]
    report = {"passed": True, "artifact_hashes_verified": len(artifact_hashes), "source_hashes_verified": len(source_hashes),
              "independent_caption_rescoring_and_image_bootstrap_match": True,
              "all_600_jobs_rejected_before_optimizer": True, "all_saved_oracle_captions_exactly_unchanged": True,
              "trained_plans_and_captions_exactly_match_saved": True,
              "failed_checkpoint_nan_parameters": 527872, "failed_checkpoint_total_parameters": 593408,
              "all_failed_parameters_except_null_X_are_nan": True,
              "reference_comparison_note": "Saved trained oracle versus its OWN trained baseline, not a new degraded optimization",
              "reference_comparisons": reference_comparisons,
              "verification_script_sha256": digest(Path(__file__)), "result_sha256": digest(directory / "results.json")}
    with open(directory / "verification.json", "x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
