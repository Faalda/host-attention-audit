"""run_rev_host_w1_donor_nullmean.py — review response W1/Q1 (HBDA-LLM 2026, reviewer #3).

"Does free-plan inversion keep its held-out gain on all-mean or donor features?"

Reuses the exact free-plan inversion machinery (run_clean_evaluation.optimize,
variant="free", 300 Adam steps, lr .05, same fault-plan A_base, same 6000-job
list) that produced the paper's own-feature "Free plan, 300" held-out gain
(+0.05862 [0.05421,0.06321], Table III), but swaps the decoder's IMAGE
features for:
  donor : results/visual_headroom_provenance/seed{S}/donor_features.pt
          (already-built, hash-chained donor features from the post-lock
          provenance control; fixed derangement, same anchor masks)
  null  : every position replaced by the saved training mean (train_mean.pt)

Held-out baseline for each condition is the SAME condition's NO-INVERSION
fault-plan caption (donor_fault / null_fault), already generated and saved
by run_visual_provenance_control.py's results.json — i.e. exactly the
"same quantity for the baseline" comparison the paper's own protocol uses,
just computed for the donor/null feature condition instead of own.

Only writes under results/revision/host_w1_donor_nullmean/. No existing
file is modified. No training, no downloads.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from run_clean_evaluation import (atomic_json, read_json, sha256, load_models, model_digest,
                                  validate_source, protocol_jobs, protocol_dump, optimize,
                                  MODEL_FILES, STEPS, LR)
from run_visual_headroom_control import routing_plans
from paper_cvpr.rescore_rouge import score, paired, image_scores

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/revision/host_w1_donor_nullmean"
MODELS = ROOT / "results/clean_small_stablekl"
CONTROL = ROOT / "results/visual_headroom_control"
PROVENANCE = ROOT / "results/visual_headroom_provenance"
SOURCE = ROOT / "results/revision/inversion_protocol_n1000_s0.json"
SEEDS = (42, 7, 13)
CONDITIONS = ("donor", "null")
BATCH = 250


def gpu_snapshot():
    import subprocess
    def q(kind, cols):
        return subprocess.check_output(
            ["nvidia-smi", f"--query-{kind}={cols}", "--format=csv,noheader,nounits"], text=True
        ).strip().splitlines()
    return {"time": time.time(), "gpus": q("gpu", "index,uuid,memory.total,memory.free")}


def check_gpu0():
    snap = gpu_snapshot()
    row = next(r.split(", ") for r in snap["gpus"] if r.split(", ")[0] == "0")
    if float(row[3]) < 24000:
        raise RuntimeError("GPU0 has less than 24000 MiB free; refusing allocation")
    return row[1]


def source_hashes(seed):
    model_dir = MODELS / f"seed{seed}"
    paths = {
        "donor_features": PROVENANCE / f"seed{seed}/donor_features.pt",
        "provenance_results": PROVENANCE / f"seed{seed}/results.json",
        "anchor_mask": CONTROL / f"seed{seed}/test/anchor_mask.pt",
        "train_mean": CONTROL / f"seed{seed}/train_mean.pt",
        "control_complete": CONTROL / f"seed{seed}/test/complete.json",
        "source": SOURCE,
    }
    for name in MODEL_FILES:
        paths[f"model_{name}"] = model_dir / name
    return {k: sha256(v) for k, v in paths.items()}


def build_images(seed, condition, mu, n=1000):
    if condition == "donor":
        x = torch.load(PROVENANCE / f"seed{seed}/donor_features.pt", map_location="cpu",
                       weights_only=True, mmap=True)
        if x.shape != (n, 256, 256):
            raise ValueError("Unexpected donor feature shape")
        return x
    if condition == "null":
        return mu[None, None].expand(n, 256, 256).contiguous()
    raise ValueError(condition)


def run_one(seed, condition, uuid):
    folder = OUT / f"seed{seed}_{condition}"
    folder.mkdir(exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    torch.cuda.set_per_process_memory_fraction(.35, 0)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    hashes = source_hashes(seed)
    atomic_json(folder / "input_hashes.json", hashes)

    tok, models = load_models(MODELS / f"seed{seed}", "cuda")
    before = model_digest(models)
    control_state = read_json(CONTROL / f"seed{seed}/test/complete.json")["state_sha256_after"]
    if before != control_state:
        raise RuntimeError("Model state does not match the state used for the paper's own-feature control")
    assert all(not m.training for model in models.values() for m in model.modules())

    records = validate_source(read_json(SOURCE))
    jobs = protocol_jobs(records)
    assert len(jobs) == 6000

    mu = torch.load(CONTROL / f"seed{seed}/train_mean.pt", map_location="cpu", weights_only=True)
    mask = torch.load(CONTROL / f"seed{seed}/test/anchor_mask.pt", map_location="cpu", weights_only=True)
    images = build_images(seed, condition, mu)

    prov = read_json(PROVENANCE / f"seed{seed}/results.json")
    baselines = prov["captions"][f"{condition}_fault"]
    if len(baselines) != 1000:
        raise ValueError("Expected 1000 baseline captions")

    captions, stats = [], []
    t0 = time.time()
    for start in range(0, len(jobs), BATCH):
        chunk = jobs[start:start + BATCH]
        idx = [j["plan_index"] for j in chunk]
        X = images[idx].cuda()
        if not torch.isfinite(X).all():
            raise ValueError("Nonfinite input features")
        a = routing_plans(mask[idx].cuda())["fault"]
        target = torch.tensor([tok.enc(j["target"], 40, bos=True, eos=True) for j in chunk], device="cuda")
        tokens, st, _ = optimize(models["guided"], X, a, target, "free", steps=STEPS, lr=LR)
        captions += [tok.dec(t) for t in tokens]
        stats += st
        print(f"seed{seed} {condition}: {min(start + BATCH, len(jobs))}/{len(jobs)} "
              f"{time.time() - t0:.0f}s", flush=True)

    after = model_digest(models)
    assert before == after
    assert all(not p.requires_grad and p.grad is None for m in models.values() for p in m.parameters())

    dump = protocol_dump(records, baselines, jobs, {"free": captions}, {"free": stats},
                          dict(seed=seed, condition=condition,
                               baseline_definition=f"{condition}_fault caption from results/visual_headroom_provenance",
                               feature_source=("results/visual_headroom_provenance donor_features.pt"
                                               if condition == "donor" else "train_mean.pt broadcast to all 256 positions"),
                               plan_a_base="fault routing plan from anchor_mask.pt (same as own-feature control)",
                               steps=STEPS, lr=LR))
    atomic_json(folder / "inversion_protocol_n1000_s0.json", dump)

    rows = [image_scores(r, "free")[0] for r in dump["records"]]
    per_image = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    heldout = per_image["heldout"]
    summary = paired(heldout[:, 0], heldout[:, 1])
    mismatch_target = per_image["mismatch_target"]
    mismatch_summary = paired(mismatch_target[:, 0], mismatch_target[:, 1])

    atomic_json(folder / "complete.json", dict(
        state_sha256_before=before, state_sha256_after=after, all_parameters_frozen=True,
        seconds=time.time() - t0, n_jobs=len(jobs)))
    atomic_json(folder / "summary.json", dict(
        seed=seed, condition=condition, heldout=summary, mismatch_target=mismatch_summary,
        heldout_per_image=heldout.tolist()))
    print(f"seed{seed} {condition} heldout gain {summary['delta']:+.5f} "
          f"CI {summary['ci95']}", flush=True)
    return summary, heldout


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    uuid0 = check_gpu0()
    pooled = {}
    per_seed = {}
    for condition in CONDITIONS:
        heldout_arrays = []
        per_seed[condition] = {}
        for seed in SEEDS:
            summary, heldout = run_one(seed, condition, uuid0)
            per_seed[condition][str(seed)] = summary
            heldout_arrays.append(heldout)
            torch.cuda.empty_cache()
        pooled_arr = np.mean(heldout_arrays, axis=0)
        pooled[condition] = paired(pooled_arr[:, 0], pooled_arr[:, 1])
        print(f"=== POOLED {condition}: heldout gain {pooled[condition]['delta']:+.5f} "
              f"CI {pooled[condition]['ci95']} ===", flush=True)

    atomic_json(OUT / "results.json", dict(
        pooled=pooled, per_seed=per_seed,
        comparison_own_feature_control="Table III free-plan held-out gain: +0.05862 [0.05421, 0.06321]",
        methodology="Identical free-plan inversion (300 Adam steps, lr .05, same fault-plan A_base, "
                    "same 6000-job list) as the paper's own-feature control; only the decoder's input "
                    "features (X) differ. Baseline per condition is that condition's no-inversion "
                    "fault-plan caption (donor_fault / null_fault), matching the paper's "
                    "'same quantity for the baseline' convention. Pooled CI: image-level seed-average "
                    "before paired bootstrap (10000 resamples, seed 0), as in the paper.",
        status="complete"))
    print("Complete:", OUT / "results.json", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        if OUT.exists():
            atomic_json(OUT / f"failure_pid{os.getpid()}.json",
                       dict(error=repr(exc), traceback=traceback.format_exc()))
        raise
