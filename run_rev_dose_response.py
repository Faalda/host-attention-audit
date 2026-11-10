"""run_rev_dose_response.py -- reviewer-requested dose-response sweep over fault severity K_a.

Addresses HBDA-LLM 2026 review W1: "The control is also one fault severity, chosen for
greatest repair gain... A dose-response over all three Ka values would give a detection
limit." The paper's reported run used K_a=16 (validation-selected, results already in
results/visual_headroom_control/). This script runs the SAME fault-construction,
fault-aware-repair, and free-plan-inversion pipeline at the two OTHER fixed grid points,
K_a in {4, 64}, with NO re-selection on validation data (fixed, pre-registered grid
points, not a new search). Test images and seeds are identical to the original run.

Only results/revision/host_w1_dose_response is written. All original directories,
checkpoints, masks, and training means are read-only. Reuses library functions from
run_visual_headroom_control.py and run_clean_evaluation.py verbatim; does not
reimplement fault/repair/inversion logic.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from run_clean_evaluation import (atomic_json, read_json, sha256, load_models, model_digest,
                                  validate_source, protocol_jobs, protocol_dump, target_audit,
                                  TEST_START, N_TEST)
from run_visual_headroom_control import (anchor_masks, fault_features, routing_plans, generate,
                                         forward_scores, run_inversions, gpu_snapshot, compact_pair)
from paper_cvpr.rescore_rouge import score, image_scores

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "results/visual_headroom_control"
OUT = ROOT / "results/revision/host_w1_dose_response"
MODELS = ROOT / "results/clean_small_stablekl"
FEATURES = ROOT / "results/dinov2_patch_features_15k.pt"
SOURCE = ROOT / "results/revision/inversion_protocol_n1000_s0.json"
SEEDS = (42, 7, 13)
NEW_KS = (4, 64)
FORWARD_BATCH = 32


def check_gpu(gpu):
    snapshot = gpu_snapshot()
    row = next(r.split(", ") for r in snapshot["gpus"] if r.split(", ")[0] == str(gpu))
    if float(row[3]) < 24000:
        raise RuntimeError(f"GPU{gpu} has less than 24000 MiB free; refusing allocation")
    return row[1], snapshot


@torch.no_grad()
def build_features_fixed_k(models, feats, mu, k, indices):
    clean = torch.cat([models["proj"](feats[chunk].cuda()).cpu()
                       for chunk in torch.tensor(indices).split(FORWARD_BATCH)])
    mask = anchor_masks(indices, k)
    x = fault_features(clean, mu, mask)
    return clean, x, mask


@torch.no_grad()
def forward_all_fixed(tok, models, clean, x, mask, native):
    captions = {name: [] for name in ("fault", "repair", "uniform", "intact")}
    for start in range(0, len(x), FORWARD_BATCH):
        sl = slice(start, start + FORWARD_BATCH)
        xc, xf = clean[sl].cuda(), x[sl].cuda()
        plans = routing_plans(mask[sl].cuda())
        a = native[sl].cuda()
        for name, p in plans.items():
            captions[name] += generate(tok, models["guided"], xf, p)
        captions["intact"] += generate(tok, models["guided"], xc, a)
    return captions


def run_seed_k(seed, k, gpu):
    folder = OUT / f"k{k}" / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    uuid, snapshot = check_gpu(gpu)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise RuntimeError("Worker must expose only the requested physical GPU by UUID")
    torch.cuda.set_per_process_memory_fraction(.35, 0)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    atomic_json(folder / "gpu_before.json", dict(**snapshot, physical_gpu=gpu, uuid=uuid, own_pid=os.getpid()))

    model_dir = MODELS / f"seed{seed}"
    tok, models = load_models(model_dir, "cuda")
    before = model_digest(models)
    assert all(not m.training for model in models.values() for m in model.modules())

    feats = torch.load(FEATURES, map_location="cpu", weights_only=True, mmap=True)
    assert feats.shape == (15000, 256, 768)
    mu = torch.load(ORIGINAL / f"seed{seed}/train_mean.pt", weights_only=True)

    records = validate_source(read_json(SOURCE))
    assert [r["refs"] for r in records] == read_json(model_dir / "caps_test.json")
    jobs = protocol_jobs(records)
    assert len(jobs) == 6000

    clean, x, mask = build_features_fixed_k(models, feats, mu, k, list(range(TEST_START, TEST_START + N_TEST)))
    native = torch.load(model_dir / "plans_test.pt", weights_only=True, mmap=True)
    forward = forward_all_fixed(tok, models, clean, x, mask, native)
    forward_score = forward_scores(forward, [r["refs"] for r in records])
    atomic_json(folder / "forward.json", dict(captions=forward, scores=forward_score))

    caps, stats, traj = run_inversions(folder, jobs, tok, models, x, mask, (0,))
    assert traj[0] == [forward["fault"][j["image"]] for j in jobs]
    for name, cc in forward.items():
        caps[name] = [cc[j["image"]] for j in jobs]
        stats[name] = [dict(steps=0, reference_free=True, inversion=False) for _ in jobs]

    dump = protocol_dump(records, forward["fault"], jobs, caps, stats,
        dict(seed=seed, k=k, note="dose-response fixed grid point, no validation re-selection",
             repair="reference-free known-mask zero-step plan switch; identical fault X",
             free_steps=300))
    for r in dump["records"]:
        r["forward"] = {n: cc[r["image"]] for n, cc in forward.items()}
    atomic_json(folder / "inversion_protocol_n1000_s0.json", dump)

    after = model_digest(models)
    assert before == after
    assert all(not p.requires_grad and p.grad is None for m in models.values() for p in m.parameters())
    atomic_json(folder / "complete.json", dict(state_sha256_before=before, state_sha256_after=after,
                all_parameters_frozen=True, gradients_none=True,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), gpu_after=gpu_snapshot()))
    print(f"k{k} seed{seed} done", flush=True)


def summarize_k(k):
    dumps = [read_json(OUT / f"k{k}/seed{s}/inversion_protocol_n1000_s0.json") for s in SEEDS]
    per_seed = {}
    for seed, dump in zip(SEEDS, dumps):
        per_seed[str(seed)] = {}
        for variant in ("free", "fault", "repair", "uniform", "intact"):
            rows = [image_scores(r, variant)[0] for r in dump["records"]]
            per_seed[str(seed)][variant] = {key: np.array([r[key] for r in rows]).tolist() for key in
                ("heldout",)}
    pooled = {}
    for variant in ("free", "fault", "repair", "uniform", "intact"):
        arr = np.mean([per_seed[str(s)][variant]["heldout"] for s in SEEDS], axis=0)
        pooled[variant] = compact_pair(arr[:, 0], arr[:, 1])
    return dict(k=k, pooled=pooled, per_seed_image_scores=per_seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--seed", type=int, choices=SEEDS)
    ap.add_argument("--k", type=int, choices=NEW_KS)
    ap.add_argument("--gpu", type=int, choices=(0, 1))
    args = ap.parse_args()
    if args.worker:
        run_seed_k(args.seed, args.k, args.gpu)
        return

    OUT.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    uuid, _ = check_gpu(1)
    for k in NEW_KS:
        for seed in SEEDS:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, PYTHONDONTWRITEBYTECODE="1",
                       HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       ATTN_ENCODER="dinov2", CUBLAS_WORKSPACE_CONFIG=":4096:8")
            log = OUT / f"k{k}_seed{seed}.log"
            with log.open("x") as stream:
                subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--worker",
                                "--seed", str(seed), "--k", str(k), "--gpu", "1"],
                               env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
            print(f"finished k{k} seed{seed} at {time.time()-t0:.1f}s", flush=True)
    summary = {str(k): summarize_k(k) for k in NEW_KS}
    summary["total_sec"] = time.time() - t0
    atomic_json(OUT / "results.json", summary)
    print("Complete:", OUT / "results.json", "total_sec", summary["total_sec"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        if OUT.exists():
            atomic_json(OUT / f"failure_pid{os.getpid()}.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        raise
