"""Isolated, offline constructed feature-space routing control.

Run with python -B run_visual_headroom_control.py. Only the two new scripts and
results/visual_headroom_control are owned. A failed validation gate stops test.
The local plan is not an externally preregistered experiment.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for clean_pipeline./analysis. imports
from clean_pipeline.run_clean_evaluation import (atomic_json, read_json, sha256, load_models,
                                  model_digest, optimize, validate_plans,
                                  validate_source, protocol_jobs, protocol_dump,
                                  target_audit, validate_training_manifest, MODEL_FILES)
from analysis.rescore_rouge import score, paired, image_scores

ROOT = Path(__file__).resolve().parent.parent  # repo root (file now lives one level down)
OUT = ROOT / "results/visual_headroom_control"
MODELS = ROOT / "results/clean_small_stablekl"
FEATURES = ROOT / "results/dinov2_patch_features_15k.pt"
CAPS = ROOT / "results/revision/coco_caps_15k.json"
SOURCE = ROOT / "results/revision/inversion_protocol_n1000_s0.json"
SEEDS = (42, 7, 13)
KS = (4, 16, 64)
HIDDEN_STEPS = (0, 1, 2, 3, 5, 10, 20, 50, 100, 300)
MASK_SEED = 20261101
EPS = 2e-6
BATCH = 250
FORWARD_BATCH = 32


def plan_config():
    return dict(seeds=list(SEEDS), candidates=list(KS), mask_seed=MASK_SEED,
                mask_mixing="numpy SeedSequence([20261101, global_feature_row]); permutation(256); nested prefix",
                train_mean_rows=[0, 13000], validation_rows=[13000, 13200],
                matching_rows=[13000, 13150], test_rows=[14000, 15000],
                ranges_half_open=True, n_patches=256, n_rows=40, eps=EPS,
                defect="retain K projected real patch vectors; replace other patches by train-only global projected mean",
                fault="eps on anchors; remaining mass uniform on background; identical 40 rows",
                repair="eps on background; remaining mass uniform on anchors; identical 40 rows",
                eligibility="pooled seed-average image repair-fault max-own5 R1 mean >= .02 and bootstrap lower > 0; repair own5 minus cyclic-next-image refs lower > 0; all finite; no empty captions",
                selection="largest eligible pooled validation repair gain; ties in candidate order",
                provenance="own versus fixed next-validation-image references; report intact-native comparison, not an intact-retention threshold",
                metric="standard rouge_score R1 F1; repository norm; max reference, no floor",
                bootstrap=dict(resamples=10000, seed=0, unit="image after averaging fixed three seeds"),
                matching=dict(images=150, jobs=750, true_slots=list(range(5)), hidden_steps=list(HIDDEN_STEPS),
                              selection="global closest mean target gain relative fault to free300 per seed; lowest-step ties",
                              test_selection=False),
                inversion=dict(steps=300, lr=.05, batch=BATCH, params=10240,
                               free_init="log(clamp(Pfault,1e-8))", hidden_init="zero input-state bias; fixed Pfault",
                               loss="native CE mean over nonpadding batch tokens; default Adam"),
                forward_batch=FORWARD_BATCH, all_five_refs=True, jobs_dropped_for_oov=0,
                gpu=dict(physical_ids=[0, 1], memory_fraction=.35, minimum_free_mib=24000,
                         shared=True, kill_or_modify_other_jobs=False),
                onehot="not run (optional)", training=False, downloads=False,
                external_preregistration=False,
                limitations=["Constructed feature-space visual-routing sanity control, not ordinary COCO predictor deficit or actual photo corruption.",
                             "Known-mask repair is reference-free and zero-step, not learned repair or inversion.",
                             "Validation is disjoint from test but was also used for model selection and K selection; matching reuses its first150 images.",
                             "Image intervals condition on three trained seeds, not training-population uncertainty; no formal upper bound."])


def anchor_masks(indices, k):
    if k not in KS:
        raise ValueError("K outside fixed grid")
    masks = np.zeros((len(indices), 256), dtype=bool)
    for i, index in enumerate(indices):
        rng = np.random.default_rng(np.random.SeedSequence([MASK_SEED, int(index)]))
        masks[i, rng.permutation(256)[:k]] = True
    return torch.from_numpy(masks)


def routing_plans(mask):
    k = mask.sum(-1, keepdim=True)
    if not ((k > 0) & (k < 256)).all():
        raise ValueError("Need nonempty anchor and background")
    fault = torch.where(mask, EPS, (1 - k * EPS) / (256 - k))
    repair = torch.where(mask, (1 - (256 - k) * EPS) / k, EPS)
    result = {}
    for name, values in (("fault", fault), ("repair", repair), ("uniform", torch.full_like(fault, 1 / 256))):
        p = values.float()[:, None].expand(-1, 40, -1).contiguous()
        p = p / p.sum(-1, keepdim=True)
        validate_plans(p)
        assert (p > 1e-6).all(), "Avoid zero/clamped-gradient fault initialization"
        result[name] = p
    return result


def fault_features(clean, mu, mask):
    assert clean.shape[1:] == (256, 256) and mu.shape == (256,)
    x = torch.where(mask[..., None], clean, mu[None, None])
    assert torch.isfinite(x).all()
    assert torch.equal(x[mask], clean[mask])
    assert torch.equal(x[~mask], mu.expand_as(x)[~mask])
    return x


def gpu_snapshot():
    def query(kind, columns):
        return subprocess.check_output(["nvidia-smi", f"--query-{kind}={columns}",
                                        "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    return {"time": time.time(), "gpus": query("gpu", "index,uuid,memory.total,memory.free"),
            "processes": query("compute-apps", "gpu_uuid,pid,process_name,used_gpu_memory")}


def compact_pair(a, b):
    return {k: v for k, v in paired(a, b).items() if k != "per_image"}


def forward_scores(captions, refs):
    return {name: [max(score(cap, ref) for ref in row) for cap, row in zip(caps, refs)]
            for name, caps in captions.items()}


def choose_k(validations):
    candidates = {}
    for k in KS:
        rows = [v["candidates"][str(k)] for v in validations]
        repair = np.mean([r["scores"]["repair"] for r in rows], axis=0)
        fault = np.mean([r["scores"]["fault"] for r in rows], axis=0)
        foreign = np.mean([r["repair_cyclic_reference_scores"] for r in rows], axis=0)
        gain, info = compact_pair(repair, fault), compact_pair(repair, foreign)
        finite = all(r["finite"] for r in rows)
        nonempty = all(all(v == 0 for v in r["empty_counts"].values()) for r in rows)
        eligible = gain["delta"] >= .02 and gain["ci95"][0] > 0 and info["ci95"][0] > 0 and finite and nonempty
        candidates[str(k)] = dict(repair_minus_fault=gain, repair_own_minus_cyclic=info,
                                  finite=finite, nonempty=nonempty, eligible=bool(eligible))
    eligible = [k for k in KS if candidates[str(k)]["eligible"]]
    chosen = max(eligible, key=lambda k: candidates[str(k)]["repair_minus_fault"]["delta"]) if eligible else None
    return dict(chosen_k=chosen, candidates=candidates, status="passed" if chosen is not None else "failed",
                test_allowed=chosen is not None, selection_uses_test=False)


@torch.no_grad()
def generate(tok, dec, x, p):
    from inversion.run_rev_inversion_protocol import dec_forward, dec_generate
    validate_plans(p)
    tokens = dec_generate(dec, x, p)
    assert torch.isfinite(dec_forward(dec, x, tokens[:, :-1], p)).all()
    return [tok.dec(t) for t in tokens.cpu().tolist()]


@torch.no_grad()
def build_features(models, feats, folder, indices, mu, k):
    clean = torch.cat([models["proj"](feats[chunk].cuda()).cpu()
                       for chunk in torch.tensor(indices).split(FORWARD_BATCH)])
    mask = anchor_masks(indices, k)
    x = fault_features(clean, mu, mask)
    for name, tensor in (("fault_features", x), ("anchor_mask", mask)):
        with (folder / (name + ".pt")).open("xb") as stream:
            torch.save(tensor, stream)
    return clean, torch.load(folder / "fault_features.pt", weights_only=True, mmap=True), mask


@torch.no_grad()
def forward_all(tok, models, clean, x, mask, native=None):
    from clean_pipeline.run_cap_improved import maskgit_cfg
    captions = {name: [] for name in ("fault", "repair", "uniform", "intact")}
    for start in range(0, len(x), FORWARD_BATCH):
        sl = slice(start, start + FORWARD_BATCH)
        xc, xf = clean[sl].cuda(), x[sl].cuda()
        plans = routing_plans(mask[sl].cuda())
        if native is None:
            raw = maskgit_cfg(models["denoiser"], xc, n_steps=4, cfg_scale=2., device="cuda")
            assert torch.isfinite(raw).all()
            a = (10 * raw).softmax(-1)
        else:
            a = native[sl].cuda()
        for name, p in plans.items():
            captions[name] += generate(tok, models["guided"], xf, p)
        captions["intact"] += generate(tok, models["guided"], xc, a)
    return captions


def run_inversions(folder, jobs, tok, models, x, mask, checkpoints):
    caps = {"free": [], "hidden": []}
    stats = {v: [] for v in caps}
    trajectory = {s: [] for s in checkpoints}
    encoded = [target_audit(tok, j) for j in jobs]
    assert len(encoded) == len(jobs)
    atomic_json(folder / "target_encoding.json", dict(jobs=[{**j, **a} for j, a in zip(jobs, encoded)],
                                                      jobs_dropped_for_oov=0))
    for variant in caps:
        for start in range(0, len(jobs), BATCH):
            tick = time.monotonic()
            chunk = jobs[start:start + BATCH]
            idx = [j["image"] for j in chunk]
            xf = x[idx].cuda()
            a = routing_plans(mask[idx].cuda())["fault"]
            target = torch.tensor([e["target_tokens"] for e in encoded[start:start + BATCH]], device="cuda")
            grid = checkpoints if variant == "hidden" else ()
            tokens, st, traj = optimize(models["guided"], xf, a, target, variant,
                                       steps=300, lr=.05, checkpoints=grid)
            cc = [tok.dec(t) for t in tokens]
            caps[variant] += cc
            for row in st:
                row["js_to_fault_init"] = row.pop("js_to_denoised")
            stats[variant] += st
            for step, values in traj.items():
                trajectory[step] += [tok.dec(t) for t in values]
            atomic_json(folder / f"{variant}_{start:04d}.json", dict(start=start, captions=cc, stats=st,
                        trajectory={s: [tok.dec(t) for t in values] for s, values in traj.items()},
                        seconds=time.monotonic() - tick, finite_gradients=True))
            print(f"{folder.name} {variant} {start + len(chunk)}/{len(jobs)} {time.monotonic()-tick:.1f}s", flush=True)
    return caps, stats, trajectory


def matching_choice(jobs, base, free, hidden):
    baseline = np.array([score(base[j["image"]], j["target"]) for j in jobs])
    fg = np.array([score(c, j["target"]) for c, j in zip(free, jobs)]) - baseline
    gains = {s: np.array([score(c, j["target"]) for c, j in zip(caps, jobs)]) - baseline
             for s, caps in hidden.items()}
    selected = min(HIDDEN_STEPS, key=lambda s: abs(gains[s].mean() - fg.mean()))
    residual = gains[selected] - fg
    return dict(step=selected, free_mean_gain=float(fg.mean()),
                sweep={s: float(g.mean()) for s, g in gains.items()},
                residual_mean=float(residual.mean()), residual_mae=float(abs(residual).mean()),
                residual_positive=int((residual > 0).sum()), residual_negative=int((residual < 0).sum()),
                residual_zero=int((residual == 0).sum()), n_jobs=len(jobs), test_selection=False)


def worker(stage, seed, gpu):
    folder = OUT / f"seed{seed}"
    dest = folder / stage
    dest.mkdir(exist_ok=False)
    snapshot = gpu_snapshot()
    row = next(r.split(", ") for r in snapshot["gpus"] if r.split(", ")[0] == str(gpu))
    if float(row[3]) < 24000:
        raise RuntimeError("Shared GPU has less than 24000 MiB free; no allocation")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == row[1]
    torch.cuda.set_per_process_memory_fraction(.35, 0)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    atomic_json(dest / "gpu_before.json", dict(**snapshot, physical_gpu=gpu, uuid=row[1],
                                              own_pid=os.getpid(), memory_cap=.35))
    model_dir = MODELS / f"seed{seed}"
    tok, models = load_models(model_dir, "cuda")
    before = model_digest(models)
    assert all(not m.training for model in models.values() for m in model.modules())
    feats = torch.load(FEATURES, map_location="cpu", weights_only=True, mmap=True)
    assert feats.shape == (15000, 256, 768)
    if stage == "validation":
        with torch.no_grad():
            total = torch.zeros(256, dtype=torch.float64, device="cuda")
            for lo in range(0, 13000, 64):
                projected = models["proj"](feats[lo:min(lo + 64, 13000)].cuda())
                assert torch.isfinite(projected).all()
                total += projected.double().sum((0, 1))
            mu = (total / (13000 * 256)).float().cpu()
        with (folder / "train_mean.pt").open("xb") as stream:
            torch.save(mu, stream)
        refs = read_json(CAPS)[13000:13200]
        assert len(refs) == 200 and all(len(r) == 5 for r in refs)
        result = dict(seed=seed, feature_indices=list(range(13000, 13200)), refs=refs, candidates={})
        for k in KS:
            sub = dest / f"k{k}"
            sub.mkdir()
            clean, x, mask = build_features(models, feats, sub, list(range(13000, 13200)), mu, k)
            caps = forward_all(tok, models, clean, x, mask)
            scores = forward_scores(caps, refs)
            foreign = [max(score(c, r) for r in refs[(i + 1) % 200]) for i, c in enumerate(caps["repair"])]
            result["candidates"][str(k)] = dict(captions=caps, scores=scores, repair_cyclic_reference_scores=foreign,
                empty_counts={n: sum(not c.strip() for c in cc) for n, cc in caps.items()}, finite=True,
                repair_minus_intact=compact_pair(scores["repair"], scores["intact"]),
                intact_minus_fault=compact_pair(scores["intact"], scores["fault"]),
                feature_sha256=sha256(sub / "fault_features.pt"), mask_sha256=sha256(sub / "anchor_mask.pt"))
            print(f"seed{seed} K{k}: " + json.dumps({n: float(np.mean(s)) for n, s in scores.items()}), flush=True)
        atomic_json(dest / "results.json", result)
    else:
        lock = read_json(OUT / "validation_lock.json")
        assert lock["status"] == "passed" and lock["test_allowed"]
        k = lock["chosen_k"]
        if stage == "calibration":
            val = read_json(folder / "validation/results.json")
            sub = folder / f"validation/k{k}"
            x = torch.load(sub / "fault_features.pt", weights_only=True, mmap=True)[:150]
            mask = torch.load(sub / "anchor_mask.pt", weights_only=True)[:150]
            refs = val["refs"][:150]
            jobs = [dict(image=i, feature_index=13000 + i, kind="true", k=s, target=r)
                    for i, rr in enumerate(refs) for s, r in enumerate(rr)]
            assert len(jobs) == 750
            caps, stats, traj = run_inversions(dest, jobs, tok, models, x, mask, HIDDEN_STEPS)
            base = val["candidates"][str(k)]["captions"]["fault"][:150]
            assert traj[0] == [base[j["image"]] for j in jobs]
            choice = matching_choice(jobs, base, caps["free"], traj)
            atomic_json(dest / "results.json", dict(choice=choice, jobs=jobs, baselines=base,
                        captions=caps, stats=stats, hidden_trajectory=traj))
        elif stage == "test":
            test_lock = read_json(OUT / "test_lock.json")
            assert test_lock["validation_lock_sha256"] == sha256(OUT / "validation_lock.json")
            for s in SEEDS:
                assert test_lock["calibration_hashes"][str(s)] == sha256(OUT / f"seed{s}/calibration/results.json")
            step = test_lock["hidden_steps"][str(seed)]
            records = validate_source(read_json(SOURCE))
            assert [r["refs"] for r in records] == read_json(CAPS)[14000:15000] == read_json(model_dir / "caps_test.json")
            jobs = protocol_jobs(records)
            assert len(jobs) == 6000
            mu = torch.load(folder / "train_mean.pt", weights_only=True)
            clean, x, mask = build_features(models, feats, dest, list(range(14000, 15000)), mu, k)
            native = torch.load(model_dir / "plans_test.pt", weights_only=True, mmap=True)
            forward = forward_all(tok, models, clean, x, mask, native)
            atomic_json(dest / "forward.json", dict(captions=forward,
                        scores=forward_scores(forward, [r["refs"] for r in records]),
                        export_intact_mismatches=[i for i, (a, b) in enumerate(zip(forward["intact"],
                            read_json(model_dir / "captions_test.json")["predicted"])) if a != b]))
            caps, stats, traj = run_inversions(dest, jobs, tok, models, x, mask, (0, step) if step else (0,))
            assert traj[0] == [forward["fault"][j["image"]] for j in jobs]
            caps["hidden_matched"] = traj[step]
            stats["hidden_matched"] = [dict(steps=step, matched_on="true validation global mean", finite=True) for _ in jobs]
            for name, cc in forward.items():
                caps[name] = [cc[j["image"]] for j in jobs]
                stats[name] = [dict(steps=0, reference_free=True, inversion=False) for _ in jobs]
            dump = protocol_dump(records, forward["fault"], jobs, caps, stats,
                dict(seed=seed, k=k, test_lock_sha256=sha256(OUT / "test_lock.json"),
                     repair="reference-free known-mask zero-step plan switch; identical fault X",
                     hidden_matched_steps=step, hidden_steps=300, free_steps=300))
            for r in dump["records"]:
                r["forward"] = {n: cc[r["image"]] for n, cc in forward.items()}
            atomic_json(dest / "inversion_protocol_n1000_s0.json", dump)
    after = model_digest(models)
    assert before == after
    assert all(not p.requires_grad and p.grad is None for m in models.values() for p in m.parameters())
    atomic_json(dest / "complete.json", dict(state_sha256_before=before, state_sha256_after=after,
                frozen_gates=models["guided"].gate_values(), all_parameters_frozen=True, gradients_none=True,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), gpu_after=gpu_snapshot()))


def run_stage(stage):
    # One process per device; the third training seed follows seed42 on GPU0.
    def queue(gpu, seeds):
        snapshot = gpu_snapshot()
        uuid = next(r.split(", ")[1] for r in snapshot["gpus"] if r.split(", ")[0] == str(gpu))
        for seed in seeds:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, PYTHONDONTWRITEBYTECODE="1",
                       HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       ATTN_ENCODER="dinov2", CUBLAS_WORKSPACE_CONFIG=":4096:8")
            log = OUT / f"seed{seed}/{stage}.log"
            with log.open("x") as stream:
                subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--worker", stage,
                                "--seed", str(seed), "--gpu", str(gpu)], env=env, stdout=stream,
                               stderr=subprocess.STDOUT, check=True, cwd=ROOT)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(queue, 0, (42, 13)), pool.submit(queue, 1, (7,))]
        for future in futures:
            future.result()


def source_hashes():
    paths = [FEATURES, CAPS, SOURCE]
    paths += [ROOT / name for name in ("run_visual_headroom_control.py", "test_visual_headroom_control.py",
        "run_clean_evaluation.py", "run_rev_inversion_protocol.py", "run_cap_improved.py",
        "run_captioning.py", "run_summarization.py", "run_sum_masked.py", "run_rev_eval_metrics.py",
        "analysis/rescore_rouge.py")]
    for seed in SEEDS:
        paths += [MODELS / f"seed{seed}" / name for name in (*MODEL_FILES, "manifest.json", "captions_test.json")]
    return {str(p.resolve()): sha256(p) for p in paths}


def summarize_test():
    dumps = [read_json(OUT / f"seed{s}/test/inversion_protocol_n1000_s0.json") for s in SEEDS]
    per_seed, pooled, matching = {}, {}, {}
    for seed, dump in zip(SEEDS, dumps):
        per_seed[str(seed)] = {}
        for variant in dump["variants"]:
            rows = [image_scores(r, variant)[0] for r in dump["records"]]
            per_seed[str(seed)][variant] = {key: np.array([r[key] for r in rows]).tolist() for key in
                ("single_ref", "heldout", "each_inversion_all_refs_avg", "mismatch_target", "mismatch_grounding")}
        matching[str(seed)] = {}
        for kind in ("true", "mismatch"):
            residuals = []
            for r in dump["records"]:
                for slot in (range(5) if kind == "true" else [-1]):
                    target = r["refs"][slot] if kind == "true" else r["mismatch_target"]
                    def caption(v):
                        return r["variants"][v]["true"][slot] if kind == "true" else r["variants"][v]["mismatch"]
                    residuals.append(score(caption("hidden_matched"), target) - score(caption("free"), target))
            rr = np.array(residuals)
            matching[str(seed)][kind] = dict(mean=float(rr.mean()), mae=float(abs(rr).mean()), n=len(rr),
                positive=int((rr > 0).sum()), negative=int((rr < 0).sum()), zero=int((rr == 0).sum()))
    for variant in dumps[0]["variants"]:
        pooled[variant] = {}
        for key in per_seed[str(SEEDS[0])][variant]:
            arr = np.mean([per_seed[str(s)][variant][key] for s in SEEDS], axis=0)
            pooled[variant][key] = compact_pair(arr[:, 0], arr[:, 1])
    empty = {str(seed): {v: {kind: sum(not c.strip() for r in dump["records"] for c in
                (r["variants"][v]["true"] if kind == "true" else [r["variants"][v]["mismatch"]]))
                           for kind in ("true", "mismatch")} for v in dump["variants"]}
             for seed, dump in zip(SEEDS, dumps)}
    repair = pooled["repair"]["each_inversion_all_refs_avg"]
    positive = repair["delta"] >= .02 and repair["ci95"][0] > 0
    return dict(pooled=pooled, per_seed_image_scores=per_seed, matching_residuals=matching, empty_counts=empty,
                locked_test_repair_gain_criterion_passed=positive,
                free_heldout_lower_positive=pooled["free"]["heldout"]["ci95"][0] > 0,
                conclusion=("Locked test confirms constructed reference-free mask-repair headroom; inspect all three audit checks separately."
                            if positive else "Validation-selected repair did not meet the same gain criterion on locked test; no confirmed positive control."),
                interval_scope="paired images, five slots then three fixed seeds averaged; not training uncertainty")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", choices=("validation", "calibration", "test"))
    ap.add_argument("--seed", type=int, choices=SEEDS)
    ap.add_argument("--gpu", type=int, choices=(0, 1))
    args = ap.parse_args()
    if args.worker:
        worker(args.worker, args.seed, args.gpu)
        return
    OUT.mkdir(exist_ok=False)
    for seed in SEEDS:
        (OUT / f"seed{seed}").mkdir()
    atomic_json(OUT / "local_plan.json", plan_config())
    atomic_json(OUT / "runtime.json", dict(pid=os.getpid(), started=time.time(),
                versions={p: importlib.metadata.version(p) for p in ("torch", "numpy", "rouge-score")}))
    atomic_json(OUT / "gpu_initial.json", gpu_snapshot())
    hashes = source_hashes()
    atomic_json(OUT / "input_hashes.json", hashes)
    for seed in SEEDS:
        folder = MODELS / f"seed{seed}"
        manifest = read_json(folder / "manifest.json")
        validate_training_manifest(manifest, hashes, folder)
        assert manifest["seed"] == seed
        for name, path in (("results/dinov2_patch_features_15k.pt", FEATURES),
                           ("results/revision/coco_caps_15k.json", CAPS),
                           ("results/revision/inversion_protocol_n1000_s0.json", SOURCE)):
            assert manifest["hashes"][name] == hashes[str(path.resolve())]
    run_stage("validation")
    vals = [read_json(OUT / f"seed{s}/validation/results.json") for s in SEEDS]
    lock = choose_k(vals)
    lock.update(local_plan_sha256=sha256(OUT / "local_plan.json"), input_hashes_sha256=sha256(OUT / "input_hashes.json"),
                validation_hashes={s: sha256(OUT / f"seed{s}/validation/results.json") for s in SEEDS})
    atomic_json(OUT / "validation_lock.json", lock)
    print(json.dumps(lock, indent=2), flush=True)
    if lock["status"] == "failed":
        assert source_hashes() == hashes
        atomic_json(OUT / "results.json", dict(status="failed_validation", test_run=False, validation=lock,
            all_consumed_inputs_unchanged=True, conclusion="No prespecified positive control. No test scoring or tuning performed."))
        return
    run_stage("calibration")
    assert source_hashes() == hashes
    atomic_json(OUT / "test_lock.json", dict(chosen_k=lock["chosen_k"],
        validation_lock_sha256=sha256(OUT / "validation_lock.json"), input_hashes_sha256=sha256(OUT / "input_hashes.json"),
        hidden_steps={s: read_json(OUT / f"seed{s}/calibration/results.json")["choice"]["step"] for s in SEEDS},
        calibration_hashes={s: sha256(OUT / f"seed{s}/calibration/results.json") for s in SEEDS},
        test_selection=False, created=time.time()))
    run_stage("test")
    result = summarize_test()
    assert source_hashes() == hashes
    result.update(status="complete", test_run=True, validation=lock, all_consumed_inputs_unchanged=True,
                  limitations=plan_config()["limitations"])
    atomic_json(OUT / "results.json", result)
    atomic_json(OUT / "artifact_hashes.json", {str(p.relative_to(OUT)): sha256(p)
                for p in sorted(OUT.rglob("*")) if p.is_file()})
    print("Complete:", OUT / "results.json", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        if OUT.exists():
            path = OUT / f"failure_pid{os.getpid()}.json"
            atomic_json(path, dict(error=repr(exc), traceback=traceback.format_exc(), time=time.time()))
        raise
