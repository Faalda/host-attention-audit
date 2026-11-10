"""HOST sensitivity on the real frozen small captioner; no model training.

Exclusive-create outputs, physical GPU0 only, local saved inputs only.
The saved trained oracle is a reference, NEVER a degraded-init optimization.
"""
import os

if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
    raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0; no other GPU is permitted")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

import run_rev_den_common as common
from run_rev_denoiser_scaling import recovery
from run_rev_eval_metrics import r1, lr1, boot_ci
from run_rev_inversion_protocol import dec_generate, make_plan, plan_stats


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    with open(path, "x") as f:
        json.dump(data, f, indent=2, allow_nan=False)


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def evaluate(jobs, captions):
    per_job, summary, comparisons = {}, {}, {}
    for name, caps in captions.items():
        assert len(caps) == len(jobs)
        per_job[name] = []
        for job, cap in zip(jobs, caps):
            heldout = [r for k, r in enumerate(job["refs"]) if k != job["slot"]] if job["kind"] == "true" else job["refs"]
            per_job[name].append({
                "target_r1": r1(cap, [job["target"]]),
                "heldout_r1": r1(cap, heldout),
                "grounding_r1": r1(cap, job["refs"]),
                "legacy_grounding_r1": lr1(cap, job["refs"]),
                "empty": float(not cap.strip()),
            })
    for kind in ("true", "mismatch"):
        groups = [[i for i, j in enumerate(jobs) if j["image"] == image and j["kind"] == kind]
                  for image in range(200)]
        means = {name: {key: np.array([np.mean([rows[i][key] for i in group]) for group in groups])
                        for key in rows[0]} for name, rows in per_job.items()}
        summary[kind] = {name: {key: float(v.mean()) for key, v in scores.items()} for name, scores in means.items()}
        comparisons[kind] = {}
        pairs = [(name, "degraded_forward") for name in captions if name != "degraded_forward"]
        if "degraded_init_free300" in captions:
            pairs += [("degraded_init_free300", name) for name in ("trained_forward", "uniform_forward", "saved_trained_oracle_reference")]
        pairs += [("uniform_forward", "trained_forward")]
        for a, b in pairs:
            stats = {}
            for key in means[a]:
                delta = means[a][key] - means[b][key]
                m, lo, hi = boot_ci(delta, n=10000, seed=0)
                stats[key] = {"delta": m, "ci95": [lo, hi], "fraction_images_positive": float((delta > 0).mean()),
                              "fraction_images_negative": float((delta < 0).mean())}
            comparisons[kind][a + "_minus_" + b] = stats
    return {"summary": summary, "paired_image_bootstrap": comparisons, "per_job": per_job}


def optimize(dec, X, A, target):
    # Independent parameter rows and fresh Adam state for EVERY job, no warm start.
    theta = torch.nn.Parameter(A.clamp(min=1e-8).log().clone())
    opt = torch.optim.Adam([theta], lr=0.05)
    valid = target[:, 1:] != 0
    trajectory, initial_ce = [], None
    max_gradient = 0.0
    for step in range(301):
        plan = make_plan(theta, "free")
        pred = dec(X, target[:, :-1], plan)
        loss = F.cross_entropy(pred[valid], target[:, 1:][valid])
        assert torch.isfinite(loss), f"Nonfinite inversion loss at step {step}"
        if step in (0, 1, 10, 50, 100, 200, 300):
            trajectory.append({"step": step, "token_mean_ce": loss.item()})
        if step in (0, 300):
            with torch.no_grad():
                ce = F.cross_entropy(pred.transpose(1, 2), target[:, 1:], reduction="none")
                ce = (ce * valid).sum(1) / valid.sum(1)
                if step == 0:
                    initial_ce = ce.cpu().tolist()
        if step == 300:
            break
        opt.zero_grad()
        loss.backward()
        assert theta.grad is not None and torch.isfinite(theta.grad).all()
        max_gradient = max(max_gradient, theta.grad.abs().max().item())
        opt.step()
    with torch.no_grad():
        tokens = dec_generate(dec, X, plan)
        stats = plan_stats(plan, A, valid.sum(1).tolist())
        for i, st in enumerate(stats):
            st["js_to_degraded_init"] = st.pop("js_to_denoised")
            st.update(initial_target_ce=initial_ce[i], final_target_ce=ce[i].item())
    return plan.detach().cpu(), theta.detach().cpu(), tokens.cpu(), stats, trajectory, max_gradient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="host_predictor_failure_frac010_n200_s0_audit")
    ap.add_argument("--batch", type=int, default=100)
    args = ap.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    root = Path(__file__).resolve().parent
    out = Path(common.OUT_DIR) / args.output
    out.mkdir(exist_ok=False)
    source_path = Path(common.OUT_DIR) / "inversion_protocol_n1000_s0.json"
    with source_path.open() as f:
        source = json.load(f)
    assert source["n_images"] == 1000 and source["n_steps"] == 300 and source["lr"] == 0.05
    records = source["records"][:200]
    assert [r["image"] for r in records] == list(range(200))
    jobs = []
    for kind in ("true", "mismatch"):
        for rec in records:
            assert len(rec["refs"]) == 5
            donor = source["records"][rec["mismatch_source"]]
            assert donor["image"] != rec["image"] and rec["mismatch_target"] in donor["refs"]
            for slot in ([0, 1] if kind == "true" else [-1]):
                jobs.append({"job": len(jobs), "image": rec["image"], "feature_index": common.TEST_START + rec["image"],
                             "kind": kind, "slot": slot, "refs": rec["refs"],
                             "target": rec["refs"][slot] if kind == "true" else rec["mismatch_target"],
                             "mismatch_source": rec["mismatch_source"]})
    config = {
        "cohort": "first 200 original 1000 test records; unchanged saved refs, slots 0/1, original 1000-image shuffle",
        "n_images": 200, "n_jobs": 600, "steps": 300, "lr": 0.05, "batch": args.batch, "seed": 0,
        "generation": "run_rev_denoiser_scaling.py: maskgit_cfg_vec K=4 CFG=2.0; softmax(10*A); native greedy max40",
        "inversion": "free plan; fresh log(clamp(degraded_plan,1e-8)) and Adam state per job; 40x256 variables",
        "loss": "native decoder CE, batch mean over nonpadding target tokens; Adam defaults",
        "selection": "fixed final step300, no best-of or floor, no reference-based output selection",
        "evaluation": "standard rouge_score R1 F1, no stemmer, repository norm; single target, max leave-one-out, max own5",
        "uncertainty": "paired 10000 image bootstrap seed0; average slots 0/1 within image before resampling",
        "forward_access": "image features and saved weights/plans only, no caption reference input",
        "oracle_reference": "unchanged saved trained-init free300 captions; comparison only, NOT new degraded-init optimization",
        "model_training": False, "downloads": False, "cuda_visible_devices": "0", "tf32": False,
        "on_manifold": "not run (optional)",
        "interpretation": "Checkpoint is the 10%-of-training-data, n_mp=2 denoiser from the scaling sweep (a real, moderately degraded predictor, not a capacity-crippled or corrupted one). If positive: demonstrates HOST detects genuine image-predictable headroom when the forward predictor is measurably worse than the deployed one; uniform rescue is diagnostic.",
    }
    write_json(out / "preregistered_jobs.json", {"config": config, "jobs": jobs})
    checkpoint_paths = {
        "projector": Path(common.SAVE_DIR) / "model_proj.pt",
        "free_decoder_recovery_only": Path(common.SAVE_DIR) / "model_free.pt",
        "native_guided_decoder": Path(common.SAVE_DIR) / "model_guided.pt",
        "trained_predictor_seed42_nmp2": Path(common.SAVE_DIR) / "model_denoiser.pt",
        "failed_predictor_seed0_frac010_nmp2": Path(common.OUT_DIR) / "rev_ckpt/scaling_frac0.10_nmp2.pt",
    }
    paths = list(checkpoint_paths.values()) + [source_path, Path(common.CACHE_PATH), Path(common.SAVE_DIR) / "plans_test.pt",
        Path(common.SAVE_DIR) / "tok_vocab.json", Path(common.CAPS_CACHE), Path(common.OUT_DIR) / "denoiser_scaling.json"]
    paths += [root / n for n in (Path(__file__).name, "run_rev_denoiser_scaling.py", "run_rev_den_common.py",
        "run_rev_inversion_protocol.py", "run_rev_eval_metrics.py", "run_cap_improved.py", "run_captioning.py", "run_summarization.py", "run_sum_masked.py")]
    print("Hashing local inputs; fixed cohort/jobs saved", flush=True)
    hashes = {str(p.relative_to(root)): sha256(p) for p in paths}
    with open(out / Path(__file__).name, "xb") as f:
        f.write(Path(__file__).read_bytes())
    write_json(out / "sources.json", {"sha256": hashes,
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "rouge-score")},
        "gpu": subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"], text=True).strip()})
    tok = common.load_tok()
    proj, free, dec = common.load_models(tok)
    trained = common.load_base_denoiser()
    failed = common.CFGDenoiser(common.D_MODEL, common.N_PATCHES, common.L_CAP, 2).to(common.DEVICE)
    models = dict(zip(checkpoint_paths, (proj, free, dec, trained, failed)))
    load_audit = {}
    for name, model in models.items():
        state = torch.load(checkpoint_paths[name], map_location="cpu", weights_only=True)
        loaded = model.load_state_dict(state, strict=True)
        model.eval().requires_grad_(False)
        entries = [{"checkpoint_key": k, "model_key": k, "shape": list(v.shape), "dtype": str(v.dtype),
                    "nonfinite_elements": int((~torch.isfinite(v)).sum()),
                    "nan_elements": int(torch.isnan(v).sum()), "inf_elements": int(torch.isinf(v).sum())}
                   for k, v in state.items()]
        load_audit[name] = {"path": str(checkpoint_paths[name]), "strict": True, "map_location": "cpu -> cuda:0 (physical0)",
                            "missing": loaded.missing_keys, "unexpected": loaded.unexpected_keys,
                            "parameters": sum(p.numel() for p in model.parameters()), "key_map": entries,
                            "nonfinite_elements": sum(e["nonfinite_elements"] for e in entries)}
    before = {name: state_hash(m) for name, m in models.items()}
    write_json(out / "checkpoint_load_audit.json", load_audit)
    assert all(not m.training for model in models.values() for m in model.modules())
    assert all(a["nonfinite_elements"] == 0 for name, a in load_audit.items() if not name.startswith("failed_"))
    failed_state_invalid = load_audit["failed_predictor_seed0_frac010_nmp2"]["nonfinite_elements"] > 0
    print(f"Strict checkpoint maps verified; failed predictor nonfinite={failed_state_invalid}", flush=True)
    feats = torch.load(common.CACHE_PATH, map_location="cpu", weights_only=True, mmap=True)
    with torch.no_grad():
        X = proj(feats[common.TEST_START:common.TEST_START + 200].to(common.DEVICE))
    assert torch.isfinite(X).all()
    plans, forward = {}, {}
    audits = {"frozen_gates": dec.gate_values(), "state_sha256_before": before}
    for name, den in (("degraded_forward", failed), ("trained_forward", trained)):
        original = common.maskgit_cfg(den, X[:8], n_steps=4, device=common.DEVICE)
        vectorized = common.maskgit_cfg_vec(den, X[:8], 4, common.CFG_SCALE)
        torch.testing.assert_close(original, vectorized, rtol=0, atol=0, equal_nan=True)
        audits[name + "_maskgit_original_equal_including_nan"] = True
        audits[name + "_raw_maskgit_nonfinite_elements_first8"] = int((~torch.isfinite(vectorized)).sum())
        plans[name] = common.gen_plans(lambda xb, sl: common.maskgit_cfg_vec(den, xb, 4, common.CFG_SCALE), X)
    plans["uniform_forward"] = torch.full_like(plans["degraded_forward"], 1.0 / common.N_PATCHES)
    assert common.CFG_SCALE == 2.0
    for name, plan in plans.items():
        audits[name + "_plan_nonfinite_elements"] = int((~torch.isfinite(plan)).sum())
        if name != "degraded_forward" or not failed_state_invalid:
            assert torch.isfinite(plan).all() and torch.allclose(plan.sum(-1), torch.ones_like(plan.sum(-1)))
        forward[name] = common.caption(dec, X, plan, tok)
        protocol_caps = [tok.dec(t) for t in dec_generate(dec, X, plan).cpu().tolist()]
        assert protocol_caps == forward[name], "Native vs protocol EOS handling changed captions"
    saved_plans = torch.load(Path(common.SAVE_DIR) / "plans_test.pt", map_location="cpu", weights_only=True)[:200]
    audits["trained_plan_vs_saved_max_abs_error"] = (plans["trained_forward"].cpu() - saved_plans).abs().max().item()
    audits["trained_caption_vs_saved_disagreements"] = sum(a != r["baseline"] for a, r in zip(forward["trained_forward"], records))
    with open(Path(common.OUT_DIR) / "denoiser_scaling.json") as f:
        scaling = json.load(f)
    old = scaling["runs"]["frac0.10_nmp2"]
    audits["failed_caption_vs_scaling_saved_disagreements"] = sum(a != b for a, b in zip(forward["degraded_forward"], old["captions"][:200]))
    audits["original_1k_failed_recovery"] = old["recovery"]
    assert audits["failed_caption_vs_scaling_saved_disagreements"] == 0
    with open(common.CAPS_CACHE) as f:
        all_refs = json.load(f)
    assert all_refs[common.TEST_START:common.TEST_START + 200] == [r["refs"] for r in records]
    # Teacher-forced plans are used ONLY for recovery diagnostics, never generation.
    A0, pair_img, ids = common.build_A0(free, X, [r["refs"] for r in records], list(range(200)), tok)
    content = torch.cat([ids[:, :-1] != 0, torch.zeros(len(ids), 1, dtype=torch.bool, device=common.DEVICE)], 1)
    audits["cohort_predictor_recovery"] = {name: recovery(den, A0, X[pair_img], content)
                                           for name, den in (("degraded", failed), ("trained", trained))}
    invalid_init = not bool(torch.isfinite(plans["degraded_forward"]).all())
    audits["failed_recovery_valid"] = not failed_state_invalid
    audits["failed_recovery_caveat"] = ("INVALID: argmax/topk on nonfinite logits is not patch recovery" if failed_state_invalid else None)
    with torch.no_grad():
        logits = failed(X, torch.full_like(plans["degraded_forward"], 1 / common.N_PATCHES),
                        torch.ones(200, common.L_CAP, device=common.DEVICE))
        audits["failed_fully_masked_logits_nonfinite_elements"] = int((~torch.isfinite(logits)).sum())
        audits["failed_fully_masked_argmax_unique_indices"] = logits.argmax(-1).unique().cpu().tolist()
        audits["teacher_plan_argmax_zero_fraction_all_rows"] = (A0.argmax(-1) == 0).float().mean().item()
        audits["teacher_plan_argmax_zero_fraction_content_rows"] = (A0.argmax(-1)[content] == 0).float().mean().item()
    forward_stats = {}
    for name, p in plans.items():
        finite = bool(torch.isfinite(p).all())
        forward_stats[name] = {
            "valid_probability_plan": finite,
            "entropy_mean_all40": float(-(p * p.log()).sum(-1).mean()) if finite else None,
            "unique_argmax_patches": int(p.argmax(-1).unique().numel()) if finite else None,
            "unique_argmax_plan_rows": int(p.argmax(-1).unique(dim=0).shape[0]) if finite else None,
            "stats_to_degraded_first39": plan_stats(p, plans["degraded_forward"], [39] * 200) if finite and not invalid_init else None,
        }
    captions = {name: [caps[j["image"]] for j in jobs] for name, caps in forward.items()}
    captions["saved_trained_oracle_reference"] = [records[j["image"]]["variants"]["free"]["true"][j["slot"]]
        if j["kind"] == "true" else records[j["image"]]["variants"]["free"]["mismatch"] for j in jobs]
    forward_metrics = evaluate(jobs, captions)
    write_json(out / "forward_results.json", {"config": config, "audit": audits, "forward_plan_stats": forward_stats,
                                            "captions": captions, "metrics": forward_metrics})
    with open(out / "forward_plans.pt", "xb") as f:
        torch.save({name: p.cpu() for name, p in plans.items()}, f)
    print("Forward results: " + json.dumps(forward_metrics["summary"]["true"]), flush=True)
    del A0, pair_img, ids, content
    if not invalid_init:
        captions["degraded_init_free300"] = []
    stats, batches = [], []
    rejected_jobs = []
    for start in range(0, len(jobs), args.batch):
        tick = time.perf_counter()
        chunk = jobs[start:start + args.batch]
        idx = [j["image"] for j in chunk]
        A = plans["degraded_forward"][idx].clone()
        target = torch.tensor([tok.enc(j["target"], common.L_CAP, bos=True, eos=True) for j in chunk], device=common.DEVICE)
        torch.testing.assert_close(A, plans["degraded_forward"][idx], rtol=0, atol=0, equal_nan=True)
        if invalid_init:
            # Check each fresh initial parameter and loss; never sanitize or run NaN Adam.
            theta = A.clamp(min=1e-8).log().clone()
            with torch.no_grad():
                pred = dec(X[idx], target[:, :-1], make_plan(theta, "free"))
                valid = target[:, 1:] != 0
                ce = F.cross_entropy(pred.transpose(1, 2), target[:, 1:], reduction="none")
                losses = (ce * valid).sum(1) / valid.sum(1)
            assert not torch.isfinite(losses).any()
            rejected_jobs.extend({"job": j["job"], "fresh_init": True,
                                  "nonfinite_initial_parameters": int((~torch.isfinite(theta[i])).sum()),
                                  "initial_target_ce": None, "initial_target_ce_is_nan": bool(torch.isnan(losses[i])),
                                  "optimizer_steps_executed": 0,
                                  "status": "rejected_nonfinite_initial_plan_and_loss"} for i, j in enumerate(chunk))
            continue
        p, theta, tokens, st, trajectory, gradient = optimize(dec, X[idx], A, target)
        assert before == {name: state_hash(m) for name, m in models.items()}
        caps = [tok.dec(t) for t in tokens.tolist()]
        captions["degraded_init_free300"].extend(caps)
        stats.extend(st)
        filename = f"inversion_jobs{start:04d}_{start + len(chunk):04d}.pt"
        with open(out / filename, "xb") as f:
            torch.save({"job_ids": [j["job"] for j in chunk], "initial_plan": A.cpu(), "final_plan": p,
                        "logits": theta, "target_tokens": target.cpu(), "tokens": tokens}, f)
        batch = {"path": filename, "sha256": sha256(out / filename), "trajectory": trajectory,
                 "max_abs_gradient": gradient, "seconds": time.perf_counter() - tick}
        batches.append(batch)
        write_json(out / filename.replace(".pt", ".json"), {**batch, "captions": caps, "stats": st})
        print(f"Degraded-init free300: {start + len(chunk)}/600; {batch['seconds']:.1f}s batch", flush=True)
    audits["state_sha256_after"] = {name: state_hash(m) for name, m in models.items()}
    assert before == audits["state_sha256_after"]
    assert all(p.grad is None and not p.requires_grad for m in models.values() for p in m.parameters())
    audits.update(all_weights_and_buffers_unchanged=True, all_model_gradients_none=True,
                  frozen_gates_after=dec.gate_values(), peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    metrics = evaluate(jobs, captions)
    positive = not invalid_init and metrics["paired_image_bootstrap"]["true"]["degraded_init_free300_minus_degraded_forward"]["heldout_r1"]["ci95"][0] > 0
    conclusion = ("Positive held-out recovery: severe predictor-failure stress test only. Not evidence informative learned routing is required; not tiny-data calibration."
                  if positive else "Reject as positive HOST sensitivity candidate: no statistically positive held-out recovery from fresh degraded initialization.")
    if invalid_init:
        conclusion = "REJECT: failed predictor has nonfinite weights and emits nonfinite plans. All 600 fresh degraded initializations have NaN loss; 300-step inversion is undefined. Empty forward captions and apparent recovery are numerical artifacts, not a valid poor predictor. Trained/uniform forward rescue does not establish positive inversion sensitivity."
    result = {"config": config, "conclusion": conclusion, "sources": hashes, "checkpoint_identification": load_audit,
              "validation": audits, "jobs": jobs, "captions": captions, "forward_plan_stats": forward_stats,
              "inversion_plan_stats": stats, "rejected_inversion_jobs": rejected_jobs, "batches": batches, "metrics": metrics,
              "total_seconds": time.perf_counter() - started}
    write_json(out / "results.json", result)
    manifest = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    write_json(out / "artifact_sha256.json", manifest)
    print(json.dumps({"conclusion": conclusion, "summary": metrics["summary"], "total_seconds": result["total_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
