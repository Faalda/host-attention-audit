"""Frozen native-captioner, gain-matched bounded/unbounded logit control.

Only the per-example 40x256 control is optimized. Run on physical GPU0 only.
All artifacts are exclusive-create in a new directory; no paper files change.
"""
import os

if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
    raise RuntimeError("Run with CUDA_VISIBLE_DEVICES=0 ONLY")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from contextlib import contextmanager
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

from run_rev_inversion_protocol import (
    CACHE_PATH, DEVICE, L_CAP, N_PATCHES, OUT_DIR, SAVE_DIR, TEST_START,
    dec_forward, dec_generate, make_plan,
)
from run_rev_matched_effect_control import load_models
from run_rev_eval_metrics import r1, objects, norm, boot_ci


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with open(path, "x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


@contextmanager
def additive_logits(dec, bias):
    """Exact native layer operations, with ONLY gate * shared_bias added."""
    originals = [layer.forward for layer in dec.layers]

    def forward_for(layer):
        def forward(x, mem, attn_plan, tgt_mask=None, tgt_pad=None, ret_attn=False):
            xn = layer.ln1(x)
            sa_out, _ = layer.sa(xn, xn, xn, attn_mask=tgt_mask, key_padding_mask=tgt_pad)
            x = x + layer.drop(sa_out)
            xn = layer.ln2(x)
            Q, K, V = layer.q(xn), layer.k(mem), layer.v(mem)
            logits = torch.bmm(Q, K.transpose(-1, -2)) * layer.scale
            gate = torch.sigmoid(layer.plan_gate)
            plan_bias = gate * torch.log(attn_plan.clamp(min=1e-6))
            logits = logits + plan_bias
            logits = logits + gate * bias[:, :x.shape[1], :]
            attn_w = F.softmax(logits, dim=-1)
            ctx = torch.bmm(attn_w, V)
            x = x + layer.drop(layer.o(ctx))
            x = x + layer.drop(layer.ff(layer.ln3(x)))
            return (x, attn_w) if ret_attn else x
        return forward

    try:
        for layer in dec.layers:
            layer.forward = forward_for(layer)
        yield
    finally:
        for layer, forward in zip(dec.layers, originals):
            layer.forward = forward


def model_digest(models):
    h = hashlib.sha256()
    for model in models:
        for name, value in model.state_dict().items():
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def optimize(dec, X, A, target, variant, steps, lr):
    # Same number of variables, Adam defaults, batches and token-mean loss.
    param = torch.nn.Parameter(
        torch.log(A.clamp(min=1e-8)).clone() if variant == "bounded"
        else torch.zeros_like(A)
    )
    opt = torch.optim.Adam([param], lr=lr)
    valid = target[:, 1:] != 0
    trajectory = []
    max_grad = 0.0
    context = additive_logits(dec, param) if variant == "unbounded" else None
    if context is not None:
        context.__enter__()
    try:
        for step in range(steps + 1):
            plan = make_plan(param, "free") if variant == "bounded" else A
            pred = dec_forward(dec, X, target[:, :-1], plan)
            loss = F.cross_entropy(pred[valid], target[:, 1:][valid])
            if step in {0, 1, 10, 50, 100, 200, steps}:
                trajectory.append({"step": step, "token_mean_ce": loss.item()})
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss: {variant}, step {step}")
            if step == steps:
                break
            opt.zero_grad()
            loss.backward()
            if param.grad is None or not torch.isfinite(param.grad).all():
                raise RuntimeError(f"Nonfinite or absent gradient: {variant}, step {step}")
            max_grad = max(max_grad, param.grad.abs().max().item())
            opt.step()
        with torch.no_grad():
            tokens = dec_generate(dec, X, plan)
            ce = F.cross_entropy(pred.transpose(1, 2), target[:, 1:], reduction="none")
            ce = (ce * valid).sum(1) / valid.sum(1)
            effective = torch.log(plan.clamp(min=1e-6)) if variant == "bounded" else torch.log(A.clamp(min=1e-6)) + param
            row_span = effective.max(-1).values - effective.min(-1).values
            used = valid.float()
            stats = [{
                "final_target_ce": ce[i].item(),
                "pre_gate_logit_span_used_mean": (row_span[i, :-1] * used[i]).sum().item() / used[i].sum().item(),
                "pre_gate_logit_span_used_max": row_span[i, :-1][valid[i]].max().item(),
                "parameter_abs_max": param[i].abs().max().item(),
                "plan_fraction_clamped_used": (float((plan[i, :-1][valid[i]] < 1e-6).float().mean())
                                               if variant == "bounded" else None),
            } for i in range(len(X))]
        return tokens.cpu(), param.detach().cpu(), stats, trajectory, max_grad
    finally:
        if context is not None:
            context.__exit__(None, None, None)


def score(jobs, captions):
    per = {}
    for method, caps in captions.items():
        rows = []
        for job, cap in zip(jobs, caps):
            refs = job["refs"]
            support = set().union(*(objects(ref) for ref in refs))
            mentioned = objects(cap)
            unsupported = mentioned - support
            heldout = [ref for k, ref in enumerate(refs) if k != job["k"]] if job["kind"] == "true" else refs
            rows.append({
                "target_r1": r1(cap, [job["target"]]),
                "grounding_r1": r1(cap, refs),
                "heldout_r1": r1(cap, heldout),
                "chair_s_proxy": float(bool(unsupported)),
                "chair_i_proxy": len(unsupported) / len(mentioned) if mentioned else 0.0,
                "unsupported_objects": sorted(unsupported),
                "mentioned_objects": sorted(mentioned),
                "exact_target_normalized": float(norm(cap) == norm(job["target"])),
            })
        per[method] = rows
    summary = {}
    comparisons = {}
    keys = ["target_r1", "grounding_r1", "heldout_r1", "chair_s_proxy", "chair_i_proxy", "exact_target_normalized"]
    for kind in ["true", "mismatch"]:
        selected = [i for i, j in enumerate(jobs) if j["kind"] == kind]
        ids = sorted({jobs[i]["image"] for i in selected})
        grouped = [[i for i in selected if jobs[i]["image"] == image] for image in ids]
        means = {method: {key: np.array([np.mean([rows[i][key] for i in group]) for group in grouped])
                          for key in keys} for method, rows in per.items()}
        summary[kind] = {method: {key: float(values.mean()) for key, values in metrics.items()}
                         for method, metrics in means.items()}
        pairs = [(a, b) for a in ["saved_free300", "bounded300", "unbounded300"]
                 for b in ["saved_baseline"]]
        pairs += [("unbounded300", "saved_free300"), ("unbounded300", "bounded300"), ("bounded300", "saved_free300")]
        comparisons[kind] = {}
        for a, b in pairs:
            comparison = {}
            for key in keys:
                delta, low, high = boot_ci(means[a][key] - means[b][key])
                comparison[key] = {"delta": delta, "ci95": [low, high]}
            comparisons[kind][f"{a}_minus_{b}"] = comparison
    return {"summary": summary, "paired_image_bootstrap": comparisons, "per_job": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="unbounded_logit_control_n200_s0")
    ap.add_argument("--batch", type=int, default=250)
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
    assert L_CAP == 40 and N_PATCHES == 256
    root = Path(__file__).resolve().parent
    out = Path(OUT_DIR) / args.output
    out.mkdir(exist_ok=False)
    source_path = Path(OUT_DIR) / "inversion_protocol_n1000_s0.json"
    with source_path.open() as f:
        source = json.load(f)
    assert source["n_steps"] == 300 and source["lr"] == 0.05
    records = source["records"][:200]
    assert [r["image"] for r in records] == list(range(200))
    jobs = []
    # Fix cohort and slots without inspecting any outcome or choosing a new shuffle.
    for kind in ["true", "mismatch"]:
        for rec in records:
            for k in ([0, 1] if kind == "true" else [-1]):
                assert len(rec["refs"]) == 5
                jobs.append({
                    "job": len(jobs), "image": rec["image"], "feature_index": TEST_START + rec["image"],
                    "kind": kind, "k": k, "refs": rec["refs"],
                    "target": rec["refs"][k] if kind == "true" else rec["mismatch_target"],
                    "mismatch_source": rec["mismatch_source"],
                })
    assert len(jobs) == 600
    config = {
        "cohort": "first 200 saved records, image indices 0..199, chosen by index before optimization",
        "slots": "references 0 and 1, plus the exact saved mismatch target from original 1000-image shuffle",
        "n_images": 200, "n_jobs": 600, "steps": 300, "lr": 0.05, "batch": args.batch,
        "optimizer": "Adam, betas=(0.9,0.999), eps=1e-8, weight_decay=0; fresh state per batch/variant",
        "loss": "native cross_entropy mean over ALL nonpadding target tokens in the batch",
        "seed": 0, "dtype": "float32", "tf32": False, "deterministic_algorithms": True,
        "control_shape_per_job": [40, 256], "control_parameters_per_job": 10240,
        "layer_sharing": "one 40x256 control shared by all 3 layers; native cross-attention is single-head",
        "baseline": "saved denoised external A from plans_test.pt, unchanged; native guided decoder",
        "bounded": "theta0=log(clamp(A,1e-8)); A_theta=softmax(theta); qk + gate_l*log(clamp(A_theta,1e-6))",
        "unbounded": "U0=0; qk + gate_l*log(clamp(A,1e-6)) + gate_l*U, no U clamp/projection/normalization",
        "gain_matching": "same frozen sigmoid(plan_gate) multiplies both controls; no added gain parameters",
        "initialization": "same effective denoised bias; bounded log-softmax roundoff checked explicitly",
        "bounded_pre_gate_span_limit": float(-np.log(1e-6)),
        "parameter_gauge": "40 rowwise additive constants are softmax-invariant; final row unused by max-40-token decoding in BOTH arms",
        "model_training": False, "cuda_visible_devices": "0", "test_start": TEST_START,
        "metrics": "standard rouge_score ROUGE-1 F1, no stemming, repository norm; best-reference heldout/grounding",
        "chair_style": "reference-only COCO synonym-set proxy, not official annotation-based CHAIR; I is macro mean unsupported/mentioned unique categories; S is any unsupported category",
        "uncertainty": "10000 paired image bootstrap resamples, seed 0; average true slots within image first",
        "saved_comparator_caveat": "original batch grouping is not retained in saved JSON; bounded rerun provides identical-batch comparator",
    }
    write_json(out / "preregistered_jobs.json", {"config": config, "jobs": jobs})
    paths = [source_path, Path(CACHE_PATH)]
    paths += [Path(SAVE_DIR) / name for name in ["model_proj.pt", "model_guided.pt", "plans_test.pt", "tok_vocab.json"]]
    paths += [root / name for name in [Path(__file__).name, "run_rev_inversion_protocol.py", "run_rev_matched_effect_control.py",
                                     "run_rev_eval_metrics.py", "run_cap_improved.py", "run_captioning.py",
                                     "run_summarization.py", "run_sum_masked.py", "run_save_plans.py"]]
    print("Hashing inputs; cohort and jobs already saved", flush=True)
    hashes = {str(path.relative_to(root)): sha256(path) for path in paths}
    hashes["preregistered_jobs.json"] = sha256(out / "preregistered_jobs.json")
    write_json(out / "sources.json", {
        "sha256": hashes,
        "versions": {p: importlib.metadata.version(p) for p in ["torch", "numpy", "rouge-score", "transformers"]},
        "gpu": subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"], text=True).strip(),
    })
    tok, proj, dec = load_models()
    models = [proj, dec]
    assert all(not p.requires_grad for m in models for p in m.parameters())
    assert all(not m.training for model in models for m in model.modules())
    before = model_digest(models)
    feats = torch.load(CACHE_PATH, weights_only=True, map_location="cpu", mmap=True)
    plans = torch.load(Path(SAVE_DIR) / "plans_test.pt", weights_only=True, map_location="cpu")
    assert plans.shape[1:] == (40, 256)
    gates = [torch.sigmoid(layer.plan_gate).item() for layer in dec.layers]

    def tensors(chunk):
        idx = [job["image"] for job in chunk]
        with torch.no_grad():
            X = proj(feats[[TEST_START + i for i in idx]].to(DEVICE))
        A = plans[idx].to(DEVICE)
        tgt = torch.tensor([tok.enc(job["target"], max_len=L_CAP, bos=True, eos=True) for job in chunk], device=DEVICE)
        return X, A, tgt

    # Small mixed-kind batch must pass before any 300-step run.
    smoke_start = time.perf_counter()
    small = [jobs[0], jobs[1], jobs[400]]
    X, A, tgt = tensors(small)
    zero = torch.zeros_like(A, requires_grad=True)
    with torch.no_grad():
        native = dec_forward(dec, X, tgt[:, :-1], A)
        native_tokens = dec_generate(dec, X, A)
        initial_free = dec_forward(dec, X, tgt[:, :-1], make_plan(A.log(), "free"))
        initial_free_tokens = dec_generate(dec, X, make_plan(A.log(), "free"))
    with additive_logits(dec, zero):
        biased = dec_forward(dec, X, tgt[:, :-1], A)
        zero_tokens = dec_generate(dec, X, A)
        valid = tgt[:, 1:] != 0
        loss = F.cross_entropy(biased[valid], tgt[:, 1:][valid])
        zero_grad = torch.autograd.grad(loss, zero)[0]
    theta = A.log().detach().requires_grad_(True)
    pred = dec_forward(dec, X, tgt[:, :-1], make_plan(theta, "free"))
    free_grad = torch.autograd.grad(F.cross_entropy(pred[valid], tgt[:, 1:][valid]), theta)[0]
    assert torch.equal(native, biased), "Zero-bias logits differ from native"
    assert torch.equal(native_tokens, zero_tokens), "Zero-bias generation differs from native"
    assert torch.equal(native_tokens, initial_free_tokens), "Initial free generation differs"
    torch.testing.assert_close(native, initial_free, rtol=1e-4, atol=2e-5)
    torch.testing.assert_close(zero_grad, free_grad, rtol=2e-3, atol=2e-7)
    assert torch.isfinite(zero_grad).all() and zero_grad.abs().max() > 0
    smoke = {}
    for variant in ["bounded", "unbounded"]:
        tokens, params, stats, trajectory, grad = optimize(dec, X, A, tgt, variant, 3, 0.05)
        smoke[variant] = {"captions": [tok.dec(t) for t in tokens.tolist()], "trajectory": trajectory, "max_gradient": grad}
    assert before == model_digest(models)
    validation = {
        "zero_logits_bitwise_identical": True, "zero_generated_tokens_bitwise_identical": True,
        "initial_bounded_logits_max_abs_error": (native - initial_free).abs().max().item(),
        "initial_bounded_tokens_identical": True,
        "initial_gradient_max_abs_error": (zero_grad - free_grad).abs().max().item(),
        "initial_gradient_relative_l2_error": ((zero_grad - free_grad).norm() / free_grad.norm()).item(),
        "finite_nonzero_gradients": True, "small_batch_model_unchanged": True,
        "smoke_jobs": [j["job"] for j in small], "smoke": smoke,
        "smoke_seconds": time.perf_counter() - smoke_start, "frozen_gates": gates,
        "model_state_sha256_before": before,
    }
    write_json(out / "validation_before_run.json", validation)
    print("PASS: zero bias bitwise identical; initial gradients gain-matched; small-batch weights unchanged", flush=True)
    captions = {"saved_baseline": [], "saved_free300": [], "regenerated_baseline": [], "bounded300": [], "unbounded300": []}
    for job in jobs:
        rec = records[job["image"]]
        captions["saved_baseline"].append(rec["baseline"])
        captions["saved_free300"].append(rec["variants"]["free"]["true"][job["k"]] if job["kind"] == "true" else rec["variants"]["free"]["mismatch"])
    all_stats = {"bounded300": [], "unbounded300": []}
    timing = {"bounded300": 0.0, "unbounded300": 0.0, "baseline_validation": 0.0}
    checkpoints = []
    for start in range(0, len(jobs), args.batch):
        chunk = jobs[start:start + args.batch]
        X, A, tgt = tensors(chunk)
        tick = time.perf_counter()
        with torch.no_grad():
            normal = dec_generate(dec, X, A)
            with additive_logits(dec, torch.zeros_like(A)):
                zero_tokens = dec_generate(dec, X, A)
                z = dec_forward(dec, X, tgt[:, :-1], A)
            plain = dec_forward(dec, X, tgt[:, :-1], A)
            assert torch.equal(normal, zero_tokens) and torch.equal(z, plain)
        captions["regenerated_baseline"].extend(tok.dec(t) for t in normal.cpu().tolist())
        torch.cuda.synchronize()
        timing["baseline_validation"] += time.perf_counter() - tick
        for variant in ["bounded", "unbounded"]:
            tick = time.perf_counter()
            tokens, params, stats, trajectory, grad = optimize(dec, X, A, tgt, variant, 300, 0.05)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - tick
            name = variant + "300"
            timing[name] += elapsed
            caps = [tok.dec(t) for t in tokens.tolist()]
            captions[name].extend(caps)
            all_stats[name].extend(stats)
            checkpoint = out / f"{name}_jobs{start:04d}_{start + len(chunk):04d}.pt"
            with checkpoint.open("xb") as f:
                torch.save({"job_ids": [j["job"] for j in chunk], "parameter": params,
                            "tokens": tokens, "target_tokens": tgt.cpu(), "stats": stats}, f)
            item = {"path": checkpoint.name, "sha256": sha256(checkpoint), "seconds": elapsed,
                    "trajectory": trajectory, "max_abs_gradient": grad, "all_gradients_finite": True}
            checkpoints.append(item)
            write_json(out / f"{name}_jobs{start:04d}_{start + len(chunk):04d}.json", {**item, "captions": caps})
            print(f"{name}: {start + len(chunk)}/600 jobs; batch {elapsed:.1f}s; total {time.perf_counter() - started:.1f}s", flush=True)
    assert all(len(caps) == len(jobs) for caps in captions.values())
    after = model_digest(models)
    assert before == after
    assert all(p.grad is None and not p.requires_grad for m in models for p in m.parameters())
    validation.update({
        "model_state_sha256_after": after, "all_model_parameters_and_buffers_bitwise_unchanged": True,
        "all_model_gradients_none": True, "all_600_zero_bias_logits_and_tokens_bitwise_identical": True,
        "saved_baseline_caption_disagreements": sum(a != b for a, b in zip(captions["saved_baseline"], captions["regenerated_baseline"])),
        "saved_free300_vs_bounded_rerun_caption_disagreements": sum(a != b for a, b in zip(captions["saved_free300"], captions["bounded300"])),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
    })
    result = {"config": config, "sources": hashes, "validation": validation, "jobs": jobs,
              "captions": captions, "control_stats": all_stats, "checkpoints": checkpoints, "timing_seconds": timing}
    score_start = time.perf_counter()
    result["metrics"] = score(jobs, captions)
    timing["scoring"] = time.perf_counter() - score_start
    timing["total"] = time.perf_counter() - started
    write_json(out / "results.json", result)
    print(json.dumps({"validation": validation, "summary": result["metrics"]["summary"], "timing_seconds": timing}, indent=2), flush=True)
    print("SAVED", out / "results.json", flush=True)


if __name__ == "__main__":
    main()
