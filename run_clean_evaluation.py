"""Offline evaluation of separately trained clean small captioners.

No training, dataset access, automatic device selection, or legacy SAVE_DIR edits.
CUDA requires BOTH --device cuda and --allow-gpu, after explicit parent approval.
Default modes: full-test forward controls, 5+1 protocol, and saved901 swaps.
Optional modes: corruption (2+1 slots), bounded (first200, 2+1 slots).
--matched-gain calibrates global gain/edit matching on 150 validation swaps,
locks both steps, then evaluates all 901 saved test swaps. No test selection.
All output directories must be new. Complete JSON files are published atomically.
"""

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
TEST_START = 14000
VAL_START = 13000
CALIBRATION_N = 150
CALIBRATION_SEED = 0
N_TEST = 1000
L_CAP = 40
N_PATCHES = 256
VARIANTS = ("free", "onehot", "hidden")
STEPS = 300
LR = 0.05
MATCH_STEPS = tuple(range(21))
CORRUPTION_FRACTIONS = (0.0, 0.25, 0.5, 1.0)
CORRUPTION_SEED = 20261030
MODEL_FILES = ("model_proj.pt", "model_free.pt", "model_denoiser.pt",
               "model_guided.pt", "tok_vocab.json", "plans_test.pt", "caps_test.json")


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    """Publish only complete, finite JSON; never overwrite an existing artifact."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def validate_source(source, n=N_TEST):
    records = source["records"]
    if source["n_images"] != n or [r["image"] for r in records] != list(range(n)):
        raise ValueError("Expected complete, ordered original test image indices")
    donors = [r["mismatch_source"] for r in records]
    if sorted(donors) != list(range(n)) or any(i == j for i, j in enumerate(donors)):
        raise ValueError("Original shuffled donors must be a derangement")
    for rec in records:
        if len(rec["refs"]) != 5 or not all(isinstance(r, str) and r for r in rec["refs"]):
            raise ValueError("Exactly five nonempty reference slots required, including duplicates")
        if rec["mismatch_target"] not in records[rec["mismatch_source"]]["refs"]:
            raise ValueError("Saved mismatch target is not a donor reference")
    return records


def protocol_jobs(records, slots=range(5), n=None):
    cohort = records if n is None else records[:n]
    jobs = [{"image": r["image"], "kind": "true", "k": k, "target": r["refs"][k]}
            for r in cohort for k in slots]
    jobs += [{"image": r["image"], "kind": "mismatch", "k": -1,
              "target": r["mismatch_target"]} for r in cohort]
    for job in jobs:
        job.update(feature_index=TEST_START + job["image"], plan_index=job["image"])
    return jobs


def validation_jobs(refs):
    """Only validation refs enter eligibility/target selection; OOVs are retained."""
    from run_rev_swap_injection import make_target
    from run_rev_eval_metrics import objects

    if len(refs) != TEST_START - VAL_START or any(
            len(row) != 5 or not all(isinstance(r, str) and r for r in row) for row in refs):
        raise ValueError("Expected exactly 1000 validation images with five reference slots")
    rng = random.Random(CALIBRATION_SEED)
    jobs = []
    for i, row in enumerate(refs):
        own = set().union(*(objects(r) for r in row))
        order = list(range(5))
        rng.shuffle(order)
        for k in order:
            swapped = make_target(row[k], own, rng)
            if swapped is not None:
                target, fake, _ = swapped
                jobs.append({"split": "validation", "feature_index": VAL_START + i,
                             "plan_index": len(jobs), "k": k, "target": target, "fake": fake,
                             "refs": row})
                break
        if len(jobs) == CALIBRATION_N:
            return jobs
    raise ValueError("Fewer than 150 eligible validation counterfactual images")


@torch.no_grad()
def validation_plans(jobs, models, feats, device, batch):
    from run_cap_improved import maskgit_cfg

    indices = [j["feature_index"] for j in jobs]
    if (len(indices) != CALIBRATION_N or len(set(indices)) != len(indices)
            or any(not VAL_START <= i < TEST_START for i in indices)
            or [j["plan_index"] for j in jobs] != list(range(CALIBRATION_N))):
        raise ValueError("Calibration plans require 150 unique validation feature indices")
    plans = []
    for start in range(0, len(indices), batch):
        X = models["proj"](feats[indices[start:start + batch]].to(device))
        if not torch.isfinite(X).all():
            raise ValueError("Nonfinite validation features")
        # Native inference mask only: no stochastic training/sample mask is applied.
        raw = maskgit_cfg(models["denoiser"], X, n_steps=4, cfg_scale=2.0, device=device)
        if not torch.isfinite(raw).all() or not torch.equal(raw, F.one_hot(raw.argmax(-1), raw.shape[-1]).float()):
            raise ValueError("K4 validation MaskGIT must commit every row to a finite onehot")
        A = F.softmax(10.0 * raw, -1)
        validate_plans(A)
        plans.append(A.cpu())
    return torch.cat(plans)


def validate_plans(plans):
    if not torch.is_tensor(plans) or plans.ndim != 3 or not plans.is_floating_point():
        raise ValueError("Plans must be a floating [image,row,patch] tensor")
    if not torch.isfinite(plans).all() or (plans < 0).any():
        raise ValueError("Nonfinite or negative plan entries")
    if not torch.allclose(plans.sum(-1), torch.ones_like(plans[..., 0]), atol=2e-6, rtol=1e-5):
        raise ValueError("Plan rows must sum to one")


def validate_training_manifest(manifest, hashes, model_dir):
    if manifest.get("status") != "complete" or manifest.get("smoke") is not False:
        raise ValueError("Evaluation requires a completed, non-smoke clean training export")
    expected = {"train": list(range(13000)), "val": list(range(13000, 14000)),
                "test": list(range(14000, 15000))}
    if manifest.get("split_ids") != expected:
        raise ValueError("Training manifest split IDs differ from strict 13k/1k/1k")
    if not manifest.get("checks", {}).get("vocab_heldout_invariant"):
        raise ValueError("Training manifest lacks train-only vocabulary audit")
    for name in (*MODEL_FILES, "captions_test.json"):
        if manifest.get("artifact_hashes", {}).get(name) != hashes[str((model_dir / name).resolve())]:
            raise ValueError(f"Training export hash mismatch: {name}")


def corrupt_plans(plans, fraction, seed=CORRUPTION_SEED):
    """Fixed nested row subsets and replacements, independent of captions/refs.

    Re-seeding a local CPU generator makes all fractions share the same row order
    and random patch choices. Replacement may coincidentally equal the old row.
    The mask denotes intervention, not necessarily a changed argmax.
    """
    validate_plans(plans)
    if fraction not in CORRUPTION_FRACTIONS:
        raise ValueError("Use the prespecified corruption fractions")
    n, rows, patches = plans.shape
    rng = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.stack([torch.randperm(rows, generator=rng) for _ in range(n)])
    mask = torch.zeros(n, rows, dtype=torch.bool)
    mask.scatter_(1, order[:, :round(rows * fraction)], True)
    indices = torch.randint(patches, (n, rows), generator=rng)
    replacements = F.softmax(10.0 * F.one_hot(indices, patches).to(plans.dtype), -1)
    mask = mask.to(plans.device)
    corrupted = torch.where(mask[..., None], replacements.to(plans.device), plans)
    validate_plans(corrupted)
    return corrupted, mask, indices


def load_models(model_dir, device):
    # Only implementations are imported; legacy loaders/main functions are never used.
    from run_summarization import Tokenizer, D_MODEL, N_HEADS, N_LAYERS, N_MP
    from run_captioning import PatchProjector, CaptionDecoder
    from run_cap_improved import SoftGuidedCaptionDecoder, CFGDenoiser

    vocab = read_json(model_dir / "tok_vocab.json")
    if (not isinstance(vocab, list) or vocab[:4] != ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
            or not all(isinstance(w, str) for w in vocab) or len(set(vocab)) != len(vocab)):
        raise ValueError("Invalid clean tokenizer vocabulary")
    tok = Tokenizer()
    tok.vocab, tok.w2i = vocab, {w: i for i, w in enumerate(vocab)}
    models = {
        "proj": PatchProjector(768, D_MODEL),
        "free": CaptionDecoder(len(vocab), D_MODEL, N_HEADS, N_LAYERS, n_src=N_PATCHES),
        "denoiser": CFGDenoiser(D_MODEL, N_PATCHES, L_CAP, N_MP),
        "guided": SoftGuidedCaptionDecoder(len(vocab), D_MODEL, N_HEADS, N_LAYERS),
    }
    for name, model in models.items():
        model.load_state_dict(torch.load(model_dir / f"model_{name}.pt", map_location="cpu", weights_only=True))
        model.eval().requires_grad_(False)
        if any(not torch.isfinite(p).all() for p in model.state_dict().values()):
            raise ValueError(f"Nonfinite checkpoint: {name}")
        model.to(device)
    return tok, models


def model_digest(models):
    digest = hashlib.sha256()
    for name, model in models.items():
        for key, value in model.state_dict().items():
            digest.update((name + "." + key).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def additive_logits(dec, bias):
    """Scoped native gate*U adaptation from the unbounded audit, no global edits."""
    originals = [layer.forward for layer in dec.layers]

    def forward_for(layer):
        def forward(x, mem, attn_plan, tgt_mask=None, tgt_pad=None, ret_attn=False):
            xn = layer.ln1(x)
            sa_out, _ = layer.sa(xn, xn, xn, attn_mask=tgt_mask, key_padding_mask=tgt_pad)
            x = x + layer.drop(sa_out)
            xn = layer.ln2(x)
            q, k, v = layer.q(xn), layer.k(mem), layer.v(mem)
            logits = torch.bmm(q, k.transpose(-1, -2)) * layer.scale
            gate = torch.sigmoid(layer.plan_gate)
            logits = logits + gate * attn_plan.clamp(min=1e-6).log()
            logits = logits + gate * bias[:, :x.shape[1]]
            weights = F.softmax(logits, -1)
            x = x + layer.drop(layer.o(torch.bmm(weights, v)))
            x = x + layer.drop(layer.ff(layer.ln3(x)))
            return (x, weights) if ret_attn else x
        return forward

    try:
        for layer in dec.layers:
            layer.forward = forward_for(layer)
        yield
    finally:
        for layer, forward in zip(dec.layers, originals):
            layer.forward = forward


def optimize(dec, X, A, target, variant, steps=STEPS, lr=LR, checkpoints=()):
    """Original Adam/token-mean algorithm, with finite checks and optional trajectory."""
    from run_rev_inversion_protocol import dec_forward, dec_generate, make_plan, plan_stats

    validate_plans(A)
    if variant not in (*VARIANTS, "bounded", "unbounded"):
        raise ValueError(variant)
    if not torch.isfinite(X).all():
        raise ValueError("Nonfinite projected features")
    if any(p.requires_grad for p in dec.parameters()) or dec.training:
        raise ValueError("Decoder must be frozen and in eval mode")
    valid = target[:, 1:] != 0
    if not valid.any(1).all():
        raise ValueError("Empty encoded target")
    initial = (torch.zeros(len(X), L_CAP, X.shape[-1], device=X.device) if variant == "hidden"
               else torch.zeros_like(A) if variant == "unbounded" else A.clamp(min=1e-8).log().clone())
    param = torch.nn.Parameter(initial)
    opt = torch.optim.Adam([param], lr=lr)
    trajectory = {}
    max_grad = 0.0
    with additive_logits(dec, param) if variant == "unbounded" else nullcontext():
        for step in range(steps + 1):
            plan = A if variant in ("hidden", "unbounded") else make_plan(param, "free" if variant == "bounded" else variant)
            delta = param if variant == "hidden" else None
            if step in checkpoints:
                trajectory[step] = dec_generate(dec, X, plan, delta).cpu().tolist()
            if step == steps:
                break
            pred = dec_forward(dec, X, target[:, :-1], plan, delta)
            loss = F.cross_entropy(pred[valid], target[:, 1:][valid])
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss at {variant} step {step}")
            opt.zero_grad()
            loss.backward()
            if param.grad is None or not torch.isfinite(param.grad).all():
                raise ValueError(f"Nonfinite/absent gradient at {variant} step {step}")
            max_grad = max(max_grad, param.grad.detach().abs().max().item())
            opt.step()
            if not torch.isfinite(param).all():
                raise ValueError("Nonfinite optimization parameter")
        validate_plans(plan)
        tokens = dec_generate(dec, X, plan, delta).cpu().tolist()
        stats = plan_stats(plan.detach(), A, valid.sum(-1).tolist())
        with torch.no_grad():
            pred = dec_forward(dec, X, target[:, :-1], plan, delta)
            if not torch.isfinite(pred).all():
                raise ValueError("Nonfinite final logits")
            ce = F.cross_entropy(pred.transpose(1, 2), target[:, 1:], reduction="none")
            ce = (ce * valid).sum(1) / valid.sum(1)
            effective = (A.clamp(min=1e-6).log() + param if variant == "unbounded"
                         else plan.clamp(min=1e-6).log())
            span = effective.max(-1).values - effective.min(-1).values
            for i, st in enumerate(stats):
                st.update(final_target_ce=ce[i].item(), max_abs_gradient_batch=max_grad,
                          pre_gate_logit_span_used_max=span[i, :-1][valid[i]].max().item())
    return tokens, stats, trajectory


def target_audit(tok, job):
    words = job["target"].lower().split()
    encoded = tok.enc(job["target"], max_len=L_CAP, bos=True, eos=True)
    result = {"target_tokens": encoded, "oov_words": [w for w in words if w not in tok.w2i],
              "encoded_unk_count": encoded.count(tok.UNK), "truncated": len(words) + 2 > L_CAP}
    if "fake" in job:
        result["fake_oov"] = any(w not in tok.w2i for w in job["fake"].lower().split())
        # Whitespace tokenization retains punctuation: a known 'dog' may be unknown as 'dog,'.
        fake_words = [w for w in words if w.strip(".,!?;:\"'()[]") == job["fake"].lower()]
        result["fake_surface_oov"] = not fake_words or any(w not in tok.w2i for w in fake_words)
    return result


def run_jobs(jobs, variants, tok, models, feats, plans, device, batch, *, checkpoints=()):
    captions = {v: [] for v in variants}
    stats = {v: [] for v in variants}
    trajectories = {s: [] for s in checkpoints}
    for variant in variants:
        for start in range(0, len(jobs), batch):
            chunk = jobs[start:start + batch]
            idx = [j["plan_index"] for j in chunk]
            with torch.no_grad():
                X = models["proj"](feats[[j["feature_index"] for j in chunk]].to(device))
            target = torch.tensor([tok.enc(j["target"], L_CAP, bos=True, eos=True) for j in chunk], device=device)
            grid = checkpoints if variant == "hidden" else ()
            tokens, st, trajectory = optimize(models["guided"], X, plans[idx].to(device), target, variant,
                                               steps=max(grid) if grid else STEPS, checkpoints=grid)
            captions[variant].extend(tok.dec(t) for t in tokens)
            stats[variant].extend(st)
            for step, values in trajectory.items():
                trajectories[step].extend(tok.dec(t) for t in values)
            print(f"{variant}: {min(start + batch, len(jobs))}/{len(jobs)} jobs", flush=True)
    return captions, stats, trajectories


@torch.no_grad()
def forward_captions(tok, models, feats, plans, device, batch, controls=True, *, feature_indices=None):
    from run_rev_inversion_protocol import dec_forward, dec_generate

    captions = {"guided_denoised": []}
    if controls:
        captions.update(free=[], guided_uniform=[])
    if feature_indices is None:
        feature_indices = list(range(TEST_START, TEST_START + len(plans)))
    if len(feature_indices) != len(plans):
        raise ValueError("Forward feature/plan alignment mismatch")
    for start in range(0, len(plans), batch):
        X = models["proj"](feats[feature_indices[start:start + batch]].to(device))
        A = plans[start:start + batch].to(device)
        if not torch.isfinite(X).all():
            raise ValueError("Nonfinite forward features")
        for name in captions:
            if name == "free":
                tokens = models["free"].generate(X)
                logits = models["free"](X, tokens[:, :-1])
            else:
                plan = torch.full_like(A, 1 / N_PATCHES) if name == "guided_uniform" else A
                tokens = dec_generate(models["guided"], X, plan)
                logits = dec_forward(models["guided"], X, tokens[:, :-1], plan)
            if not torch.isfinite(logits).all():
                raise ValueError(f"Nonfinite {name} baseline logits")
            captions[name].extend(tok.dec(t) for t in tokens.cpu().tolist())
    return captions


def protocol_dump(records, baselines, jobs, captions, stats, metadata):
    variants = list(captions)
    rows = [{"image": r["image"], "feature_index": TEST_START + r["image"], "refs": r["refs"],
             "baseline": baselines[r["image"]], "mismatch_source": r["mismatch_source"],
             "mismatch_target": r["mismatch_target"],
             "variants": {v: {"true": [None] * 5, "true_stats": [None] * 5,
                               "mismatch": None, "mismatch_stats": None} for v in variants}}
            for r in records]
    by_id = {r["image"]: r for r in rows}
    slots = sorted({j["k"] for j in jobs if j["kind"] == "true"})
    expected = {(i, "true", k) for i in by_id for k in slots} | {(i, "mismatch", -1) for i in by_id}
    actual = [(j["image"], j["kind"], j["k"]) for j in jobs]
    if not slots or any(k not in range(5) for k in slots) or len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Incomplete or duplicate image/reference jobs")
    for v in variants:
        if len(captions[v]) != len(jobs) or len(stats[v]) != len(jobs):
            raise ValueError("Incomplete caption dump")
        for job, cap, st in zip(jobs, captions[v], stats[v]):
            if not isinstance(cap, str):
                raise ValueError("Missing caption")
            row = by_id[job["image"]]["variants"][v]
            if job["kind"] == "true":
                row["true"][job["k"]], row["true_stats"][job["k"]] = cap, st
            else:
                row["mismatch"], row["mismatch_stats"] = cap, st
    if any(r["variants"][v]["mismatch"] is None for r in rows for v in variants):
        raise ValueError("Missing shuffled job")
    return {"target_mode": "random", "n_images": len(rows), "n_steps": STEPS, "lr": LR,
            "seed": 0, "variants": variants, "param_count_per_inversion": L_CAP * N_PATCHES,
            "evaluated_true_slots": slots,
            "records": rows, "metadata": metadata, "partial": False}


def swap_metrics(caption, baseline, job, refs):
    from paper_cvpr.rescore_rouge import score
    from run_rev_eval_metrics import objects

    return {"target_rouge1": score(caption, job["target"]),
            "gain_rouge1": score(caption, job["target"]) - score(baseline, job["target"]),
            "original_ref_rouge1": score(caption, refs[job["k"]]),
            "gain_original_ref_rouge1": score(caption, refs[job["k"]]) - score(baseline, refs[job["k"]]),
            "injection": float(bool(objects(job["fake"]) & objects(caption))),
            "normalized_token_edit": normalized_token_edit(caption, baseline)}


def normalized_token_edit(caption, baseline):
    """Token Levenshtein / max lengths, using the audit's ROUGE tokenizer."""
    from paper_cvpr.rescore_rouge import SCORER

    a, b = SCORER._tokenizer.tokenize(caption), SCORER._tokenizer.tokenize(baseline)
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = nxt
    return row[-1] / max(len(a), len(b), 1)


def calibrate_matched(jobs, tok, models, feats, args, metadata):
    """Validation-only API: it never receives test jobs, captions, or plans."""
    plans = validation_plans(jobs, models, feats, args.device, args.forward_batch)
    indices = [j["feature_index"] for j in jobs]
    baselines = forward_captions(tok, models, feats, plans, args.device, args.forward_batch,
                                 controls=False, feature_indices=indices)["guided_denoised"]
    free_caps, free_stats, _ = run_jobs(jobs, ["free"], tok, models, feats, plans, args.device, args.batch)
    _, _, hidden_caps = run_jobs(jobs, ["hidden"], tok, models, feats, plans, args.device,
                                 args.batch, checkpoints=MATCH_STEPS)

    def metrics(caps):
        return [swap_metrics(c, b, j, j["refs"]) for c, b, j in zip(caps, baselines, jobs)]

    free = metrics(free_caps["free"])
    hidden = {s: metrics(caps) for s, caps in hidden_caps.items()}
    effects = ("gain_rouge1", "normalized_token_edit")
    free_means = {key: sum(m[key] for m in free) / len(jobs) for key in effects}
    sweep = [{"step": s, **{key: sum(m[key] for m in hidden[s]) / len(jobs) for key in effects}}
             for s in MATCH_STEPS]
    chosen = {key: min(sweep, key=lambda row: abs(row[key] - free_means[key]))["step"] for key in effects}
    audited = [{**j, **target_audit(tok, j)} for j in jobs]
    path = args.out_dir / "matched_validation.json"
    atomic_json(path, {
        "metadata": metadata, "n": len(jobs), "jobs": audited, "baselines": baselines,
        "plan_method": "frozen projector/denoiser; native MaskGIT K4 CFG2; softmax(10 * onehot); no sample mask",
        "plan_tensor_sha256": hashlib.sha256(plans.contiguous().numpy().tobytes()).hexdigest(),
        "free300_captions": free_caps["free"], "free300_stats": free_stats["free"],
        "hidden_captions_by_step": hidden_caps, "free300_metrics": free, "hidden_metrics_by_step": hidden,
        "fake_oov_fraction": sum(j["fake_oov"] for j in audited) / len(jobs),
        "fake_surface_oov_fraction": sum(j["fake_surface_oov"] for j in audited) / len(jobs),
        "jobs_dropped_for_oov": 0,
        "step0_baseline_mismatches": [i for i, b in enumerate(baselines) if hidden_caps[0][i] != b]})
    lock = {"calibration_split": "validation", "n_calibration": len(jobs),
            "calibration_feature_indices": indices, "seed": CALIBRATION_SEED,
            "n_evaluation": 901, "evaluation_feature_range": [TEST_START, TEST_START + N_TEST],
            "steps": list(MATCH_STEPS), "selected_steps": chosen, "sweep": sweep,
            "free_mean_effects": free_means, "lowest_step_ties": True, "test_selection": False,
            "validation_artifact": {"path": path.name, "sha256": sha256(path)}}
    atomic_json(args.out_dir / "matched_calibration_lock.json", lock)
    return lock


def run_matched(jobs, free_caps, baselines, records, tok, models, feats, plans, args, metadata, lock):
    from paper_cvpr.rescore_rouge import paired

    if read_json(args.out_dir / "matched_calibration_lock.json") != lock or lock["test_selection"]:
        raise ValueError("A validation-only calibration lock is required before test matching")
    if sha256(args.out_dir / "matched_validation.json") != lock["validation_artifact"]["sha256"]:
        raise ValueError("Validation calibration artifact changed after locking")
    if len(jobs) != 901 or any(not TEST_START <= j["feature_index"] < TEST_START + N_TEST for j in jobs):
        raise ValueError("Matched evaluation requires all 901 original test jobs")
    grid = tuple(sorted({0, *lock["selected_steps"].values()}))
    _, _, all_caps = run_jobs(jobs, ["hidden"], tok, models, feats, plans, args.device,
                              args.batch, checkpoints=grid)

    def metrics(caps):
        return [swap_metrics(c, baselines[j["image"]], j, records[j["image"]]["refs"])
                for c, j in zip(caps, jobs)]

    free = metrics(free_caps)
    all_metrics = {s: metrics(caps) for s, caps in all_caps.items()}
    summaries = {}
    for effect, step in lock["selected_steps"].items():
        rows = all_metrics[step]
        summaries[effect] = {key: paired([m[key] for m in rows], [m[key] for m in free])
                             for key in free[0]}
    atomic_json(args.out_dir / "matched_gain.json", {
        "metadata": metadata, "calibration": lock, "jobs": jobs, "n_evaluation": len(jobs),
        "baselines": [baselines[j["image"]] for j in jobs], "free300_captions": free_caps,
        "hidden_captions_by_step": all_caps, "hidden_metrics_by_step": all_metrics,
        "free300_metrics": free, "global_steps_from_validation": lock["selected_steps"], "evaluation_vs_free": summaries,
        "step0_baseline_mismatches": [i for i, j in enumerate(jobs) if all_caps[0][i] != baselines[j["image"]]],
        "limitations": ["All stopping-step selection uses validation only; no per-image test selection.",
                         "Global matching is approximate; signed residuals and negative gains are retained.",
                        "Validation was also used for training-stage model selection; it is disjoint from test, not an independent calibration dataset."]})


def run_corruption(records, baselines, tok, models, feats, plans, args, metadata):
    from paper_cvpr.rescore_rouge import score, paired

    n = args.corruption_images
    cohort = records[:n]
    jobs = protocol_jobs(cohort, slots=(0, 1))
    for fraction in CORRUPTION_FRACTIONS:
        corrupted, mask, replacement = corrupt_plans(plans, fraction)
        # Decode the full cohort even for n=200, preserving the clean baseline's
        # final partial-batch boundary (200 cuts through a 32-image batch).
        base = forward_captions(tok, models, feats, corrupted, args.device, args.forward_batch, controls=False)["guided_denoised"][:n]
        if fraction == 0 and base != baselines[:n]:
            raise ValueError("Zero corruption baseline is not identical to clean forward")
        caps, stats, _ = run_jobs(jobs, ["free"], tok, models, feats, corrupted, args.device, args.batch)
        caps["repair"] = [baselines[j["image"]] for j in jobs]
        stats["repair"] = [None] * len(jobs)
        dump = protocol_dump(cohort, base, jobs, caps, stats, {
            **metadata, "corruption_fraction": fraction, "corruption_seed": CORRUPTION_SEED,
            "cohort_image_indices": [0, n], "grid_selection": False,
            "repair": "Restore ordinary trained plans; no target/reference used in repair",
            "repair_steps": 0, "free_steps": STEPS,
            "plan_validation": "finite, nonnegative, unit row sums; forward logits finite"})
        for r in dump["records"]:
            i = r["image"]
            r.update(clean_baseline=baselines[i], corruption_rows=mask[i].nonzero().flatten().tolist(),
                     replacement_patch_indices=replacement[i].tolist())
        per_image = []
        for r in dump["records"]:
            refs = r["refs"]
            base_score = max(score(r["baseline"], ref) for ref in refs)
            repair_score = max(score(r["clean_baseline"], ref) for ref in refs)
            inv = r["variants"]["free"]
            per_image.append({
                "image": r["image"], "baseline_grounding": base_score, "repair_grounding": repair_score,
                "repair_gain": repair_score - base_score,
                "free_true_single_gain": sum(score(inv["true"][k], refs[k]) - score(r["baseline"], refs[k]) for k in (0, 1)) / 2,
                "free_true_heldout_gain": sum(max(score(inv["true"][k], ref) for j, ref in enumerate(refs) if j != k)
                                               - max(score(r["baseline"], ref) for j, ref in enumerate(refs) if j != k)
                                               for k in (0, 1)) / 2,
                "free_shuffle_target_gain": score(inv["mismatch"], r["mismatch_target"]) - score(r["baseline"], r["mismatch_target"]),
                "free_shuffle_grounding_gain": max(score(inv["mismatch"], ref) for ref in refs) - base_score,
            })
        dump["per_image_scores"] = per_image
        dump["cohort_summary"] = {key: paired([r[key] for r in per_image], [0.] * len(per_image))
                                   for key in per_image[0] if key != "image"}
        dump["interpretation"] = "Measured signed gains only; no guaranteed improvement or forced positive floor."
        atomic_json(args.out_dir / f"corruption_{fraction:g}.json", dump)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--features", type=Path, default=ROOT / "results/dinov2_patch_features_15k.pt")
    ap.add_argument("--protocol-source", type=Path, default=ROOT / "results/revision/inversion_protocol_n1000_s0.json")
    ap.add_argument("--swap-source", type=Path, default=ROOT / "results/revision/swap_injection_n1000_s0.json")
    ap.add_argument("--all-caps", type=Path, default=ROOT / "results/revision/coco_caps_15k.json",
                    help="Cached captions; only rows 13000:14000 determine validation matching jobs")
    ap.add_argument("--modes", nargs="+", choices=("forward", "protocol", "counterfactual", "corruption", "bounded"),
                    default=["forward", "protocol", "counterfactual"])
    ap.add_argument("--matched-gain", action="store_true", help="Validation-only global gain and token-edit matching")
    ap.add_argument("--corruption-images", type=int, choices=(200, 1000), default=1000,
                    help="Fixed full1000 cohort by default; optional first200 subset, never used for selection")
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--forward-batch", type=int, default=32, help="Match clean training export generation batches")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--allow-gpu", action="store_true", help="Set only after explicit parent permission")
    ap.add_argument("--wait-for-gpu", action="store_true", help="Wait for the shared exclusive GPU0 lease instead of failing if busy")
    ap.add_argument("--minimum-free-mib", type=int, default=24000)
    ap.add_argument("--preflight-only", action="store_true", help="Validate local inputs on CPU, without decoding/inversion")
    return ap


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    if args.preflight_only:
        args.device = "cpu"
    if args.device == "cuda" and not args.allow_gpu:
        ap.error("CUDA is forbidden without explicit parent permission and --allow-gpu")
    if args.batch < 1 or args.forward_batch < 1:
        ap.error("Batch sizes must be positive")
    if args.minimum_free_mib < 0:
        ap.error("--minimum-free-mib must be nonnegative")
    if args.out_dir.exists():
        ap.error("--out-dir must be new; existing results are never overwritten")
    if args.matched_gain and "counterfactual" not in args.modes:
        ap.error("--matched-gain requires counterfactual mode")
    if args.device == "cpu":
        # Legacy implementation imports call is_available(); hide GPUs BEFORE those imports.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ.get("ATTN_ENCODER", "dinov2") != "dinov2":
        ap.error("Clean small evaluation requires the original DINOv2 256-patch architecture")
    if args.device == "cuda":
        # Lazy import, fail closed if unavailable. Hold the shared lock through
        # every CUDA allocation, result write, and final integrity check.
        from clean_gpu_guard import gpu_lease
        with gpu_lease(wait=args.wait_for_gpu, minimum_free_mib=args.minimum_free_mib) as lease:
            if lease["index"] != 0:
                raise RuntimeError("Evaluation requires the exclusive physical GPU0 lease")
            run_evaluation(args, gpu_uuid=lease["gpu_uuid"])
    else:
        run_evaluation(args)


def run_evaluation(args, gpu_uuid=None):
    torch.set_num_threads(4)
    started = time.perf_counter()
    source = read_json(args.protocol_source)
    records = validate_source(source)
    refs = read_json(args.model_dir / "caps_test.json")
    if refs != [r["refs"] for r in records]:
        raise ValueError("Clean caps_test does not match all 1000 original reference slots/order")
    paths = [args.model_dir / name for name in MODEL_FILES] + [args.features, args.protocol_source]
    manifest = read_json(args.model_dir / "manifest.json")
    exported = read_json(args.model_dir / "captions_test.json")
    paths += [args.model_dir / "manifest.json", args.model_dir / "captions_test.json"]
    calibration_jobs = []
    if args.matched_gain:
        all_caps = read_json(args.all_caps)
        if len(all_caps) != 15000:
            raise ValueError("Expected the unchanged 15k cached caption rows")
        calibration_jobs = validation_jobs(all_caps[VAL_START:TEST_START])
        paths += [args.all_caps, ROOT / "run_rev_swap_injection.py"]
    if gpu_uuid is not None:
        paths.append(ROOT / "clean_gpu_guard.py")
    jobs = protocol_jobs(records)
    swap_jobs = []
    if "counterfactual" in args.modes:
        saved = read_json(args.swap_source)["records"]
        swap_jobs = [{key: r[key] for key in ("image", "k", "target", "fake")} for r in saved]
        if len(swap_jobs) != 901 or len({j["image"] for j in swap_jobs}) != 901:
            raise ValueError("Expected the original 901 unique-image swap jobs in saved order")
        if any(not 0 <= j["image"] < N_TEST or not 0 <= j["k"] < 5 for j in swap_jobs):
            raise ValueError("Invalid saved swap image/reference slot")
        for job in swap_jobs:
            job.update(feature_index=TEST_START + job["image"], plan_index=job["image"])
        paths.append(args.swap_source)
    paths += [Path(__file__), ROOT / "run_rev_inversion_protocol.py", ROOT / "run_summarization.py",
              ROOT / "run_cap_improved.py", ROOT / "run_captioning.py", ROOT / "run_sum_masked.py",
              ROOT / "run_rev_eval_metrics.py", ROOT / "paper_cvpr/rescore_rouge.py"]
    hashes = {str(p.resolve()): sha256(p) for p in paths}
    validate_training_manifest(manifest, hashes, args.model_dir)
    training_inputs = [("results/dinov2_patch_features_15k.pt", args.features),
                       ("results/revision/inversion_protocol_n1000_s0.json", args.protocol_source)]
    if args.matched_gain:
        training_inputs.append(("results/revision/coco_caps_15k.json", args.all_caps))
    for original, path in training_inputs:
        if manifest.get("hashes", {}).get(original) != hashes[str(path.resolve())]:
            raise ValueError(f"Input differs from clean training audit: {original}")
    if exported["image_ids"] != list(range(TEST_START, TEST_START + N_TEST)) or exported["refs"] != refs:
        raise ValueError("Forward export image/reference alignment mismatch")
    if any(len(exported[k]) != N_TEST for k in ("free", "predicted")):
        raise ValueError("Incomplete clean forward export")
    feats = torch.load(args.features, map_location="cpu", weights_only=True, mmap=True)
    plans = torch.load(args.model_dir / "plans_test.pt", map_location="cpu", weights_only=True)
    if feats.shape != (15000, N_PATCHES, 768) or plans.shape != (N_TEST, L_CAP, N_PATCHES):
        raise ValueError("Expected unchanged 15k feature cache and 1000 clean test plans")
    validate_plans(plans)
    if any(not torch.isfinite(chunk).all() for chunk in feats[TEST_START:].split(32)):
        raise ValueError("Nonfinite test features")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    metadata = {"model_dir": str(args.model_dir.resolve()), "input_sha256": hashes,
                "split": {"train": [0, 13000], "val": [13000, 14000], "test": [14000, 15000]},
                "ranges_half_open": True, "batch": args.batch, "forward_batch": args.forward_batch,
                "steps": STEPS, "lr": LR,
                "modes": args.modes, "matched_gain": args.matched_gain, "seed": 0,
                "device": args.device, "gpu_uuid": gpu_uuid, "torch": torch.__version__, "model_training": False,
                "optimizer": "Adam default betas/eps, mean over all nonpadding batch tokens",
                "corruption": {"fractions": list(CORRUPTION_FRACTIONS), "seed": CORRUPTION_SEED,
                               "n_images": args.corruption_images, "true_slots": [0, 1],
                               "cohort_indices": [0, args.corruption_images], "grid_selection": False,
                               "selection": "fixed nested random rows, independent of refs and outcomes"},
                "matching": {"steps": list(MATCH_STEPS), "calibration_split": "validation",
                             "calibration_feature_indices": [j["feature_index"] for j in calibration_jobs],
                             "calibration_seed": CALIBRATION_SEED, "n_evaluation": 901,
                             "effects": ["gain_rouge1", "normalized_token_edit"],
                             "selection": "nearest validation mean effect to validation free300, lowest-step ties",
                             "eligibility": "first150 images with a valid original-logic swap; no vocabulary filtering",
                             "test_selection": False},
                "bounded": {"n_images": 200, "true_slots": [0, 1], "steps": STEPS,
                            "stage": "native cross-attention logits, shared control, frozen learned gates"}}
    atomic_json(args.out_dir / "planned_jobs.json", {
        "metadata": metadata, "protocol_jobs": jobs, "swap_jobs": swap_jobs, "validation_jobs": calibration_jobs,
        "references_and_donors": [{k: r[k] for k in ("image", "refs", "mismatch_source", "mismatch_target")} for r in records]})
    torch.manual_seed(0)
    random.seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    tok, models = load_models(args.model_dir, args.device)
    before = model_digest(models)
    if args.preflight_only:
        atomic_json(args.out_dir / "preflight.json", {"valid": True, "gpu_run": False, "model_state_sha256": before})
        print("CPU preflight passed; no decoding or optimization performed", flush=True)
        return
    # Lock validation choices before any test-target encoding, forward captions,
    # protocol inversion, or counterfactual outputs can be produced.
    lock = calibrate_matched(calibration_jobs, tok, models, feats, args, metadata) if args.matched_gain else None
    audit = [{**j, **target_audit(tok, j)} for j in jobs]
    swap_audit = [{**j, **target_audit(tok, j)} for j in swap_jobs]
    atomic_json(args.out_dir / "target_encoding.json", {
        "protocol": audit, "counterfactual": swap_audit, "counterfactual_jobs_dropped": 0,
        "fake_oov_fraction": sum(j["fake_oov"] for j in swap_audit) / len(swap_audit) if swap_audit else None,
        "fake_surface_oov_fraction": sum(j["fake_surface_oov"] for j in swap_audit) / len(swap_audit) if swap_audit else None})
    forward = forward_captions(tok, models, feats, plans, args.device, args.forward_batch)
    baselines = forward["guided_denoised"]
    export_mismatches = {name: [i for i, (a, b) in enumerate(zip(forward[name], exported[key])) if a != b]
                         for name, key in (("free", "free"), ("guided_denoised", "predicted"))}
    from paper_cvpr.rescore_rouge import score, paired
    values = {name: [max(score(c, r) for r in records[i]["refs"]) for i, c in enumerate(caps)]
              for name, caps in forward.items()}
    atomic_json(args.out_dir / "forward.json", {
        "metadata": metadata, "n_images": N_TEST, "image_ids": list(range(N_TEST)), "caps": forward,
        "refs": refs, "per_image_rouge1": values,
        "paired": {"denoised_minus_" + name: paired(values["guided_denoised"], values[name])
                   for name in ("free", "guided_uniform")}, "baseline_logits_finite": True,
        "training_export_caption_mismatches": export_mismatches,
        "baseline_policy": "Regenerated clean captions shared by all modes; export disagreements flagged, never rebased"})
    if "protocol" in args.modes:
        caps, stats, _ = run_jobs(jobs, VARIANTS, tok, models, feats, plans, args.device, args.batch)
        atomic_json(args.out_dir / "inversion_protocol_n1000_s0.json",
                    protocol_dump(records, baselines, jobs, caps, stats, metadata))
    if "counterfactual" in args.modes:
        caps, stats, _ = run_jobs(swap_jobs, VARIANTS, tok, models, feats, plans, args.device, args.batch)
        rows = [{**j, "baseline": baselines[j["image"]], "refs": records[j["image"]]["refs"],
                 **{v: caps[v][i] for v in VARIANTS}, "stats": {v: stats[v][i] for v in VARIANTS},
                 "scores": {v: swap_metrics(caps[v][i], baselines[j["image"]], j, records[j["image"]]["refs"])
                            for v in VARIANTS}} for i, j in enumerate(swap_audit)]
        atomic_json(args.out_dir / "swap_injection_n1000_s0.json", {
            "metadata": metadata, "n": 901, "variants": list(VARIANTS), "records": rows,
            "fake_oov_fraction": sum(j["fake_oov"] for j in swap_audit) / 901,
            "fake_surface_oov_fraction": sum(j["fake_surface_oov"] for j in swap_audit) / 901,
            "jobs_dropped": 0})
        if args.matched_gain:
            run_matched(swap_jobs, caps["free"], baselines, records, tok, models, feats, plans, args, metadata, lock)
    if "bounded" in args.modes:
        bounded_jobs = protocol_jobs(records, slots=(0, 1), n=200)
        # Preserve the original control's kind/image/slot ordering and batch boundaries.
        bounded_jobs.sort(key=lambda j: (j["kind"] != "true", j["image"], j["k"]))
        caps, stats, _ = run_jobs(bounded_jobs, ("bounded", "unbounded"), tok, models, feats, plans, args.device, args.batch)
        atomic_json(args.out_dir / "bounded_unbounded_n200.json",
                    protocol_dump(records[:200], baselines, bounded_jobs, caps, stats, metadata))
    if "corruption" in args.modes:
        run_corruption(records, baselines, tok, models, feats, plans, args, metadata)
    after = model_digest(models)
    if before != after or any(p.requires_grad or p.grad is not None for m in models.values() for p in m.parameters()):
        raise RuntimeError("Frozen model state/gradients changed")
    # Training may run concurrently; fail explicitly if any consumed artifact changed.
    if any(sha256(path) != digest for path, digest in hashes.items()):
        raise RuntimeError("An input changed during evaluation; output is not certified complete")
    atomic_json(args.out_dir / "complete.json", {"model_state_sha256_before": before,
                "model_state_sha256_after": after, "all_inputs_unchanged": True,
                "all_model_parameters_frozen": True, "elapsed_seconds": time.perf_counter() - started})
    print("Complete:", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
