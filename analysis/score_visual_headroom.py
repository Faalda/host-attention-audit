"""Offline, full-cohort visual-headroom metric sidecar. Never generate captions.

Run: PYTHONHASHSEED=0 ../.venv/bin/python -B analysis/score_visual_headroom.py
Only a fresh directory under results/visual_headroom_metrics is published.
"""

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "results/visual_headroom_control"
OUTPUT = ROOT / "results/visual_headroom_metrics"
SEEDS = (42, 7, 13)
VARIANTS = ("free", "hidden", "hidden_matched", "fault", "repair", "uniform", "intact")
FORWARD = ("fault", "repair", "uniform", "intact")
PROTOCOLS = ("single_ref", "heldout", "all_refs", "mismatch_target", "mismatch_grounding")
MEMORY_FRACTION = 0.35
MINIMUM_FREE_MIB = 24000


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    import hashlib
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_metrics(visibility):
    """Restore GPU1 after BOTH legacy imports, before any CUDA initialization."""
    torch = sys.modules.get("torch")
    require(torch is None or not torch.cuda.is_initialized(), "CUDA initialized before import guard")
    os.environ["CUDA_VISIBLE_DEVICES"] = visibility
    try:
        canonical = importlib.import_module("score_semantics")
        cider = importlib.import_module("rescore_clean_cider")
    finally:
        os.environ["CUDA_VISIBLE_DEVICES"] = visibility
    torch = sys.modules.get("torch")
    require(torch is None or not torch.cuda.is_initialized(), "Legacy import initialized CUDA")
    require(Path(canonical.__file__).resolve() == HERE / "score_semantics.py", "Wrong canonical module")
    require(Path(cider.__file__).resolve() == HERE / "rescore_clean_cider.py", "Wrong CIDEr module")
    return canonical, cider


def query_gpu():
    result = subprocess.run(
        ["nvidia-smi", "-i", "1", "--query-gpu=index,uuid,memory.total,memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15)
    lines = result.stdout.strip().splitlines()
    require(len(lines) == 1, "Expected one physical GPU1")
    index, uuid, total, free = [s.strip() for s in lines[0].split(",")]
    require(index == "1" and uuid.startswith("GPU-"), "Wrong physical GPU")
    return dict(physical_gpu=1, uuid=uuid, total_mib=int(total), free_mib=int(free))


def configure_device(torch, gpu):
    require(gpu["physical_gpu"] == 1 and gpu["free_mib"] >= MINIMUM_FREE_MIB,
            "GPU1 requires at least 24000 MiB free before acquisition")
    require(os.environ["CUDA_VISIBLE_DEVICES"] == gpu["uuid"], "GPU visibility was changed")
    require(torch.cuda.device_count() == 1, "Exactly one logical CUDA device required")
    props = torch.cuda.get_device_properties(0)
    uuid = str(props.uuid)
    if not uuid.startswith("GPU-"):
        uuid = "GPU-" + uuid
    require(uuid.lower() == gpu["uuid"].lower(), "Torch/nvidia-smi GPU UUID mismatch")
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, device=0)
    return dict(**gpu, torch_uuid=uuid, logical_device="cuda:0", name=props.name,
                cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
                memory_cap_fraction=MEMORY_FRACTION,
                memory_cap_bytes=int(props.total_memory * MEMORY_FRACTION),
                cap_scope="PyTorch allocator only; CUDA context/driver overhead excluded",
                shared=True, other_processes_signalled=False)


def validate_records(data, original, forward, native, seed, lock_hash):
    records = data["records"]
    require(data.get("partial") is False and data["n_images"] == 1000 and len(records) == 1000,
            "Full 1000-image dump required")
    require(data["variants"] == list(VARIANTS) and data["evaluated_true_slots"] == list(range(5)),
            "All variants and five slots required")
    require(data["n_steps"] == 300 and data["metadata"] == dict(
        seed=seed, k=16, test_lock_sha256=lock_hash,
        repair="reference-free known-mask zero-step plan switch; identical fault X",
        hidden_matched_steps=2, hidden_steps=300, free_steps=300), "Unexpected method metadata")
    require(len(original) == 1000 and set(forward["captions"]) == set(FORWARD), "Forward schema mismatch")
    require(native["image_ids"] == list(range(1000)) and native["n_images"] == 1000,
            "Native image order mismatch")
    require(native["refs"] == [r["refs"] for r in original], "Native reference order mismatch")
    require(all(len(forward["captions"][v]) == 1000 for v in FORWARD), "Forward length mismatch")
    for i, (r, old) in enumerate(zip(records, original)):
        for key in ("image", "refs", "mismatch_source", "mismatch_target"):
            require(r[key] == old[key], f"Original reference/job alignment mismatch: {key}, image {i}")
        require(r["image"] == i and r["feature_index"] == 14000 + i, "Test feature index mismatch")
        require(len(r["refs"]) == 5 and all(isinstance(s, str) for s in r["refs"]), "Invalid references")
        require(r["mismatch_source"] != i and r["mismatch_target"] in original[r["mismatch_source"]]["refs"],
                "Invalid shuffled donor")
        require(r["baseline"] == forward["captions"]["fault"][i], "Fault baseline mismatch")
        require(r["forward"] == {v: forward["captions"][v][i] for v in FORWARD}, "Forward row mismatch")
        require(set(r["variants"]) == set(VARIANTS), "Variant mismatch")
        for v in VARIANTS:
            jobs = r["variants"][v]
            require(len(jobs["true"]) == 5 and all(isinstance(s, str) for s in jobs["true"])
                    and isinstance(jobs["mismatch"], str), "Incomplete target jobs")
            if v in FORWARD:
                require(jobs["true"] == [r["forward"][v]] * 5 and jobs["mismatch"] == r["forward"][v],
                        "Reference-free caption is not repeated identically")
                require(jobs["true_stats"] == [dict(steps=0, reference_free=True, inversion=False)] * 5
                        and jobs["mismatch_stats"] == jobs["true_stats"][0], "Misleading forward step metadata")
            if v == "hidden_matched":
                require(all(s["steps"] == 2 for s in [*jobs["true_stats"], jobs["mismatch_stats"]]),
                        "Wrong matched hidden steps")
    require(sorted(r["mismatch_source"] for r in records) == list(range(1000)), "Donors are not a permutation")
    mismatches = [i for i in range(1000)
                  if forward["captions"]["intact"][i] != native["caps"]["guided_denoised"][i]]
    require(mismatches == forward["export_intact_mismatches"], "Intact/native mismatch receipt differs")
    require(not mismatches, "This sidecar requires exact intact/native caption equality")
    return records, mismatches


def protocol_jobs(records, variant, protocol, baseline="fault"):
    """Keep reference slots, including duplicate strings, aligned for both sides."""
    require(protocol in PROTOCOLS and baseline in ("fault", "intact"), "Unknown comparison")
    predictions, bases, refs = [], [], []
    for r in records:
        jobs = r["variants"][variant]
        for k in range(5) if protocol in PROTOCOLS[:3] else (-1,):
            predictions.append(jobs["true"][k] if k >= 0 else jobs["mismatch"])
            bases.append(r["forward"][baseline])
            refs.append([r["refs"][k]] if protocol == "single_ref" else
                        [s for j, s in enumerate(r["refs"]) if j != k] if protocol == "heldout" else
                        [r["mismatch_target"]] if protocol == "mismatch_target" else list(r["refs"]))
    return predictions, bases, refs


def protocol_values(metric, records, variant, protocol, baseline="fault"):
    preds, bases, refs = protocol_jobs(records, variant, protocol, baseline)
    return tuple(np.asarray(metric(caps, refs), dtype=float).reshape(len(records), -1).mean(1)
                 for caps in (preds, bases))


def collect_pairs(datasets):
    pairs = {}
    for records in datasets.values():
        for v in VARIANTS:
            for p in PROTOCOLS:
                preds, bases, refs = protocol_jobs(records, v, p)
                native = [r["forward"]["intact"] for r in records
                          for _ in (range(5) if p in PROTOCOLS[:3] else range(1))]
                for captions in (preds, bases, native):
                    for cap, rs in zip(captions, refs):
                        for ref in rs:
                            pairs.setdefault((cap, ref), None)
    return list(pairs)


def comparisons_for_seed(metric, records):
    rows = {}
    for p in PROTOCOLS:
        common = {}
        for v in VARIANTS:
            for base in ("fault", "intact"):
                values = protocol_values(metric, records, v, p, base)
                if base in common:
                    np.testing.assert_array_equal(values[1], common[base])
                common[base] = values[1]
                rows[f"protocol/{v}/{p}/vs_{base}"] = values
        rows[f"contrast/free_minus_repair/{p}"] = (
            rows[f"protocol/free/{p}/vs_fault"][0], rows[f"protocol/repair/{p}/vs_fault"][0])
        rows[f"contrast/free_minus_hidden_matched/{p}"] = (
            rows[f"protocol/free/{p}/vs_fault"][0], rows[f"protocol/hidden_matched/{p}/vs_fault"][0])
    refs = [r["refs"] for r in records]
    forward = {v: metric([r["forward"][v] for r in records], refs) for v in FORWARD}
    for v in FORWARD:
        for base in ("fault", "intact"):
            rows[f"forward/{v}/all_refs/vs_{base}"] = (forward[v], forward[base])
    rows["forward/repair/all_refs/vs_uniform"] = (forward["repair"], forward["uniform"])
    return rows


def verify_forward_cider(metric, records, native, old_values, cider):
    refs = [r["refs"] for r in records]
    caps = [r["forward"]["intact"] for r in records]
    values = metric.score(caps, refs)
    standard = cider.Cider().compute_score(
        {i: [cider.norm(r) for r in rs] for i, rs in enumerate(refs)},
        {i: [cider.norm(c) or "empty"] for i, c in enumerate(caps)})[1]
    standard_error = float(np.max(np.abs(values - standard)))
    require(standard_error <= 1e-12, "Full-reference fixed IDF differs from conventional forward CIDEr")
    pure = caps == native["caps"]["guided_denoised"]
    original_error = float(np.max(np.abs(values - old_values))) if pure else None
    require(not pure or original_error <= 1e-12, "Pure intact differs from original clean CIDEr forward")
    return dict(pure_intact_caption_equality=pure, standard_forward_max_abs_error=standard_error,
                original_clean_cider_max_abs_error=original_error)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=time.strftime("run_%Y%m%d_%H%M%S"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    require(args.batch_size > 0 and Path(args.run_name).name == args.run_name, "Invalid run arguments")
    out = OUTPUT / args.run_name
    require(not out.exists(), "Never overwrite an existing run")
    out.mkdir(parents=True)
    started = time.monotonic()
    inventory = {}

    def track(path, expected=None):
        path = Path(path).resolve()
        digest = sha256(path)
        require(expected is None or digest == expected, f"Existing hash receipt mismatch: {path}")
        require(str(path) not in inventory or inventory[str(path)] == digest, f"Changed input: {path}")
        inventory[str(path)] = digest
        return path

    def load(path, expected=None):
        with track(path, expected).open() as stream:
            return json.load(stream)

    print("Auditing every existing artifact and upstream input hash", flush=True)
    artifacts = load(SOURCE / "artifact_hashes.json")
    for relative, digest in artifacts.items():
        track(SOURCE / relative, digest)
    upstream = load(SOURCE / "input_hashes.json")
    for path, digest in upstream.items():
        track(path, digest)
    for name in ("score_visual_headroom.py", "test_score_visual_headroom.py", "score_semantics.py",
                 "rescore_clean_cider.py", "rescore_rouge.py"):
        track(HERE / name)
    original = load(ROOT / "results/revision/inversion_protocol_n1000_s0.json")["records"]
    require([r["refs"] for r in original] == load(ROOT / "results/revision/coco_caps_15k.json")[14000:15000],
            "Original case/punctuation reference source mismatch")
    lock = load(SOURCE / "test_lock.json")
    validation = load(SOURCE / "validation_lock.json", lock["validation_lock_sha256"])
    plan = load(SOURCE / "local_plan.json", validation["local_plan_sha256"])
    require(lock["chosen_k"] == validation["chosen_k"] == 16 and lock["test_selection"] is False
            and validation["selection_uses_test"] is False and plan["external_preregistration"] is False,
            "Unexpected local validation selection")
    track(SOURCE / "input_hashes.json", lock["input_hashes_sha256"])
    require(load(SOURCE / "independent_feature_provenance.json")["all_checks_passed"], "Feature audit failed")
    old_cider = load(HERE / "clean_cider_results.json")
    datasets, native_forwards, audits = {}, {}, {}
    for seed in SEEDS:
        require(lock["hidden_steps"][str(seed)] == 2, "Changed validation matched steps")
        track(SOURCE / f"seed{seed}/calibration/results.json", lock["calibration_hashes"][str(seed)])
        track(SOURCE / f"seed{seed}/validation/results.json", validation["validation_hashes"][str(seed)])
        done = load(SOURCE / f"seed{seed}/test/complete.json")
        require(done["all_parameters_frozen"] and done["gradients_none"]
                and done["state_sha256_before"] == done["state_sha256_after"], "Incomplete source run")
        native_path = ROOT / f"results/clean_eval_stablekl/seed{seed}/forward.json"
        native = load(native_path, old_cider["source_sha256"][native_path.relative_to(ROOT).as_posix()])
        data = load(SOURCE / f"seed{seed}/test/inversion_protocol_n1000_s0.json")
        forward = load(SOURCE / f"seed{seed}/test/forward.json")
        records, mismatches = validate_records(data, original, forward, native, seed, sha256(SOURCE / "test_lock.json"))
        datasets[seed], native_forwards[seed] = records, native
        audits[str(seed)] = dict(intact_native_caption_mismatches=mismatches)
    print("All seeds: 1000 aligned images, five original reference slots, all seven variants", flush=True)

    gpu = query_gpu()
    require(gpu["free_mib"] >= MINIMUM_FREE_MIB, "Insufficient shared GPU1 memory")
    canonical, cider = load_metrics(gpu["uuid"])
    import torch
    import bert_score.utils
    device = configure_device(torch, gpu)
    for path in canonical.SNAPSHOT.iterdir():
        if path.is_file():
            track(path)
    track(Path(bert_score.utils.__file__))
    track(Path(importlib.import_module("bert_score.scorer").__file__))
    track(Path(importlib.import_module("pycocoevalcap.cider.cider_scorer").__file__))
    metric = cider.FixedIDFCider([r["refs"] for r in original])
    df_rows = [[list(gram), count] for gram, count in sorted(metric.document_frequency.items())]
    import hashlib
    df_hash = hashlib.sha256(json.dumps(df_rows, separators=(",", ":")).encode()).hexdigest()
    require(metric.n_documents == 1000 and df_hash == old_cider["metric"]["document_frequency_sha256"],
            "IDF differs from original clean_cider_results")
    write(out / "local_config.json", dict(
        seeds=list(SEEDS), n_images=1000, target_slots=list(range(5)), variants=list(VARIANTS),
        external_preregistration=False, analysis="Local post-hoc metric extension; no outcome selection",
        gpu=device, batch_size=args.batch_size, source_hashes=inventory,
        steps=dict(free=300, hidden=300, hidden_matched=2, fault=0, repair=0, uniform=0, intact=0)))

    scorer, repaired = canonical.make_scorer()
    require(query_gpu()["uuid"] == gpu["uuid"], "GPU1 identity changed")
    smoke = canonical.smoke(scorer)
    pairs = collect_pairs(datasets)
    texts = set(s for pair in pairs for s in pair)
    tokenization = canonical.audit_tokenization(scorer._tokenizer, texts)
    print(f"BERTScore: {len(pairs)} exact-text pairs, {len(texts)} strings; GPU1 UUID {gpu['uuid']}", flush=True)
    tick = time.monotonic()
    with torch.inference_mode():
        _, _, f1 = scorer.score([p[0] for p in pairs], [p[1] for p in pairs],
                                batch_size=args.batch_size, verbose=True)
    pair_values = f1.tolist()
    require(len(pair_values) == len(pairs) and np.isfinite(pair_values).all(), "Nonfinite BERTScore")
    require(min(pair_values) >= -1.00001 and max(pair_values) <= 1.00001, "BERTScore outside raw cosine range")
    bert_seconds = time.monotonic() - tick
    scores = dict(zip(pairs, pair_values))
    write(out / "bert_pair_cache.json", dict(pairs=pairs, f1=pair_values, dtype="float32, no quantization or rounding",
                                            source_hashes=inventory))
    device.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(0), gpu_after_scoring=query_gpu())
    require(device["gpu_after_scoring"]["uuid"] == gpu["uuid"], "GPU UUID changed")
    require(device["peak_reserved_bytes"] <= device["memory_cap_bytes"], "Allocator cap exceeded")
    del scorer, f1
    torch.cuda.empty_cache()

    def bert(caps, refs):
        return np.array([max(scores[c, r] for r in rs) for c, rs in zip(caps, refs)])

    rows = {}
    ranges = {}
    for seed in SEEDS:
        print(f"Scoring/bootstrapping seed {seed}: BERTScore and CPU fixed-IDF CIDEr", flush=True)
        records = datasets[seed]
        audits[str(seed)].update(verify_forward_cider(
            metric, records, native_forwards[seed], np.array(old_cider["comparisons"][
                "forward/cider_fixed_idf/guided_denoised"]["per_seed"][str(seed)]["per_image"]["inv"]), cider))
        saved = {}
        for name, evaluate in (("bertscore", bert), ("cider_fixed_idf", metric.score)):
            values = comparisons_for_seed(evaluate, records)
            flat = np.concatenate([a for pair in values.values() for a in pair])
            require(np.isfinite(flat).all(), f"Nonfinite {name} image scores")
            require(flat.min() >= (-1.00001 if name == "bertscore" else -1e-12)
                    and flat.max() <= (1.00001 if name == "bertscore" else 10.00000001), "Metric range failed")
            ranges[f"{name}/seed{seed}"] = dict(min=float(flat.min()), max=float(flat.max()), finite=True)
            for key, pair in values.items():
                full_key = name + "/" + key
                rows.setdefault(full_key, {})[seed] = pair
                saved[full_key] = cider.paired(*pair)
        write(out / f"seed{seed}.json", dict(seed=seed, image_ids=list(range(1000)), comparisons=saved,
                                           verification=audits[str(seed)]))
    print("Pooling five-job image means across three seeds, then 10000 image bootstraps", flush=True)
    pooled = {key: cider.paired(np.mean([by_seed[s][0] for s in SEEDS], axis=0),
                               np.mean([by_seed[s][1] for s in SEEDS], axis=0))
              for key, by_seed in rows.items()}
    for key, row in pooled.items():
        row["seed_deltas"] = {str(s): float(np.mean(rows[key][s][0] - rows[key][s][1])) for s in SEEDS}
    original_results = load(SOURCE / "results.json")
    rouge_context = dict(
        free_heldout_gain=original_results["pooled"]["free"]["heldout"]["delta"],
        repair_heldout_gain=original_results["pooled"]["repair"]["heldout"]["delta"],
        repair_forward_allfive_gain=load(SOURCE / "independent_verification.json")["forward_contrasts"][
            "repair_minus_fault"]["mean"])
    # Recheck ALL inputs, including models, features, means and validation selection.
    print("Rechecking all source/model/mean/package hashes before publication", flush=True)
    for path, digest in inventory.items():
        require(sha256(path) == digest, f"Input changed during scoring: {path}")
    result = dict(
        schema_version=1, status="complete_hash_verified", seeds=list(SEEDS), n_images=1000,
        image_ids=list(range(1000)), feature_indices=list(range(14000, 15000)),
        metadata=dict(
            analysis="Local post-hoc configuration, not external preregistration", training=False,
            caption_generation=False, metric_inference_only=True, no_other_agent_posthoc_captions_scored=True,
            device=device, versions={p: importlib.metadata.version(p) for p in
                ("torch", "bert-score", "transformers", "tokenizers", "numpy", "pycocoevalcap")},
            bertscore=dict(model="cached roberta-large", snapshot=str(canonical.SNAPSHOT), num_layers=17,
                idf=False, rescale_with_baseline=False, all_layers=False, dtype="float32", autocast=False,
                tf32=False, eval=True, canonical_make_scorer_calls=1,
                normalization="Original caption/reference case and punctuation; canonical sent_encode strip/prefix space",
                tokenizer_prefix_fix=repaired, embedding_cache="Upstream bert_cos_score_idf CPU FP32, one combined call",
                multireference="max raw F1 over eligible reference slots", pairs=len(pairs), strings=len(texts),
                pair_min=min(pair_values), pair_max=max(pair_values)),
            cider=dict(name="CIDEr-D", device="cpu", n=4, sigma=6.0, scale="raw x10, NOT x100",
                n_documents=1000, reference_slots=5000, document_frequency_sha256=df_hash,
                normalization="Canonical rescore_clean_cider.norm; empty hypothesis -> empty; no PTB",
                multireference="Upstream joint CIDEr-D reference mean, NOT BERTScore-style max",
                idf="Fixed once from original 1000 test image documents, all five refs; identical to clean_cider_results",
                leakage="Excluded target still enters unsupervised frozen IDF, never similarity reference set"),
            steps=dict(free=300, hidden=300, hidden_matched=2, fault=0, repair=0, uniform=0, intact=0),
            repair="Reference-free known-mask plan switch; one saved caption repeated per slot, no reference optimization",
            shared_features="Free/hidden/hidden_matched/fault/repair/uniform use the same per-seed fault X and train-only projected mean mu; intact uses uncorrupted true-image X and native plans",
            baselines="Every protocol/forward system row has vs_fault (own fault forward) and vs_intact (native true-image forward). Both use exactly the same evaluation refs as the output; mismatch_target uses the saved donor target for both sides.",
            definitions=dict(single_ref="mean_k score(y_k,[r_k])", heldout="mean_k score(y_k,R excluding slot k); same exclusion for baseline; duplicate strings retained",
                all_refs="mean_k score(y_k,R5); not best job selection", mismatch_target="One shuffled-target job against saved donor caption",
                mismatch_grounding="One shuffled-target job against own true-image R5", forward="One reference-free caption against own R5"),
            bootstrap="10000 paired percentile image resamples; numpy.default_rng(0), reset each comparison; mean 5 jobs then mean 3 seeds then resample 1000 images; shuffle/forward one job; conditional on fixed trained seeds, no multiplicity correction",
            precision="JSON full float serialization; no quantization or rounded writes",
            limitations=["Text-reference similarity is not image grounding or factuality.",
                         "Constructed feature-space fault, not a natural-photo corruption or ordinary predictor deficit.",
                         "Validation-selected K=16 and 2-step hidden matching reused validation, not test; no semantic rematching.",
                         "No new visual-provenance captions from concurrent agents consumed."]),
        verification=dict(artifact_manifest_entries=len(artifacts), upstream_hash_entries=len(upstream),
            all_input_hashes_unchanged=True, tokenization=tokenization, smoke=smoke, seeds=audits,
            actual_image_score_ranges=ranges, same_baselines_across_variants=True, original_rouge_context=rouge_context),
        source_sha256=inventory, timings=dict(bertscore_seconds=bert_seconds, total_seconds=time.monotonic() - started),
        comparisons=pooled)
    write(out / "results.json", result)
    write(out / "artifact_hashes.json", {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()})
    print("Wrote", out, flush=True)
    for name in ("bertscore", "cider_fixed_idf"):
        for v in ("free", "hidden", "hidden_matched", "repair"):
            for base in ("fault", "intact"):
                key = f"{name}/protocol/{v}/heldout/vs_{base}"
                row = pooled[key]
                print(key, {k: row[k] for k in ("inv", "base", "delta", "ci95", "seed_deltas")}, flush=True)
    return result


if __name__ == "__main__":
    main()
