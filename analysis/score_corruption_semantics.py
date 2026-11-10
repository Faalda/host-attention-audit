"""Exploratory offline corruption BERTScore sidecar; never edits legacy artifacts.

--execute launches independent GPU workers, preferring GPU1 for seeds42/13 and
GPU0 for seed7 (120-second acquisition limit, then GPU1 fallback). --check
reaggregates saved pair scores and compares JSON bytes, without CUDA or inference.
--allow-shared explicitly permits other jobs after a 24000 MiB free-memory check;
our per-GPU cooperative locks and 35% PyTorch allocator cap remain in force.
"""

import argparse
from contextlib import contextmanager
import gc
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "clean_pipeline"))  # bare imports below match the clean_pipeline/
                                                   # test suite's own module identity (avoid a
                                                   # second, separately-patched module instance)
from clean_gpu_guard import exclusive_lock, LOCK_PATH, GPUQueryError
from run_clean_pipeline import (EVAL_ARTIFACTS, fingerprint, read_json, sha256,
                                verify_hashes, write_json)
from run_clean_semantics import (SNAPSHOT, SNAPSHOT_FILES, PACKAGES, prerequisites,
                                 validate_inputs)

HERE = Path(__file__).resolve().parent
EVAL = ROOT / "results/clean_eval_stablekl"
SEEDS = (42, 7, 13)
FRACTIONS = ("0", "0.25", "0.5", "1")
VARIANTS = ("free", "onehot", "hidden")
MEMORY_FRACTION = 0.35
OOM_BATCH_SIZES = (128, 64, 32)
CODE = ("analysis/score_corruption_semantics.py",
        "analysis/test_score_corruption_semantics.py", "analysis/score_semantics.py",
        "analysis/rescore_rouge.py", "clean_pipeline/run_clean_semantics.py",
        "clean_pipeline/run_clean_pipeline.py", "clean_pipeline/clean_gpu_guard.py")
CONFIG = {
    "metric": "canonical bert-score 0.3.13 raw F1", "model": "roberta-large",
    "snapshot_revision": SNAPSHOT.name, "num_layers": 17, "idf": False,
    "rescale_with_baseline": False, "all_layers": False, "dtype": "float32",
    "autocast": False, "tf32": False, "seeds": list(SEEDS),
    "fractions": list(FRACTIONS), "n_images": 1000, "target_slots": [0, 1],
    "reference_slots": [0, 1, 2, 3, 4],
    "selection": "All fractions, all images, all seeds; no outcome selection or subset fallback",
    "analysis_status": "Exploratory metric-specific extension after lexical results; not prespecified",
    "bootstrap": {"resamples": 10000, "seed": 0, "unit": "image",
                  "aggregation": "Average three seeds within each aligned image before resampling 1000 images",
                  "interpretation": "Conditional on these trained seeds; not training-seed uncertainty; not 3000 IID rows"},
    "definitions": {
        "single_ref": "mean over k=0,1 of F1(output_k, ref_k); identical baseline reference slots",
        "heldout": "mean over k=0,1 of max_{j!=k} F1(output_k, ref_j); baseline excludes same slot k, not matching strings",
        "all_refs": "mean over k=0,1 of max_j F1(output_k, ref_j); baseline max over all five slots",
        "mismatch_target": "F1(shuffle output, saved donor target), versus same baseline target",
        "mismatch_grounding": "max_j F1(shuffle output, own ref_j), versus own baseline max",
        "repair": "Restore ordinary trained plans; no target/reference used; same saved clean caption for both k and shuffle",
        "forward_restoration_all_refs": "Ordinary clean forward caption minus corrupted forward caption, each max over all five references",
        "free_minus_repair_heldout": "Free minus repair on exactly identical heldout reference slots",
        "baseline_loss_heldout": "Clean minus corrupted baseline; identically repair heldout gain",
        "free_gain_change_from_zero": "Free heldout gain at fraction minus free heldout gain at zero",
        "strict_k01": "Clean primary protocol restricted to target slots 0,1 for all three variants; not best5"},
    "limitations": ["Text-reference similarity does not establish image grounding or factual accuracy.",
                    "Raw BERTScore has a high similarity floor; no multiplicity correction.",
                    "A positive semantic repair gain is metric-specific evidence, not proof of the method."],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def query_gpu(index):
    """Fail closed on missing identity/telemetry, including root and graphics PIDs."""
    require(index in (0, 1), "Only explicitly authorized physical GPUs 0 and 1")
    try:
        result = subprocess.run(["nvidia-smi", "-i", str(index), "-q", "-x"],
                                check=True, capture_output=True, text=True, timeout=15)
        root = ET.fromstring(result.stdout)
        require(root.tag == "nvidia_smi_log" and len(root.findall("gpu")) == 1,
                "Expected one index-restricted GPU")
        gpu = root.find("gpu")
        uuid = gpu.findtext("uuid", "").strip()
        require(re.fullmatch(r"GPU-[A-Za-z0-9-]+", uuid), "Missing GPU UUID")
        free = re.fullmatch(r"\s*(\d+)\s+MiB\s*", gpu.findtext("fb_memory_usage/free", ""))
        util = re.fullmatch(r"\s*(\d+)\s*%\s*", gpu.findtext("utilization/gpu_util", ""))
        require(free and util and 0 <= int(util[1]) <= 100, "Missing GPU telemetry")
        tables = gpu.findall(".//processes")
        require(tables, "Missing process table")
        for table in tables:
            require((table.text or "").strip() in ("", "None")
                    and all(p.tag == "process_info" for p in table), "Unknown process table")
        processes = []
        for p in gpu.findall(".//process_info"):
            pid = int(p.findtext("pid", ""))
            require(pid > 0, "Invalid PID")
            processes.append({"pid": pid, "type": p.findtext("type", "unknown"),
                              "name": p.findtext("process_name", "unknown")})
        return {"index": index, "gpu_uuid": uuid, "free_mib": int(free[1]),
                "utilization_percent": int(util[1]), "processes": processes}
    except (OSError, subprocess.SubprocessError, ET.ParseError, ValueError) as exc:
        raise GPUQueryError(f"GPU {index} query failed closed: {exc}") from exc


@contextmanager
def gpu_lease(index, timeout=120, allow_shared=False):
    deadline = time.monotonic() + timeout
    first = True
    while True:
        if not first and time.monotonic() >= deadline:
            raise TimeoutError(f"GPU {index} not acquired within {timeout}s: {reason}")
        first = False
        lock = exclusive_lock(LOCK_PATH.with_name(f"gpu{index}.lock"))
        try:
            lock.__enter__()
        except BlockingIOError:
            reason = "cooperative lock held"
        else:
            try:
                snapshot = query_gpu(index)
                minimum_free = 24000 if allow_shared else 20000
                reason = ("GPU unavailable: " + json.dumps(snapshot)
                          if snapshot["free_mib"] < minimum_free or (not allow_shared and
                          (snapshot["processes"] or snapshot["utilization_percent"] > 5)) else None)
                if reason is None:
                    snapshot.update(allow_shared=allow_shared, minimum_free_mib=minimum_free,
                                    existing_pids=[p["pid"] for p in snapshot["processes"]])
                    # Monitor unrelated schedulers without signalling any process.
                    conflicts, stop = [], threading.Event()

                    def monitor():
                        while not stop.wait(15):
                            try:
                                now = query_gpu(index)
                                require(now["gpu_uuid"] == snapshot["gpu_uuid"], "GPU identity changed")
                                require(allow_shared or not [p for p in now["processes"] if p["pid"] != os.getpid()],
                                        "External GPU process appeared")
                            except (GPUQueryError, ValueError) as exc:
                                conflicts.append(str(exc))
                                print(str(exc), file=sys.stderr, flush=True)

                    thread = threading.Thread(target=monitor, daemon=True)
                    thread.start()
                    try:
                        yield snapshot, conflicts
                    finally:
                        stop.set()
                        thread.join()
                    return
            finally:
                lock.__exit__(None, None, None)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"GPU {index} not acquired within {timeout}s: {reason}")
        print(f"GPU {index} waiting: {reason}", flush=True)
        time.sleep(min(5, remaining))


def load_canonical(visibility):
    """Undo legacy import's GPU0 assignment before any torch CUDA initialization."""
    require(visibility.startswith("GPU-"), "Use the leased physical GPU UUID")
    torch = sys.modules.get("torch")
    require(torch is None or not torch.cuda.is_initialized(), "CUDA already initialized before canonical import")
    os.environ["CUDA_VISIBLE_DEVICES"] = visibility
    sys.path.insert(0, str(HERE))
    try:
        canonical = importlib.import_module("score_semantics")
    finally:
        os.environ["CUDA_VISIBLE_DEVICES"] = visibility
    require(Path(canonical.__file__).resolve() == HERE / "score_semantics.py", "Wrong canonical module")
    torch = sys.modules.get("torch")
    require(torch is None or not torch.cuda.is_initialized(), "Canonical import initialized CUDA")
    return canonical


def actual_device(torch, lease):
    require(torch.cuda.device_count() == 1, "Exactly one logical GPU required")
    props = torch.cuda.get_device_properties(0)
    uuid = str(props.uuid)
    if not uuid.startswith("GPU-"):
        uuid = "GPU-" + uuid
    require(uuid.lower() == lease["gpu_uuid"].lower(), "Actual torch GPU UUID differs from nvidia-smi lease")
    # Limit our allocator before make_scorer moves the model onto this GPU.
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, device=0)
    return {"physical_gpu": lease["index"], "nvidia_smi_uuid": lease["gpu_uuid"],
            "torch_device_uuid": uuid, "logical_device": "cuda:0", "gpu_name": props.name,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "memory_cap_fraction": MEMORY_FRACTION,
            "memory_cap_bytes": int(props.total_memory * MEMORY_FRACTION),
            "memory_cap_scope": "PyTorch CUDA caching allocator; excludes CUDA context/driver overhead"}


def validate_corruption(data, original, fraction):
    require(data.get("partial") is False and data["n_images"] == 1000
            and data["evaluated_true_slots"] == [0, 1] and data["variants"] == ["free", "repair"]
            and data["metadata"]["corruption_fraction"] == float(fraction), "Incomplete corruption schema")
    records = data["records"]
    require(len(original) == 1000 and [r["image"] for r in records] == list(range(1000)), "Image alignment")
    for record, clean in zip(records, original):
        for key in ("image", "feature_index", "refs", "mismatch_source", "mismatch_target"):
            require(record[key] == clean[key], f"Corruption alignment: {key}")
        require(isinstance(record["baseline"], str) and record["clean_baseline"] == clean["baseline"],
                "Invalid clean baseline")
        require(set(record["variants"]) == {"free", "repair"}, "Wrong variants")
        for jobs in record["variants"].values():
            require(len(jobs["true"]) == 5 and all(isinstance(c, str) for c in jobs["true"][:2])
                    and jobs["true"][2:] == [None] * 3 and isinstance(jobs["mismatch"], str), "Invalid k01 jobs")
        repair = record["variants"]["repair"]
        require(repair["true"][:2] == [clean["baseline"]] * 2
                and repair["mismatch"] == clean["baseline"], "Repair is not ordinary reference-free restoration")
        require(fraction != "0" or record["baseline"] == clean["baseline"], "Zero baseline changed")
    return records


def load_inputs(seed):
    folder = EVAL / f"seed{seed}"
    receipt = read_json(folder / "pipeline_eval_receipt.json")
    require(set(EVAL_ARTIFACTS) <= receipt["hashes"].keys(), "Incomplete pipeline receipt")
    verify_hashes(receipt["hashes"], folder)
    forward = read_json(folder / "forward.json")
    primary = validate_inputs(read_json(folder / "inversion_protocol_n1000_s0.json"), forward)
    datasets = {f"corruption/{f}": validate_corruption(read_json(folder / f"corruption_{f}.json"), primary, f)
                for f in FRACTIONS}
    datasets["strict_k01"] = primary
    inputs = {**receipt["hashes"], "pipeline_eval_receipt.json": sha256(folder / "pipeline_eval_receipt.json")}
    return datasets, inputs


def image_scores(record, variant, score):
    """Two true jobs per image; exclude reference by slot, preserving duplicates."""
    refs, jobs = record["refs"], record["variants"][variant]
    base = [score(record["baseline"], ref) for ref in refs]
    mat = [[score(jobs["true"][k], ref) for ref in refs] for k in (0, 1)]
    return {
        "single_ref": (sum(mat[k][k] for k in (0, 1)) / 2, (base[0] + base[1]) / 2),
        "heldout": (sum(max(mat[k][j] for j in range(5) if j != k) for k in (0, 1)) / 2,
                    sum(max(base[j] for j in range(5) if j != k) for k in (0, 1)) / 2),
        "all_refs": (sum(map(max, mat)) / 2, max(base)),
        "mismatch_target": (score(jobs["mismatch"], record["mismatch_target"]),
                            score(record["baseline"], record["mismatch_target"])),
        "mismatch_grounding": (max(score(jobs["mismatch"], ref) for ref in refs), max(base)),
    }


def collect_pairs(datasets):
    pairs = {}

    def collect(a, b):
        pairs.setdefault((a, b), None)
        return 0.0

    for name, records in datasets.items():
        for record in records:
            for variant in VARIANTS if name == "strict_k01" else ("free", "repair"):
                image_scores(record, variant, collect)
    return list(pairs)


def score_pairs(scorer, pairs, batch_size):
    # One upstream call deduplicates strings and caches canonical FP32 embeddings.
    _, _, f1 = scorer.score([p[0] for p in pairs], [p[1] for p in pairs],
                            batch_size=batch_size, verbose=True)
    values = f1.tolist()
    require(len(values) == len(pairs) and all(math.isfinite(v) for v in values), "Invalid pair scores")
    return values


def score_with_retries(torch, scorer, pairs, batch_size):
    require(batch_size in OOM_BATCH_SIZES, "Batch size must be one of 128, 64, 32")
    attempts = []
    for size in OOM_BATCH_SIZES[OOM_BATCH_SIZES.index(batch_size):]:
        try:
            values = score_pairs(scorer, pairs, size)
        except torch.cuda.OutOfMemoryError:
            attempts.append({"batch_size": size, "status": "cuda_oom"})
            if size == OOM_BATCH_SIZES[-1]:
                raise
        else:
            attempts.append({"batch_size": size, "status": "complete"})
            return values, size, attempts
        # Leave the exception scope before clearing allocations held by traceback.
        gc.collect()
        torch.cuda.empty_cache()
        print(f"OOM at batch {size}; retrying the SAME complete ordered pair list", flush=True)


def aggregate_seed(datasets, pairs, values):
    from analysis.rescore_rouge import paired

    scores = dict(zip(pairs, values))
    rows = {}
    for name, records in datasets.items():
        for variant in VARIANTS if name == "strict_k01" else ("free", "repair"):
            images = [image_scores(r, variant, lambda a, b: scores[a, b]) for r in records]
            for protocol in images[0]:
                rows[f"{name}/{variant}/{protocol}"] = paired(
                    [r[protocol][0] for r in images], [r[protocol][1] for r in images])
        if name == "strict_k01":
            continue
        free = rows[f"{name}/free/heldout"]["per_image"]
        repair = rows[f"{name}/repair/heldout"]["per_image"]
        zero = rows["corruption/0/free/heldout"]["per_image"]
        require(free["base"] == repair["base"] and repair["inv"] == zero["base"], "Unmatched repair reference slots")
        rows[f"{name}/forward_restoration_all_refs"] = rows[f"{name}/repair/all_refs"]
        rows[f"{name}/free_minus_repair_heldout"] = paired(free["inv"], repair["inv"])
        rows[f"{name}/baseline_loss_heldout"] = paired(zero["base"], free["base"])
        rows[f"{name}/free_gain_change_from_zero"] = paired(
            [a - b for a, b in zip(free["inv"], free["base"])],
            [a - b for a, b in zip(zero["inv"], zero["base"])])
    return rows


def provenance():
    distribution = importlib.metadata.distribution("bert-score")
    return {"code_sha256": {name: sha256(ROOT / name) for name in CODE},
            "snapshot_sha256": {name: sha256(SNAPSHOT / name) for name in SNAPSHOT_FILES},
            "canonical_sha256": {name: sha256(distribution.locate_file(name))
                                 for name in ("bert_score/utils.py", "bert_score/scorer.py")},
            "versions": {name: importlib.metadata.version(name) for name in PACKAGES}}


def seed_path(seed):
    return HERE / f"corruption_semantic_seed{seed}.json"


def worker(index, seeds, batch_size, allow_shared=False):
    errors = prerequisites()
    require(not errors, str(errors))
    require(batch_size in OOM_BATCH_SIZES and set(seeds) <= set(SEEDS), "Invalid worker arguments")
    # Validate hashes and shapes before attempting to initialize CUDA.
    sources = {seed: load_inputs(seed) for seed in seeds}
    require(not any(seed_path(seed).exists() for seed in seeds), "Refusing to overwrite existing seed outputs")
    prov = provenance()
    with gpu_lease(index, allow_shared=allow_shared) as (lease, conflicts):
        canonical = load_canonical(lease["gpu_uuid"])
        import torch

        device = actual_device(torch, lease)
        model_start = time.monotonic()
        scorer, repaired = canonical.make_scorer(snapshot=SNAPSHOT)
        scorer._model.eval().requires_grad_(False)
        require(str(next(scorer._model.parameters()).device) == "cuda:0", "Scorer on wrong logical device")
        model_seconds = time.monotonic() - model_start
        with torch.inference_mode():
            smoke = canonical.smoke(scorer)
            require(smoke["passed"], "Canonical smoke failed")
            for seed, (datasets, inputs) in sources.items():
                with exclusive_lock(HERE / f"corruption_semantic_seed{seed}.lock"):
                    started = time.monotonic()
                    pairs = collect_pairs(datasets)
                    texts = {t for pair in pairs for t in pair}
                    audit = canonical.audit_tokenization(scorer._tokenizer, texts, snapshot=SNAPSHOT)
                    require(audit["passed"], "Tokenization audit failed")
                    print(f"GPU{index} seed{seed}: {len(pairs)} pairs, {len(texts)} strings", flush=True)
                    values, actual_batch, attempts = score_with_retries(torch, scorer, pairs, batch_size)
                    rows = aggregate_seed(datasets, pairs, values)
                    require(not conflicts, f"GPU lease conflicts: {conflicts}")
                    final_gpu = query_gpu(index)
                    require(final_gpu["gpu_uuid"] == device["torch_device_uuid"]
                            and (allow_shared or not [p for p in final_gpu["processes"] if p["pid"] != os.getpid()]),
                            "GPU conflict at publication")
                    require(not any(m.training for m in scorer._model.modules())
                            and not any(p.requires_grad or p.grad is not None for p in scorer._model.parameters()), "Unfrozen model")
                    verify_hashes(inputs, EVAL / f"seed{seed}")
                    require(provenance() == prov, "Scorer/model provenance changed")
                    result = {"schema_version": 1, "status": "complete", "seed": seed, "config": CONFIG,
                              "input_sha256": inputs, "provenance": prov, "image_ids": list(range(1000)),
                              "ordered_pairs_sha256": fingerprint(pairs), "pair_f1": values,
                              "comparisons": rows, "unique_pairs": len(pairs), "unique_strings": len(texts),
                              "verification": {"source_hashes_unchanged": True, "smoke": smoke,
                                               "tokenization_audit": audit, "prefix_space_compatibility_fix": repaired,
                                               "offline": True, "model_downloads": False, "training": False,
                                               "model_frozen": True},
                              "execution": {**device, "pid": os.getpid(), "seed_runtime_seconds": time.monotonic() - started,
                                            "worker_model_load_seconds": model_seconds, "batch_size": actual_batch,
                                            "requested_batch_size": batch_size, "oom_batch_schedule": list(OOM_BATCH_SIZES),
                                            "scoring_attempts": attempts, "allow_shared": allow_shared,
                                            "gpu_preflight": lease, "gpu_at_publication": final_gpu,
                                            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(0)}}
                    write_json(seed_path(seed), result)
                    print(f"Published {seed_path(seed)}", flush=True)


def summary(check=False):
    import numpy as np
    from analysis.rescore_rouge import paired

    results, prov, references = {}, provenance(), None
    for seed in SEEDS:
        result = read_json(seed_path(seed))
        datasets, inputs = load_inputs(seed)
        current_refs = [r["refs"] for r in datasets["strict_k01"]]
        require(references is None or references == current_refs, "Reference slots differ across training seeds")
        references = current_refs
        pairs = collect_pairs(datasets)
        require(result["status"] == "complete" and result["config"] == CONFIG
                and result["provenance"] == prov and result["input_sha256"] == inputs
                and result["seed"] == seed and result["image_ids"] == list(range(1000))
                and result["ordered_pairs_sha256"] == fingerprint(pairs)
                and len(result["pair_f1"]) == len(pairs)
                and all(math.isfinite(v) for v in result["pair_f1"]), "Invalid seed result")
        recomputed = aggregate_seed(datasets, pairs, result["pair_f1"])
        require(json.dumps(recomputed, allow_nan=False) == json.dumps(result["comparisons"], allow_nan=False),
                "Seed summary byte check failed")
        results[seed] = result
    comparisons = {}
    for name in results[42]["comparisons"]:
        rows = [results[seed]["comparisons"][name] for seed in SEEDS]
        row = paired(np.array([r["per_image"]["inv"] for r in rows]).mean(0),
                     np.array([r["per_image"]["base"] for r in rows]).mean(0))
        row["per_seed"] = {str(seed): {k: v for k, v in r.items() if k != "per_image"}
                           for seed, r in zip(SEEDS, rows)}
        row["seed_delta_minmax"] = [min(r["delta"] for r in rows), max(r["delta"] for r in rows)]
        comparisons[name] = row
    result = {"schema_version": 1, "status": "complete", "config": CONFIG,
              "provenance": prov, "image_ids": list(range(1000)), "comparisons": comparisons,
              "seed_scientific_sha256": {str(seed): fingerprint({k: v for k, v in r.items() if k != "execution"})
                                         for seed, r in results.items()},
              "worker_execution": {str(seed): r["execution"] for seed, r in results.items()},
              "verification": {"per_seed_reaggregation_byte_equal": True,
                               "scientific_hash_excludes_gpu_and_timing_metadata": True}}
    output = HERE / "corruption_semantic_results.json"
    if check:
        require(output.read_bytes() == (json.dumps(result, indent=2, allow_nan=False) + "\n").encode(),
                "Aggregate JSON bytes differ")
        print("Exact aggregate JSON byte check passed; all seed pair-score reaggregations passed", flush=True)
    else:
        write_json(output, result)
    for fraction in FRACTIONS:
        for protocol in ("forward_restoration_all_refs", "repair/heldout", "free/heldout", "free_minus_repair_heldout"):
            row = comparisons[f"corruption/{fraction}/{protocol}"]
            print(f"fraction={fraction} {protocol}: delta={row['delta']:.9f} CI={row['ci95']} seeds={row['seed_delta_minmax']}")


def execute(batch_size, allow_shared=False):
    require("torch" not in sys.modules, "Parent must not import torch")
    require(not (HERE / "corruption_semantic_results.json").exists()
            and not any(seed_path(s).exists() for s in SEEDS), "Refusing output overwrite")
    children, logs = [], []

    def launch(gpu, seeds):
        log = (HERE / f"corruption_semantic_gpu{gpu}_{'-'.join(map(str, seeds))}_{os.getpid()}.log").open("x")
        logs.append(log)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONHASHSEED": "0",
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
               "TOKENIZERS_PARALLELISM": "false"}
        child = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "--worker",
                                  "--gpu", str(gpu), "--seeds", *map(str, seeds),
                                  "--batch-size", str(batch_size), *(["--allow-shared"] if allow_shared else [])],
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append(child)
        print(f"Worker PID={child.pid} GPU={gpu} seeds={seeds}", flush=True)
        return child

    try:
        first, second = launch(1, [42, 13]), launch(0, [7])
        code0 = second.wait()
        code1 = first.wait()
        if code1 == 75:
            raise TimeoutError(f"GPU1 acquisition failed after 120 seconds; GPU0 worker exit={code0}. "
                               "No other process was interrupted.")
        require(code1 == 0, f"GPU1 worker failed: {code1}; inspect log")
        if code0 == 75:
            print("GPU0 busy for acquisition window; did not use busy GPU0. Completing seed7 on GPU1.", flush=True)
            require(launch(1, [7]).wait() == 0, "GPU1 fallback failed")
        else:
            require(code0 == 0, f"GPU0 worker failed: {code0}; inspect log")
    finally:
        # Never kill any process, including on parent failure; reap our workers.
        for child in children:
            child.wait()
        for log in logs:
            log.close()
    summary()
    summary(check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--worker", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--aggregate", action="store_true")
    parser.add_argument("--gpu", type=int, choices=(0, 1))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS)
    parser.add_argument("--batch-size", type=int, choices=OOM_BATCH_SIZES, default=64)
    parser.add_argument("--allow-shared", action="store_true",
                        help="Permit external GPU jobs with >=24000 MiB free; cap our allocator at 35%%")
    args = parser.parse_args()
    if args.worker:
        require(args.gpu is not None and args.seeds, "Worker requires GPU and seeds")
        try:
            worker(args.gpu, args.seeds, args.batch_size, allow_shared=args.allow_shared)
        except TimeoutError as exc:
            print(str(exc), flush=True)
            return 75
    elif args.execute:
        try:
            execute(args.batch_size, allow_shared=args.allow_shared)
        except TimeoutError as exc:
            print(f"Scoring blocked: {exc}", flush=True)
            return 75
    else:
        summary(check=args.check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
