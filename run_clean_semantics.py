"""Canonical offline BERTScore for complete clean 1000-image saved captions.

Default is read-only preflight. --execute and CUDA_VISIBLE_DEVICES=0 are both
required. No model imports occur before the GPU lease, and no downloads, subset
fallbacks, training, or overwriting of published results are permitted.
"""

import argparse
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

from clean_gpu_guard import exclusive_lock, gpu_lease
from run_clean_pipeline import (EVAL_ARTIFACTS, checked_path, evaluation_complete,
                                fingerprint, read_json, sha256, verify_hashes, write_json)

ROOT = Path(__file__).resolve().parent
SNAPSHOT_REVISION = "722cf37b1afa9454edce342e7895e588b6ff1d59"
SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--roberta-large/snapshots" / SNAPSHOT_REVISION
SNAPSHOT_FILES = ("config.json", "merges.txt", "model.safetensors", "tokenizer_config.json",
                  "tokenizer.json", "vocab.json")
VARIANTS = ("free", "onehot", "hidden")
SYSTEMS = ("free", "guided_denoised", "guided_uniform")
PROTOCOLS = ("single_ref", "heldout", "each_inversion_all_refs_avg", "human_heldout",
             "mismatch_target", "mismatch_grounding")
CODE_FILES = ("run_clean_semantics.py", "run_clean_pipeline.py", "clean_gpu_guard.py",
              "paper_cvpr/score_semantics.py", "paper_cvpr/rescore_rouge.py")
PACKAGES = ("bert-score", "transformers", "tokenizers", "torch", "numpy",
            "huggingface-hub", "safetensors", "rouge-score", "regex")
CONFIG = dict(metric="canonical bert-score 0.3.13 raw F1", model="roberta-large",
              snapshot_revision=SNAPSHOT_REVISION, num_layers=17, idf=False,
              rescale_with_baseline=False, all_layers=False, dtype="float32", autocast=False,
              tf32=False, n_images=1000, variants=list(VARIANTS), forward_systems=list(SYSTEMS),
              bootstrap=dict(resamples=10000, seed=0, unit="image", method="paired percentile [2.5,97.5]"),
              selection="All 1000 test images, all five slots, all variants; no fallback/subsetting",
              normalization="canonical sent_encode strip; prefix space; original case/punctuation; max_length=512")


def prerequisites():
    """Read-only availability check; safe in the parent and on CPU machines."""
    errors = []
    for name in SNAPSHOT_FILES:
        if not (SNAPSHOT / name).is_file():
            errors.append(f"Missing offline semantic snapshot file: {SNAPSHOT / name}")
    for package in PACKAGES:
        try:
            version = importlib.metadata.version(package)
            if package == "bert-score" and version != "0.3.13":
                errors.append("Canonical semantic scoring requires bert-score==0.3.13")
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Missing semantic dependency: {package}")
    return errors


def validate_inputs(protocol, forward):
    """Clean-cohort schema, independent of the canonical 5000-image validator."""
    records = protocol["records"]
    ids = list(range(1000))
    if (protocol.get("partial") is not False or protocol.get("n_images") != 1000
            or [r["image"] for r in records] != ids or protocol.get("evaluated_true_slots") != list(range(5))
            or set(protocol["variants"]) != set(VARIANTS) or len(protocol["variants"]) != 3):
        raise ValueError("Expected complete ordered clean1000 protocol and all variants/slots")
    if (forward.get("n_images") != 1000 or forward.get("image_ids") != ids
            or forward["refs"] != [r["refs"] for r in records]):
        raise ValueError("Clean forward image/reference alignment mismatch")
    for name in SYSTEMS:
        caps = forward["caps"][name]
        if len(caps) != 1000 or not all(isinstance(c, str) for c in caps):
            raise ValueError(f"Incomplete forward system: {name}")
    donors = [r["mismatch_source"] for r in records]
    if sorted(donors) != ids or any(i == donor for i, donor in enumerate(donors)):
        raise ValueError("Expected the complete saved donor derangement")
    for record in records:
        i = record["image"]
        if record.get("feature_index") != 14000 + i:
            raise ValueError("Not a clean test feature index")
        if len(record["refs"]) != 5 or not all(isinstance(r, str) and r for r in record["refs"]):
            raise ValueError("Expected five nonempty reference slots, preserving duplicates")
        if (record["baseline"] != forward["caps"]["guided_denoised"][i]
                or record["mismatch_target"] not in records[record["mismatch_source"]]["refs"]):
            raise ValueError("Protocol baseline or shuffled target is misaligned")
        if set(record["variants"]) != set(VARIANTS):
            raise ValueError("Unexpected record variants")
        for variant in VARIANTS:
            jobs = record["variants"][variant]
            if (len(jobs["true"]) != 5 or not all(isinstance(c, str) for c in jobs["true"])
                    or not isinstance(jobs["mismatch"], str)):
                raise ValueError("Missing true/shuffle caption; partial scoring forbidden")
    return records


def load_canonical():
    # The unchanged canonical module imports `rescore_rouge` as a top-level name
    # and sets CUDA_VISIBLE_DEVICES at import. Never import it in the parent.
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("CUDA_VISIBLE_DEVICES must already be exactly '0'; no environment upgrade")
    sys.path.insert(0, str(ROOT / "paper_cvpr"))
    canonical = importlib.import_module("score_semantics")
    if Path(canonical.__file__).resolve() != ROOT / "paper_cvpr/score_semantics.py":
        raise ValueError("Unexpected canonical semantic module")
    bootstrap = importlib.import_module("rescore_rouge")
    if Path(bootstrap.__file__).resolve() != ROOT / "paper_cvpr/rescore_rouge.py":
        raise ValueError("Unexpected canonical bootstrap module")
    return canonical


def collect_pairs(canonical, records, forward):
    refs = {"caps": forward["refs"]}
    pairs = dict.fromkeys(canonical.collect_pairs(records, forward["caps"], refs, len(records)))
    # Canonical collection includes denoised/uniform; add the clean free control
    # without merging reversed pairs or changing caption case/punctuation.
    for caption, references in zip(forward["caps"]["free"], forward["refs"]):
        for reference in references:
            pairs.setdefault((caption, reference), None)
    return list(pairs)


def aggregate(canonical, records, forward, scores):
    def score(a, b):
        return scores[a, b]

    def paired(a, b):
        return canonical.paired(a, b, resamples=10000, seed=0)

    protocol = {"n_images": len(records), "image_ids": [r["image"] for r in records],
                "target_slots": list(range(5)), "variants": {}}
    for variant in VARIANTS:
        rows = [canonical.small_image(r, variant, score) for r in records]
        protocol["variants"][variant] = {key: paired([row[key][0] for row in rows],
                                                     [row[key][1] for row in rows]) for key in PROTOCOLS}
    values = {name: [max(score(c, ref) for ref in refs) for c, refs in zip(forward["caps"][name], forward["refs"])]
              for name in SYSTEMS}
    systems = {}
    for name, vals in values.items():
        summary = paired(vals, [0.0] * len(vals))
        systems[name] = {"mean": summary["inv"], "ci95": summary["inv_ci95"], "per_image": vals}
    return {"protocol": protocol, "forward": {"n_images": len(records), "image_ids": protocol["image_ids"],
            "systems": systems, "paired": {"denoised_minus_" + name: paired(values["guided_denoised"], values[name])
                                            for name in ("free", "guided_uniform")}}}


def verify_result(folder, result=None):
    """Validate before publication, or verify a saved result without GPU imports."""
    folder = checked_path(folder)
    if result is None:
        result = read_json(folder / "semantic_scores.json")
    meta = result["metadata"]
    if (result.get("status") != "complete" or meta.get("config") != CONFIG
            or meta.get("config_sha256") != fingerprint(CONFIG)
            or meta.get("cuda_visible_devices") != "0" or meta.get("training") is not False
            or meta.get("model_downloads") is not False or meta.get("model_frozen") is not True
            or meta.get("offline") is not True or not meta.get("gpu_uuid") or meta.get("device") != "cuda:0"
            or result.get("verification", {}).get("source_hashes_unchanged") is not True
            or result["verification"].get("tokenization_audit", {}).get("passed") is not True
            or result["verification"].get("smoke", {}).get("passed") is not True):
        raise ValueError("Uncertified semantic result; cannot report completion")
    if not set(EVAL_ARTIFACTS) <= result["input_sha256"].keys() or set(meta["code_sha256"]) != set(CODE_FILES):
        raise ValueError("Semantic source inventory incomplete")
    verify_hashes(result["input_sha256"], folder)
    verify_hashes(meta["code_sha256"], ROOT)
    if set(meta["snapshot_sha256"]) != set(SNAPSHOT_FILES):
        raise ValueError("Semantic model inventory incomplete")
    for name, digest in meta["snapshot_sha256"].items():
        if sha256(SNAPSHOT / name) != digest:
            raise ValueError(f"Semantic model hash mismatch: {name}")
    distribution = importlib.metadata.distribution("bert-score")
    if set(meta["canonical_sha256"]) != {"bert_score/utils.py", "bert_score/scorer.py"}:
        raise ValueError("Missing canonical implementation hashes")
    for name, digest in meta["canonical_sha256"].items():
        if sha256(distribution.locate_file(name)) != digest:
            raise ValueError(f"Canonical implementation hash mismatch: {name}")
    if set(meta["versions"]) != set(PACKAGES) or meta["versions"]["bert-score"] != "0.3.13":
        raise ValueError("Missing/noncanonical semantic package versions")
    for name, version in meta["versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Semantic package version changed: {name}")
    for dataset in ("protocol", "forward"):
        if (result["datasets"][dataset]["n_images"] != 1000
                or result["datasets"][dataset]["image_ids"] != list(range(1000))):
            raise ValueError("Incomplete semantic cohort")

    def finite_vector(values, length):
        return (isinstance(values, list) and len(values) == length
                and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values))

    def check_paired(row):
        if (row["n_images"] != 1000
                or any(not finite_vector(row["per_image"][side], 1000) for side in ("inv", "base"))
                or any(not finite_vector(row[key], 2) for key in ("ci95", "inv_ci95", "base_ci95"))
                or not finite_vector([row["inv"], row["base"], row["delta"]], 3)):
            raise ValueError("Incomplete/nonfinite per-image semantic results")
        if (any(not math.isclose(row[side], sum(row["per_image"][side]) / 1000, abs_tol=1e-10)
                for side in ("inv", "base"))
                or not math.isclose(row["delta"], row["inv"] - row["base"], abs_tol=1e-10)):
            raise ValueError("Semantic summary does not match per-image scores")

    if set(result["datasets"]["protocol"]["variants"]) != set(VARIANTS):
        raise ValueError("Incomplete semantic variants")
    for variant in VARIANTS:
        rows = result["datasets"]["protocol"]["variants"][variant]
        if set(rows) != set(PROTOCOLS):
            raise ValueError("Incomplete semantic protocols")
        for row in rows.values():
            check_paired(row)
    forward = result["datasets"]["forward"]
    if (set(forward["systems"]) != set(SYSTEMS)
            or set(forward["paired"]) != {"denoised_minus_free", "denoised_minus_guided_uniform"}):
        raise ValueError("Incomplete semantic forward controls")
    for system in forward["systems"].values():
        if (not finite_vector(system["per_image"], 1000) or not finite_vector(system["ci95"], 2)
                or not finite_vector([system["mean"]], 1)
                or not math.isclose(system["mean"], sum(system["per_image"]) / 1000, abs_tol=1e-10)):
            raise ValueError("Incomplete/nonfinite semantic forward scores")
    for name in ("free", "guided_uniform"):
        row = forward["paired"]["denoised_minus_" + name]
        check_paired(row)
        if (row["per_image"]["inv"] != forward["systems"]["guided_denoised"]["per_image"]
                or row["per_image"]["base"] != forward["systems"][name]["per_image"]):
            raise ValueError("Misaligned paired semantic forward controls")
    return result


def execute(folder, *, wait=False, batch_size=128, minimum_free_mib=20000):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("CUDA_VISIBLE_DEVICES must already be exactly '0'; no environment upgrade")
    if batch_size < 1 or minimum_free_mib < 0:
        raise ValueError("Invalid batch size or free-memory threshold")
    errors = prerequisites()
    if errors:
        raise ValueError("; ".join(errors))
    folder = checked_path(folder)
    evaluation_complete(folder)  # Fail before GPU acquisition if clean models/outputs do not exist.
    output = folder / "semantic_scores.json"
    with exclusive_lock(folder / "semantic.lock"):
        if output.exists():
            raise FileExistsError(f"Refusing semantic overwrite: {output}")
        inputs = {name: sha256(folder / name) for name in EVAL_ARTIFACTS}
        code = {name: sha256(ROOT / name) for name in CODE_FILES}
        snapshot = {name: sha256(SNAPSHOT / name) for name in SNAPSHOT_FILES}
        distribution = importlib.metadata.distribution("bert-score")
        implementation = {name: sha256(distribution.locate_file(name))
                          for name in ("bert_score/utils.py", "bert_score/scorer.py")}
        protocol = read_json(folder / "inversion_protocol_n1000_s0.json")
        forward = read_json(folder / "forward.json")
        records = validate_inputs(protocol, forward)
        status = {"pid": os.getpid(), "status": "waiting_for_gpu", "started": time.time()}
        status_path = folder / "semantic_status.json"
        write_json(status_path, status, replace=True)
        try:
            with gpu_lease(wait=wait, minimum_free_mib=minimum_free_mib) as lease:
                status.update(status="scoring", gpu_uuid=lease["gpu_uuid"], updated=time.time())
                write_json(status_path, status, replace=True)
                for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
                    os.environ[key] = "1"
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                canonical = load_canonical()
                import torch

                scorer, repaired = canonical.make_scorer(snapshot=SNAPSHOT)
                scorer._model.eval().requires_grad_(False)
                pairs = collect_pairs(canonical, records, forward)
                texts = {text for pair in pairs for text in pair}
                with torch.inference_mode():
                    smoke = canonical.smoke(scorer)
                    audit = canonical.audit_tokenization(scorer._tokenizer, texts, snapshot=SNAPSHOT)
                    if smoke.get("passed") is not True or audit.get("passed") is not True:
                        raise ValueError("Canonical smoke/tokenization audit did not pass")
                    # One combined call preserves canonical CPU FP32 embedding
                    # caching and greedy token matching. OOM is fatal, no subset retry.
                    _, _, f1 = scorer.score([p[0] for p in pairs], [p[1] for p in pairs],
                                            batch_size=batch_size, verbose=True)
                values = f1.tolist()
                if len(values) != len(pairs) or not all(math.isfinite(v) for v in values):
                    raise ValueError("Incomplete/nonfinite canonical pair scores")
                if (any(m.training for m in scorer._model.modules())
                        or any(p.requires_grad or p.grad is not None for p in scorer._model.parameters())):
                    raise ValueError("Semantic model was not frozen")
                datasets = aggregate(canonical, records, forward, dict(zip(pairs, values)))
                versions = {name: importlib.metadata.version(name) for name in PACKAGES}
                result = {"schema_version": 1, "status": "complete", "input_sha256": inputs,
                          "metadata": {"config": CONFIG, "config_sha256": fingerprint(CONFIG),
                            "code_sha256": code, "snapshot": str(SNAPSHOT), "snapshot_sha256": snapshot,
                            "canonical_sha256": implementation, "versions": versions,
                            "device": "cuda:0", "gpu_uuid": lease["gpu_uuid"], "cuda_visible_devices": "0",
                            "gpu_name": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda,
                            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(0),
                            "model_frozen": True, "training": False, "offline": True, "model_downloads": False,
                            "batch_size": batch_size, "unique_scored_pairs": len(pairs), "unique_scored_strings": len(texts),
                            "ordered_pairs_sha256": fingerprint(pairs),
                            "embedding_cache": "canonical bert_cos_score_idf combined-call CPU FP32 token embedding cache",
                            "tokenizer": {"class": type(scorer._tokenizer).__name__,
                                          "is_fast_actual": scorer._tokenizer.is_fast,
                                          "prefix_space_compatibility_fix": repaired},
                            "python": sys.version, "executable": sys.executable,
                            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                            "limitations": ["Text-reference similarity is not image-grounding or factual accuracy.",
                                            "Raw unrescaled BERTScore has a high similarity floor.",
                                            "Image bootstrap intervals are not training-seed uncertainty; no multiplicity correction."]},
                          "verification": {"source_hashes_unchanged": True, "smoke": smoke, "tokenization_audit": audit},
                          "datasets": datasets, "elapsed_seconds": time.time() - status["started"]}
                verify_result(folder, result)
                write_json(output, result)
                status.update(status="complete", completed=time.time(), output=str(output), sha256=sha256(output))
        except BaseException as exc:
            status.update(status="failed", error=repr(exc), failed=time.time())
            raise
        finally:
            status["updated"] = time.time()
            write_json(status_path, status, replace=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--minimum-free-mib", type=int, default=20000)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.minimum_free_mib < 0:
        parser.error("Batch size must be positive and minimum free memory nonnegative")
    try:
        if args.execute:
            execute(args.eval_dir, wait=args.wait_for_gpu, batch_size=args.batch_size,
                    minimum_free_mib=args.minimum_free_mib)
        else:
            errors = prerequisites()
            folder = checked_path(args.eval_dir)
            try:
                evaluation_complete(folder)
                validate_inputs(read_json(folder / "inversion_protocol_n1000_s0.json"), read_json(folder / "forward.json"))
                if (folder / "semantic_scores.json").exists():
                    errors.append("Semantic output already exists; pipeline may only skip after hash verification")
            except (OSError, ValueError, KeyError) as exc:
                errors.append(str(exc))
            print(json.dumps({"status": "blocked" if errors else "ready", "launch_performed": False,
                              "config": CONFIG, "errors": errors}, indent=2))
            return int(bool(errors))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Semantic scoring failed; no completion claimed: {exc}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
