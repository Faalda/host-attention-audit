"""Evaluate existing seed7 weights, then rescore saved captions with standard R1.

Run with the project .venv, python -B, CUDA_VISIBLE_DEVICES=0, HF_HUB_OFFLINE=1,
HF_DATASETS_OFFLINE=1, TRANSFORMERS_OFFLINE=1, ATTN_ENCODER=dinov2,
TORCH_DISABLE_NATIVE_JIT=1,
FULL_COCO_DIR=results/full_coco_s7 and FULL_COCO_DATA_DIR=results/full_coco.
Existing evaluation artifacts are never overwritten. No training or downloads.
--resume-plans continues an interrupted evaluation with saved plans and no captions.
"""

import hashlib
import argparse
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/full_coco_s7"
DATA = ROOT / "results/full_coco"
sys.path.insert(0, str(ROOT))


def file_info(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(),
            "size_bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-plans", action="store_true")
    args = parser.parse_args()
    expected_env = {
        "CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "ATTN_ENCODER": "dinov2", "TORCH_DISABLE_NATIVE_JIT": "1",
    }
    for key, value in expected_env.items():
        if os.environ.get(key) != value:
            raise RuntimeError(f"Require {key}={value}")
    for key, path in (("FULL_COCO_DIR", OUT), ("FULL_COCO_DATA_DIR", DATA)):
        if Path(os.environ.get(key, "")).resolve() != path:
            raise RuntimeError(f"Require {key}={path}")
    if not OUT.is_dir() or not DATA.is_dir() or OUT.resolve() == DATA.resolve():
        raise RuntimeError("Missing or aliased input/output directories")
    names = ("plans_test.pt", "captioning_eval.json", "captions_test.json",
             "caps_test.json", "evaluation_provenance.json", "standard_captioning_eval.json")
    allowed = {OUT / name for name in names}
    retained = {"plans_test.pt", "caps_test.json", "evaluation_provenance.json"} if args.resume_plans else set()
    for path in allowed:
        if path.name in retained:
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(f"Expected retained regular file: {path}")
            continue
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {path}")
    initial_provenance = None
    if args.resume_plans:
        initial_provenance = json.loads((OUT / "evaluation_provenance.json").read_text())
        for relative, info in initial_provenance["sources"].items():
            if relative != "analysis/evaluate_second_seed.py" and file_info(ROOT / relative) != info:
                raise RuntimeError(f"Input changed since interrupted evaluation: {relative}")
        if (OUT / "caps_test.json").read_bytes() != (DATA / "caps_test.json").read_bytes():
            raise RuntimeError("Retained references differ")
    print(f"Preflight: new output paths absent; retained artifacts: {sorted(retained)}", flush=True)

    import numpy as np
    import torch
    import rescore_rouge

    # Enforce output confinement even for the unchanged evaluator's write-mode opens.
    def audit(event, args):
        if event == "open":
            path, mode, flags = args
            if isinstance(path, (str, bytes, os.PathLike)) and flags & (
                    os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                target = Path(os.fsdecode(path)).resolve()
                if target not in allowed or target.exists():
                    raise PermissionError(f"Blocked write/overwrite: {target}")
        elif event == "os.mkdir":
            if not Path(os.fsdecode(args[0])).is_dir():
                raise PermissionError(f"Blocked directory creation: {args[0]}")
        elif event in ("os.remove", "os.rename", "os.rmdir", "os.link", "os.symlink",
                       "os.truncate", "os.chmod", "os.utime"):
            raise PermissionError(f"Blocked filesystem mutation: {event}")
        elif event in ("socket.connect", "socket.getaddrinfo"):
            raise PermissionError("Network access is disabled")

    sys.addaudithook(audit)
    import historical.run_rev_fullcoco_eval as evaluator
    import historical.run_rev_fullcoco_common as common

    cfg = json.loads((OUT / "train_config.json").read_text())
    original_cfg = json.loads((DATA / "train_config.json").read_text())
    if cfg["seed"] != 7 or original_cfg["seed"] != 42:
        raise ValueError("Unexpected training seeds")
    if {k: v for k, v in cfg.items() if k != "seed"} != {
            k: v for k, v in original_cfg.items() if k != "seed"}:
        raise ValueError("Training configurations differ beyond seed")
    if (OUT / "tok_vocab.json").read_bytes() != (DATA / "tok_vocab.json").read_bytes():
        raise ValueError("Vocabularies differ")
    refs_bytes = (DATA / "caps_test.json").read_bytes()
    refs = json.loads(refs_bytes)
    if (len(refs["caps"]) != 5000 or len(refs["keys"]) != 5000
            or len(set(refs["keys"])) != 5000 or any(len(r) != 5 for r in refs["caps"])):
        raise ValueError("Expected all 5000 images with five aligned references")
    for name, shape, dtype in (("dino_test_q.npy", (5000, 256, 768), np.int8),
                               ("dino_test_scale.npy", (5000, 768), np.float16)):
        array = np.load(DATA / name, mmap_mode="r")
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(f"Unexpected feature cache: {name} {array.shape} {array.dtype}")

    models = ("model_proj.pt", "model_free.pt", "model_denoiser.pt", "model_guided.pt")
    architecture = {}
    for name in models:
        states = [torch.load(folder / name, map_location="cpu", weights_only=True)
                  for folder in (OUT, DATA)]
        signatures = [{k: (list(v.shape), str(v.dtype)) for k, v in st.items()} for st in states]
        if signatures[0] != signatures[1]:
            raise ValueError(f"Checkpoint architecture mismatch: {name}")
        architecture[name] = {"keys_shapes_dtypes_match_original": True,
                              "state_tensor_count": len(states[0]),
                              "state_numel": sum(v.numel() for v in states[0].values())}
        del states
    if torch.cuda.device_count() != 1 or not torch.cuda.is_available():
        raise RuntimeError("Expected only physical GPU0 to be visible")

    source_names = ("historical/run_rev_fullcoco_eval.py", "historical/run_rev_fullcoco_common.py",
                    "historical/run_rev_fullcoco_train.py", "inversion/run_rev_eval_metrics.py",
                    "clean_pipeline/run_summarization.py", "clean_pipeline/run_captioning.py",
                    "clean_pipeline/run_cap_improved.py",
                    "clean_pipeline/run_sum_masked.py", "historical/run_rev_den_common.py",
                    "analysis/rescore_rouge.py", "analysis/evaluate_second_seed.py",
                    "paper_cvpr/main.tex", "paper_cvpr/supp.tex",
                    "paper_cvpr/standard_rouge_results.json")
    inputs = [ROOT / name for name in source_names]
    inputs += [folder / name for folder in (OUT, DATA)
               for name in (*models, "tok_vocab.json", "train_config.json")]
    inputs += [DATA / name for name in ("caps_test.json", "dino_test_q.npy",
                                        "dino_test_scale.npy", "captions_test.json",
                                        "captioning_eval.json")]
    sources = {str(p.relative_to(ROOT)): file_info(p) for p in inputs}
    # Include untouched training caches/checkpoints in a metadata-only mutation check.
    protected = {p: (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns)
                 for folder in (DATA, OUT) for p in folder.rglob("*") if p.is_file()}
    provenance = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {**expected_env, "FULL_COCO_DIR": str(OUT), "FULL_COCO_DATA_DIR": str(DATA)},
        "python": sys.version, "executable": sys.executable,
        "versions": {k: rescore_rouge.version(k) for k in
                     ("torch", "numpy", "rouge-score", "pycocoevalcap", "nltk")},
        "device": torch.cuda.get_device_name(0), "training": False,
        "training_seed": 7, "original_training_seed": 42, "inference_rng_seed": 0,
        "training_config": cfg, "original_training_config": original_cfg,
        "training_configs_equal_except_seed": True, "vocabularies_byte_identical": True,
        "architecture_check": architecture,
        "architecture": {k: getattr(common, k) for k in
                         ("D_MODEL", "N_HEADS", "N_LAYERS", "L_CAP", "N_MP", "N_PATCHES", "ENCODER_DIM")},
        "inference": {"entry_point": "run_rev_fullcoco_eval.main (unchanged)",
                      "maskgit_steps": 4, "cfg_scale": 2.0, "plan_softmax_scale": 10.0,
                      "decoding": "greedy", "tf32": True,
                      "note": "Inference RNG seed 0 is not a training seed or a second trained model."},
        "bootstrap": {"resamples": 10000, "seed": 0, "unit": "image",
                      "method": "paired percentile 95%", "rng": "numpy.default_rng"},
        "write_policy": "Only six new seed7 artifacts; existing-file writes and outside writes blocked",
        "sources": sources,
    }
    if not args.resume_plans:
        with (OUT / "evaluation_provenance.json").open("x") as stream:
            json.dump(provenance, stream, indent=2, allow_nan=False)
        with (OUT / "caps_test.json").open("xb") as stream:
            stream.write(refs_bytes)
    print("Verified matching configs/architecture, identical vocab and all5k cache; running evaluator", flush=True)
    if args.resume_plans:
        # Same main() setup and evaluation function; only skip rewriting saved plans.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        common.set_seed(0)
        tok = common.tok_from_vocab(json.loads((OUT / "tok_vocab.json").read_text()))
        proj, free, den, gdec = common.load_final_models(len(tok.vocab))
        feats = common.GPUFeats("test")
        evaluated, captions, plans = evaluator.evaluate_system(
            proj, free, den, gdec, feats, common.load_caps("test"), tok, save_plans=None)
        if not torch.equal(plans.cpu(), torch.load(OUT / "plans_test.pt", map_location="cpu", weights_only=True)):
            raise RuntimeError("Regenerated plans differ from retained plans; refusing inconsistent artifacts")
        evaluated["train_config"] = cfg
        with (OUT / "captioning_eval.json").open("x") as stream:
            json.dump(evaluated, stream, indent=1)
        with (OUT / "captions_test.json").open("x") as stream:
            json.dump(captions, stream)
        provenance["continuation"] = {
            "reason": "Native Triton JIT attempted an outside-cache write; write guard stopped the initial run",
            "retained_artifacts": sorted(retained), "regenerated_plans_exactly_equal": True,
            "entry_point": "Unchanged run_rev_fullcoco_eval.evaluate_system; same main setup, no plan rewrite",
            "initial_provenance": initial_provenance,
        }
        print(json.dumps(evaluated, indent=1), flush=True)
    else:
        evaluator.main()

    captions = json.loads((OUT / "captions_test.json").read_text())
    saved_refs = json.loads((OUT / "caps_test.json").read_text())
    print("Scoring standard caption-only ROUGE-1 with rescore_rouge.full_coco", flush=True)
    standard = rescore_rouge.full_coco(captions, saved_refs)
    saved_eval = json.loads((OUT / "captioning_eval.json").read_text())
    if saved_eval["n_images"] != 5000 or saved_eval["train_config"] != cfg:
        raise ValueError("Unexpected evaluation provenance")
    for relative, info in sources.items():
        if file_info(ROOT / relative) != info:
            raise RuntimeError(f"Protected source changed: {relative}")
    for path, before in protected.items():
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != before:
            raise RuntimeError(f"Protected input metadata changed: {path}")
    if {p for p in DATA.rglob("*") if p.is_file()} != {p for p in protected if p.is_relative_to(DATA)}:
        raise RuntimeError("Original run directory file inventory changed")
    artifacts = {name: file_info(OUT / name) for name in names if name != "standard_captioning_eval.json"}
    if artifacts["caps_test.json"]["sha256"] != sources["results/full_coco/caps_test.json"]["sha256"]:
        raise RuntimeError("Saved references differ")
    result = {
        "schema_version": 1, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": {"metric": "rouge_score rouge1 fmeasure", "use_stemmer": False,
                     "aggregation": "best of five references per saved caption, then mean over 5000 images",
                     "normalization": "lowercase; replace [^a-z0-9 ] by space; collapse whitespace",
                     "implementation": "rescore_rouge.full_coco", "caption_only": True,
                     "bertscore": False, "bootstrap": provenance["bootstrap"],
                     "training_seed": 7, "inference_rng_seed": 0},
        "standard_rouge1": standard,
        "existing_evaluator_metrics": saved_eval,
        "cider_provenance": "Copied unchanged from run_rev_fullcoco_eval.py output; not recomputed by R1 rescoring",
        "provenance": provenance, "artifacts": artifacts,
        "verification": {"protected_input_hashes_unchanged": True,
                         "protected_input_metadata_unchanged": True,
                         "original_directory_inventory_unchanged": True,
                         "saved_references_byte_identical": True},
        "limitations": ["Image bootstrap CIs are conditional on these trained checkpoints, not across training seeds.",
                        "This caption-only second-seed run does not establish a multi-seed gate claim.",
                        "Original training seed is 42; inference RNG seed 0 must not be labeled training seed 0."],
    }
    with (OUT / "standard_captioning_eval.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"systems": {k: {a: b for a, b in v.items() if a != "per_image"}
                                  for k, v in standard["systems"].items()},
                      "paired": {k: {a: b for a, b in v.items() if a != "per_image"}
                                 for k, v in standard["paired"].items()},
                      "verification": result["verification"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
