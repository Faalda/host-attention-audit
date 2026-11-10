"""Read-only plan by default. Only --execute starts the foreground queue.

Smoke seed42, then train/evaluate/BERTScore/CPU-score seeds 42, 7, 13.
GPU children acquire their own clean_gpu_guard lease; the parent never does.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

from clean_gpu_guard import exclusive_lock

ROOT = Path(__file__).resolve().parent.parent  # repo root (file now lives one level down)
import sys as _sys
_sys.path.insert(0, str(ROOT))  # repo root, for analysis./inversion./clean_pipeline. imports
TRAIN_ROOT = ROOT / "results/clean_small"
EVAL_ROOT = ROOT / "results/clean_eval"
SEEDS = (42, 7, 13)
MODES = ("forward", "protocol", "counterfactual", "corruption", "bounded")
TRAIN_ARTIFACTS = ("model_proj.pt", "model_free.pt", "model_denoiser.pt", "model_guided.pt",
                   "tok_vocab.json", "plans_test.pt", "caps_test.json", "captions_test.json",
                   "standard_captioning_eval.json")
EVAL_ARTIFACTS = ("planned_jobs.json", "target_encoding.json", "forward.json",
                  "inversion_protocol_n1000_s0.json", "swap_injection_n1000_s0.json",
                  "matched_validation.json", "matched_calibration_lock.json", "matched_gain.json", "bounded_unbounded_n200.json",
                  "corruption_0.json", "corruption_0.25.json", "corruption_0.5.json", "corruption_1.json",
                  "complete.json")
STAGE_ARTIFACTS = {"eval": EVAL_ARTIFACTS, "semantic": ("semantic_scores.json",), "score": ("cpu_scores.json",)}
SEMANTIC = {"status": "scheduled", "seeds": list(SEEDS), "completed_seeds": []}


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value, *, replace=False):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            temporary.replace(path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def checked_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Path escapes workspace: {path}")
    return path


def verify_hashes(hashes, base):
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Missing hash inventory")
    for name, expected in hashes.items():
        path = checked_path(base / name)
        if not isinstance(expected, str) or sha256(path) != expected:
            raise ValueError(f"Hash mismatch: {path}")


def training_action(folder, seed, smoke, resume=False):
    checked_path(folder)
    if not folder.exists() or not any(folder.iterdir()):
        return "run"
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        if {p.name for p in folder.iterdir()} == {"status.json"}:
            status = read_json(folder / "status.json")
            if (status.get("status") == "blocked_gpu_busy" and status.get("seed") == seed
                    and status.get("smoke") is smoke and status.get("training_seconds") == 0):
                return "run"  # A blocked pre-lease attempt never trained or exported.
        raise ValueError(f"Unrecognized partial training directory: {folder}")
    manifest = read_json(manifest_path)
    audit = {key: manifest[key] for key in ("config", "hashes", "split_ids", "unknown_rates",
                                          "checks", "alignment_limit")}
    if manifest.get("seed") != seed or manifest.get("smoke") is not smoke:
        raise ValueError(f"Training seed/smoke mismatch: {folder}")
    expected = fingerprint(dict(audit=audit, seed=seed, smoke=smoke))
    if manifest.get("fingerprint") != expected:
        raise ValueError(f"Training fingerprint mismatch: {folder}")
    required_inputs = {"clean_pipeline/run_clean_captioner.py", "results/dinov2_patch_features_15k.pt",
                       "results/revision/coco_caps_15k.json", "results/revision/inversion_protocol_n1000_s0.json"}
    if not required_inputs <= audit["hashes"].keys():
        raise ValueError(f"Training input inventory incomplete: {folder}")
    verify_hashes(audit["hashes"], ROOT)
    if manifest.get("status") == "complete":
        hashes = manifest.get("artifact_hashes", {})
        if not set(TRAIN_ARTIFACTS) <= hashes.keys():
            raise ValueError(f"Training export inventory incomplete: {folder}")
        verify_hashes(hashes, folder)
        return "skip_hash_verified"
    if not resume:
        raise ValueError(f"Interrupted training needs explicit --resume and matching fingerprint: {folder}")
    return "resume"


def build_stages(python=None):
    python = python or sys.executable
    stages = []
    for seed, smoke in [(42, True)] + [(seed, False) for seed in SEEDS]:
        folder = TRAIN_ROOT / ("smoke/" if smoke else "") / f"seed{seed}"
        argv = [python, "-B", str(ROOT / "clean_pipeline/run_clean_captioner.py"), "--seeds", str(seed),
                "--wait-for-gpu", "--output-root", str(TRAIN_ROOT)]
        if smoke:
            argv.append("--smoke")
        stages.append(dict(name=f"{'smoke' if smoke else 'train'}_seed{seed}", kind="train",
                           seed=seed, smoke=smoke, folder=str(folder), argv=argv, cuda="0"))
        if smoke:
            continue
        output = EVAL_ROOT / f"seed{seed}"
        argv = [python, "-B", str(ROOT / "clean_pipeline/run_clean_evaluation.py"), "--model-dir", str(folder),
                "--out-dir", str(output), "--device", "cuda", "--allow-gpu", "--wait-for-gpu",
                "--modes", *MODES, "--matched-gain"]
        stages.append(dict(name=f"eval_seed{seed}", kind="eval", seed=seed, folder=str(output), argv=argv, cuda="0"))
        stages.append(dict(name=f"semantics_seed{seed}", kind="semantic", seed=seed, folder=str(output), cuda="0",
                           argv=[python, "-B", str(ROOT / "clean_pipeline/run_clean_semantics.py"), "--execute",
                                 "--eval-dir", str(output), "--wait-for-gpu"]))
        stages.append(dict(name=f"score_seed{seed}", kind="score", seed=seed, folder=str(output), cuda="",
                           argv=[python, "-B", str(ROOT / "clean_pipeline/run_clean_pipeline.py"), "--execute",
                                 "--score-dir", str(output)]))
    return stages


def receipt_signature(stage):
    files = ("clean_pipeline/run_clean_pipeline.py", "clean_pipeline/clean_gpu_guard.py",
             "clean_pipeline/run_clean_evaluation.py",
             "clean_pipeline/run_clean_semantics.py", "analysis/score_semantics.py",
             "analysis/rescore_rouge.py", "inversion/run_rev_eval_metrics.py")
    return fingerprint({"stage": stage, "code": {name: sha256(ROOT / name) for name in files}})


def evaluation_complete(folder):
    complete = read_json(folder / "complete.json")
    if (complete.get("all_inputs_unchanged") is not True
            or complete.get("all_model_parameters_frozen") is not True
            or not complete.get("model_state_sha256_before")
            or complete.get("model_state_sha256_before") != complete.get("model_state_sha256_after")):
        raise ValueError(f"Uncertified evaluation: {folder}")
    metadata = read_json(folder / "planned_jobs.json")["metadata"]
    if set(metadata["modes"]) != set(MODES) or metadata.get("matched_gain") is not True:
        raise ValueError(f"Evaluation modes do not match pipeline: {folder}")
    verify_hashes(metadata["input_sha256"], ROOT)
    for name in EVAL_ARTIFACTS:
        if not (folder / name).is_file():
            raise ValueError(f"Missing evaluation artifact: {folder / name}")
    matched = read_json(folder / "matched_gain.json")
    if (matched.get("n_evaluation") != 901 or not isinstance(matched.get("evaluation_vs_free"), dict)
            or set(matched["evaluation_vs_free"]) != {"gain_rouge1", "normalized_token_edit"}):
        raise ValueError(f"Unexpected matched-gain output schema: {folder}")


def stage_action(stage, resume=False):
    folder = checked_path(stage["folder"])
    if stage["kind"] == "train":
        return training_action(folder, stage["seed"], stage["smoke"], resume)
    receipt_path = folder / f"pipeline_{stage['kind']}_receipt.json"
    if stage["kind"] == "eval" and not folder.exists():
        return "run"
    names = STAGE_ARTIFACTS[stage["kind"]]
    if stage["kind"] != "eval" and not receipt_path.exists() and not any((folder / name).exists() for name in names):
        return "run"
    if not receipt_path.is_file():
        raise ValueError(f"Refusing partial/unverified existing {stage['kind']} output: {folder}")
    receipt = read_json(receipt_path)
    if receipt.get("signature") != receipt_signature(stage) or not set(names) <= receipt["hashes"].keys():
        raise ValueError(f"Receipt/configuration mismatch: {receipt_path}")
    verify_hashes(receipt["hashes"], folder)
    evaluation_complete(folder)
    if stage["kind"] == "score":
        verify_hashes(read_json(folder / "cpu_scores.json")["input_sha256"], folder)
    if stage["kind"] in ("semantic", "score"):
        from run_clean_semantics import verify_result
        verify_result(folder)
    return "skip_hash_verified"


def summarize_evaluation(folder, output=None):
    """CPU-only existing scorers; optional new JSON, never overwrite a partial score."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU scoring requires CUDA_VISIBLE_DEVICES='' before imports")
    from analysis.rescore_rouge import protocol, paired
    from inversion.run_rev_eval_metrics import evaluate

    folder = checked_path(folder)
    evaluation_complete(folder)
    from run_clean_semantics import verify_result
    verify_result(folder)
    inputs = {name: sha256(folder / name) for name in (*EVAL_ARTIFACTS, "semantic_scores.json")}
    result = {"semantic_scoring": {"status": "complete", "artifact": "semantic_scores.json",
                                   "sha256": inputs["semantic_scores.json"]},
              "device": "cpu", "input_sha256": inputs, "protocols": {}}
    for name in ("inversion_protocol_n1000_s0.json", "bounded_unbounded_n200.json",
                 "corruption_0.json", "corruption_0.25.json", "corruption_0.5.json", "corruption_1.json"):
        result["protocols"][name] = {
            "standard_rouge": protocol(read_json(folder / name), require_five=name.startswith("inversion_")),
            "metrics": evaluate(str(folder / name))}
    result["forward"] = read_json(folder / "forward.json")["paired"]
    swaps = read_json(folder / "swap_injection_n1000_s0.json")
    result["counterfactual"] = {
        variant: {key: paired([r["scores"][variant][key] for r in swaps["records"]],
                              [0.0] * len(swaps["records"]))
                  for key in swaps["records"][0]["scores"][variant]}
        for variant in swaps["variants"]}
    result["matched_gain"] = read_json(folder / "matched_gain.json")["evaluation_vs_free"]
    verify_hashes(inputs, folder)
    if output is not None:
        write_json(checked_path(output), result)
    return result


def preflight(stages, resume=False):
    """Read-only static/dependency/output checks. No GPU query or child launch."""
    errors = []
    for directory in (ROOT / "results", TRAIN_ROOT):
        if not checked_path(directory).is_dir():
            errors.append(f"Required existing output parent: {directory}")
    for name in ("clean_pipeline/run_clean_captioner.py", "clean_pipeline/run_clean_evaluation.py",
                 "clean_pipeline/run_clean_semantics.py"):
        tree = ast.parse((ROOT / name).read_text())
        imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                   and node.module == "clean_gpu_guard" for alias in node.names}
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        if ("gpu_lease" not in imports
                or not any(isinstance(node.func, ast.Name) and node.func.id == "gpu_lease" for node in calls)
                or not any(isinstance(node, ast.Constant) and node.value == "--wait-for-gpu" for node in ast.walk(tree))):
            errors.append(f"{name} must integrate gpu_lease and --wait-for-gpu before execution")
    for name in ("results/dinov2_patch_features_15k.pt", "results/revision/coco_caps_15k.json",
                 "results/revision/inversion_protocol_n1000_s0.json", "results/revision/swap_injection_n1000_s0.json"):
        if not (ROOT / name).is_file():
            errors.append(f"Missing input: {name}")
    for package in ("torch", "numpy", "rouge-score", "pycocoevalcap"):
        try:
            importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Missing Python dependency in {sys.executable}: {package}")
    from run_clean_semantics import prerequisites
    errors.extend(prerequisites())
    plan = []
    for stage in stages:
        try:
            action = stage_action(stage, resume)
        except (OSError, ValueError, KeyError) as exc:
            action = "blocked"
            errors.append(f"{stage['name']}: {exc}")
        command = stage["argv"] + (["--resume"] if action == "resume" else [])
        plan.append({"stage": stage["name"], "action": action,
                     "command": f"CUDA_VISIBLE_DEVICES={shlex.quote(stage['cuda'])} {shlex.join(command)}"})
    return {"status": "ready" if not errors else "blocked", "launch_performed": False,
            "stages": plan, "errors": errors, "semantic_scoring": SEMANTIC,
            "gpu_policy": "GPU0 only; child lease waits, no parent GPU lease; external jobs audited, never killed"}


def run_pipeline(stages, resume=False):
    # The queue lock spans every stage, including CPU scoring. Children inherit
    # it so an abruptly killed runner cannot allow a second queue over its child.
    with exclusive_lock(TRAIN_ROOT / "pipeline.lock") as queue_fd:
        report = preflight(stages, resume)
        if report["errors"]:
            raise ValueError(json.dumps(report, indent=2))
        EVAL_ROOT.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        folder = TRAIN_ROOT / f"pipeline_{stamp}_{os.getpid()}"
        folder.mkdir()
        status = {"pid": os.getpid(), "started": time.time(), "status": "queued",
                  "semantic_scoring": {**SEMANTIC, "completed_seeds": []}, "stages": stages, "completed_stages": [],
                  "stage_signatures": {s["name"]: receipt_signature(s) for s in stages}}
        status_path = folder / "status.json"
        write_json(status_path, status)
        print(f"Pipeline status: {status_path}", flush=True)
        child = None
        previous_handler = signal.getsignal(signal.SIGTERM)

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        try:
            for stage in stages:
                signature = status["stage_signatures"][stage["name"]]
                if receipt_signature(stage) != signature:
                    raise ValueError("Prepared queue code changed since startup")
                action = stage_action(stage, resume)
                status.update(stage=stage["name"], status=action, updated=time.time(), child_pid=None)
                write_json(status_path, status, replace=True)
                if action != "skip_hash_verified":
                    command = stage["argv"] + (["--resume"] if action == "resume" else [])
                    log = folder / f"{stage['name']}.log"
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=stage["cuda"], ATTN_ENCODER="dinov2",
                               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                               PYTHONUNBUFFERED="1", PYTHONHASHSEED="0")
                    status.update(status="running_or_waiting_for_gpu", command=command, log=str(log))
                    with log.open("x") as stream:
                        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                                 stderr=subprocess.STDOUT, start_new_session=True,
                                                 pass_fds=(queue_fd,))
                        status.update(child_pid=child.pid, child_pgid=child.pid, updated=time.time())
                        write_json(status_path, status, replace=True)
                        code = child.wait()
                    child = None
                    status["returncode"] = code
                    if code:
                        raise subprocess.CalledProcessError(code, command)
                    if receipt_signature(stage) != signature:
                        raise ValueError("Pipeline/evaluation/scoring code changed during this stage")
                    output = Path(stage["folder"])
                    if stage["kind"] == "train":
                        if stage_action(stage, resume) != "skip_hash_verified":
                            raise ValueError("Training child exited without a verified complete export")
                    else:
                        evaluation_complete(output)
                        if stage["kind"] in ("semantic", "score"):
                            from run_clean_semantics import verify_result
                            verify_result(output)
                        names = STAGE_ARTIFACTS[stage["kind"]]
                        write_json(output / f"pipeline_{stage['kind']}_receipt.json", {
                            "signature": signature, "completed": time.time(),
                            "hashes": {name: sha256(output / name) for name in names}})
                status["completed_stages"].append({"name": stage["name"], "action": action})
                if stage["kind"] == "semantic":
                    status["semantic_scoring"]["completed_seeds"].append(stage["seed"])
            if status["semantic_scoring"]["completed_seeds"] != list(SEEDS):
                raise ValueError("Queue ended without all semantic stages")
            status["semantic_scoring"]["status"] = "complete"
            status.update(status="complete", completed=time.time(), child_pid=None)
        except BaseException as exc:
            status.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                          error=repr(exc), returncode=getattr(exc, "returncode", 130 if isinstance(exc, KeyboardInterrupt) else 1),
                          failed=time.time(), child_pid=None)
            if child is not None and child.poll() is None:
                # Only the process group created by this runner, never a GPU PID
                # discovered by nvidia-smi or any other user's process group.
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass  # The child can exit between poll() and signalling.
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
            status["updated"] = time.time()
            write_json(status_path, status, replace=True)


def main(argv=None):
    global TRAIN_ROOT, EVAL_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Run foreground queue; never backgrounds itself")
    mode.add_argument("--dry-run", action="store_true", help="Default: print commands/checks only")
    mode.add_argument("--preflight", action="store_true", help="Read-only readiness check, no children or GPU query")
    mode.add_argument("--status", action="store_true", help="Read saved pipeline and seed statuses only")
    parser.add_argument("--resume", action="store_true", help="Explicitly allow fingerprint-matching interrupted training")
    parser.add_argument("--score-dir", type=Path, help="Internal CPU scoring stage; requires --execute and GPUs hidden")
    parser.add_argument("--train-root", type=Path, default=TRAIN_ROOT,
                        help="Separate output root for a versioned training experiment")
    parser.add_argument("--eval-root", type=Path, default=EVAL_ROOT,
                        help="Separate output root for its evaluations")
    args = parser.parse_args(argv)
    try:
        TRAIN_ROOT = checked_path(args.train_root)
        EVAL_ROOT = checked_path(args.eval_root)
        if TRAIN_ROOT == EVAL_ROOT:
            raise ValueError("Training and evaluation roots must differ")
        if args.status:
            for path in sorted(TRAIN_ROOT.glob("**/status.json")):
                print(json.dumps({"path": str(path), **read_json(path)}))
            return 0
        if args.score_dir:
            if not args.execute:
                parser.error("--score-dir requires --execute")
            summarize_evaluation(args.score_dir, args.score_dir / "cpu_scores.json")
            return 0
        stages = build_stages()
        if args.execute:
            run_pipeline(stages, args.resume)
            return 0
        report = preflight(stages, args.resume)
        print(json.dumps(report, indent=2))
        return 1 if report["errors"] else 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Pipeline failed before completion: {exc}", file=sys.stderr, flush=True)
        return getattr(exc, "returncode", 1) or 1
    except KeyboardInterrupt:
        print("Pipeline interrupted; see recorded stage/log. No completion claimed.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
