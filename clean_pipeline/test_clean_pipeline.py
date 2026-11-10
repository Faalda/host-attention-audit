"""CPU-only orchestration tests; no model imports, GPU queries, or real jobs."""

from contextlib import ExitStack
import ast
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import clean_gpu_guard as guard
import run_clean_pipeline as pipeline
import run_clean_semantics as semantic


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir="/tmp/opencode")))
        self.train = self.root / "results/clean_small"
        self.train.mkdir(parents=True)
        self.stack.enter_context(patch.object(pipeline, "ROOT", self.root))
        self.stack.enter_context(patch.object(guard, "ROOT", self.root))
        self.stack.enter_context(patch.object(pipeline, "TRAIN_ROOT", self.train))
        self.stack.enter_context(patch.object(pipeline, "EVAL_ROOT", self.root / "results/clean_eval"))
        self.spawn = self.stack.enter_context(patch.object(pipeline.subprocess, "Popen"))
        self.stack.enter_context(patch.object(pipeline.importlib.metadata, "version", return_value="test"))
        self.stack.enter_context(patch.object(semantic, "prerequisites", return_value=[]))
        for name in ("clean_pipeline/run_clean_captioner.py", "clean_pipeline/run_clean_evaluation.py",
                     "clean_pipeline/run_clean_semantics.py"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("from clean_gpu_guard import gpu_lease\n"
                                          "flag = '--wait-for-gpu'\nwith gpu_lease(): pass\n")
        for name in ("clean_pipeline/run_clean_pipeline.py", "clean_pipeline/clean_gpu_guard.py",
                     "inversion/run_rev_eval_metrics.py",
                     "analysis/rescore_rouge.py", "analysis/score_semantics.py", "results/dinov2_patch_features_15k.pt",
                     "results/revision/coco_caps_15k.json", "results/revision/inversion_protocol_n1000_s0.json",
                     "results/revision/swap_injection_n1000_s0.json"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture")

    def manifest(self, *, complete=False):
        folder = self.train / "seed42"
        folder.mkdir(exist_ok=True)
        inputs = ("clean_pipeline/run_clean_captioner.py", "results/dinov2_patch_features_15k.pt",
                  "results/revision/coco_caps_15k.json", "results/revision/inversion_protocol_n1000_s0.json")
        audit = dict(config={"epochs": 40}, hashes={p: pipeline.sha256(self.root / p) for p in inputs},
                     split_ids={}, unknown_rates={}, checks={}, alignment_limit="fixture")
        manifest = dict(**audit, seed=42, smoke=False, status="complete" if complete else "failed",
                        fingerprint=pipeline.fingerprint(dict(audit=audit, seed=42, smoke=False)))
        if complete:
            for name in pipeline.TRAIN_ARTIFACTS:
                (folder / name).write_text("artifact")
            manifest["artifact_hashes"] = {name: pipeline.sha256(folder / name) for name in pipeline.TRAIN_ARTIFACTS}
        pipeline.write_json(folder / "manifest.json", manifest, replace=True)
        return folder, manifest

    def evaluation(self):
        folder = pipeline.EVAL_ROOT / "seed42"
        folder.mkdir(parents=True)
        for name in pipeline.EVAL_ARTIFACTS:
            pipeline.write_json(folder / name, {})
        pipeline.write_json(folder / "complete.json", dict(all_inputs_unchanged=True,
                            all_model_parameters_frozen=True, model_state_sha256_before="abc",
                            model_state_sha256_after="abc"), replace=True)
        pipeline.write_json(folder / "planned_jobs.json", {"metadata": {
            "modes": list(pipeline.MODES), "matched_gain": True,
            "input_sha256": {str(self.root / "clean_pipeline/run_clean_evaluation.py"):
                              pipeline.sha256(self.root / "clean_pipeline/run_clean_evaluation.py")}}}, replace=True)
        pipeline.write_json(folder / "matched_gain.json", {"n_evaluation": 901,
                            "evaluation_vs_free": {"gain_rouge1": {}, "normalized_token_edit": {}}}, replace=True)
        return folder

    def test_exact_stage_order_and_commands(self):
        stages = pipeline.build_stages("python-test")
        self.assertEqual([s["name"] for s in stages], ["smoke_seed42", "train_seed42", "eval_seed42",
                         "semantics_seed42", "score_seed42", "train_seed7", "eval_seed7", "semantics_seed7", "score_seed7",
                         "train_seed13", "eval_seed13", "semantics_seed13", "score_seed13"])
        self.assertIn("--smoke", stages[0]["argv"])
        for stage in stages:
            self.assertNotIn("--resume", stage["argv"])
            if stage["kind"] == "score":
                self.assertEqual(stage["cuda"], "")
            else:
                self.assertEqual(stage["cuda"], "0")
                self.assertIn("--wait-for-gpu", stage["argv"])
            if stage["kind"] == "eval":
                self.assertIn("--matched-gain", stage["argv"])
                self.assertTrue(set(pipeline.MODES) <= set(stage["argv"]))
            if stage["kind"] == "semantic":
                self.assertIn("--execute", stage["argv"])
                self.assertIn(str(self.root / "clean_pipeline/run_clean_semantics.py"), stage["argv"])
        self.spawn.assert_not_called()

    def test_versioned_roots_reach_training_and_evaluation(self):
        train = self.root / "results/clean_small_stablekl"
        evaluation = self.root / "results/clean_eval_stablekl"
        train.mkdir()
        with patch.object(pipeline, "run_pipeline") as run:
            self.assertEqual(pipeline.main([
                "--execute", "--train-root", str(train), "--eval-root", str(evaluation),
            ]), 0)
        stages = run.call_args.args[0]
        for stage in stages:
            if stage["kind"] == "train":
                argv = stage["argv"]
                self.assertEqual(argv[argv.index("--output-root") + 1], str(train))
                self.assertTrue(Path(stage["folder"]).is_relative_to(train))
            else:
                self.assertTrue(Path(stage["folder"]).is_relative_to(evaluation))
        self.spawn.assert_not_called()

    def test_default_dry_run_preflight_and_status_never_spawn_or_write(self):
        before = set(self.root.rglob("*"))
        with patch("builtins.print"):
            for argv in ([], ["--dry-run"], ["--preflight"], ["--status"]):
                self.assertEqual(pipeline.main(argv), 0)
        self.assertEqual(set(self.root.rglob("*")), before)
        self.spawn.assert_not_called()

    def test_preflight_blocks_unintegrated_gpu_children(self):
        (self.root / "clean_pipeline/run_clean_evaluation.py").write_text("pass\n")
        report = pipeline.preflight(pipeline.build_stages())
        self.assertEqual(report["status"], "blocked")
        self.assertIn("gpu_lease", report["errors"][0])

    def test_blocked_only_training_folder_is_fresh_not_resume(self):
        folder = self.train / "seed42"
        folder.mkdir()
        pipeline.write_json(folder / "status.json", dict(status="blocked_gpu_busy", seed=42, smoke=False,
                                                         training_seconds=0))
        self.assertEqual(pipeline.training_action(folder, 42, False, resume=True), "run")
        (folder / "checkpoint.pt").touch()
        with self.assertRaisesRegex(ValueError, "Unrecognized partial"):
            pipeline.training_action(folder, 42, False, resume=True)

    def test_resume_is_explicit_and_fingerprint_verified(self):
        folder, manifest = self.manifest()
        with self.assertRaisesRegex(ValueError, "explicit --resume"):
            pipeline.training_action(folder, 42, False)
        self.assertEqual(pipeline.training_action(folder, 42, False, resume=True), "resume")
        manifest["fingerprint"] = "wrong"
        pipeline.write_json(folder / "manifest.json", manifest, replace=True)
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            pipeline.training_action(folder, 42, False, resume=True)

    def test_complete_training_checks_export_and_current_input_hashes(self):
        folder, _ = self.manifest(complete=True)
        self.assertEqual(pipeline.training_action(folder, 42, False), "skip_hash_verified")
        (folder / "plans_test.pt").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            pipeline.training_action(folder, 42, False)
        folder, _ = self.manifest(complete=True)
        (self.root / "clean_pipeline/run_clean_captioner.py").write_text("changed code")
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            pipeline.training_action(folder, 42, False, resume=True)

    def test_complete_evaluation_without_receipt_is_not_skipped(self):
        self.evaluation()
        with self.assertRaisesRegex(ValueError, "partial/unverified"):
            pipeline.stage_action(pipeline.build_stages()[2])

    def test_current_evaluator_filename_and_matching_schema(self):
        folder = self.evaluation()
        self.assertIn("planned_jobs.json", pipeline.EVAL_ARTIFACTS)
        self.assertNotIn("preregistered_jobs.json", pipeline.EVAL_ARTIFACTS)
        pipeline.evaluation_complete(folder)
        pipeline.write_json(folder / "matched_gain.json", {"evaluation_vs_free": {}}, replace=True)
        with self.assertRaisesRegex(ValueError, "matched-gain output schema"):
            pipeline.evaluation_complete(folder)

    def test_output_contract_matches_actual_evaluator_source_without_importing_it(self):
        tree = ast.parse(Path(__file__).with_name("run_clean_evaluation.py").read_text())
        strings = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        for name in pipeline.EVAL_ARTIFACTS:
            if not name.startswith("corruption_"):
                self.assertIn(name, strings)
        self.assertIn("evaluation_vs_free", strings)

    def test_evaluation_receipt_checks_all_artifacts_and_code(self):
        folder = self.evaluation()
        stage = pipeline.build_stages()[2]
        pipeline.write_json(folder / "pipeline_eval_receipt.json", {
            "signature": pipeline.receipt_signature(stage),
            "hashes": {name: pipeline.sha256(folder / name) for name in pipeline.EVAL_ARTIFACTS}})
        self.assertEqual(pipeline.stage_action(stage), "skip_hash_verified")
        (folder / "forward.json").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            pipeline.stage_action(stage)

    def test_partial_cpu_scores_fail_instead_of_overwriting(self):
        folder = self.evaluation()
        pipeline.write_json(folder / "cpu_scores.json", {})
        with self.assertRaisesRegex(ValueError, "partial/unverified"):
            pipeline.stage_action(pipeline.build_stages()[4])
        with self.assertRaises(FileExistsError):
            pipeline.write_json(folder / "cpu_scores.json", {"bad": "overwrite"})
        self.assertEqual(pipeline.read_json(folder / "cpu_scores.json"), {})

    def test_duplicate_queue_fails_before_preflight_or_spawn(self):
        with guard.exclusive_lock(self.train / "pipeline.lock"):
            with self.assertRaises(BlockingIOError):
                pipeline.run_pipeline(pipeline.build_stages())
        self.spawn.assert_not_called()

    def test_failed_child_records_code_log_pid_and_stops_next_commands(self):
        child = self.spawn.return_value
        child.pid = 1234
        child.wait.return_value = 17
        child.poll.return_value = 17
        with patch("builtins.print"):
            self.assertEqual(pipeline.main(["--execute"]), 17)
        self.spawn.assert_called_once()
        args, kwargs = self.spawn.call_args
        self.assertIn("--smoke", args[0])
        self.assertTrue(kwargs["start_new_session"])
        self.assertTrue(kwargs["pass_fds"])
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")
        status = pipeline.read_json(next(self.train.glob("pipeline_*/status.json")))
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["returncode"], 17)
        self.assertEqual(status["stage"], "smoke_seed42")
        self.assertTrue(Path(status["log"]).is_file())
        self.assertEqual(status["semantic_scoring"]["status"], "scheduled")
        self.assertEqual(status["semantic_scoring"]["completed_seeds"], [])

    def test_scoring_requires_hidden_gpus_and_does_not_import_scorers_early(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="0"):
            with self.assertRaisesRegex(ValueError, "CPU scoring requires"):
                pipeline.summarize_evaluation(self.root)

    def test_sequential_success_holds_queue_lock_but_never_parent_gpu_lease(self):
        stages = pipeline.build_stages()
        calls = {}

        def action(stage, resume=False):
            count = calls.get(stage["name"], 0)
            calls[stage["name"]] = count + 1
            return "skip_hash_verified" if stage["kind"] == "train" and count else "run"

        def spawn(command, **kwargs):
            with self.assertRaises(BlockingIOError), guard.exclusive_lock(self.train / "pipeline.lock"):
                pass
            stage = next(s for s in stages if s["argv"] == command)
            folder = Path(stage["folder"])
            folder.mkdir(parents=True, exist_ok=True)
            if stage["kind"] != "train":
                for name in pipeline.STAGE_ARTIFACTS[stage["kind"]]:
                    pipeline.write_json(folder / name, {})
            return Mock(pid=1234, wait=Mock(return_value=0))

        self.spawn.side_effect = spawn
        with patch.object(pipeline, "preflight", return_value={"errors": []}), \
                patch.object(pipeline, "stage_action", side_effect=action), \
                patch.object(pipeline, "evaluation_complete"), \
                patch.object(semantic, "verify_result"), \
                patch.object(guard, "gpu_lease", side_effect=AssertionError("Parent acquired GPU lease")), \
                patch("builtins.print"):
            pipeline.run_pipeline(stages)
        self.assertEqual([c.args[0] for c in self.spawn.call_args_list], [s["argv"] for s in stages])
        status = pipeline.read_json(next(self.train.glob("pipeline_*/status.json")))
        self.assertEqual(status["status"], "complete")
        self.assertEqual(len(status["completed_stages"]), 13)
        self.assertEqual(status["semantic_scoring"], {"status": "complete", "seeds": [42, 7, 13], "completed_seeds": [42, 7, 13]})
        self.assertEqual(pipeline.SEMANTIC["completed_seeds"], [])

    def test_interruption_signals_only_own_group_and_records_failure(self):
        child = self.spawn.return_value
        child.pid = 1234
        child.wait.side_effect = [KeyboardInterrupt(), 0]
        child.poll.return_value = None
        with patch.object(pipeline.os, "killpg") as kill, patch("builtins.print"):
            self.assertEqual(pipeline.main(["--execute"]), 130)
        kill.assert_called_once_with(1234, pipeline.signal.SIGTERM)
        status = pipeline.read_json(next(self.train.glob("pipeline_*/status.json")))
        self.assertEqual(status["status"], "interrupted")
        self.assertEqual(status["returncode"], 130)
        self.spawn.assert_called_once()

    def test_summarizer_calls_cpu_functions_and_includes_verified_semantics(self):
        folder = self.evaluation()
        pipeline.write_json(folder / "semantic_scores.json", {"status": "complete"})
        pipeline.write_json(folder / "forward.json", {"paired": {"forward": 1}}, replace=True)
        pipeline.write_json(folder / "swap_injection_n1000_s0.json", {
            "variants": ["free"], "records": [{"scores": {"free": {"gain_rouge1": -0.2}}}]}, replace=True)
        rouge = SimpleNamespace(protocol=Mock(return_value={"rouge": "scored"}), paired=Mock(return_value={"delta": -0.2}))
        metrics = SimpleNamespace(evaluate=Mock(return_value={"metrics": "scored"}))
        with patch.dict("sys.modules", {"analysis.rescore_rouge": rouge, "inversion.run_rev_eval_metrics": metrics}), \
                patch.object(semantic, "verify_result") as verified, \
                patch.dict(os.environ, CUDA_VISIBLE_DEVICES=""):
            summary = pipeline.summarize_evaluation(folder, folder / "cpu_scores.json")
        verified.assert_called_once_with(folder)
        self.assertEqual(rouge.protocol.call_count, 6)
        self.assertEqual(metrics.evaluate.call_count, 6)
        self.assertEqual(summary["semantic_scoring"]["status"], "complete")
        self.assertEqual(summary["semantic_scoring"]["sha256"], pipeline.sha256(folder / "semantic_scores.json"))
        self.assertEqual(summary["device"], "cpu")
        self.assertEqual(summary["counterfactual"]["free"]["gain_rouge1"]["delta"], -0.2)
        self.spawn.assert_not_called()

    def test_semantic_partial_output_and_receipt_hashes(self):
        folder = self.evaluation()
        stage = pipeline.build_stages()[3]
        self.assertEqual(pipeline.stage_action(stage), "run")
        pipeline.write_json(folder / "semantic_scores.json", {"status": "complete"})
        with self.assertRaisesRegex(ValueError, "partial/unverified"):
            pipeline.stage_action(stage)
        pipeline.write_json(folder / "pipeline_semantic_receipt.json", {
            "signature": pipeline.receipt_signature(stage),
            "hashes": {"semantic_scores.json": pipeline.sha256(folder / "semantic_scores.json")}})
        with patch.object(semantic, "verify_result") as verified:
            self.assertEqual(pipeline.stage_action(stage), "skip_hash_verified")
            verified.assert_called_once_with(folder)
        (folder / "semantic_scores.json").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            pipeline.stage_action(stage)

    def test_semantic_failure_never_claims_complete_or_launches_cpu_scoring(self):
        stages = pipeline.build_stages()
        semantic_stage = stages[3]
        child = self.spawn.return_value
        child.pid = 1234
        child.wait.return_value = 1
        child.poll.return_value = 1
        with patch.object(pipeline, "build_stages", return_value=[semantic_stage]), \
                patch("builtins.print"):
            self.assertEqual(pipeline.main(["--execute"]), 1)
        self.spawn.assert_called_once()
        status = pipeline.read_json(next(self.train.glob("pipeline_*/status.json")))
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["semantic_scoring"]["completed_seeds"], [])
        self.assertNotEqual(status["semantic_scoring"]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
