"""CPU-only clean semantic schema, canonical bookkeeping and mocked execution."""

from contextlib import ExitStack, contextmanager, nullcontext
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import clean_gpu_guard as guard
import run_clean_pipeline as pipeline
import run_clean_semantics as clean


def fixture(n=1000):
    refs = ["A cat.", "A cat.", "cat sits", "small cat", "cat on mat"]
    records = [{"image": i, "feature_index": 14000 + i, "refs": refs.copy(), "baseline": "A cat.",
                "mismatch_source": (i + 1) % n, "mismatch_target": "cat sits",
                "variants": {v: {"true": refs.copy(), "mismatch": "shuffled cat"} for v in clean.VARIANTS}}
               for i in range(n)]
    protocol = {"n_images": n, "partial": False, "evaluated_true_slots": list(range(5)),
                "variants": list(clean.VARIANTS), "records": records}
    forward = {"n_images": n, "image_ids": list(range(n)), "refs": [refs.copy() for _ in range(n)],
               "caps": {"guided_denoised": ["A cat."] * n, "guided_uniform": ["uniform cat"] * n,
                        "free": ["FREE unique!"] * n}}
    return protocol, forward


class SchemaTests(unittest.TestCase):
    def test_clean1000_validates_without_canonical5000_validator(self):
        protocol, forward = fixture()
        self.assertEqual(len(clean.validate_inputs(protocol, forward)), 1000)

    def test_partial_variants_slots_forward_ids_donors_and_alignment_rejected(self):
        for defect in ("partial", "cohort", "slots", "variants", "true", "shuffle", "forward", "ids", "donors", "baseline", "feature"):
            protocol, forward = fixture()
            rec = protocol["records"][0]
            if defect == "partial":
                protocol["partial"] = True
            elif defect == "cohort":
                protocol["records"].pop()
            elif defect == "slots":
                protocol["evaluated_true_slots"] = [0, 1]
            elif defect == "variants":
                protocol["variants"].pop()
            elif defect == "true":
                rec["variants"]["free"]["true"][0] = None
            elif defect == "shuffle":
                rec["variants"]["hidden"]["mismatch"] = None
            elif defect == "forward":
                forward["caps"]["free"].pop()
            elif defect == "ids":
                forward["image_ids"].reverse()
            elif defect == "donors":
                rec["mismatch_source"] = 0
            elif defect == "baseline":
                rec["baseline"] = "old baseline"
            elif defect == "feature":
                rec["feature_index"] = 0
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                clean.validate_inputs(protocol, forward)

    def test_import_has_no_cuda_environment_upgrade_or_torch(self):
        code = ("import os,sys; os.environ['CUDA_VISIBLE_DEVICES']=''; "
                "import run_clean_semantics,run_clean_pipeline; "
                "assert os.environ['CUDA_VISIBLE_DEVICES']==''; "
                "assert 'torch' not in sys.modules and 'score_semantics' not in sys.modules")
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(__file__).resolve().parent,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_real_canonical_collection_aggregation_and_bootstrap_bookkeeping(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="0"):
            canonical = clean.load_canonical()  # Pure NumPy/ROUGE imports; never make_scorer.
        protocol, forward = fixture(2)
        records = protocol["records"]
        pairs = clean.collect_pairs(canonical, records, forward)
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertIn(("FREE unique!", "A cat."), pairs)
        self.assertIn(("shuffled cat", "cat sits"), pairs)
        scores = {p: float(p[0] == p[1]) for p in pairs}
        original = canonical.paired
        with patch.object(canonical, "paired", wraps=original) as paired:
            result = clean.aggregate(canonical, records, forward, scores)
        self.assertEqual(paired.call_count, 23)
        for call in paired.call_args_list:
            self.assertEqual(call.kwargs, {"resamples": 10000, "seed": 0})
        self.assertEqual(result["protocol"]["variants"]["free"]["heldout"]["inv"], 0.4)
        self.assertEqual(set(result["forward"]["systems"]), set(clean.SYSTEMS))
        self.assertEqual(result["forward"]["paired"]["denoised_minus_free"]["delta"], 1.0)


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir="/tmp/opencode")))
        for module in (clean, pipeline, guard):
            self.stack.enter_context(patch.object(module, "ROOT", self.root))
        self.folder = self.root / "results/clean_eval/seed42"
        self.folder.mkdir(parents=True)
        for name in clean.CODE_FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("source fixture")
        snapshot = self.root / "snapshot"
        snapshot.mkdir()
        for name in clean.SNAPSHOT_FILES:
            (snapshot / name).write_text("model fixture")
        self.stack.enter_context(patch.object(clean, "SNAPSHOT", snapshot))
        package = self.root / "package"
        (package / "bert_score").mkdir(parents=True)
        for name in ("utils.py", "scorer.py"):
            (package / "bert_score" / name).write_text("canonical fixture")
        self.stack.enter_context(patch.object(clean.importlib.metadata, "distribution",
                                             return_value=SimpleNamespace(locate_file=lambda name: package / name)))
        self.stack.enter_context(patch.object(clean.importlib.metadata, "version",
                                             side_effect=lambda name: "0.3.13" if name == "bert-score" else "test"))
        self.stack.enter_context(patch.dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
        for name in pipeline.EVAL_ARTIFACTS:
            pipeline.write_json(self.folder / name, {})
        pipeline.write_json(self.folder / "complete.json", dict(all_inputs_unchanged=True,
                            all_model_parameters_frozen=True, model_state_sha256_before="hash",
                            model_state_sha256_after="hash"), replace=True)
        pipeline.write_json(self.folder / "planned_jobs.json", {"metadata": {
            "modes": list(pipeline.MODES), "matched_gain": True,
            "input_sha256": {"clean_pipeline/run_clean_semantics.py": pipeline.sha256(self.root / "clean_pipeline/run_clean_semantics.py")}}}, replace=True)
        pipeline.write_json(self.folder / "matched_gain.json", {"n_evaluation": 901,
                            "evaluation_vs_free": {"gain_rouge1": {}, "normalized_token_edit": {}}}, replace=True)
        protocol, forward = fixture()
        pipeline.write_json(self.folder / "inversion_protocol_n1000_s0.json", protocol, replace=True)
        pipeline.write_json(self.folder / "forward.json", forward, replace=True)
        self.events = []

        @contextmanager
        def lease(**kwargs):
            self.events.append("lease_enter")
            try:
                yield {"gpu_uuid": "GPU-fixture", "index": 0}
            finally:
                self.events.append("lease_exit")

        self.lease = self.stack.enter_context(patch.object(clean, "gpu_lease", side_effect=lease))
        self.param = SimpleNamespace(requires_grad=True, grad=None)
        model = SimpleNamespace(training=False)
        model.eval = lambda: model

        def freeze(flag):
            self.param.requires_grad = flag
            self.events.append("freeze")
            return model

        model.requires_grad_ = freeze
        model.modules = lambda: [model]
        model.parameters = lambda: [self.param]

        def score(candidates, references, **kwargs):
            self.events.append("score")
            self.assertFalse(self.param.requires_grad)
            return None, None, SimpleNamespace(tolist=lambda: [0.5] * len(candidates))

        self.scorer = SimpleNamespace(_model=model, _tokenizer=SimpleNamespace(is_fast=False), score=Mock(side_effect=score))

        def make_scorer(**kwargs):
            self.events.append("make_scorer")
            return self.scorer, True

        self.canonical = SimpleNamespace(make_scorer=Mock(side_effect=make_scorer),
            smoke=Mock(side_effect=lambda scorer: self.events.append("smoke") or {"passed": True}),
            audit_tokenization=Mock(side_effect=lambda *args, **kwargs: self.events.append("audit") or {"passed": True}))
        self.loader = self.stack.enter_context(patch.object(clean, "load_canonical",
                                              side_effect=lambda: self.events.append("import") or self.canonical))
        self.stack.enter_context(patch.object(clean, "collect_pairs", return_value=[("A cat.", "cat sits")]))
        row = {"n_images": 1000, "per_image": {"inv": [0.5] * 1000, "base": [0.5] * 1000},
               "inv": 0.5, "base": 0.5, "delta": 0.0, "ci95": [0.0, 0.0],
               "inv_ci95": [0.5, 0.5], "base_ci95": [0.5, 0.5]}
        self.datasets = {"protocol": {"n_images": 1000, "image_ids": list(range(1000)),
                         "variants": {v: {p: row for p in clean.PROTOCOLS} for v in clean.VARIANTS}},
                         "forward": {"n_images": 1000, "image_ids": list(range(1000)),
                           "systems": {name: {"per_image": [0.5] * 1000, "mean": 0.5, "ci95": [0.5, 0.5]}
                                       for name in clean.SYSTEMS},
                           "paired": {"denoised_minus_free": row, "denoised_minus_guided_uniform": row}}}
        self.aggregation = self.stack.enter_context(patch.object(clean, "aggregate", return_value=self.datasets))
        torch = SimpleNamespace(inference_mode=nullcontext, version=SimpleNamespace(cuda="fixture"),
                                cuda=SimpleNamespace(get_device_name=lambda i: "mock GPU", max_memory_allocated=lambda i: 0))
        self.stack.enter_context(patch.dict(sys.modules, torch=torch))

    def test_call_order_canonical_path_and_provenance(self):
        clean.execute(self.folder, wait=True)
        self.assertEqual(self.events, ["lease_enter", "import", "make_scorer", "freeze", "smoke", "audit", "score", "lease_exit"])
        self.lease.assert_called_once_with(wait=True, minimum_free_mib=20000)
        self.canonical.make_scorer.assert_called_once_with(snapshot=clean.SNAPSHOT)
        self.scorer.score.assert_called_once_with(["A cat."], ["cat sits"], batch_size=128, verbose=True)
        result = clean.verify_result(self.folder)
        self.assertEqual(result["metadata"]["config"], clean.CONFIG)
        self.assertEqual(result["metadata"]["gpu_uuid"], "GPU-fixture")
        self.assertFalse(result["metadata"]["training"])
        self.assertFalse(result["metadata"]["model_downloads"])
        self.assertEqual(set(result["metadata"]["snapshot_sha256"]), set(clean.SNAPSHOT_FILES))
        self.assertEqual(result["metadata"]["code_sha256"]["analysis/score_semantics.py"],
                         pipeline.sha256(self.root / "analysis/score_semantics.py"))
        with self.assertRaises(FileExistsError):
            clean.execute(self.folder)
        self.lease.assert_called_once()

    def test_default_and_preflight_do_not_import_or_acquire_gpu(self):
        for mode in ([], ["--preflight"], ["--dry-run"]):
            with patch("builtins.print"):
                self.assertEqual(clean.main(["--eval-dir", str(self.folder), *mode]), 0)
        self.lease.assert_not_called()
        self.loader.assert_not_called()
        self.assertFalse((self.folder / "semantic_scores.json").exists())
        self.assertFalse((self.folder / "semantic.lock").exists())

    def test_invalid_visible_devices_never_upgrade_or_acquire(self):
        for value in ("", "1", "0,1"):
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=value):
                with self.assertRaisesRegex(ValueError, "no environment upgrade"):
                    clean.execute(self.folder)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], value)
        self.lease.assert_not_called()
        self.loader.assert_not_called()

    def test_missing_clean_outputs_or_snapshot_fail_before_gpu(self):
        with self.assertRaises(OSError):
            clean.execute(self.root / "missing")
        (clean.SNAPSHOT / "model.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "Missing offline"):
            clean.execute(self.folder)
        self.lease.assert_not_called()
        self.loader.assert_not_called()

    def test_busy_and_oom_never_publish_semantic_results(self):
        self.lease.side_effect = guard.GPUBusyError("occupied")
        with self.assertRaises(guard.GPUBusyError):
            clean.execute(self.folder)
        self.loader.assert_not_called()
        self.assertFalse((self.folder / "semantic_scores.json").exists())
        self.lease.side_effect = lambda **kwargs: nullcontext({"gpu_uuid": "GPU-test", "index": 0})
        self.canonical.make_scorer.side_effect = RuntimeError("CUDA out of memory")
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            clean.execute(self.folder)
        self.assertFalse((self.folder / "semantic_scores.json").exists())
        self.assertEqual(pipeline.read_json(self.folder / "semantic_status.json")["status"], "failed")

    def test_failed_token_audit_or_nonfinite_scores_are_not_published(self):
        self.canonical.audit_tokenization.side_effect = None
        self.canonical.audit_tokenization.return_value = {"passed": False}
        with self.assertRaisesRegex(ValueError, "audit did not pass"):
            clean.execute(self.folder)
        self.scorer.score.assert_not_called()
        self.canonical.audit_tokenization.return_value = {"passed": True}
        self.scorer.score.side_effect = lambda *args, **kwargs: (None, None, SimpleNamespace(tolist=lambda: [float("nan")]))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            clean.execute(self.folder)
        self.assertFalse((self.folder / "semantic_scores.json").exists())

    def test_changed_source_not_published_and_published_hashes_rechecked(self):
        original = (self.root / "analysis/score_semantics.py").read_text()

        def changed(*args):
            (self.root / "analysis/score_semantics.py").write_text("changed")
            return self.datasets

        self.aggregation.side_effect = changed
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            clean.execute(self.folder)
        self.assertFalse((self.folder / "semantic_scores.json").exists())
        (self.root / "analysis/score_semantics.py").write_text(original)
        self.aggregation.side_effect = None
        clean.execute(self.folder)
        (clean.SNAPSHOT / "vocab.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "model hash mismatch"):
            clean.verify_result(self.folder)

    def test_fake_completion_metadata_rejected(self):
        clean.execute(self.folder)
        path = self.folder / "semantic_scores.json"
        original = pipeline.read_json(path)
        for key, value in (("training", True), ("model_downloads", True), ("model_frozen", False), ("gpu_uuid", "")):
            result = copy.deepcopy(original)
            result["metadata"][key] = value
            pipeline.write_json(path, result, replace=True)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Uncertified"):
                clean.verify_result(self.folder)

    def test_incomplete_aggregation_is_never_published_as_complete(self):
        self.datasets["forward"]["systems"].pop("free")
        with self.assertRaisesRegex(ValueError, "Incomplete semantic forward controls"):
            clean.execute(self.folder)
        self.assertFalse((self.folder / "semantic_scores.json").exists())
        self.assertEqual(pipeline.read_json(self.folder / "semantic_status.json")["status"], "failed")


if __name__ == "__main__":
    unittest.main()
