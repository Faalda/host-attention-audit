"""CPU-only regression tests; no models, downloads, or CUDA initialization."""

import copy
import json
import os
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from analysis import score_corruption_semantics as scoring


def record(i=0):
    return {"image": i, "feature_index": 14000 + i, "refs": ["same", "same", "r2", "r3", "r4"],
            "baseline": "base", "clean_baseline": "base", "mismatch_source": (i + 1) % 1000,
            "mismatch_target": "donor", "variants": {
                "free": {"true": ["true0", "true1", None, None, None], "mismatch": "shuffle"},
                "repair": {"true": ["base", "base", None, None, None], "mismatch": "base"}}}


def xml(process="", memory="24000 MiB"):
    return f"""<nvidia_smi_log><gpu><uuid>GPU-test-uuid</uuid>
    <fb_memory_usage><free>{memory}</free></fb_memory_usage>
    <utilization><gpu_util>0 %</gpu_util></utilization><processes>{process}</processes>
    </gpu></nvidia_smi_log>"""


class CorruptionSemanticsTests(unittest.TestCase):
    def test_parent_import_has_no_torch_or_cuda_environment_mutation(self):
        command = [sys.executable, "-B", "-c",
                   "import os,sys; os.environ['CUDA_VISIBLE_DEVICES']='sentinel'; "
                   "import analysis.score_corruption_semantics; "
                   "assert 'torch' not in sys.modules; assert os.environ['CUDA_VISIBLE_DEVICES']=='sentinel'"]
        subprocess.run(command, cwd=scoring.ROOT, check=True)

    def test_real_canonical_import_restores_gpu1_before_torch_import(self):
        command = [sys.executable, "-B", "-c",
                   "import os,sys; from analysis.score_corruption_semantics import load_canonical; "
                   "m=load_canonical('GPU-physical1'); "
                   "assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-physical1'; "
                   "assert 'torch' not in sys.modules; assert callable(m.make_scorer)"]
        subprocess.run(command, cwd=scoring.ROOT, check=True)

    def test_import_error_also_restores_visibility(self):
        def fail(name):
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            raise RuntimeError("broken import")
        with patch.dict(os.environ), patch.object(scoring.importlib, "import_module", side_effect=fail):
            with self.assertRaises(RuntimeError):
                scoring.load_canonical("GPU-physical1")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-physical1")

    def test_reject_cuda_initialized_before_import(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))
        with patch.dict(sys.modules, torch=torch):
            with self.assertRaisesRegex(ValueError, "already initialized"):
                scoring.load_canonical("GPU-physical1")

    def test_actual_uuid_not_hardcoded_physical_zero(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1,
                    set_per_process_memory_fraction=Mock(),
                    get_device_properties=lambda index: SimpleNamespace(uuid="physical1", name="test", total_memory=48 * 1024**3)))
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="GPU-physical1"):
            actual = scoring.actual_device(torch, {"index": 1, "gpu_uuid": "GPU-physical1"})
            self.assertEqual(actual["physical_gpu"], 1)
            self.assertEqual(actual["torch_device_uuid"], "GPU-physical1")
            torch.cuda.set_per_process_memory_fraction.assert_called_once_with(0.35, device=0)
            self.assertEqual(actual["memory_cap_fraction"], 0.35)
            self.assertEqual(actual["memory_cap_bytes"], int(48 * 1024**3 * 0.35))
            with self.assertRaisesRegex(ValueError, "UUID differs"):
                scoring.actual_device(torch, {"index": 0, "gpu_uuid": "GPU-physical0"})

    def test_query_includes_graphics_and_unowned_processes(self):
        process = "<process_info><pid>3349205</pid><type>G</type><process_name>root-job</process_name></process_info>"
        with patch.object(scoring.subprocess, "run", return_value=SimpleNamespace(stdout=xml(process))) as run:
            actual = scoring.query_gpu(1)
            self.assertEqual(actual["processes"][0]["pid"], 3349205)
            self.assertEqual(actual["processes"][0]["type"], "G")
            self.assertEqual(run.call_args.args[0], ["nvidia-smi", "-i", "1", "-q", "-x"])

    def test_query_fails_closed(self):
        for text in (xml(memory="N/A"), xml().replace("<processes></processes>", ""),
                     xml("N/A"), xml("<unknown/>"), "bad XML"):
            with self.subTest(text=text), patch.object(scoring.subprocess, "run", return_value=SimpleNamespace(stdout=text)):
                with self.assertRaises(scoring.GPUQueryError):
                    scoring.query_gpu(0)

    def test_gpu0_uses_existing_lock_and_busy_does_not_yield(self):
        snapshot = {"processes": [{"pid": 3349205}], "free_mib": 24000, "utilization_percent": 0}
        with patch.object(scoring, "exclusive_lock") as lock, patch.object(scoring, "query_gpu", return_value=snapshot):
            with self.assertRaises(TimeoutError):
                with scoring.gpu_lease(0, timeout=0):
                    self.fail("Busy GPU must not yield")
            lock.assert_called_once_with(scoring.LOCK_PATH)

    def test_gpu1_separate_lock_body_error_not_retried(self):
        snapshot = {"processes": [], "free_mib": 24000, "utilization_percent": 0, "gpu_uuid": "GPU-one"}
        with patch.object(scoring, "exclusive_lock") as lock, patch.object(scoring, "query_gpu", return_value=snapshot) as query:
            with self.assertRaisesRegex(RuntimeError, "body failed"):
                with scoring.gpu_lease(1):
                    raise RuntimeError("body failed")
            lock.assert_called_once_with(scoring.LOCK_PATH.with_name("gpu1.lock"))
            self.assertEqual(query.call_count, 1)

    def test_shared_requires_24000_mib_and_records_existing_pids(self):
        for free in (23999, 24000):
            snapshot = {"processes": [{"pid": 123, "type": "G", "name": "root-job"}],
                        "free_mib": free, "utilization_percent": 99, "gpu_uuid": "GPU-one"}
            with self.subTest(free=free), patch.object(scoring, "exclusive_lock"), \
                    patch.object(scoring, "query_gpu", return_value=snapshot):
                if free < 24000:
                    with self.assertRaises(TimeoutError):
                        with scoring.gpu_lease(1, timeout=0, allow_shared=True):
                            self.fail("Insufficient free memory")
                else:
                    with scoring.gpu_lease(1, timeout=0, allow_shared=True) as (lease, conflicts):
                        self.assertTrue(lease["allow_shared"])
                        self.assertEqual(lease["minimum_free_mib"], 24000)
                        self.assertEqual(lease["existing_pids"], [123])
                        self.assertEqual(lease["processes"], snapshot["processes"])
                        self.assertEqual(conflicts, [])

    def test_shared_still_respects_cooperative_lock(self):
        with patch.object(scoring, "exclusive_lock") as lock, patch.object(scoring, "query_gpu") as query:
            lock.return_value.__enter__.side_effect = BlockingIOError()
            with self.assertRaises(TimeoutError):
                with scoring.gpu_lease(1, timeout=0, allow_shared=True):
                    self.fail("Cannot bypass our lock")
            query.assert_not_called()

    def test_oom_retries_only_predeclared_batches_same_complete_pairs(self):
        class OOM(RuntimeError):
            pass
        torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OOM, empty_cache=Mock()))
        pairs = [("a", "b"), ("b", "a")]
        with patch.object(scoring, "score_pairs", side_effect=[OOM(), OOM(), [0.1, 0.2]]) as score:
            values, batch, attempts = scoring.score_with_retries(torch, None, pairs, 128)
        self.assertEqual(values, [0.1, 0.2])
        self.assertEqual(batch, 32)
        self.assertEqual([c.args[2] for c in score.call_args_list], [128, 64, 32])
        self.assertTrue(all(c.args[1] is pairs for c in score.call_args_list))
        self.assertEqual([a["status"] for a in attempts], ["cuda_oom", "cuda_oom", "complete"])
        self.assertEqual(torch.cuda.empty_cache.call_count, 2)
        with patch.object(scoring, "score_pairs", side_effect=OOM()):
            with self.assertRaises(OOM):
                scoring.score_with_retries(torch, None, pairs, 32)
        with patch.object(scoring, "score_pairs", side_effect=ValueError("bad scores")) as score:
            with self.assertRaises(ValueError):
                scoring.score_with_retries(torch, None, pairs, 128)
            self.assertEqual(score.call_count, 1)

    def test_validate_exact_two_slots_and_reference_free_repair(self):
        records = [record(i) for i in range(1000)]
        data = {"partial": False, "n_images": 1000, "evaluated_true_slots": [0, 1],
                "variants": ["free", "repair"], "metadata": {"corruption_fraction": 0.25}, "records": records}
        original = copy.deepcopy(records)
        self.assertEqual(scoring.validate_corruption(data, original, "0.25"), records)
        for mutation in (lambda d: d["records"].pop(),
                         lambda d: d["records"][0]["variants"]["free"]["true"].__setitem__(2, "extra"),
                         lambda d: d["records"][0]["variants"]["repair"]["true"].__setitem__(1, "target"),
                         lambda d: d["records"][0]["refs"].__setitem__(0, "different")):
            bad = copy.deepcopy(data)
            mutation(bad)
            with self.assertRaises(ValueError):
                scoring.validate_corruption(bad, original, "0.25")

    def test_duplicate_reference_strings_retained_and_no_best5(self):
        r = record()
        def score(a, b):
            return 1.0 if a == "true0" and b == "same" else 0.0
        rows = scoring.image_scores(r, "free", score)
        self.assertEqual(rows["heldout"], (0.5, 0.0))
        self.assertEqual(rows["single_ref"], (0.5, 0.0))
        self.assertFalse(any("best5" in key for key in rows))

    def test_allrefs_and_heldout_are_distinct(self):
        r = record()
        r["refs"] = ["r0", "r1", "r2", "r3", "r4"]
        score = lambda a, b: float(a == "base" and b == "r0")
        rows = scoring.image_scores(r, "repair", score)
        self.assertEqual(rows["all_refs"], (1.0, 1.0))
        self.assertEqual(rows["heldout"], (0.5, 0.5))

    def test_full_ordered_pair_dedup_and_single_upstream_call(self):
        datasets = {"corruption/0": [record(), record()]}
        pairs = scoring.collect_pairs(datasets)
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertIn(("shuffle", "donor"), pairs)
        self.assertNotIn(("donor", "shuffle"), pairs)
        from unittest.mock import Mock
        scorer = Mock()
        scorer.score.return_value = (None, None, SimpleNamespace(tolist=lambda: [0.5] * len(pairs)))
        self.assertEqual(scoring.score_pairs(scorer, pairs, 128), [0.5] * len(pairs))
        scorer.score.assert_called_once()

    def test_reaggregation_exact_and_signed_repair_gain(self):
        zero, corrupt = record(), record()
        corrupt["baseline"] = "corrupt"
        datasets = {"corruption/0": [zero], "corruption/1": [corrupt]}
        pairs = scoring.collect_pairs(datasets)
        values = [0.9 if a == "base" else 0.8 if a == "corrupt" else 0.95 for a, _ in pairs]
        first = scoring.aggregate_seed(datasets, pairs, values)
        second = scoring.aggregate_seed(datasets, pairs, values)
        self.assertEqual(json.dumps(first), json.dumps(second))
        self.assertAlmostEqual(first["corruption/1/repair/heldout"]["delta"], 0.1)
        self.assertEqual(first["corruption/1/repair/heldout"], first["corruption/1/baseline_loss_heldout"])
        self.assertEqual(first["corruption/0/repair/heldout"]["delta"], 0)
        self.assertEqual(first["corruption/1/forward_restoration_all_refs"], first["corruption/1/repair/all_refs"])

    def test_receipt_hash_failure_prevents_loading_captions(self):
        receipt = {"hashes": {name: "bad" for name in scoring.EVAL_ARTIFACTS}}
        with patch.object(scoring, "read_json", return_value=receipt) as read, \
                patch.object(scoring, "verify_hashes", side_effect=ValueError("Hash mismatch")):
            with self.assertRaisesRegex(ValueError, "Hash mismatch"):
                scoring.load_inputs(42)
            self.assertEqual(read.call_count, 1)

    def test_seed_average_precedes_1000_image_bootstrap_not_3000_iid(self):
        from analysis.rescore_rouge import paired
        protocols = ("forward_restoration_all_refs", "repair/heldout", "free/heldout", "free_minus_repair_heldout")
        arrays = ([0.0, 3.0] * 500, [3.0, 0.0] * 500, [0.0] * 1000)
        results = []
        for seed, values in zip(scoring.SEEDS, arrays):
            row = paired(values, [0.0] * 1000)
            rows = {f"corruption/{fraction}/{protocol}": row for fraction in scoring.FRACTIONS for protocol in protocols}
            results.append({"status": "complete", "config": scoring.CONFIG, "provenance": {},
                            "input_sha256": {}, "seed": seed, "image_ids": list(range(1000)),
                            "ordered_pairs_sha256": scoring.fingerprint([]), "pair_f1": [],
                            "comparisons": rows, "execution": {"physical_gpu": 1, "seed_runtime_seconds": 123}})
        with patch.object(scoring, "provenance", return_value={}), \
                patch.object(scoring, "read_json", side_effect=results), \
                patch.object(scoring, "load_inputs", return_value=({"strict_k01": [record()]}, {})), \
                patch.object(scoring, "collect_pairs", return_value=[]), \
                patch.object(scoring, "aggregate_seed", side_effect=[r["comparisons"] for r in results]), \
                patch.object(scoring, "write_json") as write, patch("builtins.print"):
            scoring.summary()
        summary = write.call_args.args[1]
        for row in summary["comparisons"].values():
            self.assertEqual(row["n_images"], 1000)
            self.assertEqual(row["delta"], 1.0)
            self.assertEqual(row["ci95"], [1.0, 1.0])
        for seed, result in zip(scoring.SEEDS, results):
            expected = scoring.fingerprint({k: v for k, v in result.items() if k != "execution"})
            self.assertEqual(summary["seed_scientific_sha256"][str(seed)], expected)


if __name__ == "__main__":
    unittest.main()
