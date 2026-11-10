"""CPU tests for slot alignment, frozen scoring and GPU fail-closed guards."""

import copy
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_visual_headroom as target


def record():
    return dict(refs=["Repeat!", "Repeat!", "Third.", "Fourth", "Fifth"],
                mismatch_target="Donor Caption!", baseline="fault caption",
                forward={v: v + " caption" for v in target.FORWARD},
                variants={v: dict(true=[v + str(k) for k in range(5)], mismatch=v + " shuffled")
                          for v in target.VARIANTS})


class ScoringTests(unittest.TestCase):
    def test_duplicate_slots_and_baseline_alignment(self):
        r = record()
        preds, bases, refs = target.protocol_jobs([r], "free", "heldout")
        self.assertEqual(preds, ["free0", "free1", "free2", "free3", "free4"])
        self.assertEqual(refs[0], ["Repeat!", "Third.", "Fourth", "Fifth"])
        self.assertEqual(refs[2], ["Repeat!", "Repeat!", "Fourth", "Fifth"])
        self.assertEqual(bases, ["fault caption"] * 5)
        for variant in target.VARIANTS:
            _, other, other_refs = target.protocol_jobs([r], variant, "heldout")
            self.assertEqual((other, other_refs), (bases, refs))
        _, native, native_refs = target.protocol_jobs([r], "free", "heldout", "intact")
        self.assertEqual(native, ["intact caption"] * 5)
        self.assertEqual(native_refs, refs)

    def test_single_shuffle_original_text_and_five_job_mean(self):
        r = record()
        self.assertEqual(target.protocol_jobs([r], "free", "single_ref")[2], [[s] for s in r["refs"]])
        self.assertEqual(target.protocol_jobs([r], "free", "mismatch_target")[2], [["Donor Caption!"]])
        self.assertEqual(target.protocol_jobs([r], "free", "mismatch_grounding")[2], [r["refs"]])
        metric = lambda caps, refs: [int(c[-1]) if c.startswith("free") else 10 for c in caps]
        a, b = target.protocol_values(metric, [r], "free", "heldout")
        np.testing.assert_array_equal(a, [2.0])
        np.testing.assert_array_equal(b, [10.0])

    def test_common_baseline_all_variants(self):
        metric = lambda caps, refs: np.array([len(c) + len(rs) / 10 for c, rs in zip(caps, refs)])
        rows = target.comparisons_for_seed(metric, [record()])
        for p in target.PROTOCOLS:
            for baseline in ("fault", "intact"):
                for variant in target.VARIANTS:
                    np.testing.assert_array_equal(rows[f"protocol/{variant}/{p}/vs_{baseline}"][1],
                                                  rows[f"protocol/free/{p}/vs_{baseline}"][1])

    def test_gpu_uuid_memory_cap_and_rejections(self):
        gpu = dict(physical_gpu=1, uuid="GPU-test-one", total_mib=49140, free_mib=24000)
        cuda = Mock()
        cuda.device_count.return_value = 1
        cuda.get_device_properties.return_value = SimpleNamespace(uuid="test-one", name="test", total_memory=10000)
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"]):
            metadata = target.configure_device(SimpleNamespace(cuda=cuda), gpu)
            cuda.set_per_process_memory_fraction.assert_called_once_with(0.35, device=0)
            self.assertEqual(metadata["memory_cap_bytes"], 3500)
            self.assertEqual(metadata["physical_gpu"], 1)
            cuda.get_device_properties.return_value.uuid = "other"
            with self.assertRaisesRegex(ValueError, "UUID mismatch"):
                target.configure_device(SimpleNamespace(cuda=cuda), gpu)
            cuda.device_count.return_value = 2
            with self.assertRaisesRegex(ValueError, "Exactly one"):
                target.configure_device(SimpleNamespace(cuda=cuda), gpu)
            with self.assertRaisesRegex(ValueError, "24000"):
                target.configure_device(SimpleNamespace(cuda=cuda), dict(gpu, free_mib=23999))
        cuda.set_per_process_memory_fraction.assert_called_once()

    def test_legacy_import_visibility_restored_before_cuda(self):
        modules = [SimpleNamespace(__file__=str(target.HERE / n))
                   for n in ("score_semantics.py", "rescore_clean_cider.py")]

        def legacy(name):
            os.environ["CUDA_VISIBLE_DEVICES"] = "0" if name == "score_semantics" else ""
            return modules[0 if name == "score_semantics" else 1]

        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="GPU-test-one"), \
                patch.object(target.importlib, "import_module", side_effect=legacy):
            target.load_metrics("GPU-test-one")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-test-one")
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="GPU-test-one"), \
                patch.object(target.importlib, "import_module", side_effect=RuntimeError("failure")):
            with self.assertRaises(RuntimeError):
                target.load_metrics("GPU-test-one")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-test-one")

    def test_frozen_cider_forward_and_duplicate_refs(self):
        _, cider = target.load_metrics("GPU-test-one")
        refs = [["a cat sits on a mat", "a cat sits on a mat", "cat on mat", "a small cat", "the cat sits"],
                ["a dog runs through grass", "dog in grass", "a running dog", "dog on a field", "a dog runs"]]
        metric = cider.FixedIDFCider(refs)
        caps = [rs[0] for rs in refs]
        records = [dict(refs=rs, forward=dict(intact=c)) for c, rs in zip(caps, refs)]
        values = metric.score(caps, refs)
        audit = target.verify_forward_cider(metric, records, dict(caps=dict(guided_denoised=caps)), values, cider)
        self.assertTrue(audit["pure_intact_caption_equality"])
        self.assertLessEqual(audit["standard_forward_max_abs_error"], 1e-12)
        before = dict(metric.document_frequency)
        metric.score(caps, [rs[1:] for rs in refs])
        self.assertEqual(dict(metric.document_frequency), before)
        with self.assertRaisesRegex(ValueError, "original clean CIDEr"):
            target.verify_forward_cider(metric, records, dict(caps=dict(guided_denoised=caps)), values + 1, cider)

    def test_original_reference_alignment_rejected(self):
        # Validation must reject a slot permutation before scoring or deduplication.
        old = [dict(record(), image=i,
                    mismatch_source=(i + 1) % 1000) for i in range(1000)]
        data = dict(records=copy.deepcopy(old), partial=False, n_images=1000, variants=list(target.VARIANTS),
                    evaluated_true_slots=list(range(5)), n_steps=300, metadata=dict(
                        seed=42, k=16, test_lock_sha256="hash", hidden_matched_steps=2, hidden_steps=300,
                        free_steps=300, repair="reference-free known-mask zero-step plan switch; identical fault X"))
        for i, r in enumerate(data["records"]):
            r["feature_index"] = 14000 + i
        data["records"][0]["refs"][2:4] = list(reversed(data["records"][0]["refs"][2:4]))
        forward = dict(captions={v: [v] * 1000 for v in target.FORWARD})
        native = dict(image_ids=list(range(1000)), n_images=1000, refs=[r["refs"] for r in old])
        with self.assertRaisesRegex(ValueError, "alignment mismatch: refs"):
            target.validate_records(data, old, forward, native, 42, "hash")


if __name__ == "__main__":
    unittest.main()
