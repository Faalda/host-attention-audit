"""CPU bookkeeping tests; optional real-model smoke: score_semantics.py --smoke-only."""

import unittest
from unittest.mock import patch

import numpy as np

import score_semantics as sem


def record(refs=None, outputs=None, baseline="a", image=0):
    refs = refs or ["a", "b", "c", "d", "e"]
    return {"image": image, "refs": refs, "baseline": baseline, "mismatch_target": "alien",
            "variants": {v: {"true": outputs or refs, "mismatch": "alien"} for v in sem.VARIANTS}}


def exact(a, b):
    return float(a == b)


class SemanticTests(unittest.TestCase):
    def test_legacy_tokenization_audit_and_prefix_regression(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(sem.SNAPSHOT), local_files_only=True,
                                                  use_fast=False, add_prefix_space=True)
        if not hasattr(tokenizer, "build_inputs_with_special_tokens"):
            tokenizer.build_inputs_with_special_tokens = lambda ids: [0, *ids, 2]
        texts = {"", "A cat sits.", "a cat sits.", "  Don't change CASE!  ",
                 "tabs\tand  spaces\ninside", "caf\u00e9", "a " * 600}
        result = sem.audit_tokenization(tokenizer, texts)
        self.assertTrue(result["passed"])
        self.assertEqual(result["empty_strings"], 1)
        self.assertEqual(result["truncated_strings"], 1)
        self.assertEqual(result["unique_strings_checked"], len(texts))

    def test_fit_reference_is_excluded_for_both_sides(self):
        values = sem.small_image(record(), "free", exact)
        self.assertEqual(values["single_ref"], (1.0, 0.2))
        self.assertEqual(values["heldout"], (0.0, 0.8))
        self.assertEqual(values["each_inversion_all_refs_avg"], (1.0, 1.0))

    def test_exclude_slot_not_duplicate_string(self):
        values = sem.small_image(record(refs=["a", "a", "c", "d", "e"]), "free", exact)
        self.assertEqual(values["heldout"], (0.4, 1.0))

    def test_max_per_job_then_average_not_global_max(self):
        values = sem.small_image(record(outputs=["b", "c", "x", "x", "x"]), "free", exact)
        self.assertEqual(values["heldout"], (0.4, 0.8))

    def test_missing_slot_is_rejected_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            sem.small_image(record(outputs=["a", None, "c", "d", "e"]), "free", exact)

    def test_validation_aligns_images_and_rejects_incomplete_variant(self):
        records = [record(image=2), record(image=0)]
        source = {"records": records, "n_images": 2, "variants": list(sem.VARIANTS)}
        captions = {name: ["a"] * 5000 for name in sem.SYSTEMS}
        refs = {"keys": list(range(5000)), "caps": [records[0]["refs"]] * 5000}
        self.assertEqual([r["image"] for r in sem.validate(source, captions, refs)], [0, 2])
        records[0]["variants"]["hidden"]["true"] = ["a", None, "c", "d", "e"]
        with self.assertRaises(ValueError):
            sem.validate(source, captions, refs)

    def test_paired_image_aggregation_and_fullcoco_direction(self):
        records = [record(), record(outputs=["x"] * 5, image=1)]
        captions = {"guided_denoised": ["a", "x"], "guided_uniform": ["x", "x"]}
        refs = {"keys": ["i0", "i1"], "caps": [records[0]["refs"]] * 2}
        original = sem.paired
        with patch.object(sem, "paired", side_effect=lambda a, b: original(a, b, resamples=101)):
            result = sem.aggregate(records, captions, refs, 2, exact)
        self.assertEqual(result["small"]["variants"]["free"]["single_ref"]["inv"], 0.5)
        full = result["fullcoco_uniform"]
        self.assertEqual(full["systems"]["guided_denoised"]["mean"], 0.5)
        self.assertEqual(full["paired"]["denoised_minus_uniform"]["delta"], 0.5)
        self.assertEqual(full["image_ids"], ["i0", "i1"])

    def test_pair_collection_covers_all_metrics_and_deduplicates(self):
        r = record()
        caps = {name: ["a"] for name in sem.SYSTEMS}
        refs = {"caps": [r["refs"]]}
        pairs = sem.collect_pairs([r], caps, refs, 1)
        self.assertEqual(len(pairs), len(set(pairs)))
        cache = {pair: exact(*pair) for pair in pairs}
        for variant in sem.VARIANTS:
            self.assertEqual(sem.small_image(r, variant, exact),
                             sem.small_image(r, variant, lambda a, b: cache[a, b]))
        self.assertIn(("alien", "alien"), cache)
        self.assertIn(("a", "alien"), cache)

    def test_bootstrap_is_paired_seed_zero_image_unit(self):
        a, b = np.array([0.2, 0.8, 0.4]), np.array([0.4, 0.1, 0.9])
        idx = np.random.default_rng(0).integers(0, 3, (10000, 3))
        expected = np.percentile((a - b)[idx].mean(1), [2.5, 97.5])
        np.testing.assert_array_equal(sem.paired(a, b)["ci95"], expected)
        self.assertEqual(sem.paired(a, a)["ci95"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
