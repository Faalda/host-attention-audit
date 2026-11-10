"""Run: python -B -m unittest discover -s paper_cvpr -p test_rescore_rouge.py -v"""

import unittest
from unittest.mock import patch

import numpy as np

import rescore_rouge as rescore


def record(refs, captions, baseline="a", image=0):
    return {"image": image, "refs": refs, "baseline": baseline,
            "mismatch_target": "alien", "variants": {"free": {
                "true": captions, "mismatch": "alien"}}}


class AggregationTests(unittest.TestCase):
    def test_best5_is_diagonal_not_all_25_pairs(self):
        r = record(["a", "b", "c", "d", "e"], ["b", "c", "d", "e", "a"])
        values, slots = rescore.image_scores(r, "free")
        self.assertEqual(slots, list(range(5)))
        self.assertEqual(values["best5_unclipped"], (0.0, 1.0))
        self.assertEqual(values["best5_clipped"], (1.0, 1.0))
        self.assertEqual(values["each_inversion_all_refs_avg"], (1.0, 1.0))
        self.assertEqual(values["single_ref"], (0.0, 0.2))
        self.assertEqual(values["heldout"], (1.0, 0.8))

    def test_duplicate_reference_excluded_by_slot_not_value(self):
        values, _ = rescore.image_scores(record(["cat", "cat", "dog"],
                                                ["cat", None, None], "cat"), "free")
        self.assertEqual(values["heldout"], (1.0, 1.0))
        self.assertEqual(values["human_heldout"], (1.0, 1.0))
        self.assertNotIn("best5_clipped", values)

    def test_all_refs_uses_max_then_average(self):
        values, _ = rescore.image_scores(record(["a", "b", "c", "d", "e"],
                                                ["a", "b", "x", "x", "x"], "z"), "free")
        self.assertAlmostEqual(values["each_inversion_all_refs_avg"][0], 0.4)
        self.assertEqual(values["heldout"], (0.0, 0.0))

    def test_missing_jobs_not_zero_filled_and_images_equally_weighted(self):
        records = [record(["a", "b"], ["a", None], "z", 3),
                   record(["a", "b"], ["z", "z"], "z", 1)]
        original = rescore.paired
        with patch.object(rescore, "paired", side_effect=lambda a, b: original(a, b, resamples=20)):
            result = rescore.protocol({"n_images": 2, "variants": ["free"], "records": records})
        self.assertEqual(result["image_ids"], [1, 3])
        self.assertEqual(result["variants"]["free"]["protocol"]["single_ref"]["inv"], 0.5)

    def test_human_uses_only_evaluated_slots(self):
        values, slots = rescore.image_scores(record(["a", "b", "b"], ["a", None, None]), "free")
        self.assertEqual(slots, [0])
        self.assertEqual(values["human_heldout"], (0.0, 0.0))

    def test_shuffled_fit_and_grounding_are_distinct(self):
        values, _ = rescore.image_scores(record(["a", "b"], ["a", "b"]), "free")
        self.assertEqual(values["mismatch_target"], (1.0, 0.0))
        self.assertEqual(values["mismatch_grounding"], (0.0, 1.0))

    def test_standard_metric_multiset_and_normalization(self):
        self.assertAlmostEqual(rescore.score("CAT, cat!", "cat"), 2 / 3)
        self.assertEqual(rescore.score("", "cat"), 0.0)
        self.assertEqual(rescore.score("running", "run"), 0.0)

    def test_bootstrap_matches_unchunked_seed_zero(self):
        a, b = np.array([0.2, 0.8, 0.4]), np.array([0.4, 0.1, 0.9])
        indices = np.random.default_rng(0).integers(0, 3, (103, 3))
        expected = np.percentile((a - b)[indices].mean(1), [2.5, 97.5])
        actual = rescore.paired(a, b, resamples=103, batch=7)
        np.testing.assert_array_equal(actual["ci95"], expected)
        self.assertEqual(actual, rescore.paired(a, b, resamples=103, batch=100))

    def test_paired_not_independent_bootstrap(self):
        result = rescore.paired([0.25, 0.75], [0.25, 0.75], resamples=25)
        self.assertEqual(result["ci95"], [0.0, 0.0])
        self.assertEqual(result["frac_images_pos"], 0.0)

    def test_forward_dummy_mismatch_fields_are_not_reported(self):
        r = record(["a", "b"], ["a", "a"])
        original = rescore.paired
        with patch.object(rescore, "paired", side_effect=lambda a, b: original(a, b, resamples=20)):
            result = rescore.protocol({"n_images": 1, "variants": ["free"], "records": [r]},
                                      include_mismatch=False)
        self.assertEqual(set(result["variants"]["free"]["protocol"]), {"single_ref", "heldout"})

    def test_fullcoco_single_caption_best_reference_and_paired_direction(self):
        refs = {"keys": list(range(5000)), "caps": [["a", "b", "c", "d", "e"]] * 5000}
        captions = {"free": ["z"] * 5000, "guided_denoised": ["a"] * 5000,
                    "guided_uniform": ["a b"] * 5000}
        original = rescore.paired
        with patch.object(rescore, "paired", side_effect=lambda a, b: original(a, b, resamples=20)):
            result = rescore.full_coco(captions, refs)
        self.assertEqual(result["systems"]["guided_denoised"]["mean"], 1.0)
        self.assertAlmostEqual(result["paired"]["denoised_minus_uniform"]["delta"], 1 / 3)
        captions["free"] = []
        with self.assertRaises(ValueError):
            rescore.full_coco(captions, refs)

    def test_validation(self):
        with self.assertRaises(ValueError):
            rescore.paired([], [])
        with self.assertRaises(ValueError):
            rescore.paired([float("nan")], [0])
        r = record(["a", "b"], ["a", "b"])
        with self.assertRaises(ValueError):
            rescore.protocol({"n_images": 2, "variants": ["free"], "records": [r, r]})
        with self.assertRaises(ValueError):
            rescore.protocol({"n_images": 1, "variants": ["free"], "records": [r]}, require_five=True)


if __name__ == "__main__":
    unittest.main()
