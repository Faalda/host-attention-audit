"""CPU synthetic contracts for the clean three-model summary."""

import copy
import unittest

import numpy as np

from analysis.summarize_clean_results import (
    aggregate, compare_saved, finite_json, mask_audit, verify_provenance_refresh,
)
import torch
from analysis.rescore_rouge import image_scores, paired


class SummaryTests(unittest.TestCase):
    def test_provenance_refresh_cannot_change_results_or_experiment_inputs(self):
        old = {"comparisons": {"gain": 0.1}, "source_sha256": {
            "paper_cvpr/main.tex": "old", "results/captions.json": "unchanged"}}
        new = copy.deepcopy(old)
        new["source_sha256"]["paper_cvpr/main.tex"] = "new"
        verify_provenance_refresh(old, new)
        new["comparisons"]["gain"] = 0.2
        with self.assertRaisesRegex(ValueError, "Scientific results changed"):
            verify_provenance_refresh(old, new)
        new = copy.deepcopy(old)
        new["source_sha256"]["results/captions.json"] = "changed"
        with self.assertRaisesRegex(ValueError, "Experiment source hashes changed"):
            verify_provenance_refresh(old, new)

    def test_bootstrap_averages_seeds_before_images(self):
        rows = [paired([1, 0, 0], [0, 0, 0]), paired([0, 1, 0], [0, 0, 0]),
                paired([0, 0, 1], [0, 0, 0])]
        result = aggregate(rows)
        np.testing.assert_allclose(result["ci95"], [1 / 3, 1 / 3])
        self.assertLess(rows[0]["ci95"][0], result["ci95"][0])
        self.assertEqual(set(result["per_seed"]), {"42", "7", "13"})

    def test_strict_slots_preserve_duplicates_and_never_claim_best5(self):
        record = {"refs": ["cat", "cat", "dog", "bird", "horse"], "baseline": "dog",
                  "mismatch_target": "boat", "variants": {"free": {
                      "true": ["cat", "cat", None, None, None], "mismatch": "boat"}}}
        values, slots = image_scores(record, "free")
        self.assertEqual(slots, [0, 1])
        self.assertNotIn("best5_clipped", values)
        self.assertEqual(values["heldout"], (1, 1))

    def test_repair_allrefs_is_not_heldout(self):
        record = {"refs": ["cat", "dog", "bird", "horse", "boat"], "baseline": "none",
                  "mismatch_target": "dog", "variants": {"repair": {
                      "true": ["cat", "cat", None, None, None], "mismatch": "cat"}}}
        values, _ = image_scores(record, "repair")
        self.assertEqual(values["heldout"][0], 0.5)
        self.assertEqual(values["mismatch_grounding"][0], 1)

    def test_nonfinite_and_corrupted_saved_scores_rejected(self):
        with self.assertRaises(ValueError):
            finite_json({"nested": [float("nan")]})
        row = paired([1, 2], [0, 1])
        saved = copy.deepcopy(row)
        saved["per_image"]["inv"][0] += 1
        with self.assertRaises(ValueError):
            compare_saved(row, saved)
        with self.assertRaises(ValueError):
            aggregate([row, row])

    def test_mask_counts_loss_rows_not_allocated_rows(self):
        rng = torch.Generator().manual_seed(20261030)
        torch.randperm(40, generator=rng)
        replacement = torch.randint(256, (1, 40), generator=rng)
        records = [{"corruption_rows": list(range(40)), "replacement_patch_indices": replacement[0].tolist()}]
        # Every row selected, but only two rows contribute to the target loss.
        encodings = [{"kind": kind, "k": k, "image": 0, "target_tokens": [1, 4, 2] + [0] * 37}
                     for kind, k in (("true", 0), ("true", 1), ("mismatch", -1))]
        plans = torch.nn.functional.one_hot(replacement, 256).float()
        result = mask_audit(records, encodings, plans, "1")
        self.assertEqual(result["true"]["selected_loss_bearing_rows"], 4)
        self.assertEqual(result["true"]["changed_argmax_loss_bearing_rows"], 0)
        self.assertEqual(result["nominal_selected_rows_per_image"], 40)
        records[0]["corruption_rows"].pop()
        with self.assertRaises(ValueError):
            mask_audit(records, encodings, plans, "1")


if __name__ == "__main__":
    unittest.main()
