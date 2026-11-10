"""CPU only: python -B -m unittest discover -s paper_cvpr -p test_calibrate_host.py -v"""

from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import unittest

import calibrate_host as calibration


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = calibration.run()
        cls.tasks = cls.result["tasks"]
        cls.grid = cls.result["error_grid"]

    def test_complete_balanced_population_and_query_independence(self):
        self.assertEqual(len(self.tasks), 200)
        self.assertEqual(Counter(t["true_route"] for t in self.tasks), {0: 100, 1: 100})
        counts = Counter((t["regions"][t["true_route"]]["object"], t["query"], t["true_route"])
                         for t in self.tasks)
        self.assertEqual(set(counts.values()), {5})
        self.assertEqual(len(counts), 40)
        for task in self.tasks:
            self.assertEqual(calibration.cue_predictor(task["regions"], task["query"]), task["true_route"])
            other_query = next(q for q in calibration.CUES if q != task["query"])
            self.assertEqual(calibration.cue_predictor(task["regions"], other_query), 1 - task["true_route"])
        self.assertEqual(self.tasks, calibration.make_tasks())

    def test_donor_is_bijective_and_both_objects_are_absent(self):
        self.assertEqual({t["shuffled_donor_id"] for t in self.tasks}, set(range(200)))
        for task in self.tasks:
            donor = self.tasks[task["shuffled_donor_id"]]
            self.assertTrue({r["object"] for r in task["regions"]}.isdisjoint(
                r["object"] for r in donor["regions"]))

    def test_noise_has_exact_counts_and_nested_masks(self):
        previous = set()
        for count in calibration.ERROR_COUNTS:
            ids = calibration.error_ids(self.tasks, count)
            self.assertEqual(len(ids), count)
            self.assertLessEqual(previous, ids)
            previous = ids
        for invalid in (-1, 201, 0.5):
            with self.assertRaises(ValueError):
                calibration.error_ids(self.tasks, invalid)

    def test_standard_multiset_metric_and_no_stemming(self):
        self.assertAlmostEqual(calibration.score("CAT, cat!", "cat"), 2 / 3)
        self.assertEqual(calibration.score("running", "run"), 0)
        self.assertEqual(calibration.score("canine", "dog"), 0)

    def test_oracle_enumeration_uses_target_only_and_preserves_ties(self):
        for task in self.tasks:
            captions = [calibration.decode(task["regions"], r) for r in (0, 1)]
            for k, target in enumerate(task["refs"]):
                for baseline in (0, 1):
                    selected = calibration.exact_route(captions, target, baseline)
                    self.assertEqual(calibration.score(captions[selected], target),
                                     max(calibration.score(c, target) for c in captions))
                    self.assertEqual(selected, baseline if k in (2, 4) else task["true_route"])
        # No hidden semantic or query oracle breaks a lexical tie.
        self.assertEqual(calibration.exact_route(["dog", "cat"], "canine", 1), 1)

    def test_heldout_exclusion_is_by_slot_and_not_best_of_targets(self):
        scores = calibration.reference_scores("cat", ["cat", "dog", "bird"], 0)
        self.assertEqual(scores, {"own_reference_best": 0, "own_reference_mean": 0})
        scores = calibration.reference_scores("cat", ["cat", "cat", "dog"], 0)
        self.assertEqual(scores, {"own_reference_best": 1, "own_reference_mean": 0.5})
        with self.assertRaises(ValueError):
            calibration.reference_scores("cat", ["cat"], 0)

    def test_heldout_scores_agree_with_existing_host_rescorer(self):
        import rescore_rouge

        task = self.tasks[0]
        captions = [calibration.decode(task["regions"], r) for r in (0, 1)]
        baseline = 1 - task["true_route"]
        outputs = [captions[calibration.exact_route(captions, target, baseline)]
                   for target in task["refs"]]
        record = {"refs": task["refs"], "baseline": captions[baseline],
                  "variants": {"toy": {"true": outputs}}}
        values, _ = rescore_rouge.image_scores(record, "toy")
        for index, candidates in enumerate((outputs, [captions[baseline]] * 5)):
            expected = sum(calibration.reference_scores(c, task["refs"], k)["own_reference_best"]
                           for k, c in enumerate(candidates)) / 5
            self.assertAlmostEqual(values["heldout"][index], expected)

    def test_known_headroom_and_lexical_tie_loss(self):
        # Independent hand count: 7-token decoder, 6/5/6/6/5-token refs.
        # Correct max is 6/12 with slot 0 removed, otherwise 8/13;
        # wrong-route max is 6/13 in every leave-one-out slot.
        route_gain_per_error = ((6 / 12 - 6 / 13) + 2 * (8 / 13 - 6 / 13)) / 5
        cue_gain_per_error = ((6 / 12 - 6 / 13) + 4 * (8 / 13 - 6 / 13)) / 5
        for row in self.grid:
            q = row["wrong_route_fraction"]
            systems = row["true"]["systems"]
            self.assertAlmostEqual(systems["route_oracle"]["delta_own_reference_best"], q * route_gain_per_error)
            self.assertAlmostEqual(systems["cue_predictor"]["delta_own_reference_best"], q * cue_gain_per_error)
            self.assertAlmostEqual(row["true_oracle_wrong_route_fraction"], 0.4 * q)
            self.assertEqual(row["target_type_heldout_best_gain"]["synonym"], 0)
            self.assertEqual(systems["cue_predictor"]["desired_object_mention"], 1)
            for i, gain in enumerate(row["per_task_true_heldout_best_gain"]):
                self.assertAlmostEqual(gain, route_gain_per_error if i in row["corrupted_task_ids"] else 0)

    def test_unsupported_targets_have_scaffold_overlap_but_no_injection(self):
        for row in self.grid:
            systems = row["shuffled"]["systems"]
            route = systems["route_oracle"]
            self.assertGreater(route["target_fit"], 0)
            self.assertEqual(route["target_fit"], systems["baseline"]["target_fit"])
            self.assertEqual(route["unsupported_object_mention"], 0)
            self.assertEqual(systems["target_copy"]["target_fit"], 1)
            self.assertEqual(systems["target_copy"]["unsupported_object_mention"], 1)
            self.assertGreater(systems["target_copy"]["delta_own_reference_best"], 0)

    def test_copy_matching_is_exact_per_job_not_a_capacity_claim(self):
        self.assertAlmostEqual(calibration.copy_dose(0.2, 0.6), 0.5)
        self.assertEqual(calibration.copy_dose(1, 1), 0)
        with self.assertRaises(ValueError):
            calibration.copy_dose(0.6, 0.2)
        for row in self.grid:
            self.assertLess(row["maximum_per_job_copy_fit_matching_error"], 1e-12)
            for condition in ("true", "shuffled"):
                systems = row[condition]["systems"]
                self.assertAlmostEqual(systems["fit_matched_copy"]["target_fit"], systems["route_oracle"]["target_fit"])
            self.assertEqual(row["shuffled"]["matched_copy_zero_dose_fraction"], 1)
            self.assertEqual(row["shuffled"]["matched_copy_mean_dose"], 0)

    def test_labeling_and_no_model_imports(self):
        self.assertTrue(all(self.result["verification"].values()))
        self.assertFalse(self.result["satisfies_requested_degraded_model_calibration"])
        self.assertIn("NOT VISION-MODEL EMPIRICAL", self.result["label"])
        code = ("import sys; import calibrate_host; "
                "assert not any(m.split('.')[0] in {'torch', 'transformers', 'tensorflow', 'jax'} for m in sys.modules)")
        subprocess.run([sys.executable, "-B", "-c", code], check=True,
                       cwd=Path(calibration.__file__).parent)

    def test_saved_results_match_current_computation_if_present(self):
        path = Path(calibration.__file__).with_name("calibrate_host_results.json")
        if path.exists():
            self.assertEqual(json.loads(path.read_text()), json.loads(json.dumps(self.result)))


if __name__ == "__main__":
    unittest.main()
