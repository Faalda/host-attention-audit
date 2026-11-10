"""Synthetic CPU contracts; no torch, inference, or original artifact writes."""

import ast
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pycocoevalcap.cider.cider import Cider

from analysis.rescore_clean_cider import (
    FixedIDFCider, FrozenCiderScorer, ROOT, SEEDS, aggregate, norm, paired,
    protocol_scores, publish, validate_dump,
)


class FixedCiderTests(unittest.TestCase):
    def setUp(self):
        self.refs = [[s] * 5 for s in ("red cat on mat", "blue dog in park", "green bird near tree")]

    def test_norm_matches_legacy_without_importing_model_dependencies(self):
        tree = ast.parse((ROOT / "inversion/run_rev_eval_metrics.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "norm")
        namespace = {"re": __import__("re")}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "legacy_norm", "exec"), namespace)
        for text in ("A Cat's 42...DOG!", "  \t\n ", "caf\u00e9 x_y <UNK>", "A-B / C"):
            self.assertEqual(norm(text), namespace["norm"](text))

    def test_standard_forward_equivalence_and_batch_independence(self):
        predictions = ["red cat cat on mat", "blue dog", ""]
        refs = copy.deepcopy(self.refs)
        refs[0][1] = "cat resting on a rug"
        standard = Cider().compute_score({i: rs for i, rs in enumerate(refs)},
                                         {i: [p or "empty"] for i, p in enumerate(predictions)})[1]
        for batch in (1, 2, 100):
            # New scorer each time, so cache cannot hide batch-dependent IDF.
            actual = FixedIDFCider(refs).score(predictions, refs, batch)
            np.testing.assert_array_equal(actual, standard)

    def test_singleton_batch_duplicate_jobs_and_unseen_hypotheses_do_not_change_df(self):
        metric = FixedIDFCider(self.refs)
        before = dict(metric.document_frequency)
        self.assertEqual(before[("red",)], 1)  # five copies are one document
        prediction, references = ["red cat cat on mat"], [[self.refs[0][0]]]
        first = metric.score(prediction, references, 1)[0]
        repeated = metric.score(prediction * 17, references * 17, 3)
        np.testing.assert_array_equal(repeated, [first] * 17)
        metric.score(["unseen nonsense"], references)
        self.assertEqual(dict(metric.document_frequency), before)
        with self.assertRaises(TypeError):
            metric.document_frequency[("red",)] = 2
        scorer = FrozenCiderScorer(metric.document_frequency, 3)
        scorer += (prediction[0], references[0])
        scorer.compute_cider()
        self.assertEqual(scorer.ref_len, np.log(3.0))
        with self.assertRaises(RuntimeError):
            scorer.compute_doc_freq()

    def test_identical_short_captions_and_zero_idf_orders(self):
        for length in range(1, 5):
            caption = " ".join("red cat on mat".split()[:length])
            refs = [[caption] * 5, ["blue dog in park"] * 5]
            score = FixedIDFCider(refs).score([caption], [[caption]])[0]
            self.assertAlmostEqual(score, 2.5 * length)
        metric = FixedIDFCider([["cat"] * 5, ["cat"] * 5])
        self.assertEqual(metric.score(["cat"], [["cat"]])[0], 0)

    def test_target_slot_exclusion_duplicates_and_shuffle_references(self):
        record = {"refs": ["cat", "cat", "dog", "bird", "horse"], "baseline": "dog",
                  "mismatch_target": "boat", "variants": {"free": {
                      "true": ["cat", "cat", None, None, None], "mismatch": "boat"}}}

        class Capture:
            def __init__(self):
                self.calls = []

            def score(self, predictions, references, batch_size):
                self.calls.append((predictions, references))
                return np.arange(len(predictions), dtype=float)

        metric = Capture()
        a, b = protocol_scores(metric, [record], "free", "heldout", (0, 1))
        self.assertEqual(a.tolist(), [0.5])
        self.assertEqual(b.tolist(), [0.5])
        for _, refs in metric.calls:
            self.assertEqual(refs, [["cat", "dog", "bird", "horse"]] * 2)
        for protocol, expected in (("single_ref", [["cat"], ["cat"]]),
                                   ("mismatch_target", [["boat"]]),
                                   ("mismatch_grounding", [record["refs"]]),
                                   ("each_inversion_all_refs_avg", [record["refs"]] * 2)):
            metric = Capture()
            protocol_scores(metric, [record], "free", protocol, (0, 1))
            self.assertTrue(all(rs == expected for _, rs in metric.calls))

    def test_bootstrap_jobs_then_models_then_images(self):
        rows = {s: (np.eye(3)[i], np.zeros(3)) for i, s in enumerate(SEEDS)}
        result = aggregate(rows)
        np.testing.assert_allclose(result["ci95"], [1 / 3, 1 / 3])
        self.assertLess(result["per_seed"]["42"]["ci95"][0], 1 / 3)
        delta = np.array([1., -2., 3.])
        indices = np.random.default_rng(0).integers(0, 3, (10000, 3))
        expected = np.percentile(delta[indices].mean(1), [2.5, 97.5])
        np.testing.assert_array_equal(paired(delta, np.zeros(3))["ci95"], expected)
        with self.assertRaises(ValueError):
            aggregate({42: rows[42]})

    def test_missing_slots_and_misaligned_inputs_rejected(self):
        record = {"image": 0, "feature_index": 14000, "refs": self.refs[0],
                  "mismatch_source": 1, "mismatch_target": "dog", "baseline": "cat",
                  "variants": {"free": {"true": ["cat"] * 5, "mismatch": "dog"}}}
        data = {"partial": False, "n_images": 1, "evaluated_true_slots": list(range(5)),
                "variants": ["free"], "records": [record]}
        validate_dump(data, [record], range(5), ("free",))
        data = copy.deepcopy(data)
        data["records"][0]["variants"]["free"]["true"][3] = None
        with self.assertRaises(ValueError):
            validate_dump(data, [record], range(5), ("free",))
        with self.assertRaises(ValueError):
            FixedIDFCider(self.refs).score(["cat"], [])
        with self.assertRaises(ValueError):
            paired([float("nan")], [0])

    def test_publication_never_overwrites_and_check_is_read_only(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sidecar.json"
            publish(path, "original\n")
            with self.assertRaises(FileExistsError):
                publish(path, "replacement\n")
            publish(path, "original\n", check=True)
            with self.assertRaises(ValueError):
                publish(path, "replacement\n", check=True)
            self.assertEqual(path.read_text(), "original\n")
            self.assertEqual(list(Path(folder).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
