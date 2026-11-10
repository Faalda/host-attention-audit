"""CPU-only unit tests and independent saved-result verification."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["ATTN_ENCODER"] = "dinov2"

import unittest
import numpy as np
import torch
import run_visual_headroom_control as control


class ControlTests(unittest.TestCase):
    def test_nested_row_independent_masks(self):
        indices = [13000, 13001, 14000]
        masks = [control.anchor_masks(indices, k) for k in control.KS]
        for k, mask in zip(control.KS, masks):
            self.assertEqual(mask.sum(1).tolist(), [k] * 3)
            self.assertTrue(torch.equal(mask[1], control.anchor_masks([13001], k)[0]))
        self.assertTrue((~masks[0] | masks[1]).all())
        self.assertTrue((~masks[1] | masks[2]).all())
        self.assertFalse(torch.equal(masks[0][0], masks[0][1]))

    def test_plans_above_clamp_and_train_mean_provenance(self):
        for k in control.KS:
            mask = control.anchor_masks([13000, 13001], k)
            plans = control.routing_plans(mask)
            for p in plans.values():
                self.assertEqual(p.shape, (2, 40, 256))
                self.assertTrue((p > 1e-6).all())
                torch.testing.assert_close(p.sum(-1), torch.ones(2, 40))
                self.assertTrue(torch.equal(p[:, 0], p[:, 39]))
            self.assertTrue((plans["fault"][:, 0][mask] < plans["repair"][:, 0][mask]).all())
            theta = plans["fault"].log().requires_grad_()
            (theta.softmax(-1).clamp(min=1e-6).log()[:, 0] * mask).sum().backward()
            self.assertTrue(torch.isfinite(theta.grad).all())
            self.assertGreater(float(theta.grad.abs().max()), 0)
            clean, mu = torch.randn(2, 256, 256), torch.randn(256)
            x = control.fault_features(clean, mu, mask)
            self.assertTrue(torch.equal(x[mask], clean[mask]))
            self.assertTrue(torch.equal(x[~mask], mu.expand_as(x)[~mask]))

    def test_gate_failure_and_fixed_ties(self):
        def values(gain):
            return [dict(candidates={str(k): dict(scores=dict(repair=[.3 + gain]*200, fault=[.3]*200),
                repair_cyclic_reference_scores=[.1]*200, finite=True, empty_counts=dict(fault=0, repair=0))
                for k in control.KS}) for _ in control.SEEDS]
        self.assertIsNone(control.choose_k(values(.019))["chosen_k"])
        self.assertEqual(control.choose_k(values(.021))["chosen_k"], 4)
        invalid = values(.1)
        for row in invalid:
            for candidate in row["candidates"].values():
                candidate["repair_cyclic_reference_scores"] = [.8]*200
        self.assertIsNone(control.choose_k(invalid)["chosen_k"])

    def test_matching_global_not_per_image(self):
        jobs = [dict(image=0, target="a dog"), dict(image=1, target="a cat")]
        base = ["a", "a"]
        free = ["a dog", "a"]
        hidden = {s: (["a dog", "a"] if s in (2, 3) else base) for s in control.HIDDEN_STEPS}
        result = control.matching_choice(jobs, base, free, hidden)
        self.assertEqual(result["step"], 2)
        self.assertEqual(result["residual_mae"], 0)
        self.assertFalse(result["test_selection"])

    def test_oov_kept_all_five_duplicate_slots(self):
        from clean_pipeline.run_summarization import Tokenizer
        tok = Tokenizer()
        tok.vocab = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "a"]
        tok.w2i = {w: i for i, w in enumerate(tok.vocab)}
        job = dict(target="a zzfakewordzz", fake="zzfakewordzz")
        audit = control.target_audit(tok, job)
        self.assertTrue(audit["fake_oov"])
        self.assertIn(tok.UNK, audit["target_tokens"])
        records = [dict(image=i, refs=[job["target"]]*5, mismatch_target="a", mismatch_source=1-i) for i in range(2)]
        jobs = control.protocol_jobs(records)
        self.assertEqual(len(jobs), 12)
        self.assertEqual([j["k"] for j in jobs[:5]], list(range(5)))

    def test_optimizer_native_parity_and_frozen_weights(self):
        from clean_pipeline.run_cap_improved import SoftGuidedCaptionDecoder
        from inversion.run_rev_inversion_protocol import invert
        torch.set_num_threads(2)
        torch.manual_seed(11)
        dec = SoftGuidedCaptionDecoder(12, 256, 4, 1).eval().requires_grad_(False)
        x = torch.randn(2, 256, 256)
        p = control.routing_plans(control.anchor_masks([13000, 13001], 4))["fault"]
        target = torch.tensor([[1, 4, 5, 2] + [0]*36, [1, 6, 2] + [0]*37])
        before = control.model_digest(dict(dec=dec))
        for variant in ("free", "hidden"):
            tokens, stats, trajectory = control.optimize(dec, x, p, target, variant, steps=2, checkpoints=(0, 1, 2))
            _, _, expected = invert(dec, x, p, target, variant, 2, .05)
            self.assertEqual(tokens, expected.tolist())
            self.assertEqual(tokens, trajectory[2])
            self.assertGreater(stats[0]["max_abs_gradient_batch"], 0)
        self.assertEqual(before, control.model_digest(dict(dec=dec)))
        self.assertTrue(all(p.grad is None for p in dec.parameters()))

    @unittest.skipUnless((control.OUT / "results.json").exists(), "Live results not yet available")
    def test_independent_saved_results(self):
        from rouge_score import rouge_scorer
        import re
        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)
        def score(a, b):
            def norm(s):
                return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())
            return scorer.score(norm(b), norm(a))["rouge1"].fmeasure
        results = control.read_json(control.OUT / "results.json")
        lock = control.read_json(control.OUT / "validation_lock.json")
        vals = [control.read_json(control.OUT / f"seed{s}/validation/results.json") for s in control.SEEDS]
        eligible = []
        gains = {}
        for k in control.KS:
            delta, provenance = [], []
            for val in vals:
                row, refs = val["candidates"][str(k)], val["refs"]
                own = np.array([max(score(c, r) for r in rr) for c, rr in zip(row["captions"]["repair"], refs)])
                fault = np.array([max(score(c, r) for r in rr) for c, rr in zip(row["captions"]["fault"], refs)])
                foreign = np.array([max(score(c, r) for r in refs[(i+1)%200]) for i, c in enumerate(row["captions"]["repair"])])
                np.testing.assert_allclose(own, row["scores"]["repair"], atol=1e-14)
                np.testing.assert_allclose(fault, row["scores"]["fault"], atol=1e-14)
                delta.append(own-fault)
                provenance.append(own-foreign)
            d, info = np.mean(delta, axis=0), np.mean(provenance, axis=0)
            rng = np.random.default_rng(0)
            idx = rng.integers(0, 200, (10000, 200))
            lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
            np.testing.assert_allclose([lo, hi], lock["candidates"][str(k)]["repair_minus_fault"]["ci95"], atol=1e-14)
            gains[k] = d.mean()
            finite = all(v["candidates"][str(k)]["finite"] for v in vals)
            empty = sum(sum(v["candidates"][str(k)]["empty_counts"].values()) for v in vals)
            if d.mean() >= .02 and lo > 0 and np.percentile(info[idx].mean(1), 2.5) > 0 and finite and empty == 0:
                eligible.append(k)
        chosen = max(eligible, key=lambda k: gains[k]) if eligible else None
        self.assertEqual(chosen, lock["chosen_k"])
        if chosen is None:
            self.assertFalse(results["test_run"])
            self.assertFalse((control.OUT / "test_lock.json").exists())
            self.assertFalse(any((control.OUT / f"seed{s}/test").exists() for s in control.SEEDS))
        else:
            test_lock = control.read_json(control.OUT / "test_lock.json")
            pooled_heldout = {v: [] for v in ("free", "hidden", "hidden_matched", "repair")}
            for seed in control.SEEDS:
                calibration = control.read_json(control.OUT / f"seed{seed}/calibration/results.json")
                self.assertEqual(len(calibration["jobs"]), 750)
                self.assertEqual(sorted(set(j["feature_index"] for j in calibration["jobs"])), list(range(13000, 13150)))
                means = {int(s): np.mean([score(c, j["target"]) for c, j in zip(cc, calibration["jobs"])])
                         for s, cc in calibration["hidden_trajectory"].items()}
                free_mean = np.mean([score(c, j["target"]) for c, j in zip(calibration["captions"]["free"], calibration["jobs"])])
                best = min(control.HIDDEN_STEPS, key=lambda s: abs(means[s]-free_mean))
                self.assertEqual(best, test_lock["hidden_steps"][str(seed)])
                dump = control.read_json(control.OUT / f"seed{seed}/test/inversion_protocol_n1000_s0.json")
                self.assertEqual([r["image"] for r in dump["records"]], list(range(1000)))
                for r in dump["records"]:
                    self.assertEqual(r["feature_index"], 14000+r["image"])
                    for v in dump["variants"]:
                        self.assertEqual(len(r["variants"][v]["true"]), 5)
                        self.assertTrue(all(isinstance(c, str) for c in r["variants"][v]["true"]))
                    self.assertEqual(r["variants"]["repair"]["true"], [r["forward"]["repair"]]*5)
                    self.assertEqual(r["variants"]["repair"]["mismatch"], r["forward"]["repair"])
                for v in pooled_heldout:
                    delta = []
                    for r in dump["records"]:
                        refs = r["refs"]
                        delta.append(np.mean([max(score(r["variants"][v]["true"][s], ref) for t, ref in enumerate(refs) if t != s)
                            - max(score(r["baseline"], ref) for t, ref in enumerate(refs) if t != s) for s in range(5)]))
                    pooled_heldout[v].append(delta)
            for v, arrays in pooled_heldout.items():
                delta = np.mean(arrays, axis=0)
                rng = np.random.default_rng(0)
                idx = rng.integers(0, 1000, (10000, 1000))
                ci = np.percentile(delta[idx].mean(1), [2.5, 97.5])
                np.testing.assert_allclose(ci, results["pooled"][v]["heldout"]["ci95"], atol=1e-14)
                self.assertAlmostEqual(float(delta.mean()), results["pooled"][v]["heldout"]["delta"], places=12)
        for seed in control.SEEDS:
            stages = ("validation", "calibration", "test") if chosen is not None else ("validation",)
            for stage in stages:
                audit = control.read_json(control.OUT / f"seed{seed}/{stage}/complete.json")
                self.assertEqual(audit["state_sha256_before"], audit["state_sha256_after"])
                self.assertTrue(audit["gradients_none"])
        for path, digest in control.read_json(control.OUT / "input_hashes.json").items():
            self.assertEqual(control.sha256(path), digest, path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
