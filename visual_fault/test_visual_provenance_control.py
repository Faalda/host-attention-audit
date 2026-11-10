"""CPU tests. No writes to the original experiment; no CUDA allocation."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ATTN_ENCODER"] = "dinov2"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import inspect
import unittest
import numpy as np
import torch

import run_visual_provenance_control as control


class ProvenanceTests(unittest.TestCase):
    def test_derangement_fixed_and_reference_blind(self):
        a, attempts = control.donor_permutation()
        b, again = control.donor_permutation()
        self.assertEqual(a, b)
        self.assertEqual(attempts, again)
        self.assertEqual(sorted(a), list(range(1000)))
        self.assertTrue(all(i != p for i, p in enumerate(a)))
        self.assertEqual(list(inspect.signature(control.donor_permutation).parameters), ["n"])
        self.assertEqual(list(inspect.signature(control.donor_memory).parameters), ["projected_donors", "recipient_mask", "mu"])
        with self.assertRaises(ValueError):
            control.donor_permutation(1)

    def test_recipient_positions_and_masks_not_permuted_memories(self):
        from run_visual_headroom_control import anchor_masks
        clean = torch.arange(4 * 256 * 256).reshape(4, 256, 256).float()
        masks = anchor_masks(list(range(14000, 14004)), 16)
        donors = [1, 2, 3, 0]
        mu = torch.full((256,), -17.)
        memory = control.donor_memory(clean[donors], masks, mu)
        self.assertTrue(torch.equal(memory[masks], clean[donors][masks]))
        self.assertTrue(torch.equal(memory[~masks], mu.expand_as(memory)[~masks]))
        wrong = control.donor_memory(clean, masks, mu)[donors]
        self.assertFalse(torch.equal(memory, wrong))
        broken = clean.clone()
        broken[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            control.donor_memory(broken, masks, mu)

    def test_prefix_bos_no_eos_pad_and_truncation(self):
        from clean_pipeline.run_summarization import Tokenizer
        tok = Tokenizer()
        tok.vocab = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "a", "dog"]
        tok.w2i = {w: i for i, w in enumerate(tok.vocab)}
        prefix = control.caption_prefix(tok, ["a dog", "", "a " * 60, "zzunknownzz"])
        self.assertEqual(tuple(prefix.shape), (4, 39))
        self.assertEqual(prefix[0, :4].tolist(), [1, 4, 5, 0])
        self.assertEqual((prefix != 0).sum(1).tolist(), [3, 1, 39, 2])
        self.assertFalse((prefix == 2).any())
        self.assertEqual(prefix[3, 1].item(), 3)

    def test_capture_native_equivalence_and_exception_restoration(self):
        from clean_pipeline.run_cap_improved import SoftGuidedCaptionDecoder, SoftGuidedDecoderLayer
        from inversion.run_rev_inversion_protocol import dec_forward
        from run_visual_headroom_control import anchor_masks
        torch.set_num_threads(2)
        torch.manual_seed(4)
        dec = SoftGuidedCaptionDecoder(9, 256, 4, 3).eval().requires_grad_(False)
        class_forward = SoftGuidedDecoderLayer.forward
        before = control.model_digest(dict(dec=dec))
        masks = anchor_masks([14000, 14001], 16)
        plans = control.routing_plans(masks)
        memory = torch.randn(2, 256, 256)
        prefix = torch.tensor([[1, 4, 5, 0], [1, 6, 0, 0]])
        result = control.anchor_attention(dec, memory, plans, masks, prefix)
        for layers in result.values():
            self.assertEqual(len(layers), 3)
            for rows in layers:
                self.assertEqual([len(r) for r in rows], [3, 2])
                self.assertTrue(all(0 <= v <= 1 for r in rows for v in r))
        with self.assertRaisesRegex(RuntimeError, "test exception"):
            with control.capture_attention(dec):
                raise RuntimeError("test exception")
        self.assertIs(class_forward, SoftGuidedDecoderLayer.forward)
        self.assertTrue(all("forward" not in layer.__dict__ for layer in dec.layers))
        self.assertEqual(before, control.model_digest(dict(dec=dec)))
        with torch.no_grad():
            null = torch.randn(1, 1, 256).expand(2, 256, 256).contiguous()
            fault = dec_forward(dec, null, prefix, plans["fault"])
            repair = dec_forward(dec, null, prefix, plans["repair"])
        self.assertTrue(torch.isfinite(fault).all())
        self.assertLess((fault-repair).abs().max().item(), 1e-4)

    def test_interaction_pairing_and_reference_swap(self):
        refs = [["a dog"]*5, ["a cat"]*5]
        caps = dict(own_fault=["a", "a"], own_repair=["a dog", "a cat"],
                    donor_fault=["a", "a"], donor_repair=["a cat", "a dog"],
                    null_fault=["a", "a"], null_repair=["a", "a"])
        scores = control.score_outputs(caps, refs, [1, 0])
        self.assertEqual(scores["own_refs"]["own_repair"], [1., 1.])
        self.assertEqual(scores["donor_refs"]["donor_repair"], [1., 1.])
        self.assertEqual(scores["own_refs"]["donor_repair"], [.5, .5])
        rows = [dict(seed=s, captions=caps, scores=scores,
                     attention={c: [dict(image_mean=[.1, .2])]*3 for c in ("fault", "repair")},
                     prior_anchor_mass=dict(fault=.000032, repair=.99952),
                     null=dict(identical_captions_across_masks_and_plans=True,
                               shared_prefix_logits_within_1e_minus4_tolerance=True)) for s in control.SEEDS]
        result = control.aggregate(rows)
        self.assertAlmostEqual(result["comparisons"]["interaction"]["delta"], .5)
        self.assertTrue(result["confirms_visual_basis_in_this_control"])
        for row in rows:
            row["scores"]["own_refs"]["donor_repair"] = [1., 1.]
        self.assertFalse(control.aggregate(rows)["interaction_and_repair_difference_positive"])

    @unittest.skipUnless((control.OUT / "results.json").exists(), "Live provenance results unavailable")
    def test_independent_completed_diagnostic(self):
        from rouge_score import rouge_scorer
        from clean_pipeline.run_captioning import PatchProjector
        import re
        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)
        def score(cap, ref):
            norm = lambda s: " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())
            return scorer.score(norm(ref), norm(cap))["rouge1"].fmeasure
        plan = control.read_json(control.OUT / "local_plan.json")
        summary = control.read_json(control.OUT / "results.json")
        self.assertFalse(plan["preregistered"])
        self.assertFalse(plan["new_selection"])
        self.assertEqual(plan["k"], 16)
        self.assertEqual(plan["donor_permutation"], control.donor_permutation()[0])
        features = torch.load(control.FEATURES, map_location="cpu", weights_only=True, mmap=True)
        interactions, repairs = [], []
        torch.set_num_threads(2)
        for seed in control.SEEDS:
            result = control.read_json(control.OUT / f"seed{seed}/results.json")
            self.assertEqual(result["image_ids"], list(range(1000)))
            self.assertEqual(result["feature_indices"], list(range(14000, 15000)))
            self.assertEqual(result["donor_indices"], plan["donor_permutation"])
            self.assertEqual(result["state_sha256_before"], result["state_sha256_after"])
            self.assertTrue(result["native_layer_methods_restored"])
            original = control.read_json(control.ORIGINAL / f"seed{seed}/test/forward.json")["captions"]
            for condition in ("fault", "repair"):
                self.assertEqual(original[condition], result["captions"]["own_" + condition])
                prefix = result["autogenerated_own_fault_prefix_tokens"]
                for layer in result["attention"][condition]:
                    values = layer["per_valid_prefix_position"]
                    self.assertEqual([len(v) for v in values], [sum(t != 0 for t in p) for p in prefix])
                    np.testing.assert_allclose(layer["image_mean"], [np.mean(v) for v in values], atol=1e-14)
            metrics = {name: np.array([max(score(cap, r) for r in refs) for cap, refs in zip(cc, result["refs"])])
                       for name, cc in result["captions"].items()}
            for name, values in metrics.items():
                np.testing.assert_allclose(values, result["scores"]["own_refs"][name], atol=1e-14)
            interactions.append(metrics["own_repair"]-metrics["own_fault"]-metrics["donor_repair"]+metrics["donor_fault"])
            repairs.append(metrics["own_repair"]-metrics["donor_repair"])
            x = torch.load(control.OUT / f"seed{seed}/donor_features.pt", map_location="cpu", weights_only=True, mmap=True)
            mask = torch.load(control.ORIGINAL / f"seed{seed}/test/anchor_mask.pt", map_location="cpu", weights_only=True)
            mu = torch.load(control.ORIGINAL / f"seed{seed}/train_mean.pt", map_location="cpu", weights_only=True)
            proj = PatchProjector(768, 256).eval().requires_grad_(False)
            proj.load_state_dict(torch.load(control.MODELS / f"seed{seed}/model_proj.pt", map_location="cpu", weights_only=True))
            ids = [0, 17, 499, 999]
            with torch.no_grad():
                projected = proj(features[[14000 + result["donor_indices"][i] for i in ids]])
            torch.testing.assert_close(x[ids][mask[ids]], projected[mask[ids]], rtol=1e-5, atol=2e-5)
            for start in range(0, 1000, 32):
                xx, mm = x[start:start+32], mask[start:start+32]
                self.assertTrue(torch.equal(xx[~mm], mu.expand_as(xx)[~mm]))
            null = result["null"]
            self.assertEqual(null["shared_fault_prefix_tokens"], result["autogenerated_own_fault_prefix_tokens"])
            self.assertEqual(null["max_abs_valid_prefix_logits"], max(null["per_image_max_abs_valid_prefix_logits"]))
            self.assertEqual(null["identical_captions_across_masks_and_plans"], len(set(result["captions"]["null_fault"] + result["captions"]["null_repair"])) == 1)
        for name, rows in (("interaction", interactions), ("repair_own_minus_donor", repairs)):
            delta = np.mean(rows, axis=0)
            rng = np.random.default_rng(0)
            idx = rng.integers(0, 1000, (10000, 1000))
            ci = np.percentile(delta[idx].mean(1), [2.5, 97.5])
            self.assertAlmostEqual(summary["comparisons"][name]["delta"], float(delta.mean()), places=13)
            np.testing.assert_allclose(summary["comparisons"][name]["ci95"], ci, atol=1e-14)
        snapshots = control.read_json(control.OUT / "sources.json")
        for mapping in snapshots.values():
            for path, expected in mapping.items():
                self.assertEqual(control.sha256(path), expected, path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
