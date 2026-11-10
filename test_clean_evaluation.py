"""CPU-only tests; never load training checkpoints, features, or remote datasets."""

import os

# Set before any legacy implementation imports or torch device discovery.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["ATTN_ENCODER"] = "dinov2"

import json
import math
from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import run_clean_evaluation as clean
from run_cap_improved import SoftGuidedCaptionDecoder
from run_rev_inversion_protocol import dec_forward, dec_generate, invert, make_plan
from run_summarization import Tokenizer


def source_fixture():
    refs = [[f"image {i} reference {k}" for k in range(5)] for i in range(3)]
    refs[0][1] = refs[0][0]  # Equal text must remain two distinct slots.
    return {"n_images": 3, "records": [
        {"image": i, "refs": refs[i], "baseline": "OLD DO NOT REUSE",
         "mismatch_source": (i + 1) % 3, "mismatch_target": refs[(i + 1) % 3][2]}
        for i in range(3)]}


class ProtocolTests(unittest.TestCase):
    def test_saved_slots_and_donors_preserved(self):
        source = source_fixture()
        records = clean.validate_source(source, n=3)
        jobs = clean.protocol_jobs(records)
        self.assertEqual(len(jobs), 18)
        self.assertEqual([j["k"] for j in jobs[:5]], list(range(5)))
        self.assertEqual(jobs[0]["target"], jobs[1]["target"])
        self.assertEqual([j["target"] for j in jobs[15:]], [r["mismatch_target"] for r in records])
        small = clean.protocol_jobs(records, slots=(0, 1), n=2)
        self.assertEqual(len(small), 6)
        self.assertEqual(small[-1]["target"], records[1]["mismatch_target"])
        self.assertEqual(records[1]["mismatch_source"], 2)  # Donor outside subset is retained.

    def test_bad_derangements_rejected(self):
        for defect in ("self", "duplicate", "target", "order", "slots"):
            source = source_fixture()
            if defect == "self":
                source["records"][0]["mismatch_source"] = 0
            elif defect == "duplicate":
                source["records"][0]["mismatch_source"] = 2
            elif defect == "target":
                source["records"][0]["mismatch_target"] = "not a donor caption"
            elif defect == "order":
                source["records"].reverse()
            else:
                source["records"][0]["refs"].pop()
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                clean.validate_source(source, n=3)

    def test_real_cached_cohort_without_model_loading(self):
        source = clean.read_json(clean.ROOT / "results/revision/inversion_protocol_n1000_s0.json")
        records = clean.validate_source(source)
        self.assertEqual(len(clean.protocol_jobs(records)), 6000)
        swaps = clean.read_json(clean.ROOT / "results/revision/swap_injection_n1000_s0.json")["records"]
        self.assertEqual(len(swaps), 901)
        self.assertEqual(len({j["image"] for j in swaps}), 901)
        all_caps = clean.read_json(clean.ROOT / "results/revision/coco_caps_15k.json")
        validation = clean.validation_jobs(all_caps[13000:14000])
        self.assertEqual(len(validation), 150)
        self.assertTrue(all(j["refs"] == all_caps[j["feature_index"]] for j in validation))
        self.assertFalse({j["feature_index"] for j in validation} & {14000 + j["image"] for j in swaps})

    def test_new_vocab_oov_not_filtered(self):
        tok = Tokenizer()
        tok.build(["a cat dog"])
        job = {"target": "a zebra dog,", "fake": "zebra"}
        audit = clean.target_audit(tok, job)
        self.assertTrue(audit["fake_oov"])
        self.assertTrue(audit["fake_surface_oov"])
        self.assertEqual(audit["encoded_unk_count"], 2)
        self.assertEqual(audit["target_tokens"], tok.enc(job["target"], 40, bos=True, eos=True))
        punctuated = clean.target_audit(tok, {"target": "a dog,", "fake": "dog"})
        self.assertFalse(punctuated["fake_oov"])
        self.assertTrue(punctuated["fake_surface_oov"])

    def test_protocol_compatible_with_both_scorers(self):
        from paper_cvpr import rescore_rouge
        from run_rev_eval_metrics import evaluate

        records = source_fixture()["records"]
        for slots in (range(5), (0, 1)):
            jobs = clean.protocol_jobs(records, slots=slots)
            # Deliberately worse captions ensure negative gains are not floored.
            caps = {"free": ["unrelated" for _ in jobs]}
            stats = {"free": [None] * len(jobs)}
            baseline = [r["refs"][0] for r in records]
            dump = clean.protocol_dump(records, baseline, jobs, caps, stats, {})
            self.assertEqual(dump["records"][0]["baseline"], baseline[0])
            self.assertNotEqual(dump["records"][0]["baseline"], records[0]["baseline"])
            scored = rescore_rouge.protocol(dump, require_five=len(slots) == 5)
            self.assertLess(scored["variants"]["free"]["protocol"]["single_ref"]["delta"], 0)
            self.assertEqual(scored["variants"]["free"]["target_slots_by_image"], [list(slots)] * 3)
            with tempfile.TemporaryDirectory(dir="/tmp/opencode") as folder:
                path = Path(folder) / "protocol.json"
                clean.atomic_json(path, dump)
                result = evaluate(str(path))
                self.assertLess(result["free"]["single_ref"]["rouge1"]["delta"], 0)

    def test_incomplete_dump_rejected(self):
        records = source_fixture()["records"]
        jobs = clean.protocol_jobs(records)
        with self.assertRaises(ValueError):
            clean.protocol_dump(records, ["base"] * 3, jobs, {"free": ["x"]}, {"free": [None]}, {})
        jobs[0] = jobs[1]
        with self.assertRaises(ValueError):
            clean.protocol_dump(records, ["base"] * 3, jobs, {"free": ["x"] * len(jobs)},
                                {"free": [None] * len(jobs)}, {})

    def test_atomic_json_no_nan_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as folder:
            path = Path(folder) / "results.json"
            with self.assertRaises(ValueError):
                clean.atomic_json(path, {"score": float("nan")})
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(folder).iterdir()), [])
            clean.atomic_json(path, {"score": -0.5})
            with self.assertRaises(FileExistsError):
                clean.atomic_json(path, {"score": 1})
            self.assertEqual(json.loads(path.read_text()), {"score": -0.5})
            self.assertEqual(len(list(Path(folder).iterdir())), 1)

    def test_gpu_guard_before_loading_inputs(self):
        with patch.object(clean, "read_json", side_effect=AssertionError("must not read inputs")):
            with self.assertRaises(SystemExit):
                clean.main(["--model-dir", "missing", "--out-dir", "missing", "--device", "cuda"])
        self.assertFalse(torch.cuda.is_initialized())

    def test_manifest_rejects_incomplete_smoke_and_wrong_splits(self):
        folder = Path("/tmp/opencode/models")
        manifest = {"status": "complete", "smoke": False,
                    "split_ids": {"train": list(range(13000)), "val": list(range(13000, 14000)),
                                  "test": list(range(14000, 15000))},
                    "checks": {"vocab_heldout_invariant": True},
                    "artifact_hashes": {n: "hash" for n in (*clean.MODEL_FILES, "captions_test.json")}}
        hashes = {str(folder / n): "hash" for n in manifest["artifact_hashes"]}
        clean.validate_training_manifest(manifest, hashes, folder)
        for key, value in (("status", "running"), ("smoke", True), ("split_ids", {}), ("checks", {}), ("artifact_hashes", {})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                clean.validate_training_manifest({**manifest, key: value}, hashes, folder)

    def test_gpu_lease_surrounds_execution_and_releases_on_error(self):
        events = []

        @contextmanager
        def lease(**kwargs):
            self.assertEqual(kwargs, {"wait": True, "minimum_free_mib": 24000})
            events.append("acquire")
            try:
                yield {"gpu_uuid": "GPU0-test-uuid", "index": 0}
            finally:
                events.append("release")

        def execute(args, gpu_uuid=None):
            self.assertEqual(events, ["acquire"])
            self.assertEqual(gpu_uuid, "GPU0-test-uuid")
            self.assertEqual(args.device, "cuda")
            events.append("execute")
            raise RuntimeError("test failure")

        with patch.dict(sys.modules, {"clean_gpu_guard": SimpleNamespace(gpu_lease=lease)}):
            with patch.object(clean, "run_evaluation", side_effect=execute), self.assertRaisesRegex(RuntimeError, "test failure"):
                clean.main(["--model-dir", "missing", "--out-dir", "missing",
                            "--device", "cuda", "--allow-gpu", "--wait-for-gpu"])
        self.assertEqual(events, ["acquire", "execute", "release"])
        self.assertFalse(torch.cuda.is_initialized())

    def test_cpu_preflight_never_acquires_gpu_lease(self):
        def no_lease(**kwargs):
            raise AssertionError("CPU preflight must not inspect or lock a GPU")

        with patch.dict(sys.modules, {"clean_gpu_guard": SimpleNamespace(gpu_lease=no_lease)}):
            with patch.object(clean, "run_evaluation") as execute:
                clean.main(["--model-dir", "missing", "--out-dir", "missing", "--device", "cuda", "--preflight-only"])
                self.assertEqual(execute.call_args.args[0].device, "cpu")
        self.assertFalse(torch.cuda.is_initialized())

    def test_real_lease_interface_blocks_busy_gpu_before_execution(self):
        import clean_gpu_guard

        @contextmanager
        def lock(path):
            yield None

        busy = {"gpu_uuid": "GPU-test", "index": 0, "free_mib": 48000,
                "utilization_percent": 0, "processes": [{"pid": os.getpid() + 1}]}
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), \
                patch.object(clean_gpu_guard, "exclusive_lock", side_effect=lock), \
                patch.object(clean_gpu_guard, "query_gpu0", return_value=busy), \
                patch.object(clean, "run_evaluation") as execute:
            with self.assertRaises(clean_gpu_guard.GPUBusyError):
                clean.main(["--model-dir", "missing", "--out-dir", "missing", "--device", "cuda", "--allow-gpu"])
            execute.assert_not_called()
        self.assertFalse(torch.cuda.is_initialized())

    def test_validation_jobs_deterministic_disjoint_and_oov_independent(self):
        refs = [["a cat on a chair"] * 5 for _ in range(1000)]
        refs[0] = ["a person"] * 5  # No eligible non-person object to replace.
        jobs = clean.validation_jobs(refs)
        self.assertEqual(jobs, clean.validation_jobs(refs))
        self.assertEqual(len(jobs), 150)
        self.assertEqual([j["feature_index"] for j in jobs], list(range(13001, 13151)))
        self.assertEqual([j["plan_index"] for j in jobs], list(range(150)))
        self.assertTrue(all("image" not in j for j in jobs))
        self.assertFalse({j["feature_index"] for j in jobs} & set(range(14000, 15000)))
        tok = Tokenizer()
        tok.build(["a"])
        self.assertTrue(all(clean.target_audit(tok, j)["fake_oov"] for j in jobs))
        with self.assertRaises(ValueError):
            clean.validation_jobs(refs[:999])
        with self.assertRaises(ValueError):
            clean.validation_jobs([["a person"] * 5 for _ in range(1000)])

    def test_token_edit_normalization(self):
        self.assertEqual(clean.normalized_token_edit("", ""), 0)
        self.assertEqual(clean.normalized_token_edit("A cat!", "a cat"), 0)
        self.assertEqual(clean.normalized_token_edit("a zebra", "a cat"), .5)
        self.assertEqual(clean.normalized_token_edit("zebra a", "a cat"), 1)

    def test_matched_validation_locks_before_all901_test_jobs(self):
        val_jobs = [{"split": "validation", "feature_index": 13000 + i, "plan_index": i,
                     "k": 0, "target": "a zebra", "fake": "zebra", "refs": ["a cat"] * 5} for i in range(150)]
        test_jobs = [{"image": i, "feature_index": 14000 + i, "plan_index": i,
                      "k": 0, "target": "a bus", "fake": "bus"} for i in range(901)]
        records = [{"refs": ["a cat"] * 5} for _ in test_jobs]
        calls = []
        tok = Tokenizer()
        tok.build(["a cat zebra bus"])
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as folder:
            args = SimpleNamespace(out_dir=Path(folder), device="cpu", batch=250, forward_batch=32)

            def generate(chunk, variants, tok, models, feats, plans, device, batch, checkpoints=()):
                is_test = chunk[0]["feature_index"] >= 14000
                calls.append(("test" if is_test else "validation", len(chunk), variants))
                if is_test:
                    lock = clean.read_json(args.out_dir / "matched_calibration_lock.json")
                    self.assertEqual(lock["selected_steps"], {"gain_rouge1": 1, "normalized_token_edit": 2})
                    self.assertEqual(tuple(checkpoints), (0, 1, 2))
                else:
                    self.assertFalse((args.out_dir / "matched_calibration_lock.json").exists())
                    self.assertTrue(all(13000 <= j["feature_index"] < 14000 and j["target"] == "a zebra" for j in chunk))
                caps = {step: ["a cat" if step == 0 else "zebra a" if step == 1 else "a zebra"] * len(chunk)
                        for step in checkpoints}
                return {"free": ["a zebra"] * len(chunk)}, {"free": [None] * len(chunk)}, caps

            def baseline(tok, models, feats, plans, device, batch, controls, feature_indices):
                self.assertEqual(feature_indices, list(range(13000, 13150)))
                return {"guided_denoised": ["a cat"] * 150}

            plans = torch.full((150, 40, 4), .25)
            with patch.object(clean, "validation_plans", return_value=plans), \
                    patch.object(clean, "forward_captions", side_effect=baseline), \
                    patch.object(clean, "run_jobs", side_effect=generate):
                lock = clean.calibrate_matched(val_jobs, tok, None, None, args, {})
                # Even adversarial test free outcomes cannot alter the validation lock.
                clean.run_matched(test_jobs, ["unrelated"] * 901, ["a cat"] * 901, records,
                                  tok, None, None, None, args, {}, lock)
            validation = clean.read_json(args.out_dir / "matched_validation.json")
            self.assertEqual(len(validation["jobs"]), 150)
            self.assertEqual(len(validation["hidden_captions_by_step"]), 21)
            self.assertEqual(validation["jobs_dropped_for_oov"], 0)
            result = clean.read_json(args.out_dir / "matched_gain.json")
            self.assertEqual(calls, [("validation", 150, ["free"]), ("validation", 150, ["hidden"]), ("test", 901, ["hidden"])])
            self.assertEqual(result["global_steps_from_validation"], {"gain_rouge1": 1, "normalized_token_edit": 2})
            self.assertEqual(result["n_evaluation"], 901)
            self.assertEqual(result["step0_baseline_mismatches"], [])
            self.assertEqual(len(result["hidden_captions_by_step"]["2"]), 901)
            self.assertNotIn("evaluation_selected_steps", result)

    def test_validation_maskgit_finite_masks_and_exact_k4_sharpening(self):
        class Denoiser:
            def __init__(self):
                self.masks = []

            def cfg_predict(self, X, A, mask, scale):
                self.masks.append(mask.clone())
                clean.validate_plans(A)
                self_case.assertTrue(torch.isfinite(mask).all())
                self_case.assertTrue(((mask == 0) | (mask == 1)).all())
                self_case.assertEqual(scale, 2.)
                # Deterministic, finite image-conditioned patch preferences.
                return X[:, :, 0][:, None, :].expand_as(A)

        self_case = self
        den = Denoiser()
        feats = torch.randn(14000, 4, 2)
        jobs = [{"feature_index": 13000 + i, "plan_index": i} for i in range(150)]
        models = {"proj": torch.nn.Identity(), "denoiser": den}
        before = torch.get_rng_state().clone()
        plans = clean.validation_plans(jobs, models, feats, "cpu", 150)
        expected = (10 * torch.nn.functional.one_hot(feats[13000:13150, :, 0].argmax(-1), 4).float()).softmax(-1)
        self.assertTrue(torch.equal(plans, expected[:, None, :].expand_as(plans)))
        self.assertEqual([int(mask[0].sum()) for mask in den.masks], [40, 30, 20, 10, 0])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        with patch.object(den, "cfg_predict", side_effect=lambda X, A, mask, scale: torch.full_like(A, float("nan"))):
            with self.assertRaises(ValueError):
                clean.validation_plans(jobs, models, feats, "cpu", 150)
        jobs[-1]["feature_index"] = 14000
        with self.assertRaises(ValueError):
            clean.validation_plans(jobs, models, feats, "cpu", 150)

    def test_job_feature_indices_are_global_and_plan_indices_are_local(self):
        tok = Tokenizer()
        tok.build(["a cat"])
        feats = torch.arange(15000, dtype=torch.float32).reshape(-1, 1, 1)
        plans = torch.tensor([.1, .2, .3, .4]).expand(2, 40, 4).clone()
        plans[1] = plans[1].flip(-1)
        models = {"proj": torch.nn.Identity(), "guided": None}
        jobs = [{"feature_index": 13027, "plan_index": 1, "target": "a cat"},
                {"image": 7, "feature_index": 14007, "plan_index": 0, "target": "a cat"}]

        def optimize(dec, X, A, target, variant, steps, checkpoints):
            self.assertEqual(X.flatten().tolist(), [13027., 14007.])
            self.assertTrue(torch.equal(A, plans[[1, 0]]))
            return [[1, 2], [1, 2]], [None, None], {}

        with patch.object(clean, "optimize", side_effect=optimize):
            captions, _, _ = clean.run_jobs(jobs, ["free"], tok, models, feats, plans, "cpu", 2)
        self.assertEqual(captions["free"], ["", ""])


class CorruptionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.plans = torch.rand(4, 40, 8).softmax(-1)

    def test_zero_exact_identity_and_no_mutation(self):
        before = self.plans.clone()
        result, mask, _ = clean.corrupt_plans(self.plans, 0)
        self.assertTrue(torch.equal(result, before))
        self.assertFalse(mask.any())
        self.assertTrue(torch.equal(self.plans, before))

    def test_all_rows_valid_sharpened_random_onehot(self):
        result, mask, indices = clean.corrupt_plans(self.plans, 1)
        self.assertTrue(mask.all())
        clean.validate_plans(result)
        expected = (10 * torch.nn.functional.one_hot(indices, 8).float()).softmax(-1)
        self.assertTrue(torch.equal(result, expected))
        self.assertTrue(torch.isfinite(result.clamp(min=1e-6).log()).all())

    def test_nested_deterministic_and_rng_independent(self):
        state = torch.get_rng_state().clone()
        previous = torch.zeros(4, 40, dtype=torch.bool)
        for fraction in clean.CORRUPTION_FRACTIONS:
            result, mask, indices = clean.corrupt_plans(self.plans, fraction)
            self.assertTrue(torch.equal(mask.sum(1), torch.full((4,), round(40 * fraction))))
            self.assertFalse((previous & ~mask).any())
            again, again_mask, again_indices = clean.corrupt_plans(self.plans, fraction)
            self.assertTrue(torch.equal(result, again))
            self.assertTrue(torch.equal(mask, again_mask))
            self.assertTrue(torch.equal(indices, again_indices))
            self.assertTrue(torch.equal(result[~mask], self.plans[~mask]))
            previous = mask
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_nan_inf_negative_and_nonunit_rejected_even_at_zero(self):
        for value in (float("nan"), float("inf"), -0.1, 2.):
            plans = self.plans.clone()
            plans[0, 0, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                clean.corrupt_plans(plans, 0)

    def test_corruption_runner_uses_each_actual_baseline_and_restores_clean(self):
        records = source_fixture()["records"]
        plans = self.plans[:3]
        baselines = [r["refs"][0] for r in records]
        seen = []

        def forward(tok, models, feats, A, device, batch, controls):
            seen.append(A.clone())
            return {"guided_denoised": baselines if torch.equal(A, plans) else ["unrelated"] * 3}

        def jobs(jobs, variants, tok, models, feats, A, device, batch):
            self.assertTrue(torch.equal(A, seen[-1]))
            return {"free": ["unrelated"] * len(jobs)}, {"free": [None] * len(jobs)}, {}

        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as folder:
            args = SimpleNamespace(corruption_images=3, out_dir=Path(folder), device="cpu", batch=2, forward_batch=2)
            with patch.object(clean, "forward_captions", side_effect=forward), patch.object(clean, "run_jobs", side_effect=jobs):
                clean.run_corruption(records, baselines, None, None, None, plans, args, {})
            zero = clean.read_json(Path(folder) / "corruption_0.json")
            full = clean.read_json(Path(folder) / "corruption_1.json")
            self.assertEqual(len(seen), 4)
            self.assertTrue(torch.equal(seen[0], plans))
            self.assertLess(zero["per_image_scores"][0]["free_true_single_gain"], 0)
            self.assertEqual(zero["per_image_scores"][0]["repair_gain"], 0)
            self.assertGreater(full["per_image_scores"][0]["repair_gain"], 0)
            self.assertEqual(full["records"][0]["variants"]["repair"]["true"][:2], [baselines[0]] * 2)
            self.assertEqual(full["records"][0]["baseline"], "unrelated")
            self.assertEqual(full["records"][0]["variants"]["free"]["true"][2:], [None] * 3)
            self.assertIn("cohort_summary", full)
            self.assertNotIn("split_summaries", full)
            self.assertFalse(full["metadata"]["grid_selection"])

    def test_first200_corruption_keeps_full1000_forward_batch_boundaries(self):
        records = [{"image": i, "refs": ["a cat"] * 5, "mismatch_source": (i + 1) % 1000,
                    "mismatch_target": "a cat"} for i in range(1000)]
        plans = torch.full((1000, 40, 4), .25)
        # Simulate a decoder sensitive to batch size: image199 must see 32, not 8.
        baseline = [str(min(32, 1000 - (i // 32) * 32)) for i in range(1000)]

        def forward(tok, models, feats, A, device, batch, controls):
            self.assertEqual(len(A), 1000)
            return {"guided_denoised": baseline}

        def jobs(jobs, variants, tok, models, feats, A, device, batch):
            self.assertEqual(len(jobs), 600)
            return {"free": ["a cat"] * len(jobs)}, {"free": [None] * len(jobs)}, {}

        from paper_cvpr import rescore_rouge
        paired = rescore_rouge.paired
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as folder:
            args = SimpleNamespace(corruption_images=200, out_dir=Path(folder), device="cpu", batch=250, forward_batch=32)
            with patch.object(clean, "forward_captions", side_effect=forward), \
                    patch.object(clean, "run_jobs", side_effect=jobs), \
                    patch.object(rescore_rouge, "paired", side_effect=lambda a, b: paired(a, b, resamples=20)):
                clean.run_corruption(records, baseline, None, None, None, plans, args, {})
            zero = clean.read_json(Path(folder) / "corruption_0.json")
            self.assertEqual(zero["n_images"], 200)
            self.assertEqual(zero["records"][199]["baseline"], "32")
            self.assertEqual(zero["cohort_summary"]["repair_gain"]["delta"], 0)


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        self.dec = SoftGuidedCaptionDecoder(9, 8, 2, 1, dropout=0).eval().requires_grad_(False)
        self.X = torch.randn(2, 4, 8)
        self.A = torch.rand(2, 40, 4).softmax(-1)
        self.target = torch.tensor([[1, 4, 5, 2] + [0] * 36, [1, 6, 2] + [0] * 37])

    def test_matches_original_inversion_and_finite_gradients(self):
        before = clean.model_digest({"guided": self.dec})
        for variant in clean.VARIANTS:
            with self.subTest(variant=variant):
                expected_plan, expected_delta, expected = invert(self.dec, self.X, self.A, self.target, variant, 2, .05)
                actual, stats, trajectory = clean.optimize(self.dec, self.X, self.A, self.target, variant, steps=2,
                                                            checkpoints=(0, 1, 2))
                self.assertEqual(actual, expected.tolist())
                self.assertEqual(trajectory[2], actual)
                self.assertTrue(all(math.isfinite(v) for st in stats for v in st.values()))
                self.assertGreater(stats[0]["max_abs_gradient_batch"], 0)
                logits = dec_forward(self.dec, self.X, self.target[:, :-1], expected_plan, expected_delta)
                ce = torch.nn.functional.cross_entropy(logits.transpose(1, 2), self.target[:, 1:], reduction="none")
                valid = self.target[:, 1:] != 0
                expected_ce = (ce * valid).sum(1) / valid.sum(1)
                self.assertEqual([st["final_target_ce"] for st in stats], expected_ce.tolist())
        self.assertEqual(before, clean.model_digest({"guided": self.dec}))
        self.assertTrue(all(p.grad is None for p in self.dec.parameters()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_zero_unbounded_identity_gradient_and_restoration(self):
        original = [layer.forward for layer in self.dec.layers]
        bias = torch.zeros_like(self.A, requires_grad=True)
        native = dec_forward(self.dec, self.X, self.target[:, :-1], self.A)
        with clean.additive_logits(self.dec, bias):
            actual = dec_forward(self.dec, self.X, self.target[:, :-1], self.A)
            self.assertTrue(torch.equal(actual, native))
            valid = self.target[:, 1:] != 0
            loss = torch.nn.functional.cross_entropy(actual[valid], self.target[:, 1:][valid])
            unbounded_grad = torch.autograd.grad(loss, bias)[0]
        theta = self.A.log().requires_grad_(True)
        pred = dec_forward(self.dec, self.X, self.target[:, :-1], make_plan(theta, "free"))
        bounded_grad = torch.autograd.grad(torch.nn.functional.cross_entropy(pred[valid], self.target[:, 1:][valid]), theta)[0]
        torch.testing.assert_close(unbounded_grad, bounded_grad, atol=2e-7, rtol=2e-3)
        self.assertTrue(torch.isfinite(unbounded_grad).all())
        self.assertGreater(unbounded_grad.abs().max().item(), 0)
        with self.assertRaisesRegex(RuntimeError, "test cleanup"):
            with clean.additive_logits(self.dec, bias):
                raise RuntimeError("test cleanup")
        self.assertEqual(original, [layer.forward for layer in self.dec.layers])
        for variant in ("bounded", "unbounded"):
            _, stats, _ = clean.optimize(self.dec, self.X, self.A, self.target, variant, steps=2)
            self.assertGreater(stats[0]["max_abs_gradient_batch"], 0)

    def test_corrupted_plan_is_actual_initialization(self):
        corrupted, _, _ = clean.corrupt_plans(self.A, .5)
        tokens, _, _ = clean.optimize(self.dec, self.X, corrupted, self.target, "free", steps=0)
        expected = dec_generate(self.dec, self.X, make_plan(corrupted.log(), "free"))
        self.assertEqual(tokens, expected.tolist())
        _, stats, _ = clean.optimize(self.dec, self.X, corrupted, self.target, "free", steps=2)
        self.assertGreater(stats[0]["max_abs_gradient_batch"], 0)

    def test_nan_inputs_and_unfrozen_model_rejected(self):
        bad = self.A.clone()
        bad[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            clean.optimize(self.dec, self.X, bad, self.target, "free", steps=1)
        bad_X = self.X.clone()
        bad_X[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            clean.optimize(self.dec, bad_X, self.A, self.target, "free", steps=1)
        self.dec.requires_grad_(True)
        with self.assertRaises(ValueError):
            clean.optimize(self.dec, self.X, self.A, self.target, "free", steps=1)


if __name__ == "__main__":
    unittest.main()
