"""CPU contract/equivalence tests; no dataset downloads or CUDA allocation."""
import os

# Hide devices before legacy imports discover CUDA; CPU Adam also checks streams.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import random
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

import run_clean_captioner as clean
from run_cap_improved import maskgit_cfg


class CleanCaptionerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.assertFalse(torch.cuda.is_initialized())
        self.enterContext(patch.object(torch.cuda, "_lazy_init",
                                       side_effect=AssertionError("CPU test attempted CUDA initialization")))
        self.enterContext(patch.object(torch._C, "_cuda_init",
                                       side_effect=AssertionError("CPU test attempted native CUDA initialization")))

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_split_disjoint_and_overlap_rejected(self):
        ids = clean.check_splits(clean.SPLITS)
        self.assertEqual([len(ids[s]) for s in ("train", "val", "test")], [13000, 1000, 1000])
        with self.assertRaises(ValueError):
            clean.check_splits({"train": (0, 2), "val": (1, 3)})

    def test_heldout_vocab_invariance_and_unknown(self):
        caps = [["training caption"]] * 13000 + [["heldoutword"]] * 2000
        a = clean.build_vocab(caps)
        caps[13000:] = [["other heldout tokens"]] * 2000
        b = clean.build_vocab(caps)
        self.assertEqual(a.vocab, b.vocab)
        self.assertEqual(a.enc("heldoutword", 4, bos=True), [1, 3, 2, 0])
        self.assertEqual(a.dec([1, 3, 2, 4]), "<UNK>")

    def test_nonfinite_rejected(self):
        clean.finite([torch.ones(3)], "valid")
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(FloatingPointError):
                clean.finite([torch.tensor([value])], "invalid")

    def test_maskgit_exact_and_sharpening(self):
        torch.manual_seed(8)
        den = clean.CFGDenoiser(256, 256, 40, 2).eval()
        X = torch.randn(2, 256, 256)
        for k in (1, 2, 4, 8):
            a = maskgit_cfg(den, X, k, device="cpu")
            b = clean.maskgit_vec(den, X, k)
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(b, F.one_hot(b.argmax(-1), 256).float()))
            self.assertTrue(torch.equal((a * 10).softmax(-1), (b * 10).softmax(-1)))

    def test_free_objective(self):
        torch.manual_seed(1)
        dec = clean.CaptionDecoder(12, 256, 4, 3, n_src=256).eval()
        X, ids = torch.randn(2, 256, 256), torch.randint(1, 12, (2, 40))
        expected = F.cross_entropy(dec(X, ids[:, :-1]).reshape(-1, 12), ids[:, 1:].reshape(-1), ignore_index=0)
        actual = clean.objective("free", {"free": dec}, X, ids)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_denoiser_objective_deterministic_and_gradients(self):
        den = clean.CFGDenoiser(256, 256, 40, 2)
        X = torch.randn(2, 256, 256)
        A = torch.randn(2, 40, 256).softmax(-1)
        ids = torch.ones(2, 40, dtype=torch.long)
        losses = []
        for _ in range(2):
            random.seed(123)
            torch.manual_seed(123)
            losses.append(clean.objective("denoiser", {"denoiser": den}, X, ids, A))
        torch.testing.assert_close(*losses, rtol=0, atol=0)
        losses[0].backward()
        clean.finite([p.grad for p in den.parameters() if p.grad is not None], "gradients")

    def test_guided_objective_matches_original(self):
        dec = clean.SoftGuidedCaptionDecoder(12, 256, 4, 3).eval()
        X = torch.randn(2, 256, 256)
        ids = torch.randint(1, 12, (2, 40))
        ids[:, 25:] = 0
        plan = (10 * F.one_hot(torch.randint(256, (2, 40)), 256).float()).softmax(-1)
        logits, attn = dec(X, ids[:, :-1], plan, ret_last_attn=True)
        valid = ids[:, 1:] != 0
        p = plan[:, :39]
        kl = (p * (p.clamp(min=1e-9).log() - attn.clamp(min=1e-9).log())).sum(-1)
        H = -(p * p.clamp(min=1e-9).log()).sum(-1)
        conf = (1 - H / clean.math.log(256)).clamp(0, 1).detach()
        sal = clean.patch_salience(X).unsqueeze(1).expand_as(attn)
        expected = (F.cross_entropy(logits.reshape(-1, 12), ids[:, 1:].reshape(-1), ignore_index=0)
                    + 0.1 * (conf * kl)[valid].mean()
                    + 0.05 * F.kl_div((sal[valid] + 1e-8).log(), attn[valid], reduction="batchmean"))
        actual = clean.objective("guided", {"guided": dec}, X, ids, plan=plan)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        actual_grads = torch.autograd.grad(actual, tuple(dec.parameters()))
        expected_grads = torch.autograd.grad(expected, tuple(dec.parameters()))
        for a, b in zip(actual_grads, expected_grads):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)

    def test_denoiser_salience_equivalence_and_underflow(self):
        torch.manual_seed(17)
        X = torch.randn(2, 256, 256)
        ids = torch.ones(2, 40, dtype=torch.long)
        A = torch.randn(2, 40, 256).softmax(-1)
        mask = torch.ones(2, 40)
        mask[:, ::3] = 0
        m = mask.bool()
        for extreme in (False, True):
            with self.subTest(extreme=extreme):
                logits = torch.randn(2, 40, 256)
                if extreme:
                    logits[..., 0] = -120
                logits.requires_grad_()
                sal = clean.patch_salience(X).unsqueeze(1).expand_as(logits)
                logsal = (sal[m] + 1e-8).log()
                ce = F.cross_entropy(logits[m], A.argmax(-1)[m])
                native = ce + 0.05 * F.kl_div(logsal, logits.softmax(-1)[m], reduction="batchmean")
                with patch.object(clean, "mask_attn", return_value=(A, mask)):
                    actual = clean.objective("denoiser", {"denoiser": Mock(return_value=logits)}, X, ids, A)
                actual_grad = torch.autograd.grad(actual, logits)[0]
                native_grad = torch.autograd.grad(native, logits)[0]
                self.assertTrue(torch.isfinite(actual_grad).all())
                torch.testing.assert_close(actual, native, rtol=1e-6, atol=1e-7)
                if extreme:
                    self.assertTrue((logits.softmax(-1)[m] == 0).any())
                    self.assertFalse(torch.isfinite(native_grad).all())
                    # FP64 keeps these probabilities positive, providing a native-KL reference.
                    ref = logits.detach().double().requires_grad_()
                    reference = (F.cross_entropy(ref[m], A.argmax(-1)[m])
                                 + 0.05 * F.kl_div(logsal.double(), ref.softmax(-1)[m], reduction="batchmean"))
                    reference_grad = torch.autograd.grad(reference, ref)[0].float()
                else:
                    reference_grad = native_grad
                torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-5, atol=1e-7)

    def test_guided_salience_underflow(self):
        torch.manual_seed(18)
        X = torch.randn(2, 256, 256)
        ids = torch.randint(1, 12, (2, 40))
        ids[:, 25:] = 0
        plan = (10 * F.one_hot(torch.randint(256, (2, 40)), 256).float()).softmax(-1)
        logits = torch.randn(2, 39, 12)
        scores = torch.randn(2, 39, 256)
        scores[..., 0] = -120
        scores.requires_grad_()
        attn = scores.softmax(-1)
        self.assertTrue((attn == 0).any())
        model = Mock(return_value=(logits, attn))
        actual = clean.objective("guided", {"guided": model}, X, ids, plan=plan)
        actual_grad = torch.autograd.grad(actual, scores, retain_graph=True)[0]
        valid = ids[:, 1:] != 0
        sal = clean.patch_salience(X).unsqueeze(1).expand_as(attn)
        logsal = (sal[valid] + 1e-8).log()
        old_sal = F.kl_div(logsal, attn[valid], reduction="batchmean")
        self.assertTrue(torch.isfinite(old_sal))
        self.assertFalse(torch.isfinite(torch.autograd.grad(old_sal, scores)[0]).all())
        # Reference the entire guided objective in FP64, where -120 does not underflow.
        ref = scores.detach().double().requires_grad_()
        ref_attn = ref.softmax(-1)
        p = plan[:, :39].double()
        logp = p.clamp_min(1e-9).log()
        conf = (1 + (p * logp).sum(-1) / clean.math.log(256)).clamp(0, 1).detach()
        kl = (p * (logp - ref_attn.clamp_min(1e-9).log())).sum(-1)
        reference = (F.cross_entropy(logits.double().flatten(0, 1), ids[:, 1:].reshape(-1), ignore_index=0)
                     + 0.1 * (conf * kl)[valid].mean()
                     + 0.05 * F.kl_div(logsal.double(), ref_attn[valid], reduction="batchmean"))
        self.assertTrue(torch.isfinite(actual_grad).all())
        torch.testing.assert_close(actual, reference.float(), rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(actual_grad, torch.autograd.grad(reference, ref)[0].float(),
                                   rtol=1e-5, atol=1e-7)

    def test_output_root_preflight_status_and_seed_isolation(self):
        from clean_gpu_guard import LOCK_PATH

        self.assertEqual(LOCK_PATH, clean.ROOT / "results/clean_small/gpu0.lock")
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as tmp:
            old, new = Path(tmp) / "old", Path(tmp) / "new"
            argv = ["run_clean_captioner.py", "--output-root", str(new)]
            with patch.object(clean, "OUT", old), \
                    patch.object(clean, "preflight", return_value=([], None, None, {"audit": "new"})) as preflight:
                with patch("sys.argv", argv + ["--preflight"]):
                    clean.main()
                self.assertEqual(json.loads((new / "preflight.json").read_text()), {"audit": "new"})
                preflight.reset_mock()
                for smoke in (False, True):
                    folder = new / ("smoke" if smoke else "") / "seed42"
                    folder.mkdir(parents=True)
                    clean.write_json(folder / "manifest.json", {})
                    clean.write_json(folder / "status.json", {"status": "sentinel"})
                    args = SimpleNamespace(output_root=new, smoke=smoke, resume=False)
                    with self.assertRaisesRegex(FileExistsError, "refusing overwrite"):
                        clean.run_seed(args, 42, None, None, None, None, None)
                with patch("sys.argv", argv + ["--status"]), patch("builtins.print") as output:
                    clean.main()
                self.assertEqual(output.call_count, 2)
                for call in output.call_args_list:
                    self.assertTrue(Path(call.args[0]).is_relative_to(new))
                    self.assertIn("sentinel", call.args[1])
                preflight.assert_not_called()
                self.assertFalse(old.exists())

    def test_main_shared_gpu_lease_all_seeds(self):
        for wait in (False, True):
            with self.subTest(wait=wait), tempfile.TemporaryDirectory(dir="/tmp/opencode") as tmp:
                events = []
                lease_info = dict(index=0, gpu_uuid="GPU-mocked")

                @contextmanager
                def lease(**kwargs):
                    self.assertEqual(kwargs, dict(wait=wait, poll_seconds=30,
                                                   timeout_seconds=None, minimum_free_mib=24000))
                    events.append("acquire")
                    try:
                        yield lease_info
                    finally:
                        events.append("release")

                def run_seed(args, seed, caps, features, tok, audit, info):
                    self.assertIs(info, lease_info)
                    self.assertEqual(args.output_root, Path(tmp))
                    events.append(seed)

                guard = SimpleNamespace(gpu_lease=Mock(side_effect=lease))
                argv = ["run_clean_captioner.py"] + (["--wait-for-gpu"] if wait else [])
                with patch.dict("sys.modules", clean_gpu_guard=guard), patch("sys.argv", argv), \
                        patch.object(clean, "OUT", Path(tmp)), \
                        patch.object(clean, "preflight", return_value=([], None, None, {})), \
                        patch.object(clean, "run_seed", side_effect=run_seed), \
                        patch.object(torch.cuda, "device_count", return_value=1), \
                        patch.object(torch.cuda, "set_device", side_effect=lambda _: events.append("cuda")), \
                        patch.object(torch.backends.cuda, "matmul", SimpleNamespace(allow_tf32=True)) as matmul, \
                        patch.object(torch.backends, "cudnn", SimpleNamespace(allow_tf32=True)) as cudnn, \
                        patch.object(torch, "use_deterministic_algorithms"):
                    clean.main()
                    self.assertFalse(matmul.allow_tf32)
                    self.assertFalse(cudnn.allow_tf32)
                guard.gpu_lease.assert_called_once()
                self.assertEqual(events, ["acquire", "cuda", 42, 7, 13, "release"])

    def test_main_gpu_guard_fail_closed_and_cpu_preflight(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as tmp:
            guard = SimpleNamespace(gpu_lease=Mock(side_effect=RuntimeError("GPU busy")))
            with patch.dict("sys.modules", clean_gpu_guard=guard), \
                    patch.object(clean, "OUT", Path(tmp)), \
                    patch.object(clean, "preflight", return_value=([], None, None, {})), \
                    patch.object(clean, "run_seed") as run, \
                    patch.object(torch.cuda, "device_count") as count, \
                    patch.object(torch.cuda, "set_device") as device:
                with patch("sys.argv", ["run_clean_captioner.py", "--preflight"]):
                    clean.main()
                guard.gpu_lease.assert_not_called()
                with patch("sys.argv", ["run_clean_captioner.py"]), self.assertRaisesRegex(RuntimeError, "GPU busy"):
                    clean.main()
                guard.gpu_lease.assert_called_once()
                with patch.dict("sys.modules", clean_gpu_guard=None), \
                        patch("sys.argv", ["run_clean_captioner.py"]), self.assertRaises(ModuleNotFoundError):
                    clean.main()
                run.assert_not_called()
                count.assert_not_called()
                device.assert_not_called()
            self.assertFalse(list(Path(tmp).glob("seed*/manifest.json")))

    def test_output_lock_exclusive_and_released(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as tmp:
            folder = Path(tmp)
            with clean.run_lock(folder):
                with self.assertRaisesRegex(RuntimeError, "output already in use"):
                    with clean.run_lock(folder):
                        self.fail("concurrent run acquired output lock")
            with self.assertRaises(InterruptedError):
                with clean.run_lock(folder):
                    raise InterruptedError("test release on failure")
            with clean.run_lock(folder):
                pass

    def test_compact_image_cache_preserves_row_ids(self):
        cache = clean.ImageCache(torch.tensor([0, 13000, 14000]), torch.tensor([[1.], [2.], [3.]]))
        self.assertEqual(cache[torch.tensor([14000, 0])].tolist(), [[3.], [1.]])
        with self.assertRaises(IndexError):
            cache[torch.tensor([1])]

    def test_epoch_checkpoint_resume_exact_cpu(self):
        self.check_checkpoint_resume(before_checkpoint=False)

    def test_epoch_checkpoint_resume_before_checkpoint_cpu(self):
        self.check_checkpoint_resume(before_checkpoint=True)

    def check_checkpoint_resume(self, before_checkpoint):
        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(8, 12)
                self.drop = torch.nn.Dropout(0.1)

            def forward(self, src, ids):
                return self.head(self.drop(src.mean(1)))[:, None, :].expand(-1, ids.shape[1], -1)

        def state():
            return dict(python=random.getstate(), cpu=torch.get_rng_state())

        def restore(st):
            random.setstate(st["python"])
            torch.set_rng_state(st["cpu"])

        def setup():
            random.seed(42)
            torch.manual_seed(42)
            models = dict(proj=torch.nn.Linear(8, 8), free=Decoder())
            features = torch.randn(16, 4, 8)
            data = dict(features=features)
            for split, lo in (("train", 0), ("val", 8)):
                data[split] = (torch.arange(lo, lo + 8).repeat(8), torch.randint(1, 12, (64, 40)), None)
            return models, data, dict(fingerprint="test", seed=42, stages={})

        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as tmp, patch.object(clean, "rng_state", state), \
                patch.object(clean, "restore_rng", restore):
            full, resumed = Path(tmp) / "full", Path(tmp) / "resumed"
            full.mkdir()
            resumed.mkdir()
            m, d, manifest = setup()
            clean.train_stage("free", m, d, 2, full, manifest, False)
            expected = {k: {n: p.detach().clone() for n, p in model.state_dict().items()} for k, model in m.items()}
            m, d, manifest2 = setup()
            original_save = clean.save_tensor

            def interrupt(path, value):
                if before_checkpoint and path.name == "checkpoint_free.pt" and value["epoch"] == 2:
                    raise InterruptedError("simulated interruption before durable epoch checkpoint")
                original_save(path, value)
                if not before_checkpoint and path.name == "checkpoint_free.pt" and value["epoch"] == 1:
                    raise InterruptedError("simulated interruption after durable epoch checkpoint")

            with patch.object(clean, "save_tensor", interrupt), self.assertRaises(InterruptedError):
                clean.train_stage("free", m, d, 2, resumed, manifest2, False)
            # Simulate an old append-only log/manifest ahead of the durable checkpoint.
            (resumed / "epochs.jsonl").write_text('{"stage":"free","epoch":2}\n' * 2)
            m, d, manifest2 = setup()
            manifest2["stages"]["guided"] = dict(history=[dict(epoch=99)])
            clean.train_stage("free", m, d, 2, resumed, manifest2, True)
            for name, model in m.items():
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(expected[name][key], value))
            self.assertEqual(manifest["stages"]["free"]["best"], manifest2["stages"]["free"]["best"])
            for a, b in zip(manifest["stages"]["free"]["history"], manifest2["stages"]["free"]["history"]):
                self.assertEqual(a["train_loss"], b["train_loss"])
                self.assertEqual(a["val_loss"], b["val_loss"])
            rows = [json.loads(line) for line in (resumed / "epochs.jsonl").read_text().splitlines()]
            self.assertEqual([(row["stage"], row["epoch"]) for row in rows], [("free", 1), ("free", 2)])
            self.assertEqual(set(manifest2["stages"]), {"free"})
            checkpoint = torch.load(resumed / "checkpoint_free.pt", weights_only=False)
            for row, committed in zip(rows, checkpoint["history"]):
                self.assertEqual({key: row[key] for key in committed}, committed)
            self.assertFalse((resumed / "epochs.jsonl.tmp").exists())


if __name__ == "__main__":
    unittest.main()
