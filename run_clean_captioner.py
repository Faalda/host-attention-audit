"""Leakage-free small captioner. Cached data only; GPU 0 only.

Run --preflight or --status without allocating CUDA tensors. --smoke uses two
minibatches per split/stage in a separate directory and is never a full result.
Resume an interrupted run with the identical command plus --resume.
Use --output-root to isolate runs with different source/configuration fingerprints.
"""
import os

# Respect callers that deliberately hide GPUs for CPU-only imports/preflight.
# Actual training is restricted to visibility "0" by the shared GPU lease.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["ATTN_ENCODER"] = "dinov2"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from run_summarization import Tokenizer, D_MODEL, N_HEADS, N_LAYERS, L_SUM, BATCH, LR, N_MP
from run_captioning import PatchProjector, CaptionDecoder
from run_cap_improved import (CFGDenoiser, SoftGuidedCaptionDecoder, patch_salience,
                              EPOCHS_FREE, EPOCHS_DEN, EPOCHS_GUIDED,
                              CFG_DROP_P, CFG_SCALE, SAL_LAMBDA, KL_LAMBDA)
from run_sum_masked import mask_attn

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/clean_small"
FEATURES = ROOT / "results/dinov2_patch_features_15k.pt"
CAPS = ROOT / "results/revision/coco_caps_15k.json"
PROTOCOL = ROOT / "results/revision/inversion_protocol_n1000_s0.json"
SPLITS = {"train": (0, 13000), "val": (13000, 14000), "test": (14000, 15000)}
CONFIG = dict(d_model=256, heads=4, layers=3, length=40, patches=256,
              encoder_dim=768, n_mp=2, vocab=6000, batch=32, lr=3e-4,
              epochs=dict(free=40, denoiser=60, guided=50), optimizer="Adam",
              scheduler="CosineAnnealingLR", clip_grad_norm=1.0,
              cfg_scale=2.0, cfg_drop=0.15, sal_lambda=0.05, kl_lambda=0.1,
              oracle_fraction=0.5, mask_rate=[0.1, 0.95], predicted_k=[1, 2, 4, 8],
               precision="float32; TF32 disabled", patience=None,
               numerics=dict(salience_kl="stablekl-v1",
                             denoiser="exp(log_softmax) * (log_softmax - log_salience)",
                             guided="p * (log(clamp_min(p, dtype.tiny)) - log_salience)"),
              selection="minimum deterministic validation batch-mean training objective",
              validation_rng="seed + 100000; reset every validation; restored afterward",
              checkpoint="every epoch; optimizer, scheduler, RNG, model, best choice",
              plan="softmax(10 * maskgit_cfg(K=4, CFG=2)); onehot commitments",
              optimizations=["GPU FP32 features and teacher plans", "vectorized MaskGIT",
                             "frozen predicted plans cached per image and K", "foreach Adam"],
               deviations=["train-only vocabulary", "train-only phases 2 and 3",
                           "validation-selected stages, full epoch budgets",
                           "stable salience KL backward at softmax underflow; guided log-only dtype.tiny floor"],
              id_namespace="cached row index, not original COCO image ID")


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save_tensor(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


@contextmanager
def run_lock(folder):
    """Hold a nonblocking, per-output lock through resume checks and all writes."""
    with (folder / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"output already in use: {folder}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def sync_checkpoint_history(folder, manifest):
    """Checkpoints are authoritative; logs/manifest may lag or contain a failed epoch."""
    stages, lines = {}, []
    for stage in CONFIG["epochs"]:
        path = folder / f"checkpoint_{stage}.pt"
        if not path.exists():
            continue
        st = torch.load(path, weights_only=False, map_location="cpu")
        assert st["fingerprint"] == manifest["fingerprint"]
        history = st["history"]
        assert [row["epoch"] for row in history] == list(range(1, st["epoch"] + 1))
        best = dict(epoch=None, val_loss=None)
        for row in history:
            if best["val_loss"] is None or row["val_loss"] < best["val_loss"]:
                best = dict(epoch=row["epoch"], val_loss=row["val_loss"])
            lines.append(json.dumps(dict(stage=stage, **row, best=best), allow_nan=False) + "\n")
        assert best == st["best"]
        stages[stage] = dict(best=best, history=history,
                             complete=st["epoch"] == st["scheduler"]["T_max"],
                             seconds=sum(row["seconds"] for row in history))
    tmp = folder / "epochs.jsonl.tmp"
    tmp.write_text("".join(lines))
    tmp.replace(folder / "epochs.jsonl")
    manifest["stages"] = stages
    write_json(folder / "manifest.json", manifest)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_vocab(caps):
    tok = Tokenizer()
    tok.build([c for row in caps[:13000] for c in row], max_vocab=6000)
    return tok


def check_splits(splits):
    ids = {name: set(range(*bounds)) for name, bounds in splits.items()}
    for a in ids:
        for b in ids:
            if a != b and ids[a] & ids[b]:
                raise ValueError(f"overlapping split IDs: {a}, {b}")
    return {name: sorted(values) for name, values in ids.items()}


def preflight():
    assert (D_MODEL, N_HEADS, N_LAYERS, L_SUM, BATCH, LR, N_MP) == (256, 4, 3, 40, 32, 3e-4, 2)
    assert (EPOCHS_FREE, EPOCHS_DEN, EPOCHS_GUIDED) == (40, 60, 50)
    assert (CFG_SCALE, CFG_DROP_P, SAL_LAMBDA, KL_LAMBDA) == (2.0, 0.15, 0.05, 0.1)
    caps = json.loads(CAPS.read_text())
    assert len(caps) == 15000
    assert all(len(row) == 5 and all(isinstance(c, str) and c.strip() for c in row) for row in caps)
    records = json.loads(PROTOCOL.read_text())["records"]
    assert len(records) == 1000
    for i, rec in enumerate(records):
        assert rec["image"] == i and rec["refs"] == caps[14000 + i], f"reference mismatch {i}"
    feats = torch.load(FEATURES, weights_only=True, mmap=True, map_location="cpu")
    assert feats.shape == (15000, 256, 768) and feats.dtype == torch.float32
    for chunk in feats.split(64):
        if not torch.isfinite(chunk).all():
            raise FloatingPointError("nonfinite feature cache")
    tok = build_vocab(caps)
    assert len(tok.vocab) == 6000 and tok.vocab[:4] == ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
    assert tok.enc("zzheldoutonlysentinelzz", 3, bos=True) == [1, 3, 2]
    changed = caps[:13000] + [["zzheldoutonlysentinelzz"] * 5 for _ in range(2000)]
    assert build_vocab(changed).vocab == tok.vocab
    unknown = {}
    for name, (lo, hi) in SPLITS.items():
        words = [w.lower() for row in caps[lo:hi] for c in row for w in c.split()]
        n = sum(w not in tok.w2i for w in words)
        unknown[name] = dict(unknown=n, tokens=len(words), rate=n / len(words))
    files = [FEATURES, CAPS, PROTOCOL, Path(__file__), ROOT / "run_captioning.py",
             ROOT / "run_cap_improved.py", ROOT / "run_summarization.py", ROOT / "run_sum_masked.py",
             ROOT / "run_rev_fullcoco_common.py", ROOT / "run_save_plans.py"]
    audit = dict(config=CONFIG, hashes={str(p.relative_to(ROOT)): sha256(p) for p in files},
                 split_ids=check_splits(SPLITS), unknown_rates=unknown,
                 checks=dict(feature_shape=list(feats.shape), feature_finite=True,
                             all_test_refs_equal_existing_protocol=True,
                             first_test_refs=caps[14000], last_test_refs=caps[14999],
                             vocab_heldout_invariant=True, unknown_id=3, disjoint_split_ids=True),
                 alignment_limit="Cache has no original image IDs; row order and all 1000 protocol refs checked.")
    print(json.dumps(dict(preflight="passed", unknown_rates=unknown,
                          first_test_refs=caps[14000], last_test_refs=caps[14999])), flush=True)
    return caps, feats, tok, audit


@torch.no_grad()
def maskgit_vec(den, X, n_steps=4):
    B, N, _ = X.shape
    A = torch.full((B, 40, N), 1.0 / N, device=X.device)
    mask = torch.ones(B, 40, device=X.device)
    remaining = 40
    for _ in range(n_steps):
        probs = den.cfg_predict(X, A, mask, scale=2.0).softmax(-1)
        conf = probs.max(-1).values.masked_fill(~mask.bool(), -1e9)
        n = min(max(1, 40 // n_steps), remaining)
        if not n:
            break
        top = conf.topk(n, dim=-1).indices
        pred = probs.argmax(-1).gather(1, top)
        A.scatter_(1, top.unsqueeze(-1).expand(-1, -1, N), F.one_hot(pred, N).float())
        mask.scatter_(1, top, 0)
        remaining -= n
    final = den.cfg_predict(X, A, mask, scale=2.0).softmax(-1)
    committed = (mask == 0).unsqueeze(-1).float()
    return committed * A + (1 - committed) * final


def objective(stage, models, X, ids, A0=None, plan=None):
    if stage == "free":
        logits = models["free"](X, ids[:, :-1])
        return F.cross_entropy(logits.flatten(0, 1), ids[:, 1:].reshape(-1), ignore_index=0)
    if stage == "denoiser":
        At, mask = mask_attn(A0, random.uniform(0.1, 0.95))
        logits = models["denoiser"](X, At, mask, drop_X=random.random() < 0.15)
        m = mask.bool()
        if not m.any():
            raise RuntimeError("empty masked batch; no silent skipped batches")
        ce = F.cross_entropy(logits[m], A0.argmax(-1)[m])
        # KL's differentiable probability target has a singular backward at zero.
        logpred = logits.log_softmax(-1)
        sal = patch_salience(X).unsqueeze(1).expand_as(logpred)
        loss_sal = (logpred[m].exp() * (logpred[m] - (sal[m] + 1e-8).log())).sum(-1).mean()
        return ce + 0.05 * loss_sal
    logits, attn = models["guided"](X, ids[:, :-1], plan, ret_last_attn=True)
    ce = F.cross_entropy(logits.flatten(0, 1), ids[:, 1:].reshape(-1), ignore_index=0)
    p = plan[:, :ids.shape[1] - 1]
    logp = p.clamp(min=1e-9).log()
    kl = (p * (logp - attn.clamp(min=1e-9).log())).sum(-1)
    confidence = (1 + (p * logp).sum(-1) / math.log(256)).clamp(0, 1).detach()
    valid = ids[:, 1:] != 0
    sal = patch_salience(X).unsqueeze(1).expand_as(attn)
    # Attention exposes probabilities only. Floor the log, not its multiplier:
    # zero mass contributes zero and has finite upstream softmax gradients.
    prob = attn[valid]
    logprob = prob.clamp_min(torch.finfo(prob.dtype).tiny).log()
    loss_sal = (prob * (logprob - (sal[valid] + 1e-8).log())).sum(-1).mean()
    return ce + 0.1 * (confidence * kl)[valid].mean() + 0.05 * loss_sal


def finite(tensors, label):
    # Infinity norms check every element without concatenating large parameters.
    values = torch.stack(torch._foreach_norm([t.detach() for t in tensors], float("inf")))
    if not torch.isfinite(values).all():
        raise FloatingPointError(f"nonfinite {label}")


def rng_state():
    return dict(python=random.getstate(), cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(0))


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["cpu"].cpu())
    torch.cuda.set_rng_state(state["cuda"].cpu(), 0)


def pairs(caps, tok, bounds, smoke):
    images, ids = [], []
    for i in range(*bounds):
        for c in caps[i]:
            images.append(i)
            ids.append(tok.enc(c, 40, bos=True, eos=True))
    if smoke:
        images, ids = images[:64], ids[:64]
    return torch.tensor(images, device="cuda"), torch.tensor(ids, device="cuda")


class ImageCache:
    """Compact GPU storage addressed by original cache-row IDs (also in smoke)."""

    def __init__(self, image_ids, values):
        self.values = values
        self.lookup = torch.full((15000,), -1, device=values.device, dtype=torch.long)
        self.lookup[image_ids] = torch.arange(len(image_ids), device=values.device)

    def __getitem__(self, ids):
        idx = self.lookup[ids]
        if (idx < 0).any():
            raise IndexError("uncached image ID")
        return self.values[idx]


def train_stage(stage, models, data, epochs, folder, manifest, resume):
    names = ["proj", "free"] if stage == "free" else [stage]
    params = [p for name in names for p in models[name].parameters()]
    for name, model in models.items():
        model.eval()
        for p in model.parameters():
            p.requires_grad_(name in names)
            p.grad = None
    opt = torch.optim.Adam(params, lr=3e-4, foreach=True)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ckpt = folder / f"checkpoint_{stage}.pt"
    history, start, best = [], 0, dict(epoch=None, val_loss=None)
    if ckpt.exists():
        if not resume:
            raise FileExistsError(f"use --resume: {ckpt}")
        st = torch.load(ckpt, weights_only=False, map_location="cpu")
        assert st["fingerprint"] == manifest["fingerprint"]
        for name in names:
            models[name].load_state_dict(st["models"][name])
        opt.load_state_dict(st["optimizer"])
        sch.load_state_dict(st["scheduler"])
        restore_rng(st["rng"])
        history, start, best = st["history"], st["epoch"], st["best"]
    sync_checkpoint_history(folder, manifest)
    for ep in range(start + 1, epochs + 1):
        t0 = time.monotonic()
        write_json(folder / "status.json", dict(seed=manifest["seed"], stage=stage, epoch=ep,
                                                 batch=0, status="training", updated=time.time()))
        for name in names:
            models[name].train()
        losses = []
        images, ids_all, A_all = data["train"]
        order = torch.randperm(len(images), device=images.device)
        for batch, idx in enumerate(order.split(32)):
            ids = ids_all[idx]
            X = models["proj"](data["features"][images[idx]]) if stage == "free" else data["X"][images[idx]]
            A0 = None if A_all is None else A_all[idx]
            plan = None
            if stage == "guided":
                if random.random() < 0.5:
                    plan = (10 * A0).softmax(-1)
                else:
                    k = random.choice([1, 2, 4, 8])
                    plan = (10 * F.one_hot(data["pred"][k][images[idx]].long(), 256).float()).softmax(-1)
            loss = objective(stage, models, X, ids, A0, plan)
            finite([loss], "training loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True, foreach=True)
            opt.step()
            finite(params, "updated parameters")
            losses.append(loss.detach())
            if (batch + 1) % 250 == 0:
                progress = dict(seed=manifest["seed"], stage=stage, epoch=ep, batch=batch + 1,
                                batches=math.ceil(len(images) / 32), status="training",
                                elapsed_epoch_seconds=time.monotonic() - t0, updated=time.time())
                write_json(folder / "status.json", progress)
                print(json.dumps(progress), flush=True)
        sch.step()
        for model in models.values():
            model.eval()
        saved_rng = rng_state()
        random.seed(manifest["seed"] + 100000)
        torch.manual_seed(manifest["seed"] + 100000)
        val_losses = []
        with torch.no_grad():
            images, ids_all, A_all = data["val"]
            for lo in range(0, len(images), 32):
                idx = slice(lo, lo + 32)
                ids = ids_all[idx]
                X = models["proj"](data["features"][images[idx]]) if stage == "free" else data["X"][images[idx]]
                A0 = None if A_all is None else A_all[idx]
                plan = None
                if stage == "guided":
                    if random.random() < 0.5:
                        plan = (10 * A0).softmax(-1)
                    else:
                        k = random.choice([1, 2, 4, 8])
                        plan = (10 * F.one_hot(data["pred"][k][images[idx]].long(), 256).float()).softmax(-1)
                loss = objective(stage, models, X, ids, A0, plan)
                finite([loss], "validation loss")
                assert not loss.requires_grad
                val_losses.append(loss)
        restore_rng(saved_rng)
        value = torch.stack(val_losses).mean().item()
        if best["val_loss"] is None or value < best["val_loss"]:
            best = dict(epoch=ep, val_loss=value)
            for name in names:
                save_tensor(folder / f"model_{name}.pt", models[name].state_dict())
        row = dict(epoch=ep, train_loss=torch.stack(losses).mean().item(), val_loss=value,
                   seconds=time.monotonic() - t0, train_batches=len(losses), val_batches=len(val_losses),
                   finite_loss_checks=len(losses) + len(val_losses),
                   finite_gradient_checks=len(losses), finite_parameter_checks=len(losses),
                   rejected_nonfinite=0)
        history.append(row)
        save_tensor(ckpt, dict(fingerprint=manifest["fingerprint"], epoch=ep, best=best,
                              models={name: models[name].state_dict() for name in names},
                              optimizer=opt.state_dict(), scheduler=sch.state_dict(),
                              rng=rng_state(), history=history))
        sync_checkpoint_history(folder, manifest)
        write_json(folder / "status.json", dict(seed=manifest["seed"], stage=stage, epoch=ep,
                                                 status="epoch_complete", best=best, updated=time.time()))
        print(json.dumps(dict(seed=manifest["seed"], stage=stage, **row, best=best)), flush=True)
    for name in names:
        models[name].load_state_dict(torch.load(folder / f"model_{name}.pt", weights_only=True))
        models[name].eval()
        models[name].requires_grad_(False)
        for p in models[name].parameters():
            p.grad = None


@torch.no_grad()
def build_teacher_cache(models, data):
    proj, free = models["proj"], models["free"]
    used = torch.unique(torch.cat([data[s][0] for s in ("train", "val")]))
    values = torch.empty(len(used), 256, 256, device="cuda")
    for lo in range(0, len(used), 32):
        values[lo:lo + 32] = proj(data["features"][used[lo:lo + 32]])
    finite([values], "projected train/val features")
    X = values if len(used) == 14000 else ImageCache(used, values)
    data["X"] = X
    for split in ("train", "val"):
        images, ids, _ = data[split]
        A = torch.zeros(len(images), 40, 256, device="cuda")
        for lo in range(0, len(images), 32):
            _, attn = free(X[images[lo:lo + 32]], ids[lo:lo + 32, :-1], ret_last_attn=True)
            A[lo:lo + 32, :39] = attn
        finite([A], "teacher plans")
        data[split] = images, ids, A


@torch.no_grad()
def build_pred_cache(models, data, folder):
    used = torch.unique(torch.cat([data[s][0] for s in ("train", "val")]))
    pred = {}
    h = hashlib.sha256()
    for k in (1, 2, 4, 8):
        pred[k] = torch.zeros(14000, 40, dtype=torch.uint8, device="cuda")
        for idx in used.split(32):
            A = maskgit_vec(models["denoiser"], data["X"][idx], k)
            finite([A], "predicted plans")
            assert torch.equal(A, F.one_hot(A.argmax(-1), 256).float())
            pred[k][idx] = A.argmax(-1).to(torch.uint8)
        h.update(pred[k][used].cpu().numpy().tobytes())
        print(f"cached frozen predicted plans K={k}", flush=True)
    data["pred"] = pred
    write_json(folder / "predicted_plan_cache.json", dict(sha256=h.hexdigest(), ks=[1, 2, 4, 8],
                                                         image_ids=used.cpu().tolist(), deterministic=True))


@torch.no_grad()
def export(models, features, caps, tok, folder, smoke):
    from run_rev_eval_metrics import r1, corpus_bleu_cider
    n = 64 if smoke else 1000
    refs = caps[14000:14000 + n]
    plans, free_caps, guided_caps = [], [], []
    for lo in range(14000, 14000 + n, 32):
        X = models["proj"](features[lo:min(lo + 32, 14000 + n)])
        plan = (10 * maskgit_vec(models["denoiser"], X, 4)).softmax(-1)
        finite([plan], "test plans")
        plans.append(plan.cpu())
        free_caps += [tok.dec(row) for row in models["free"].generate(X).cpu().tolist()]
        guided_caps += [tok.dec(row) for row in models["guided"].generate(X, plan).cpu().tolist()]
    plans = torch.cat(plans)
    assert plans.shape == (n, 40, 256)
    torch.testing.assert_close(plans.sum(-1), torch.ones(n, 40))
    save_tensor(folder / "plans_test.pt", plans)
    write_json(folder / "caps_test.json", refs)
    write_json(folder / "captions_test.json", dict(image_ids=list(range(14000, 14000 + n)),
                                                   refs=refs, free=free_caps, predicted=guided_caps))
    scores = dict(protocol="frozen forward only; best-ref standard ROUGE-1; pycocoevalcap CIDEr",
                  tokenization="run_rev_eval_metrics.norm; lowercase punctuation stripping, no PTB",
                  test_images=n, checkpoint_selection_uses_test=False, smoke=smoke)
    for name, captions in (("free", free_caps), ("predicted", guided_caps)):
        _, cider = corpus_bleu_cider(captions, refs)
        scores[name] = dict(rouge1=sum(r1(p, r) for p, r in zip(captions, refs)) / n,
                            cider=float(cider.mean()))
    write_json(folder / "standard_captioning_eval.json", scores)
    # Validate the actual serialized artifacts, not just in-memory values.
    assert json.loads((folder / "caps_test.json").read_text()) == refs
    assert json.loads((folder / "tok_vocab.json").read_text()) == tok.vocab
    assert torch.equal(torch.load(folder / "plans_test.pt", weights_only=True), plans)
    for name, model in models.items():
        state = torch.load(folder / f"model_{name}.pt", weights_only=True, map_location="cpu")
        assert state.keys() == model.state_dict().keys()
        for key, value in state.items():
            assert torch.equal(value, model.state_dict()[key].cpu())
    return scores


def run_seed(args, seed, caps, features_cpu, tok, audit, lease):
    folder = args.output_root / ("smoke" if args.smoke else "") / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    with run_lock(folder):
        if (folder / "manifest.json").exists() and not args.resume:
            raise FileExistsError(f"refusing overwrite: {folder}; use --resume")
        fingerprint = hashlib.sha256(json.dumps(dict(audit=audit, seed=seed, smoke=args.smoke),
                                                sort_keys=True).encode()).hexdigest()
        manifest = dict(**audit, seed=seed, smoke=args.smoke, fingerprint=fingerprint,
                        status="running", stages={}, torch_version=torch.__version__,
                        gpu=torch.cuda.get_device_name(0), started=time.time(),
                        gpu_lease=lease,
                        gpu_total_mib=torch.cuda.get_device_properties(0).total_memory // (1024 * 1024),
                        minimum_free_mib=24000,
                        deterministic_algorithms=True, cpu_threads=torch.get_num_threads(),
                        effective_epochs={s: 1 if args.smoke else n for s, n in CONFIG["epochs"].items()},
                        effective_train_pairs=64 if args.smoke else 65000,
                        effective_val_pairs=64 if args.smoke else 5000)
        if args.resume and (folder / "manifest.json").exists():
            manifest = json.loads((folder / "manifest.json").read_text())
            assert manifest["fingerprint"] == fingerprint, "resume configuration/input hash mismatch"
            if manifest["status"] == "complete":
                print(f"already complete: {folder}", flush=True)
                return
        write_json(folder / "manifest.json", manifest)
        write_json(folder / "tok_vocab.json", tok.vocab)
        random.seed(seed)
        torch.manual_seed(seed)
        t0 = time.monotonic()
        try:
            if args.smoke:
                used = torch.tensor(list(range(13)) + list(range(13000, 13013)) + list(range(14000, 14064)))
                features = ImageCache(used.cuda(), features_cpu[used].cuda())
            else:
                features = features_cpu.to("cuda")
            models = dict(proj=PatchProjector(768, 256).cuda(),
                          free=CaptionDecoder(6000, 256, 4, 3, n_src=256).cuda(),
                          denoiser=CFGDenoiser(256, 256, 40, 2).cuda(),
                          guided=SoftGuidedCaptionDecoder(6000, 256, 4, 3).cuda())
            for model in models.values():
                finite(list(model.parameters()), "initial parameters")
            data = dict(features=features)
            for split in ("train", "val"):
                images, ids = pairs(caps, tok, SPLITS[split], args.smoke)
                data[split] = (images, ids, None)
            train_stage("free", models, data, 1 if args.smoke else 40, folder, manifest, args.resume)
            build_teacher_cache(models, data)
            train_stage("denoiser", models, data, 1 if args.smoke else 60, folder, manifest, args.resume)
            build_pred_cache(models, data, folder)
            train_stage("guided", models, data, 1 if args.smoke else 50, folder, manifest, args.resume)
            manifest["forward_eval"] = export(models, features, caps, tok, folder, args.smoke)
            manifest["status"] = "complete"
            manifest["session_seconds"] = time.monotonic() - t0
            manifest["training_seconds"] = sum(s["seconds"] for s in manifest["stages"].values())
            manifest["completed"] = time.time()
            manifest["artifact_hashes"] = {p.name: sha256(p) for p in folder.iterdir()
                                             if p.name.startswith("model_") or p.name in
                                             ("plans_test.pt", "tok_vocab.json", "caps_test.json",
                                              "captions_test.json", "standard_captioning_eval.json")}
            write_json(folder / "manifest.json", manifest)
            write_json(folder / "status.json", dict(status="complete", seed=seed, smoke=args.smoke,
                                                     training_seconds=manifest["training_seconds"],
                                                     updated=time.time()))
            print(f"COMPLETE {folder} training_seconds={manifest['training_seconds']:.1f}", flush=True)
            del models, data, features
            torch.cuda.empty_cache()
        except BaseException as exc:
            manifest["status"] = "failed"
            manifest["failure"] = dict(error=repr(exc), time=time.time())
            write_json(folder / "manifest.json", manifest)
            write_json(folder / "status.json", dict(status="failed", seed=seed, error=repr(exc), updated=time.time()))
            raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 13])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--output-root", type=Path, default=OUT,
                    help="artifact root (the shared GPU 0 lease path is unchanged)")
    ap.add_argument("--wait-for-gpu", action="store_true",
                    help="wait for the shared GPU 0 lease; otherwise fail immediately if busy")
    args = ap.parse_args()
    if args.status:
        for path in sorted(args.output_root.glob("**/status.json")):
            print(str(path), path.read_text())
        return
    torch.set_num_threads(4)
    caps, features_cpu, tok, audit = preflight()
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "preflight.json", audit)
    if args.preflight:
        return
    # No fallback: a missing/busy guard must fail before CUDA allocation.
    from clean_gpu_guard import gpu_lease

    with gpu_lease(wait=args.wait_for_gpu, poll_seconds=30, timeout_seconds=None,
                   minimum_free_mib=24000) as lease:
        assert lease["index"] == 0 and lease["gpu_uuid"]
        assert torch.cuda.device_count() == 1
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        for seed in args.seeds:
            run_seed(args, seed, caps, features_cpu, tok, audit, lease)


if __name__ == "__main__":
    main()
