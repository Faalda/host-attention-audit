"""
run_rev_fullcoco_common.py — shared helpers for revision experiment E9 (full-scale COCO captioner).

Streaming / GPU-resident versions of the original recipe (run_save_plans.py):
  * DINOv2 features are stored int8 + per-(image,channel) fp16 scale (run_rev_fullcoco_features.py)
    and kept on the GPU as int8; each minibatch is dequantised on the fly.
  * The projector is frozen after Phase 1 (as in the original, where A0CapDataset stores X=proj(patches)
    computed once after Phase 1), so X = proj(dequant(q)) per batch is the same quantity.
  * Teacher-forced plans A0 (head-averaged last-layer cross-attention of the frozen free captioner,
    padded with zero rows to L_CAP) are recomputed per batch instead of being stored; the free
    captioner is in eval mode, so this is deterministic and identical to extract_A0_cap.
  * Every loss / masking / CFG-drop / gating / salience term is copied line-by-line from
    run_captioning.train_free_cap, run_cap_improved.train_cfg_denoiser and train_guided_mixed.
    The only recipe changes are batch size / LR / #epochs (recorded in the run config).
"""
import os, sys, json, random, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_summarization import Tokenizer, DEVICE, D_MODEL, N_HEADS, N_LAYERS, L_SUM as L_CAP, N_MP
from run_captioning import PatchProjector, CaptionDecoder, N_CAP_PER
from run_cap_improved import (N_PATCHES, ENCODER_DIM, SoftGuidedCaptionDecoder, CFGDenoiser,
                              patch_salience, CFG_SCALE, CFG_DROP_P, SAL_LAMBDA, KL_LAMBDA)
from run_sum_masked import mask_attn
from run_rev_den_common import maskgit_cfg_vec

ROOT = "/home/faldabbas/myproject/attn_diffusion_gen"
FC = os.environ.get("FULL_COCO_DIR", os.path.join(ROOT, "results", "full_coco"))
FC_DATA = os.environ.get("FULL_COCO_DATA_DIR", FC)
CKPT = os.path.join(FC, "ckpt")
os.makedirs(CKPT, exist_ok=True)
N_VAL_HOLD = 2000


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class GPUFeats:
    """int8 DINOv2 features on the GPU; get(idx) -> fp32 [B,256,768]."""
    def __init__(self, split, n=None, start=0, device=DEVICE, chunk=4000):
        q = np.load(os.path.join(FC_DATA, f"dino_{split}_q.npy"), mmap_mode="r")
        s = np.load(os.path.join(FC_DATA, f"dino_{split}_scale.npy"), mmap_mode="r")
        n = n if n is not None else q.shape[0] - start
        self.q = torch.empty(n, N_PATCHES, ENCODER_DIM, dtype=torch.int8, device=device)
        for i in range(0, n, chunk):
            j = min(n, i + chunk)
            self.q[i:j] = torch.from_numpy(np.ascontiguousarray(q[start + i:start + j])).to(device)
        self.s = torch.from_numpy(np.ascontiguousarray(s[start:start + n])).to(device)
        self.n = n

    def get(self, idx):
        return self.q[idx].float() * self.s[idx].float().unsqueeze(1)

    def __len__(self): return self.n


def load_caps(split):
    return json.load(open(os.path.join(FC_DATA, f"caps_{split}.json")))["caps"]


def build_tok(train_caps, max_vocab=8000):
    tok = Tokenizer()
    tok.build([c for caps in train_caps for c in caps], max_vocab=max_vocab)
    return tok


def tok_from_vocab(vocab):
    tok = Tokenizer(); tok.vocab = vocab; tok.w2i = {w: i for i, w in enumerate(vocab)}
    return tok


def encode_pairs(caps, tok, img_ids):
    """Image-major (image, caption) pairs like CocoCapDataset. Returns (pair_img [P], pair_ids [P,L]) on GPU."""
    pi, pids = [], []
    for i in img_ids:
        for c in caps[i]:
            pi.append(i); pids.append(tok.enc(c, L_CAP, bos=True, eos=True))
    return torch.tensor(pi, device=DEVICE), torch.tensor(pids, device=DEVICE)


def epoch_batches(n, bs):
    perm = torch.randperm(n, device=DEVICE)
    return [perm[i:i + bs] for i in range(0, n, bs)]   # DataLoader(shuffle=True, drop_last=False)


@torch.no_grad()
def teacher_forced_A0(proj, free, feats, img_idx, ids):
    """== extract_A0_cap for one batch: X = proj(patches); A0 = last-layer head-avg cross-attn, padded to L_CAP."""
    X = proj(feats.get(img_idx))
    _, attn = free(X, ids[:, :-1], ret_last_attn=True)
    B, L1, NP = attn.shape
    A0 = torch.cat([attn, torch.zeros(B, L_CAP - L1, NP, device=attn.device)], 1)
    return A0, X


# ── checkpointing ─────────────────────────────────────────────────────────
def save_ckpt(name, ep, models, opt, sch, extra=None):
    st = {"ep": ep, "models": {k: m.state_dict() for k, m in models.items()},
          "opt": opt.state_dict(), "sch": sch.state_dict(),
          "rng": {"py": random.getstate(), "np": np.random.get_state(),
                  "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
          "extra": extra or {}}
    tmp = os.path.join(CKPT, f"{name}.pt.tmp")
    torch.save(st, tmp); os.replace(tmp, os.path.join(CKPT, f"{name}.pt"))


def load_ckpt(name, models, opt, sch):
    p = os.path.join(CKPT, f"{name}.pt")
    if not os.path.exists(p):
        return 0
    st = torch.load(p, map_location=DEVICE, weights_only=False)
    for k, m in models.items():
        m.load_state_dict(st["models"][k])
    opt.load_state_dict(st["opt"]); sch.load_state_dict(st["sch"])
    random.setstate(st["rng"]["py"]); np.random.set_state(st["rng"]["np"])
    torch.set_rng_state(st["rng"]["torch"]); torch.cuda.set_rng_state_all(st["rng"]["cuda"])
    print(f"  resumed {name} at epoch {st['ep']}", flush=True)
    return st["ep"]


# ── Phase 1: == run_captioning.train_free_cap ─────────────────────────────
def train_free_cap_stream(proj, dec, feats, pair_img, pair_ids, n_epochs, bs, lr, name="phase1",
                          val=None):
    params = list(proj.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    start = load_ckpt(name, {"proj": proj, "dec": dec}, opt, sch)
    for ep in range(start + 1, n_epochs + 1):
        proj.train(); dec.train(); tot = 0; n = 0; t0 = time.time()
        for idx in epoch_batches(len(pair_img), bs):
            patches = feats.get(pair_img[idx]); ids = pair_ids[idx]
            X = proj(patches)
            logits = dec(X, ids[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1), ignore_index=0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); tot += loss.item(); n += 1
        sch.step()
        msg = f"    [{name}] ep {ep:3d} cap_loss={tot / n:.4f} ({time.time() - t0:.0f}s)"
        if val is not None:
            msg += f"  val_loss={val_loss_free(proj, dec, feats, *val):.4f}"
        print(msg, flush=True)
        save_ckpt(name, ep, {"proj": proj, "dec": dec}, opt, sch)
    proj.eval(); dec.eval()


@torch.no_grad()
def val_loss_free(proj, dec, feats, pair_img, pair_ids, bs=500):
    proj.eval(); dec.eval(); tot = 0; cnt = 0
    for s in range(0, len(pair_img), bs):
        ids = pair_ids[s:s + bs]
        logits = dec(proj(feats.get(pair_img[s:s + bs])), ids[:, :-1])
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1),
                               ignore_index=0, reduction="sum").item()
        cnt += (ids[:, 1:] != 0).sum().item()
    proj.train(); dec.train()
    return tot / cnt


# ── Phase 2: == run_cap_improved.train_cfg_denoiser ───────────────────────
def train_cfg_denoiser_stream(den, proj, free, feats, pair_img, pair_ids, n_epochs, bs, lr, name="phase2"):
    opt = torch.optim.Adam(den.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    start = load_ckpt(name, {"den": den}, opt, sch)
    for ep in range(start + 1, n_epochs + 1):
        den.train(); tot = 0; n = 0; t0 = time.time()
        for idx in epoch_batches(len(pair_img), bs):
            A0, X = teacher_forced_A0(proj, free, feats, pair_img[idx], pair_ids[idx])
            mr = random.uniform(0.1, 0.95)
            A_t, mask = mask_attn(A0, mr)
            drop = (random.random() < CFG_DROP_P)
            logits = den(X, A_t, mask, drop_X=drop)
            tgt = A0.argmax(-1)
            m = mask.bool()
            if not m.any():
                continue
            loss_ce = F.cross_entropy(logits[m], tgt[m])
            pred_plan = F.softmax(logits, dim=-1)
            sal = patch_salience(X)
            sal_exp = sal.unsqueeze(1).expand_as(pred_plan)
            pred_m = pred_plan[m]; sal_m = sal_exp[m]
            loss_sal = F.kl_div(torch.log(sal_m + 1e-8), pred_m, reduction='batchmean')
            loss = loss_ce + SAL_LAMBDA * loss_sal
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(den.parameters(), 1.0)
            opt.step(); tot += loss.item(); n += 1
        sch.step()
        print(f"    [{name}] ep {ep:3d} cfg_den_loss={tot / max(n, 1):.4f} ({time.time() - t0:.0f}s)", flush=True)
        save_ckpt(name, ep, {"den": den}, opt, sch)
    den.eval()


# ── Phase 3: == run_cap_improved.train_guided_mixed ───────────────────────
def train_guided_mixed_stream(gdec, den, proj, free, feats, pair_img, pair_ids, n_epochs, bs, lr,
                              name="phase3", oracle_frac=0.5):
    opt = torch.optim.Adam(gdec.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    start = load_ckpt(name, {"gdec": gdec}, opt, sch)
    den.eval()
    max_H = math.log(N_PATCHES)
    for ep in range(start + 1, n_epochs + 1):
        gdec.train(); tot_ce = tot_kl = tot_sal = 0; cnt = 0; t0 = time.time()
        for idx in epoch_batches(len(pair_img), bs):
            A0, X = teacher_forced_A0(proj, free, feats, pair_img[idx], pair_ids[idx])
            ids = pair_ids[idx]
            if random.random() < oracle_frac:
                plan = F.softmax(A0 * 10, dim=-1)
            else:
                with torch.no_grad():
                    k = random.choice([1, 2, 4, 8])
                    A_sampled = maskgit_cfg_vec(den, X, n_steps=k)
                plan = F.softmax(A_sampled * 10, dim=-1)
            tgt_in = ids[:, :-1]; tgt_out = ids[:, 1:]
            pad_mask = (tgt_out == 0)
            logits, attn_w = gdec(X, tgt_in, plan, ret_last_attn=True)
            loss_ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1), ignore_index=0)
            L = tgt_in.shape[1]
            plan_L = plan[:, :L, :]
            attn_w_safe = attn_w.clamp(min=1e-9)
            kl = (plan_L * (torch.log(plan_L.clamp(min=1e-9)) - torch.log(attn_w_safe))).sum(-1)
            plan_H = -(plan_L * torch.log(plan_L.clamp(min=1e-9))).sum(-1)
            conf_w = (1.0 - plan_H / max_H).clamp(0, 1).detach()
            valid = ~pad_mask[:, :L]
            loss_kl = (conf_w * kl)[valid].mean() if valid.any() else kl.mean() * 0
            sal = patch_salience(X)
            sal_exp_dec = sal.unsqueeze(1).expand_as(attn_w)
            attn_valid = attn_w[valid[:, :attn_w.shape[1]]]
            sal_valid = sal_exp_dec[valid[:, :attn_w.shape[1]]]
            loss_sal = F.kl_div(torch.log(sal_valid + 1e-8), attn_valid, reduction='batchmean')
            loss = loss_ce + KL_LAMBDA * loss_kl + SAL_LAMBDA * loss_sal
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(gdec.parameters(), 1.0)
            opt.step()
            tot_ce += loss_ce.item(); tot_kl += loss_kl.item(); tot_sal += loss_sal.item(); cnt += 1
        sch.step()
        g = [f"{x:.3f}" for x in gdec.gate_values()]
        print(f"    [{name}] ep {ep:3d} ce={tot_ce / cnt:.4f} kl={tot_kl / cnt:.4f} sal={tot_sal / cnt:.4f} "
              f"gates={g} ({time.time() - t0:.0f}s)", flush=True)
        save_ckpt(name, ep, {"gdec": gdec}, opt, sch)
    gdec.eval()


def freeze(*ms):
    for m in ms:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)


def load_final_models(vocab_len, device=DEVICE):
    proj = PatchProjector(ENCODER_DIM, D_MODEL).to(device)
    free = CaptionDecoder(vocab_len, D_MODEL, N_HEADS, N_LAYERS, n_src=N_PATCHES).to(device)
    den = CFGDenoiser(D_MODEL, N_PATCHES, L_CAP, N_MP).to(device)
    gdec = SoftGuidedCaptionDecoder(vocab_len, D_MODEL, N_HEADS, N_LAYERS).to(device)
    for name, m in [("model_proj", proj), ("model_free", free), ("model_denoiser", den), ("model_guided", gdec)]:
        m.load_state_dict(torch.load(os.path.join(FC, f"{name}.pt"), map_location=device, weights_only=True))
    freeze(proj, free, den, gdec)
    return proj, free, den, gdec
