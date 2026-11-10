"""
run_rev_den_common.py — shared helpers for revision experiments E7 (remedies with seeds)
and E4 (denoiser scaling). Nothing here modifies existing scripts or result files.

Design notes
  * All training data are kept on the GPU (X per IMAGE, A0 per (image, caption) pair, plus an
    image index), so a minibatch is an index gather rather than a DataLoader stack of
    [32,256,256] CPU tensors. The optimisation recipe (batch 32, shuffle each epoch, Adam
    LR=3e-4, cosine over epochs, per-batch random mask rate U(0.1,0.95), per-batch CFG drop
    p=0.15, grad-clip 1.0, CE on masked rows vs A0.argmax, + SAL_LAMBDA * salience KL for the
    image-only denoiser) is copied line-by-line from run_cap_improved.train_cfg_denoiser /
    run_text_denoiser.train_text_denoiser / run_oracle_distillation.finetune_distilled.
  * maskgit_cfg_vec / maskgit_text_cfg_vec are loop-free versions of the original maskgit
    functions (scatter instead of python loops); verify_maskgit() checks exact equality.
"""
import os, sys, json, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for clean_pipeline./inversion. imports
from clean_pipeline.run_summarization import Tokenizer, DEVICE, D_MODEL, N_HEADS, N_LAYERS, L_SUM as L_CAP, BATCH, LR, N_MP
from clean_pipeline.run_captioning import PatchProjector, CaptionDecoder, N_CAP_PER
from clean_pipeline.run_cap_improved import (N_PATCHES, ENCODER_DIM, N_IMAGES, VAL_IMAGES, TEST_IMAGES,
                              RESULTS_DIR, CACHE_PATH, SoftGuidedCaptionDecoder, CFGDenoiser,
                              maskgit_cfg, patch_salience, CFG_SCALE, CFG_DROP_P, SAL_LAMBDA)
from clean_pipeline.run_sum_masked import mask_attn
from inversion.run_rev_eval_metrics import lr1, corpus_bleu_cider, boot_ci

SAVE_DIR = os.path.join(RESULTS_DIR, "saved_plans")
OUT_DIR = os.path.join(RESULTS_DIR, "revision")
os.makedirs(OUT_DIR, exist_ok=True)
TEST_START = N_IMAGES + VAL_IMAGES
N_TV = N_IMAGES + VAL_IMAGES
CAPS_CACHE = os.path.join(OUT_DIR, "coco_caps_15k.json")   # ~4 MB text cache


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_tok():
    vocab = json.load(open(os.path.join(SAVE_DIR, "tok_vocab.json")))
    tok = Tokenizer(); tok.vocab = vocab; tok.w2i = {w: i for i, w in enumerate(vocab)}
    return tok


def load_caps():
    if os.path.exists(CAPS_CACHE):
        return json.load(open(CAPS_CACHE))
    from datasets import load_dataset
    ds = load_dataset("clip-benchmark/wds_mscoco_captions", split="train")
    ds = ds.remove_columns([c for c in ds.column_names if c != "txt"])
    caps = []
    for item in ds.select(range(N_TV + TEST_IMAGES)):
        c = [x.strip() for x in item["txt"].strip().split("\n") if x.strip()]
        caps.append(c[:N_CAP_PER])
    json.dump(caps, open(CAPS_CACHE, "w"))
    return caps


def load_models(tok):
    proj = PatchProjector(ENCODER_DIM, D_MODEL).to(DEVICE)
    proj.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_proj.pt"), map_location=DEVICE, weights_only=True))
    free = CaptionDecoder(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS, n_src=N_PATCHES).to(DEVICE)
    free.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_free.pt"), map_location=DEVICE, weights_only=True))
    guided = SoftGuidedCaptionDecoder(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)
    guided.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_guided.pt"), map_location=DEVICE, weights_only=True))
    for m in (proj, free, guided):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    return proj, free, guided


def load_base_denoiser():
    d = CFGDenoiser(D_MODEL, N_PATCHES, L_CAP, N_MP).to(DEVICE)
    d.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_denoiser.pt"), map_location=DEVICE, weights_only=True))
    d.eval()
    return d


@torch.no_grad()
def project_all(proj, n=None, bs=500):
    """X = proj(DINOv2 patches) for images [0, n) -> GPU fp32 [n,256,256]."""
    feats = torch.load(CACHE_PATH, weights_only=True, mmap=True)
    n = n or feats.shape[0]
    out = torch.empty(n, N_PATCHES, D_MODEL, device=DEVICE)
    for i in range(0, n, bs):
        out[i:i + bs] = proj(feats[i:min(n, i + bs)].to(DEVICE))
    del feats
    return out


@torch.no_grad()
def build_A0(free, X, caps, img_ids, tok, bs=250):
    """Equivalent of run_captioning.extract_A0_cap over CocoCapDataset(pairs image-major):
    teacher-forced last-layer cross-attention of the free captioner, padded to L_CAP rows.
    Returns A0 [P,L,N] (GPU), pair->image index [P] (GPU), cap ids [P,L] (GPU)."""
    pair_img, pair_ids = [], []
    for i in img_ids:
        for c in caps[i]:
            pair_img.append(i); pair_ids.append(tok.enc(c, L_CAP, bos=True, eos=True))
    pair_img = torch.tensor(pair_img, device=DEVICE)
    pair_ids = torch.tensor(pair_ids, device=DEVICE)
    P = len(pair_img)
    A0 = torch.zeros(P, L_CAP, N_PATCHES, device=DEVICE)
    for s in range(0, P, bs):
        ids = pair_ids[s:s + bs]
        _, attn = free(X[pair_img[s:s + bs]], ids[:, :-1], ret_last_attn=True)
        A0[s:s + bs, :attn.shape[1]] = attn
    return A0, pair_img, pair_ids


# ── vectorised MaskGIT (identical semantics to run_cap_improved.maskgit_cfg) ──
def _maskgit_core(pred_fn, B, N, n_steps, device):
    A = torch.full((B, L_CAP, N), 1.0 / N, device=device)
    mask = torch.ones(B, L_CAP, device=device)
    n_per_step = max(1, L_CAP // n_steps)
    for _ in range(n_steps):
        probs = F.softmax(pred_fn(A, mask), dim=-1)
        conf = probs.max(-1).values.masked_fill(~mask.bool(), -1e9)
        n_unmask = min(n_per_step, int(mask.sum(-1).min().item()))
        if n_unmask == 0:
            break
        _, topk = conf.topk(n_unmask, dim=-1)                       # [B,k]
        sel_mask = mask.gather(1, topk) > 0                          # original's `if mask[b,idx]>0`
        pred = probs.argmax(-1).gather(1, topk)                      # [B,k]
        onehot = F.one_hot(pred, N).float()                          # [B,k,N]
        cur = A.gather(1, topk.unsqueeze(-1).expand(-1, -1, N))
        new = torch.where(sel_mask.unsqueeze(-1), onehot, cur)
        A = A.scatter(1, topk.unsqueeze(-1).expand(-1, -1, N), new)
        mask = mask.scatter(1, topk, torch.where(sel_mask, torch.zeros_like(pred, dtype=mask.dtype),
                                                 mask.gather(1, topk)))
    A_out = F.softmax(pred_fn(A, mask), dim=-1)
    committed = (mask == 0).unsqueeze(-1).float()
    return committed * A + (1 - committed) * A_out


@torch.no_grad()
def maskgit_cfg_vec(den, X, n_steps=4, cfg_scale=CFG_SCALE):
    return _maskgit_core(lambda A, m: den.cfg_predict(X, A, m, scale=cfg_scale),
                         X.shape[0], X.shape[1], n_steps, X.device)


@torch.no_grad()
def maskgit_text_cfg_vec(den, X, t_vec, n_steps=4, scale_x=2.0, scale_t=1.5):
    return _maskgit_core(lambda A, m: den.cfg_predict_text(X, A, m, t_vec, scale_x, scale_t),
                         X.shape[0], X.shape[1], n_steps, X.device)


def verify_maskgit(den, X, n=64):
    a = maskgit_cfg(den, X[:n], n_steps=4, device=DEVICE)
    b = maskgit_cfg_vec(den, X[:n], n_steps=4)
    return float((a - b).abs().max())


# ── plan generation + captioning + scoring ─────────────────────────────────
@torch.no_grad()
def gen_plans(plan_fn, X, bs=200):
    """plan_fn(Xb, sl) -> raw maskgit output; returns sharpened plans softmax(10*A) on GPU."""
    out = []
    for s in range(0, len(X), bs):
        out.append(F.softmax(plan_fn(X[s:s + bs], slice(s, s + bs)) * 10.0, dim=-1))
    return torch.cat(out)


@torch.no_grad()
def caption(guided, X, plans, tok, bs=200):
    caps = []
    for s in range(0, len(X), bs):
        toks = guided.generate(X[s:s + bs], plans[s:s + bs])
        caps += [tok.dec(r) for r in toks.cpu().tolist()]
    return caps


def score(preds, refs_list):
    r1 = np.array([lr1(p, r) for p, r in zip(preds, refs_list)])
    b4, cid = corpus_bleu_cider(preds, refs_list)
    return {"lr1": r1, "bleu4": np.asarray(b4), "cider": np.asarray(cid)}


def summarize(sc):
    return {k: round(float(np.mean(v)), 4) for k, v in sc.items()}


def paired(sc_a, sc_b):
    out = {}
    for k in ("lr1", "cider"):
        m, lo, hi = boot_ci(sc_a[k] - sc_b[k])
        out[k] = {"delta": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    return out


# ── training loops (recipes copied from the original scripts) ──────────────
def _epoch_batches(n, bs):
    perm = torch.randperm(n, device=DEVICE)
    return [perm[i:i + bs] for i in range(0, n, bs)]   # DataLoader(drop_last=False) semantics


def train_cfg_denoiser_gpu(den, A0, pair_img, X, n_epochs, log_every=10, tag=""):
    """== run_cap_improved.train_cfg_denoiser (CE + SAL_LAMBDA*salience KL)."""
    opt = torch.optim.Adam(den.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    sal_all = None
    for ep in range(1, n_epochs + 1):
        den.train(); tot = 0.0; n = 0; t0 = time.time()
        for idx in _epoch_batches(len(A0), BATCH):
            a0 = A0[idx]; x = X[pair_img[idx]]
            mr = random.uniform(0.1, 0.95)
            A_t, mask = mask_attn(a0, mr)
            drop = (random.random() < CFG_DROP_P)
            logits = den(x, A_t, mask, drop_X=drop)
            tgt = a0.argmax(-1); m = mask.bool()
            if not m.any():
                continue
            loss_ce = F.cross_entropy(logits[m], tgt[m])
            pred_m = F.softmax(logits, dim=-1)[m]
            sal_m = patch_salience(x).unsqueeze(1).expand(-1, L_CAP, -1)[m]
            loss_sal = F.kl_div(torch.log(sal_m + 1e-8), pred_m, reduction='batchmean')
            loss = loss_ce + SAL_LAMBDA * loss_sal
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(den.parameters(), 1.0)
            opt.step(); tot += loss.item(); n += 1
        sch.step()
        if ep % log_every == 0 or ep == 1:
            print(f"    [{tag}] ep {ep:3d} cfg_den_loss={tot / max(n, 1):.4f} ({time.time() - t0:.1f}s/ep)", flush=True)
    den.eval()


def train_text_gpu(den, A0, pair_img, X, t_img, n_epochs, lr=LR, log_every=10, tag="", text_drop=0.15):
    """== run_text_denoiser.train_text_denoiser (lr=LR) and
    run_oracle_distillation.finetune_distilled (lr=LR*0.5). CE only (no salience term).
    t_img: text embedding per IMAGE index (GPU), gathered through pair_img."""
    opt = torch.optim.Adam(den.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    for ep in range(1, n_epochs + 1):
        den.train(); tot = 0.0; n = 0; t0 = time.time()
        for idx in _epoch_batches(len(A0), BATCH):
            a0 = A0[idx]; x = X[pair_img[idx]]; t = t_img[pair_img[idx]]
            mr = random.uniform(0.1, 0.95)
            A_t, mask = mask_attn(a0, mr)
            drop_X = (random.random() < CFG_DROP_P)
            drop_t = (random.random() < text_drop)
            logits = den(x, A_t, mask, t_vec=t, drop_X=drop_X, drop_t=drop_t)
            tgt = a0.argmax(-1); m = mask.bool()
            if not m.any():
                continue
            loss = F.cross_entropy(logits[m], tgt[m])
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(den.parameters(), 1.0)
            opt.step(); tot += loss.item(); n += 1
        sch.step()
        if ep % log_every == 0 or ep == 1:
            print(f"    [{tag}] ep {ep:3d} loss={tot / max(n, 1):.4f} ({time.time() - t0:.1f}s/ep)", flush=True)
    den.eval()


def train_img_distill_gpu(den, A0, pair_img, X, n_epochs, lr, tag=""):
    """Image-only analogue of finetune_distilled (CE only, CFG drop, no text)."""
    opt = torch.optim.Adam(den.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    for ep in range(1, n_epochs + 1):
        den.train(); tot = 0.0; n = 0
        for idx in _epoch_batches(len(A0), BATCH):
            a0 = A0[idx]; x = X[pair_img[idx]]
            A_t, mask = mask_attn(a0, random.uniform(0.1, 0.95))
            logits = den(x, A_t, mask, drop_X=(random.random() < CFG_DROP_P))
            tgt = a0.argmax(-1); m = mask.bool()
            if not m.any():
                continue
            loss = F.cross_entropy(logits[m], tgt[m])
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(den.parameters(), 1.0)
            opt.step(); tot += loss.item(); n += 1
        sch.step()
        if ep % 10 == 0 or ep == 1:
            print(f"    [{tag}] ep {ep:3d} loss={tot / max(n, 1):.4f}", flush=True)
    den.eval()


@torch.no_grad()
def encode_clip_text(captions, bs=256):
    from transformers import CLIPTokenizer, CLIPTextModel
    ctok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    cm = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    out = []
    for i in range(0, len(captions), bs):
        inp = ctok(captions[i:i + bs], padding=True, truncation=True, max_length=77, return_tensors="pt").to(DEVICE)
        out.append(cm(**inp).pooler_output.float())
    del cm
    torch.cuda.empty_cache()
    return torch.cat(out)
