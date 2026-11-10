"""
run_rev_inversion_protocol.py — revision experiment E1 (AAAI-27 reviews R1-W1/F1/F2, R2-Q2, R3-W1).

Symmetric, un-floored plan-inversion protocol for the small captioner.

For every test image i (refs R_i, |R_i| = 5) and every parameterisation v:
  * true-target:  for each k, invert toward r_k and decode y_k.
      - single-ref       : score(y_k, r_k)            vs  score(y_base, r_k)
      - held-out-ref     : best_{r in R_i \\ r_k} score(y_k, r)  vs  best_{r in R_i \\ r_k} score(y_base, r)
  * shuffled-target: invert toward m_i, a reference of a different image j = pi(i) (derangement).
      - target agreement : score(y_m, m_i)  vs  chance agreement score(y_base, m_i)
      - grounding        : best_{r in R_i} score(y_m, r)  vs  best_{r in R_i} score(y_base, r)
No max(baseline, .) floor anywhere. All captions are saved; metrics are computed by
run_rev_eval_metrics.py so that every metric uses exactly the same outputs.

Parameterisations (same 40x256 = 10,240 free variables in all three):
  free    : A0 = softmax(logits)                      (the paper's original inversion)
  onehot  : rows are one-hot patch indices, consumed as softmax(10*onehot) exactly like
            denoiser output; straight-through gradients. Restricts search to the set of
            plans the denoiser can emit (on-manifold oracle).
  hidden  : plan fixed to the denoised plan; optimise an additive bias on the decoder's
            input token states (40 positions x 256 dims). Matched-dimensional non-routing
            control: how much can an equally large generic steering vector achieve?

Usage:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 run_rev_inversion_protocol.py --n-images 1000
"""

import os, sys, json, argparse, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from run_summarization import Tokenizer, DEVICE, D_MODEL, N_HEADS, N_LAYERS, L_SUM as L_CAP
from run_captioning import PatchProjector, N_CAP_PER
from run_cap_improved import (N_PATCHES, ENCODER_DIM, N_IMAGES, VAL_IMAGES,
                              RESULTS_DIR, CACHE_PATH, SoftGuidedCaptionDecoder)

SAVE_DIR = os.path.join(RESULTS_DIR, "saved_plans")
OUT_DIR = os.path.join(RESULTS_DIR, "revision")
os.makedirs(OUT_DIR, exist_ok=True)
TEST_START = N_IMAGES + VAL_IMAGES
SHARPEN = 10.0


# ── decoder forward with optional additive input-state bias ─────────────────
def dec_forward(dec, X, tgt, plan, delta=None):
    B, L = tgt.shape
    x = dec.embed(tgt) + dec.pos(torch.arange(L, device=tgt.device))
    if delta is not None:
        x = x + delta[:, :L]
    causal = nn.Transformer.generate_square_subsequent_mask(L, device=tgt.device)
    pad = (tgt == 0)
    for layer in dec.layers:
        x = layer(x, X, plan[:, :L, :], tgt_mask=causal, tgt_pad=pad)
    return dec.head(x)


@torch.no_grad()
def dec_generate(dec, X, plan, delta=None, bos=1, eos=2):
    B = X.shape[0]
    toks = torch.full((B, 1), bos, device=X.device, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool, device=X.device)
    for _ in range(L_CAP - 1):
        logits = dec_forward(dec, X, toks, plan, delta)
        nxt = logits[:, -1].argmax(-1)
        nxt = torch.where(done, torch.zeros_like(nxt), nxt)
        toks = torch.cat([toks, nxt.unsqueeze(1)], 1)
        done |= (nxt == eos)
        if done.all():
            break
    return toks


def make_plan(logits, variant):
    if variant == "free":
        return F.softmax(logits, -1)
    if variant == "onehot":
        soft = F.softmax(logits, -1)
        hard = F.one_hot(logits.argmax(-1), logits.shape[-1]).float()
        st = hard + soft - soft.detach()
        return F.softmax(SHARPEN * st, -1)
    raise ValueError(variant)


def invert(dec, X, A_base, tgt, variant, n_steps, lr):
    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
    valid = tgt_out != 0
    if variant == "hidden":
        param = nn.Parameter(torch.zeros(X.shape[0], L_CAP, X.shape[-1], device=X.device))
    else:
        param = nn.Parameter(torch.log(A_base.clamp(min=1e-8)).clone())
    opt = torch.optim.Adam([param], lr=lr)
    for _ in range(n_steps):
        if variant == "hidden":
            pred = dec_forward(dec, X, tgt_in, A_base, delta=param)
        else:
            pred = dec_forward(dec, X, tgt_in, make_plan(param, variant))
        loss = F.cross_entropy(pred[valid], tgt_out[valid])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        if variant == "hidden":
            plan, delta = A_base, param.detach()
        else:
            plan, delta = make_plan(param.detach(), variant), None
        toks = dec_generate(dec, X, plan, delta)
    return plan, delta, toks


def plan_stats(plan, A_base, lengths):
    """Per-example stats over the first `length` rows (the rows actually used)."""
    out = []
    ent = -(plan * plan.clamp(min=1e-12).log()).sum(-1)                 # [B,L]
    am_changed = (plan.argmax(-1) != A_base.argmax(-1)).float()          # [B,L]
    m = 0.5 * (plan + A_base)
    js = 0.5 * ((plan * ((plan + 1e-12) / (m + 1e-12)).log()).sum(-1)
                + (A_base * ((A_base + 1e-12) / (m + 1e-12)).log()).sum(-1))
    for b, L in enumerate(lengths):
        L = max(1, int(L))
        out.append({"entropy": round(ent[b, :L].mean().item(), 4),
                    "argmax_changed": round(am_changed[b, :L].mean().item(), 4),
                    "js_to_denoised": round(js[b, :L].mean().item(), 4)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=1000)
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--variants", nargs="+", default=["free", "onehot", "hidden"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-mode", choices=["random", "knn"], default="random",
                    help="shuffled target from a random other image, or from the nearest DINOv2 neighbour")
    ap.add_argument("--mismatch-only", action="store_true", help="skip the 5 true-target inversions")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)

    with open(os.path.join(SAVE_DIR, "tok_vocab.json")) as f:
        vocab = json.load(f)
    tok = Tokenizer(); tok.vocab = vocab; tok.w2i = {w: i for i, w in enumerate(vocab)}

    proj = PatchProjector(ENCODER_DIM, D_MODEL).to(DEVICE)
    proj.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_proj.pt"), map_location=DEVICE, weights_only=True))
    dec = SoftGuidedCaptionDecoder(len(vocab), D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)
    dec.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_guided.pt"), map_location=DEVICE, weights_only=True))
    proj.eval(); dec.eval()
    for p in list(proj.parameters()) + list(dec.parameters()):
        p.requires_grad_(False)

    feats = torch.load(CACHE_PATH, weights_only=True)
    plans_test = torch.load(os.path.join(SAVE_DIR, "plans_test.pt"), weights_only=True)

    from datasets import load_dataset
    ds = load_dataset("clip-benchmark/wds_mscoco_captions", split="train")
    n = args.n_images
    refs = []
    for i in range(n):
        txt = ds[TEST_START + i]["txt"]
        refs.append([c.strip() for c in txt.strip().split("\n") if c.strip()][:N_CAP_PER])

    # derangement for shuffled targets (fixed seed)
    rng = random.Random(args.seed)
    perm = list(range(n))
    while any(p == i for i, p in enumerate(perm)):
        rng.shuffle(perm)
    if args.target_mode == "knn":
        g = F.normalize(feats[TEST_START:TEST_START + n].float().mean(1), dim=-1)
        sim = g @ g.T
        sim.fill_diagonal_(-1)
        perm = sim.argmax(1).tolist()
    mism = [rng.choice(refs[perm[i]]) for i in range(n)]

    # baseline captions
    base_caps = []
    with torch.no_grad():
        for s in range(0, n, args.batch):
            idx = list(range(s, min(n, s + args.batch)))
            X = proj(feats[[TEST_START + i for i in idx]].to(DEVICE))
            A = plans_test[idx].to(DEVICE)
            base_caps += [tok.dec(t) for t in dec_generate(dec, X, A).cpu().tolist()]

    # job list: (image, kind, k, target string)
    jobs = [] if args.mismatch_only else [(i, "true", k, r) for i in range(n) for k, r in enumerate(refs[i])]
    jobs += [(i, "mismatch", -1, mism[i]) for i in range(n)]

    records = {i: {"image": i, "refs": refs[i], "baseline": base_caps[i],
                   "mismatch_source": perm[i], "mismatch_target": mism[i],
                   "variants": {v: {"true": [None] * len(refs[i]), "mismatch": None,
                                    "true_stats": [None] * len(refs[i]), "mismatch_stats": None}
                                for v in args.variants}} for i in range(n)}
    t0 = time.time()
    for v in args.variants:
        for s in range(0, len(jobs), args.batch):
            chunk = jobs[s:s + args.batch]
            img_idx = [j[0] for j in chunk]
            X = proj(feats[[TEST_START + i for i in img_idx]].to(DEVICE))
            A = plans_test[img_idx].to(DEVICE)
            tgt = torch.tensor([tok.enc(j[3], max_len=L_CAP, bos=True, eos=True) for j in chunk],
                               dtype=torch.long, device=DEVICE)
            plan, delta, toks = invert(dec, X, A, tgt, v, args.n_steps, args.lr)
            lengths = (tgt[:, 1:] != 0).sum(-1).tolist()
            stats = plan_stats(plan, A, lengths)
            caps = [tok.dec(t) for t in toks.cpu().tolist()]
            for (i, kind, k, _), c, st in zip(chunk, caps, stats):
                rv = records[i]["variants"][v]
                if kind == "true":
                    rv["true"][k] = c; rv["true_stats"][k] = st
                else:
                    rv["mismatch"] = c; rv["mismatch_stats"] = st
            print(f"[{v}] {min(s + args.batch, len(jobs))}/{len(jobs)} jobs  {time.time() - t0:.0f}s", flush=True)

    out = {"target_mode": args.target_mode, "n_images": n, "n_steps": args.n_steps, "lr": args.lr, "seed": args.seed,
           "variants": args.variants, "param_count_per_inversion": L_CAP * N_PATCHES,
           "records": [records[i] for i in range(n)]}
    path = os.path.join(OUT_DIR, f"inversion_protocol_n{n}_s{args.seed}{args.tag}.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print("saved", path)


if __name__ == "__main__":
    main()
