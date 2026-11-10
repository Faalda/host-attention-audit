"""
run_rev_fullcoco_eval.py — revision experiment E9: captioning evaluation of the full-COCO model
on all 5,000 Karpathy-test images (split="test").

Systems: free captioner (greedy); guided decoder + denoised plan (vectorised maskgit K=4, CFG 2.0,
softmax(10*A)); guided decoder + uniform plan. Metrics: best-of-5 legacy ROUGE-1 (lr1), BLEU-4 and
CIDEr-D (run_rev_eval_metrics helpers; BLEU-4 reported both as mean per-image and corpus-level),
bootstrap CIs over images, paired deltas. Plus gates sigma(gamma) per layer and plan concentration.
Saves denoised test plans (fp32) to results/full_coco/plans_test.pt for the protocol scripts.

Usage: CUDA_VISIBLE_DEVICES=0 python run_rev_fullcoco_eval.py
"""
import sys
import os
from run_rev_fullcoco_common import *
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for clean_pipeline./inversion. imports
from inversion.run_rev_eval_metrics import lr1, corpus_bleu_cider, boot_ci, norm
from pycocoevalcap.bleu.bleu import Bleu


@torch.no_grad()
def project(proj, feats, bs=500):
    out = torch.empty(len(feats), N_PATCHES, D_MODEL, device=DEVICE)
    for s in range(0, len(feats), bs):
        idx = torch.arange(s, min(len(feats), s + bs), device=DEVICE)
        out[s:s + len(idx)] = proj(feats.get(idx))
    return out


@torch.no_grad()
def denoised_plans(den, X, bs=250, k=4):
    return torch.cat([F.softmax(maskgit_cfg_vec(den, X[s:s + bs], k, CFG_SCALE) * 10.0, dim=-1)
                      for s in range(0, len(X), bs)])


@torch.no_grad()
def gen_tokens(model, X, plans=None, bs=250):
    out = []
    for s in range(0, len(X), bs):
        if plans is None:
            t = model.generate(X[s:s + bs])
        else:
            t = model.generate(X[s:s + bs], plans[s:s + bs])
        out += t.cpu().tolist()
    return out


def score(preds, refs):
    r1 = np.array([lr1(p, r) for p, r in zip(preds, refs)])
    b4, cid = corpus_bleu_cider(preds, refs)
    gts = {i: [norm(r) for r in rs] for i, rs in enumerate(refs)}
    res = {i: [norm(p) if norm(p) else "empty"] for i, p in enumerate(preds)}
    bc, _ = Bleu(4).compute_score(gts, res, verbose=0)
    return {"lr1": r1, "bleu4": np.asarray(b4), "cider": np.asarray(cid), "bleu4_corpus": float(bc[3])}


def summ(sc):
    out = {}
    for k in ("lr1", "bleu4", "cider"):
        m, lo, hi = boot_ci(sc[k] - 0.0)
        out[k] = round(m, 4); out[k + "_ci95"] = [round(lo, 4), round(hi, 4)]
    out["bleu4_corpus"] = round(sc["bleu4_corpus"], 4)
    return out


def paired(a, b):
    return {k: dict(zip(("delta", "lo", "hi"), [round(x, 4) for x in boot_ci(a[k] - b[k])])) for k in ("lr1", "cider")}


def plan_concentration(plans, toks):
    """Over the rows a caption actually consumes while generating (tokens after BOS, incl. the EOS step):
    used rows, distinct argmax patches, share of rows whose argmax is the caption's most-used patch."""
    used, distinct, top_rows, per_share = [], [], 0, []
    am = plans.argmax(-1).cpu()
    for b, t in enumerate(toks):
        gen = t[1:]
        L = (gen.index(2) + 1) if 2 in gen else len(gen)   # rows up to and incl. the EOS step (post-EOS tokens ignored)
        L = max(1, L)
        a = am[b, :L]
        cnt = torch.bincount(a, minlength=N_PATCHES)
        used.append(L); distinct.append(int((cnt > 0).sum())); top_rows += int(cnt.max()); per_share.append(float(cnt.max()) / L)
    return {"mean_used_rows": round(float(np.mean(used)), 2),
            "mean_distinct_argmax": round(float(np.mean(distinct)), 3),
            "median_distinct_argmax": float(np.median(distinct)),
            "share_rows_on_top_patch_pooled": round(top_rows / sum(used), 4),
            "share_rows_on_top_patch_mean_per_caption": round(float(np.mean(per_share)), 4)}


def evaluate_system(proj, free, den, gdec, feats, refs, tok, save_plans=None):
    X = project(proj, feats)
    t = time.time()
    plans = denoised_plans(den, X)
    if save_plans:
        torch.save(plans.cpu(), save_plans)
    uni = torch.full((len(X), L_CAP, N_PATCHES), 1.0 / N_PATCHES, device=DEVICE)
    toks = {"free": gen_tokens(free, X), "guided_denoised": gen_tokens(gdec, X, plans),
            "guided_uniform": gen_tokens(gdec, X, uni)}
    caps = {k: [tok.dec(r) for r in v] for k, v in toks.items()}
    sc = {k: score(v, refs) for k, v in caps.items()}
    res = {"n_images": len(X), "systems": {k: summ(v) for k, v in sc.items()},
           "paired": {"denoised_minus_uniform": paired(sc["guided_denoised"], sc["guided_uniform"]),
                      "denoised_minus_free": paired(sc["guided_denoised"], sc["free"])},
           "gates": [round(g, 4) for g in gdec.gate_values()],
           "plan_concentration": plan_concentration(plans, toks["guided_denoised"]),
           "sec": round(time.time() - t, 1)}
    return res, caps, plans


def main():
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    set_seed(0)
    tok = tok_from_vocab(json.load(open(os.path.join(FC, "tok_vocab.json"))))
    proj, free, den, gdec = load_final_models(len(tok.vocab))
    feats = GPUFeats("test")
    refs = load_caps("test")
    res, caps, _ = evaluate_system(proj, free, den, gdec, feats, refs, tok,
                                   save_plans=os.path.join(FC, "plans_test.pt"))
    res["train_config"] = json.load(open(os.path.join(FC, "train_config.json")))
    json.dump(res, open(os.path.join(FC, "captioning_eval.json"), "w"), indent=1)
    json.dump(caps, open(os.path.join(FC, "captions_test.json"), "w"))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
