"""
run_rev_denoiser_scaling.py — revision experiment E4: denoiser data / capacity scaling.

Fresh image-only CFGDenoisers trained with the run_cap_improved.train_cfg_denoiser recipe
(60 epochs, batch 32, Adam 3e-4 cosine, mask rate U(0.1,0.95), CFG drop 0.15, CE + 0.05*salience
KL) on the free captioner's teacher-forced A0 of train+val images (0..13999, 5 captions each),
seed 0:
  (i)  data fraction f in {0.10, 0.25, 0.50, 1.00} of the 14000 images (nested random subsets,
       all 5 captions of a kept image), n_mp = 2;  same number of EPOCHS (so fewer steps at low f).
  (ii) capacity n_mp in {1, 2, 4} at f = 1.00 (n_mp = 2 is the f = 1.00 run of (i)).
Evaluation (identical for every model, incl. the saved seed-42 model_denoiser.pt reference):
  * recovery: Top-1 / Top-5 at mask rates 0.5 and 1.0 on the TEST images' (14000..14999)
    free-captioner plans (5000 plans), denoiser(X, A_t, mask) conditional forward, argmax
    target, over masked rows (definition of run_cap_improved.recovery_test; all 40 rows incl.
    rows at padded caption positions). Also reported over 'content' rows only (rows whose
    teacher-forcing input token is not PAD). Same masks for every model (fixed RNG).
  * captions: MaskGIT K=4, CFG 2.0, softmax(10A), model_guided.pt, greedy; best-of-5 legacy
    ROUGE-1, BLEU-4, CIDEr-D on the 1000 test images.
  * no-routing reference: guided decoder with a UNIFORM plan (every row 1/256).
Output: results/revision/denoiser_scaling.json
"""
import time
from run_rev_den_common import *
from run_cap_improved import EPOCHS_DEN

CKPT_DIR = os.path.join(OUT_DIR, "rev_ckpt"); os.makedirs(CKPT_DIR, exist_ok=True)
EPOCHS = int(os.environ.get("REV_SMOKE_EPOCHS", EPOCHS_DEN))
FRACS = [0.10, 0.25, 0.50, 1.00]
NMPS = [1, 4]
SEED = 0


@torch.no_grad()
def recovery(den, A0_te, X_pairs, content):
    out = {}
    tgt = A0_te.argmax(-1)
    for mr in (0.5, 1.0):
        torch.manual_seed(999)   # identical masks for every model
        t1 = t5 = t1c = t5c = 0.0; n = nc = 0
        for s in range(0, len(A0_te), 500):
            a0 = A0_te[s:s + 500]
            A_t, mask = mask_attn(a0, mr)
            logits = den(X_pairs[s:s + 500], A_t, mask)
            m = mask.bool(); mc = m & content[s:s + 500]
            top5 = logits.topk(5, -1).indices
            h1 = logits.argmax(-1) == tgt[s:s + 500]
            h5 = (top5 == tgt[s:s + 500].unsqueeze(-1)).any(-1)
            t1 += h1[m].sum().item(); t5 += h5[m].sum().item(); n += m.sum().item()
            t1c += h1[mc].sum().item(); t5c += h5[mc].sum().item(); nc += mc.sum().item()
        out[str(mr)] = {"top1": round(t1 / n, 4), "top5": round(t5 / n, 4),
                        "top1_content": round(t1c / nc, 4), "top5_content": round(t5c / nc, 4)}
    return out


def main():
    T0 = time.time()
    tok = load_tok(); caps = load_caps()
    proj, free, guided = load_models(tok)
    X_all = project_all(proj)
    X_te = X_all[TEST_START:TEST_START + TEST_IMAGES]
    refs = caps[TEST_START:TEST_START + TEST_IMAGES]
    A0_tv, pimg_tv, _ = build_A0(free, X_all, caps, list(range(N_TV)), tok)
    A0_te, pimg_te, ids_te = build_A0(free, X_all, caps, list(range(TEST_START, TEST_START + TEST_IMAGES)), tok)
    X_pairs_te = X_all[pimg_te]                     # 5000 x 256 x 256 (1.3 GB on GPU)
    content = (ids_te[:, :-1] != 0)
    content = torch.cat([content, torch.zeros(len(content), 1, dtype=torch.bool, device=DEVICE)], 1)  # row 39 = zero pad row
    print("A0 tv", A0_tv.shape, "A0 test", A0_te.shape, flush=True)

    res = {"config": {"epochs": EPOCHS, "seed": SEED, "fracs": FRACS, "nmp": [1, 2, 4],
                      "n_tv_images": N_TV, "recovery_plans": len(A0_te), "note_epochs":
                      "same epoch count for every fraction (steps scale with data)"}, "runs": {}}

    def evaluate(den, name, train_sec=None, n_imgs=None, n_mp=None, n_params=None):
        cp = caption(guided, X_te, gen_plans(lambda Xb, sl: maskgit_cfg_vec(den, Xb, 4, CFG_SCALE), X_te), tok)
        sc = score(cp, refs)
        r = {"train_images": n_imgs, "n_mp": n_mp, "n_params": n_params, "train_sec": train_sec,
             "recovery": recovery(den, A0_te, X_pairs_te, content), "captions_metrics": summarize(sc),
             "captions": cp}
        print(name, r["recovery"], r["captions_metrics"], flush=True)
        return r, sc

    base = load_base_denoiser()
    rb, sc_base = evaluate(base, "saved_seed42", n_imgs=N_TV, n_mp=N_MP,
                           n_params=sum(p.numel() for p in base.parameters()))
    assert abs(rb["captions_metrics"]["lr1"] - 0.4876) < 1e-4
    res["reference_saved_model_denoiser_seed42"] = rb
    uni = torch.full((TEST_IMAGES, L_CAP, N_PATCHES), 1.0 / N_PATCHES, device=DEVICE)
    cu = caption(guided, X_te, uni, tok); scu = score(cu, refs)
    res["uniform_plan"] = {"captions_metrics": summarize(scu), "vs_saved_baseline": paired(scu, sc_base), "captions": cu}
    print("uniform", res["uniform_plan"]["captions_metrics"], flush=True)

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(N_TV, generator=g)
    configs = [(f, 2) for f in FRACS] + [(1.0, k) for k in NMPS]
    for f, k in configs:
        name = f"frac{f:.2f}_nmp{k}"
        keep = torch.zeros(N_TV, dtype=torch.bool); keep[perm[:int(round(f * N_TV))]] = True
        sel = keep.to(DEVICE)[pimg_tv].nonzero().squeeze(1)
        set_seed(SEED)
        den = CFGDenoiser(D_MODEL, N_PATCHES, L_CAP, k).to(DEVICE)
        t1 = time.time()
        train_cfg_denoiser_gpu(den, A0_tv[sel], pimg_tv[sel], X_all, EPOCHS, tag=name)
        ts = round(time.time() - t1, 1)
        torch.save(den.state_dict(), os.path.join(CKPT_DIR, f"scaling_{name}.pt"))
        r, sc = evaluate(den, name, ts, int(keep.sum()), k, sum(p.numel() for p in den.parameters()))
        r["vs_saved_baseline"] = paired(sc, sc_base)
        r["vs_uniform"] = paired(sc, scu)
        res["runs"][name] = r
        json.dump(res, open(os.path.join(OUT_DIR, "denoiser_scaling.json" if "REV_SMOKE_EPOCHS" not in os.environ
                                         else "denoiser_scaling_SMOKE.json"), "w"), indent=1)
        del den; torch.cuda.empty_cache()
    res["total_sec"] = round(time.time() - T0, 1)
    json.dump(res, open(os.path.join(OUT_DIR, "denoiser_scaling.json" if "REV_SMOKE_EPOCHS" not in os.environ
                                     else "denoiser_scaling_SMOKE.json"), "w"), indent=1)
    print("total sec", res["total_sec"])


if __name__ == "__main__":
    main()
