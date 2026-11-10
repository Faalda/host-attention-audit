"""
run_rev_distill_contentmask.py — follow-up to E7 (deviation from the original recipe, clearly
labelled). Diagnostic finding in E7: in the oracle-inverted plans, 69% of rows lie at padded
caption positions, receive no gradient during inversion, and stay exactly uniform, so their
distillation target argmax(uniform) = patch 0. The original finetune_distilled recipe trains
CE on those rows too, and the distilled denoiser collapses to "patch 0" for ~96% of rows.
Here the distillation CE is restricted to CONTENT rows (rows whose teacher-forcing input token
is not PAD); everything else is as in run_rev_remedies_seeds.py:
  (b-fix)  text model of seed s (rev_ckpt/text_ft_s{s}.pt) -> distill 30 ep, LR*0.5, text=CLIP(ref0)
  (b'-fix) image-only model_denoiser.pt -> distill 30 ep, LR*0.5
Same evaluation protocol, same shared baseline. Output: results/revision/remedies_seeds_contentmask.json
"""
from run_rev_den_common import *
from run_text_denoiser import TextCFGDenoiser, CLIP_DIM
from run_plan_inversion import tokenize_caption

N_DISTILL = 2000; EPOCHS_DIST = 30; SEEDS = [0, 1, 2]


def train_masked(den, A0, content, X, t_img, n_epochs, lr, text):
    opt = torch.optim.Adam(den.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    for ep in range(1, n_epochs + 1):
        den.train()
        for idx in [p for p in torch.randperm(len(A0), device=DEVICE).split(BATCH)]:
            a0 = A0[idx]; x = X[idx]
            A_t, mask = mask_attn(a0, random.uniform(0.1, 0.95))
            dX = random.random() < CFG_DROP_P
            if text:
                logits = den(x, A_t, mask, t_vec=t_img[idx], drop_X=dX, drop_t=(random.random() < 0.15))
            else:
                logits = den(x, A_t, mask, drop_X=dX)
            m = mask.bool() & content[idx]
            if not m.any():
                continue
            loss = F.cross_entropy(logits[m], a0.argmax(-1)[m])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(den.parameters(), 1.0); opt.step()
        sch.step()
    den.eval()


def main():
    tok = load_tok(); caps = load_caps()
    proj, free, guided = load_models(tok)
    X_all = project_all(proj)
    X_te = X_all[TEST_START:TEST_START + TEST_IMAGES]; refs = caps[TEST_START:TEST_START + TEST_IMAGES]
    base = load_base_denoiser()
    sc_base = score(caption(guided, X_te, gen_plans(lambda Xb, sl: maskgit_cfg_vec(base, Xb, 4, CFG_SCALE), X_te), tok), refs)
    A_or = torch.load(os.path.join(OUT_DIR, f"oracle_plans_train_{N_DISTILL}_inv100.pt"), weights_only=True).to(DEVICE)
    ids = torch.cat([tokenize_caption(tok, caps[i][0]) for i in range(N_DISTILL)]).to(DEVICE)
    content = torch.cat([ids[:, :-1] != 0, torch.zeros(N_DISTILL, 1, dtype=torch.bool, device=DEVICE)], 1)
    t_train = encode_clip_text([caps[i][0] for i in range(N_DISTILL)])
    free_caps = open(os.path.join(SAVE_DIR, "captions_test.txt")).read().split("\n")
    t_test = {"ref0": encode_clip_text([r[0] for r in refs]), "free": encode_clip_text(free_caps)}
    zeros = torch.zeros(TEST_IMAGES, CLIP_DIM, device=DEVICE)
    X_tr = X_all[:N_DISTILL]
    res = {"note": __doc__, "baseline": summarize(sc_base), "seeds": {}}
    arrs = {}
    for s in SEEDS:
        R = {}
        set_seed(s)
        d = TextCFGDenoiser(D_MODEL, N_PATCHES, L_CAP, N_MP, CLIP_DIM).to(DEVICE)
        d.load_state_dict(torch.load(os.path.join(OUT_DIR, "rev_ckpt", f"text_ft_s{s}.pt"), weights_only=True))
        train_masked(d, A_or, content, X_tr, t_train, EPOCHS_DIST, LR * 0.5, True)
        for name, bt, T in [("beta0", 0.0, zeros), ("beta1_ref0_LEAK", 1.0, t_test["ref0"]), ("beta1_free", 1.0, t_test["free"])]:
            pl = gen_plans(lambda Xb, sl: maskgit_text_cfg_vec(d, Xb, T[sl], 4, 2.0, bt), X_te)
            am = pl.argmax(-1)
            cp = caption(guided, X_te, pl, tok); sc = score(cp, refs)
            R[f"distill_text_fix/{name}"] = {"metrics": summarize(sc), "vs_baseline": paired(sc, sc_base),
                                             "patch0_row_frac": round(float((am == 0).float().mean()), 4), "captions": cp}
            arrs.setdefault(f"distill_text_fix/{name}", []).append(sc)
        set_seed(s)
        d2 = load_base_denoiser()
        train_masked(d2, A_or, content, X_tr, None, EPOCHS_DIST, LR * 0.5, False)
        pl = gen_plans(lambda Xb, sl: maskgit_cfg_vec(d2, Xb, 4, CFG_SCALE), X_te)
        cp = caption(guided, X_te, pl, tok); sc = score(cp, refs)
        R["distill_imgonly_fix/cfg"] = {"metrics": summarize(sc), "vs_baseline": paired(sc, sc_base),
                                        "patch0_row_frac": round(float((pl.argmax(-1) == 0).float().mean()), 4), "captions": cp}
        arrs.setdefault("distill_imgonly_fix/cfg", []).append(sc)
        print(s, {k: (v["metrics"], v["vs_baseline"]["lr1"], v["patch0_row_frac"]) for k, v in R.items()}, flush=True)
        res["seeds"][str(s)] = R
    agg = {}
    for k, L in arrs.items():
        e = {}
        for met in ("lr1", "cider"):
            v = [float(x[met].mean()) for x in L]
            m, lo, hi = boot_ci(np.mean([x[met] for x in L], 0) - sc_base[met])
            e[met] = {"per_seed": [round(x, 4) for x in v], "mean": round(float(np.mean(v)), 4),
                      "std": round(float(np.std(v, ddof=1)), 4), "delta_mean": round(m, 4),
                      "seedavg_perimage_delta_ci95": [round(lo, 4), round(hi, 4)]}
        agg[k] = e
    res["aggregate"] = agg
    print(json.dumps(agg, indent=1))
    json.dump(res, open(os.path.join(OUT_DIR, "remedies_seeds_contentmask.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
