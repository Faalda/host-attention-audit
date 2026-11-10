"""
run_rev_remedies_seeds.py — revision experiment E7: the two "remedies" with repeated seeds,
all evaluated against ONE shared image-only baseline (results/saved_plans/model_denoiser.pt)
with ONE evaluation protocol:
  1000 test images (14000..14999), MaskGIT K=4, image CFG 2.0, plan = softmax(10*A),
  SoftGuidedCaptionDecoder (model_guided.pt), greedy, EOS-terminated decoding;
  best-of-5 legacy set-unigram ROUGE-1 and CIDEr-D vs the 5 references.

(a) text-conditioned fine-tuning (run_text_denoiser.py recipe): TextCFGDenoiser initialised
    from model_denoiser.pt, 80 epochs, LR 3e-4, train images 0..12999 x 5 captions, text =
    CLIP(first reference) of the image (shared by all 5 A0s of the image, as in the original).
(b) oracle-plan distillation (run_oracle_distillation.py recipe): invert plans for train images
    0..1999 toward their first reference (100 Adam steps, lr 0.05, zero init), then fine-tune
    the seed's text-conditioned model from (a) on them for 30 epochs at LR*0.5 with text =
    CLIP(first reference). (The original started from the v1 text model checkpoint.)
(b') extra, not in the original: the same distillation applied to the image-only
    CFGDenoiser (init model_denoiser.pt, no text), isolating distillation from text-FT.

Text at test time (beta > 0): 'ref0' = CLIP(first test reference) -> TARGET LEAKAGE (the
same text is one of the 5 references we score against; this is what the old scripts did);
'free' = CLIP(free captioner's own greedy caption, results/saved_plans/captions_test.txt)
-> no leakage. beta = 0 nulls the text (image-only inference) and is the main number.

Output: results/revision/remedies_seeds.json (+ small checkpoints in results/revision/rev_ckpt/).
"""
import argparse, time
from run_rev_den_common import *
from run_text_denoiser import TextCFGDenoiser, CLIP_DIM, maskgit_text_cfg
from run_plan_inversion import tokenize_caption

CKPT_DIR = os.path.join(OUT_DIR, "rev_ckpt"); os.makedirs(CKPT_DIR, exist_ok=True)
EPOCHS_TEXT = int(os.environ.get("REV_SMOKE_EPOCHS", 80))
EPOCHS_DIST = int(os.environ.get("REV_SMOKE_EPOCHS", 30))
N_DISTILL = 2000
N_INV_STEPS = 100
INV_LR = 0.05


def invert_batch(guided, X, tgt_toks):
    """Batched run_plan_inversion.invert_plan (A_init=None). Loss = sum over images of each
    image's own mean token CE, so every image receives exactly its single-image gradient."""
    B = X.shape[0]
    logits = nn.Parameter(torch.zeros(B, L_CAP, N_PATCHES, device=DEVICE))
    opt = torch.optim.Adam([logits], lr=INV_LR)
    tgt_in, tgt_out = tgt_toks[:, :-1], tgt_toks[:, 1:]
    valid = (tgt_out != 0).float()
    for _ in range(N_INV_STEPS):
        plan = F.softmax(logits, dim=-1)
        pred = guided(X, tgt_in, plan[:, :tgt_in.shape[1], :])
        ce = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), tgt_out.reshape(-1), reduction="none").view(B, -1)
        loss = ((ce * valid).sum(1) / valid.sum(1)).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return F.softmax(logits.detach(), dim=-1)


def oracle_plans(guided, X_tr, caps, tok):
    path = os.path.join(OUT_DIR, f"oracle_plans_train_{N_DISTILL}_inv{N_INV_STEPS}.pt")  # 82 MB
    if os.path.exists(path):
        return torch.load(path, weights_only=True).to(DEVICE)
    out = []
    for s in range(0, N_DISTILL, 200):
        tt = torch.cat([tokenize_caption(tok, caps[i][0]) for i in range(s, s + 200)]).to(DEVICE)
        out.append(invert_batch(guided, X_tr[s:s + 200], tt))
        print(f"  inverted {s + 200}/{N_DISTILL}", flush=True)
    A = torch.cat(out)
    torch.save(A.cpu(), path)
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    T0 = time.time()
    set_seed(1234)
    tok = load_tok(); caps = load_caps()
    proj, free, guided = load_models(tok)
    base = load_base_denoiser()
    X_all = project_all(proj)
    X_te = X_all[TEST_START:TEST_START + TEST_IMAGES]
    refs = caps[TEST_START:TEST_START + TEST_IMAGES]
    res = {"protocol": {"n_test": TEST_IMAGES, "maskgit_K": 4, "cfg_scale_x": CFG_SCALE, "sharpen": 10.0,
                        "decoding": "greedy, EOS-terminated (tok.dec)", "metric_refs": "all 5",
                        "epochs_text": EPOCHS_TEXT, "epochs_distill": EPOCHS_DIST, "n_distill": N_DISTILL,
                        "inv_steps": N_INV_STEPS, "inv_lr": INV_LR, "seeds": args.seeds},
           "notes": {"beta_ref0": "text = CLIP(first test reference): TARGET LEAKAGE (that caption is among the 5 scored references)",
                     "beta_free": "text = CLIP(free captioner's own greedy caption): no leakage",
                     "beta0": "text nulled (null_t) -> image-only inference; main number"}}

    # ── shared baseline ──
    base_caps = caption(guided, X_te, gen_plans(lambda Xb, sl: maskgit_cfg_vec(base, Xb, 4, CFG_SCALE), X_te), tok)
    sc_base = score(base_caps, refs)
    res["baseline"] = {"metrics": summarize(sc_base), "captions": base_caps}
    print("baseline", res["baseline"]["metrics"], flush=True)
    assert abs(res["baseline"]["metrics"]["lr1"] - 0.4876) < 1e-4

    # ── text embeddings ──
    t_train = encode_clip_text([caps[i][0] for i in range(N_IMAGES)])
    old = torch.load(os.path.join(RESULTS_DIR, "text_denoiser", "train_text_embs.pt"), weights_only=True)
    res["check_train_text_embs_vs_old_cache_maxabs"] = float((t_train.cpu() - old).abs().max())
    free_caps_test = open(os.path.join(SAVE_DIR, "captions_test.txt")).read().split("\n")
    assert len(free_caps_test) == TEST_IMAGES
    t_test = {"ref0": encode_clip_text([r[0] for r in refs]), "free": encode_clip_text(free_caps_test)}
    print("text embs ok; diff vs old cache", res["check_train_text_embs_vs_old_cache_maxabs"], flush=True)

    # ── A0 for text fine-tuning: train images only (as run_text_denoiser.py) ──
    A0, pimg, _ = build_A0(free, X_all, caps, list(range(N_IMAGES)), tok)
    print("A0 train pairs", A0.shape, flush=True)

    # ── oracle plans (deterministic; shared by all seeds) ──
    t1 = time.time()
    A_or = oracle_plans(guided, X_all[:N_DISTILL], caps, tok)
    res["oracle_inversion_sec"] = round(time.time() - t1, 1)
    oldc = os.path.join(RESULTS_DIR, "oracle_distillation", "oracle_plans_train_2000.pt")
    if os.path.exists(oldc):
        o = torch.load(oldc, weights_only=True)["A0"][:N_DISTILL].to(DEVICE)
        res["check_oracle_vs_old_cache"] = {"maxabs": float((o - A_or).abs().max()),
                                            "argmax_agree": float((o.argmax(-1) == A_or.argmax(-1)).float().mean())}
        del o
        print("oracle vs old cache", res["check_oracle_vs_old_cache"], flush=True)
    # distilled-plan captions (sanity: how well do the targets themselves caption?)
    or_caps = caption(guided, X_all[:N_DISTILL], A_or, tok)
    res["oracle_train_fit_lr1_vs_ref0"] = round(float(np.mean([lr1(c, [caps[i][0]]) for i, c in enumerate(or_caps)])), 4)

    # verify vectorised text maskgit once
    tmp = TextCFGDenoiser(D_MODEL, N_PATCHES, L_CAP, N_MP, CLIP_DIM).to(DEVICE)
    tmp.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_denoiser.pt"), weights_only=True), strict=False)
    tmp.eval()
    a = maskgit_text_cfg(tmp, X_te[:32], t_test["ref0"][:32], n_steps=4, scale_x=2.0, scale_t=1.0, device=DEVICE)
    b = maskgit_text_cfg_vec(tmp, X_te[:32], t_test["ref0"][:32], 4, 2.0, 1.0)
    res["check_text_maskgit_vec_maxabs"] = float((a - b).abs().max()); del tmp

    def eval_text_model(den):
        out = {}
        zeros = torch.zeros(TEST_IMAGES, CLIP_DIM, device=DEVICE)
        confs = [("beta0", 0.0, zeros), ("beta1_ref0_LEAK", 1.0, t_test["ref0"]),
                 ("beta1.5_ref0_LEAK", 1.5, t_test["ref0"]), ("beta1_free", 1.0, t_test["free"])]
        for name, bt, T in confs:
            pl = gen_plans(lambda Xb, sl: maskgit_text_cfg_vec(den, Xb, T[sl], 4, 2.0, bt), X_te)
            cp = caption(guided, X_te, pl, tok)
            sc = score(cp, refs)
            out[name] = {"metrics": summarize(sc), "vs_baseline": paired(sc, sc_base), "captions": cp,
                         "_lr1": sc["lr1"], "_cider": sc["cider"]}
        return out

    res["seeds"] = {}
    for seed in args.seeds:
        print(f"\n===== seed {seed} =====", flush=True)
        R = {}
        # (a) text fine-tuning
        set_seed(seed); t1 = time.time()
        den = TextCFGDenoiser(D_MODEL, N_PATCHES, L_CAP, N_MP, CLIP_DIM).to(DEVICE)
        miss, unexp = den.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_denoiser.pt"), weights_only=True), strict=False)
        train_text_gpu(den, A0, pimg, X_all, t_train, EPOCHS_TEXT, lr=LR, tag=f"text s{seed}")
        torch.save(den.state_dict(), os.path.join(CKPT_DIR, f"text_ft_s{seed}.pt"))
        R["text_ft"] = {"train_sec": round(time.time() - t1, 1), "init_missing": list(miss), **eval_text_model(den)}
        print("text_ft", {k: v["metrics"] for k, v in R["text_ft"].items() if isinstance(v, dict) and "metrics" in v}, flush=True)

        # (b) distillation from this seed's text model
        set_seed(seed); t1 = time.time()
        pidx = torch.arange(N_DISTILL, device=DEVICE)
        train_text_gpu(den, A_or, pidx, X_all, t_train, EPOCHS_DIST, lr=LR * 0.5, log_every=5, tag=f"distill s{seed}")
        torch.save(den.state_dict(), os.path.join(CKPT_DIR, f"distill_text_s{seed}.pt"))
        R["distill_text"] = {"train_sec": round(time.time() - t1, 1), **eval_text_model(den)}
        print("distill", {k: v["metrics"] for k, v in R["distill_text"].items() if isinstance(v, dict) and "metrics" in v}, flush=True)
        del den

        # (b') image-only distillation from the shared baseline
        set_seed(seed); t1 = time.time()
        d2 = load_base_denoiser()
        train_img_distill_gpu(d2, A_or, pidx, X_all, EPOCHS_DIST, lr=LR * 0.5, tag=f"imgdistill s{seed}")
        torch.save(d2.state_dict(), os.path.join(CKPT_DIR, f"distill_img_s{seed}.pt"))
        cp = caption(guided, X_te, gen_plans(lambda Xb, sl: maskgit_cfg_vec(d2, Xb, 4, CFG_SCALE), X_te), tok)
        sc = score(cp, refs)
        R["distill_imgonly"] = {"train_sec": round(time.time() - t1, 1),
                                "cfg_image_only": {"metrics": summarize(sc), "vs_baseline": paired(sc, sc_base),
                                                   "captions": cp, "_lr1": sc["lr1"], "_cider": sc["cider"]}}
        print("distill_imgonly", R["distill_imgonly"]["cfg_image_only"]["metrics"], flush=True)
        del d2
        res["seeds"][str(seed)] = R

    # ── aggregate across seeds ──
    agg = {}
    for model in ("text_ft", "distill_text", "distill_imgonly"):
        for conf in res["seeds"][str(args.seeds[0])][model]:
            if conf == "train_sec" or conf == "init_missing":
                continue
            entry = {}
            for k, key in (("lr1", "_lr1"), ("cider", "_cider")):
                vals = [float(np.mean(res["seeds"][str(s)][model][conf][key])) for s in args.seeds]
                dl = [v - float(sc_base[k].mean()) for v in vals]
                seedmean_delta = np.mean([res["seeds"][str(s)][model][conf][key] for s in args.seeds], 0) - sc_base[k]
                m, lo, hi = boot_ci(seedmean_delta)
                entry[k] = {"per_seed": [round(v, 4) for v in vals], "mean": round(float(np.mean(vals)), 4),
                            "std": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0,
                            "delta_per_seed": [round(d, 4) for d in dl],
                            "delta_mean": round(float(np.mean(dl)), 4),
                            "delta_std": round(float(np.std(dl, ddof=1)), 4) if len(dl) > 1 else 0.0,
                            "seedavg_perimage_delta_ci95": [round(lo, 4), round(hi, 4)]}
            agg[f"{model}/{conf}"] = entry
    # distillation vs its own parent text model (same seed, same conf)
    for conf in ("beta0", "beta1_free", "beta1_ref0_LEAK"):
        per = []
        for s in args.seeds:
            d = np.asarray(res["seeds"][str(s)]["distill_text"][conf]["_lr1"]) - np.asarray(res["seeds"][str(s)]["text_ft"][conf]["_lr1"])
            m, lo, hi = boot_ci(d); per.append([round(m, 4), round(lo, 4), round(hi, 4)])
        agg[f"distill_text_minus_text_ft/{conf}/lr1_per_seed_[delta,lo,hi]"] = per
    res["aggregate"] = agg
    # strip private arrays
    for s in res["seeds"].values():
        for model in s.values():
            for conf in model.values():
                if isinstance(conf, dict):
                    conf.pop("_lr1", None); conf.pop("_cider", None)
    res["total_sec"] = round(time.time() - T0, 1)
    json.dump(res, open(os.path.join(OUT_DIR, "remedies_seeds.json" if "REV_SMOKE_EPOCHS" not in os.environ else "remedies_seeds_SMOKE.json"), "w"), indent=1)
    print(json.dumps(agg, indent=1))
    print("total sec", res["total_sec"])


if __name__ == "__main__":
    main()
