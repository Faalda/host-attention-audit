"""
run_rev_den_sanity.py — pre-flight checks for E7/E4 (writes results/revision/den_sanity.json).
  1. vectorised maskgit == original maskgit_cfg (max abs diff).
  2. baseline model_denoiser.pt, K=4, CFG 2.0, softmax(10A), greedy -> best-of-5 legacy ROUGE-1
     on 1000 test images; compare with plans_test.pt and the paper's 0.4876.
  3. reproduce the old 0.4561 (first 500) setting of run_text_cfg_sweep.py to explain it.
  4. time one denoiser training epoch.
"""
import time, json
from run_rev_den_common import *


def main():
    set_seed(0)
    tok = load_tok(); caps = load_caps()
    proj, free, guided = load_models(tok)
    den = load_base_denoiser()
    X_te = project_all(proj)[TEST_START:TEST_START + TEST_IMAGES].contiguous()
    torch.cuda.empty_cache()
    refs = caps[TEST_START:TEST_START + TEST_IMAGES]
    assert all(len(r) == 5 for r in refs)
    out = {}

    # 1. maskgit equivalence (original loops, batch 32 like the original scripts)
    diffs = [verify_maskgit(den, X_te[s:s + 32], n=32) for s in (0, 32, 500)]
    out["maskgit_vec_vs_orig_maxabs"] = max(diffs)
    print("maskgit max abs diff:", diffs, flush=True)

    # 2. baseline reproduction
    plans = gen_plans(lambda Xb, sl: maskgit_cfg_vec(den, Xb, 4, CFG_SCALE), X_te)
    saved = torch.load(os.path.join(SAVE_DIR, "plans_test.pt"), weights_only=True).to(DEVICE)
    out["plans_vs_saved_maxabs"] = float((plans - saved).abs().max())
    out["plans_argmax_agree"] = float((plans.argmax(-1) == saved.argmax(-1)).float().mean())
    caps_new = caption(guided, X_te, plans, tok)
    caps_saved = caption(guided, X_te, saved, tok)
    out["captions_identical_frac"] = float(np.mean([a == b for a, b in zip(caps_new, caps_saved)]))
    sc_new = score(caps_new, refs); sc_saved = score(caps_saved, refs)
    out["baseline_new"] = summarize(sc_new); out["baseline_saved_plans"] = summarize(sc_saved)
    out["baseline_new_first500_lr1"] = round(float(sc_new["lr1"][:500].mean()), 4)
    inv = json.load(open(os.path.join(SAVE_DIR, "inversion_scores_n1000.json")))
    out["paper_baseline_mean"] = inv["baseline_mean"]
    print(json.dumps(out, indent=1), flush=True)

    # 3. old run_text_cfg_sweep.py image-only setting: plan NOT sharpened; decode keeps post-EOS tokens
    def dec_noeos(r): return " ".join(tok.vocab[t] for t in r if t > 2)
    old = {}
    for sharpen in (False, True):
        for eos_stop in (False, True):
            hyps = []
            for s in range(0, 500, BATCH):
                Xb = X_te[s:s + BATCH]
                p = maskgit_cfg_vec(den, Xb, 4, CFG_SCALE)
                if sharpen:
                    p = F.softmax(p * 10.0, dim=-1)
                t = guided.generate(Xb, p).cpu().tolist()
                hyps += [tok.dec(r) if eos_stop else dec_noeos(r) for r in t]
            old[f"sharpen={sharpen},eos_stop={eos_stop}"] = round(float(np.mean([lr1(h, r) for h, r in zip(hyps, refs[:500])])), 4)
    out["old_sweep_settings_first500_lr1"] = old
    print(json.dumps(old, indent=1), flush=True)

    # 4. timing of one epoch of train_cfg_denoiser_gpu on 20% of train+val
    X_all = project_all(proj, N_TV)
    A0, pimg, _ = build_A0(free, X_all, caps, list(range(N_TV)), tok)
    out["A0_pairs"] = len(A0)
    d2 = CFGDenoiser(D_MODEL, N_PATCHES, L_CAP, N_MP).to(DEVICE)
    sub = torch.arange(0, len(A0) // 5, device=DEVICE)
    t0 = time.time()
    train_cfg_denoiser_gpu(d2, A0[sub], pimg[sub], X_all, 1, tag="timing")
    torch.cuda.synchronize()
    out["sec_per_epoch_full_tv_est"] = round((time.time() - t0) * 5, 1)
    print("est sec/epoch (70k pairs):", out["sec_per_epoch_full_tv_est"])
    json.dump(out, open(os.path.join(OUT_DIR, "den_sanity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
