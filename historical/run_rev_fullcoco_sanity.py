"""
run_rev_fullcoco_sanity.py — E9 pre-flight: does the new streaming/int8 pipeline reproduce the old one?
  1. new int8 features of train-split images 14000..14999 vs the old fp32 cache (same images).
  2. OLD checkpoints (results/saved_plans) evaluated with the NEW eval code on (a) old fp32 features,
     (b) new int8 features: best-of-5 legacy ROUGE-1 should be 0.4876 (paper / den_sanity.json);
     denoised plans vs saved plans_test.pt.
  3. calibration of the plan-concentration statistics against the paper text (11.4 rows, 1.5 distinct,
     90% of rows on the top patch) and of the uniform-plan decode (first 500: 0.499 / CIDEr 0.870).
  4. teacher-forced A0 via the streaming helper == run_rev_den_common.build_A0 on a batch.
Writes results/full_coco/sanity_old_ckpt.json.
"""
import sys
import os
from run_rev_fullcoco_eval import *
from run_rev_den_common import build_A0
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for clean_pipeline./inversion. imports
from clean_pipeline.run_cap_improved import CACHE_PATH

OLD = os.path.join(ROOT, "results", "saved_plans")
T0 = 14000


class FP32Feats:
    def __init__(self, x): self.x = x
    def get(self, idx): return self.x[idx]
    def __len__(self): return len(self.x)


def main():
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    out = {}
    old = torch.load(CACHE_PATH, weights_only=True, mmap=True)
    x_old = old[T0:T0 + 1000].float().to(DEVICE)
    new8 = GPUFeats("train", n=1000, start=T0)
    x_new = new8.get(torch.arange(1000, device=DEVICE))
    out["feat_rel_l2_err"] = round(float((x_new - x_old).norm() / x_old.norm()), 5)
    out["feat_max_abs_err"] = round(float((x_new - x_old).abs().max()), 4)
    print(out, flush=True)

    tok = tok_from_vocab(json.load(open(os.path.join(OLD, "tok_vocab.json"))))
    proj = PatchProjector(ENCODER_DIM, D_MODEL).to(DEVICE)
    free = CaptionDecoder(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS, n_src=N_PATCHES).to(DEVICE)
    den = CFGDenoiser(D_MODEL, N_PATCHES, L_CAP, N_MP).to(DEVICE)
    gdec = SoftGuidedCaptionDecoder(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)
    for n_, m in [("model_proj", proj), ("model_free", free), ("model_denoiser", den), ("model_guided", gdec)]:
        m.load_state_dict(torch.load(os.path.join(OLD, f"{n_}.pt"), map_location=DEVICE, weights_only=True))
    freeze(proj, free, den, gdec)
    refs = load_caps("train")[T0:T0 + 1000]
    saved = torch.load(os.path.join(OLD, "plans_test.pt"), weights_only=True).to(DEVICE)

    for name, fs in [("old_fp32", FP32Feats(x_old)), ("new_int8", new8)]:
        set_seed(0)
        res, caps, plans = evaluate_system(proj, free, den, gdec, fs, refs, tok)
        res["plans_vs_saved_argmax_agree"] = round(float((plans.argmax(-1) == saved.argmax(-1)).float().mean()), 4)
        res["plans_vs_saved_maxabs"] = float((plans - saved).abs().max())
        uni500 = score(caps["guided_uniform"][:500], refs[:500]); den500 = score(caps["guided_denoised"][:500], refs[:500])
        res["first500"] = {"uniform": {k: round(float(np.mean(uni500[k])), 4) for k in ("lr1", "cider")},
                           "denoised": {k: round(float(np.mean(den500[k])), 4) for k in ("lr1", "cider")}}
        out[name] = res
        out[name + "_captions_first5"] = {k: v[:5] for k, v in caps.items()}
        if name == "old_fp32":
            caps_old = caps
        else:
            out["caption_identical_frac_int8_vs_fp32"] = {k: round(float(np.mean([a == b for a, b in zip(caps[k], caps_old[k])])), 4) for k in caps}
        print(name, json.dumps(res, indent=1), flush=True)

    # 4. streaming teacher-forced A0 == build_A0 (fp32 features)
    fs = FP32Feats(x_old)
    Xall = torch.stack([proj(x_old[i:i + 100]) for i in range(0, 1000, 100)]).reshape(1000, N_PATCHES, D_MODEL)
    A_ref, pimg, pids = build_A0(free, Xall, refs, list(range(40)), tok)
    A_new, _ = teacher_forced_A0(proj, free, fs, pimg, pids)
    out["A0_stream_vs_build_maxabs"] = float((A_new - A_ref).abs().max())
    print("A0 diff", out["A0_stream_vs_build_maxabs"])
    json.dump(out, open(os.path.join(FC, "sanity_old_ckpt.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
