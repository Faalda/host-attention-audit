"""
run_rev_matched_effect_control.py — pre-registered control, Table "tab:open" row 1:
"Hidden-state bias early-stopped to the free plan's target fit."

Reuses the exact 901 counterfactual swap-injection jobs from
swap_injection_n1000_s0.json so every number is directly comparable to the
paper's existing free/onehot/hidden-at-300-steps rows.

Question: is the free plan's near-zero (0.2%) counterfactual-object injection
rate because routing through a simplex-constrained channel is intrinsically
grounded (account A), or simply because the free plan's optimizer reaches a
much smaller realized effect size (target-fit) than the hidden-state bias at
the same step count, i.e. selectivity is a function of effect size alone
(account B)?

Method: calibrate the number of inversion steps at which the HIDDEN-STATE
bias reaches the SAME mean target-fit (legacy ROUGE-1 vs the edited target)
that the free plan reaches at its usual 300 steps, then measure the
hidden-state bias's counterfactual-injection rate at that matched step count.
  - If injection stays far above the free plan's 0.2% even at matched fit:
    account (A) -- the channel's constraint, not the amount it was pushed,
    is what's doing the grounding.
  - If injection falls toward the free plan's rate once fit is matched:
    account (B) -- selectivity was a function of effect size alone.

Calibration subset: first 150 jobs (cheap sweep). Full measurement: all 901.
"""
import os, sys, json
import torch

sys.path.insert(0, os.path.dirname(__file__))
from run_rev_inversion_protocol import (
    Tokenizer, DEVICE, D_MODEL, N_HEADS, N_LAYERS, L_CAP, PatchProjector,
    ENCODER_DIM, CACHE_PATH, SoftGuidedCaptionDecoder, SAVE_DIR, OUT_DIR,
    TEST_START, invert,
)
from run_rev_eval_metrics import objects, lr1, boot_ci

CALIB_STEPS = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300]
CALIB_N = 150


def load_models():
    vocab = json.load(open(os.path.join(SAVE_DIR, "tok_vocab.json")))
    tok = Tokenizer(); tok.vocab = vocab; tok.w2i = {w: i for i, w in enumerate(vocab)}
    proj = PatchProjector(ENCODER_DIM, D_MODEL).to(DEVICE)
    proj.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_proj.pt"), map_location=DEVICE, weights_only=True))
    dec = SoftGuidedCaptionDecoder(len(vocab), D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)
    dec.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_guided.pt"), map_location=DEVICE, weights_only=True))
    proj.eval(); dec.eval()
    for p in list(proj.parameters()) + list(dec.parameters()):
        p.requires_grad_(False)
    return tok, proj, dec


def main():
    tok, proj, dec = load_models()
    feats = torch.load(CACHE_PATH, weights_only=True)
    plans = torch.load(os.path.join(SAVE_DIR, "plans_test.pt"), weights_only=True)
    swap = json.load(open(os.path.join(OUT_DIR, "swap_injection_n1000_s0.json")))
    recs = swap["records"]  # each: image, k, target, fake, free, onehot, hidden (at 300 steps)
    print(f"loaded {len(recs)} swap jobs from swap_injection_n1000_s0.json", flush=True)

    # Free plan's target-fit at its usual 300 steps (already computed captions, no rerun needed).
    free_fit_full = [lr1(r["free"], [r["target"]]) for r in recs]
    free_fit_mean = sum(free_fit_full) / len(free_fit_full)
    print(f"free plan mean fit-to-target (300 steps, n={len(recs)}): {free_fit_mean:.4f}", flush=True)

    def run_hidden(ch_recs, n_steps):
        idx = [r["image"] for r in ch_recs]
        X = proj(feats[[TEST_START + i for i in idx]].to(DEVICE))
        A = plans[idx].to(DEVICE)
        tgt = torch.tensor([tok.enc(r["target"], max_len=L_CAP, bos=True, eos=True) for r in ch_recs], device=DEVICE)
        _, _, toks = invert(dec, X, A, tgt, "hidden", n_steps, 0.05)
        return [tok.dec(t) for t in toks.cpu().tolist()]

    # --- Calibration: find the step count whose mean hidden-state fit-to-target
    # is closest to the free plan's mean fit-to-target, on a cheap subset. ---
    calib_recs = recs[:CALIB_N]
    calib_target_mean = sum(lr1(r["free"], [r["target"]]) for r in calib_recs) / len(calib_recs)
    print(f"calibration subset (n={CALIB_N}) free fit-to-target mean: {calib_target_mean:.4f}", flush=True)

    calib_results = []
    for n_steps in CALIB_STEPS:
        caps = run_hidden(calib_recs, n_steps)
        fit = [lr1(c, [r["target"]]) for c, r in zip(caps, calib_recs)]
        fit_mean = sum(fit) / len(fit)
        calib_results.append({"steps": n_steps, "fit_to_target_mean": round(fit_mean, 4)})
        print(f"  calib steps={n_steps:4d}  hidden fit-to-target={fit_mean:.4f}", flush=True)

    best = min(calib_results, key=lambda r: abs(r["fit_to_target_mean"] - calib_target_mean))
    matched_steps = best["steps"]
    print(f"MATCHED STEP COUNT: {matched_steps} (hidden fit {best['fit_to_target_mean']:.4f} "
          f"vs free-plan target {calib_target_mean:.4f})", flush=True)

    # --- Full measurement at the matched step count, all 901 jobs. ---
    all_caps_matched = []
    for s in range(0, len(recs), 300):
        ch = recs[s:s + 300]
        all_caps_matched += run_hidden(ch, matched_steps)
        print(f"  matched-run progress {min(s+300, len(recs))}/{len(recs)}", flush=True)

    def rates(caps, recs_):
        inj = [float(r["fake"] in objects(c) or r["fake"] in c.lower().split()) for c, r in zip(caps, recs_)]
        fit_target = [lr1(c, [r["target"]]) for c, r in zip(caps, recs_)]
        return inj, fit_target

    free_inj = [float(r["fake"] in objects(r["free"]) or r["fake"] in r["free"].lower().split()) for r in recs]
    hidden300_inj = [float(r["fake"] in objects(r["hidden"]) or r["fake"] in r["hidden"].lower().split()) for r in recs]
    matched_inj, matched_fit = rates(all_caps_matched, recs)

    out = {
        "n": len(recs),
        "calibration": {"subset_n": CALIB_N, "free_plan_target_fit": round(calib_target_mean, 4),
                        "sweep": calib_results, "matched_steps": matched_steps},
        "free_300steps": {"injection": round(sum(free_inj) / len(free_inj), 4),
                           "fit_to_target": round(free_fit_mean, 4)},
        "hidden_300steps": {"injection": round(sum(hidden300_inj) / len(hidden300_inj), 4)},
        f"hidden_{matched_steps}steps_matched": {
            "injection": round(sum(matched_inj) / len(matched_inj), 4),
            "injection_vs_free300_ci": [round(x, 4) for x in boot_ci([a - b for a, b in zip(matched_inj, free_inj)])],
            "injection_vs_hidden300_ci": [round(x, 4) for x in boot_ci([a - b for a, b in zip(matched_inj, hidden300_inj)])],
            "fit_to_target": round(sum(matched_fit) / len(matched_fit), 4),
        },
        "examples": [{"target": r["target"], "fake": r["fake"], "free": r["free"],
                      "hidden_300": r["hidden"], f"hidden_{matched_steps}_matched": c}
                     for r, c in list(zip(recs, all_caps_matched))[:20]],
    }
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=2))

    print("", flush=True)
    print("VERDICT:", flush=True)
    matched_inj_rate = out[f"hidden_{matched_steps}steps_matched"]["injection"]
    free_rate = out["free_300steps"]["injection"]
    hidden_rate = out["hidden_300steps"]["injection"]
    if abs(matched_inj_rate - free_rate) < abs(matched_inj_rate - hidden_rate):
        print(f"  Matched-effect injection ({matched_inj_rate:.4f}) sits closer to free plan's rate "
              f"({free_rate:.4f}) than to full hidden-state's ({hidden_rate:.4f})"
              f" -> supports account (B): selectivity tracks effect size.")
    else:
        print(f"  Matched-effect injection ({matched_inj_rate:.4f}) stays far closer to full hidden-state's "
              f"rate ({hidden_rate:.4f}) than to the free plan's ({free_rate:.4f}) even at matched fit"
              f" -> supports account (A): the channel's constraint itself is doing the grounding.")

    with open(os.path.join(OUT_DIR, "matched_effect_control.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved results/revision/matched_effect_control.json")


if __name__ == "__main__":
    main()
