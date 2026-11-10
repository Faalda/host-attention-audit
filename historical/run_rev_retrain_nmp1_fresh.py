"""
run_rev_retrain_nmp1_fresh.py — one-off fresh retrain of the frac=1.00, n_mp=1 denoiser.

Context: the saved checkpoint results/revision/rev_ckpt/scaling_frac1.00_nmp1.pt is
corrupted (527,872/593,408 parameters are NaN on disk), which forced
run_rev_host_predictor_failure.py's known-headroom calibration to reject all 600 jobs
before the optimizer ran (see results/revision/host_predictor_failure_n200_s0_audit/).
The previously RECORDED recovery for this exact config in denoiser_scaling.json
(top1=0.0353, top1_content=0.0055, train_sec=522.3) shows training previously completed
and was evaluated successfully, so the corruption is most likely to the on-disk artifact,
not an inherent training instability -- but this script verifies that directly rather
than assuming it, by retraining with the identical recipe/seed and auditing every step
for nonfinite values.

This does NOT touch the original (corrupted) checkpoint, the original
denoiser_scaling.json, or any other file in the frac/nmp sweep. It writes a new
checkpoint under a new name and a standalone result record for comparison.
"""
import os

if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
    raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0; no other GPU is permitted")

import json
import time
from pathlib import Path

import torch

from run_rev_den_common import (
    D_MODEL, N_PATCHES, L_CAP, N_TV, TEST_START,
    CFGDenoiser, OUT_DIR, SAVE_DIR,
    set_seed, load_tok, load_caps, build_A0, project_all,
    load_models, gen_plans, caption, score, summarize, maskgit_cfg_vec, CFG_SCALE,
)
from run_rev_denoiser_scaling import recovery
from run_cap_improved import EPOCHS_DEN

SEED = 0
EPOCHS = EPOCHS_DEN
N_MP = 1
CKPT_OUT = os.path.join(OUT_DIR, "rev_ckpt", "scaling_frac1.00_nmp1_RETRAIN.pt")
RESULT_OUT = os.path.join(OUT_DIR, "retrain_nmp1_fresh.json")


def train_monitored(den, A0, pair_img, X, n_epochs, tag=""):
    """Identical recipe to run_rev_den_common.train_cfg_denoiser_gpu, with added
    nonfinite accounting (loss AND gradient norm, pre-step) instead of a silent
    `continue`-on-nan that never reports whether it fired."""
    import random
    import torch.nn.functional as F
    from run_rev_den_common import _epoch_batches
    from run_sum_masked import mask_attn
    from run_cap_improved import patch_salience, CFG_DROP_P, SAL_LAMBDA
    from run_rev_den_common import LR, BATCH

    opt = torch.optim.Adam(den.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    n_batches = n_skipped_loss = n_skipped_grad = n_steps = 0
    worst_grad_norm = 0.0
    for ep in range(1, n_epochs + 1):
        den.train(); tot = 0.0; n = 0; t0 = time.time()
        for idx in _epoch_batches(len(A0), BATCH):
            n_batches += 1
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
                n_skipped_loss += 1
                continue
            opt.zero_grad(); loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(den.parameters(), 1.0)
            gnorm = float(gnorm)
            if not (gnorm == gnorm) or gnorm in (float("inf"), float("-inf")):  # nan/inf check
                n_skipped_grad += 1
                opt.zero_grad()
                continue
            worst_grad_norm = max(worst_grad_norm, gnorm)
            opt.step(); n_steps += 1
            tot += loss.item(); n += 1
            bad = sum(int((~torch.isfinite(p)).sum()) for p in den.parameters())
            if bad:
                raise RuntimeError(f"Nonfinite parameter appeared mid-training at ep{ep}, "
                                    f"batch {n_batches}: {bad} elements")
        sch.step()
        print(f"    [{tag}] ep {ep:3d} cfg_den_loss={tot / max(n, 1):.4f} "
              f"({time.time() - t0:.1f}s/ep) skipped_loss={n_skipped_loss} "
              f"skipped_grad={n_skipped_grad}", flush=True)
    den.eval()
    return {"n_batches": n_batches, "n_steps": n_steps, "n_skipped_loss": n_skipped_loss,
            "n_skipped_grad": n_skipped_grad, "worst_grad_norm": worst_grad_norm}


def main():
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert not Path(CKPT_OUT).exists(), f"refuse to overwrite existing {CKPT_OUT}"
    DEVICE = "cuda:0"
    t_start = time.time()

    tok = load_tok()
    caps = load_caps()
    proj, free, guided = load_models(tok)
    X_all = project_all(proj)

    A0_tv, pimg_tv, _ = build_A0(free, X_all, caps, list(range(N_TV)), tok)

    set_seed(SEED)
    den = CFGDenoiser(D_MODEL, N_PATCHES, L_CAP, N_MP).to(DEVICE)
    n_params = sum(p.numel() for p in den.parameters())
    print(f"Training fresh frac1.00_nmp{N_MP} denoiser, {n_params} params, seed={SEED}, "
          f"{EPOCHS} epochs, {len(A0_tv)} training pairs", flush=True)

    t0 = time.time()
    train_audit = train_monitored(den, A0_tv, pimg_tv, X_all, EPOCHS, tag=f"frac1.00_nmp{N_MP}_retrain")
    train_sec = round(time.time() - t0, 1)

    bad_final = sum(int((~torch.isfinite(p)).sum()) for p in den.parameters())
    assert bad_final == 0, f"Final model has {bad_final} nonfinite parameters"
    print(f"Training finished in {train_sec}s; 0 nonfinite parameters in final model.", flush=True)

    torch.save(den.state_dict(), CKPT_OUT)
    # reload from disk to make sure what actually got written is clean (catches write corruption)
    reloaded = torch.load(CKPT_OUT, map_location="cpu", weights_only=True)
    reload_bad = sum(int((~torch.isfinite(v)).sum()) for v in reloaded.values())
    assert reload_bad == 0, f"Checkpoint on disk has {reload_bad} nonfinite elements after reload"

    # Evaluate with the exact same recipe as run_rev_denoiser_scaling.py's evaluate()
    N_IMAGES_TEST = 1000
    X_te = X_all[TEST_START:TEST_START + N_IMAGES_TEST]
    refs = caps[TEST_START:TEST_START + N_IMAGES_TEST]
    A0_te, pimg_te, ids_te = build_A0(free, X_all, caps, list(range(TEST_START, TEST_START + N_IMAGES_TEST)), tok)
    X_pairs_te = X_all[pimg_te]
    content = (ids_te[:, :-1] != 0)
    content = torch.cat([content, torch.zeros(len(content), 1, dtype=torch.bool, device=DEVICE)], 1)

    rec = recovery(den, A0_te, X_pairs_te, content)
    print("recovery:", rec, flush=True)

    cp = caption(guided, X_te, gen_plans(lambda Xb, sl: maskgit_cfg_vec(den, Xb, 4, CFG_SCALE), X_te), tok)
    sc = score(cp, refs)
    cap_metrics = summarize(sc)
    print("caption metrics:", cap_metrics, flush=True)

    result = {
        "config": {"frac": 1.00, "n_mp": N_MP, "seed": SEED, "epochs": EPOCHS,
                   "n_params": n_params, "n_train_pairs": int(len(A0_tv))},
        "purpose": "Fresh, independently-verified replacement for the corrupted "
                   "scaling_frac1.00_nmp1.pt, to unblock the known-headroom calibration "
                   "in run_rev_host_predictor_failure.py. Does not modify the original "
                   "sweep's checkpoint or denoiser_scaling.json.",
        "train_audit": train_audit,
        "train_sec": train_sec,
        "final_model_nonfinite_parameters": bad_final,
        "checkpoint_reload_nonfinite_elements": reload_bad,
        "checkpoint_path": CKPT_OUT,
        "recovery": rec,
        "caption_metrics": cap_metrics,
        "comparison_to_original_recorded_run": {
            "original_recovery_top1": 0.0353, "original_recovery_top1_content": 0.0055,
            "original_train_sec": 522.3,
            "note": "Compare manually; same seed/recipe/data so values should be close, "
                    "not bit-identical (GPU nondeterminism in attention/matmul ops is not "
                    "pinned here).",
        },
        "total_sec": round(time.time() - t_start, 1),
    }
    with open(RESULT_OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ("train_audit",)}, indent=2))


if __name__ == "__main__":
    main()
