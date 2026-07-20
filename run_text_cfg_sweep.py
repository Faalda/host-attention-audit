"""
run_text_cfg_sweep.py — Sweep text-CFG guidance scale for TextCFGDenoiser.

Runs inference at multiple scale_t values (text guidance strength) while
fixing scale_x=2.0.  Reports Top-1 / Top-5 recovery for each scale.
Shows whether the 0.168→0.269 gain saturates or keeps climbing.
Also reports the pre-fine-tuning image-only CFGDenoiser's ROUGE-1 on the
same images, as an honest "before text-conditioning fine-tune" baseline.

Usage:
    python3 run_text_cfg_sweep.py
    python3 run_text_cfg_sweep.py --n-eval 500
"""

import os, sys, json, argparse, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from run_summarization import Tokenizer, DEVICE, D_MODEL, N_HEADS, N_LAYERS, L_SUM as L_CAP, BATCH, N_MP
from run_captioning import PatchProjector, CaptionDecoder, CocoCapDataset, collate_cap, extract_A0_cap, N_CAP_PER
from run_cap_improved import (
    ENCODER, N_PATCHES, ENCODER_DIM,
    N_IMAGES, VAL_IMAGES, TEST_IMAGES,
    RESULTS_DIR, CACHE_PATH,
    SoftGuidedCaptionDecoder, CFGDenoiser, maskgit_cfg, CFG_SCALE,
)
from run_sum_masked import mask_attn, MASK_RATES
from run_text_denoiser import TextCFGDenoiser, CLIP_DIM, TEXT_DIR, encode_captions_clip, maskgit_text_cfg
from run_summarization import eval_rouge

SAVE_DIR = os.path.join(RESULTS_DIR, "saved_plans")

SCALE_X_VALUES = [2.0]                            # fix image CFG
SCALE_T_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]  # sweep text CFG
N_SAMPLES      = 512


@torch.no_grad()
def recovery_at_scale(denoiser, A0_all, X_all, t_embs, scale_t, scale_x=2.0,
                      n_samples=N_SAMPLES, device=DEVICE):
    idx  = torch.randperm(len(A0_all))[:n_samples]
    A0   = A0_all[idx].to(device)
    X    = X_all[idx].to(device)
    t    = t_embs[idx].to(device)

    rand1 = 1.0 / N_PATCHES
    rand5 = 5.0 / N_PATCHES

    # Use mask_rate=0.50 as representative point
    A_t, mask = mask_attn(A0, 0.50)
    logits = denoiser.cfg_predict_text(X, A_t, mask, t, scale_x=scale_x, scale_t=scale_t)
    tgt = A0.argmax(-1)
    m   = mask.bool()

    top1 = (logits.argmax(-1) == tgt)[m].float().mean().item()
    top5_i = logits.topk(5, -1).indices
    top5 = (top5_i == tgt.unsqueeze(-1).expand_as(top5_i)).any(-1)[m].float().mean().item()
    return top1, top5, top1 / rand1, top5 / rand5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-eval", type=int, default=100,
                        help="Number of test images for the ROUGE-1 sweep "
                             "and the image-only baseline")
    args = parser.parse_args()
    n_eval = args.n_eval

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    with open(os.path.join(SAVE_DIR, "tok_vocab.json")) as f:
        vocab = json.load(f)
    tok = Tokenizer()
    tok.vocab = vocab
    tok.w2i = {w: i for i, w in enumerate(vocab)}

    # ── Models ────────────────────────────────────────────────────────────────
    proj = PatchProjector(ENCODER_DIM, D_MODEL).to(DEVICE)
    proj.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_proj.pt"),
                                    map_location=DEVICE, weights_only=True))
    proj.eval()

    free_dec = CaptionDecoder(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS,
                              n_src=N_PATCHES).to(DEVICE)
    free_dec.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_free.pt"),
                                        map_location=DEVICE, weights_only=True))
    free_dec.eval()

    guided_dec = SoftGuidedCaptionDecoder(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)
    guided_dec.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_guided.pt"),
                                           map_location=DEVICE, weights_only=True))
    guided_dec.eval()

    # Pre-fine-tuning image-only CFGDenoiser — the honest "before text
    # conditioning" baseline, evaluated on the exact same images/method.
    image_only_denoiser = CFGDenoiser(D_MODEL, N_PATCHES, L_CAP, N_MP).to(DEVICE)
    image_only_denoiser.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, "model_denoiser.pt"),
                   map_location=DEVICE, weights_only=True))
    image_only_denoiser.eval()

    denoiser = TextCFGDenoiser(D_MODEL, N_PATCHES, L_CAP, N_MP, CLIP_DIM).to(DEVICE)
    ckpt = os.path.join(TEXT_DIR, "model_text_denoiser_v2.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(TEXT_DIR, "model_text_denoiser.pt")
    denoiser.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    print(f"  Loaded from {ckpt}")
    denoiser.eval()
    print("Models loaded.")

    # ── Test oracle A₀ + CLIP embeddings ─────────────────────────────────────
    print("Loading patch features...")
    patch_feats = torch.load(CACHE_PATH, weights_only=True)
    test_start  = N_IMAGES + VAL_IMAGES
    test_feats  = patch_feats[test_start: test_start + TEST_IMAGES]

    print("Loading MSCOCO test captions...")
    from datasets import load_dataset
    ds = load_dataset("clip-benchmark/wds_mscoco_captions", split="train")
    test_caps = []
    test_caps_flat = []
    for item in ds.select(range(test_start, test_start + TEST_IMAGES)):
        caps = [c.strip() for c in item["txt"].strip().split("\n") if c.strip()][:N_CAP_PER]
        test_caps.append(caps)
        test_caps_flat.append(caps[0])

    print("Extracting oracle A₀ (test)...")
    test_ds     = CocoCapDataset(test_feats, test_caps, tok, proj)
    test_loader = DataLoader(test_ds, BATCH, shuffle=False, collate_fn=collate_cap)
    A0_test, X_test, _, _ = extract_A0_cap(proj, free_dec, test_loader, DEVICE)
    print(f"  A0_test: {A0_test.shape}")

    print("Encoding test captions with CLIP...")
    text_emb_path = os.path.join(TEXT_DIR, "test_text_embs.pt")
    if os.path.exists(text_emb_path):
        t_embs = torch.load(text_emb_path, weights_only=True)
    else:
        t_embs = encode_captions_clip(test_caps_flat, device=DEVICE)
        torch.save(t_embs, text_emb_path)
    # Repeat to match A0_test (one per caption-image pair)
    repeat = len(A0_test) // len(t_embs)
    t_embs_rep = t_embs.repeat_interleave(repeat, dim=0)
    if len(t_embs_rep) < len(A0_test):
        t_embs_rep = torch.cat([t_embs_rep, t_embs[:len(A0_test)-len(t_embs_rep)]], dim=0)
    print(f"  text_embs: {t_embs.shape}")

    # ── Recovery sweep ────────────────────────────────────────────────────────
    print(f"\n--- Text-CFG scale sweep (mask_rate=0.50, n={N_SAMPLES}) ---")
    print(f"{'scale_t':>8}  {'Top-1':>8}  {'×rand':>8}  {'Top-5':>8}  {'×rand5':>8}")
    print("-" * 52)
    for scale_t in SCALE_T_VALUES:
        t1, t5, x1, x5 = recovery_at_scale(
            denoiser, A0_test, X_test, t_embs_rep,
            scale_t=scale_t, scale_x=2.0
        )
        print(f"{scale_t:8.1f}  {t1:8.3f}  {x1:8.1f}x  {t5:8.3f}  {x5:8.1f}x")

    # ── ROUGE sweep (caption quality at each scale) ────────────────────────────
    # Use one X per image: X_test has 5*TEST_IMAGES rows (one per caption-image pair),
    # stride by N_CAP_PER to get the canonical first-caption X for each image.
    X_test_unique = X_test[::N_CAP_PER][:TEST_IMAGES]  # [TEST_IMAGES, N, D]
    refs_1st = [test_caps[i][0] for i in range(n_eval)]

    # ── Image-only baseline (pre-fine-tuning CFGDenoiser, no text at all) ──────
    print(f"\n--- Image-only baseline (pre-fine-tuning), {n_eval} test images ---")
    hyps_baseline = []
    for i in range(0, n_eval, BATCH):
        X_b = X_test_unique[i:i+BATCH].to(DEVICE)
        plan = maskgit_cfg(image_only_denoiser, X_b, n_steps=4, cfg_scale=CFG_SCALE, device=DEVICE)
        toks = guided_dec.generate(X_b, plan)
        for row in toks:
            words = [tok.vocab[t] for t in row.tolist() if t > 2]
            hyps_baseline.append(" ".join(words))
    baseline_rouge = eval_rouge(hyps_baseline[:n_eval], refs_1st)
    print(f"  Image-only ROUGE-1: {baseline_rouge:.4f}")

    print(f"\n--- ROUGE-1 sweep over {n_eval} test images ---")
    print(f"{'scale_t':>8}  {'ROUGE-1':>8}  {'vs.baseline':>12}")
    oracle_rouge = 0.6858  # best-of-5-reference oracle-inversion ceiling, n=1000
    print("-" * 34)
    sweep_results = []
    for scale_t in SCALE_T_VALUES:
        hyps = []
        for i in range(0, n_eval, BATCH):
            X_b = X_test_unique[i:i+BATCH].to(DEVICE)
            t_b = t_embs[i:i+BATCH].to(DEVICE)   # one per image, aligned
            plan = maskgit_text_cfg(denoiser, X_b, t_b, n_steps=4,
                                    scale_x=2.0, scale_t=scale_t, device=DEVICE)
            toks = guided_dec.generate(X_b, plan)
            for row in toks:
                words = [tok.vocab[t] for t in row.tolist() if t > 2]
                hyps.append(" ".join(words))
        r1 = eval_rouge(hyps[:n_eval], refs_1st)
        vs_baseline = r1 - baseline_rouge
        print(f"{scale_t:8.1f}  {r1:8.4f}  {vs_baseline:12.4f}")
        sweep_results.append({"scale_t": scale_t, "rouge1": round(r1, 4),
                              "vs_baseline": round(vs_baseline, 4)})

    print("\n--- LaTeX rows ---")
    print("scale_t & Top-1 & Top-5 & ROUGE-1 \\\\")

    results_path = os.path.join(TEXT_DIR, f"cfg_sweep_results_n{n_eval}.json")
    with open(results_path, "w") as f:
        json.dump({
            "n_eval": n_eval,
            "image_only_baseline_rouge1": round(baseline_rouge, 4),
            "oracle_inversion_ceiling_rouge1": oracle_rouge,
            "sweep": sweep_results,
        }, f, indent=2)
    print(f"Results saved → {results_path}")


if __name__ == "__main__":
    main()
