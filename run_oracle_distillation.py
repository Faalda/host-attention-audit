"""
run_oracle_distillation.py — Oracle-plan distillation.

Bridges plan inversion (C) and the forward denoiser (A) into one coherent
method: precompute oracle-inverted plans for a training subset, then
fine-tune the TextCFGDenoiser to predict those plans directly.

Pipeline:
  1. Invert A₀* for N_DISTILL images toward their first reference caption.
  2. Fine-tune the TextCFGDenoiser on (X, text, A₀*) instead of free-decoder A₀.
  3. Evaluate prediction gap = oracle_ROUGE − forward_ROUGE before and after.

Usage:
    python3 run_oracle_distillation.py
    python3 run_oracle_distillation.py --skip-precompute --skip-finetune --n-eval 100
"""

import os, sys, json, random, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from run_summarization import Tokenizer, DEVICE, D_MODEL, N_HEADS, N_LAYERS, L_SUM as L_CAP, BATCH, LR, N_MP
from run_captioning import PatchProjector, CaptionDecoder, N_CAP_PER
from run_cap_improved import (
    ENCODER, N_PATCHES, ENCODER_DIM, N_IMAGES, VAL_IMAGES, TEST_IMAGES,
    RESULTS_DIR, CACHE_PATH, SoftGuidedCaptionDecoder, CFG_DROP_P,
)
from run_sum_masked import mask_attn, MASK_RATES
from run_plan_inversion import invert_plan, tokenize_caption
from run_text_denoiser import TextCFGDenoiser, TextA0Dataset, CLIP_DIM, TEXT_DROP_P, TEXT_DIR
from run_summarization import eval_rouge

SAVE_DIR  = os.path.join(RESULTS_DIR, "saved_plans")
DIST_DIR  = os.path.join(RESULTS_DIR, "oracle_distillation")
os.makedirs(DIST_DIR, exist_ok=True)

TEST_START = N_IMAGES + VAL_IMAGES


def precompute_oracle_plans(proj, guided_dec, tok, patch_feats, caps_list,
                            n_images, n_inv_steps=100, inv_lr=0.05, device=DEVICE):
    cache_path = os.path.join(DIST_DIR, f"oracle_plans_train_{n_images}.pt")
    if os.path.exists(cache_path):
        print(f"  Loading cached oracle plans from {cache_path}")
        data = torch.load(cache_path, weights_only=True)
        return data["A0"], data["X"]

    proj.eval()
    guided_dec.eval()
    A0_list, X_list = [], []

    for i in range(n_images):
        X = proj(patch_feats[i].unsqueeze(0).to(device)).detach()
        tgt_toks = tokenize_caption(tok, caps_list[i][0])
        A0_oracle = invert_plan(guided_dec, X, tgt_toks, A_init=None,
                                n_steps=n_inv_steps, lr=inv_lr, verbose=False)
        A0_list.append(A0_oracle[0].cpu())
        X_list.append(X[0].cpu())
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{n_images}] oracle plans precomputed")

    A0_all = torch.stack(A0_list)
    X_all  = torch.stack(X_list)
    torch.save({"A0": A0_all, "X": X_all}, cache_path)
    print(f"  Saved oracle plans → {cache_path}")
    return A0_all, X_all


def finetune_distilled(denoiser, A0_oracle, X_all, text_embs, n_epochs=30, device=DEVICE):
    repeat = len(A0_oracle) // len(text_embs) if len(text_embs) < len(A0_oracle) else 1
    if repeat > 1:
        text_embs = text_embs[:len(A0_oracle) // repeat].repeat_interleave(repeat, dim=0)
    text_embs = text_embs[:len(A0_oracle)]

    ds     = TextA0Dataset(A0_oracle, X_all, text_embs)
    loader = DataLoader(ds, BATCH, shuffle=True, num_workers=2, pin_memory=True)
    opt    = torch.optim.Adam(denoiser.parameters(), lr=LR * 0.5)
    sch    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)

    for ep in range(1, n_epochs + 1):
        denoiser.train()
        tot = 0; n = 0
        for A0, X, t in loader:
            A0 = A0.to(device); X = X.to(device); t = t.to(device)
            mr   = random.uniform(0.1, 0.95)
            A_t, mask = mask_attn(A0, mr)
            drop_X = (random.random() < CFG_DROP_P)
            drop_t = (random.random() < TEXT_DROP_P)
            logits = denoiser(X, A_t, mask, t_vec=t, drop_X=drop_X, drop_t=drop_t)
            tgt = A0.argmax(-1)
            m   = mask.bool()
            if not m.any(): continue
            loss = F.cross_entropy(logits[m], tgt[m])
            if torch.isnan(loss) or torch.isinf(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1
        sch.step()
        if ep % 5 == 0:
            avg = tot / n if n else float("nan")
            print(f"    ep {ep:3d}  distill_loss={avg:.4f}")


def _generate_captions(denoiser, guided_dec, proj, patch_feats, text_embs,
                       start, n, tok, device):
    from run_text_denoiser import maskgit_text_cfg
    hyps = []
    for i in range(start, start + n):
        X = proj(patch_feats[i].unsqueeze(0).to(device))
        t = text_embs[i - start].unsqueeze(0).to(device) if text_embs is not None else None
        if t is not None:
            plan = maskgit_text_cfg(denoiser, X, t, n_steps=4, device=device)
        else:
            from run_cap_improved import maskgit_cfg, CFG_SCALE
            plan = maskgit_cfg(denoiser, X, n_steps=4, cfg_scale=CFG_SCALE, device=device)
        with torch.no_grad():
            toks = guided_dec.generate(X, plan)
        words = []
        for t_id in toks[0].tolist():
            if t_id == 2: break
            if t_id > 2: words.append(tok.vocab[t_id])
        hyps.append(" ".join(words))
    return hyps


def eval_prediction_gap(denoiser, guided_dec, proj, patch_feats, test_caps,
                        tok, device, n_images=50, text_embs=None):
    denoiser.eval()
    guided_dec.eval()
    proj.eval()

    forward_hyps, oracle_hyps, refs = [], [], []

    for i in range(n_images):
        ref = test_caps[i][0]
        refs.append(ref)

        # Forward: denoiser plan
        X = proj(patch_feats[TEST_START + i].unsqueeze(0).to(device))
        t_vec = text_embs[i].unsqueeze(0).to(device) if text_embs is not None else None
        if t_vec is not None:
            from run_text_denoiser import maskgit_text_cfg
            plan_fwd = maskgit_text_cfg(denoiser, X, t_vec, n_steps=4, device=device)
        else:
            from run_cap_improved import maskgit_cfg, CFG_SCALE
            plan_fwd = maskgit_cfg(denoiser, X, n_steps=4, cfg_scale=CFG_SCALE)
        with torch.no_grad():
            toks_fwd = guided_dec.generate(X.detach(), plan_fwd)
        fwd_cap = tok.dec(toks_fwd[0].tolist())
        forward_hyps.append(fwd_cap)

        # Oracle: inverted plan
        tgt_toks = tokenize_caption(tok, ref)
        X_det = X.detach()
        A_inv = invert_plan(guided_dec, X_det, tgt_toks, A_init=None,
                            n_steps=100, lr=0.05, verbose=False)
        with torch.no_grad():
            toks_inv = guided_dec.generate(X_det, A_inv)
        oracle_hyps.append(tok.dec(toks_inv[0].tolist()))

        if (i + 1) % 10 == 0:
            r_fwd = eval_rouge(forward_hyps, refs)
            r_ora = eval_rouge(oracle_hyps, refs)
            print(f"  [{i+1}/{n_images}]  forward={r_fwd:.4f}  oracle={r_ora:.4f}  gap={r_ora-r_fwd:+.4f}")

    r_fwd = eval_rouge(forward_hyps, refs)
    r_ora = eval_rouge(oracle_hyps, refs)
    return {"forward_rouge": r_fwd, "oracle_rouge": r_ora, "gap": r_ora - r_fwd}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-distill",       type=int,   default=2000)
    parser.add_argument("--n-inv-steps",     type=int,   default=100)
    parser.add_argument("--n-epochs",        type=int,   default=30)
    parser.add_argument("--n-eval",          type=int,   default=50)
    parser.add_argument("--skip-precompute", action="store_true")
    parser.add_argument("--skip-finetune",   action="store_true")
    args = parser.parse_args()

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    with open(os.path.join(SAVE_DIR, "tok_vocab.json")) as f:
        vocab = json.load(f)
    tok = Tokenizer()
    tok.vocab = vocab
    tok.w2i = {w: i for i, w in enumerate(vocab)}
    print(f"Vocab: {len(tok.vocab)}")

    # ── Models ─────────────────────────────────────────────────────────────────
    proj = PatchProjector(ENCODER_DIM, D_MODEL).to(DEVICE)
    proj.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_proj.pt"),
                                    map_location=DEVICE, weights_only=True))
    proj.eval()

    guided_dec = SoftGuidedCaptionDecoder(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)
    guided_dec.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_guided.pt"),
                                           map_location=DEVICE, weights_only=True))
    guided_dec.eval()

    denoiser = TextCFGDenoiser(D_MODEL, N_PATCHES, L_CAP, N_MP, CLIP_DIM).to(DEVICE)
    text_den_path = os.path.join(TEXT_DIR, "model_text_denoiser.pt")
    denoiser.load_state_dict(torch.load(text_den_path, map_location=DEVICE, weights_only=True))
    print("Models loaded.")

    # ── Patch features ─────────────────────────────────────────────────────────
    patch_feats = torch.load(CACHE_PATH, weights_only=True)
    print(f"Patch features: {patch_feats.shape}")

    # ── CLIP text embeddings ───────────────────────────────────────────────────
    text_embs_path = os.path.join(TEXT_DIR, "train_text_embs.pt")
    text_embs = torch.load(text_embs_path, weights_only=True)   # [N_IMAGES, 512]
    print(f"CLIP text embs: {text_embs.shape}")

    # ── Train captions ─────────────────────────────────────────────────────────
    print("Loading MSCOCO captions...")
    from datasets import load_dataset
    ds_hf = load_dataset("clip-benchmark/wds_mscoco_captions", split="train")
    train_caps = []
    for item in ds_hf.select(range(args.n_distill)):
        caps = [c.strip() for c in item["txt"].strip().split("\n") if c.strip()]
        train_caps.append(caps[:N_CAP_PER])
    test_caps = []
    for item in ds_hf.select(range(TEST_START, TEST_START + args.n_eval)):
        caps = [c.strip() for c in item["txt"].strip().split("\n") if c.strip()]
        test_caps.append(caps[:N_CAP_PER])
    print(f"  {len(train_caps)} train, {len(test_caps)} test captions")

    # ── Baseline gap (before distillation) ────────────────────────────────────
    print("\n--- Prediction gap BEFORE distillation ---")
    test_text_embs = None
    test_text_emb_path = os.path.join(DIST_DIR, f"test_text_embs_{args.n_eval}.pt")
    if os.path.exists(test_text_emb_path):
        test_text_embs = torch.load(test_text_emb_path, weights_only=True)
    else:
        from run_text_denoiser import encode_captions_clip
        test_caps_flat = [c[0] for c in test_caps]
        test_text_embs = encode_captions_clip(test_caps_flat, device=DEVICE)
        torch.save(test_text_embs, test_text_emb_path)
    before = eval_prediction_gap(denoiser, guided_dec, proj, patch_feats,
                                  test_caps, tok, DEVICE,
                                  n_images=args.n_eval, text_embs=test_text_embs)
    print(f"\nBefore distillation:")
    print(f"  Forward ROUGE-1 : {before['forward_rouge']:.4f}")
    print(f"  Oracle ROUGE-1  : {before['oracle_rouge']:.4f}")
    print(f"  Prediction gap  : {before['gap']:+.4f}")

    # ── Precompute oracle plans ────────────────────────────────────────────────
    print(f"\n--- Precomputing oracle plans (n={args.n_distill}) ---")
    A0_oracle, X_oracle = precompute_oracle_plans(
        proj, guided_dec, tok, patch_feats[:args.n_distill], train_caps,
        n_images=args.n_distill, n_inv_steps=args.n_inv_steps, device=DEVICE,
    )
    print(f"  A0_oracle: {A0_oracle.shape},  X_oracle: {X_oracle.shape}")

    # ── Fine-tune ──────────────────────────────────────────────────────────────
    dist_ckpt = os.path.join(DIST_DIR, "model_distilled_denoiser.pt")
    if args.skip_finetune and os.path.exists(dist_ckpt):
        denoiser.load_state_dict(torch.load(dist_ckpt, map_location=DEVICE, weights_only=True))
        print(f"Loaded distilled denoiser from {dist_ckpt}")
    else:
        print(f"\n--- Fine-tuning on oracle plans ({args.n_epochs} epochs) ---")
        finetune_distilled(denoiser, A0_oracle, X_oracle,
                           text_embs[:args.n_distill],
                           n_epochs=args.n_epochs, device=DEVICE)
        torch.save(denoiser.state_dict(), dist_ckpt)
        print(f"Saved → {dist_ckpt}")

    # ── After-distillation gap ─────────────────────────────────────────────────
    print("\n--- Prediction gap AFTER distillation ---")
    after = eval_prediction_gap(denoiser, guided_dec, proj, patch_feats,
                                 test_caps, tok, DEVICE,
                                 n_images=args.n_eval, text_embs=test_text_embs)
    print(f"\nAfter distillation:")
    print(f"  Forward ROUGE-1 : {after['forward_rouge']:.4f}")
    print(f"  Oracle ROUGE-1  : {after['oracle_rouge']:.4f}")
    print(f"  Prediction gap  : {after['gap']:+.4f}")

    print(f"\n=== Gap reduction: {before['gap']:+.4f} → {after['gap']:+.4f}  "
          f"(Δgap = {after['gap'] - before['gap']:+.4f}) ===")

    results_path = os.path.join(DIST_DIR, f"distillation_results_n{args.n_eval}.json")
    with open(results_path, "w") as f:
        json.dump({
            "n_eval": args.n_eval,
            "n_distill": args.n_distill,
            "before": {k: round(v, 4) for k, v in before.items()},
            "after": {k: round(v, 4) for k, v in after.items()},
        }, f, indent=2)
    print(f"Results saved → {results_path}")


if __name__ == "__main__":
    main()
