"""
run_plan_inversion.py — Option C: gradient-based plan inversion.

Given image X and a target caption string, finds A₀* = argmin_{A₀ ∈ Δ^N^L}
-log p(target | X, A₀) by back-propagating through the guided decoder.
A₀ is parameterized as softmax(logits) to stay on the simplex.

Usage:
    python3 run_plan_inversion.py
    python3 run_plan_inversion.py --eval
    python3 run_plan_inversion.py --n-steps 500 --lr 0.02
"""

import os, sys, json, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from run_summarization import Tokenizer, DEVICE, D_MODEL, N_HEADS, N_LAYERS, L_SUM as L_CAP, N_MP
from run_captioning import PatchProjector, N_CAP_PER
from run_cap_improved import (
    ENCODER, N_PATCHES, ENCODER_DIM, N_IMAGES, VAL_IMAGES, TEST_IMAGES,
    RESULTS_DIR, CACHE_PATH,
    SoftGuidedCaptionDecoder,
)
from run_summarization import eval_rouge

SAVE_DIR   = os.path.join(RESULTS_DIR, "saved_plans")
TEST_START = N_IMAGES + VAL_IMAGES


def tokenize_caption(tok, caption, max_len=L_CAP):
    ids = tok.enc(caption, max_len=max_len, bos=True, eos=True)
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)   # [1, L]


def invert_plan(guided_dec, X, target_toks, A_init=None, n_steps=300, lr=0.05, verbose=False):
    guided_dec.eval()
    X = X.detach()   # only optimise logits, not upstream proj weights
    B = X.shape[0]
    if A_init is not None:
        init = torch.log(A_init.clamp(min=1e-8))
    else:
        init = torch.zeros(B, L_CAP, N_PATCHES, device=X.device)

    logits = nn.Parameter(init.clone())
    opt = torch.optim.Adam([logits], lr=lr)

    tgt_in  = target_toks[:, :-1].to(X.device)   # [B, L-1]
    tgt_out = target_toks[:, 1:].to(X.device)     # [B, L-1]
    pad_mask = (tgt_out != 0)

    for step in range(1, n_steps + 1):
        plan = F.softmax(logits, dim=-1)           # [B, L, N]
        pred = guided_dec(X, tgt_in, plan[:, :tgt_in.shape[1], :])  # [B, L-1, V]
        loss = F.cross_entropy(
            pred[pad_mask], tgt_out[pad_mask], reduction='mean'
        )
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and step % 50 == 0:
            print(f"    step {step:4d}  loss={loss.item():.4f}")

    return F.softmax(logits.detach(), dim=-1)


def caption_from_plan(guided_dec, X, plan, tok):
    guided_dec.eval()
    with torch.no_grad():
        toks = guided_dec.generate(X, plan)
    return tok.dec(toks[0].tolist())


def plan_diff_summary(A_orig, A_edited):
    orig_argmax   = A_orig.argmax(-1)    # [B, L]
    edited_argmax = A_edited.argmax(-1)
    changed = (orig_argmax != edited_argmax).sum().item()
    total   = orig_argmax.numel()
    return {"changed_slots": changed, "total_slots": total, "frac": changed / total}


def run_editing_demo(guided_dec, proj, patch_feats, plans_test, tok, examples, n_steps, lr):
    for img_idx, target_str in examples:
        X_raw = patch_feats[TEST_START + img_idx].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            X = proj(X_raw)

        A_init  = plans_test[img_idx].unsqueeze(0).to(DEVICE)   # denoised plan
        cap_orig = caption_from_plan(guided_dec, X, A_init, tok)

        tgt_toks = tokenize_caption(tok, target_str)
        A_edited = invert_plan(guided_dec, X, tgt_toks, A_init=A_init,
                               n_steps=n_steps, lr=lr, verbose=True)
        cap_edited = caption_from_plan(guided_dec, X, A_edited, tok)

        diff = plan_diff_summary(A_init, A_edited)
        print(f"\n[img {img_idx}]")
        print(f"  target  : {target_str}")
        print(f"  original: {cap_orig}")
        print(f"  inverted: {cap_edited}")
        print(f"  plan diff: {diff['changed_slots']}/{diff['total_slots']} slots changed ({diff['frac']:.1%})")


def run_eval(guided_dec, proj, patch_feats, plans_test, tok, n_images=100, n_steps=300, lr=0.05):
    from datasets import load_dataset
    print(f"Loading MSCOCO test captions...")
    ds = load_dataset("clip-benchmark/wds_mscoco_captions", split="train")

    baseline_scores, invert_scores = [], []

    for img_idx in range(n_images):
        item = ds[TEST_START + img_idx]
        refs = [c.strip() for c in item["txt"].strip().split("\n") if c.strip()][:N_CAP_PER]

        X_raw = patch_feats[TEST_START + img_idx].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            X = proj(X_raw)

        A_base = plans_test[img_idx].unsqueeze(0).to(DEVICE)
        cap_base = caption_from_plan(guided_dec, X, A_base, tok)
        base_r = max(eval_rouge([cap_base], [r]) for r in refs)
        baseline_scores.append(base_r)

        best_r = base_r
        for ref_cap in refs:
            tgt_toks = tokenize_caption(tok, ref_cap)
            A_inv = invert_plan(guided_dec, X, tgt_toks, A_init=A_base,
                                n_steps=n_steps, lr=lr, verbose=False)
            cap_inv = caption_from_plan(guided_dec, X, A_inv, tok)
            r = eval_rouge([cap_inv], [ref_cap])
            if r > best_r:
                best_r = r
        invert_scores.append(best_r)

        if (img_idx + 1) % 10 == 0:
            b_mean = sum(baseline_scores) / len(baseline_scores)
            i_mean = sum(invert_scores) / len(invert_scores)
            print(f"  [{img_idx+1}/{n_images}]  baseline={b_mean:.4f}  inverted={i_mean:.4f}")

    b_mean = sum(baseline_scores) / len(baseline_scores)
    i_mean = sum(invert_scores) / len(invert_scores)
    print(f"\nResults over {n_images} images:")
    print(f"  Baseline ROUGE-1  : {b_mean:.4f}")
    print(f"  Best-invert ROUGE-1: {i_mean:.4f}")
    print(f"  Delta             : {i_mean - b_mean:+.4f}")

    import json, os
    out = {
        "n_images": n_images,
        "baseline_mean": round(b_mean, 4),
        "oracle_mean":   round(i_mean, 4),
        "delta_mean":    round(i_mean - b_mean, 4),
        "baseline_scores": [round(s, 4) for s in baseline_scores],
        "oracle_scores":   [round(s, 4) for s in invert_scores],
    }
    save_path = os.path.join(SAVE_DIR, f"inversion_scores_n{n_images}.json")
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Per-image scores saved → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval",     action="store_true")
    parser.add_argument("--n-steps",  type=int,   default=300)
    parser.add_argument("--lr",       type=float, default=0.05)
    parser.add_argument("--n-images", type=int,   default=100)
    args = parser.parse_args()

    with open(os.path.join(SAVE_DIR, "tok_vocab.json")) as f:
        vocab = json.load(f)
    tok = Tokenizer()
    tok.vocab = vocab
    tok.w2i = {w: i for i, w in enumerate(vocab)}
    print(f"Vocab: {len(tok.vocab)} tokens")

    proj = PatchProjector(ENCODER_DIM, D_MODEL).to(DEVICE)
    proj.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_proj.pt"),
                                    map_location=DEVICE, weights_only=True))
    proj.eval()

    guided_dec = SoftGuidedCaptionDecoder(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)
    guided_dec.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_guided.pt"),
                                           map_location=DEVICE, weights_only=True))
    guided_dec.eval()
    print("Models loaded.")

    patch_feats = torch.load(CACHE_PATH, weights_only=True)
    print(f"Patch features: {patch_feats.shape}")

    plans_test = torch.load(os.path.join(SAVE_DIR, "plans_test.pt"), weights_only=True)
    print(f"Plans test: {plans_test.shape}")

    if args.eval:
        run_eval(guided_dec, proj, patch_feats, plans_test, tok,
                 n_images=args.n_images, n_steps=args.n_steps, lr=args.lr)
    else:
        examples = [
            (0,  "a dog running through the grass"),
            (10, "a woman in a red dress"),
            (20, "a plate of vegetables on a table"),
            (50, "a group of people standing near water"),
            (97, "a little girl eating a banana"),
        ]
        run_editing_demo(guided_dec, proj, patch_feats, plans_test, tok,
                         examples, n_steps=args.n_steps, lr=args.lr)


if __name__ == "__main__":
    main()
