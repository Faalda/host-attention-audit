"""
Tier 1.2 — Soft-metric + random-perturbation baseline for plan arithmetic.

Hardens the arithmetic results by adding:
  A) Soft CLIP-text cosine delta (continuous, not binary noun-match)
  B) Random-perturbation baseline (same edit magnitude, random direction)

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 \\
        run_soft_metric_baseline.py > results/soft_metric.log 2>&1
Prerequisite: run_save_plans.py must have completed.
"""

import os, sys, json, re
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from run_summarization import Tokenizer, DEVICE, D_MODEL, N_HEADS, N_LAYERS, L_SUM as L_CAP, N_MP
from run_captioning import PatchProjector, CaptionDecoder
from run_cap_improved import (ENCODER, N_PATCHES, ENCODER_DIM, N_IMAGES, VAL_IMAGES, TEST_IMAGES,
                               RESULTS_DIR, SoftGuidedCaptionDecoder, CFGDenoiser, maskgit_cfg)

SAVE_DIR   = os.path.join(RESULTS_DIR, "saved_plans")
SEED       = 42
T_SOFTMAX  = 10.0
N_TEST_PER_PAIR = 30

NOUN_PAIRS = [
    ("train",  "car"),
    ("dog",    "cat"),
    ("chair",  "couch"),
    ("pizza",  "sandwich"),
    ("horse",  "cow"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def caption_contains(caption, noun):
    cap = caption.lower()
    variants = {noun, noun + "s", noun + "es"}
    return any(v in cap.split() or v + "," in cap.split() or v + "." in cap.split()
               for v in variants)


def load_clip_text_model(device):
    from transformers import CLIPModel, CLIPTokenizer
    print("  Loading CLIP text model (openai/clip-vit-base-patch32)...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, tokenizer


@torch.no_grad()
def clip_text_sim(clip_model, clip_tok, text1, text2, device):
    """Cosine similarity between two text strings via CLIP text encoder."""
    inp1 = clip_tok([text1], return_tensors="pt", padding=True, truncation=True).to(device)
    inp2 = clip_tok([text2], return_tensors="pt", padding=True, truncation=True).to(device)
    def _get_feat(inp):
        f = clip_model.get_text_features(**inp)
        if not isinstance(f, torch.Tensor):
            f = clip_model.text_projection(clip_model.text_model(**inp).pooler_output)
        return F.normalize(f, dim=-1)
    e1 = _get_feat(inp1)
    e2 = _get_feat(inp2)
    return float((e1 * e2).sum().item())


def plan_arithmetic(A_src, c_src, c_tgt, alpha=1.0):
    A_edit = A_src - alpha * c_src + alpha * c_tgt
    A_edit = A_edit.clamp(min=0)
    row_sum = A_edit.sum(-1, keepdim=True).clamp(min=1e-8)
    return A_edit / row_sum


def plan_random_perturb(A_src, c_src, c_tgt, rng):
    """Add Gaussian noise with same std as the centroid diff, random direction."""
    diff = c_tgt - c_src                              # [L, N]
    sigma = diff.std().item()
    noise = torch.tensor(rng.normal(0, sigma, A_src.shape), dtype=A_src.dtype)
    A_pert = A_src + noise
    A_pert = A_pert.clamp(min=0)
    row_sum = A_pert.sum(-1, keepdim=True).clamp(min=1e-8)
    return A_pert / row_sum


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models(save_dir, device):
    meta_path = os.path.join(save_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found in {save_dir} — run run_save_plans.py first")
    meta = json.load(open(meta_path))

    tok = Tokenizer()
    tok.vocab = json.load(open(os.path.join(save_dir, "tok_vocab.json")))
    tok.word2id = {w: i for i, w in enumerate(tok.vocab)}

    V = len(tok.vocab)
    proj       = PatchProjector(ENCODER_DIM, D_MODEL).to(device)
    free_dec   = CaptionDecoder(V, D_MODEL, N_HEADS, N_LAYERS, n_src=N_PATCHES).to(device)
    denoiser   = CFGDenoiser(D_MODEL, N_PATCHES, L_CAP, N_MP).to(device)
    guided_dec = SoftGuidedCaptionDecoder(V, D_MODEL, N_HEADS, N_LAYERS).to(device)

    proj.load_state_dict(torch.load(os.path.join(save_dir, "model_proj.pt"),      weights_only=True), strict=False)
    free_dec.load_state_dict(torch.load(os.path.join(save_dir, "model_free.pt"),  weights_only=True), strict=False)
    denoiser.load_state_dict(torch.load(os.path.join(save_dir, "model_denoiser.pt"), weights_only=True), strict=False)
    guided_dec.load_state_dict(torch.load(os.path.join(save_dir, "model_guided.pt"), weights_only=True), strict=False)

    proj.eval(); free_dec.eval(); denoiser.eval(); guided_dec.eval()
    return proj, free_dec, denoiser, guided_dec, tok, meta


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading saved plans and labels...")
    if not os.path.exists(os.path.join(SAVE_DIR, "plans_tv.pt")):
        raise FileNotFoundError("run_save_plans.py must complete before this script")

    plans_tv   = torch.load(os.path.join(SAVE_DIR, "plans_tv.pt"),   weights_only=True)
    plans_test = torch.load(os.path.join(SAVE_DIR, "plans_test.pt"), weights_only=True)
    noun_labels_tv   = json.load(open(os.path.join(SAVE_DIR, "noun_labels_tv.json")))
    noun_labels_test = json.load(open(os.path.join(SAVE_DIR, "noun_labels_test.json")))
    print(f"  plans_tv:   {plans_tv.shape}")
    print(f"  plans_test: {plans_test.shape}")

    print("Loading patch features for test images...")
    feat_path = os.path.join(RESULTS_DIR, f"{ENCODER}_patch_features_15k.pt")
    all_feats = torch.load(feat_path, weights_only=True)
    test_feats = all_feats[N_IMAGES + VAL_IMAGES: N_IMAGES + VAL_IMAGES + TEST_IMAGES]
    print(f"  test_feats: {test_feats.shape}")

    print("Loading model checkpoints...")
    proj, free_dec, denoiser, guided_dec, tok, meta = load_models(SAVE_DIR, DEVICE)

    print("Loading CLIP text model...")
    clip_model, clip_tok = load_clip_text_model(DEVICE)

    # ── Build centroids from train+val ────────────────────────────────────────
    print("Building concept centroids from train+val plans...")
    noun_to_plans = {}
    for i, noun in enumerate(noun_labels_tv):
        if noun is None:
            continue
        if noun not in noun_to_plans:
            noun_to_plans[noun] = []
        noun_to_plans[noun].append(plans_tv[i])

    centroids = {}
    for noun, plist in noun_to_plans.items():
        if len(plist) >= 15:
            centroids[noun] = torch.stack(plist).mean(0)  # [L, N]

    print(f"  Available centroids: {sorted(centroids.keys())[:10]} ...")

    # ── Evaluate each noun pair ────────────────────────────────────────────────
    results = {}
    for src_noun, tgt_noun in NOUN_PAIRS:
        if src_noun not in centroids or tgt_noun not in centroids:
            print(f"  SKIP {src_noun}→{tgt_noun}: centroid missing")
            continue

        c_src = centroids[src_noun].to(DEVICE)
        c_tgt = centroids[tgt_noun].to(DEVICE)
        tgt_concept = f"a photo of a {tgt_noun}"

        # Find test images with src_noun as dominant label
        source_indices = [i for i, n in enumerate(noun_labels_test) if n == src_noun]
        source_indices = source_indices[:N_TEST_PER_PAIR]
        if len(source_indices) < 5:
            print(f"  SKIP {src_noun}→{tgt_noun}: only {len(source_indices)} source test images")
            continue

        arith_target   = arith_soft_delta   = arith_pos_soft   = 0
        random_target  = random_soft_delta  = random_pos_soft  = 0
        n = len(source_indices)

        print(f"\n  Evaluating {src_noun}→{tgt_noun}  (n={n})")

        with torch.no_grad():
            for idx in source_indices:
                feat = test_feats[idx].unsqueeze(0).to(DEVICE)
                X    = proj(feat)

                # Source caption (free decode)
                src_ids  = free_dec.generate(X)[0].cpu().tolist()
                src_cap  = tok.dec(src_ids)
                sim_src  = clip_text_sim(clip_model, clip_tok, src_cap, tgt_concept, DEVICE)

                # ── Centroid arithmetic ──
                A0_mg    = maskgit_cfg(denoiser, X, n_steps=4, device=DEVICE)
                A0_plan  = F.softmax(A0_mg * T_SOFTMAX, dim=-1)
                A0_src   = A0_plan[0]                          # [L, N]
                A0_arith = plan_arithmetic(A0_src, c_src, c_tgt).unsqueeze(0)
                arith_ids = guided_dec.generate(X, A0_arith)[0].cpu().tolist()
                arith_cap = tok.dec(arith_ids)
                sim_arith = clip_text_sim(clip_model, clip_tok, arith_cap, tgt_concept, DEVICE)

                arith_target  += caption_contains(arith_cap, tgt_noun)
                arith_soft_delta += sim_arith - sim_src
                arith_pos_soft   += (sim_arith > sim_src)

                # ── Random perturbation ──
                A0_pert = plan_random_perturb(A0_src.cpu(), c_src.cpu(), c_tgt.cpu(), rng).to(DEVICE)
                A0_pert = A0_pert.unsqueeze(0)
                rand_ids = guided_dec.generate(X, A0_pert)[0].cpu().tolist()
                rand_cap = tok.dec(rand_ids)
                sim_rand = clip_text_sim(clip_model, clip_tok, rand_cap, tgt_concept, DEVICE)

                random_target    += caption_contains(rand_cap, tgt_noun)
                random_soft_delta += sim_rand - sim_src
                random_pos_soft   += (sim_rand > sim_src)

        a_rate  = arith_target   / n
        a_sdel  = arith_soft_delta  / n
        a_spos  = arith_pos_soft    / n
        r_rate  = random_target  / n
        r_sdel  = random_soft_delta / n
        r_spos  = random_pos_soft   / n

        print(f"    Arithmetic:  target={a_rate:.1%}  soft_delta={a_sdel:+.4f}  frac_pos={a_spos:.1%}")
        print(f"    Random:      target={r_rate:.1%}  soft_delta={r_sdel:+.4f}  frac_pos={r_spos:.1%}")

        results[f"{src_noun}→{tgt_noun}"] = {
            "n": n,
            "arithmetic": {
                "target_appeared":   round(a_rate, 3),
                "soft_delta":        round(a_sdel, 4),
                "frac_positive_soft": round(a_spos, 3),
            },
            "random": {
                "target_appeared":   round(r_rate, 3),
                "soft_delta":        round(r_sdel, 4),
                "frac_positive_soft": round(r_spos, 3),
            },
        }

    if not results:
        print("No results produced — check centroid availability")
        return

    # ── Summary ───────────────────────────────────────────────────────────────
    arith_sdels = [v["arithmetic"]["soft_delta"]  for v in results.values()]
    rand_sdels  = [v["random"]["soft_delta"]       for v in results.values()]
    mean_a = float(np.mean(arith_sdels))
    mean_r = float(np.mean(rand_sdels))

    print(f"\n{'='*60}")
    print(f"SOFT-METRIC BASELINE SUMMARY  ({len(results)} pairs)")
    print(f"  Arithmetic mean soft-delta: {mean_a:+.4f}")
    print(f"  Random     mean soft-delta: {mean_r:+.4f}")
    print(f"  Arithmetic beats random:    {mean_a > mean_r}")
    if mean_a > mean_r:
        print("  → Centroid arithmetic has directional signal (even if weak)")
    else:
        print("  → Centroid arithmetic has NO directional signal vs random noise")
    print(f"{'='*60}")

    out = {
        "encoder":    ENCODER,
        "n_patches":  N_PATCHES,
        "t_softmax":  T_SOFTMAX,
        "noun_pairs": results,
        "summary": {
            "arithmetic_mean_soft_delta": round(mean_a, 4),
            "random_mean_soft_delta":     round(mean_r, 4),
            "arithmetic_beats_random_soft": bool(mean_a > mean_r),
        },
    }
    path = os.path.join(RESULTS_DIR, "soft_metric_baseline.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
