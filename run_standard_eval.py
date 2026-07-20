"""
Standard captioning metrics evaluation (CIDEr-D, BLEU-4, METEOR, ROUGE-L).

Evaluates:
  (1) Free decoder — baseline, no attention plans
  (2) Guided decoder — our method, guided by denoised A₀ plans

Compares to published ClipCap / OSCAR numbers for reference.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 \\
        run_standard_eval.py > results/standard_eval.log 2>&1
"""

import os, sys, json, warnings
import torch
import torch.nn.functional as F
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from run_summarization  import Tokenizer, DEVICE, D_MODEL, N_HEADS, N_LAYERS, L_SUM as L_CAP, N_MP
from run_captioning     import PatchProjector, CaptionDecoder, N_CAP_PER
from run_cap_improved   import (
    ENCODER, N_PATCHES, ENCODER_DIM, N_IMAGES, VAL_IMAGES, TEST_IMAGES,
    RESULTS_DIR, CACHE_PATH, SoftGuidedCaptionDecoder, CFGDenoiser, maskgit_cfg
)

SAVE_DIR  = os.path.join(RESULTS_DIR, "saved_plans")
OUT_JSON  = os.path.join(RESULTS_DIR, "standard_eval.json")
SEED      = 42
torch.manual_seed(SEED)

# ── Published baseline numbers (Karpathy test, trained on ~83K images) ────────
PUBLISHED = {
    "ClipCap (83K)":  {"CIDEr": 113.1, "BLEU-4": 33.5, "METEOR": 27.2, "ROUGE-L": 56.4},
    "OSCAR-base (83K)": {"CIDEr": 137.6, "BLEU-4": 36.5, "METEOR": 30.6, "ROUGE-L": 58.0},
}


# ── Metric helpers ─────────────────────────────────────────────────────────────
def run_scorers(refs, hyps):
    """
    refs: {str_id: [ref1, ref2, ...]}
    hyps: {str_id: [hypothesis]}
    Returns dict of metric → score.
    """
    from pycocoevalcap.bleu.bleu   import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider

    scores = {}

    # BLEU-1..4
    bleu_scorer = Bleu(4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bleu, _ = bleu_scorer.compute_score(refs, hyps)
    for i, b in enumerate(bleu, 1):
        scores[f"BLEU-{i}"] = round(b * 100, 2)

    # ROUGE-L
    rouge_scorer = Rouge()
    r, _ = rouge_scorer.compute_score(refs, hyps)
    scores["ROUGE-L"] = round(r * 100, 2)

    # CIDEr-D
    cider_scorer = Cider()
    c, _ = cider_scorer.compute_score(refs, hyps)
    scores["CIDEr"] = round(c * 100, 2)

    # METEOR (requires Java; skip gracefully if unavailable)
    try:
        from pycocoevalcap.meteor.meteor import Meteor
        meteor_scorer = Meteor()
        m, _ = meteor_scorer.compute_score(refs, hyps)
        scores["METEOR"] = round(m * 100, 2)
        meteor_scorer.meteor_p.kill()
    except Exception as e:
        print(f"  METEOR unavailable ({e}); skipping")
        scores["METEOR"] = None

    return scores


# ── Caption generation ─────────────────────────────────────────────────────────
@torch.no_grad()
def gen_free(proj, free_dec, feats, tok, batch=32):
    proj.eval(); free_dec.eval()
    caps = []
    for i in range(0, len(feats), batch):
        X   = proj(feats[i:i+batch].to(DEVICE))
        ids = free_dec.generate(X).cpu().tolist()
        new_caps = [tok.dec(row) for row in ids]
        caps += new_caps
        if (i // batch) % 10 == 0:
            print(f"    free: {i+len(new_caps)}/{len(feats)}", flush=True)
    return caps


@torch.no_grad()
def gen_guided(proj, guided_dec, feats, denoiser, tok, batch=32):
    proj.eval(); guided_dec.eval(); denoiser.eval()
    caps = []
    for i in range(0, len(feats), batch):
        X  = proj(feats[i:i+batch].to(DEVICE))
        A0 = maskgit_cfg(denoiser, X, n_steps=4, device=DEVICE)
        A0 = F.softmax(A0 * 10.0, dim=-1)
        ids = guided_dec.generate(X, A0).cpu().tolist()
        new_caps = [tok.dec(row) for row in ids]
        caps += new_caps
        if (i // batch) % 10 == 0:
            print(f"    guided: {i+len(new_caps)}/{len(feats)}", flush=True)
    return caps


def main():
    n_tv = N_IMAGES + VAL_IMAGES

    # ── 1. Load features ───────────────────────────────────────────────────────
    print("Loading cached patch features...")
    patch_feats = torch.load(CACHE_PATH, weights_only=True)
    test_feats  = patch_feats[n_tv : n_tv + TEST_IMAGES]
    print(f"  test_feats: {test_feats.shape}")

    # ── 2. Load reference captions ─────────────────────────────────────────────
    print("Loading MSCOCO reference captions for test set...")
    ds = load_dataset("clip-benchmark/wds_mscoco_captions", split="train")
    total = N_IMAGES + VAL_IMAGES + TEST_IMAGES
    ref_caps = []
    for item in ds.select(range(n_tv, total)):
        caps = [c.strip() for c in item["txt"].strip().split("\n") if c.strip()]
        ref_caps.append(caps[:N_CAP_PER])
    print(f"  {len(ref_caps)} images × up to {N_CAP_PER} captions each")

    refs = {str(i): ref_caps[i] for i in range(len(ref_caps))}

    # ── 3. Load tokenizer & models ─────────────────────────────────────────────
    print("Loading tokenizer...")
    tok_vocab_path = os.path.join(SAVE_DIR, "tok_vocab.json")
    with open(tok_vocab_path) as f:
        vocab = json.load(f)
    tok = Tokenizer()
    tok.vocab = vocab
    tok.w2i   = {w: i for i, w in enumerate(vocab)}

    print("Loading models...")
    proj = PatchProjector(ENCODER_DIM, D_MODEL).to(DEVICE)
    proj.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_proj.pt"),
                                    weights_only=True, map_location=DEVICE))

    free_dec = CaptionDecoder(len(vocab), D_MODEL, N_HEADS, N_LAYERS,
                              n_src=N_PATCHES).to(DEVICE)
    free_dec.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_free.pt"),
                                        weights_only=True, map_location=DEVICE))

    denoiser = CFGDenoiser(D_MODEL, N_PATCHES, L_CAP, n_mp=N_MP).to(DEVICE)
    denoiser.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_denoiser.pt"),
                                        weights_only=True, map_location=DEVICE))

    guided_dec = SoftGuidedCaptionDecoder(len(vocab), D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)
    guided_dec.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_guided.pt"),
                                          weights_only=True, map_location=DEVICE))
    print("  Models loaded OK")

    # ── 4a. Use saved free captions (already generated) ───────────────────────
    print("\nLoading saved free-decoder captions...")
    with open(os.path.join(SAVE_DIR, "captions_test.txt")) as f:
        free_caps = [l.rstrip("\n") for l in f]
    # Trim / pad to TEST_IMAGES
    free_caps = free_caps[:TEST_IMAGES]
    if len(free_caps) < TEST_IMAGES:
        print(f"  Warning: only {len(free_caps)} captions; padding with UNK")
        free_caps += ["<UNK>"] * (TEST_IMAGES - len(free_caps))
    print(f"  {len(free_caps)} free captions loaded")

    # ── 4b. Generate guided captions ──────────────────────────────────────────
    print("\nGenerating guided captions (guided decoder + denoised A₀)...")
    guided_caps = gen_guided(proj, guided_dec, test_feats, denoiser, tok)
    print(f"  {len(guided_caps)} guided captions generated")

    # ── 5. Compute metrics ─────────────────────────────────────────────────────
    print("\nComputing metrics...")

    hyps_free   = {str(i): [free_caps[i]]   for i in range(TEST_IMAGES)}
    hyps_guided = {str(i): [guided_caps[i]] for i in range(len(guided_caps))}

    # Use only images with valid refs in both conditions
    n_eval = min(len(refs), len(free_caps), len(guided_caps))
    r_eval = {str(i): refs[str(i)] for i in range(n_eval)}
    hf_eval = {str(i): [free_caps[i]]   for i in range(n_eval)}
    hg_eval = {str(i): [guided_caps[i]] for i in range(n_eval)}

    print(f"  Evaluating on {n_eval} images")

    print("  Scoring free decoder...")
    scores_free = run_scorers(r_eval, hf_eval)

    print("  Scoring guided decoder...")
    scores_guided = run_scorers(r_eval, hg_eval)

    # ── 6. Print table ─────────────────────────────────────────────────────────
    metrics = ["CIDEr", "BLEU-4", "METEOR", "ROUGE-L"]
    col_w   = 14

    header = f"{'Model':<30}" + "".join(f"{m:>{col_w}}" for m in metrics)
    sep    = "-" * len(header)

    print(f"\n{'='*60}")
    print("  STANDARD CAPTIONING METRICS")
    print(f"{'='*60}")
    print(header)
    print(sep)

    def row(name, sc):
        vals = [sc.get(m, "—") for m in metrics]
        cells = []
        for v in vals:
            cells.append(f"{v:>{col_w}.1f}" if isinstance(v, float) else f"{'—':>{col_w}}")
        return f"{name:<30}" + "".join(cells)

    print(row(f"Ours-Free   (13K train)", scores_free))
    print(row(f"Ours-Guided (13K train)", scores_guided))
    print(sep)
    for name, sc in PUBLISHED.items():
        print(row(name, sc))
    print(sep)
    print("Note: published numbers on Karpathy test split (83K train); ours on")
    print("      our 1K held-out set (HuggingFace MSCOCO subset, 13K train).")
    print(f"{'='*60}\n")

    # ── 7. Save JSON ───────────────────────────────────────────────────────────
    result = {
        "n_test": n_eval,
        "training_images": N_IMAGES,
        "our_free":   scores_free,
        "our_guided": scores_guided,
        "published":  PUBLISHED,
        "sample_captions": {
            "free":   free_caps[:5],
            "guided": guided_caps[:5],
            "refs":   [ref_caps[i] for i in range(5)],
        }
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
