"""
Plan Arithmetic in CLIP-ViT-B/32 space (compare with DINOv2 version).

Key difference from run_plan_arithmetic.py:
  - CLIP ViT-B/32 patch features: 49 patches (7×7 grid, 32px each) @ 768-dim
  - CLIP features co-trained with text → plan dimensions correlate with
    semantic concepts → arithmetic should work better than DINOv2

Hypothesis: CLIP-space plan arithmetic hits ≥40% target-noun rate
(vs DINOv2 which may fall below due to cross-modal geometry mismatch).

Outputs:
  results/plan_arithmetic_clip.json
  results/plan_arithmetic_clip_viewer.html

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 \
        run_clip_plan_arithmetic.py > results/plan_arith_clip.log 2>&1
"""

import os, sys, json, io, base64, re

# Must set before any import from run_cap_improved (module-level constants)
os.environ["ATTN_ENCODER"] = "clip"

from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from run_summarization import Tokenizer, DEVICE, D_MODEL, N_HEADS, N_LAYERS, L_SUM as L_CAP, BATCH, LR, N_MP
from run_captioning import (PatchProjector, CaptionDecoder, CocoCapDataset,
                            collate_cap, A0CapDataset, train_free_cap, extract_A0_cap, N_CAP_PER)
from run_cap_improved import (ENCODER, N_PATCHES, ENCODER_DIM,
                               N_IMAGES, VAL_IMAGES, TEST_IMAGES,
                               EPOCHS_FREE, EPOCHS_DEN, EPOCHS_GUIDED,
                               SEEDS, RESULTS_DIR, CACHE_PATH,
                               extract_patch_features,
                               SoftGuidedCaptionDecoder, CFGDenoiser,
                               train_cfg_denoiser, train_guided_mixed,
                               maskgit_cfg, recovery_test)

assert ENCODER == "clip", f"Expected clip, got {ENCODER} — env var not picked up"
print(f"  Encoder: {ENCODER}  N_PATCHES={N_PATCHES}  ENCODER_DIM={ENCODER_DIM}")

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_OK = True
except Exception:
    SPACY_OK = False
    print("WARNING: spaCy not available, using regex noun extraction")

SEED           = 42
T_SOFTMAX      = 10.0
TEST_OFFSET    = N_IMAGES + VAL_IMAGES

COCO_NOUNS = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis",
    "snowboard","kite","bat","glove","skateboard","surfboard","bottle","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli",
    "carrot","pizza","donut","cake","chair","couch","bed","toilet","monitor",
    "laptop","mouse","keyboard","phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy","toothbrush",
    "cellphone","cell","racket","racquet","hotdog","hot dog",
}

NOUN_PAIRS = [
    ("train",  "car"),
    ("dog",    "cat"),
    ("chair",  "couch"),
    ("pizza",  "sandwich"),
    ("horse",  "cow"),
]
MIN_CENTROID_N  = 15
N_TEST_PER_PAIR = 30
N_DEMO_PER_PAIR = 4

# CLIP-ViT-B/32: 7×7 grid, 32px patches on 224px input
PATCH_GRID = 7
PATCH_SIZE = 32


# ── Noun helpers ──────────────────────────────────────────────────────────────
def extract_dominant_noun(caption: str) -> str | None:
    cap_lower = caption.lower()
    if SPACY_OK:
        doc = nlp(cap_lower)
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN") and token.lemma_ in COCO_NOUNS:
                return token.lemma_
    for word in cap_lower.split():
        w = re.sub(r"[^a-z]", "", word)
        if w in COCO_NOUNS:
            return w
    return None


def caption_contains(caption: str, noun: str) -> bool:
    cap = caption.lower()
    variants = {noun, noun + "s", noun + "es"}
    return any(v in cap.split() or v + "," in cap.split() or v + "." in cap.split()
               for v in variants)


# ── Plan arithmetic ───────────────────────────────────────────────────────────
def plan_arithmetic(A_src, centroid_src, centroid_tgt, alpha=1.0):
    A_edit = A_src - alpha * centroid_src + alpha * centroid_tgt
    A_edit = A_edit.clamp(min=0)
    row_sum = A_edit.sum(-1, keepdim=True).clamp(min=1e-8)
    return A_edit / row_sum


# ── Image helpers ─────────────────────────────────────────────────────────────
def pil_to_b64(img: Image.Image, quality=90) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def highlight_diff(img, A_before, A_after):
    """Red = decreased attention, green = increased attention."""
    diff = (A_after - A_before).sum(0)
    diff = diff / diff.abs().max().clamp(min=1e-8)
    img  = img.copy().resize((224, 224)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for pid in range(len(diff)):
        row, col = divmod(pid, PATCH_GRID)
        x0, y0 = col * PATCH_SIZE, row * PATCH_SIZE
        x1, y1 = x0 + PATCH_SIZE, y0 + PATCH_SIZE
        v = diff[pid].item()
        if v > 0.15:
            draw.rectangle([x0,y0,x1,y1], fill=(40,160,40,int(200*v)))
        elif v < -0.15:
            draw.rectangle([x0,y0,x1,y1], fill=(200,40,40,int(200*(-v))))
    return Image.alpha_composite(img, overlay).convert("RGB")


# ── Training + extraction ─────────────────────────────────────────────────────
def train_and_extract(patch_feats, caps_per_img, tok, seed=SEED):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    n_tv = N_IMAGES + VAL_IMAGES
    train_feats = patch_feats[:N_IMAGES]
    tv_feats    = patch_feats[:n_tv]
    test_feats  = patch_feats[n_tv: n_tv + TEST_IMAGES]
    train_caps  = caps_per_img[:N_IMAGES]
    tv_caps     = caps_per_img[:n_tv]
    test_caps   = caps_per_img[n_tv: n_tv + TEST_IMAGES]

    proj       = PatchProjector(ENCODER_DIM, D_MODEL).to(DEVICE)
    free_dec   = CaptionDecoder(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS,
                                n_src=N_PATCHES).to(DEVICE)
    denoiser   = CFGDenoiser(D_MODEL, N_PATCHES, L_CAP, N_MP).to(DEVICE)
    guided_dec = SoftGuidedCaptionDecoder(
                     len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)

    train_ds     = CocoCapDataset(train_feats, train_caps, tok, proj)
    train_loader = DataLoader(train_ds, BATCH, shuffle=True, collate_fn=collate_cap)

    print("  Phase 1: free decoder...")
    train_free_cap(proj, free_dec, train_loader, EPOCHS_FREE, DEVICE)

    tv_ds     = CocoCapDataset(tv_feats, tv_caps, tok, proj)
    tv_loader = DataLoader(tv_ds, BATCH, shuffle=False, collate_fn=collate_cap)
    print("  Extracting A₀ plans (train+val)...")
    A0_all, X_all, _, ids_all = extract_A0_cap(proj, free_dec, tv_loader, DEVICE)
    A0_ds = A0CapDataset(A0_all, X_all, ids_all)
    print(f"  A₀ dataset: {len(A0_ds)} plans")

    print("  Phase 2: CFG denoiser...")
    train_cfg_denoiser(denoiser, A0_ds, EPOCHS_DEN, DEVICE)

    print("  Phase 3: guided decoder...")
    train_guided_mixed(guided_dec, A0_ds, denoiser, EPOCHS_GUIDED, DEVICE)

    # Free captions for noun labelling
    tv_ds2     = CocoCapDataset(tv_feats, tv_caps, tok, proj)
    tv_loader2 = DataLoader(tv_ds2, BATCH, shuffle=False, collate_fn=collate_cap)
    free_captions_tv = []
    with torch.no_grad():
        proj.eval(); free_dec.eval()
        for batch in tv_loader2:
            X = proj(batch["patches"].to(DEVICE))
            preds = free_dec.generate(X).cpu().tolist()
            free_captions_tv += [tok.dec(p) for p in preds]

    return (proj, free_dec, guided_dec, denoiser,
            A0_all, X_all, free_captions_tv,
            test_feats, test_caps)


# ── Build concept centroids ───────────────────────────────────────────────────
def build_centroids(A0_all, free_captions):
    noun_plans = defaultdict(list)
    for A0, cap in zip(A0_all, free_captions):
        noun = extract_dominant_noun(cap)
        if noun:
            noun_plans[noun].append(A0)

    centroids = {}
    counts = {}
    for noun, plans in noun_plans.items():
        if len(plans) >= MIN_CENTROID_N:
            stacked = torch.stack(plans)
            mean    = stacked.mean(0)
            centroids[noun] = mean
            counts[noun] = len(plans)

    print(f"  Centroids for {len(centroids)} nouns (min {MIN_CENTROID_N} plans each)")
    for noun, plans in sorted(noun_plans.items(), key=lambda x: -len(x[1]))[:15]:
        flag = "✓" if noun in centroids else "✗"
        print(f"    {flag} {noun:15s}: {len(plans):4d} plans")
    return centroids, counts


# ── Evaluate arithmetic ───────────────────────────────────────────────────────
@torch.no_grad()
def eval_arithmetic(proj, free_dec, guided_dec, denoiser,
                    test_feats, test_caps, tok,
                    centroids, noun_pairs, coco_imgs):
    proj.eval(); free_dec.eval(); guided_dec.eval(); denoiser.eval()
    results = {}

    for src_noun, tgt_noun in noun_pairs:
        if src_noun not in centroids or tgt_noun not in centroids:
            print(f"  SKIP {src_noun}→{tgt_noun}: centroid missing")
            continue

        c_src = centroids[src_noun].to(DEVICE)
        c_tgt = centroids[tgt_noun].to(DEVICE)

        source_indices = []
        for i, caps in enumerate(test_caps):
            cap = caps[0] if caps else ""
            if caption_contains(cap, src_noun):
                source_indices.append(i)
            if len(source_indices) >= N_TEST_PER_PAIR:
                break

        if len(source_indices) < 5:
            print(f"  SKIP {src_noun}→{tgt_noun}: only {len(source_indices)} source images")
            continue

        n_hit_tgt = n_drop_src = n_both = 0
        demo_records = []

        for idx in source_indices:
            feat = test_feats[idx].unsqueeze(0).to(DEVICE)
            X    = proj(feat)

            free_ids = free_dec.generate(X)[0].cpu().tolist()
            free_cap = tok.dec(free_ids)

            A0_mg   = maskgit_cfg(denoiser, X, n_steps=4, device=DEVICE)
            A0_plan = F.softmax(A0_mg * T_SOFTMAX, dim=-1)
            A0_src  = A0_plan[0]

            A0_edit       = plan_arithmetic(A0_src, c_src, c_tgt)
            A0_edit_batch = A0_edit.unsqueeze(0)

            edit_ids = guided_dec.generate(X, A0_edit_batch)[0].cpu().tolist()
            edit_cap = tok.dec(edit_ids)

            hit_tgt  = caption_contains(edit_cap, tgt_noun)
            drop_src = not caption_contains(edit_cap, src_noun)
            both     = hit_tgt and drop_src

            n_hit_tgt  += hit_tgt
            n_drop_src += drop_src
            n_both     += both

            if len(demo_records) < N_DEMO_PER_PAIR and (both or len(demo_records) < 2):
                orig_img = coco_imgs.get(TEST_OFFSET + idx)
                diff_img = highlight_diff(orig_img, A0_src.cpu(),
                                          A0_edit.cpu()) if orig_img else None
                demo_records.append({
                    "test_idx": idx,
                    "free_cap": free_cap,
                    "edit_cap": edit_cap,
                    "hit_tgt":  bool(hit_tgt),
                    "drop_src": bool(drop_src),
                    "img_orig": pil_to_b64(orig_img.resize((224,224))) if orig_img else None,
                    "img_diff": pil_to_b64(diff_img) if diff_img else None,
                })

        n = len(source_indices)
        rate_tgt  = n_hit_tgt  / n
        rate_drop = n_drop_src / n
        rate_both = n_both     / n

        print(f"\n  {src_noun}→{tgt_noun}  (n={n})")
        print(f"    target appeared:  {rate_tgt:.1%}  ({n_hit_tgt}/{n})")
        print(f"    source dropped:   {rate_drop:.1%}  ({n_drop_src}/{n})")
        print(f"    both (clean sub): {rate_both:.1%}  ({n_both}/{n})")
        for dr in demo_records[:3]:
            sym = "✓" if dr["drop_src"] and dr["hit_tgt"] else ("~" if dr["hit_tgt"] else "✗")
            print(f"      {sym} '{dr['free_cap'][:55]}' → '{dr['edit_cap'][:55]}'")

        results[f"{src_noun}→{tgt_noun}"] = {
            "src_noun":              src_noun,
            "tgt_noun":              tgt_noun,
            "n":                     n,
            "rate_target_appeared":  round(rate_tgt,  3),
            "rate_source_dropped":   round(rate_drop, 3),
            "rate_both":             round(rate_both, 3),
            "demo":                  demo_records,
        }

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    clip_cache = os.path.join(RESULTS_DIR, "clip_patch_features_15k.pt")

    if os.path.exists(clip_cache):
        print(f"Loading cached CLIP features from {clip_cache}...")
        patch_feats = torch.load(clip_cache, weights_only=True)
    else:
        print("Extracting CLIP ViT-B/32 patch features for 15K images...")
        from datasets import load_dataset as _lds
        ds    = _lds("clip-benchmark/wds_mscoco_captions", split="train")
        total = N_IMAGES + VAL_IMAGES + TEST_IMAGES
        imgs  = []
        for item in ds.select(range(total)):
            img = item["jpg"]
            if img.mode != "RGB": img = img.convert("RGB")
            imgs.append(img)
        patch_feats = extract_patch_features(imgs, encoder="clip", device=DEVICE)
        torch.save(patch_feats, clip_cache)
        print(f"  Saved to {clip_cache}")
    print(f"  patch_feats: {patch_feats.shape}  (expected [15000, 49, 768])")

    total = N_IMAGES + VAL_IMAGES + TEST_IMAGES
    print("Loading MSCOCO captions...")
    ds = load_dataset("clip-benchmark/wds_mscoco_captions", split="train")
    caps_per_img = []
    for item in ds.select(range(total)):
        caps = [c.strip() for c in item["txt"].strip().split("\n") if c.strip()]
        caps_per_img.append(caps[:N_CAP_PER])

    tok = Tokenizer()
    tok.build([c for caps in caps_per_img for c in caps], max_vocab=6000)
    print(f"  Vocab: {len(tok.vocab)}")

    print("Loading test images for demo...")
    test_offset = N_IMAGES + VAL_IMAGES
    coco_imgs = {}
    for ds_idx in range(test_offset, test_offset + min(TEST_IMAGES, N_TEST_PER_PAIR * len(NOUN_PAIRS))):
        if ds_idx < len(ds):
            img = ds[ds_idx]["jpg"]
            if img.mode != "RGB": img = img.convert("RGB")
            coco_imgs[ds_idx] = img

    print("\nTraining 3-phase model (CLIP encoder, N_PATCHES=49)...")
    (proj, free_dec, guided_dec, denoiser,
     A0_all, X_all, free_captions_tv,
     test_feats, test_caps) = train_and_extract(patch_feats, caps_per_img, tok)

    print("\nBuilding concept centroids...")
    centroids, centroid_counts = build_centroids(A0_all, free_captions_tv)

    print("\nEvaluating plan arithmetic (CLIP space)...")
    results = eval_arithmetic(
        proj, free_dec, guided_dec, denoiser,
        test_feats, test_caps[:TEST_IMAGES], tok,
        centroids, NOUN_PAIRS, coco_imgs,
    )

    if results:
        rates = [v["rate_target_appeared"] for v in results.values()]
        both  = [v["rate_both"]            for v in results.values()]
        print(f"\n{'='*60}")
        print(f"CLIP-SPACE PLAN ARITHMETIC SUMMARY")
        print(f"Mean target-appeared rate: {np.mean(rates):.1%}")
        print(f"Mean clean-substitution:   {np.mean(both):.1%}")
        if np.mean(rates) >= 0.40:
            print("→ CLIP plan space supports additive edits — Path 2 is low-risk")
        elif np.mean(rates) >= 0.20:
            print("→ Partial linearity — Path 2 may work with soft conditioning")
        else:
            print("→ CLIP plan space also non-linear — Path 3 (flow matching) needed")
        print(f"{'='*60}")

    out = {
        "encoder":        "clip",
        "n_patches":      N_PATCHES,
        "seed":           SEED,
        "min_centroid_n": MIN_CENTROID_N,
        "n_test_per_pair": N_TEST_PER_PAIR,
        "t_softmax":      T_SOFTMAX,
        "centroid_counts": {k: v for k, v in centroid_counts.items()
                            if any(k in pair for pair in NOUN_PAIRS)},
        "results":        results,
        "summary": {
            "mean_target_appeared": round(float(np.mean(rates)), 3) if results else None,
            "mean_clean_sub":       round(float(np.mean(both)),  3) if results else None,
        },
    }
    path = os.path.join(RESULTS_DIR, "plan_arithmetic_clip.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
