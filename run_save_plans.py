"""
Train the 3-phase model and save A₀ plans + labels + checkpoints to disk.
This is the shared prerequisite for all four downstream analysis scripts.

Saves to results/saved_plans/:
  plans_tv.pt          — [14000, L_cap, N_patches] float32  (train+val)
  plans_test.pt        — [1000,  L_cap, N_patches] float32  (test split)
  captions_tv.txt      — one free caption per line (14000 lines)
  captions_test.txt    — one free caption per line (1000 lines)
  noun_labels_tv.json  — dominant COCO noun per plan, or null  (14000 entries)
  noun_labels_test.json
  model_proj.pt        — PatchProjector state_dict
  model_free.pt        — CaptionDecoder   state_dict
  model_denoiser.pt    — CFGDenoiser      state_dict
  model_guided.pt      — SoftGuidedCaptionDecoder state_dict
  tok_vocab.json       — tokenizer vocab list (for decoding captions)
  meta.json            — shapes, N_PATCHES, ENCODER_DIM, etc.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 \\
        run_save_plans.py > results/save_plans.log 2>&1
"""

import os, sys, json, re
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from run_summarization import Tokenizer, DEVICE, D_MODEL, N_HEADS, N_LAYERS, L_SUM as L_CAP, BATCH, LR, N_MP
from run_captioning import (PatchProjector, CaptionDecoder, CocoCapDataset,
                            collate_cap, A0CapDataset, train_free_cap,
                            extract_A0_cap, N_CAP_PER)
from run_cap_improved import (ENCODER, N_PATCHES, ENCODER_DIM,
                               N_IMAGES, VAL_IMAGES, TEST_IMAGES,
                               EPOCHS_FREE, EPOCHS_DEN, EPOCHS_GUIDED,
                               RESULTS_DIR, CACHE_PATH, extract_patch_features,
                               SoftGuidedCaptionDecoder, CFGDenoiser,
                               train_cfg_denoiser, train_guided_mixed, maskgit_cfg)

SAVE_DIR = os.path.join(RESULTS_DIR, "saved_plans")
SEED     = 42

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

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_OK = True
except Exception:
    SPACY_OK = False


def extract_dominant_noun(caption):
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


@torch.no_grad()
def generate_captions_and_plans(proj, free_dec, denoiser, feats, tok, batch_size=32):
    """Generate free captions and A₀ plans for a feature tensor."""
    proj.eval(); free_dec.eval(); denoiser.eval()
    all_caps, all_plans = [], []
    for i in range(0, len(feats), batch_size):
        X_raw = feats[i:i+batch_size].to(DEVICE)
        X = proj(X_raw)
        ids = free_dec.generate(X).cpu().tolist()
        all_caps += [tok.dec(ids_i) for ids_i in ids]
        A0 = maskgit_cfg(denoiser, X, n_steps=4, device=DEVICE)
        A0 = F.softmax(A0 * 10.0, dim=-1).cpu()
        all_plans.append(A0)
        if (i // batch_size) % 20 == 0:
            print(f"    {i+len(X_raw)}/{len(feats)}")
    return all_caps, torch.cat(all_plans, dim=0)


def main():
    import random
    random.seed(SEED); torch.manual_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Load features ─────────────────────────────────────────────────────────
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached features from {CACHE_PATH}...")
        patch_feats = torch.load(CACHE_PATH, weights_only=True)
    else:
        print("Extracting patch features...")
        from datasets import load_dataset as _lds
        ds = _lds("clip-benchmark/wds_mscoco_captions", split="train")
        total = N_IMAGES + VAL_IMAGES + TEST_IMAGES
        imgs = []
        for item in ds.select(range(total)):
            img = item["jpg"]
            if img.mode != "RGB": img = img.convert("RGB")
            imgs.append(img)
        patch_feats = extract_patch_features(imgs, encoder=ENCODER, device=DEVICE)
        torch.save(patch_feats, CACHE_PATH)
    print(f"  features: {patch_feats.shape}")

    # ── Load captions ─────────────────────────────────────────────────────────
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

    # ── Train 3-phase model ───────────────────────────────────────────────────
    n_tv = N_IMAGES + VAL_IMAGES
    train_feats = patch_feats[:N_IMAGES]
    tv_feats    = patch_feats[:n_tv]
    test_feats  = patch_feats[n_tv: n_tv + TEST_IMAGES]
    train_caps  = caps_per_img[:N_IMAGES]
    tv_caps     = caps_per_img[:n_tv]

    proj       = PatchProjector(ENCODER_DIM, D_MODEL).to(DEVICE)
    free_dec   = CaptionDecoder(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS,
                                n_src=N_PATCHES).to(DEVICE)
    denoiser   = CFGDenoiser(D_MODEL, N_PATCHES, L_CAP, N_MP).to(DEVICE)
    guided_dec = SoftGuidedCaptionDecoder(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)

    train_ds     = CocoCapDataset(train_feats, train_caps, tok, proj)
    train_loader = DataLoader(train_ds, BATCH, shuffle=True, collate_fn=collate_cap)

    print("Phase 1: free decoder...")
    train_free_cap(proj, free_dec, train_loader, EPOCHS_FREE, DEVICE)

    tv_ds     = CocoCapDataset(tv_feats, tv_caps, tok, proj)
    tv_loader = DataLoader(tv_ds, BATCH, shuffle=False, collate_fn=collate_cap)
    print("Extracting A₀ plans (train+val)...")
    A0_all, X_all, _, ids_all = extract_A0_cap(proj, free_dec, tv_loader, DEVICE)
    A0_ds = A0CapDataset(A0_all, X_all, ids_all)
    print(f"  A₀ dataset: {len(A0_ds)} plans")

    print("Phase 2: CFG denoiser...")
    train_cfg_denoiser(denoiser, A0_ds, EPOCHS_DEN, DEVICE)

    print("Phase 3: guided decoder...")
    train_guided_mixed(guided_dec, A0_ds, denoiser, EPOCHS_GUIDED, DEVICE)

    # ── Generate captions + denoised plans for train+val ─────────────────────
    print("\nGenerating captions + denoised plans for train+val (14K images)...")
    caps_tv, plans_tv = generate_captions_and_plans(proj, free_dec, denoiser, tv_feats, tok)
    noun_labels_tv = [extract_dominant_noun(c) for c in caps_tv]

    print("Generating captions + denoised plans for test (1K images)...")
    caps_test, plans_test = generate_captions_and_plans(proj, free_dec, denoiser, test_feats, tok)
    noun_labels_test = [extract_dominant_noun(c) for c in caps_test]

    # ── Save everything ───────────────────────────────────────────────────────
    print(f"\nSaving to {SAVE_DIR}/ ...")

    torch.save(plans_tv,   os.path.join(SAVE_DIR, "plans_tv.pt"))
    torch.save(plans_test, os.path.join(SAVE_DIR, "plans_test.pt"))
    print(f"  plans_tv:   {plans_tv.shape}")
    print(f"  plans_test: {plans_test.shape}")

    with open(os.path.join(SAVE_DIR, "captions_tv.txt"),   "w") as f:
        f.write("\n".join(caps_tv))
    with open(os.path.join(SAVE_DIR, "captions_test.txt"), "w") as f:
        f.write("\n".join(caps_test))

    with open(os.path.join(SAVE_DIR, "noun_labels_tv.json"),   "w") as f:
        json.dump(noun_labels_tv,   f)
    with open(os.path.join(SAVE_DIR, "noun_labels_test.json"), "w") as f:
        json.dump(noun_labels_test, f)

    torch.save(proj.state_dict(),       os.path.join(SAVE_DIR, "model_proj.pt"))
    torch.save(free_dec.state_dict(),   os.path.join(SAVE_DIR, "model_free.pt"))
    torch.save(denoiser.state_dict(),   os.path.join(SAVE_DIR, "model_denoiser.pt"))
    torch.save(guided_dec.state_dict(), os.path.join(SAVE_DIR, "model_guided.pt"))

    tok_vocab_path = os.path.join(SAVE_DIR, "tok_vocab.json")
    with open(tok_vocab_path, "w") as f:
        json.dump(tok.vocab, f)

    meta = {
        "encoder":      ENCODER,
        "n_patches":    N_PATCHES,
        "encoder_dim":  ENCODER_DIM,
        "l_cap":        L_CAP,
        "n_images_tv":  n_tv,
        "n_images_test": TEST_IMAGES,
        "d_model":      D_MODEL,
        "n_heads":      N_HEADS,
        "n_layers":     N_LAYERS,
        "plans_tv_shape":   list(plans_tv.shape),
        "plans_test_shape": list(plans_test.shape),
    }
    with open(os.path.join(SAVE_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # noun distribution
    from collections import Counter
    dist = Counter(n for n in noun_labels_tv if n)
    top15 = dist.most_common(15)
    print("\nTop nouns in train+val:")
    for noun, cnt in top15:
        print(f"  {noun:15s}: {cnt}")

    print(f"\nDone. All files in {SAVE_DIR}/")


if __name__ == "__main__":
    main()
