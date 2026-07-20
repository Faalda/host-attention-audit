"""
run_text_denoiser.py — Option A: Text-conditioned plan diffusion.

Trains p_φ(A₀ | X, t) where t is a CLIP text embedding of a desired
caption description.  At inference time, a user provides a text prompt
(e.g. "a cat sitting on a blue chair") and the model generates an
attention plan conditioned on both the image features X and the text t.

Architecture: TextCFGDenoiser extends CFGDenoiser with a text projection
  h ← h + text_proj(t).unsqueeze(1)
applied after the initial row projection.  CFG is applied independently
to X and t, so four prediction modes are possible:
  (cond X, cond t)  → (uncond X, cond t)  → (cond X, uncond t)
  → (uncond X, uncond t)
During inference we use the double-CFG: uncond + α*(Xcond-uncond) + β*(tcond-uncond).

Usage:
    python3 run_text_denoiser.py [--demo]
"""

import os, sys, json, random, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from run_summarization import Tokenizer, DEVICE, D_MODEL, N_HEADS, N_LAYERS, L_SUM as L_CAP, BATCH, LR, N_MP
from run_captioning import (
    PatchProjector, CaptionDecoder, CocoCapDataset, collate_cap,
    extract_A0_cap, N_CAP_PER,
)
from run_cap_improved import (
    ENCODER, N_PATCHES, ENCODER_DIM,
    N_IMAGES, VAL_IMAGES, TEST_IMAGES,
    RESULTS_DIR, CACHE_PATH,
    CFGDenoiser, SoftGuidedCaptionDecoder,
    maskgit_cfg, CFG_SCALE, CFG_DROP_P,
)
from run_sum_masked import mask_attn, MASK_RATES

SAVE_DIR   = os.path.join(RESULTS_DIR, "saved_plans")
TEXT_DIR   = os.path.join(RESULTS_DIR, "text_denoiser")
CLIP_DIM   = 512    # CLIP ViT-B/32 text embedding dimension
TEXT_DROP_P = 0.15  # probability to drop text conditioning during training
EPOCHS_TEXT = 80
CFG_X_SCALE = 2.0
CFG_T_SCALE = 1.5   # text guidance is less aggressive than visual guidance

os.makedirs(TEXT_DIR, exist_ok=True)


# ── Text-conditioned denoiser ─────────────────────────────────────────────────

class TextCFGDenoiser(CFGDenoiser):
    """
    Extends CFGDenoiser with CLIP text conditioning: p_φ(A₀ | X, t).

    Text embedding t ∈ R^{clip_dim} is projected to D and added to each
    row position after the initial row projection.  Both X and t can be
    independently nulled for CFG, giving:
        A_hat = uncond + α*(cond_X - uncond) + β*(cond_t - uncond)
    """
    def __init__(self, D, n_src=N_PATCHES, n_tgt=L_CAP, n_mp=N_MP, clip_dim=CLIP_DIM):
        super().__init__(D, n_src, n_tgt, n_mp)
        self.text_proj = nn.Linear(clip_dim, D)
        self.null_t    = nn.Parameter(torch.zeros(1, D))

    def _forward_once(self, X_src, A_t, mask, t_vec=None):
        B, L, N = A_t.shape
        h = F.gelu(self.row_proj(A_t)) + self.mask_emb(mask.long())

        # Text bias: broadcast to all L positions
        if t_vec is not None:
            h = h + self.text_proj(t_vec).unsqueeze(1)   # [B, D] → [B, 1, D]
        else:
            h = h + self.null_t.unsqueeze(0)

        for mp, ln in zip(self.msg_projs, self.mp_norms):
            msgs = torch.bmm(A_t, mp(X_src))
            h    = ln(h + msgs)
        h2, _ = self.pos_attn(h, h, h, need_weights=False)
        h = self.pos_norm(h + h2)
        Q = F.normalize(self.q_proj(h), dim=-1)
        K = self.k_proj(X_src)
        return torch.bmm(Q, K.transpose(-1, -2)) / self.scale   # [B,L,N]

    def forward(self, X_src, A_t, mask, t_vec=None, drop_X=False, drop_t=False):
        X = (self.null_X.expand(X_src.shape[0], -1, -1) if drop_X else X_src)
        t = (None if drop_t else t_vec)
        return self._forward_once(X, A_t, mask, t_vec=t)

    @torch.no_grad()
    def cfg_predict_text(self, X_src, A_t, mask, t_vec,
                         scale_x=CFG_X_SCALE, scale_t=CFG_T_SCALE):
        """Double-CFG: blend along X and t axes."""
        null_X = self.null_X.expand(X_src.shape[0], -1, -1)
        cond   = self._forward_once(X_src,  A_t, mask, t_vec)
        uncond = self._forward_once(null_X,  A_t, mask, None)
        x_only = self._forward_once(X_src,  A_t, mask, None)   # cond X, uncond t
        t_only = self._forward_once(null_X,  A_t, mask, t_vec) # uncond X, cond t
        return (uncond
                + scale_x * (x_only - uncond)
                + scale_t * (t_only - uncond)
                + scale_x * scale_t / max(scale_x, scale_t) * (cond - x_only - t_only + uncond))


# ── Dataset ───────────────────────────────────────────────────────────────────

class TextA0Dataset(Dataset):
    """
    (A0, X, text_emb) triples for training TextCFGDenoiser.

    A0        [L, N]   oracle attention plan (softmax probabilities)
    X         [N, D]   projected patch features
    text_emb  [512]    CLIP text embedding of the reference caption
    """
    def __init__(self, A0_all, X_all, text_embs):
        assert len(A0_all) == len(X_all) == len(text_embs)
        self.A0   = A0_all
        self.X    = X_all
        self.text = text_embs

    def __len__(self):  return len(self.A0)
    def __getitem__(self, i): return self.A0[i], self.X[i], self.text[i]


# ── CLIP text encoding ────────────────────────────────────────────────────────

def encode_captions_clip(captions, batch_size=256, device=DEVICE):
    """
    Encode a list of strings with CLIP ViT-B/32 text encoder.
    Returns [N, 512] float32 tensor on CPU.
    """
    from transformers import CLIPTokenizer, CLIPTextModel
    print("  Loading CLIP text encoder (ViT-B/32)...")
    tok   = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()
    all_embs = []
    for i in range(0, len(captions), batch_size):
        batch = captions[i: i + batch_size]
        inputs = tok(batch, padding=True, truncation=True,
                     max_length=77, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        # Use pooled (EOS) representation — shape [B, 512]
        emb = out.pooler_output.cpu().float()
        all_embs.append(emb)
        if (i // batch_size) % 10 == 0:
            print(f"    {i+len(batch)}/{len(captions)}")
    return torch.cat(all_embs, dim=0)


# ── Training ──────────────────────────────────────────────────────────────────

def train_text_denoiser(denoiser, ds, n_epochs, device):
    loader = DataLoader(ds, BATCH, shuffle=True, num_workers=2, pin_memory=True)
    opt = torch.optim.Adam(denoiser.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)

    for ep in range(1, n_epochs + 1):
        denoiser.train()
        tot = 0; n = 0
        for A0, X, t in loader:
            A0 = A0.to(device); X = X.to(device); t = t.to(device)
            mr = random.uniform(0.1, 0.95)
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
        if ep % 10 == 0:
            avg = tot / n if n else float('nan')
            print(f"    ep {ep:3d}  text_den_loss={avg:.4f}")


# ── MaskGIT with double-CFG ───────────────────────────────────────────────────

@torch.no_grad()
def maskgit_text_cfg(denoiser, X_src, t_vec, n_steps=4,
                     scale_x=CFG_X_SCALE, scale_t=CFG_T_SCALE, device=None):
    if device is None: device = X_src.device
    B, N, D = X_src.shape
    A    = torch.full((B, L_CAP, N), 1.0 / N, device=device)
    mask = torch.ones(B, L_CAP, device=device)
    n_per_step = max(1, L_CAP // n_steps)

    for _ in range(n_steps):
        logits = denoiser.cfg_predict_text(X_src, A, mask, t_vec, scale_x, scale_t)
        probs  = F.softmax(logits, dim=-1)
        conf   = probs.max(-1).values.masked_fill(~mask.bool(), -1e9)
        n_unmask = min(n_per_step, int(mask.sum(-1).min().item()))
        if n_unmask == 0: break
        _, topk = conf.topk(n_unmask, dim=-1)
        pred = probs.argmax(-1)
        for b in range(B):
            for idx in topk[b]:
                if mask[b, idx] > 0:
                    A[b, idx] = 0; A[b, idx, pred[b, idx]] = 1.0; mask[b, idx] = 0

    final   = denoiser.cfg_predict_text(X_src, A, mask, t_vec, scale_x, scale_t)
    A_out   = F.softmax(final, dim=-1)
    committed = (mask == 0).unsqueeze(-1).float()
    return committed * A + (1 - committed) * A_out


# ── Demo: text-guided caption editing ────────────────────────────────────────

@torch.no_grad()
def demo_text_guided(denoiser, guided_dec, proj, X_raw, text_prompts, tok, device):
    """
    Given a batch of images (raw patch features before proj) and text prompts,
    generate captions conditioned on (X, t).
    """
    from transformers import CLIPTokenizer, CLIPTextModel
    clip_tok   = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_model.eval()
    inputs = clip_tok(text_prompts, padding=True, truncation=True,
                      max_length=77, return_tensors="pt").to(device)
    t_vec = clip_model(**inputs).pooler_output.float()   # [B, 512]

    X = proj(X_raw.to(device))   # [B, N, D]
    plan = maskgit_text_cfg(denoiser, X, t_vec, n_steps=4, device=device)
    toks = guided_dec.generate(X, plan)

    captions = []
    for row in toks:
        words = []
        for t in row.tolist():
            if t == 2: break     # EOS
            if t > 2: words.append(tok.vocab[t])
        captions.append(" ".join(words))
    return captions


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true",
                        help="Run text-guided demo after training (or load from disk)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Load saved TextCFGDenoiser and skip training")
    args = parser.parse_args()

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    with open(os.path.join(SAVE_DIR, "tok_vocab.json")) as f:
        vocab = json.load(f)
    tok = Tokenizer()
    tok.vocab = vocab
    tok.w2i = {w: i for i, w in enumerate(vocab)}
    print(f"Vocab: {len(tok.vocab)} tokens")

    # ── Saved base models ─────────────────────────────────────────────────────
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
    print("Base models loaded.")

    # ── Patch features ────────────────────────────────────────────────────────
    print("Loading patch features...")
    patch_feats = torch.load(CACHE_PATH, weights_only=True)
    train_feats = patch_feats[:N_IMAGES]
    print(f"  train: {train_feats.shape}")

    # ── MSCOCO reference captions (train split) ────────────────────────────────
    print("Loading MSCOCO train captions...")
    from datasets import load_dataset
    ds_hf = load_dataset("clip-benchmark/wds_mscoco_captions", split="train")
    train_caps_flat = []   # one representative caption per image
    train_caps_all  = []   # all captions for oracle extraction
    for item in ds_hf.select(range(N_IMAGES)):
        caps = [c.strip() for c in item["txt"].strip().split("\n") if c.strip()]
        caps = caps[:N_CAP_PER]
        train_caps_all.append(caps)
        train_caps_flat.append(caps[0])   # first caption as text condition
    print(f"  {len(train_caps_flat)} captions")

    # ── Oracle A₀ ─────────────────────────────────────────────────────────────
    print("Extracting oracle A₀...")
    train_ds     = CocoCapDataset(train_feats, train_caps_all, tok, proj)
    train_loader = DataLoader(train_ds, BATCH, shuffle=False, collate_fn=collate_cap)
    A0_train, X_train, _, _ = extract_A0_cap(proj, free_dec, train_loader, DEVICE)
    print(f"  A0: {A0_train.shape},  X: {X_train.shape}")

    # ── CLIP text embeddings ──────────────────────────────────────────────────
    text_emb_path = os.path.join(TEXT_DIR, "train_text_embs.pt")
    if os.path.exists(text_emb_path):
        print("Loading cached CLIP text embeddings...")
        text_embs = torch.load(text_emb_path, weights_only=True)
    else:
        print("Encoding captions with CLIP...")
        text_embs = encode_captions_clip(train_caps_flat, device=DEVICE)
        torch.save(text_embs, text_emb_path)
        print(f"  Saved → {text_emb_path}")
    print(f"  text_embs: {text_embs.shape}")   # [N_images, 512]

    # extract_A0_cap returns one A0 per (image, caption) pair, so repeat each
    # text embedding N_CAP_PER times to align with A0_train
    repeat = len(A0_train) // len(text_embs)
    text_embs_rep = text_embs.repeat_interleave(repeat, dim=0)
    if len(text_embs_rep) < len(A0_train):
        pad = text_embs[:len(A0_train) - len(text_embs_rep)]
        text_embs_rep = torch.cat([text_embs_rep, pad], dim=0)
    print(f"  text_embs repeated: {text_embs_rep.shape} (repeat={repeat})")

    # ── Build dataset ─────────────────────────────────────────────────────────
    text_ds = TextA0Dataset(A0_train.cpu(), X_train.cpu(), text_embs_rep.cpu())

    # ── TextCFGDenoiser ───────────────────────────────────────────────────────
    denoiser = TextCFGDenoiser(D_MODEL, N_PATCHES, L_CAP, N_MP, CLIP_DIM).to(DEVICE)

    # Initialise from CFGDenoiser weights so the image-conditioning path starts
    # from a converged checkpoint. Only text_proj and null_t are randomly init.
    cfg_state = torch.load(os.path.join(SAVE_DIR, "model_denoiser.pt"),
                           map_location=DEVICE, weights_only=True)
    missing, unexpected = denoiser.load_state_dict(cfg_state, strict=False)
    print(f"  Init from CFGDenoiser: missing={missing}, unexpected={unexpected}")

    save_path = os.path.join(TEXT_DIR, "model_text_denoiser_v2.pt")

    if args.skip_train and os.path.exists(save_path):
        denoiser.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
        print(f"Loaded TextCFGDenoiser from {save_path}")
    else:
        n_params = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
        print(f"Fine-tuning TextCFGDenoiser ({n_params/1e6:.2f}M params) for {EPOCHS_TEXT} epochs...")
        train_text_denoiser(denoiser, text_ds, EPOCHS_TEXT, DEVICE)
        torch.save(denoiser.state_dict(), save_path)
        print(f"Saved → {save_path}")

    # ── Quick recovery check ──────────────────────────────────────────────────
    denoiser.eval()
    print("\n--- Recovery check (n=256) ---")
    idx = torch.randperm(len(A0_train))[:256]
    A0s = A0_train[idx].to(DEVICE)
    Xs  = X_train[idx].to(DEVICE)
    ts  = text_embs_rep[idx].to(DEVICE)
    rand1 = 1.0 / N_PATCHES; rand5 = 5.0 / N_PATCHES
    print(f"{'Mask':>6}  {'Top-1':>8}  {'×rand':>8}  {'Top-5':>8}")
    print("-" * 42)
    with torch.no_grad():
        for mr in MASK_RATES:
            A_t, mask = mask_attn(A0s, mr)
            logits = denoiser(Xs, A_t, mask, t_vec=ts)
            tgt  = A0s.argmax(-1)
            m    = mask.bool()
            top1 = (logits.argmax(-1) == tgt)[m].float().mean().item()
            top5_i = logits.topk(5, -1).indices
            top5 = (top5_i == tgt.unsqueeze(-1).expand_as(top5_i)).any(-1)[m].float().mean().item()
            print(f"{mr:6.2f}  {top1:8.3f}  {top1/rand1:8.1f}x  {top5:8.3f}")

    # ── Demo ──────────────────────────────────────────────────────────────────
    if args.demo:
        print("\n--- Text-guided caption demo ---")
        test_start = N_IMAGES + VAL_IMAGES
        test_feats = patch_feats[test_start: test_start + 4]

        prompts = [
            "a dog playing in the grass",
            "a person surfing on the ocean waves",
            "a bowl of food on a table",
            "children playing in a park",
        ]
        captions = demo_text_guided(
            denoiser, guided_dec, proj,
            test_feats, prompts, tok, DEVICE,
        )
        for prompt, cap in zip(prompts, captions):
            print(f"  Prompt: {prompt!r}")
            print(f"  Caption: {cap}")
            print()


if __name__ == "__main__":
    main()
