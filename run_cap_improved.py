"""
Improved captioning with attention diffusion.

Fixes applied:
  1. Non-determinism: MSCOCO 5 captions/image → denoiser must use A_t
  2. Distribution shift: guided decoder trained with mixed oracle+noisy plans
  3. CFG: denoiser trained with/without X conditioning
  4. Soft gate: learnable plan influence (replaces fixed +log(plan))
  5. KL attention supervision: direct loss on decoder cross-attention vs plan
  6. Plan confidence weighting: high-entropy plans get lower supervision weight

Encoder options (set ENCODER below):
  "clip"    — CLIP ViT-B/32, 49 patches @ 768-dim
  "dinov2"  — DINOv2 ViT-B/14, 256 patches @ 768-dim (richer spatial features)
"""

import os, sys, json, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
from run_summarization import (
    Tokenizer,
    DEVICE, D_MODEL, N_HEADS, N_LAYERS, L_SUM as L_CAP,
    BATCH, LR, N_MP, eval_rouge,
)
from run_captioning import (
    PatchProjector, CaptionDecoder, GuidedCaptionDecoder,
    CocoCapDataset, collate_cap, A0CapDataset,
    train_free_cap, extract_A0_cap,
    N_PATCHES as N_PATCHES_CLIP, CLIP_DIM, N_CAP_PER,
)
from run_sum_masked import (
    mask_attn, MASK_RATES,
)

# ── Encoder choice ─────────────────────────────────────────────────────────
ENCODER = os.environ.get("ATTN_ENCODER", "dinov2")   # override: ATTN_ENCODER=clip

_ENCODER_CFG = {
    "clip":   {"n_patches": 49,  "dim": 768, "model_id": "openai/clip-vit-base-patch32"},
    "dinov2": {"n_patches": 256, "dim": 768, "model_id": "facebook/dinov2-base"},
}
N_PATCHES  = _ENCODER_CFG[ENCODER]["n_patches"]
ENCODER_DIM = _ENCODER_CFG[ENCODER]["dim"]

N_IMAGES    = 13000
VAL_IMAGES  = 1000
TEST_IMAGES = 1000
EPOCHS_FREE = 40
EPOCHS_DEN  = 60
EPOCHS_GUIDED = 50
K_LIST      = [1, 2, 4, 8]
SEEDS       = [42, 7, 13]
CFG_SCALE   = 2.0   # classifier-free guidance scale > 1 sharpens predictions
CFG_DROP_P  = 0.15  # probability of dropping X during denoiser training
KL_LAMBDA   = 0.1   # weight on KL attention supervision loss
SAL_LAMBDA  = 0.05  # weight on patch-salience grounding loss (fix colour hallucination)
RESULTS_DIR = "/home/faldabbas/myproject/attn_diffusion_gen/results"
CACHE_PATH  = os.path.join(RESULTS_DIR, f"{ENCODER}_patch_features_15k.pt")

# ── Patch feature extraction (CLIP or DINOv2) ─────────────────────────────
def extract_patch_features(images, encoder=ENCODER, batch_size=64, device=DEVICE):
    """
    Extract patch-level features for a list of PIL images.
    Returns [N_img, N_patches, dim] tensor.

    CLIP  ViT-B/32 → [N, 49,  768]   7×7  grid, semantic alignment
    DINOv2 ViT-B/14 → [N, 256, 768]  16×16 grid, richer spatial structure
    """
    if encoder == "clip":
        from transformers import CLIPProcessor, CLIPVisionModel
        print(f"  Loading CLIP ViT-B/32...")
        model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            inputs = proc(images=batch, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs)
            patches = out.last_hidden_state[:, 1:, :]   # skip CLS → [B,49,768]
            all_feats.append(patches.cpu().float())
            if (i // batch_size) % 5 == 0:
                print(f"    {i+len(batch)}/{len(images)}")
        return torch.cat(all_feats, dim=0)

    elif encoder == "dinov2":
        from transformers import AutoModel
        import numpy as np
        print(f"  Loading DINOv2 ViT-B/14 (256 patches per image)...")
        model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
        model.eval()

        # Standard ImageNet normalisation — same values DINOv2 was trained with
        _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        def _preprocess(imgs):
            """Resize short-side→256, center-crop 224×224, normalize. Pure PIL+numpy."""
            from PIL import Image as PILImage
            tensors = []
            for img in imgs:
                w, h = img.size
                scale = 256 / min(w, h)
                nw, nh = int(round(w * scale)), int(round(h * scale))
                img = img.resize((nw, nh), PILImage.BICUBIC)
                left, top = (nw - 224) // 2, (nh - 224) // 2
                img = img.crop((left, top, left + 224, top + 224))
                arr = np.array(img, dtype=np.float32) / 255.0   # [224,224,3]
                arr = (arr - _MEAN) / _STD
                tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
            return torch.stack(tensors)   # [B, 3, 224, 224]

        all_feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            pixel_values = _preprocess(batch).to(device)
            with torch.no_grad():
                out = model(pixel_values=pixel_values)
            # last_hidden_state: [B, 257, 768] — index 0 is CLS, 1..256 are patches
            patches = out.last_hidden_state[:, 1:, :]   # [B, 256, 768]
            all_feats.append(patches.cpu().float())
            if (i // batch_size) % 5 == 0:
                print(f"    {i+len(batch)}/{len(images)}")
        return torch.cat(all_feats, dim=0)

    else:
        raise ValueError(f"Unknown encoder: {encoder}")

# ── Fixes 4+5+6: Soft-gated guided decoder with KL supervision ────────────
class SoftGuidedDecoderLayer(nn.Module):
    """
    Guided decoder layer with three improvements over GuidedDecoderLayer:

    Fix 4 — Soft gate: plan influence is scaled by sigmoid(plan_gate) per layer.
      Initialised near 0 so training starts plan-free and gate opens as plan quality
      improves.  Replaces the hard +log(plan) of fixed weight 1.

    Fix 5 — Returns actual cross-attention weights (needed for KL supervision).

    Fix 6 — Plan confidence weighting is applied in the training loop, not here.
    """
    def __init__(self, D, H, dropout=0.1):
        super().__init__()
        self.sa    = nn.MultiheadAttention(D, H, dropout=dropout, batch_first=True)
        self.q     = nn.Linear(D, D)
        self.k     = nn.Linear(D, D)
        self.v     = nn.Linear(D, D)
        self.o     = nn.Linear(D, D)
        self.ff    = nn.Sequential(nn.Linear(D, D*4), nn.GELU(), nn.Linear(D*4, D))
        self.ln1   = nn.LayerNorm(D)
        self.ln2   = nn.LayerNorm(D)
        self.ln3   = nn.LayerNorm(D)
        self.drop  = nn.Dropout(dropout)
        self.scale = D ** -0.5
        # Learnable gate: starts at 0 → sigmoid(0)=0.5, but init negative so gate ≈ 0.1
        self.plan_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x, mem, attn_plan, tgt_mask=None, tgt_pad=None, ret_attn=False):
        B, L, D = x.shape
        xn = self.ln1(x)
        sa_out, _ = self.sa(xn, xn, xn, attn_mask=tgt_mask, key_padding_mask=tgt_pad)
        x = x + self.drop(sa_out)

        xn = self.ln2(x)
        Q = self.q(xn)
        K = self.k(mem)
        V = self.v(mem)
        logits = torch.bmm(Q, K.transpose(-1, -2)) * self.scale   # [B,L,N]

        # Fix 4: soft gate — plan influence scales from ~0 up to 1 as gate opens
        gate = torch.sigmoid(self.plan_gate)
        plan_bias = gate * torch.log(attn_plan.clamp(min=1e-6))
        logits = logits + plan_bias

        attn_w = F.softmax(logits, dim=-1)   # [B,L,N]
        ctx = torch.bmm(attn_w, V)
        x = x + self.drop(self.o(ctx))
        x = x + self.drop(self.ff(self.ln3(x)))
        return (x, attn_w) if ret_attn else x


class SoftGuidedCaptionDecoder(nn.Module):
    """Guided decoder using SoftGuidedDecoderLayer throughout."""
    def __init__(self, V, D, H, NL, dropout=0.1):
        super().__init__()
        self.embed  = nn.Embedding(V, D)
        self.pos    = nn.Embedding(L_CAP + 2, D)
        self.layers = nn.ModuleList([SoftGuidedDecoderLayer(D, H, dropout)
                                     for _ in range(NL)])
        self.head   = nn.Linear(D, V)

    def forward(self, src, tgt, attn_plan, ret_last_attn=False):
        B, L = tgt.shape
        x = self.embed(tgt) + self.pos(torch.arange(L, device=tgt.device))
        causal  = nn.Transformer.generate_square_subsequent_mask(L, device=tgt.device)
        tgt_pad = (tgt == 0)
        attn_w  = None
        for i, layer in enumerate(self.layers):
            last = ret_last_attn and (i == len(self.layers) - 1)
            out  = layer(x, src, attn_plan[:, :L, :], tgt_mask=causal,
                         tgt_pad=tgt_pad, ret_attn=last)
            if last: x, attn_w = out
            else:    x = out
        logits = self.head(x)
        return (logits, attn_w) if ret_last_attn else logits

    @torch.no_grad()
    def generate(self, src, attn_plan, max_len=L_CAP, bos=1, eos=2):
        B = src.shape[0]
        toks = torch.full((B, 1), bos, device=src.device, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len - 1):
            logits = self.forward(src, toks, attn_plan)
            next_t = logits[:, -1].argmax(-1)
            toks   = torch.cat([toks, next_t.unsqueeze(1)], dim=1)
            done  |= (next_t == eos)
            if done.all(): break
        return toks

    def gate_values(self):
        """Diagnostic: return the learned gate value per layer."""
        return [float(torch.sigmoid(l.plan_gate).item()) for l in self.layers]


# ── Fix 3: Denoiser with classifier-free guidance ──────────────────────────
class CFGDenoiser(nn.Module):
    """
    GNN denoiser with CFG support.
    During training: randomly drop X (replace with zeros) with probability CFG_DROP_P.
    During inference: interpolate between conditional and unconditional predictions.

    A_0_hat = uncond + cfg_scale * (cond - uncond)

    This sharpens the attention plan toward high-confidence predictions.
    """
    def __init__(self, D, n_src=N_PATCHES, n_tgt=L_CAP, n_mp=N_MP):
        super().__init__()
        self.row_proj  = nn.Linear(n_src, D)
        self.mask_emb  = nn.Embedding(2, D)
        self.msg_projs = nn.ModuleList([nn.Linear(D, D) for _ in range(n_mp)])
        self.mp_norms  = nn.ModuleList([nn.LayerNorm(D) for _ in range(n_mp)])
        self.pos_attn  = nn.MultiheadAttention(D, 4, batch_first=True)
        self.pos_norm  = nn.LayerNorm(D)
        self.q_proj    = nn.Linear(D, D)
        self.k_proj    = nn.Linear(D, D)
        # Null token for CFG unconditional path
        self.null_X    = nn.Parameter(torch.zeros(1, n_src, D))
        self.scale     = D ** -0.5
        self.n_src     = n_src

    def _forward_once(self, X_src, A_t, mask):
        B, L, N = A_t.shape
        h = F.gelu(self.row_proj(A_t)) + self.mask_emb(mask.long())
        for mp, ln in zip(self.msg_projs, self.mp_norms):
            msgs = torch.bmm(A_t, mp(X_src))
            h    = ln(h + msgs)
        h2, _ = self.pos_attn(h, h, h, need_weights=False)
        h = self.pos_norm(h + h2)
        Q = F.normalize(self.q_proj(h), dim=-1)
        K = self.k_proj(X_src)
        return torch.bmm(Q, K.transpose(-1, -2)) / self.scale   # [B,L,N]

    def forward(self, X_src, A_t, mask, drop_X=False):
        if drop_X:
            X_null = self.null_X.expand(X_src.shape[0], -1, -1)
            return self._forward_once(X_null, A_t, mask)
        return self._forward_once(X_src, A_t, mask)

    @torch.no_grad()
    def cfg_predict(self, X_src, A_t, mask, scale=CFG_SCALE):
        """Apply CFG: blend conditional and unconditional predictions."""
        cond   = self._forward_once(X_src, A_t, mask)
        uncond = self._forward_once(
            self.null_X.expand(X_src.shape[0], -1, -1), A_t, mask
        )
        return uncond + scale * (cond - uncond)

# ── Fix 3: MaskGIT inference with CFG ─────────────────────────────────────
@torch.no_grad()
def maskgit_cfg(denoiser, X_src, n_steps, cfg_scale=CFG_SCALE, device=None):
    if device is None: device = X_src.device
    B, N, D = X_src.shape
    A = torch.full((B, L_CAP, N), 1.0 / N, device=device)
    mask = torch.ones(B, L_CAP, device=device)

    n_per_step = max(1, L_CAP // n_steps)

    for step in range(n_steps):
        # CFG prediction
        logits = denoiser.cfg_predict(X_src, A, mask, scale=cfg_scale)
        probs  = F.softmax(logits, dim=-1)
        conf   = probs.max(-1).values.masked_fill(~mask.bool(), -1e9)

        # Unmask most confident positions
        n_unmask = min(n_per_step, int(mask.sum(-1).min().item()))
        if n_unmask == 0: break
        _, topk = conf.topk(n_unmask, dim=-1)

        pred = probs.argmax(-1)
        for b in range(B):
            for idx in topk[b]:
                if mask[b, idx] > 0:
                    A[b, idx] = 0
                    A[b, idx, pred[b, idx]] = 1.0
                    mask[b, idx] = 0

    # Final CFG pass for remaining
    final = denoiser.cfg_predict(X_src, A, mask, scale=cfg_scale)
    A_out = F.softmax(final, dim=-1)
    committed = (mask == 0).unsqueeze(-1).float()
    return committed * A + (1 - committed) * A_out

# ── Patch salience grounding ──────────────────────────────────────────────
def patch_salience(X):
    """
    Visual salience: how much each patch deviates from the image mean.
    Foreground objects (colour-bearing regions) deviate most from the
    homogeneous background → highest salience.

    Uses z-score normalisation before softmax so the result is robust to
    the absolute feature magnitude (DINOv2 norms vary across images).
    Temperature 0.5 gives a soft but peaked distribution (entropy ~2.6
    vs 5.55 uniform for 256 patches).
    """
    mean_f = X.mean(1, keepdim=True)                                    # [B,1,D]
    deviation = (X - mean_f).norm(dim=-1)                               # [B,N]
    dev_z = (deviation - deviation.mean(1, keepdim=True)) / \
            (deviation.std(1, keepdim=True) + 1e-8)                     # z-scores
    return F.softmax(dev_z * 0.5, dim=-1)                               # [B,N]

# ── Fix 1 + 2: Training ───────────────────────────────────────────────────
def train_cfg_denoiser(denoiser, A0_ds, n_epochs, device):
    """
    Train CFG denoiser: masked noise + random X dropout.
    This simultaneously handles:
    - Fix 1 (non-determinism): multiple A_0 per image forces A_t usage
    - Fix 3 (CFG): X dropped randomly to learn unconditional direction
    """
    loader = DataLoader(A0_ds, BATCH, shuffle=True)
    opt = torch.optim.Adam(denoiser.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    for ep in range(1, n_epochs + 1):
        denoiser.train()
        tot = 0; n = 0
        for A0, X, _ in loader:
            A0 = A0.to(device); X = X.to(device)
            mr = random.uniform(0.1, 0.95)
            A_t, mask = mask_attn(A0, mr)
            # CFG: drop X with probability CFG_DROP_P
            drop = (random.random() < CFG_DROP_P)
            logits = denoiser(X, A_t, mask, drop_X=drop)
            tgt = A0.argmax(-1)
            m = mask.bool()
            if not m.any(): continue
            loss_ce = F.cross_entropy(logits[m], tgt[m])

            # Salience grounding: push denoised plan toward visually salient patches
            # Prevents the denoiser from attending to uninformative background patches
            # (which causes colour hallucination when foreground colour is ignored)
            pred_plan = F.softmax(logits, dim=-1)          # [B, L, N]
            sal = patch_salience(X)                        # [B, N]
            sal_exp = sal.unsqueeze(1).expand_as(pred_plan) # [B, L, N]
            pred_m  = pred_plan[m]                         # [M, N]
            sal_m   = sal_exp[m]                           # [M, N]
            # KL(pred_plan ‖ salience): F.kl_div(log_target, source) = KL(source‖target)
            loss_sal = F.kl_div(torch.log(sal_m + 1e-8), pred_m, reduction='batchmean')

            loss = loss_ce + SAL_LAMBDA * loss_sal
            if torch.isnan(loss) or torch.isinf(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1
        sch.step()
        if ep % 10 == 0:
            print(f"    ep {ep:3d}  cfg_den_loss={tot/n:.4f}" if n > 0 else f"    ep {ep:3d}  cfg_den_loss=nan (all batches skipped)")

def train_guided_mixed(guided_dec, A0_ds, denoiser, n_epochs, device,
                       oracle_frac=0.5):
    """
    Fixes 2+4+5+6: mixed plans + soft gate + KL supervision + confidence weighting.

    Loss = L_ce  +  λ_kl * confidence_weight * KL(plan || decoder_attn)

    KL term: encourages decoder's actual cross-attention to align with the plan.
    Confidence weight: plan entropy determines how much to trust the plan —
      high-entropy (uniform) plan → weight≈0, confident plan → weight≈1.
    This prevents the supervision signal from pulling the decoder toward
    uninformative plans during early training.
    """
    loader = DataLoader(A0_ds, BATCH, shuffle=True)
    opt = torch.optim.Adam(guided_dec.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    denoiser.eval()
    max_H = math.log(N_PATCHES)   # entropy of uniform plan over N patches

    for ep in range(1, n_epochs + 1):
        guided_dec.train()
        tot_ce = 0; tot_kl = 0; tot_sal = 0; cnt = 0
        for A0, X, ids in loader:
            A0 = A0.to(device); X = X.to(device); ids = ids.to(device)

            if random.random() < oracle_frac:
                plan = F.softmax(A0 * 10, dim=-1)
            else:
                with torch.no_grad():
                    k = random.choice([1, 2, 4, 8])
                    A_sampled = maskgit_cfg(denoiser, X, n_steps=k, device=device)
                plan = F.softmax(A_sampled * 10, dim=-1)

            tgt_in  = ids[:, :-1]
            tgt_out = ids[:, 1:]
            pad_mask = (tgt_out == 0)

            # Forward with attention weights returned from last layer
            logits, attn_w = guided_dec(X, tgt_in, plan, ret_last_attn=True)

            # Fix 2: CE loss
            loss_ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                      tgt_out.reshape(-1), ignore_index=0)

            # Fix 5: KL(plan || attn_w) — plan is the target, attn_w should match
            # KL = Σ plan * log(plan / attn) = Σ plan*(log(plan)-log(attn))
            L = tgt_in.shape[1]
            plan_L = plan[:, :L, :]                         # [B,L,N]
            attn_w_safe = attn_w.clamp(min=1e-9)            # numerical safety
            kl = (plan_L * (torch.log(plan_L.clamp(min=1e-9)) -
                            torch.log(attn_w_safe))).sum(-1) # [B,L]

            # Fix 6: confidence weight = 1 - normalised plan entropy
            plan_H = -(plan_L * torch.log(plan_L.clamp(min=1e-9))).sum(-1) # [B,L]
            conf_w = (1.0 - plan_H / max_H).clamp(0, 1).detach()           # [B,L]

            # Average only over non-padding positions
            valid = ~pad_mask[:, :L]
            if valid.any():
                loss_kl = (conf_w * kl)[valid].mean()
            else:
                loss_kl = kl.mean() * 0

            # Salience grounding on decoder cross-attention (same principle as denoiser)
            sal = patch_salience(X)                               # [B, N]
            sal_exp_dec = sal.unsqueeze(1).expand_as(attn_w)     # [B, L-1, N]
            attn_valid  = attn_w[valid[:, :attn_w.shape[1]]]
            sal_valid   = sal_exp_dec[valid[:, :attn_w.shape[1]]]
            loss_sal = F.kl_div(torch.log(sal_valid + 1e-8),
                                attn_valid, reduction='batchmean')

            loss = loss_ce + KL_LAMBDA * loss_kl + SAL_LAMBDA * loss_sal
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(guided_dec.parameters(), 1.0)
            opt.step()
            tot_ce += loss_ce.item(); tot_kl += loss_kl.item()
            tot_sal += loss_sal.item(); cnt += 1
        sch.step()
        if ep % 10 == 0:
            gates = guided_dec.gate_values() if hasattr(guided_dec, 'gate_values') else []
            gate_str = f"  gates={[f'{g:.2f}' for g in gates]}" if gates else ""
            print(f"    ep {ep:3d}  ce={tot_ce/cnt:.4f}  kl={tot_kl/cnt:.4f}"
                  f"  sal={tot_sal/cnt:.4f}{gate_str}")

# ── Diversity metric ──────────────────────────────────────────────────────
@torch.no_grad()
def diversity_score(guided_dec, denoiser, X_src, tok, n_samples=8, device=DEVICE):
    """
    For each image, generate n_samples captions using different A_0 samples.
    Diversity = fraction of distinct unigrams across samples (Self-BLEU inverse).
    """
    B = X_src.shape[0]
    all_caps = [[] for _ in range(B)]

    for _ in range(n_samples):
        A_s = maskgit_cfg(denoiser, X_src, n_steps=4, device=device)
        plan = F.softmax(A_s * 10, dim=-1)
        toks = guided_dec.generate(X_src, plan)
        for b in range(B):
            cap = tok.dec(toks[b].cpu().tolist())
            all_caps[b].append(cap)

    # Distinct-1: unique unigrams / total unigrams (higher = more diverse)
    scores = []
    for caps in all_caps:
        all_words = " ".join(caps).split()
        unique = len(set(all_words))
        total  = len(all_words)
        scores.append(unique / total if total > 0 else 0)
    return float(np.mean(scores))

# ── Recovery test ─────────────────────────────────────────────────────────
@torch.no_grad()
def recovery_test(denoiser, A0_all, X_all, n_samples=512, device=DEVICE):
    idx = torch.randperm(len(A0_all))[:n_samples]
    A0  = A0_all[idx].to(device)
    X   = X_all[idx].to(device)
    results = {}
    for mr in MASK_RATES:
        A_t, mask = mask_attn(A0, mr)
        logits = denoiser(X, A_t, mask)
        pred = logits.argmax(-1)
        tgt  = A0.argmax(-1)
        m = mask.bool()
        if not m.any(): results[mr] = float('nan'); continue
        results[mr] = (pred == tgt)[m].float().mean().item()
    return results

# ── Evaluation ────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(proj, free_dec, guided_dec, denoiser, test_loader, tok, device):
    proj.eval(); free_dec.eval(); guided_dec.eval(); denoiser.eval()
    free_preds, oracle_free_preds, oracle_self_preds, refs = [], [], [], []
    k_preds = {k: [] for k in K_LIST}
    div_scores = []

    for batch in test_loader:
        patches = batch["patches"].to(device)
        ids     = batch["cap_ids"].to(device)
        X = proj(patches)
        B = X.shape[0]

        # Free baseline
        free_preds += [tok.dec(r) for r in free_dec.generate(X).cpu().tolist()]

        # Oracle (free): free decoder's attention → guided decoder
        # Misaligned: plan comes from a different model's representations
        _, attn_w_free = free_dec(X, ids[:, :-1], ret_last_attn=True)
        pad = torch.zeros(B, L_CAP - attn_w_free.shape[1], N_PATCHES, device=device)
        A_free = torch.cat([attn_w_free, pad], dim=1).detach()
        plan_free = F.softmax(A_free * 10, dim=-1)
        oracle_free_preds += [tok.dec(r) for r in guided_dec.generate(X, plan_free).cpu().tolist()]

        # Oracle (self): guided decoder runs once with uniform plan → extracts its
        # own cross-attention → uses that as the plan for a second pass.
        # Uniform plan biases cancel in softmax, so this is effectively unguided attn.
        uniform = torch.full((B, L_CAP, N_PATCHES), 1.0 / N_PATCHES, device=device)
        _, attn_w_self = guided_dec(X, ids[:, :-1], uniform, ret_last_attn=True)
        pad2 = torch.zeros(B, L_CAP - attn_w_self.shape[1], N_PATCHES, device=device)
        A_self = torch.cat([attn_w_self, pad2], dim=1).detach()
        plan_self = F.softmax(A_self * 10, dim=-1)
        oracle_self_preds += [tok.dec(r) for r in guided_dec.generate(X, plan_self).cpu().tolist()]

        refs += batch["ref"]

        # CFG denoiser K steps
        for k in K_LIST:
            A_mg = maskgit_cfg(denoiser, X, n_steps=k, device=device)
            plan = F.softmax(A_mg * 10, dim=-1)
            k_preds[k] += [tok.dec(r) for r in guided_dec.generate(X, plan).cpu().tolist()]

        # Diversity on first batch only (expensive)
        if len(div_scores) == 0:
            div = diversity_score(guided_dec, denoiser, X, tok, n_samples=8, device=device)
            div_scores.append(div)

    return {
        "free":         eval_rouge(free_preds,         refs),
        "oracle_free":  eval_rouge(oracle_free_preds,  refs),  # free dec attn → guided dec
        "oracle_self":  eval_rouge(oracle_self_preds,  refs),  # guided dec's own attn
        **{f"K{k}":     eval_rouge(k_preds[k],         refs) for k in K_LIST},
        "diversity":    float(np.mean(div_scores)),
    }

# ── Main ──────────────────────────────────────────────────────────────────
def run_seed(seed, patch_feats, caps_per_img, tok):
    print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    n_train = N_IMAGES; n_val = VAL_IMAGES; n_test = TEST_IMAGES

    train_feats = patch_feats[:n_train]
    tv_feats    = patch_feats[:n_train+n_val]
    test_feats  = patch_feats[n_train+n_val:n_train+n_val+n_test]

    train_caps = caps_per_img[:n_train]
    tv_caps    = caps_per_img[:n_train+n_val]
    test_caps  = caps_per_img[n_train+n_val:n_train+n_val+n_test]

    # N_PATCHES is set at module level based on ENCODER ("clip"=49, "dinov2"=256)
    proj        = PatchProjector(ENCODER_DIM, D_MODEL).to(DEVICE)
    free_dec    = CaptionDecoder(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS,
                                 n_src=N_PATCHES).to(DEVICE)
    denoiser    = CFGDenoiser(D_MODEL, N_PATCHES, L_CAP, N_MP).to(DEVICE)
    # Fix 4+5+6: SoftGuidedCaptionDecoder replaces GuidedCaptionDecoder
    guided_dec  = SoftGuidedCaptionDecoder(len(tok.vocab), D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)

    n_den = sum(p.numel() for p in denoiser.parameters())
    print(f"  Encoder={ENCODER}  N_PATCHES={N_PATCHES}  "
          f"CFGDenoiser={n_den:,} params  KL_LAMBDA={KL_LAMBDA}")

    # Phase 1: train image-captioning model
    train_ds = CocoCapDataset(train_feats, train_caps, tok, proj)
    train_loader = DataLoader(train_ds, BATCH, shuffle=True, collate_fn=collate_cap)
    print("  Phase 1: training captioning model...")
    train_free_cap(proj, free_dec, train_loader, EPOCHS_FREE, DEVICE)

    # Extract A_0 for ALL train+val captions (5 per image → non-determinism)
    print("  Extracting A_0 (5 captions × image)...")
    tv_ds = CocoCapDataset(tv_feats, tv_caps, tok, proj)
    tv_loader = DataLoader(tv_ds, BATCH, shuffle=False, collate_fn=collate_cap)
    A0_all, X_all, _, cap_ids_all = extract_A0_cap(proj, free_dec, tv_loader, DEVICE)
    A0_ds = A0CapDataset(A0_all, X_all, cap_ids_all)
    print(f"  A_0 dataset: {len(A0_ds)} pairs "
          f"({len(tv_caps)} images × 5 captions)")

    # Phase 2: train CFG denoiser
    print("  Phase 2: training CFG denoiser (masked + CFG dropout)...")
    train_cfg_denoiser(denoiser, A0_ds, EPOCHS_DEN, DEVICE)

    # Phase 3: train guided decoder with MIXED plans (fixes distribution shift)
    # Offload proj/free_dec to CPU — Phase 3 only needs denoiser + guided_dec on GPU
    proj.cpu(); free_dec.cpu()
    del train_loader, tv_loader
    torch.cuda.empty_cache()
    print("  Phase 3: training guided decoder (50% oracle / 50% denoiser)...")
    train_guided_mixed(guided_dec, A0_ds, denoiser, EPOCHS_GUIDED, DEVICE)
    proj.to(DEVICE); free_dec.to(DEVICE)

    # Recovery test
    A0_test, X_test, _, _ = extract_A0_cap(proj, free_dec,
        DataLoader(CocoCapDataset(test_feats, test_caps, tok, proj),
                   BATCH, shuffle=False, collate_fn=collate_cap), DEVICE)
    rec = recovery_test(denoiser, A0_test, X_test)

    # ROUGE + diversity
    test_ds     = CocoCapDataset(test_feats, test_caps, tok, proj)
    test_loader = DataLoader(test_ds, BATCH, shuffle=False, collate_fn=collate_cap)
    print("  Evaluating...")
    scores = evaluate(proj, free_dec, guided_dec, denoiser, test_loader, tok, DEVICE)

    print("\n  ROUGE-1 F1:")
    for k, v in scores.items():
        if k == "diversity":
            print(f"    {'diversity':20s}: {v:.4f}  (distinct-1, higher=more diverse)")
            continue
        delta = v - scores["free"]
        tag = f"  Δfree={delta:+.4f}" if k != "free" else "  ← baseline"
        print(f"    {k:20s}: {v:.4f}{tag}")

    # Learned gate values: show how much the decoder learned to trust the plan
    if hasattr(guided_dec, 'gate_values'):
        gates = guided_dec.gate_values()
        print(f"\n  Learned plan gates (per layer, 0=ignore plan, 1=full plan):")
        for i, g in enumerate(gates):
            bar = "█" * int(g * 20)
            print(f"    layer {i}: {g:.3f}  {bar}")

    print(f"\n  Recovery (random baseline = 1/{N_PATCHES} = {1/N_PATCHES:.3f}):")
    for mr, acc in sorted(rec.items()):
        print(f"   mask={mr:.2f}  acc={acc:.3f}  vs_random={acc - 1/N_PATCHES:+.3f}")

    return scores, rec


# ── Plan editing demo ─────────────────────────────────────────────────────
@torch.no_grad()
def plan_editing_demo(proj, free_dec, guided_dec, denoiser, test_feats,
                      test_caps, tok, device, n_images=6):
    """
    The key paper figure: shows that editing A_0 rows at test time
    predictably changes the output caption — something impossible with
    a standard seq2seq model.

    Three interventions per image:
      1. baseline   — free decoder caption (no plan)
      2. denoised   — guided decoder + K4 denoised plan
      3. suppressed — zero the top-3 most-attended patch rows → caption
                      loses the objects in those regions
      4. swapped    — inject A_0 plan from a different image → output
                      shifts toward the donor image's content

    Returns list of dicts with captions + attention maps for HTML render.
    """
    proj.eval(); free_dec.eval(); guided_dec.eval(); denoiser.eval()
    results = []

    for img_i in range(min(n_images, len(test_feats))):
        patches = test_feats[img_i].unsqueeze(0).to(device)   # [1, N, D_enc]
        X = proj(patches)                                       # [1, N, D]
        N = X.shape[1]                                          # actual patch count
        side = int(N ** 0.5)
        ref = test_caps[img_i][0] if test_caps[img_i] else ""

        # ── baseline (free decoder) ──────────────────────────────────────
        free_toks  = free_dec.generate(X)
        free_cap   = tok.dec(free_toks[0].cpu().tolist())

        # ── extract oracle A_0 from free decoder ────────────────────────
        dummy_ids = free_toks[:, :-1]
        _, attn_w = free_dec(X, dummy_ids, ret_last_attn=True)   # [1, L-1, N]
        L1 = attn_w.shape[1]
        pad = torch.zeros(1, L_CAP - L1, N, device=device)
        A0 = torch.cat([attn_w, pad], dim=1).detach()            # [1, L, N]

        # ── denoised plan (K=4) ──────────────────────────────────────────
        A_den  = maskgit_cfg(denoiser, X, n_steps=4, device=device)
        plan_d = F.softmax(A_den * 10, dim=-1)
        den_cap = tok.dec(guided_dec.generate(X, plan_d)[0].cpu().tolist())

        # ── intervention 1: suppress top-attended patches ────────────────
        # Find the 3 patches with highest total attention across all rows
        patch_importance = A0[0].sum(0)                          # [N]
        top3 = patch_importance.topk(3).indices
        A_sup = A0.clone()
        A_sup[0, :, top3] = 0
        row_sums = A_sup[0].sum(-1, keepdim=True).clamp(min=1e-9)
        A_sup[0] = A_sup[0] / row_sums                          # renormalize
        plan_sup = F.softmax(A_sup * 10, dim=-1)
        sup_cap = tok.dec(guided_dec.generate(X, plan_sup)[0].cpu().tolist())

        # ── intervention 2: swap plan from another image ─────────────────
        donor_i   = (img_i + 1) % n_images
        d_patches = test_feats[donor_i].unsqueeze(0).to(device)
        d_X       = proj(d_patches)
        d_toks    = free_dec.generate(d_X)
        _, d_attn = free_dec(d_X, d_toks[:, :-1], ret_last_attn=True)
        d_L1 = d_attn.shape[1]
        d_pad = torch.zeros(1, L_CAP - d_L1, N, device=device)
        A_donor = torch.cat([d_attn, d_pad], dim=1).detach()
        plan_sw = F.softmax(A_donor * 10, dim=-1)
        swap_cap = tok.dec(guided_dec.generate(X, plan_sw)[0].cpu().tolist())

        # ── store attention map as 2D grid (for heatmap) ─────────────────
        # A0[0]: [L, N] → take first 8 non-padding rows, reshape each to [side, side]
        words_free = free_cap.replace("<BOS>","").replace("<EOS>","").split()
        attn_maps = []
        for row_i in range(min(8, L1)):
            am = A0[0, row_i].cpu().reshape(side, side).tolist()
            attn_maps.append({
                "word": words_free[row_i] if row_i < len(words_free) else f"pos{row_i}",
                "map": am,
            })

        results.append({
            "image_idx":  img_i,
            "reference":  ref,
            "free":       free_cap,
            "denoised":   den_cap,
            "suppressed": sup_cap,
            "swapped":    swap_cap,
            "donor_idx":  donor_i,
            "donor_free": tok.dec(free_dec.generate(d_X)[0].cpu().tolist()),
            "attn_maps":  attn_maps,
            "top3_patches": top3.cpu().tolist(),
            "grid_side":  side,
        })

        print(f"\n  Image {img_i}:")
        print(f"    ref:        {ref[:80]}")
        print(f"    free:       {free_cap[:80]}")
        print(f"    denoised:   {den_cap[:80]}")
        print(f"    suppressed: {sup_cap[:80]}  [patched {top3.cpu().tolist()}]")
        print(f"    swapped:    {swap_cap[:80]}  [plan from img {donor_i}]")

    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    from datasets import load_dataset
    print("Loading MSCOCO train split (82K images, using first 15K)...")
    ds = load_dataset("clip-benchmark/wds_mscoco_captions", split="train")
    total = N_IMAGES + VAL_IMAGES + TEST_IMAGES

    images, caps_per_img = [], []
    for item in ds.select(range(total)):
        img = item["jpg"]
        if img.mode != "RGB": img = img.convert("RGB")
        images.append(img)
        caps = [c.strip() for c in item["txt"].strip().split("\n") if c.strip()]
        caps_per_img.append(caps[:N_CAP_PER])

    # Load cached patch features (encoder-specific cache file)
    if os.path.exists(CACHE_PATH):
        print(f"  Loading cached {ENCODER} patch features from {CACHE_PATH}...")
        patch_feats = torch.load(CACHE_PATH, weights_only=True)
    else:
        print(f"  Extracting {ENCODER} patch features (one-time ~5 min)...")
        patch_feats = extract_patch_features(images, encoder=ENCODER, device=DEVICE)
        torch.save(patch_feats, CACHE_PATH)
        print(f"  Saved to {CACHE_PATH}")
    print(f"  Patch features: {patch_feats.shape}  "
          f"(expected [N, {N_PATCHES}, {ENCODER_DIM}])")

    # Vocabulary from all captions
    tok = Tokenizer()
    tok.build([cap for caps in caps_per_img for cap in caps], max_vocab=6000)
    print(f"  Vocab: {len(tok.vocab)}")

    all_scores, all_recs = [], []
    for seed in SEEDS:
        scores, rec = run_seed(seed, patch_feats, caps_per_img, tok)
        all_scores.append(scores); all_recs.append(rec)

    keys = ["free", "oracle_free", "oracle_self"] + [f"K{k}" for k in K_LIST]
    free_mean = float(np.mean([s["free"] for s in all_scores]))
    print(f"\n{'='*60}\nSUMMARY — {len(SEEDS)} seeds\n{'='*60}")
    for key in keys:
        m = float(np.mean([s[key] for s in all_scores]))
        std = float(np.std([s[key] for s in all_scores]))
        if key == "free":
            tag = "  ← baseline"
        elif key == "oracle_free":
            tag = f"  Δfree={m-free_mean:+.4f}  (free dec plan → guided dec)"
        elif key == "oracle_self":
            tag = f"  Δfree={m-free_mean:+.4f}  (guided dec's own plan)"
        else:
            tag = f"  Δfree={m-free_mean:+.4f}"
        print(f"  {key:20s}: {m:.4f} ± {std:.4f}{tag}")
    div_m = float(np.mean([s["diversity"] for s in all_scores]))
    print(f"  {'diversity':20s}: {div_m:.4f}  (standard seq2seq = 0.0, always same output)")

    print(f"\n  Recovery (random = 1/{N_PATCHES} = {1/N_PATCHES:.4f}):")
    for mr in MASK_RATES:
        vals = [r[mr] for r in all_recs if not math.isnan(r[mr])]
        m = float(np.mean(vals)) if vals else float('nan')
        print(f"   mask={mr:.2f}  acc={m:.3f}  vs_random={m - 1/N_PATCHES:+.3f}")

    out = {
        "config": {"encoder": ENCODER, "CFG_SCALE": CFG_SCALE, "oracle_frac": 0.5,
                   "n_patches": N_PATCHES, "n_cap_per": N_CAP_PER,
                   "n_train": N_IMAGES, "sal_lambda": SAL_LAMBDA},
        "scores": all_scores,
        "recovery": [{str(k): v for k, v in r.items()} for r in all_recs],
    }
    path = os.path.join(RESULTS_DIR, f"cap_improved_{ENCODER}_15k_sal.json")
    with open(path, "w") as f: json.dump(out, f, indent=2)
    print(f"\nFull results → {path}")

    # ── Plan editing demo (run on last seed's trained model) ──────────────
    # Re-run last seed to get trained models in scope
    print("\n\nRunning plan editing demo on last seed...")
    seed = SEEDS[-1]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    n_train = N_IMAGES; n_val = VAL_IMAGES
    test_feats_demo = patch_feats[n_train+n_val:n_train+n_val+TEST_IMAGES]
    test_caps_demo  = caps_per_img[n_train+n_val:n_train+n_val+TEST_IMAGES]

    # Rebuild models with same seed (use stored checkpoint from last run_seed call)
    tok2 = tok  # same vocabulary
    proj2       = PatchProjector(ENCODER_DIM, D_MODEL).to(DEVICE)
    free_dec2   = CaptionDecoder(len(tok2.vocab), D_MODEL, N_HEADS, N_LAYERS,
                                 n_src=N_PATCHES).to(DEVICE)
    denoiser2   = CFGDenoiser(D_MODEL, N_PATCHES, L_CAP, N_MP).to(DEVICE)
    guided_dec2 = SoftGuidedCaptionDecoder(len(tok2.vocab), D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)

    # Train quickly for demo (fewer epochs — just enough to get coherent outputs)
    train_ds2 = CocoCapDataset(patch_feats[:n_train], caps_per_img[:n_train], tok2, proj2)
    train_loader2 = DataLoader(train_ds2, BATCH, shuffle=True, collate_fn=collate_cap)
    print("  Training demo models (20 epochs)...")
    train_free_cap(proj2, free_dec2, train_loader2, 20, DEVICE)
    tv_ds2 = CocoCapDataset(patch_feats[:n_train+n_val],
                             caps_per_img[:n_train+n_val], tok2, proj2)
    tv_loader2 = DataLoader(tv_ds2, BATCH, shuffle=False, collate_fn=collate_cap)
    A0_d, X_d, _, ids_d = extract_A0_cap(proj2, free_dec2, tv_loader2, DEVICE)
    A0_ds2 = A0CapDataset(A0_d, X_d, ids_d)
    train_cfg_denoiser(denoiser2, A0_ds2, 20, DEVICE)
    train_guided_mixed(guided_dec2, A0_ds2, denoiser2, 20, DEVICE)

    demo_results = plan_editing_demo(
        proj2, free_dec2, guided_dec2, denoiser2,
        test_feats_demo, test_caps_demo, tok2, DEVICE, n_images=6
    )
    demo_path = os.path.join(RESULTS_DIR, f"plan_editing_demo_{ENCODER}.json")
    with open(demo_path, "w") as f: json.dump(demo_results, f, indent=2)
    print(f"\nPlan editing demo → {demo_path}")

if __name__ == "__main__":
    main()
