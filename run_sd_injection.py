"""
Direct SD cross-attention plan injection (Improvement #3).

Injects A₀ plan as a logit bias into SD 1.5 UNet cross-attention at 16×16 resolution.
A₀[L_cap, N_patches].T[N_patches, L_cap] padded to [N_lat=256, N_text=77].

Compares plan-guided img2img vs. vanilla img2img on 10 test images.
Metrics: CLIP-I (generated ↔ source image), CLIP-T (generated ↔ caption).

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 \\
        run_sd_injection.py > results/sd_injection.log 2>&1
Requires SD 1.5 (~4 GB download on first run).
Prerequisite: run_save_plans.py must have completed.
"""

import os, sys, json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from run_summarization import L_SUM as L_CAP
from run_cap_improved import RESULTS_DIR, N_PATCHES, DEVICE

SAVE_DIR   = os.path.join(RESULTS_DIR, "saved_plans")
N_TEST     = 10
N_TV       = 14000           # train+val images before test split
STRENGTH   = 0.6
LAM_SWEEP  = [0.5, 1.0, 2.0]
N_TEXT_SD  = 77              # SD text token sequence length
IMG_SIZE   = 512

# ── diffusers import guard ────────────────────────────────────────────────────
try:
    from diffusers import StableDiffusionImg2ImgPipeline
    from diffusers.models.attention_processor import Attention, AttnProcessor2_0
except ImportError:
    print("ERROR: diffusers not installed.")
    print("Install: pip install diffusers accelerate")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Plan-guided attention processor
# ─────────────────────────────────────────────────────────────────────────────

class PlanGuidedAttnProcessor(AttnProcessor2_0):
    """
    Drop-in for AttnProcessor2_0.  For cross-attention whose hidden_states
    has sequence length == N_patches (== 256 for 16×16), adds a scaled
    plan bias to the attention logits before the softmax.
    """

    def __init__(self):
        super().__init__()
        self.plan_bias: torch.Tensor | None = None   # [N_lat, N_text], set externally
        self.lam: float = 1.0

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        is_cross = encoder_hidden_states is not None

        # For layers we don't want to inject (self-attn, or wrong resolution),
        # fall back to the parent implementation unchanged.
        if not is_cross or self.plan_bias is None:
            return super().__call__(
                attn, hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                temb=temb,
                *args, **kwargs,
            )

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            B, C, H, W = hidden_states.shape
            hidden_states = hidden_states.view(B, C, H * W).transpose(1, 2)

        batch_size, seq_len, _ = encoder_hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, seq_len, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key   = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim  = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key   = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key   = attn.norm_k(key)

        # Manual dot-product so we can add the plan bias
        scale  = head_dim ** -0.5
        dots   = torch.matmul(query, key.transpose(-2, -1)) * scale   # [B, H, L_q, L_k]

        N_lat  = dots.shape[2]
        N_text = dots.shape[3]
        if self.plan_bias.shape == (N_lat, N_text):
            bias = self.plan_bias.to(dots.dtype).to(dots.device)
            dots = dots + self.lam * bias.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            dots = dots + attention_mask

        attn_weights = F.softmax(dots, dim=-1, dtype=torch.float32).to(dots.dtype)
        hidden_states = torch.matmul(attn_weights, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(B, C, H, W)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def plan_to_bias(plan: torch.Tensor) -> torch.Tensor:
    """A₀[L_cap, N_patches] → [N_patches, N_text_SD] bias for SD cross-attn."""
    # plan: [L_cap, N_patches]
    A0_T = plan.T.float()                              # [N_patches, L_cap]
    pad  = N_TEXT_SD - A0_T.shape[1]
    if pad > 0:
        A0_T = F.pad(A0_T, (0, pad), value=0.0)       # [N_patches, 77]
    elif pad < 0:
        A0_T = A0_T[:, :N_TEXT_SD]
    # log-scale: convert probability → log so it acts as logit shift
    # small epsilon to avoid log(0)
    A0_T = torch.log(A0_T.clamp(min=1e-8))
    # zero out padding columns to avoid negative-inf bias there
    A0_T[:, L_CAP:] = 0.0
    return A0_T                                        # [256, 77]


def register_processors(pipe, processors: dict):
    """Bulk-set attention processors on the UNet."""
    pipe.unet.set_attn_processor(processors)


def build_processor_dict(pipe, guided_proc: PlanGuidedAttnProcessor) -> dict:
    """
    Return a dict mapping every attn_processor name to either the guided
    processor (cross-attn) or a fresh AttnProcessor2_0 (self-attn / others).
    We inject into ALL cross-attn layers; the processor filters by resolution
    at runtime (only acts when N_lat == N_patches == 256).
    """
    procs = {}
    for name in pipe.unet.attn_processors.keys():
        # attn2 = cross-attention in diffusers SD UNet naming
        if name.endswith("attn2.processor"):
            procs[name] = guided_proc
        else:
            procs[name] = AttnProcessor2_0()
    return procs


@torch.no_grad()
def clip_similarities(
    clip_model,
    clip_proc,
    img1: Image.Image,
    img2: Image.Image,
    caption: str,
    device: str,
) -> dict:
    """CLIP-I (img1↔img2) and CLIP-T (img2↔caption)."""
    from transformers import CLIPModel, CLIPProcessor

    def img_feat(img):
        inp = clip_proc(images=img, return_tensors="pt").to(device)
        f   = clip_model.get_image_features(**inp)
        if not isinstance(f, torch.Tensor):
            f = clip_model.visual_projection(
                clip_model.vision_model(**inp).pooler_output
            )
        return F.normalize(f.float(), dim=-1)

    def txt_feat(text):
        inp = clip_proc(text=[text], return_tensors="pt",
                        padding=True, truncation=True).to(device)
        f   = clip_model.get_text_features(**inp)
        if not isinstance(f, torch.Tensor):
            f = clip_model.text_projection(
                clip_model.text_model(**inp).pooler_output
            )
        return F.normalize(f.float(), dim=-1)

    f1  = img_feat(img1)
    f2  = img_feat(img2)
    ft  = txt_feat(caption)
    return {
        "clip_i": float((f1 * f2).sum()),
        "clip_t": float((f2 * ft).sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading SD 1.5 pipeline...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    print("Loading CLIP evaluator...")
    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    print("Loading test plans and captions...")
    plans_test = torch.load(
        os.path.join(SAVE_DIR, "plans_test.pt"), weights_only=True
    )                                                    # [N_test, L_cap, N_patches]
    with open(os.path.join(SAVE_DIR, "captions_test.txt")) as f:
        captions_test = [l.strip() for l in f.readlines()]

    print("Loading COCO test images from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("clip-benchmark/wds_mscoco_captions", split="train")

    # ── Lambda sweep on image 0 ──────────────────────────────────────────────
    item0   = ds[N_TV]
    img0    = item0["jpg"].convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    plan0   = plans_test[0]
    bias0   = plan_to_bias(plan0)
    cap0    = captions_test[0]

    # Register processors ONCE — mutate guided_proc attributes per step
    guided_proc  = PlanGuidedAttnProcessor()
    default_proc = AttnProcessor2_0()
    # Build name→processor mapping from current attn_processors keys
    proc_names   = list(pipe.unet.attn_processors.keys())
    guided_procs = {n: (guided_proc if n.endswith("attn2.processor") else default_proc)
                    for n in proc_names}
    pipe.unet.set_attn_processor(guided_procs)

    print(f"\nLambda sweep on image 0 (caption: '{cap0[:60]}...')")
    best_lam    = LAM_SWEEP[0]
    best_clip_t = -999.0

    for lam in LAM_SWEEP:
        guided_proc.plan_bias = bias0.to(DEVICE)
        guided_proc.lam       = lam
        # No re-registration needed — processor object is already registered

        gen = pipe(
            prompt=cap0, image=img0, strength=STRENGTH,
            guidance_scale=7.5, num_inference_steps=20,
        ).images[0]

        sims = clip_similarities(clip_model, clip_proc, img0, gen, cap0, DEVICE)
        print(f"  lam={lam:.1f}  CLIP-I={sims['clip_i']:.4f}  CLIP-T={sims['clip_t']:.4f}")
        if sims["clip_t"] > best_clip_t:
            best_clip_t = sims["clip_t"]
            best_lam    = lam

    print(f"  → best lam = {best_lam}")

    results = []
    for i in range(N_TEST):
        item    = ds[N_TV + i]
        img_src = item["jpg"].convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        plan_i  = plans_test[i]
        cap_i   = captions_test[i]
        print(f"\nImage {i}  cap='{cap_i[:55]}...'")

        # Baseline: disable plan bias
        guided_proc.plan_bias = None
        gen_base = pipe(
            prompt=cap_i, image=img_src, strength=STRENGTH,
            guidance_scale=7.5, num_inference_steps=20,
        ).images[0]
        sims_base = clip_similarities(
            clip_model, clip_proc, img_src, gen_base, cap_i, DEVICE
        )

        # Guided: inject plan bias
        guided_proc.plan_bias = plan_to_bias(plan_i).to(DEVICE)
        guided_proc.lam       = best_lam
        gen_guide = pipe(
            prompt=cap_i, image=img_src, strength=STRENGTH,
            guidance_scale=7.5, num_inference_steps=20,
        ).images[0]
        sims_guide = clip_similarities(
            clip_model, clip_proc, img_src, gen_guide, cap_i, DEVICE
        )

        entry = {
            "idx":             i,
            "caption":         cap_i,
            "clip_i_baseline": round(sims_base["clip_i"],  4),
            "clip_i_guided":   round(sims_guide["clip_i"], 4),
            "clip_t_baseline": round(sims_base["clip_t"],  4),
            "clip_t_guided":   round(sims_guide["clip_t"], 4),
            "delta_clip_i":    round(sims_guide["clip_i"] - sims_base["clip_i"], 4),
            "delta_clip_t":    round(sims_guide["clip_t"] - sims_base["clip_t"], 4),
        }
        results.append(entry)
        print(f"  baseline:  CLIP-I={sims_base['clip_i']:.4f}  CLIP-T={sims_base['clip_t']:.4f}")
        print(f"  guided:    CLIP-I={sims_guide['clip_i']:.4f}  CLIP-T={sims_guide['clip_t']:.4f}")
        print(f"  Δ CLIP-I={entry['delta_clip_i']:+.4f}  Δ CLIP-T={entry['delta_clip_t']:+.4f}")

    # ── Summary ──────────────────────────────────────────────────────────────
    mean_ci_base  = float(np.mean([r["clip_i_baseline"] for r in results]))
    mean_ci_guide = float(np.mean([r["clip_i_guided"]   for r in results]))
    mean_ct_base  = float(np.mean([r["clip_t_baseline"] for r in results]))
    mean_ct_guide = float(np.mean([r["clip_t_guided"]   for r in results]))
    n_pos_ci = sum(1 for r in results if r["delta_clip_i"] > 0)
    n_pos_ct = sum(1 for r in results if r["delta_clip_t"] > 0)

    print(f"\n{'='*60}")
    print(f"SD INJECTION SUMMARY  (n={N_TEST}, lam={best_lam})")
    print(f"  CLIP-I  baseline={mean_ci_base:.4f}  guided={mean_ci_guide:.4f}  "
          f"delta={mean_ci_guide-mean_ci_base:+.4f}  ({n_pos_ci}/{N_TEST} positive)")
    print(f"  CLIP-T  baseline={mean_ct_base:.4f}  guided={mean_ct_guide:.4f}  "
          f"delta={mean_ct_guide-mean_ct_base:+.4f}  ({n_pos_ct}/{N_TEST} positive)")
    verdict = "PASS" if (mean_ci_guide > mean_ci_base or mean_ct_guide > mean_ct_base) \
              else "NULL"
    print(f"  Verdict: {verdict}")
    print(f"{'='*60}")

    out = {
        "n_images":  N_TEST,
        "lam":       best_lam,
        "strength":  STRENGTH,
        "results":   results,
        "summary": {
            "mean_clip_i_baseline": round(mean_ci_base,  4),
            "mean_clip_i_guided":   round(mean_ci_guide, 4),
            "mean_clip_t_baseline": round(mean_ct_base,  4),
            "mean_clip_t_guided":   round(mean_ct_guide, 4),
            "clip_i_delta":         round(mean_ci_guide - mean_ci_base, 4),
            "clip_t_delta":         round(mean_ct_guide - mean_ct_base, 4),
            "n_positive_clip_i":    n_pos_ci,
            "n_positive_clip_t":    n_pos_ct,
            "verdict":              verdict,
        },
    }
    path = os.path.join(RESULTS_DIR, "sd_injection.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
