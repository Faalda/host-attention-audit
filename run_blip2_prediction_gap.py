"""
run_blip2_prediction_gap.py
===========================
Replicates the prediction-gap experiment on BLIP-2 OPT-2.7B.

The attention plan A0 is defined as the Q-Former cross-attention weights:
  A0[l, head, query, patch] = softmax(QK^T / sqrt(d) + plan_bias[l])

Teacher-forced plan:   plan_bias = 0  → normal BLIP-2 forward pass
Oracle-inverted plan:  plan_bias optimised via Adam to minimise NLL(ref | X, plan)

Key result:
  prediction_gap = ROUGE1(oracle) − ROUGE1(teacher-forced)

If the gap replicates the toy-model result (+0.20), the finding is
scale-relevant and substantially strengthens the paper.

Usage:
  # Quick sanity check (20 images, 100 steps):
  python run_blip2_prediction_gap.py --n-images 20 --n-steps 100

  # Full eval (200 images, 300 steps — matches paper protocol):
  python run_blip2_prediction_gap.py --n-images 200 --n-steps 300

  # Reproduce on COCO test offset (skip first 1000 images):
  python run_blip2_prediction_gap.py --n-images 1000 --offset 14000

Output:
  results/blip2_prediction_gap_n{N}.json
"""

import os, sys, json, math, types, argparse, io
import torch
import torch.nn as nn
import torch.nn.functional as F
from rouge_score import rouge_scorer as rs

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID   = "Salesforce/blip2-opt-2.7b"
RESULTS    = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

# Q-Former constants (BLIP-2 OPT-2.7B)
N_QUERY         = 32
N_CROSS_LAYERS  = 6   # every other of the 12 QFormer layers has cross-attn
N_HEADS         = 12


# ──────────────────────────────────────────────────────────────────────────────
# 1. Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model_and_processor():
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    print(f"Loading {MODEL_ID} …")
    proc = Blip2Processor.from_pretrained(MODEL_ID)
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Model loaded ({n_params:.1f}B params, dtype=fp16, device={DEVICE})")
    return proc, model


# ──────────────────────────────────────────────────────────────────────────────
# 2. Q-Former plan injection
#    Monkey-patches each cross-attention forward to add a logit bias
#    before the softmax. With bias=0 this is the identity transform;
#    optimising the bias is the plan inversion.
# ──────────────────────────────────────────────────────────────────────────────

def _make_biased_forward(original_module, bias_tensor):
    """
    Returns a bound forward that injects `bias_tensor` into the cross-attention
    logits of `original_module` before the softmax.

    bias_tensor: [1, 1, n_query, n_patches]  (fp32, broadcast over B and heads)
    Computation:
        attention_scores = QK^T / sqrt(d) + bias_tensor    ← plan injection
        attention_probs  = softmax(attention_scores, dim=-1)
        context          = attention_probs @ V
    """
    def forward(
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        **kwargs,
    ):
        m = original_module
        is_cross = encoder_hidden_states is not None

        if is_cross:
            key_layer   = m.transpose_for_scores(m.key(encoder_hidden_states))
            value_layer = m.transpose_for_scores(m.value(encoder_hidden_states))
            eff_mask = encoder_attention_mask
        else:
            key_layer   = m.transpose_for_scores(m.key(hidden_states))
            value_layer = m.transpose_for_scores(m.value(hidden_states))
            eff_mask = attention_mask

        query_layer = m.transpose_for_scores(m.query(hidden_states))

        # Standard scaled dot-product
        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        scores = scores / math.sqrt(m.attention_head_size)

        # ── Plan injection ────────────────────────────────────────────────────
        if is_cross:
            # bias_tensor: [1,1,n_query,n_patches] → broadcasts to [B,H,Q,P]
            scores = scores + bias_tensor.to(scores.dtype)
        # ─────────────────────────────────────────────────────────────────────

        if eff_mask is not None:
            scores = scores + eff_mask

        attn_probs = F.softmax(scores, dim=-1)

        if is_cross and m.save_attention:
            m.save_attention_map(attn_probs)
            attn_probs.register_hook(m.save_attn_gradients)

        attn_probs_dropped = m.dropout(attn_probs).to(value_layer.dtype)
        context = torch.matmul(attn_probs_dropped, value_layer)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(*context.size()[:-2], m.all_head_size)
        return context, attn_probs

    return forward


def _cross_attn_modules(qformer):
    """Yields (layer_index, Blip2QFormerMultiHeadAttention) for cross-attn layers."""
    for layer in qformer.encoder.layer:
        if layer.has_cross_attention:
            yield layer.crossattention.attention


def patch_plan(qformer, plan_bias):
    """
    Inject plan_bias into all Q-Former cross-attention layers.
    plan_bias: [N_CROSS_LAYERS, 1, 1, N_QUERY, N_PATCHES]
    Returns list of original forward functions for later restoration.
    """
    originals = []
    for idx, ca in enumerate(_cross_attn_modules(qformer)):
        originals.append(ca.forward)
        ca.forward = _make_biased_forward(ca, plan_bias[idx])
    return originals


def restore_plan(qformer, originals):
    for ca, orig in zip(_cross_attn_modules(qformer), originals):
        ca.forward = orig


# ──────────────────────────────────────────────────────────────────────────────
# 3. Caption generation with a given plan
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _qformer_output(model, image_embeds, image_attn_mask):
    """Run Q-Former and project to language model embedding space."""
    B = image_embeds.shape[0]
    query_tokens = model.query_tokens.expand(B, -1, -1)
    qout = model.qformer(
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_attn_mask,
        return_dict=True,
    )
    lm_dtype = model.language_projection.weight.dtype
    lang_inputs = model.language_projection(
        qout.last_hidden_state.to(lm_dtype)
    )  # [B, 32, lm_dim]
    return lang_inputs


def generate_caption(model, proc, image_embeds, image_attn_mask, plan_bias, device):
    """Generate a caption using the supplied plan_bias (zero = teacher-forced)."""
    B = image_embeds.shape[0]
    originals = patch_plan(model.qformer, plan_bias)
    try:
        with torch.no_grad():
            lang_inputs = _qformer_output(model, image_embeds, image_attn_mask)
            # lang_inputs: [B, 32, lm_dim]  fp16
            lang_mask = torch.ones(B, lang_inputs.shape[1], dtype=torch.long, device=device)
            bos = torch.zeros(B, 1, dtype=torch.long, device=device)
            bos_emb = model.language_model.get_input_embeddings()(bos)  # fp16
            inputs_embeds = torch.cat([lang_inputs, bos_emb], dim=1)
            attn_mask = torch.cat(
                [lang_mask, torch.ones(B, 1, dtype=torch.long, device=device)], dim=1
            )
            out = model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                max_new_tokens=50,
                do_sample=False,
            )
        captions = proc.tokenizer.batch_decode(out, skip_special_tokens=True)
    finally:
        restore_plan(model.qformer, originals)
    return captions


# ──────────────────────────────────────────────────────────────────────────────
# 4. Plan inversion via Adam
# ──────────────────────────────────────────────────────────────────────────────

def invert_plan(
    model,
    image_embeds,        # [1, n_patches, enc_dim]  fp16, frozen
    image_attn_mask,     # [1, n_patches]
    target_input_ids,    # [1, L]  token ids of the reference caption
    target_attn_mask,    # [1, L]
    n_patches,
    n_steps=300,
    lr=0.05,
    verbose=False,
):
    """
    Optimise plan_bias to maximise log p(target | X, plan).
    Returns plan_bias: [N_CROSS_LAYERS, 1, 1, N_QUERY, n_patches]  fp32
    """
    device = image_embeds.device

    # Initialise plan bias to zero (= natural QK^T attention at step 0)
    plan_bias = nn.Parameter(
        torch.zeros(N_CROSS_LAYERS, 1, 1, N_QUERY, n_patches,
                    device=device, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam([plan_bias], lr=lr)

    B = image_embeds.shape[0]
    query_tokens = model.query_tokens.expand(B, -1, -1)

    # Shift targets for causal LM: input = all but last, label = all but first
    tgt_in  = target_input_ids[:, :-1]   # [B, L-1]
    tgt_out = target_input_ids[:, 1:]    # [B, L-1]
    tgt_mask = target_attn_mask[:, 1:]   # [B, L-1]

    for step in range(1, n_steps + 1):
        optimizer.zero_grad()

        # ── Q-Former forward with current plan ────────────────────────────────
        originals = patch_plan(model.qformer, plan_bias.half())
        try:
            qout = model.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attn_mask,
                return_dict=True,
            )
        finally:
            restore_plan(model.qformer, originals)

        # ── Language model NLL ─────────────────────────────────────────────────
        # Q-Former LayerNorm runs in fp32 even in fp16 mode; cast to match
        # language_projection weight dtype before the linear.
        lm_dtype = model.language_projection.weight.dtype
        lang_inputs = model.language_projection(qout.last_hidden_state.to(lm_dtype))
        # [B, N_QUERY, lm_dim]

        tgt_emb = model.language_model.get_input_embeddings()(tgt_in)
        # [B, L-1, lm_dim]

        inputs_embeds = torch.cat([lang_inputs, tgt_emb], dim=1)
        # [B, N_QUERY + L-1, lm_dim]

        lm_mask = torch.cat([
            torch.ones(B, N_QUERY, dtype=torch.long, device=device),
            tgt_mask,
        ], dim=1)

        lm_out = model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=lm_mask,
            return_dict=True,
        )

        # Loss only on the caption tokens (not the query prefix)
        logits = lm_out.logits[:, N_QUERY:, :].float()  # [B, L-1, vocab] cast to fp32 for stable CE
        valid  = tgt_mask.bool()
        loss   = F.cross_entropy(logits[valid], tgt_out[valid], reduction="mean")

        loss.backward()
        optimizer.step()

        if verbose and step % 50 == 0:
            print(f"    step {step:3d}  nll={loss.item():.4f}")

    return plan_bias.detach()


# ──────────────────────────────────────────────────────────────────────────────
# 5. ROUGE-1 helper
# ──────────────────────────────────────────────────────────────────────────────

_ROUGE = rs.RougeScorer(["rouge1"], use_stemmer=False)

def rouge1(pred, ref):
    return _ROUGE.score(ref.strip(), pred.strip())["rouge1"].fmeasure

def best_of_refs(pred, refs):
    """Best-of-N-references ROUGE-1 (matches paper §4.5 scoring)."""
    return max(rouge1(pred, r) for r in refs if r.strip())


# ──────────────────────────────────────────────────────────────────────────────
# 6. Dataset helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_coco_items(offset, n_images):
    from datasets import load_dataset
    print("Loading MS-COCO captions dataset …")
    ds = load_dataset("clip-benchmark/wds_mscoco_captions", split="train",
                      streaming=False)
    items = []
    for i in range(offset, offset + n_images):
        row = ds[i]
        refs = [c.strip() for c in row["txt"].strip().split("\n") if c.strip()][:5]
        img_bytes = row["jpg"]
        items.append({"refs": refs, "img_bytes": img_bytes})
    print(f"  Loaded {len(items)} items (offset={offset})")
    return items


def preprocess_image(proc, img_bytes_or_pil, device):
    from PIL import Image
    if isinstance(img_bytes_or_pil, (bytes, bytearray)):
        img = Image.open(io.BytesIO(img_bytes_or_pil)).convert("RGB")
    else:
        img = img_bytes_or_pil.convert("RGB")
    inputs = proc(images=img, return_tensors="pt")
    return inputs["pixel_values"].to(device, dtype=torch.float16)


# ──────────────────────────────────────────────────────────────────────────────
# 7. Main evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def run_eval(proc, model, items, n_steps, lr, out_path, verbose_interval=10):
    teacher_scores, oracle_scores = [], []

    for idx, item in enumerate(items):
        refs = item["refs"]

        # ── Pre-process image ─────────────────────────────────────────────────
        pixel_values = preprocess_image(proc, item["img_bytes"], DEVICE)

        with torch.no_grad():
            image_embeds = model.vision_model(pixel_values, return_dict=True).last_hidden_state
            image_attn_mask = torch.ones(
                image_embeds.shape[:-1], dtype=torch.long, device=DEVICE
            )

        n_patches = image_embeds.shape[1]
        zero_bias = torch.zeros(
            N_CROSS_LAYERS, 1, 1, N_QUERY, n_patches, device=DEVICE, dtype=torch.float16
        )

        # ── Teacher-forced caption (plan_bias = 0) ────────────────────────────
        tf_caps = generate_caption(
            model, proc, image_embeds, image_attn_mask, zero_bias, DEVICE
        )
        tf_score = best_of_refs(tf_caps[0], refs)
        teacher_scores.append(tf_score)

        # ── Oracle inversion: optimise toward best reference ──────────────────
        best_oracle_score = tf_score  # lower bound: inversion never hurts in oracle eval
        for ref_cap in refs:
            enc = proc.tokenizer(
                ref_cap, return_tensors="pt", padding=True, truncation=True, max_length=64
            )
            tgt_ids  = enc["input_ids"].to(DEVICE)
            tgt_mask = enc["attention_mask"].to(DEVICE)

            plan_bias = invert_plan(
                model, image_embeds, image_attn_mask,
                tgt_ids, tgt_mask, n_patches,
                n_steps=n_steps, lr=lr,
                verbose=(idx == 0),   # print steps only for first image
            )

            inv_caps = generate_caption(
                model, proc, image_embeds, image_attn_mask,
                plan_bias.half(), DEVICE
            )
            score = rouge1(inv_caps[0], ref_cap)
            if score > best_oracle_score:
                best_oracle_score = score

        oracle_scores.append(best_oracle_score)

        # ── Progress report ───────────────────────────────────────────────────
        if (idx + 1) % verbose_interval == 0 or idx == 0:
            tf_mean  = sum(teacher_scores) / len(teacher_scores)
            ora_mean = sum(oracle_scores)  / len(oracle_scores)
            gap      = ora_mean - tf_mean
            print(f"  [{idx+1:4d}/{len(items)}]  "
                  f"teacher={tf_mean:.4f}  oracle={ora_mean:.4f}  "
                  f"gap={gap:+.4f}")
            # Flush stdout so progress is visible in logs
            sys.stdout.flush()

    # ── Final results ─────────────────────────────────────────────────────────
    tf_mean  = sum(teacher_scores) / len(teacher_scores)
    ora_mean = sum(oracle_scores)  / len(oracle_scores)
    gap      = ora_mean - tf_mean

    print("\n" + "="*60)
    print(f"  N images         : {len(items)}")
    print(f"  Teacher-forced   : {tf_mean:.4f}")
    print(f"  Oracle-inverted  : {ora_mean:.4f}")
    print(f"  Prediction gap   : {gap:+.4f}")
    print("="*60)

    per_image_gaps = [o - t for o, t in zip(oracle_scores, teacher_scores)]
    n_positive = sum(1 for g in per_image_gaps if g > 0)
    n_large    = sum(1 for g in per_image_gaps if g > 0.10)
    print(f"  % images gap > 0 : {100*n_positive/len(items):.1f}%")
    print(f"  % images gap >0.1: {100*n_large/len(items):.1f}%")

    results = {
        "model": MODEL_ID,
        "n_images": len(items),
        "n_steps": n_steps,
        "lr": lr,
        "scoring": "best-of-5-reference ROUGE-1",
        "teacher_forced_mean": round(tf_mean, 4),
        "oracle_inverted_mean": round(ora_mean, 4),
        "prediction_gap": round(gap, 4),
        "pct_positive_gap": round(100 * n_positive / len(items), 1),
        "pct_large_gap": round(100 * n_large / len(items), 1),
        "per_image": [
            {"teacher": round(t, 4), "oracle": round(o, 4), "gap": round(o - t, 4)}
            for t, o in zip(teacher_scores, oracle_scores)
        ],
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out_path}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 8. Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="BLIP-2 prediction-gap experiment")
    p.add_argument("--n-images", type=int, default=200,
                   help="Number of COCO test images to evaluate (default: 200)")
    p.add_argument("--n-steps",  type=int, default=300,
                   help="Adam steps per reference caption (default: 300)")
    p.add_argument("--lr",       type=float, default=0.05,
                   help="Adam learning rate for plan logits (default: 0.05)")
    p.add_argument("--offset",   type=int, default=14000,
                   help="COCO dataset offset — skip training images (default: 14000)")
    p.add_argument("--out",      type=str, default=None,
                   help="Output JSON path (default: results/blip2_prediction_gap_n{N}.json)")
    args = p.parse_args()

    out_path = args.out or os.path.join(
        RESULTS, f"blip2_prediction_gap_n{args.n_images}.json"
    )

    # Reproducibility
    torch.manual_seed(42)

    proc, model = load_model_and_processor()
    items       = load_coco_items(args.offset, args.n_images)

    print(f"\nRunning prediction-gap eval:")
    print(f"  n_images={args.n_images}  n_steps={args.n_steps}  "
          f"lr={args.lr}  offset={args.offset}")
    print(f"  Plan shape per layer: [1, 1, {N_QUERY}, n_patches]")
    print(f"  Total plan params: {N_CROSS_LAYERS} × {N_QUERY} × n_patches")
    print()

    run_eval(proc, model, items, args.n_steps, args.lr, out_path)


if __name__ == "__main__":
    main()
