"""
run_rev_blip2_protocol.py — revision experiment E2 (R1-W2/F2/F3, R2-Q1, R3-W1).

Same symmetric, un-floored protocol as run_rev_inversion_protocol.py, on BLIP-2 OPT-2.7B.
Per image: native caption (bias = 0, i.e. BLIP-2's own image-conditioned Q-Former attention),
inversions toward references r_0 and r_1 (scored single-ref and held-out against the other
four), and an inversion toward a reference of a different image (shuffled target).

Parameterisations:
  full  : per-layer logit bias [6 layers, 32 queries, 257 patches] = 49,344 params, shared over
          the 12 heads (the paper's original setting).
  rank2 : bias = U V with U [6,32,2], V [6,2,257]  = 3,468 params.
  query : bias = 0; optimise an additive offset to the 32 learned query embeddings
          (32 x 768 = 24,576 params). Non-routing control: steering what the queries ask
          for rather than where they look.
Output format matches the small-model dump so run_rev_eval_metrics.py scores both.
"""
import os, sys, json, argparse, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from run_blip2_prediction_gap import (load_model_and_processor, load_coco_items, preprocess_image,
                                      N_CROSS_LAYERS, N_QUERY)


# transformers>=5 removed transpose_for_scores; eager cross-attention with an additive logit bias.
def _biased_forward(m, bias_l):
    d = m.attention_head_size

    def forward(hidden_states, attention_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, **kwargs):
        is_cross = encoder_hidden_states is not None
        cur = encoder_hidden_states if is_cross else hidden_states
        mask = encoder_attention_mask if is_cross else attention_mask
        q = m.query(hidden_states).view(*hidden_states.shape[:-1], -1, d).transpose(1, 2)
        k = m.key(cur).view(*cur.shape[:-1], -1, d).transpose(1, 2)
        v = m.value(cur).view(*cur.shape[:-1], -1, d).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)).float() * m.scaling
        if is_cross:
            scores = scores + bias_l.float()
        if mask is not None and torch.is_floating_point(mask):
            scores = scores + mask.float()
        probs = F.softmax(scores, -1).to(v.dtype)
        out = torch.matmul(probs, v).transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
        return out, probs
    return forward


def _cross_attn_modules(qformer):
    for layer in qformer.encoder.layer:
        if getattr(layer, "has_cross_attention", False):
            yield layer.crossattention.attention


def patch_plan(qformer, bias):
    originals = []
    for idx, ca in enumerate(_cross_attn_modules(qformer)):
        originals.append(ca.forward)
        ca.forward = _biased_forward(ca, bias[idx])
    return originals


def restore_plan(qformer, originals):
    for ca, orig in zip(_cross_attn_modules(qformer), originals):
        ca.forward = orig

DEVICE = "cuda"
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "revision")
os.makedirs(OUT_DIR, exist_ok=True)


def lm_inputs(model, image_embeds, bias, query_delta):
    B = image_embeds.shape[0]
    q = model.query_tokens.expand(B, -1, -1)
    if query_delta is not None:
        q = q + query_delta.to(q.dtype)
    originals = patch_plan(model.qformer, bias)
    try:
        qout = model.qformer(query_embeds=q, encoder_hidden_states=image_embeds,
                             encoder_attention_mask=torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=DEVICE),
                             return_dict=True)
    finally:
        restore_plan(model.qformer, originals)
    lm_dtype = model.language_projection.weight.dtype
    return model.language_projection(qout.last_hidden_state.to(lm_dtype))


@torch.no_grad()
def generate(model, proc, image_embeds, bias, query_delta=None, bos_id=None):
    B = image_embeds.shape[0]
    li = lm_inputs(model, image_embeds, bias, query_delta)
    bos_id = model.config.text_config.bos_token_id if bos_id is None else bos_id
    bos = torch.full((B, 1), bos_id, dtype=torch.long, device=DEVICE)
    emb = torch.cat([li, model.language_model.get_input_embeddings()(bos)], 1)
    out = model.language_model.generate(inputs_embeds=emb, attention_mask=torch.ones(emb.shape[:2], dtype=torch.long, device=DEVICE),
                                        max_new_tokens=50, do_sample=False)
    return [c.strip() for c in proc.tokenizer.batch_decode(out, skip_special_tokens=True)]


def build_bias(variant, params, B, P):
    if variant == "full":
        return params["bias"]
    if variant == "rank2":
        return (params["U"] @ params["V"])                       # [6,B,1,Q,P]
    return torch.zeros(N_CROSS_LAYERS, B, 1, N_QUERY, P, device=DEVICE)


def invert(model, proc, image_embeds, targets, variant, n_steps, lr):
    B, P = image_embeds.shape[0], image_embeds.shape[1]
    params = {}
    if variant == "full":
        params["bias"] = nn.Parameter(torch.zeros(N_CROSS_LAYERS, B, 1, N_QUERY, P, device=DEVICE))
    elif variant == "rank2":
        params["U"] = nn.Parameter(0.01 * torch.randn(N_CROSS_LAYERS, B, 1, N_QUERY, 2, device=DEVICE))
        params["V"] = nn.Parameter(torch.zeros(N_CROSS_LAYERS, B, 1, 2, P, device=DEVICE))
    else:
        params["q"] = nn.Parameter(torch.zeros(B, N_QUERY, model.query_tokens.shape[-1], device=DEVICE))
    opt = torch.optim.Adam(list(params.values()), lr=lr)
    enc = proc.tokenizer(targets, return_tensors="pt", padding=True, truncation=True, max_length=64)
    ids, am = enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)
    tgt_in, tgt_out, tgt_m = ids[:, :-1], ids[:, 1:], am[:, 1:]
    for _ in range(n_steps):
        bias = build_bias(variant, params, B, P)
        li = lm_inputs(model, image_embeds, bias.half(), params.get("q"))
        emb = torch.cat([li, model.language_model.get_input_embeddings()(tgt_in)], 1)
        mask = torch.cat([torch.ones(B, N_QUERY, dtype=torch.long, device=DEVICE), am[:, :-1]], 1)
        logits = model.language_model(inputs_embeds=emb, attention_mask=mask, return_dict=True).logits[:, N_QUERY:].float()
        valid = tgt_m.bool()
        loss = F.cross_entropy(logits[valid], tgt_out[valid])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        bias = build_bias(variant, params, B, P).half()
        caps = generate(model, proc, image_embeds, bias, params.get("q"))
        nb = bias.float().abs().mean().item()
    return caps, nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=200)
    ap.add_argument("--offset", type=int, default=14000)
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--refs-per-image", type=int, default=2)
    ap.add_argument("--variants", nargs="+", default=["full", "rank2", "query"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    proc, model = load_model_and_processor()
    items = load_coco_items(args.offset, args.n_images)
    n = len(items)
    refs = [it["refs"] for it in items]
    rng = random.Random(args.seed)
    perm = list(range(n))
    while any(p == i for i, p in enumerate(perm)):
        rng.shuffle(perm)
    mism = [rng.choice(refs[perm[i]]) for i in range(n)]

    def embeds(idx):
        pv = torch.cat([preprocess_image(proc, items[i]["img_bytes"], DEVICE) for i in idx])
        with torch.no_grad():
            return model.vision_model(pv, return_dict=True).last_hidden_state

    base, base_bos0 = [], []
    for s in range(0, n, args.batch):
        idx = list(range(s, min(n, s + args.batch)))
        E = embeds(idx)
        zb = torch.zeros(N_CROSS_LAYERS, 1, 1, N_QUERY, E.shape[1], device=DEVICE).half()
        base += generate(model, proc, E, zb)
        base_bos0 += generate(model, proc, E, zb, bos_id=0)   # the original script's start token
    print("native captions done", flush=True)

    jobs = [(i, "true", k, refs[i][k]) for i in range(n) for k in range(min(args.refs_per_image, len(refs[i])))]
    jobs += [(i, "mismatch", -1, mism[i]) for i in range(n)]
    records = [{"image": i, "refs": refs[i], "baseline": base[i], "baseline_bos0": base_bos0[i], "mismatch_source": perm[i], "mismatch_target": mism[i],
                "variants": {v: {"true": [None] * len(refs[i]), "mismatch": None, "bias_abs_mean": []} for v in args.variants}}
               for i in range(n)]
    out_path = os.path.join(OUT_DIR, f"blip2_protocol_n{n}_s{args.seed}.json")
    t0 = time.time()
    for v in args.variants:
        for s in range(0, len(jobs), args.batch):
            chunk = jobs[s:s + args.batch]
            E = embeds([j[0] for j in chunk])
            caps, nb = invert(model, proc, E, [j[3] for j in chunk], v, args.n_steps, args.lr)
            for (i, kind, k, _), c in zip(chunk, caps):
                rv = records[i]["variants"][v]
                if kind == "true":
                    rv["true"][k] = c
                else:
                    rv["mismatch"] = c
                rv["bias_abs_mean"].append(nb)
            print(f"[{v}] {min(s + args.batch, len(jobs))}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
            if (s // args.batch) % 10 == 0:
                json.dump({"n_images": n, "variants": args.variants, "partial": True, "records": records}, open(out_path, "w"))
    json.dump({"n_images": n, "n_steps": args.n_steps, "lr": args.lr, "variants": args.variants,
               "param_counts": {"full": N_CROSS_LAYERS * N_QUERY * 257, "rank2": N_CROSS_LAYERS * 2 * (N_QUERY + 257),
                                "query": N_QUERY * 768},
               "records": records}, open(out_path, "w"))
    print("saved", out_path)


if __name__ == "__main__":
    main()
