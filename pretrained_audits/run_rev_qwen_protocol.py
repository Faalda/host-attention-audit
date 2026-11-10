"""
run_rev_qwen_protocol.py — fourth-architecture test of the routing-vs-steering protocol (revision).

Qwen2.5-VL-7B is a decoder-only VLM: caption tokens attend to image tokens through the language model's
own self-attention (no cross-attention, no Q-Former). Two interfaces with exactly the same number of
free variables (7,168), optimised per image toward a target caption:
  route : additive bias on the attention logits from text queries (positions after the image) to the
          256 image-token keys, one value per layer and image token (28 x 256), shared over heads/queries.
  resid : additive steering vector on the residual stream of text positions at two middle layers (2 x 3584).
Targets per image: its first reference (true), a reference of another image (shuffled), and its own
reference with one COCO object swapped for an absent one (swap injection).
Output format matches the other protocol dumps (+ "swap" fields).
"""
import os, sys, json, argparse, random, time, io
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for inversion. imports
from run_blip2_prediction_gap import load_coco_items
from inversion.run_rev_eval_metrics import objects
from inversion.run_rev_swap_injection import make_target

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
PROMPT = "Describe this image in one short sentence."
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "revision")
DEVICE = "cuda"
STATE = {"bias": None, "img_pos": None, "img_end": None}


def routing_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import repeat_kv
    k = repeat_kv(key, module.num_key_value_groups); v = repeat_kv(value, module.num_key_value_groups)
    # fp32 logits: the bf16 eager path of this model degenerates during generation
    w = torch.matmul(query.float(), k.float().transpose(2, 3)) * scaling
    if attention_mask is not None:
        w = w + attention_mask[:, :, :, : k.shape[-2]].float()
    if STATE["bias"] is not None:
        B, _, Q, K = w.shape
        b = torch.zeros(B, K, device=w.device, dtype=torch.float32)
        b.scatter_(1, STATE["img_pos"], STATE["bias"][module.layer_idx])
        q_abs = torch.arange(K - Q, K, device=w.device)
        rows = (q_abs.unsqueeze(0) > STATE["img_end"].unsqueeze(1)).float()          # [B,Q] text queries only
        w = w + (rows.unsqueeze(-1) * b.unsqueeze(1)).unsqueeze(1)
    w = F.softmax(w, dim=-1)
    out = torch.matmul(w, v.float()).to(query.dtype).transpose(1, 2).contiguous()
    return out, None


class Runner:
    def __init__(self, res):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AttentionInterface, AttentionMaskInterface
        from transformers.masking_utils import eager_mask
        AttentionInterface.register("routing_bias", routing_attention)
        AttentionMaskInterface.register("routing_bias", eager_mask)
        self.proc = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=res * res, max_pixels=res * res)
        self.proc.tokenizer.padding_side = "left"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation={"text_config": "routing_bias", "vision_config": "sdpa", "": "sdpa"}
        ).to(DEVICE).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.layers = self.model.model.language_model.layers
        self.img_tok = self.model.config.image_token_id
        self.hooks = []
        self.res = res

    def batch(self, imgs, targets=None):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        txt = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        texts = [txt + (t.strip() + "<|im_end|>" if targets else "") for t in (targets or [None] * len(imgs))]
        imgs = [i.resize((self.res, self.res)) for i in imgs]
        enc = self.proc(text=texts, images=imgs, return_tensors="pt", padding=True).to(DEVICE)
        n_prompt = None
        if targets:
            p = self.proc(text=[txt] * len(imgs), images=imgs, return_tensors="pt", padding=True)
            n_prompt = p["attention_mask"].sum(1).to(DEVICE)       # prompt length per sample (unpadded)
        ids = enc["input_ids"]
        pos = [(r == self.img_tok).nonzero().squeeze(1) for r in ids]
        STATE["img_pos"] = torch.stack(pos)
        STATE["img_end"] = STATE["img_pos"].max(1).values
        return enc, n_prompt

    def set_resid(self, vec, layers):
        self.clear_resid()
        for j, li in enumerate(layers):
            def hook(mod, inp, out, j=j):
                h = out[0] if isinstance(out, tuple) else out
                T = h.shape[1]; K_total = self._cur_len
                q_abs = torch.arange(K_total - T, K_total, device=h.device)
                rows = (q_abs.unsqueeze(0) > STATE["img_end"].unsqueeze(1)).to(h.dtype).unsqueeze(-1)
                h = h + rows * vec[j].unsqueeze(1).to(h.dtype)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            self.hooks.append(self.layers[li].register_forward_hook(hook))

    def clear_resid(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    @torch.no_grad()
    def generate(self, imgs, bias=None, resid=None, resid_layers=None):
        enc, _ = self.batch(imgs)
        STATE["bias"] = bias
        if resid is not None:
            self.set_resid(resid, resid_layers)
        base_len = enc["input_ids"].shape[1]
        # track absolute length for hooks during cached generation
        orig_fwd = self.model.model.language_model.forward
        runner = self
        def fwd(*a, **k):
            pkv = k.get("past_key_values")
            ln = k["inputs_embeds"].shape[1] if k.get("inputs_embeds") is not None else k["input_ids"].shape[1]
            past = pkv.get_seq_length() if pkv is not None else 0
            runner._cur_len = past + ln
            return orig_fwd(*a, **k)
        self.model.model.language_model.forward = fwd
        try:
            out = self.model.generate(**enc, max_new_tokens=40, do_sample=False)
        finally:
            self.model.model.language_model.forward = orig_fwd
            STATE["bias"] = None; self.clear_resid()
        return [c.strip() for c in self.proc.batch_decode(out[:, base_len:], skip_special_tokens=True)]

    def invert(self, imgs, targets, variant, n_steps, lr, resid_layers):
        enc, n_prompt = self.batch(imgs, targets)
        B, T = enc["input_ids"].shape
        n_img = STATE["img_pos"].shape[1]
        if variant == "route":
            param = nn.Parameter(torch.zeros(len(self.layers), B, n_img, device=DEVICE))
        else:
            param = nn.Parameter(torch.zeros(len(resid_layers), B, self.model.config.text_config.hidden_size, device=DEVICE))
            self.set_resid(param, resid_layers)
        opt = torch.optim.Adam([param], lr=lr)
        ids = enc["input_ids"]; am = enc["attention_mask"]
        pad = (am == 0).sum(1)                                   # left padding
        start = pad + n_prompt                                   # first target token position
        tgt_mask = torch.arange(T, device=DEVICE).unsqueeze(0) >= start.unsqueeze(1)
        self._cur_len = T
        for _ in range(n_steps):
            STATE["bias"] = param if variant == "route" else None
            logits = self.model(**enc).logits[:, :-1].float()
            lab = ids[:, 1:]; m = tgt_mask[:, 1:] & (am[:, 1:] == 1)
            loss = F.cross_entropy(logits[m], lab[m])
            opt.zero_grad(); loss.backward(); opt.step()
        STATE["bias"] = None; self.clear_resid()
        with torch.no_grad():
            if variant == "route":
                return self.generate(imgs, bias=param.detach())
            return self.generate(imgs, resid=param.detach(), resid_layers=resid_layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=150)
    ap.add_argument("--offset", type=int, default=14000)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--res", type=int, default=448)
    ap.add_argument("--variants", nargs="+", default=["route", "resid"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); rng = random.Random(args.seed)
    R = Runner(args.res)
    L = len(R.layers); resid_layers = [L // 2 - 1, L // 2]
    items = load_coco_items(args.offset, args.n_images)
    imgs = [Image.open(io.BytesIO(it["img_bytes"])).convert("RGB") if isinstance(it["img_bytes"], (bytes, bytearray))
            else it["img_bytes"].convert("RGB") for it in items]
    n = len(items); refs = [it["refs"] for it in items]
    perm = list(range(n))
    while any(p == i for i, p in enumerate(perm)):
        rng.shuffle(perm)
    mism = [rng.choice(refs[perm[i]]) for i in range(n)]
    swaps = []
    for i in range(n):
        own = set().union(*[objects(r) for r in refs[i]]); s = None
        for k in rng.sample(range(len(refs[i])), len(refs[i])):
            s = make_target(refs[i][k], own, rng)
            if s:
                swaps.append((k,) + s); break
        if not s:
            swaps.append(None)

    base = []
    for s in range(0, n, args.batch):
        base += R.generate(imgs[s:s + args.batch])
    print("native example:", base[:2], flush=True)
    jobs = [(i, "true", refs[i][0]) for i in range(n)] + [(i, "mismatch", mism[i]) for i in range(n)]
    jobs += [(i, "swap", swaps[i][1]) for i in range(n) if swaps[i]]
    recs = [{"image": i, "refs": refs[i], "baseline": base[i], "mismatch_source": perm[i], "mismatch_target": mism[i],
             "swap_k": swaps[i][0] if swaps[i] else None, "swap_target": swaps[i][1] if swaps[i] else None,
             "swap_fake": swaps[i][2] if swaps[i] else None, "swap_fake_obj": swaps[i][3] if swaps[i] else None,
             "variants": {v: {"true": [None] * len(refs[i]), "mismatch": None, "swap": None} for v in args.variants}} for i in range(n)]
    path = os.path.join(OUT_DIR, f"qwen_protocol_n{n}_s{args.seed}.json")
    t0 = time.time()
    for v in args.variants:
        for s in range(0, len(jobs), args.batch):
            ch = jobs[s:s + args.batch]
            caps = R.invert([imgs[j[0]] for j in ch], [j[2] for j in ch], v, args.n_steps, args.lr, resid_layers)
            for (i, kind, _), c in zip(ch, caps):
                if kind == "true":
                    recs[i]["variants"][v]["true"][0] = c
                else:
                    recs[i]["variants"][v][kind] = c
            print(f"[{v}] {min(s + args.batch, len(jobs))}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
            json.dump({"model": MODEL_ID, "partial": True, "variants": args.variants, "records": recs}, open(path, "w"))
    json.dump({"model": MODEL_ID, "n_images": n, "n_steps": args.n_steps, "lr": args.lr, "variants": args.variants, "res": args.res,
               "resid_layers": resid_layers, "param_counts": {"route": L * 256, "resid": len(resid_layers) * R.model.config.text_config.hidden_size},
               "records": recs}, open(path, "w"))
    print("saved", path)


if __name__ == "__main__":
    main()
