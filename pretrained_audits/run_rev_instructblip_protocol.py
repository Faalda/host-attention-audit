"""
run_rev_instructblip_protocol.py — third-architecture replication of the BLIP-2 protocol (revision).

InstructBLIP (Vicuna-7B, instruction-aware Q-Former). Same plan interface as run_rev_blip2_protocol.py:
an additive logit bias on every Q-Former cross-attention layer ([n_cross, 32 queries, 257 patches], shared
over heads, init 0), plus the query-embedding offset control. Per image: native caption, inversion toward
r_0 (single-ref / held-out) and toward a caption of another image (shuffled target).
Prompt: "A short image caption:" (both to the Q-Former and the LLM), greedy decoding, 40 new tokens.
Output format matches the other protocol dumps (run_rev_eval_metrics.py / run_rev_selectivity.py).
"""
import os, sys, json, argparse, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from run_blip2_prediction_gap import load_coco_items, preprocess_image
from run_rev_blip2_protocol import _biased_forward

DEVICE = "cuda"
MODEL_ID = "Salesforce/instructblip-vicuna-7b"
PROMPT = "A short image caption:"
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "revision")


def cross_modules(model):
    return [l.crossattention.attention for l in model.qformer.encoder.layer if getattr(l, "has_cross_attention", False)]


class Runner:
    def __init__(self):
        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        self.proc = InstructBlipProcessor.from_pretrained(MODEL_ID)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.cross = cross_modules(self.model)
        self.nq = self.model.query_tokens.shape[1]
        q = self.proc.qformer_tokenizer([PROMPT], return_tensors="pt")
        self.q_ids, self.q_mask = q["input_ids"].to(DEVICE), q["attention_mask"].to(DEVICE)
        t = self.proc.tokenizer([PROMPT], return_tensors="pt", add_special_tokens=True)
        self.p_ids = t["input_ids"].to(DEVICE)

    def patch(self, bias):
        orig = []
        for i, ca in enumerate(self.cross):
            orig.append(ca.forward); ca.forward = _biased_forward(ca, bias[i])
        return orig

    def restore(self, orig):
        for ca, o in zip(self.cross, orig):
            ca.forward = o

    def lang(self, E, bias, qdelta):
        B = E.shape[0]
        qt = self.model.query_tokens.expand(B, -1, -1)
        if qdelta is not None:
            qt = qt + qdelta.to(qt.dtype)
        qmask = torch.cat([torch.ones(B, self.nq, dtype=torch.long, device=DEVICE), self.q_mask.expand(B, -1)], 1)
        orig = self.patch(bias)
        try:
            out = self.model.qformer(input_ids=self.q_ids.expand(B, -1), attention_mask=qmask, query_embeds=qt,
                                     encoder_hidden_states=E,
                                     encoder_attention_mask=torch.ones(E.shape[:-1], dtype=torch.long, device=DEVICE),
                                     return_dict=True)
        finally:
            self.restore(orig)
        h = out.last_hidden_state[:, :self.nq]
        return self.model.language_projection(h.to(self.model.language_projection.weight.dtype))

    def embed(self, ids):
        return self.model.language_model.get_input_embeddings()(ids)

    @torch.no_grad()
    def generate(self, E, bias, qdelta=None):
        B = E.shape[0]
        li = self.lang(E, bias, qdelta)
        emb = torch.cat([li, self.embed(self.p_ids.expand(B, -1))], 1)
        out = self.model.language_model.generate(inputs_embeds=emb, attention_mask=torch.ones(emb.shape[:2], dtype=torch.long, device=DEVICE),
                                                 max_new_tokens=40, do_sample=False, num_beams=1)
        return [c.strip() for c in self.proc.tokenizer.batch_decode(out, skip_special_tokens=True)]

    def invert(self, E, targets, variant, n_steps, lr):
        B, P = E.shape[0], E.shape[1]
        nl = len(self.cross)
        if variant == "full":
            param = nn.Parameter(torch.zeros(nl, B, 1, self.nq, P, device=DEVICE))
        else:
            param = nn.Parameter(torch.zeros(B, self.nq, self.model.query_tokens.shape[-1], device=DEVICE))
        zero = torch.zeros(nl, B, 1, self.nq, P, device=DEVICE)
        opt = torch.optim.Adam([param], lr=lr)
        tok = self.proc.tokenizer([" " + t.strip() for t in targets], return_tensors="pt", padding=True,
                                  add_special_tokens=False, truncation=True, max_length=48)
        ids, am = tok["input_ids"].to(DEVICE), tok["attention_mask"].to(DEVICE)
        eos = torch.full((B, 1), self.proc.tokenizer.eos_token_id, device=DEVICE)
        ids = torch.cat([ids, eos], 1); am = torch.cat([am, torch.ones_like(eos)], 1)
        P_len = self.p_ids.shape[1]
        for _ in range(n_steps):
            bias = param if variant == "full" else zero
            li = self.lang(E, bias.half(), param if variant == "query" else None)
            emb = torch.cat([li, self.embed(self.p_ids.expand(B, -1)), self.embed(ids)], 1)
            mask = torch.cat([torch.ones(B, self.nq + P_len, dtype=torch.long, device=DEVICE), am], 1)
            logits = self.model.language_model(inputs_embeds=emb, attention_mask=mask).logits.float()
            start = self.nq + P_len - 1
            pred = logits[:, start:start + ids.shape[1]]
            v = am.bool()
            loss = F.cross_entropy(pred[v], ids[v])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            bias = (param if variant == "full" else zero).half()
            return self.generate(E, bias, param.detach() if variant == "query" else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=200)
    ap.add_argument("--offset", type=int, default=14000)
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--variants", nargs="+", default=["full", "query"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    R = Runner()
    items = load_coco_items(args.offset, args.n_images)
    n = len(items); refs = [it["refs"] for it in items]
    rng = random.Random(args.seed)
    perm = list(range(n))
    while any(p == i for i, p in enumerate(perm)):
        rng.shuffle(perm)
    mism = [rng.choice(refs[perm[i]]) for i in range(n)]

    def E_of(idx):
        pv = torch.cat([R.proc.image_processor(images=__import__("PIL.Image", fromlist=["Image"]).open(__import__("io").BytesIO(items[i]["img_bytes"])).convert("RGB")
                                               if isinstance(items[i]["img_bytes"], (bytes, bytearray)) else items[i]["img_bytes"].convert("RGB"),
                                               return_tensors="pt")["pixel_values"] for i in idx]).to(DEVICE, torch.float16)
        with torch.no_grad():
            return R.model.vision_model(pixel_values=pv, return_dict=True).last_hidden_state

    base = []
    nl = len(R.cross)
    for s in range(0, n, args.batch):
        idx = list(range(s, min(n, s + args.batch))); E = E_of(idx)
        base += R.generate(E, torch.zeros(nl, 1, 1, R.nq, E.shape[1], device=DEVICE).half())
    print("native done; example:", base[:2], flush=True)
    jobs = [(i, "true", 0, refs[i][0]) for i in range(n)] + [(i, "mismatch", -1, mism[i]) for i in range(n)]
    recs = [{"image": i, "refs": refs[i], "baseline": base[i], "mismatch_source": perm[i], "mismatch_target": mism[i],
             "variants": {v: {"true": [None] * len(refs[i]), "mismatch": None} for v in args.variants}} for i in range(n)]
    path = os.path.join(OUT_DIR, f"instructblip_protocol_n{n}_s{args.seed}.json")
    t0 = time.time()
    for v in args.variants:
        for s in range(0, len(jobs), args.batch):
            chunk = jobs[s:s + args.batch]
            caps = R.invert(E_of([j[0] for j in chunk]), [j[3] for j in chunk], v, args.n_steps, args.lr)
            for (i, kind, k, _), c in zip(chunk, caps):
                if kind == "true":
                    recs[i]["variants"][v]["true"][k] = c
                else:
                    recs[i]["variants"][v]["mismatch"] = c
            print(f"[{v}] {min(s + args.batch, len(jobs))}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    json.dump({"model": MODEL_ID, "n_images": n, "n_steps": args.n_steps, "lr": args.lr, "variants": args.variants,
               "param_counts": {"full": nl * R.nq * 257, "query": R.nq * R.model.query_tokens.shape[-1]}, "records": recs}, open(path, "w"))
    print("saved", path)


if __name__ == "__main__":
    main()
