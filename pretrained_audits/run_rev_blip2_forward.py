"""
run_rev_blip2_forward.py — revision experiment E3 (R2-Q1 "learned forward predictor for BLIP-2", R1-W2).

Trains a forward plan predictor for BLIP-2's Q-Former cross-attention, end-to-end through the
frozen model with the captioning NLL on COCO training captions, and evaluates it without any
access to test references.

  img   : image-conditioned low-rank bias  B_l[q,p] = sum_r U_l[q,r] * (W_l e_p)_r   (e_p: ViT patch feature)
  const : learned image-INDEPENDENT low-rank bias B_l = U_l V_l of the same rank (controls for COCO
          caption style / dataset-level position priors that any fine-tuning would pick up)
Both are compared with native BLIP-2 (bias 0) and, via run_rev_eval_metrics.py, with the per-image
oracle of run_rev_blip2_protocol.py on identical references (single-ref / held-out for refs r_0, r_1).
Output format matches the protocol dumps: for forward variants the same caption fills every slot.
"""
import os, sys, json, argparse, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from run_blip2_prediction_gap import load_model_and_processor, load_coco_items, preprocess_image, N_CROSS_LAYERS, N_QUERY
from run_rev_blip2_protocol import lm_inputs, generate, DEVICE, OUT_DIR


class BiasPredictor(nn.Module):
    def __init__(self, enc_dim, n_patches, rank, image_conditioned):
        super().__init__()
        self.ic = image_conditioned
        self.U = nn.Parameter(0.01 * torch.randn(N_CROSS_LAYERS, N_QUERY, rank))
        if image_conditioned:
            self.W = nn.Parameter(torch.zeros(N_CROSS_LAYERS, rank, enc_dim))
            self.ln = nn.LayerNorm(enc_dim)
        else:
            self.V = nn.Parameter(torch.zeros(N_CROSS_LAYERS, rank, n_patches))

    def forward(self, E):                      # E: [B, P, enc_dim]
        B, P, _ = E.shape
        if self.ic:
            Vb = torch.einsum("lrd,bpd->lbrp", self.W, self.ln(E.float()))       # [6,B,r,P]
        else:
            Vb = self.V.unsqueeze(1).expand(-1, B, -1, -1)
        bias = torch.einsum("lqr,lbrp->lbqp", self.U, Vb)                      # [6,B,Q,P]
        return bias.unsqueeze(2)                                               # [6,B,1,Q,P]


def nll(model, proc, E, bias, caps):
    B = E.shape[0]
    enc = proc.tokenizer(caps, return_tensors="pt", padding=True, truncation=True, max_length=64)
    ids, am = enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)
    li = lm_inputs(model, E, bias.half(), None)
    emb = torch.cat([li, model.language_model.get_input_embeddings()(ids[:, :-1])], 1)
    mask = torch.cat([torch.ones(B, N_QUERY, dtype=torch.long, device=DEVICE), am[:, :-1]], 1)
    logits = model.language_model(inputs_embeds=emb, attention_mask=mask, return_dict=True).logits[:, N_QUERY:].float()
    v = am[:, 1:].bool()
    return F.cross_entropy(logits[v], ids[:, 1:][v])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=13000)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--offset", type=int, default=14000)
    ap.add_argument("--refs-per-image", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); rng = random.Random(args.seed)

    proc, model = load_model_and_processor()
    from datasets import load_dataset
    ds = load_dataset("clip-benchmark/wds_mscoco_captions", split="train")
    enc_dim = model.config.vision_config.hidden_size

    def embeds(pvs):
        with torch.no_grad():
            return model.vision_model(torch.cat(pvs), return_dict=True).last_hidden_state

    results = {}
    predictors = {}
    for variant in ["const", "img"]:
        pred = BiasPredictor(enc_dim, 257, args.rank, variant == "img").to(DEVICE)
        opt = torch.optim.AdamW(pred.parameters(), lr=args.lr, weight_decay=0.0)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
        t0 = time.time(); run = 0.0
        for step in range(1, args.steps + 1):
            idx = [rng.randrange(args.n_train) for _ in range(args.batch)]
            rows = [ds[i] for i in idx]
            caps = [rng.choice([c.strip() for c in r["txt"].strip().split("\n") if c.strip()][:5]) for r in rows]
            E = embeds([preprocess_image(proc, r["jpg"], DEVICE) for r in rows])
            loss = nll(model, proc, E, pred(E), caps)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
            if step % 200 == 0:
                print(f"[{variant}] step {step}/{args.steps} nll~{run:.4f} {time.time() - t0:.0f}s", flush=True)
        predictors[variant] = pred.eval()
        torch.save(pred.state_dict(), os.path.join(OUT_DIR, f"blip2_forward_{variant}_r{args.rank}_s{args.seed}.pt"))

    # evaluation on the same test images / reference slots as run_rev_blip2_protocol.py
    items = load_coco_items(args.offset, args.n_eval)
    records = []
    for s in range(0, len(items), args.batch):
        chunk = items[s:s + args.batch]
        E = embeds([preprocess_image(proc, it["img_bytes"], DEVICE) for it in chunk])
        zb = torch.zeros(N_CROSS_LAYERS, 1, 1, N_QUERY, E.shape[1], device=DEVICE).half()
        base = generate(model, proc, E, zb)
        out = {}
        with torch.no_grad():
            for v, p in predictors.items():
                out[v] = generate(model, proc, E, p(E).half())
        for j, it in enumerate(chunk):
            i = s + j
            k = min(args.refs_per_image, len(it["refs"]))
            records.append({"image": i, "refs": it["refs"], "baseline": base[j], "mismatch_target": it["refs"][0],
                            "variants": {v: {"true": [out[v][j]] * k + [None] * (len(it["refs"]) - k), "mismatch": out[v][j]}
                                         for v in predictors}})
        print(f"eval {min(s + args.batch, len(items))}/{len(items)}", flush=True)
    path = os.path.join(OUT_DIR, f"blip2_forward_n{args.n_eval}_r{args.rank}_s{args.seed}.json")
    json.dump({"n_images": len(items), "variants": list(predictors), "rank": args.rank, "steps": args.steps,
               "batch": args.batch, "lr": args.lr, "records": records}, open(path, "w"))
    print("saved", path)


if __name__ == "__main__":
    main()
