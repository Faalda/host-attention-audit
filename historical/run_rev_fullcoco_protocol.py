"""
run_rev_fullcoco_protocol.py — E9: symmetric inversion protocol (logic of run_rev_inversion_protocol.py,
unchanged: variants free / onehot / hidden, 300 Adam steps, lr 0.05, invert toward each of the 5 refs +
one shuffled target from a fixed seed-0 derangement) pointed at the full-COCO model:
  checkpoints/vocab : results/full_coco/model_*.pt, tok_vocab.json
  features          : results/full_coco/dino_test_q.npy (int8, dequantised)
  plans (A_base)    : results/full_coco/plans_test.pt (denoised, softmax(10*maskgit_K4))
  refs              : Karpathy test split, first 5 captions, images [0, n)
Also writes the human leave-one-out ceiling (each ref vs the other 4; legacy ROUGE-1 and CIDEr-D).
Output JSON has the same format as run_rev_inversion_protocol.py.

Usage: CUDA_VISIBLE_DEVICES=0 python run_rev_fullcoco_protocol.py --n-images 1000
"""
import sys
import os
import argparse
from run_rev_fullcoco_common import *
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for clean_pipeline./inversion. imports
from inversion.run_rev_inversion_protocol import dec_generate, invert, plan_stats
from inversion.run_rev_eval_metrics import lr1, corpus_bleu_cider, boot_ci


def human_ceiling(refs):
    r1, preds, others, ids = [], [], [], []
    for i, R in enumerate(refs):
        for k in range(len(R)):
            o = [x for j, x in enumerate(R) if j != k]
            r1.append(lr1(R[k], o)); preds.append(R[k]); others.append(o); ids.append(i)
    _, cid = corpus_bleu_cider(preds, others)
    ids = np.array(ids)
    per_img_r1 = np.array([np.mean(np.array(r1)[ids == i]) for i in range(len(refs))])
    per_img_c = np.array([np.mean(cid[ids == i]) for i in range(len(refs))])
    return {"legacy_rouge1": [round(x, 4) for x in boot_ci(per_img_r1)],
            "cider": [round(x, 4) for x in boot_ci(per_img_c)], "n_images": len(refs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=1000)
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--variants", nargs="+", default=["free", "onehot", "hidden"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)

    tok = tok_from_vocab(json.load(open(os.path.join(FC, "tok_vocab.json"))))
    proj, _, _, dec = load_final_models(len(tok.vocab))
    n = args.n_images
    feats = GPUFeats("test", n=n)
    plans_test = torch.load(os.path.join(FC, "plans_test.pt"), weights_only=True)
    refs = load_caps("test")[:n]

    rng = random.Random(args.seed)
    perm = list(range(n))
    while any(p == i for i, p in enumerate(perm)):
        rng.shuffle(perm)
    mism = [rng.choice(refs[perm[i]]) for i in range(n)]

    def X_of(idx):
        with torch.no_grad():
            return proj(feats.get(torch.tensor(idx, device=DEVICE)))

    base_caps = []
    with torch.no_grad():
        for s in range(0, n, args.batch):
            idx = list(range(s, min(n, s + args.batch)))
            base_caps += [tok.dec(t) for t in dec_generate(dec, X_of(idx), plans_test[idx].to(DEVICE)).cpu().tolist()]

    jobs = [(i, "true", k, r) for i in range(n) for k, r in enumerate(refs[i])]
    jobs += [(i, "mismatch", -1, mism[i]) for i in range(n)]
    records = {i: {"image": i, "refs": refs[i], "baseline": base_caps[i],
                   "mismatch_source": perm[i], "mismatch_target": mism[i],
                   "variants": {v: {"true": [None] * len(refs[i]), "mismatch": None,
                                    "true_stats": [None] * len(refs[i]), "mismatch_stats": None}
                                for v in args.variants}} for i in range(n)}
    t0 = time.time()
    for v in args.variants:
        for s in range(0, len(jobs), args.batch):
            chunk = jobs[s:s + args.batch]
            img_idx = [j[0] for j in chunk]
            X = X_of(img_idx)
            A = plans_test[img_idx].to(DEVICE)
            tgt = torch.tensor([tok.enc(j[3], max_len=L_CAP, bos=True, eos=True) for j in chunk],
                               dtype=torch.long, device=DEVICE)
            plan, delta, toks = invert(dec, X, A, tgt, v, args.n_steps, args.lr)
            lengths = (tgt[:, 1:] != 0).sum(-1).tolist()
            stats = plan_stats(plan, A, lengths)
            caps = [tok.dec(t) for t in toks.cpu().tolist()]
            for (i, kind, k, _), c, st in zip(chunk, caps, stats):
                rv = records[i]["variants"][v]
                if kind == "true":
                    rv["true"][k] = c; rv["true_stats"][k] = st
                else:
                    rv["mismatch"] = c; rv["mismatch_stats"] = st
            print(f"[{v}] {min(s + args.batch, len(jobs))}/{len(jobs)} jobs  {time.time() - t0:.0f}s", flush=True)

    out = {"target_mode": "random", "n_images": n, "n_steps": args.n_steps, "lr": args.lr, "seed": args.seed,
           "variants": args.variants, "param_count_per_inversion": L_CAP * N_PATCHES,
           "model": "full_coco (results/full_coco)", "records": [records[i] for i in range(n)]}
    path = os.path.join(FC, f"inversion_protocol_n{n}_s{args.seed}{args.tag}.json")
    json.dump(out, open(path, "w"))
    print("saved", path)
    hc = human_ceiling(refs)
    json.dump(hc, open(os.path.join(FC, f"human_ceiling_n{n}.json"), "w"), indent=1)
    print("human ceiling", hc)


if __name__ == "__main__":
    main()
