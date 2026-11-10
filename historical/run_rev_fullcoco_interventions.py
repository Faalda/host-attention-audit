"""
run_rev_fullcoco_interventions.py — E9: corrected suppression/transplant study (logic of
run_rev_interventions.py main(), copied verbatim) pointed at the full-COCO model: checkpoints, vocab,
int8 test features, denoised plans (results/full_coco/plans_test.pt) and Karpathy-test refs.
Summary uses run_rev_interventions.summarize unchanged.

Usage: CUDA_VISIBLE_DEVICES=0 python run_rev_fullcoco_interventions.py --n-images 500
"""
import argparse
from run_rev_fullcoco_common import *
from run_rev_inversion_protocol import dec_generate
from run_rev_eval_metrics import objects, lr1, corpus_bleu_cider, boot_ci, norm
from run_rev_interventions import suppress, grid_neighbours, summarize

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=500)
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 10])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed); torch.manual_seed(args.seed)

    vocab = json.load(open(os.path.join(FC, "tok_vocab.json")))
    tok = tok_from_vocab(vocab)
    proj, _, _, dec = load_final_models(len(vocab))
    n = args.n_images
    feats = GPUFeats("test", n=n)
    plans = torch.load(os.path.join(FC, "plans_test.pt"), weights_only=True)
    refs = load_caps("test")[:n]

    def decode(i, P):
        with torch.no_grad():
            X = proj(feats.get(torch.tensor([i], device=DEVICE)))
            toks = dec_generate(dec, X, P.unsqueeze(0).to(DEVICE))[0].tolist()
        return tok.dec(toks), toks

    recs = []
    perm = list(range(n))
    while any(p == i for i, p in enumerate(perm)):
        rng.shuffle(perm)
    for i in range(n):
        P = plans[i]
        base, toks = decode(i, P)
        L_used = max(1, len([t for t in toks[1:] if t != 0]))   # rows consumed while generating (incl. EOS step)
        used = P[:L_used]
        mass = used.sum(0)
        am = used.argmax(-1)
        base_objs = objects(base)
        # primary noun: first COCO object word in the unedited caption, and its row
        words = toks[1:]
        prim, prim_row = None, None
        for t_idx, w in enumerate(words):
            if w in (0, 2):
                break
            o = objects(vocab[w])
            if o:
                prim, prim_row = next(iter(o)), t_idx      # token at output index t_idx+1 is predicted by row t_idx
                break
        rec = {"image": i, "refs": refs[i], "baseline": base, "L_used": L_used, "primary": prim, "supp": {}, "transplant": {}}
        for K in args.ks:
            strat = {}
            strat["top"] = mass.topk(K).indices.tolist()
            strat["bottom"] = (-mass).topk(K).indices.tolist()
            strat["random"] = rng.sample(range(N_PATCHES), K)
            if prim_row is not None:
                strat["token"] = grid_neighbours(int(am[prim_row]), K)
            for name, S in strat.items():
                Ps = suppress(P, S)
                cap, _ = decode(i, Ps)
                rows_hit = int(sum(int(a) in set(S) for a in am.tolist()))
                rec["supp"][f"{name}_K{K}"] = {"cap": cap, "rows_unrouted": rows_hit}
            # magnitude-matched: unroute as many used rows as `top` did, at random
            n_hit = rec["supp"][f"top_K{K}"]["rows_unrouted"]
            rows = rng.sample(range(L_used), min(n_hit, L_used))
            Pr = P.clone()
            if rows:
                Pr[rows] = 1.0 / N_PATCHES
            cap, _ = decode(i, Pr)
            rec["supp"][f"rows_K{K}"] = {"cap": cap, "rows_unrouted": len(rows)}
        # transplant
        j = perm[i]
        Pd = plans[j]
        pi = torch.randperm(N_PATCHES)
        Pp = Pd[:, pi]
        rs = torch.randperm(L_CAP)
        Pw = Pd[rs]
        Pu = torch.full_like(Pd, 1.0 / N_PATCHES)
        for name, Q in [("donor", Pd), ("permuted", Pp), ("rowshuf", Pw), ("uniform", Pu)]:
            rec["transplant"][name] = decode(i, Q)[0]
        rec["donor_index"] = j
        rec["donor_refs"] = refs[j]
        rec["donor_own_caption"] = decode(j, Pd)[0]
        recs.append(rec)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{n}", flush=True)
    path = os.path.join(FC, f"interventions_n{n}_s{args.seed}.json")
    json.dump({"n": n, "ks": args.ks, "records": recs}, open(path, "w"))
    print("saved", path)
    summarize(path)



if __name__ == "__main__":
    main()
