"""Counterfactual object injection for the full-COCO captioner."""
import sys
import os
import argparse
from run_rev_fullcoco_common import *
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for clean_pipeline./inversion. imports
from inversion.run_rev_inversion_protocol import dec_generate, invert
from inversion.run_rev_swap_injection import make_target
from inversion.run_rev_eval_metrics import objects, lr1, boot_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)

    tok = tok_from_vocab(json.load(open(os.path.join(FC, "tok_vocab.json"))))
    proj, _, _, dec = load_final_models(len(tok.vocab))
    feats = GPUFeats("test", n=args.n_images)
    plans = torch.load(os.path.join(FC, "plans_test.pt"), weights_only=True)
    prev = json.load(open(os.path.join(FC, "inversion_protocol_n1000_s0.json")))["records"][:args.n_images]

    rng = random.Random(args.seed); jobs = []
    for rec in prev:
        own = set().union(*[objects(r) for r in rec["refs"]])
        order = list(range(len(rec["refs"]))); rng.shuffle(order)
        for k in order:
            target = make_target(rec["refs"][k], own, rng)
            if target and target[1] in tok.w2i:
                jobs.append((rec["image"], k, *target))
                break

    out = {v: [] for v in ("free", "onehot", "hidden")}
    for v in out:
        for s in range(0, len(jobs), args.batch):
            chunk = jobs[s:s + args.batch]; idx = torch.tensor([j[0] for j in chunk], device=DEVICE)
            with torch.no_grad():
                X = proj(feats.get(idx))
            tgt = torch.tensor([tok.enc(j[2], L_CAP, bos=True, eos=True) for j in chunk], device=DEVICE)
            _, _, toks = invert(dec, X, plans[idx.cpu()].to(DEVICE), tgt, v, 300, 0.05)
            out[v] += [tok.dec(t) for t in toks.cpu().tolist()]
        print("done", v, flush=True)

    base = [prev[j[0]]["baseline"] for j in jobs]
    def rates(caps):
        inj = [float(j[4] in objects(c) or j[3] in c.lower().split()) for c, j in zip(caps, jobs)]
        fit = [lr1(c, [prev[j[0]]["refs"][j[1]]]) for c, j in zip(caps, jobs)]
        return inj, fit
    b_inj, b_fit = rates(base)
    result = {"n": len(jobs), "baseline": {"injection": float(np.mean(b_inj)), "fit_to_original_ref": float(np.mean(b_fit))}, "variants": {}}
    for v, caps in out.items():
        inj, fit = rates(caps)
        result["variants"][v] = {
            "injection": float(np.mean(inj)),
            "injection_delta_ci": [float(x) for x in boot_ci(np.subtract(inj, b_inj))],
            "fit_to_original_ref": float(np.mean(fit)),
            "fit_delta_ci": [float(x) for x in boot_ci(np.subtract(fit, b_fit))],
        }
    result["records"] = [{"image": j[0], "target": j[2], "fake": j[3], **{v: out[v][i] for v in out}}
                         for i, j in enumerate(jobs)]
    path = os.path.join(FC, f"swap_injection_n{args.n_images}_s{args.seed}.json")
    json.dump(result, open(path, "w"), indent=1)
    print(json.dumps(result["baseline"] | result["variants"], indent=1)); print("saved", path)


if __name__ == "__main__":
    main()
