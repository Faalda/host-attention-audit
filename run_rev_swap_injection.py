"""
run_rev_swap_injection.py — counterfactual object-injection test (revision, positive-evidence experiment).

Target = one of the image's own references with one COCO-object word replaced by a COCO object that
appears in none of the image's references (e.g. "a dog on a couch" -> "a giraffe on a couch").
Invert toward it with each parameterisation and measure:
  injection  : fraction of outputs that mention the fake object (hallucination injected)
  retention  : fit to the unchanged part of the target (legacy ROUGE-1 vs the original reference)
A grounding-selective routing channel should keep injection near the baseline rate while still moving
toward the rest of the target; a generic control channel should inject the fake object.
"""
import os, sys, json, random, re
import torch

sys.path.insert(0, os.path.dirname(__file__))
from run_rev_inversion_protocol import (Tokenizer, DEVICE, D_MODEL, N_HEADS, N_LAYERS, L_CAP, PatchProjector, N_CAP_PER,
                                        N_PATCHES, ENCODER_DIM, CACHE_PATH, SoftGuidedCaptionDecoder, SAVE_DIR, OUT_DIR,
                                        TEST_START, invert)
from run_rev_eval_metrics import COCO_OBJ, objects, lr1, boot_ci

# single-word, unambiguous object names usable as replacements
REPL = ["giraffe", "elephant", "zebra", "horse", "cow", "sheep", "bear", "dog", "cat", "bird", "airplane", "train",
        "bus", "truck", "boat", "motorcycle", "bicycle", "pizza", "banana", "broccoli", "cake", "laptop", "umbrella",
        "kite", "surfboard", "skateboard", "clock", "toilet", "refrigerator", "suitcase", "frisbee", "vase"]
SYN2OBJ = {s: o for o, ss in COCO_OBJ.items() for s in ss if " " not in s}


def make_target(ref, own_objs, rng):
    words = ref.strip().split()
    idx = [i for i, w in enumerate(words) if re.sub(r"[^a-z]", "", w.lower()) in SYN2OBJ
           and SYN2OBJ[re.sub(r"[^a-z]", "", w.lower())] != "person"]
    if not idx:
        return None
    i = rng.choice(idx)
    cands = [r for r in REPL if SYN2OBJ.get(r, r) not in own_objs and r not in own_objs]
    if not cands:
        return None
    fake = rng.choice(cands)
    punct = re.sub(r"[A-Za-z]", "", words[i])
    words[i] = fake + punct
    return " ".join(words), fake, SYN2OBJ.get(fake, fake)


def main():
    n, seed = 1000, 0
    rng = random.Random(seed); torch.manual_seed(seed)
    vocab = json.load(open(os.path.join(SAVE_DIR, "tok_vocab.json")))
    tok = Tokenizer(); tok.vocab = vocab; tok.w2i = {w: i for i, w in enumerate(vocab)}
    proj = PatchProjector(ENCODER_DIM, D_MODEL).to(DEVICE)
    proj.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_proj.pt"), map_location=DEVICE, weights_only=True))
    dec = SoftGuidedCaptionDecoder(len(vocab), D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)
    dec.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model_guided.pt"), map_location=DEVICE, weights_only=True))
    proj.eval(); dec.eval()
    for p in list(proj.parameters()) + list(dec.parameters()):
        p.requires_grad_(False)
    feats = torch.load(CACHE_PATH, weights_only=True)
    plans = torch.load(os.path.join(SAVE_DIR, "plans_test.pt"), weights_only=True)
    prev = json.load(open(os.path.join(OUT_DIR, "inversion_protocol_n1000_s0.json")))["records"]

    jobs = []
    for rec in prev[:n]:
        R = rec["refs"]; own = set().union(*[objects(r) for r in R])
        order = list(range(len(R))); rng.shuffle(order)
        for k in order:
            m = make_target(R[k], own, rng)
            if m and m[1] in tok.w2i:
                jobs.append((rec["image"], k, m[0], m[1], m[2])); break
    print(len(jobs), "swap targets; e.g.", jobs[0][2], flush=True)
    out = {v: [] for v in ["free", "onehot", "hidden"]}
    for v in out:
        for s in range(0, len(jobs), 500):
            ch = jobs[s:s + 500]
            idx = [j[0] for j in ch]
            X = proj(feats[[TEST_START + i for i in idx]].to(DEVICE))
            A = plans[idx].to(DEVICE)
            tgt = torch.tensor([tok.enc(j[2], max_len=L_CAP, bos=True, eos=True) for j in ch], device=DEVICE)
            _, _, toks = invert(dec, X, A, tgt, v, 300, 0.05)
            out[v] += [tok.dec(t) for t in toks.cpu().tolist()]
        print("done", v, flush=True)

    res = {"n": len(jobs), "variants": {}}
    base = [prev[j[0]]["baseline"] for j in jobs]
    def rates(caps):
        inj = [float(j[4] in objects(c) or j[3] in c.lower().split()) for c, j in zip(caps, jobs)]
        ret = [lr1(c, [prev[j[0]]["refs"][j[1]]]) for c, j in zip(caps, jobs)]
        return inj, ret
    b_inj, b_ret = rates(base)
    res["baseline"] = {"injection": round(sum(b_inj) / len(b_inj), 4), "fit_to_original_ref": round(sum(b_ret) / len(b_ret), 4)}
    print("baseline", res["baseline"])
    for v, caps in out.items():
        inj, ret = rates(caps)
        res["variants"][v] = {"injection": round(sum(inj) / len(inj), 4),
                              "injection_delta_ci": [round(x, 4) for x in boot_ci([a - b for a, b in zip(inj, b_inj)])],
                              "fit_to_original_ref": round(sum(ret) / len(ret), 4),
                              "fit_delta_ci": [round(x, 4) for x in boot_ci([a - b for a, b in zip(ret, b_ret)])]}
        print(v, res["variants"][v], flush=True)
    res["examples"] = [{"ref": prev[j[0]]["refs"][j[1]], "target": j[2], "baseline": b, **{v: out[v][q] for v in out}}
                       for q, (j, b) in enumerate(zip(jobs[:30], base[:30]))]
    res["records"] = [{"image": j[0], "k": j[1], "target": j[2], "fake": j[3], **{v: out[v][q] for v in out}} for q, j in enumerate(jobs)]
    json.dump(res, open(os.path.join(OUT_DIR, "swap_injection_n1000_s0.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
