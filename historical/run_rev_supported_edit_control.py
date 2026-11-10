"""
run_rev_supported_edit_control.py — pre-registered positive control, Table "tab:open" row 4:
"Within-image reference swap (foreground A->B, both present)."

The paper's existing counterfactual-injection test (run_rev_swap_injection.py) only shows
that the plan REFUSES to import an object that is absent from the image. That is consistent
with two different explanations: (A) the plan is a grounded editor that imports supported
content on demand and refuses unsupported content, or (B) the plan is simply too weak/limited
in capacity to import *anything*, so the low injection rate is a capability ceiling, not
selective grounding.

This control tests the missing half: take an image whose five references jointly mention at
least two distinct, REAL, PRESENT COCO objects (both actually in the image, evidenced by
appearing in its own reference captions), build a target that substitutes one real, present
object for the other (A -> B, not a fake/absent object), and invert toward it. If the plan can
be legitimately steered to mention B (a present, supported edit) while it could not be steered
to mention an absent object in the counterfactual test, that is the positive half of the
selectivity claim: refusal in the counterfactual test was not a blanket incapacity.
"""
import os, sys, json, random, re
import torch

sys.path.insert(0, os.path.dirname(__file__))
from run_rev_inversion_protocol import (
    Tokenizer, DEVICE, D_MODEL, N_HEADS, N_LAYERS, L_CAP, PatchProjector,
    ENCODER_DIM, CACHE_PATH, SoftGuidedCaptionDecoder, SAVE_DIR, OUT_DIR,
    TEST_START, invert,
)
from run_rev_eval_metrics import COCO_OBJ, objects, lr1, boot_ci

SYN2OBJ = {s: o for o, ss in COCO_OBJ.items() for s in ss if " " not in s}


def make_positive_target(refs, own, rng):
    """Find (ref index, word index) mentioning some real object A, where at least one
    other real object B != A is also present (in `own`); substitute A's word for a
    single-word surface form of B. Returns None if no such swap exists for this image."""
    candidates = []
    for k, ref in enumerate(refs):
        words = ref.strip().split()
        for i, w in enumerate(words):
            wl = re.sub(r"[^a-z]", "", w.lower())
            if wl in SYN2OBJ and SYN2OBJ[wl] != "person":
                a_obj = SYN2OBJ[wl]
                others = [o for o in own if o != a_obj and o != "person"]
                if others:
                    candidates.append((k, i, a_obj, others))
    if not candidates:
        return None
    k, i, a_obj, others = rng.choice(candidates)
    b_obj = rng.choice(others)
    b_words = [s for s in COCO_OBJ[b_obj] if " " not in s]
    if not b_words:
        return None
    b_word = rng.choice(b_words)
    words = refs[k].strip().split()
    punct = re.sub(r"[A-Za-z]", "", words[i])
    words[i] = b_word + punct
    return k, " ".join(words), a_obj, b_obj, b_word


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
        if len(own) < 2:
            continue
        m = make_positive_target(R, own, rng)
        if m and m[4] in tok.w2i:
            jobs.append({"image": rec["image"], "k": m[0], "target": m[1],
                         "a_obj": m[2], "b_obj": m[3], "b_word": m[4],
                         "baseline": rec["baseline"], "ref_used": R[m[0]]})
    print(f"{len(jobs)} positive-edit targets; e.g. '{jobs[0]['ref_used']}' -> '{jobs[0]['target']}'"
          f" ({jobs[0]['a_obj']} -> {jobs[0]['b_obj']})", flush=True)

    out_caps = {v: [] for v in ["free", "onehot", "hidden"]}
    for v in out_caps:
        for s in range(0, len(jobs), 500):
            ch = jobs[s:s + 500]
            idx = [j["image"] for j in ch]
            X = proj(feats[[TEST_START + i for i in idx]].to(DEVICE))
            A = plans[idx].to(DEVICE)
            tgt = torch.tensor([tok.enc(j["target"], max_len=L_CAP, bos=True, eos=True) for j in ch], device=DEVICE)
            _, _, toks = invert(dec, X, A, tgt, v, 300, 0.05)
            out_caps[v] += [tok.dec(t) for t in toks.cpu().tolist()]
        print("done", v, flush=True)

    def success_rate(caps):
        succ = [float(j["b_obj"] in objects(c) or j["b_word"] in c.lower().split()) for c, j in zip(caps, jobs)]
        kept_a = [float(j["a_obj"] in objects(c)) for c, j in zip(caps, jobs)]
        fit = [lr1(c, [j["target"]]) for c, j in zip(caps, jobs)]
        return succ, kept_a, fit

    base_caps = [j["baseline"] for j in jobs]
    b_succ, b_kept_a, b_fit = success_rate(base_caps)

    res = {"n": len(jobs),
           "baseline": {"b_obj_already_mentioned_rate": round(sum(b_succ) / len(b_succ), 4),
                        "fit_to_target": round(sum(b_fit) / len(b_fit), 4)},
           "variants": {}}
    print("baseline (chance rate B already mentioned):", res["baseline"], flush=True)
    for v, caps in out_caps.items():
        succ, kept_a, fit = success_rate(caps)
        res["variants"][v] = {
            "positive_edit_success_rate": round(sum(succ) / len(succ), 4),
            "success_vs_baseline_ci": [round(x, 4) for x in boot_ci([a - b for a, b in zip(succ, b_succ)])],
            "a_obj_retained_rate": round(sum(kept_a) / len(kept_a), 4),
            "fit_to_target": round(sum(fit) / len(fit), 4),
        }
        print(v, res["variants"][v], flush=True)

    # Load the counterfactual (absent-object) injection rate for direct contrast.
    try:
        swap = json.load(open(os.path.join(OUT_DIR, "swap_injection_n1000_s0.json")))
        res["contrast_absent_object_injection"] = {v: swap["variants"][v]["injection"] for v in swap["variants"]}
    except FileNotFoundError:
        res["contrast_absent_object_injection"] = None

    res["examples"] = [{"ref_used": j["ref_used"], "target": j["target"], "a_obj": j["a_obj"], "b_obj": j["b_obj"],
                        "baseline": b, **{v: out_caps[v][q] for v in out_caps}}
                       for q, (j, b) in enumerate(zip(jobs[:30], base_caps[:30]))]
    res["records"] = [{"image": j["image"], "k": j["k"], "target": j["target"], "a_obj": j["a_obj"],
                       "b_obj": j["b_obj"], **{v: out_caps[v][q] for v in out_caps}} for q, j in enumerate(jobs)]

    print("", flush=True)
    print("VERDICT:", flush=True)
    free_succ = res["variants"]["free"]["positive_edit_success_rate"]
    free_absent_inj = res["contrast_absent_object_injection"]["free"] if res["contrast_absent_object_injection"] else None
    print(f"  free plan: positive-edit (supported, B present) success = {free_succ:.4f}"
          + (f"  vs  absent-object (counterfactual) injection = {free_absent_inj:.4f}" if free_absent_inj is not None else ""), flush=True)
    if free_absent_inj is not None and free_succ > free_absent_inj * 3:
        print("  -> Free plan imports SUPPORTED content far more readily than UNSUPPORTED content: "
              "selectivity is not a blanket incapacity, it discriminates by image support.", flush=True)
    else:
        print("  -> Free plan's positive-edit success is NOT clearly higher than its absent-object injection rate: "
              "the earlier low injection rate may reflect limited capacity more than genuine grounding.", flush=True)

    with open(os.path.join(OUT_DIR, "supported_edit_control.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("saved results/revision/supported_edit_control.json")


if __name__ == "__main__":
    main()
