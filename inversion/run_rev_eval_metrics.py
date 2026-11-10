"""
run_rev_eval_metrics.py — scores the caption dumps written by run_rev_inversion_protocol.py
and run_rev_blip2_protocol.py (revision experiments E1/E2/E8).

Protocols (all paired, all un-floored):
  single_ref : inversion toward r_k scored against r_k        | baseline scored against r_k
  heldout    : inversion toward r_k scored against R \\ {r_k}   | baseline scored against R \\ {r_k}
  mismatch_target    : inversion toward m scored against m     | baseline scored against m (chance)
  mismatch_grounding : inversion toward m scored against R     | baseline scored against R
Metrics: ROUGE-1 (paper's set-unigram F1, for continuity), ROUGE-1/ROUGE-L (rouge_score),
BLEU-4 and CIDEr-D (pycocoevalcap; simple lowercase/punct tokenisation because Java/PTB is
unavailable), exact match, and an object-hallucination proxy (COCO-80 objects mentioned in
the caption but in none of the image's five references).
95% CIs: paired bootstrap over images (10^4 resamples).

Usage: python run_rev_eval_metrics.py results/revision/inversion_protocol_n1000_s0.json
"""
import sys, json, re
import numpy as np
from rouge_score import rouge_scorer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider

import os
sys.path.insert(0, __file__.rsplit("/", 1)[0])
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for clean_pipeline. imports
from clean_pipeline.run_summarization import rouge1_f1 as legacy_r1

_RS = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)

COCO_OBJ = {
    "person": ["person", "people", "man", "men", "woman", "women", "boy", "boys", "girl", "girls", "child",
               "children", "kid", "kids", "baby", "player", "players", "guy", "lady", "skier", "surfer", "skateboarder"],
    "bicycle": ["bicycle", "bike", "bikes", "bicycles"], "car": ["car", "cars"], "motorcycle": ["motorcycle", "motorcycles", "motorbike"],
    "airplane": ["airplane", "airplanes", "plane", "planes", "jet", "jets"], "bus": ["bus", "buses"], "train": ["train", "trains"],
    "truck": ["truck", "trucks"], "boat": ["boat", "boats", "ship"], "traffic light": ["traffic light", "stoplight"],
    "fire hydrant": ["hydrant"], "stop sign": ["stop sign"], "bench": ["bench", "benches"], "bird": ["bird", "birds"],
    "cat": ["cat", "cats", "kitten"], "dog": ["dog", "dogs", "puppy"], "horse": ["horse", "horses"], "sheep": ["sheep"],
    "cow": ["cow", "cows", "cattle"], "elephant": ["elephant", "elephants"], "bear": ["bear", "bears"], "zebra": ["zebra", "zebras"],
    "giraffe": ["giraffe", "giraffes"], "backpack": ["backpack"], "umbrella": ["umbrella", "umbrellas"], "handbag": ["handbag", "purse"],
    "tie": ["tie"], "suitcase": ["suitcase", "luggage"], "frisbee": ["frisbee"], "skis": ["skis", "ski"], "snowboard": ["snowboard"],
    "kite": ["kite", "kites"], "baseball bat": ["bat"], "skateboard": ["skateboard"], "surfboard": ["surfboard"],
    "tennis racket": ["racket", "racquet"], "bottle": ["bottle", "bottles"], "wine glass": ["wine glass"], "cup": ["cup", "cups", "mug"],
    "fork": ["fork"], "knife": ["knife", "knives"], "spoon": ["spoon"], "bowl": ["bowl", "bowls"], "banana": ["banana", "bananas"],
    "apple": ["apple", "apples"], "sandwich": ["sandwich", "sandwiches"], "orange": ["oranges"], "broccoli": ["broccoli"],
    "carrot": ["carrot", "carrots"], "hot dog": ["hot dog", "hotdog"], "pizza": ["pizza", "pizzas"], "donut": ["donut", "donuts", "doughnut", "doughnuts"],
    "cake": ["cake", "cakes"], "chair": ["chair", "chairs"], "couch": ["couch", "sofa"], "potted plant": ["plant", "plants"],
    "bed": ["bed", "beds"], "dining table": ["table", "tables"], "toilet": ["toilet", "toilets"], "tv": ["tv", "television"],
    "laptop": ["laptop", "laptops"], "mouse": ["mouse"], "remote": ["remote"], "keyboard": ["keyboard"], "cell phone": ["phone", "cellphone"],
    "microwave": ["microwave"], "oven": ["oven"], "toaster": ["toaster"], "sink": ["sink", "sinks"], "refrigerator": ["refrigerator", "fridge"],
    "book": ["book", "books"], "clock": ["clock", "clocks"], "vase": ["vase", "vases"], "scissors": ["scissors"],
    "teddy bear": ["teddy"], "hair drier": ["hair dryer"], "toothbrush": ["toothbrush"],
}
_SYN = [(re.compile(r"\b" + re.escape(s) + r"\b"), obj) for obj, ss in COCO_OBJ.items() for s in ss]


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def objects(s):
    s = norm(s)
    return {obj for pat, obj in _SYN if pat.search(s)}


def r1(p, refs):  # rouge_score ROUGE-1 F, best over refs
    return max(_RS.score(norm(r), norm(p))["rouge1"].fmeasure for r in refs)


def rl(p, refs):
    return max(_RS.score(norm(r), norm(p))["rougeL"].fmeasure for r in refs)


def lr1(p, refs):
    return max(legacy_r1(p, r) for r in refs)


def corpus_bleu_cider(preds, refs_list):
    gts = {i: [norm(r) for r in rs] for i, rs in enumerate(refs_list)}
    res = {i: [norm(p) if norm(p) else "empty"] for i, p in enumerate(preds)}
    b, b_per = Bleu(4).compute_score(gts, res, verbose=0)
    c, c_per = Cider().compute_score(gts, res)
    return np.array(b_per[3]), np.array(c_per)


def boot_ci(delta, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(delta, float)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def pair_metrics(preds_inv, preds_base, refs_list, image_ids):
    """Returns per-metric (inv mean, base mean, delta, CI) with the bootstrap unit = image."""
    out = {}
    per = {}
    per["legacy_rouge1"] = ([lr1(p, r) for p, r in zip(preds_inv, refs_list)],
                            [lr1(p, r) for p, r in zip(preds_base, refs_list)])
    per["rouge1"] = ([r1(p, r) for p, r in zip(preds_inv, refs_list)], [r1(p, r) for p, r in zip(preds_base, refs_list)])
    per["rougeL"] = ([rl(p, r) for p, r in zip(preds_inv, refs_list)], [rl(p, r) for p, r in zip(preds_base, refs_list)])
    bi, ci_ = corpus_bleu_cider(preds_inv, refs_list)
    bb, cb = corpus_bleu_cider(preds_base, refs_list)
    per["bleu4"] = (bi, bb)
    per["cider"] = (ci_, cb)
    per["exact"] = ([float(any(norm(p) == norm(x) for x in r)) for p, r in zip(preds_inv, refs_list)],
                    [float(any(norm(p) == norm(x) for x in r)) for p, r in zip(preds_base, refs_list)])
    ids = np.asarray(image_ids)
    for k, (a, b) in per.items():
        a, b = np.asarray(a, float), np.asarray(b, float)
        # aggregate to image level before bootstrapping (several jobs per image)
        uniq = np.unique(ids)
        da = np.array([a[ids == u].mean() for u in uniq])
        db = np.array([b[ids == u].mean() for u in uniq])
        m, lo, hi = boot_ci(da - db)
        out[k] = {"inv": round(float(da.mean()), 4), "base": round(float(db.mean()), 4),
                  "delta": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
                  "frac_images_pos": round(float((da - db > 0).mean()), 3),
                  "frac_images_neg": round(float((da - db < 0).mean()), 3)}
    return out


def halluc(preds, refs_list):
    rates, per_cap = [], []
    for p, rs in zip(preds, refs_list):
        po = objects(p)
        ro = set().union(*[objects(r) for r in rs])
        h = len(po - ro)
        per_cap.append(float(h > 0))
        rates.append(h / len(po) if po else 0.0)
    return {"chair_i_proxy": round(float(np.mean(rates)), 4), "chair_s_proxy": round(float(np.mean(per_cap)), 4)}


def evaluate(path):
    d = json.load(open(path))
    recs = d["records"]
    res = {"file": path, "n_images": len(recs)}
    for v in d["variants"]:
        sr_i, sr_b, sr_refs, sr_ids = [], [], [], []
        ho_i, ho_b, ho_refs = [], [], []
        for rec in recs:
            R = rec["refs"]
            for k, y in enumerate(rec["variants"][v]["true"]):
                if y is None:
                    continue
                sr_i.append(y); sr_b.append(rec["baseline"]); sr_refs.append([R[k]]); sr_ids.append(rec["image"])
                ho_i.append(y); ho_b.append(rec["baseline"]); ho_refs.append([x for j, x in enumerate(R) if j != k])
        mm_i = [rec["variants"][v]["mismatch"] for rec in recs]
        mm_b = [rec["baseline"] for rec in recs]
        ids1 = [rec["image"] for rec in recs]
        vr = {
            "single_ref": pair_metrics(sr_i, sr_b, sr_refs, sr_ids),
            "heldout": pair_metrics(ho_i, ho_b, ho_refs, sr_ids),
            "mismatch_target": pair_metrics(mm_i, mm_b, [[rec["mismatch_target"]] for rec in recs], ids1),
            "mismatch_grounding": pair_metrics(mm_i, mm_b, [rec["refs"] for rec in recs], ids1),
            "halluc_true_inv": halluc(sr_i, [rec["refs"] for rec in recs for _ in rec["variants"][v]["true"] if _ is not None]),
            "halluc_mismatch_inv": halluc(mm_i, [rec["refs"] for rec in recs]),
            "halluc_baseline": halluc(mm_b, [rec["refs"] for rec in recs]),
        }
        st = [s for rec in recs for s in (rec["variants"][v].get("true_stats") or []) if s]
        if st:
            vr["plan_stats_true"] = {k: round(float(np.mean([s[k] for s in st])), 4) for k in st[0]}
        sm = [rec["variants"][v].get("mismatch_stats") for rec in recs if rec["variants"][v].get("mismatch_stats")]
        if sm:
            vr["plan_stats_mismatch"] = {k: round(float(np.mean([s[k] for s in sm])), 4) for k in sm[0]}
        res[v] = vr
    return res


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = evaluate(p)
        outp = p.replace(".json", "_metrics.json")
        json.dump(r, open(outp, "w"), indent=1)
        for v in [k for k in r if k not in ("file", "n_images")]:
            print(f"== {v}")
            for proto in ["single_ref", "heldout", "mismatch_target", "mismatch_grounding"]:
                m = r[v][proto]
                print(f"  {proto:19s} " + "  ".join(
                    f"{k}: {m[k]['inv']:.3f}/{m[k]['base']:.3f} (Δ{m[k]['delta']:+.3f} [{m[k]['ci95'][0]:+.3f},{m[k]['ci95'][1]:+.3f}])"
                    for k in ["legacy_rouge1", "rougeL", "cider"]))
            print("  halluc true/mism/base:", r[v]["halluc_true_inv"], r[v]["halluc_mismatch_inv"], r[v]["halluc_baseline"])
            print("  plan stats:", r[v].get("plan_stats_true"), r[v].get("plan_stats_mismatch"))
        print("saved", outp)
