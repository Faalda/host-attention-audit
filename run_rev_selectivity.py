"""run_rev_selectivity.py — grounding selectivity of an inversion interface (revision, positive-evidence test).
For shuffled-target inversions: of the target's content words NOT already in the baseline caption, what fraction
does the inverted caption contain, split by whether the word occurs in the image's own references
(present) or not (absent)? A routing channel should insert present words far more often than absent ones;
a generic control channel inserts both. Bootstrap CI over images for the ratio."""
import sys, json, numpy as np
from run_rev_eval_metrics import norm
STOP = set("a an the of on in with and is are to at by for from it its his her their this that some two three near "
           "next top up down while as be has have there".split())


def selectivity(path, key="mismatch", target_key="mismatch_target", n_boot=2000, seed=0):
    d = json.load(open(path)); out = {}
    for v in d["variants"]:
        rows = []
        for rec in d["records"]:
            y = rec["variants"][v].get(key)
            if y is None:
                continue
            tgt = set(norm(rec[target_key]).split()) - STOP
            own = set(w for r in rec["refs"] for w in norm(r).split())
            b = set(norm(rec["baseline"]).split()); yy = set(norm(y).split())
            cand = tgt - b
            p = [w in yy for w in cand if w in own]; a = [w in yy for w in cand if w not in own]
            rows.append((sum(p), len(p), sum(a), len(a)))
        R = np.array(rows, float)
        def stat(M):
            pr = M[:, 0].sum() / max(M[:, 1].sum(), 1); ar = M[:, 2].sum() / max(M[:, 3].sum(), 1)
            return pr, ar, pr / max(ar, 1e-4)
        pr, ar, ratio = stat(R)
        rng = np.random.default_rng(seed)
        bs = np.array([stat(R[rng.integers(0, len(R), len(R))]) for _ in range(n_boot)])
        out[v] = {"present_rate": round(pr, 4), "absent_rate": round(ar, 4), "ratio": round(ratio, 2),
                  "present_rate_ci": [round(x, 4) for x in np.percentile(bs[:, 0], [2.5, 97.5])],
                  "absent_rate_ci": [round(x, 4) for x in np.percentile(bs[:, 1], [2.5, 97.5])],
                  "ratio_ci": [round(x, 2) for x in np.percentile(bs[:, 2], [2.5, 97.5])],
                  "n_present_words": int(R[:, 1].sum()), "n_absent_words": int(R[:, 3].sum())}
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = selectivity(p); print(p)
        for v, m in r.items():
            print(f"  {v:8s} present {m['present_rate']:.3f} {m['present_rate_ci']}  absent {m['absent_rate']:.3f} {m['absent_rate_ci']}  "
                  f"ratio {m['ratio']:.1f}x {m['ratio_ci']}  (n={m['n_present_words']}/{m['n_absent_words']})")
        json.dump(r, open(p.replace(".json", "_selectivity.json"), "w"), indent=1)
