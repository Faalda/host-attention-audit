"""Audit whether Qwen routing and residual interventions differ before decoding."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()
    data = json.load(open(args.path))
    recs = data["records"]
    fields = ("true", "mismatch", "swap")
    out = {"n_images": len(recs), "comparisons": {}}
    for field in fields:
        pairs = [(r["variants"]["route"].get(field), r["variants"]["resid"].get(field)) for r in recs]
        if field == "true":
            pairs = [(a[0], b[0]) for a, b in pairs if a and b and a[0] is not None and b[0] is not None]
        else:
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        out["comparisons"][field] = {"n": len(pairs), "exact_match_rate": sum(a == b for a, b in pairs) / len(pairs)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
