"""CPU-only, frozen-test-corpus CIDEr-D sidecar; no inference or legacy edits.

Run with ../.venv/bin/python3 -B paper_cvpr/rescore_clean_cider.py.
Publication is exclusive; --check recomputes and compares bytes without writing.
"""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import tempfile
from types import MappingProxyType

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.cider.cider_scorer import CiderScorer

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 7, 13)
VARIANTS = ("free", "onehot", "hidden")
PROTOCOLS = ("single_ref", "heldout", "mismatch_target", "mismatch_grounding")
ALLREFS = "each_inversion_all_refs_avg"
SYSTEMS = ("free", "guided_uniform", "guided_denoised")
FRACTIONS = ("0", "0.25", "0.5", "1")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def norm(text):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenCiderScorer(CiderScorer):
    """Reuse upstream clipping, length penalty and averaging, freezing only IDF."""

    def __init__(self, document_frequency, n_documents):
        self._log_n = np.log(float(n_documents))
        super().__init__(n=4, sigma=6.0)
        self.document_frequency = document_frequency

    @property
    def ref_len(self):
        return self._log_n

    @ref_len.setter
    def ref_len(self, value):
        # Upstream __init__ and compute_cider assign None/log(job count).
        # Neither may replace the frozen log(image count).
        pass

    def compute_doc_freq(self):
        raise RuntimeError("Frozen document frequencies cannot be refitted")

    def compute_score(self, option=None, verbose=0):
        scores = np.asarray(self.compute_cider(), dtype=float)
        return float(scores.mean()), scores


class FixedIDFCider:
    def __init__(self, references):
        require(len(references) > 1, "IDF needs at least two image documents")
        require(all(len(rs) == 5 and all(isinstance(r, str) and r for r in rs)
                    for rs in references), "Expected five reference slots per image")
        corpus = CiderScorer()
        for refs in references:
            corpus += (None, [norm(r) for r in refs])
        corpus.compute_doc_freq()
        self.n_documents = len(references)
        # Counter returns zero for unseen hypothesis ngrams without inserting keys.
        self.document_frequency = MappingProxyType(Counter(corpus.document_frequency))
        self.cache = {}

    def score(self, predictions, references, batch_size=256):
        require(len(predictions) == len(references) and len(predictions) > 0,
                "Nonempty aligned predictions/references required")
        require(batch_size > 0, "Positive batch size required")
        require(all(isinstance(p, str) for p in predictions)
                and all(rs and all(isinstance(r, str) for r in rs) for rs in references),
                "Missing prediction or references")
        keys = [(norm(p) or "empty", tuple(norm(r) for r in rs))
                for p, rs in zip(predictions, references)]
        missing = list(dict.fromkeys(key for key in keys if key not in self.cache))
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            scorer = FrozenCiderScorer(self.document_frequency, self.n_documents)
            for prediction, refs in batch:
                scorer += (prediction, refs)
            _, values = scorer.compute_score()
            require(np.isfinite(values).all(), "Nonfinite CIDEr-D score")
            self.cache.update(zip(batch, map(float, values)))
        return np.array([self.cache[key] for key in keys])


def protocol_scores(metric, records, variant, protocol, slots=range(5), batch_size=256):
    require(protocol in (*PROTOCOLS, ALLREFS), "Unknown protocol")
    slots = tuple(slots)
    require(slots and len(set(slots)) == len(slots) and all(k in range(5) for k in slots),
            "Invalid target slots")
    predictions, baselines, references = [], [], []
    for record in records:
        refs = record["refs"]
        for k in slots if protocol in ("single_ref", "heldout", ALLREFS) else (-1,):
            predictions.append(record["variants"][variant]["true"][k] if k >= 0
                               else record["variants"][variant]["mismatch"])
            baselines.append(record["baseline"])
            references.append([refs[k]] if protocol == "single_ref" else
                              [r for j, r in enumerate(refs) if j != k] if protocol == "heldout" else
                              [record["mismatch_target"]] if protocol == "mismatch_target" else refs)
    return tuple(metric.score(caps, references, batch_size).reshape(len(records), -1).mean(1)
                 for caps in (predictions, baselines))


def paired(a, b, resamples=10000, seed=0):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    require(a.ndim == 1 and a.size and a.shape == b.shape
            and np.isfinite(a).all() and np.isfinite(b).all(), "Invalid paired vectors")
    delta = a - b
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, resamples, 250):
        indices = rng.integers(0, len(a), (min(250, resamples - start), len(a)))
        draws.extend(delta[indices].mean(1))
    return {"n_images": len(a), "inv": float(a.mean()), "base": float(b.mean()),
            "delta": float(delta.mean()), "ci95": np.percentile(draws, [2.5, 97.5]).tolist(),
            "frac_images_pos": float((delta > 0).mean()),
            "frac_images_neg": float((delta < 0).mean()),
            "per_image": {"inv": a.tolist(), "base": b.tolist()}}


def aggregate(rows):
    require(set(rows) == set(SEEDS), "Exactly three trained seeds required")
    a = np.array([rows[s][0] for s in SEEDS], dtype=float)
    b = np.array([rows[s][1] for s in SEEDS], dtype=float)
    require(a.ndim == 2 and a.shape == b.shape, "Misaligned seed/image arrays")
    result = paired(a.mean(0), b.mean(0))
    result["per_seed"] = {str(s): paired(*rows[s]) for s in SEEDS}
    result["seed_minmax"] = {key: [min(r[key] for r in result["per_seed"].values()),
                                       max(r[key] for r in result["per_seed"].values())]
                             for key in ("inv", "base", "delta")}
    return result


def validate_dump(data, original, slots, variants):
    n = len(original)
    require(data.get("partial") is False and data["n_images"] == n
            and data["evaluated_true_slots"] == list(slots)
            and data["variants"] == list(variants), "Incomplete or unexpected dump")
    require([r["image"] for r in data["records"]] == list(range(n)), "Image alignment mismatch")
    for record, source in zip(data["records"], original):
        for key in ("image", "feature_index", "refs", "mismatch_source", "mismatch_target"):
            require(record[key] == source[key], f"Protocol mismatch: {key}")
        require(isinstance(record["baseline"], str) and set(record["variants"]) == set(variants),
                "Missing baseline/variant")
        for variant in variants:
            jobs = record["variants"][variant]
            require(len(jobs["true"]) == 5 and isinstance(jobs["mismatch"], str), "Missing jobs")
            require(all(isinstance(c, str) if k in slots else c is None
                        for k, c in enumerate(jobs["true"])), "Incomplete target slots")


def publish(path, text, check=False):
    path = Path(path)
    if check:
        require(path.read_text() == text, "Sidecar is not byte-for-byte reproducible")
        return
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic publication, never replacement.
    finally:
        os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=ROOT / "results/clean_eval_stablekl")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("clean_cider_results.json"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    require(args.batch_size > 0, "Positive batch size required")
    require(args.check or not args.output.exists(), "Refusing to overwrite sidecar; use --check")
    inventory = {}

    def track(path):
        path = path.resolve()
        name = path.relative_to(ROOT).as_posix()
        digest = sha256(path)
        require(name not in inventory or inventory[name] == digest, f"Input changed: {name}")
        inventory[name] = digest
        return path

    def load(path, receipt=None):
        path = track(path)
        if receipt is not None:
            require(inventory[path.relative_to(ROOT).as_posix()] == receipt["hashes"][path.name],
                    f"Receipt hash mismatch: {path.name}")
        with path.open() as stream:
            return json.load(stream)

    for name in ("rescore_clean_cider.py", "test_rescore_clean_cider.py"):
        track(Path(__file__).with_name(name))
    track(ROOT / "run_rev_eval_metrics.py")
    package = importlib.metadata.distribution("pycocoevalcap")
    package_paths = {name: Path(package.locate_file(name)) for name in
                     ("pycocoevalcap/cider/cider.py", "pycocoevalcap/cider/cider_scorer.py")}
    package_hashes = {name: sha256(path) for name, path in package_paths.items()}
    rows, forward_errors = {}, {}
    identity = metric = None

    def add(key, seed, values):
        require(seed not in rows.setdefault(key, {}), "Duplicate comparison")
        rows[key][seed] = values

    for seed in SEEDS:
        print(f"Scoring fixed-IDF seed {seed}", flush=True)
        folder = args.eval_root / f"seed{seed}"
        receipt = load(folder / "pipeline_eval_receipt.json")
        complete = load(folder / "complete.json", receipt)
        require(complete["all_inputs_unchanged"] and complete["all_model_parameters_frozen"]
                and complete["model_state_sha256_before"] == complete["model_state_sha256_after"],
                "Incomplete source evaluation")
        data = load(folder / "inversion_protocol_n1000_s0.json", receipt)
        records = data["records"]
        require(len(records) == 1000, "Expected full 1000-image test corpus")
        validate_dump(data, records, range(5), VARIANTS)
        current = [{k: r[k] for k in ("image", "feature_index", "refs", "mismatch_source", "mismatch_target")}
                   for r in records]
        if identity is None:
            identity = current
            metric = FixedIDFCider([r["refs"] for r in records])
        require(current == identity, "Seeds do not share images, references and donors")
        require(sorted(r["mismatch_source"] for r in records) == list(range(1000)), "Invalid donor permutation")
        for i, r in enumerate(records):
            require(r["feature_index"] == 14000 + i and r["mismatch_source"] != i
                    and r["mismatch_target"] in records[r["mismatch_source"]]["refs"],
                    "Invalid test index or shuffled target")
        for variant in VARIANTS:
            for protocol in (*PROTOCOLS, ALLREFS):
                add(f"primary/cider_fixed_idf/{variant}/{protocol}", seed,
                    protocol_scores(metric, records, variant, protocol, batch_size=args.batch_size))
            for protocol in ("single_ref", "heldout", ALLREFS):
                add(f"strict_k01/cider_fixed_idf/{variant}/{protocol}", seed,
                    protocol_scores(metric, records, variant, protocol, (0, 1), args.batch_size))
        forward = load(folder / "forward.json", receipt)
        refs = [r["refs"] for r in records]
        require(forward["n_images"] == 1000 and forward["image_ids"] == list(range(1000))
                and forward["refs"] == refs and set(forward["caps"]) == set(SYSTEMS), "Forward misalignment")
        require(forward["caps"]["guided_denoised"] == [r["baseline"] for r in records],
                "Protocol baseline differs from forward")
        vectors = {}
        forward_errors[str(seed)] = {}
        for system in SYSTEMS:
            captions = forward["caps"][system]
            values = metric.score(captions, refs, args.batch_size)
            standard = Cider().compute_score(
                {i: [norm(r) for r in rs] for i, rs in enumerate(refs)},
                {i: [norm(c) or "empty"] for i, c in enumerate(captions)})[1]
            error = float(np.max(np.abs(values - standard)))
            require(error <= 1e-12, "Fixed IDF disagrees with standard full-reference forward")
            forward_errors[str(seed)][system] = error
            vectors[system] = values
            add(f"forward/cider_fixed_idf/{system}", seed, (values, np.zeros(1000)))
        for system in ("free", "guided_uniform"):
            add(f"forward/cider_fixed_idf/denoised_minus_{system}", seed,
                (vectors["guided_denoised"], vectors[system]))
        for fraction in FRACTIONS:
            corruption = load(folder / f"corruption_{fraction}.json", receipt)
            validate_dump(corruption, records, (0, 1), ("free", "repair"))
            require(corruption["metadata"]["corruption_fraction"] == float(fraction), "Wrong corruption fraction")
            for r, clean in zip(corruption["records"], records):
                require(r["clean_baseline"] == clean["baseline"]
                        and r["variants"]["repair"]["true"][:2] == [clean["baseline"]] * 2
                        and r["variants"]["repair"]["mismatch"] == clean["baseline"], "Invalid clean repair")
                require(fraction != "0" or r["baseline"] == clean["baseline"], "Zero corruption baseline changed")
            prefix = f"corruption/{fraction}/cider_fixed_idf"
            for variant in ("free", "repair"):
                for protocol in (*PROTOCOLS, ALLREFS):
                    add(f"{prefix}/{variant}/{protocol}", seed,
                        protocol_scores(metric, corruption["records"], variant, protocol, (0, 1), args.batch_size))
            free, repair = (rows[f"{prefix}/{v}/heldout"][seed] for v in ("free", "repair"))
            require(np.array_equal(free[1], repair[1]), "Nonidentical corruption baselines")
            zero = rows["corruption/0/cider_fixed_idf/free/heldout"][seed]
            require(np.array_equal(zero[1], repair[0]), "Repair does not recover clean heldout baseline")
            add(f"{prefix}/free_minus_repair_heldout", seed, (free[0], repair[0]))
            add(f"{prefix}/baseline_loss_heldout", seed, (zero[1], free[1]))
            add(f"{prefix}/free_gain_change_from_zero", seed, (free[0] - free[1], zero[0] - zero[1]))

    print("Bootstrapping paired images, after averaging jobs then three fixed models", flush=True)
    comparisons = {key: aggregate(by_seed) for key, by_seed in rows.items()}
    df_rows = [[list(gram), count] for gram, count in sorted(metric.document_frequency.items())]
    df_hash = hashlib.sha256(json.dumps(df_rows, separators=(",", ":")).encode()).hexdigest()
    result = {
        "schema_version": 1, "status": "complete_hash_verified", "device": "cpu",
        "training_or_model_inference_performed": False, "seeds": list(SEEDS),
        "eval_root": args.eval_root.resolve().relative_to(ROOT).as_posix(),
        "image_ids": list(range(1000)), "feature_indices": list(range(14000, 15000)),
        "metric": {
            "name": "CIDEr-D fixed test-corpus IDF", "comparison_namespace": "cider_fixed_idf",
            "implementation": "pycocoevalcap CiderScorer.compute_cider; n=4, sigma=6, clipping, Gaussian length penalty, raw x10 scale (not x100)",
            "normalization": "Exact run_rev_eval_metrics.norm: lowercase ASCII alphanumeric, punctuation to spaces, collapse whitespace; empty hypothesis -> 'empty'; no PTB",
            "idf": "Computed ONCE from all five reference slots of all 1000 test images, union of 1..4-grams within each image document; log(1000)-log(max(1,df)); frozen across protocols, seeds, jobs and bootstrap",
            "n_documents": metric.n_documents, "n_reference_slots": 5000,
            "n_distinct_ngrams": len(df_rows), "document_frequency_sha256": df_hash,
            "document_frequency_serialization": "sha256 of JSON sorted [ngram-token-list,count] rows, separators=(',',':')",
            "scope": "Controlled fixed-statistics cross-protocol comparison, NOT a claim of conventional per-protocol refitted-IDF CIDEr. Different reference sets still define different tasks; common IDF does not equalize their difficulty.",
            "leakage_policy": "All test references enter unsupervised scoring IDF only, including excluded target slots. This post-hoc CPU rescore performs no optimization, training, model/step selection or caption changes. Heldout similarity excludes target slot exactly; not a fully reference-unseen IDF estimate.",
            "identical_captions": "Not always 10: absent higher-order ngrams and zero-IDF orders contribute zero; multi-reference scores average similarities, not max.",
        },
        "statistical_policy": {
            "ci": "Paired percentile image bootstrap, 10000 resamples, numpy.default_rng(0), 95%; average five jobs within each image, then three aligned seed scores, THEN resample 1000 images; IDF stays fixed",
            "exceptions": "Shuffle and forward: one job/image; strict_k01 and corruption: two true jobs/image (k=0,1)",
            "scope": "Conditional on these three trained models and this frozen reference corpus; NOT training-seed uncertainty, not 3000 IID images; no multiplicity correction",
            "seed_minmax": "Descriptive range, not CI",
        },
        "definitions": {
            "single_ref": "Seen target: mean_k score(y_k,[r_k]) vs same baseline refs, no flooring",
            "heldout": "Mean_k score(y_k,R excluding slot k), baseline excludes same slot; duplicate strings in other slots retained",
            "mismatch_target": "Shuffled inversion vs baseline, scored against saved donor target only",
            "mismatch_grounding": "Shuffled inversion vs baseline, scored jointly against own five references",
            ALLREFS: "Mean_k score(y_k,R), joint multi-reference CIDEr-D (not max)",
            "forward": "All five refs; system rows have base=0 as absolute-score reporting only; denoised_minus_* are paired controls",
            "corruption": "k=0,1; free and reference-free repair share corrupted baseline and exactly matched reference slots; free_minus_repair_heldout is paired gain difference",
        },
        "verification": {"consumed_dumps_match_eval_receipts": True, "sources_unchanged": True,
                         "standard_forward_max_abs_error": forward_errors,
                         "upstream_scope": "Caption dumps and completion receipts verified; upstream model/feature files are not loaded or re-audited by this scoring-only tool"},
        "integration": "Load comparisons as a separate cider_fixed_idf namespace alongside clean_results_summary.json. Do not relabel old cider rows. Use primary/cider_fixed_idf/{variant}/{protocol}; inv/base/delta/ci95 and per_seed fields are summary-compatible. Source hashes bind the sidecar to the dumps. Summary and papers are not edited.",
        "comparisons": comparisons,
        "versions": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                     for name in ("numpy", "pycocoevalcap")}},
        "package_sha256": package_hashes, "source_sha256": inventory,
    }
    for name, digest in inventory.items():
        require(sha256(ROOT / name) == digest, f"Source changed during rescore: {name}")
    for name, path in package_paths.items():
        require(sha256(path) == package_hashes[name], f"Package changed: {name}")
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    publish(args.output, text, args.check)
    print("Verified" if args.check else "Wrote", args.output)
    for key, row in comparisons.items():
        if key.startswith(("primary/", "forward/")):
            print(f"{key}: {row['inv']:.9f}/{row['base']:.9f}; delta={row['delta']:+.9f} CI={row['ci95']}")
    return result


if __name__ == "__main__":
    main()
