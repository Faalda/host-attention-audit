"""Offline saved-caption BERTScore; GPU 0 only, no generation or training.

Run: PYTHONHASHSEED=0 ../.venv/bin/python -B paper_cvpr/score_semantics.py
Writes semantic_results.json only. A timing pilot selects the complete datasets
or the prespecified first 200 small / first 500 full-COCO images, never random jobs.
"""

import os

# Set before importing torch/transformers, including when launched without env flags.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

import numpy as np

from rescore_rouge import paired, version


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path("/home/faldabbas/.cache/huggingface/hub/models--roberta-large/snapshots/"
                "722cf37b1afa9454edce342e7895e588b6ff1d59")
VARIANTS = ("free", "onehot", "hidden")
SYSTEMS = ("guided_denoised", "guided_uniform")


def file_info(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.absolute()), "sha256": digest,
            "size_bytes": path.stat().st_size}


def validate(small, captions, references):
    records = sorted(small["records"], key=lambda r: r["image"])
    ids = [r["image"] for r in records]
    if (small.get("partial") or len(records) != small["n_images"] or
            len(set(ids)) != len(ids) or set(small["variants"]) != set(VARIANTS)):
        raise ValueError("Incomplete small protocol or unexpected variants")
    for record in records:
        if len(record["refs"]) != 5 or not all(isinstance(r, str) for r in record["refs"]):
            raise ValueError("Expected five reference slots")
        if not isinstance(record["baseline"], str) or not isinstance(record["mismatch_target"], str):
            raise ValueError("Missing baseline or mismatch target")
        for variant in VARIANTS:
            jobs = record["variants"][variant]
            if len(jobs["true"]) != 5 or not all(isinstance(c, str) for c in jobs["true"]):
                raise ValueError("All variants must have the same five complete target slots")
            if not isinstance(jobs["mismatch"], str):
                raise ValueError("Missing mismatch caption")
    refs, keys = references["caps"], references["keys"]
    if len(refs) != 5000 or len(keys) != 5000 or len(set(keys)) != 5000:
        raise ValueError("Expected complete 5000-image reference source")
    if any(len(rs) != 5 or not all(isinstance(r, str) for r in rs) for rs in refs):
        raise ValueError("Expected five string references per full-COCO image")
    for name in SYSTEMS:
        if len(captions[name]) != 5000 or not all(isinstance(c, str) for c in captions[name]):
            raise ValueError("Caption/reference length mismatch")
    return records


def small_image(record, variant, score):
    """Exclude target k by SLOT for both output and baseline, then average jobs."""
    refs = record["refs"]
    jobs = record["variants"][variant]
    if len(refs) != 5 or len(jobs["true"]) != 5 or any(c is None for c in jobs["true"]):
        raise ValueError("Expected all five aligned target slots")
    base = np.array([score(record["baseline"], r) for r in refs])
    mat = np.array([[score(c, r) for r in refs] for c in jobs["true"]])
    held = [max(mat[k, j] for j in range(5) if j != k) for k in range(5)]
    held_base = [max(base[j] for j in range(5) if j != k) for k in range(5)]
    human = [max(score(refs[k], refs[j]) for j in range(5) if j != k) for k in range(5)]
    return {
        "single_ref": (float(mat.diagonal().mean()), float(base.mean())),
        "heldout": (float(np.mean(held)), float(np.mean(held_base))),
        "each_inversion_all_refs_avg": (float(mat.max(1).mean()), float(base.max())),
        "human_heldout": (float(np.mean(human)), float(np.mean(held_base))),
        "mismatch_target": (score(jobs["mismatch"], record["mismatch_target"]),
                            score(record["baseline"], record["mismatch_target"])),
        "mismatch_grounding": (max(score(jobs["mismatch"], r) for r in refs), float(base.max())),
    }


def collect_pairs(records, captions, references, full_n):
    # Ordered deduplication saves matching work too; do not merge reversed pairs.
    pairs = {}

    def collect(a, b):
        pairs.setdefault((a, b), None)
        return 0.0

    for record in records:
        for variant in VARIANTS:
            small_image(record, variant, collect)
    for name in SYSTEMS:
        for caption, refs in zip(captions[name][:full_n], references["caps"][:full_n]):
            for ref in refs:
                collect(caption, ref)
    return list(pairs)


def aggregate(records, captions, references, full_n, score):
    small = {"n_images": len(records), "image_ids": [r["image"] for r in records],
             "target_slots": list(range(5)), "variants": {}}
    for variant in VARIANTS:
        rows = [small_image(r, variant, score) for r in records]
        small["variants"][variant] = {
            key: paired([row[key][0] for row in rows], [row[key][1] for row in rows])
            for key in rows[0]
        }
    values = {name: [max(score(c, r) for r in refs)
                     for c, refs in zip(captions[name][:full_n], references["caps"][:full_n])]
              for name in SYSTEMS}
    comparison = paired(values[SYSTEMS[0]], values[SYSTEMS[1]])
    full = {"n_images": full_n, "image_ids": references["keys"][:full_n],
            "systems": {
                name: {"mean": comparison[side], "ci95": comparison[side + "_ci95"],
                       "per_image": values[name]}
                for name, side in zip(SYSTEMS, ("inv", "base"))},
            "paired": {"denoised_minus_uniform": comparison}}
    return {"small": small, "fullcoco_uniform": full}


def make_scorer(snapshot=SNAPSHOT):
    import torch
    from bert_score import BERTScorer
    from bert_score.utils import sent_encode
    from transformers import AutoTokenizer

    if version("bert-score") != "0.3.13":
        raise RuntimeError("Requires bert-score==0.3.13; install with --no-deps")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly physical GPU 0 must be visible")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    scorer = BERTScorer(model_type=str(snapshot), num_layers=17, idf=False,
                        rescale_with_baseline=False, device="cuda:0", use_fast_tokenizer=False)
    # Transformers 5 replaced the slow RoBERTa tokenizer and ignores the encode-
    # time prefix flag used by canonical sent_encode. Restore that exact setting.
    expected = [0, 83, 4758, 6476, 4, 2]  # <s> + byte-BPE(" A cat sits.") + </s>
    repaired = sent_encode(scorer._tokenizer, "A cat sits.") != expected
    if repaired:
        scorer._tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), local_files_only=True, use_fast=False, add_prefix_space=True)
    if sent_encode(scorer._tokenizer, "A cat sits.") != expected:
        raise RuntimeError("Canonical RoBERTa prefix-space tokenization failed")
    if not hasattr(scorer._tokenizer, "build_inputs_with_special_tokens"):
        # Canonical sent_encode uses this removed v5 method for empty strings.
        scorer._tokenizer.build_inputs_with_special_tokens = lambda ids: [0, *ids, 2]
    scorer._model.float().eval()
    if len(scorer._model.encoder.layer) != 17 or any(m.training for m in scorer._model.modules()):
        raise RuntimeError("Layer/eval-mode invariant failed")
    return scorer, repaired


def audit_tokenization(tokenizer, texts, snapshot=SNAPSHOT):
    """Independently check legacy RoBERTa byte-BPE IDs on every scored string.

    This deliberately audits the old regex/merge algorithm, not a second call to
    the same Rust tokenizer. Literal special-token strings require a separate
    audit and are rejected rather than silently accepted by this natural-text run.
    """
    import regex
    from bert_score.utils import sent_encode

    vocab = json.loads((snapshot / "vocab.json").read_text())
    merges = (snapshot / "merges.txt").read_text().splitlines()[1:]
    ranks = {tuple(line.split()): i for i, line in enumerate(merges) if line}
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    unicode_values = byte_values.copy()
    next_codepoint = 256
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            unicode_values.append(next_codepoint)
            next_codepoint += 1
    byte_encoder = dict(zip(byte_values, map(chr, unicode_values)))
    pattern = regex.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")

    @lru_cache(maxsize=None)
    def bpe(token):
        word = tuple(byte_encoder[b] for b in token.encode("utf-8"))
        while len(word) > 1:
            pair = min(zip(word, word[1:]), key=lambda p: ranks.get(p, float("inf")))
            if pair not in ranks:
                break
            merged, i = [], 0
            while i < len(word):
                if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
                    merged.append(word[i] + word[i + 1])
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = tuple(merged)
        return [vocab[token] for token in word]

    digest = hashlib.sha256()
    maximum, truncated, empty = 0, 0, 0
    for text in sorted(texts):
        if any(special in text for special in tokenizer.all_special_tokens):
            raise ValueError("Literal special token requires an explicit legacy-tokenization audit")
        stripped = text.strip()
        ids = [idx for token in pattern.findall(" " + stripped) for idx in bpe(token)] if stripped else []
        maximum = max(maximum, len(ids) + 2)
        truncated += len(ids) > 510
        empty += not stripped
        expected = [0, *ids[:510], 2]
        if sent_encode(tokenizer, text) != expected:
            raise ValueError(f"Legacy byte-BPE tokenization mismatch for {text!r}")
        digest.update(json.dumps([text, expected], ensure_ascii=True).encode() + b"\n")
    return {"passed": True, "unique_strings_checked": len(texts),
            "method": "independent legacy RoBERTa regex + byte mapping + ranked BPE merges from cached vocab/merges",
            "max_untruncated_tokens_including_special": maximum,
            "truncated_strings": truncated, "empty_strings": empty,
            "sorted_text_and_token_ids_sha256": digest.hexdigest()}


def smoke(scorer):
    from bert_score.utils import sent_encode

    a, b = "A cat sits on the mat.", "A dog runs through a field."
    cands, refs = [a, a, b, "", a], [a, b, a, a, ""]
    first = np.stack([x.numpy() for x in scorer.score(cands, refs, batch_size=32)])
    second = np.stack([x.numpy() for x in scorer.score(cands, refs, batch_size=32)])
    np.testing.assert_allclose(first[:, 0], 1, atol=2e-6, rtol=0)
    np.testing.assert_allclose(first[2, 1], first[2, 2], atol=2e-6, rtol=0)
    np.testing.assert_allclose(first[0, 1], first[1, 2], atol=2e-6, rtol=0)
    np.testing.assert_allclose(first, second, atol=2e-6, rtol=0)
    np.testing.assert_array_equal(first[:, 3:], 0)
    assert sent_encode(scorer._tokenizer, "A cat sits.") == [0, 83, 4758, 6476, 4, 2]
    assert sent_encode(scorer._tokenizer, "  A cat sits.  ") == [0, 83, 4758, 6476, 4, 2]
    assert sent_encode(scorer._tokenizer, "a cat sits.") != sent_encode(scorer._tokenizer, "A cat sits.")
    return {"passed": True, "identical_f1": float(first[2, 0]),
            "swapped_f1_abs_error": float(abs(first[2, 1] - first[2, 2])),
            "repeat_max_abs_error": float(np.max(abs(first - second))),
            "empty_f1": first[2, 3:].tolist(), "prefix_case_and_strip_checks": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    started = time.perf_counter()
    scorer, repaired = make_scorer()
    checks = {"smoke": smoke(scorer)}
    if args.smoke_only:
        print(json.dumps(checks, indent=2))
        return
    timings = {"model_load_and_smoke_seconds": time.perf_counter() - started}
    sources = {}

    def load(relative):
        path = ROOT / relative
        sources[relative] = file_info(path)
        return json.loads(path.read_text())

    small = load("results/revision/inversion_protocol_n1000_s0.json")
    captions = load("results/full_coco/captions_test.json")
    references = load("results/full_coco/caps_test.json")
    records = validate(small, captions, references)
    all_pairs = collect_pairs(records, captions, references, 5000)
    unique_texts = set(text for pair in all_pairs for text in pair)
    tick = time.perf_counter()
    checks["tokenization_audit"] = audit_tokenization(scorer._tokenizer, unique_texts)
    timings["tokenization_audit_seconds"] = time.perf_counter() - tick
    # Pilot spans the entire ordered pair list, not an outcome-selected subset.
    pilot = [all_pairs[i] for i in np.linspace(0, len(all_pairs) - 1, 512, dtype=int)]
    tick = time.perf_counter()
    scorer.score([p[0] for p in pilot], [p[1] for p in pilot], batch_size=args.batch_size)
    timings["pilot_seconds"] = time.perf_counter() - tick
    pilot_unique = len(set(text for pair in pilot for text in pair))
    # Conservative: scales the entire pilot (encoding AND matching) by pair count.
    projected = timings["pilot_seconds"] * len(all_pairs) / len(pilot) * 1.25
    fallback = projected + time.perf_counter() - started > 900
    selection = {
        "policy": "Complete sources unless projected/actual run exceeds 900 seconds; then first 200 small images sorted by image ID and first 500 full-COCO rows, all variants and slots",
        "projected_scoring_seconds_conservative": projected,
        "available_small_images": len(records), "available_fullcoco_images": 5000,
        "full_unique_pairs": len(all_pairs), "full_unique_strings": len(unique_texts),
        "pilot_pairs": len(pilot), "pilot_unique_strings": pilot_unique,
        "fallback_used": fallback, "reason": "pilot projection" if fallback else "full run fits budget",
    }

    def timeout(signum, frame):
        raise TimeoutError("Full scoring reached the 15-minute budget")

    signal.signal(signal.SIGALRM, timeout)
    while True:
        selected = records[:200] if fallback else records
        full_n = 500 if fallback else 5000
        pairs = collect_pairs(selected, captions, references, full_n) if fallback else all_pairs
        print(f"Scoring small n={len(selected)} all 3 variants; full-COCO n={full_n}; "
              f"{len(pairs)} unique pairs; projected full seconds={projected:.1f}", flush=True)
        tick = time.perf_counter()
        if not fallback:
            signal.alarm(max(1, int(900 - (tick - started))))
        try:
            _, _, f1 = scorer.score([p[0] for p in pairs], [p[1] for p in pairs],
                                    batch_size=args.batch_size, verbose=True)
            break
        except TimeoutError:
            fallback = True
            selection.update(fallback_used=True, reason="actual full run timeout")
            timings["abandoned_full_scoring_seconds"] = time.perf_counter() - tick
        finally:
            signal.alarm(0)
    timings["scoring_seconds"] = time.perf_counter() - tick
    scores = dict(zip(pairs, f1.tolist()))
    if not np.isfinite(f1.numpy()).all():
        raise ValueError("Nonfinite BERTScore")
    tick = time.perf_counter()
    datasets = aggregate(selected, captions, references, full_n, lambda a, b: scores[a, b])
    timings["aggregation_bootstrap_seconds"] = time.perf_counter() - tick
    for relative, info in sources.items():
        if file_info(ROOT / relative) != info:
            raise RuntimeError(f"Source changed during scoring: {relative}")
    checks["complete_input_validation"] = True
    checks["source_hashes_unchanged"] = True
    import torch
    import bert_score.utils
    snapshot_files = {p.name: file_info(p) for p in sorted(SNAPSHOT.iterdir()) if p.is_file()}
    timings["total_seconds_before_json_write"] = time.perf_counter() - started
    result = {
        "schema_version": 1,
        "metadata": {
            "metric": "canonical bert-score 0.3.13 raw F1", "model": "roberta-large",
            "snapshot": str(SNAPSHOT), "snapshot_revision": SNAPSHOT.name,
            "snapshot_files": snapshot_files, "num_layers": 17, "idf": False,
            "rescale_with_baseline": False, "all_layers": False,
            "dtype": "float32", "autocast": False, "tf32": False, "model_eval": True,
            "device": "cuda:0", "physical_gpu": 0, "cuda_visible_devices": "0",
            "gpu_name": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "offline": True, "model_downloads": False, "training": False,
            "batch_size": args.batch_size, "unique_scored_pairs": len(pairs),
            "unique_scored_strings": len(set(text for pair in pairs for text in pair)),
            "embedding_cache": "canonical bert_cos_score_idf deduplicates all strings in one combined call; CPU FP32 token embedding cache",
            "tokenizer": {"class": type(scorer._tokenizer).__name__,
                          "is_fast_actual": scorer._tokenizer.is_fast,
                          "use_fast_requested": False, "prefix_space_compatibility_fix": repaired,
                          "normalization": "canonical sent_encode strip; prefix space; original case/punctuation; max_length=512 truncation"},
            "python": sys.version, "executable": sys.executable,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "versions": {k: version(k) for k in ("bert-score", "transformers", "tokenizers", "torch", "numpy", "huggingface-hub", "safetensors", "rouge-score", "regex")},
            "sources": sources,
            "implementation": file_info(Path(__file__).resolve()),
            "tests": file_info(Path(__file__).with_name("test_score_semantics.py")),
            "bootstrap_implementation": file_info(Path(__file__).with_name("rescore_rouge.py")),
            "canonical_implementation": file_info(Path(bert_score.utils.__file__)),
            "bootstrap": {"resamples": 10000, "seed": 0, "unit": "image",
                          "method": "paired percentile [2.5,97.5]", "rng": "numpy.default_rng PCG64",
                          "seed_reset_per_comparison": True},
            "definitions": {
                "heldout": "mean_k max_{j != k} F1(output_k, ref_j), k=0..4; baseline excludes identical slot k, then average within image",
                "single_ref": "mean_k F1(output_k, ref_k), baseline mean_k F1(baseline, ref_k)",
                "each_inversion_all_refs_avg": "mean_k max_j F1(output_k, ref_j)",
                "human_heldout": "mean_k max_{j != k} F1(ref_k, ref_j)",
                "mismatch_target": "F1(mismatch_output, mismatch_target), paired against same baseline target",
                "mismatch_grounding": "max_j F1(mismatch_output, own_ref_j)",
                "fullcoco_uniform": "max_j F1(single_saved_caption, ref_j) over all five refs, then mean over images",
                "per_image": "Arrays follow image_ids; inv is intervention/denoised, base is baseline/uniform; delta=inv-base"},
            "limitations": [
                "Text-reference similarity is not an image-grounding or factuality test.",
                "Raw unrescaled BERTScore has a high similarity floor; absolute levels are not calibrated accuracy.",
                "Full-COCO captions lack their own image IDs; alignment relies on the saved evaluator's shared row order, validated by complete lengths, reference keys and source hashes.",
                "Installed Transformers 5 uses a tokenizers-backed RoBERTa tokenizer even with use_fast=False; restored prefix-space behavior was checked against independent legacy byte-BPE for every unique input string.",
                "The removed tokenizer build_inputs_with_special_tokens method is restored for canonical empty-string handling (<s>, </s>).",
                "Unused RoBERTa pooler parameters are initialized at load; BERTScore uses only token hidden states, not the pooler.",
                "Intervals are image-bootstrap uncertainty for these saved outputs, not training-seed uncertainty; no multiple-comparison adjustment.",
            ],
        },
        "selection": selection, "timings": timings, "verification": checks, "datasets": datasets,
    }
    out = Path(__file__).with_name("semantic_results.json")
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Wrote", out, flush=True)
    for variant in VARIANTS:
        for metric, row in datasets["small"]["variants"][variant].items():
            print(variant, metric, {k: row[k] for k in ("n_images", "inv", "base", "delta", "ci95")})
    for name, row in datasets["fullcoco_uniform"]["systems"].items():
        print(name, {k: row[k] for k in ("mean", "ci95")})
    print("denoised_minus_uniform", {k: datasets["fullcoco_uniform"]["paired"]["denoised_minus_uniform"][k]
                                     for k in ("n_images", "inv", "base", "delta", "ci95")})
    print("Timings", timings)


if __name__ == "__main__":
    main()
