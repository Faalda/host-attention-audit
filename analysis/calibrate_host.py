"""Analytic synthetic sanity check, NOT vision-model empirical sensitivity.

CPU only; no training, tensors, model imports, downloads, or optimization.
Run: ../.venv/bin/python -B analysis/calibrate_host.py (from project root,
adjust the interpreter path if necessary). Writes only calibrate_host_results.json.

The finite world contains 200 symbolic two-region tasks, not rendered images or
200 object categories. Object identities are supplied, not visually recognized.
A binary route selects one supported object's fixed caption. An independently
assigned region marker answers the query without caption references. Corrupting
that answer supplies a known number of recoverable errors, not measured errors.

Five templated paraphrases include two synonym-only targets. Standard ROUGE has
no synonym knowledge: these targets tie across routes, and ties retain baseline.
Thus the lexical oracle need not recover all known semantic routing headroom.
Scaffold overlap gives wrong-route captions nonzero baseline scores; held-out
mean is reported alongside HOST's leave-one-reference-out maximum. References
are distinct strings but NOT independent human descriptions or held-out scenes.

The nonrouting control selects baseline or the supplied target string. Both
channels have two choices, but copying has a target-dependent payload/codebook:
this is NOT a dimensionality/architecture/optimizer-matched HOST experiment.
An additional exact two-output mixture matches target fit PER JOB in expectation;
it is neither a deterministic edit nor an independently calibrated model. Zero
route gain forces zero copy dose and cannot establish structural selectivity.

This checks bookkeeping/sensitivity under deliberately constructed lexical
headroom. It cannot replace the requested degraded-trained-predictor experiment,
validate vision sensitivity, or alter any paper claim. No confidence intervals:
we enumerate this entire designed finite population, not a sampled population.
"""

from collections import defaultdict
from functools import lru_cache
import hashlib
import importlib.metadata
from itertools import product
import json
import math
from pathlib import Path
import platform

from rouge_score import rouge_scorer, tokenizers


# Single-token, disjoint names keep absent-object membership unambiguous.
OBJECTS = (
    ("dog", "canine"), ("cat", "feline"), ("car", "automobile"),
    ("bicycle", "bike"), ("sofa", "couch"), ("rabbit", "bunny"),
    ("horse", "equine"), ("airplane", "plane"), ("boat", "vessel"),
    ("phone", "telephone"),
)
CUES = ("striped", "dotted")
ERROR_COUNTS = (0, 10, 20, 50, 100, 200)
SCORER = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)
TOKENIZER = tokenizers.DefaultTokenizer(use_stemmer=False)


@lru_cache(maxsize=50000)
def score(prediction, reference):
    return SCORER.score(reference, prediction)["rouge1"].fmeasure


def references(object_id):
    noun, synonym = OBJECTS[object_id]
    return [f"a {noun} appears in the picture", f"the image shows a {noun}",
            f"there is a {synonym} in view", f"a {noun} can be seen here",
            f"this scene contains a {synonym}"]


def decode(regions, route):
    if route not in (0, 1):
        raise ValueError("Exactly two routes are allowed")
    return f"a {OBJECTS[regions[route]['object']][0]} is visible in the scene"


def cue_predictor(regions, query):
    """Reference-free symbolic predictor; no task ID or true-route label input."""
    matches = [i for i, region in enumerate(regions) if region["cue"] == query]
    if len(matches) != 1:
        raise ValueError("Query must identify exactly one region")
    return matches[0]


def make_tasks():
    tasks = []
    # 25 object pairs x 2 layouts x 2 marker assignments x 2 queries = 200.
    for a, b, layout, cue_layout, query in product(range(5), range(5, 10),
                                                  range(2), range(2), CUES):
        pair = (a, b) if layout == 0 else (b, a)
        regions = [{"object": obj, "cue": CUES[i ^ cue_layout]}
                   for i, obj in enumerate(pair)]
        route = CUES.index(query) ^ cue_layout
        tasks.append({"id": len(tasks), "regions": regions, "query": query,
                      "true_route": route, "refs": references(pair[route])})
    # A fixed bijection on tasks: advance BOTH category indices modulo five.
    # Donor queried object is therefore absent from both recipient regions.
    lookup = {(tuple(r["object"] for r in t["regions"]),
               tuple(r["cue"] for r in t["regions"]), t["query"]): t["id"]
              for t in tasks}
    for task in tasks:
        shifted = tuple((r["object"] // 5) * 5 + (r["object"] + 1) % 5
                        for r in task["regions"])
        task["shuffled_donor_id"] = lookup[
            shifted, tuple(r["cue"] for r in task["regions"]), task["query"]]
    return tasks


def error_ids(tasks, count):
    if not isinstance(count, int) or not 0 <= count <= len(tasks):
        raise ValueError("Error count must be an integer in [0, number of tasks]")
    # Nested deterministic corruption, independent of caption wording/query.
    ranked = sorted((t["id"] for t in tasks),
                    key=lambda i: hashlib.sha256(f"host-route-noise-v1:{i}".encode()).digest())
    return set(ranked[:count])


def exact_route(captions, target, baseline_route):
    """Enumerate both candidates using ONLY the fitting target, not held-out refs."""
    if len(captions) != 2 or baseline_route not in (0, 1):
        raise ValueError("Expected two captions and a binary baseline route")
    fits = [score(c, target) for c in captions]
    other = 1 - baseline_route
    return other if fits[other] > fits[baseline_route] else baseline_route


def reference_scores(caption, refs, excluded_slot=None):
    values = [score(caption, r) for j, r in enumerate(refs) if j != excluded_slot]
    if not values:
        raise ValueError("No held-out references")
    return {"own_reference_best": max(values),
            "own_reference_mean": math.fsum(values) / len(values)}


def copy_dose(base_fit, route_fit):
    """Exact expectation: (1-p)*base_fit + p*1 = route_fit."""
    if not 0 <= base_fit <= route_fit <= 1:
        raise ValueError("Exact route must not reduce fitting-target score")
    return 0.0 if base_fit == 1 else (route_fit - base_fit) / (1 - base_fit)


def evaluate(tasks, count):
    errors = error_ids(tasks, count)
    sums = defaultdict(lambda: defaultdict(list))
    doses = defaultdict(list)
    match_errors = []
    route_correct = []
    baseline_wrong = []
    recovered_wrong = []
    by_target_type = defaultdict(list)
    task_gains = []
    for task in tasks:
        regions, refs = task["regions"], task["refs"]
        correct = task["true_route"]
        recovered = cue_predictor(regions, task["query"])
        baseline = recovered ^ int(task["id"] in errors)
        baseline_wrong.append(float(baseline != correct))
        recovered_wrong.append(float(recovered != correct))
        captions = [decode(regions, r) for r in (0, 1)]
        supported = {r["object"] for r in regions}
        desired = regions[correct]["object"]
        donor = tasks[task["shuffled_donor_id"]]
        gains = []
        for condition, targets in (("true", refs), ("shuffled", donor["refs"])):
            for k, target in enumerate(targets):
                route = exact_route(captions, target, baseline)

                def metrics(caption):
                    tokens = set(TOKENIZER.tokenize(caption))
                    named = {i for i, names in enumerate(OBJECTS) if tokens.intersection(names)}
                    return {"target_fit": score(caption, target),
                            **reference_scores(caption, refs, k if condition == "true" else None),
                            "desired_object_mention": float(desired in named),
                            "unsupported_object_mention": float(bool(named - supported))}

                values = {"baseline": metrics(captions[baseline]),
                          "route_oracle": metrics(captions[route]),
                          "cue_predictor": metrics(captions[recovered]),
                          "target_copy": metrics(target)}
                dose = copy_dose(values["baseline"]["target_fit"],
                                 values["route_oracle"]["target_fit"])
                values["fit_matched_copy"] = {
                    key: (1 - dose) * value + dose * values["target_copy"][key]
                    for key, value in values["baseline"].items()}
                doses[condition].append(dose)
                match_errors.append(abs(values["fit_matched_copy"]["target_fit"]
                                        - values["route_oracle"]["target_fit"]))
                for system, measurements in values.items():
                    for key, value in measurements.items():
                        sums[condition, system][key].append(value)
                if condition == "true":
                    gain = (values["route_oracle"]["own_reference_best"]
                            - values["baseline"]["own_reference_best"])
                    gains.append(gain)
                    by_target_type["synonym" if k in (2, 4) else "canonical"].append(gain)
                    route_correct.append(float(route == correct))
        task_gains.append(math.fsum(gains) / len(gains))
    mean = lambda xs: math.fsum(xs) / len(xs)
    result = {"error_count": count, "wrong_route_fraction": mean(baseline_wrong),
              "corrupted_task_ids": sorted(errors),
              "true_oracle_wrong_route_fraction": 1 - mean(route_correct),
              "cue_predictor_wrong_route_fraction": mean(recovered_wrong),
              "per_task_true_heldout_best_gain": task_gains,
              "target_type_heldout_best_gain": {k: mean(v) for k, v in by_target_type.items()},
              "maximum_per_job_copy_fit_matching_error": max(match_errors)}
    for condition in ("true", "shuffled"):
        systems = {system: {key: mean(v) for key, v in values.items()}
                   for (c, system), values in sums.items() if c == condition}
        base = systems["baseline"]
        for measurements in systems.values():
            measurements["delta_own_reference_best"] = measurements["own_reference_best"] - base["own_reference_best"]
            measurements["delta_own_reference_mean"] = measurements["own_reference_mean"] - base["own_reference_mean"]
        result[condition] = {"systems": systems,
                             "matched_copy_mean_dose": mean(doses[condition]),
                             "matched_copy_zero_dose_fraction": mean([float(p == 0) for p in doses[condition]])}
    return result


def run():
    tasks = make_tasks()
    grid = [evaluate(tasks, n) for n in ERROR_COUNTS]
    close = lambda a, b: math.isclose(a, b, abs_tol=1e-12)
    all_wrong = grid[-1]["true"]["systems"]
    checks = {
        "200_tasks_and_bijective_absent_object_donors": len(tasks) == 200
        and len({t["shuffled_donor_id"] for t in tasks}) == 200
        and all(not {r["object"] for r in t["regions"]}.intersection(
                    r["object"] for r in tasks[t["shuffled_donor_id"]]["regions"]) for t in tasks),
        "distinct_paraphrases_no_decoder_copy": all(len(set(t["refs"])) == 5
            and all(decode(t["regions"], r) not in t["refs"] for r in (0, 1)) for t in tasks),
        "zero_error_no_oracle_headroom": close(grid[0]["true"]["systems"]["route_oracle"]["delta_own_reference_best"], 0),
        "positive_errors_positive_heldout_best_and_mean": all(
            row["true"]["systems"]["route_oracle"]["delta_own_reference_best"] > 0
            and row["true"]["systems"]["route_oracle"]["delta_own_reference_mean"] > 0 for row in grid[1:]),
        "known_fraction_and_partial_lexical_recovery": all(
            close(row["wrong_route_fraction"], row["error_count"] / 200)
            and close(row["cue_predictor_wrong_route_fraction"], 0)
            and close(row["true_oracle_wrong_route_fraction"], 0.4 * row["wrong_route_fraction"])
            and close(row["target_type_heldout_best_gain"]["synonym"], 0) for row in grid),
        "linear_known_error_response": all(close(
            row["true"]["systems"][s]["delta_own_reference_best"],
            row["wrong_route_fraction"] * all_wrong[s]["delta_own_reference_best"])
            for row in grid for s in ("route_oracle", "cue_predictor")),
        "shuffled_route_cannot_inject_or_improve_fit": all(
            close(row["shuffled"]["systems"]["route_oracle"]["unsupported_object_mention"], 0)
            and close(row["shuffled"]["systems"]["route_oracle"]["target_fit"],
                      row["shuffled"]["systems"]["baseline"]["target_fit"]) for row in grid),
        "copy_can_fit_and_inject_shuffled_targets": all(
            close(row["shuffled"]["systems"]["target_copy"]["target_fit"], 1)
            and close(row["shuffled"]["systems"]["target_copy"]["unsupported_object_mention"], 1) for row in grid),
        "shuffled_copy_exposes_shared_template_confound": all(
            row["shuffled"]["systems"]["target_copy"]["delta_own_reference_best"] > 0 for row in grid),
        "exact_per_job_expected_fit_match": all(row["maximum_per_job_copy_fit_matching_error"] < 1e-12 for row in grid),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Synthetic validation failed; no results written: {checks}")
    return {
        "label": "ANALYTIC SYNTHETIC SANITY CHECK; NOT VISION-MODEL EMPIRICAL SENSITIVITY",
        "credible_minimal_synthetic_check": True,
        "satisfies_requested_degraded_model_calibration": False,
        "metadata": {
            "device": "cpu", "model_inference": False, "optimization": False,
            "images": False, "training": False, "downloads": False,
            "python": platform.python_version(), "rouge_score_version": importlib.metadata.version("rouge-score"),
            "metric": "standard rouge_score ROUGE-1 F1, use_stemmer=False; default alphanumeric lowercase tokenizer",
            "population": "200 tasks; 10 categories; 25 cross-group pairs; all layouts, marker assignments and queries",
            "references": "5 hand-written template paraphrases: 3 canonical noun, 2 synonym noun; no learned paraphraser",
            "heldout": "true: mean over fitting slots k of max/mean score over j != k; same exclusion for every system",
            "shuffled": "fixed absent-object donor permutation; each donor ref is a target; own-reference scores use all 5 recipient refs",
            "oracle": "argmax target ROUGE over exactly 2 supported route captions; ties retain baseline; no held-out input",
            "corruption": "exact count of flipped cue-predictor routes; nested SHA256 task-ID order; no noise fitting",
            "recoverability": "deterministic cue lookup using supplied symbolic regions and query only; not a learned image predictor",
            "matched_control": "two-output baseline/target-copy mixture; per-job expected target fit matched analytically, not architecture matched",
            "uncertainty": "no CIs/p-values: complete designed finite population; floating-point scores, exact discrete enumeration",
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "tests_sha256": hashlib.sha256(Path(__file__).with_name("test_calibrate_host.py").read_bytes()).hexdigest(),
        },
        "interpretation": [
            "Yes only as a minimal analytic implementation check: known route errors yield held-out lexical gain without optimization.",
            "The oracle repairs only canonical-target errors; synonym-only targets tie. The independent cue reader repairs all errors.",
            "Shared scaffolds produce nonzero wrong-route and shuffled-target overlap. Absent-object rejection is built into the output support.",
            "Shuffled target copying increases own-reference overlap despite injecting an absent object, because donor and recipient share templates. True-target copying does not improve leave-one-out maximum here. Neither outcome is imposed by the evaluator.",
            "Shuffled routing has zero gain; fit matching then forces zero copy dose. This matched null cannot distinguish channel structure.",
            "Names, decoder, noise, cues and lexical overlap are constructed. These are computed toy results, not observed vision experiments.",
            "Reference slots are held out, not templates/scenes. No learned generalization, optimizer sensitivity, real paraphrase robustness or architecture comparison is established.",
            "The finite oracle maximizes target ROUGE directly, unlike the paper's gradient-based likelihood inversion; this does not calibrate that optimizer.",
            "The requested degraded-trained-model positive calibration remains unrun; no paper claims or sources were changed.",
        ],
        "verification": checks, "object_lexicon": OBJECTS, "tasks": tasks, "error_grid": grid,
    }


def main():
    result = run()
    path = Path(__file__).with_name("calibrate_host_results.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(result["label"])
    for row in result["error_grid"]:
        systems = row["true"]["systems"]
        print(f"wrong={row['wrong_route_fraction']:.2f} "
              f"base={systems['baseline']['own_reference_best']:.6f} "
              f"route_gain={systems['route_oracle']['delta_own_reference_best']:+.6f} "
              f"cue_gain={systems['cue_predictor']['delta_own_reference_best']:+.6f}")
    print("Validated and wrote", path)


if __name__ == "__main__":
    main()
