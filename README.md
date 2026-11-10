# Auditing Attention-Level Oracles with Held-Out and Shuffled Targets

Code for the HOST protocol paper: an audit for target-conditioned attention
inversion ("oracle") results in image captioning, combining held-out
references, shuffled targets, and dimension-matched intervention controls
to separate genuine image-predictable improvement from fitting the
optimisation target.

This replaces an earlier, unrelated snapshot of this repository, and the
code is organised into one directory per pipeline stage (previously
everything sat flat at the repo root). The code below is the one actually
used to produce every number in the current paper; the previous commit's
plan-arithmetic / geometry-diagnostic scripts belonged to an earlier,
superseded phase of this project and have been removed.

## Layout

- **`clean_pipeline/`** — shared base classes, the three-stage clean
  training pipeline, and the primary evaluation driver.
- **`inversion/`** — plan inversion and the counterfactual swap cohort;
  depended on by every other stage below, so it stays its own package
  rather than nested under any one of them.
- **`visual_fault/`** — the constructed feature-space fault, fault-aware
  repair, the donor/all-mean provenance checks, and the severity
  dose-response sweep.
- **`matched_effect/`** — validation-calibrated effect matching and the
  unbounded-bias control.
- **`pretrained_audits/`** — the BLIP-2, InstructBLIP, and Qwen2.5-VL
  secondary audits.
- **`analysis/`** — rescoring, table/figure assembly, and corruption/semantic
  scoring scripts that consume saved results; no CVPR-specific content
  despite scripts here still saying so in a few historical path strings.
- **`historical/`** — ancillary archives the paper's Limitations section
  explicitly marks as non-primary (see below); not meant to be run to
  reproduce a numbered result.

## Setup

```bash
pip install -r requirements.txt
```

All scripts assume a working directory containing a `results/` folder
(created automatically) for checkpoints, cached features, and JSON result
files. Data is a 15,000-image MS-COCO subset with five references each,
split 13,000/1,000/1,000 (train/val/test) and held disjoint at every
training stage. A single CUDA GPU is expected; scripts default to CUDA.

## Pipeline order

1. **`clean_pipeline/run_summarization.py`**, **`clean_pipeline/run_captioning.py`**,
   **`clean_pipeline/run_sum_masked.py`** — shared base classes (tokenizer,
   patch projector, caption decoder, masking utilities). Not run directly;
   imported by everything else.
2. **`clean_pipeline/run_cap_improved.py`** — trains the three-stage clean
   pipeline (free captioner, CFG denoiser, guided decoder) across seeds
   42/7/13.
3. **`clean_pipeline/clean_gpu_guard.py`**, **`clean_pipeline/run_clean_pipeline.py`**,
   **`clean_pipeline/run_clean_captioner.py`**, **`clean_pipeline/run_clean_semantics.py`**,
   **`clean_pipeline/run_clean_evaluation.py`** — orchestrate training, GPU
   leasing, and the primary evaluation (forward controls, the 5+1
   held-out/shuffled protocol, and the counterfactual swap cohort).
4. **`inversion/run_rev_inversion_protocol.py`** — free/on-manifold/hidden-state
   plan inversion; produces the primary decomposition (nominal gap →
   single-reference → held-out).
5. **`inversion/run_rev_swap_injection.py`** — the 901-image counterfactual
   object-injection cohort.

## Reproducing each paper result

| Paper section | Script(s) |
|---|---|
| Table II (nominal-gap decomposition), Table III (HOST protocol) | `inversion/run_rev_inversion_protocol.py`, `clean_pipeline/run_clean_evaluation.py` |
| Reference-redundancy null (Sec. IV-C) | rescoring of the same inversion-protocol output against the shuffled job's donor-image references; see `analysis/rescore_rouge.py` |
| Table IV (constructed visual fault, fault-aware repair) | `visual_fault/run_visual_headroom_control.py`, `analysis/score_visual_headroom.py` |
| Fig. 2 (recovery depends on the correct image: repair *and* free-plan inversion, Sec. IV-E) | `visual_fault/run_visual_provenance_control.py`, `visual_fault/run_rev_host_w1_donor_nullmean.py`, `analysis/make_visual_recovery_figure.py` (reads both results, plotted as two clearly-separated groups since they score different protocols: repair against all five references, inversion held-out) |
| Fault-severity (K_a) dose-response | `visual_fault/run_rev_dose_response.py` |
| Table V (validation-calibrated effect matching) | `matched_effect/run_rev_matched_effect_control.py`, `matched_effect/run_rev_matched_effect_audit.py` |
| Unbounded-bias control (Sec. V-C) | `matched_effect/run_rev_unbounded_logit_control.py`, `matched_effect/run_rev_unbounded_logit_audit.py` |
| Content-word selectivity | `matched_effect/run_rev_selectivity.py` |
| BLIP-2 audit (Sec. VI) | `pretrained_audits/run_blip2_prediction_gap.py`, `pretrained_audits/run_rev_blip2_protocol.py`, `pretrained_audits/run_rev_blip2_forward.py` |
| InstructBLIP audit (Sec. VI) | `pretrained_audits/run_rev_instructblip_protocol.py` |
| Qwen2.5-VL audit (Sec. VI) | `pretrained_audits/run_rev_qwen_protocol.py`, `pretrained_audits/run_rev_qwen_audit.py` |
| Tables/figures assembly | `analysis/make_tables.py`, `analysis/summarize_clean_results.py`, `analysis/calibrate_host.py` |

`test_*.py` (under `clean_pipeline/`, `visual_fault/`, and `analysis/`) are
the unit/regression tests for the corresponding scripts above. `analysis/`'s
own test suite is discovered from the repo root with
`python -B -m unittest discover -s analysis -p "test_*.py"`; the others run
with `pytest clean_pipeline/ visual_fault/` (or plain `unittest discover`
from the repo root). `matched_effect/` and `pretrained_audits/` have no
dedicated test files; their scripts import-check cleanly but exercise GPU-
and model-dependent code paths that this code-only checkout can't run.

A handful of tests that read real cached results (e.g.
`results/revision/inversion_protocol_n1000_s0.json`,
`results/visual_headroom_provenance/results.json`) will fail here by
design, since no checkpoints or result JSONs are distributed with this
repository (see below) — that's a missing-fixture failure, not a code bug.

## `historical/`

Ancillary archives discussed in the paper's Limitations as **not**
clean-pipeline replications and **not** evidence for any primary claim:
full-COCO-scale forward/inversion runs on a non-clean split
(`run_rev_fullcoco_*.py`), an earlier denoiser data/capacity-scaling sweep
and remedy attempts (`run_rev_den_*.py`, `run_rev_denoiser_scaling.py`,
`run_rev_distill_contentmask.py`, `run_rev_remedies_seeds.py`,
`run_rev_retrain_nmp1_fresh.py`, `run_rev_supported_edit_control.py`), and
a predictor-failure diagnostic (`run_rev_host_predictor_failure*.py`,
`verify_rev_host_predictor_failure.py`). Kept for transparency, not meant
to be run to reproduce a numbered table. A few of these reference two
scripts from an earlier, pre-clean-pipeline phase (`run_text_denoiser.py`,
`run_plan_inversion.py`, `run_rev_interventions.py`) that predate this
reorganisation and aren't part of this repository at all; that gap is not
new here and isn't resolved by this reorganisation.

## Notes on reproducibility

- All primary-pipeline scripts hash-lock their inputs (trained-checkpoint
  and config SHA-256s recorded alongside every result JSON, now keyed by
  each file's path including its subdirectory) and assert frozen model
  weights / no gradient leakage before writing output.
- Reported confidence intervals are paired image-level bootstraps (10,000
  resamples, seed 0) after averaging the three training seeds; they
  condition on three fixed trained models, not on independent images.
- No checkpoints, cached features, or result JSONs are included in this
  repository — only the code that produces them.

## License

MIT — see `LICENSE`.
