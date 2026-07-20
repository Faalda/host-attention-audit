# Attention Plans as Diffusion Latents for Controllable Image Captioning

Code for reproducing the experiments in the paper. Cross-attention in an
image captioner is made an explicit diffusion latent (the *attention
plan*), enabling test-time editing (patch suppression, plan transplant),
gradient-based *plan inversion* to quantify the prediction gap, and a suite
of plan-space geometry diagnostics.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # required for noun extraction (suppression eval)
```

All scripts assume a working directory containing a `results/` folder
(created automatically) for checkpoints, cached features, and JSON result
files. Data is MS-COCO, loaded via `datasets.load_dataset("clip-benchmark/wds_mscoco_captions")`
(downloaded automatically on first use). A single GPU with ~16GB+ memory is
recommended; scripts default to CUDA if available and fall back to CPU.

## Pipeline order

The core pipeline must run once before most downstream experiments/diagnostics,
since they load cached DINOv2 features and trained checkpoints from `results/`.

1. **`run_captioning.py`** / **`run_summarization.py`** / **`run_sum_masked.py`** —
   shared base classes (patch projector, caption decoder, tokenizer, masking
   utilities, ROUGE evaluation). Not run directly; imported by everything else.
2. **`run_cap_improved.py`** — trains the full 4-phase pipeline (free
   captioner, GNN-style CFG denoiser, guided decoder) and evaluates the
   MaskGIT `K`-step sweep. Produces the main captioning-quality table
   (ROUGE-1 vs. `K`, free/oracle baselines) and caches patch features +
   checkpoints used by every later script.
3. **`run_save_plans.py`** — extracts and saves attention plans (`A0`) plus
   labels needed by the plan-space geometry diagnostics.

## Reproducing each result

| Paper result | Script(s) |
|---|---|
| Captioning quality, `K`-sweep (main ROUGE-1 table) | `run_cap_improved.py` |
| Standard metrics: CIDEr-D, BLEU-4, ROUGE-L | `run_standard_eval.py` |
| Recovery accuracy (Top-1/Top-5 vs. mask rate) | `run_recovery_top5.py` |
| Capacity-fidelity tradeoff (GNN vs. cross-attention denoiser) | `run_cap_xattn.py`, `run_extended_seeds.py` (extends the seed count; edit `NEW_SEEDS` to add more) |
| Patch suppression + 3-strategy comparison (random/global/token-conditioned) | `run_exp3_token_suppression.py` |
| Qualitative editing figures (suppression, transplant examples) | `run_make_figures.py` |
| Plan inversion / prediction gap | `run_plan_inversion.py --eval --n-images 1000` |
| Prediction-gap distribution figure | `plot_inversion_histogram.py` (run after plan inversion) |
| Text-conditioned denoiser (forward-side gap-closing attempt) | `run_text_denoiser.py` (train), `run_text_cfg_sweep.py --n-eval 500` (guidance-scale sweep + image-only baseline) |
| Oracle-plan distillation (forward-side gap-closing attempt) | `run_oracle_distillation.py --n-eval 500` |
| BLIP-2 (OPT-2.7B) prediction-gap replication | `run_blip2_prediction_gap.py` |
| Plan arithmetic: DINOv2 vs. CLIP encoder | `run_plan_arithmetic.py`, `run_clip_plan_arithmetic.py` |

### Plan-space geometry diagnostics (D1–D10)

| Diagnostic | Script |
|---|---|
| D1: Covariance anisotropy | `run_anisotropy_analysis.py` |
| D2: Nearest-neighbour plan retrieval | `run_plan_retrieval.py` |
| D3: Logistic probe (location vs. concept) | `run_where_vs_what_probe.py` |
| D4: ZCA whitening | `run_whitening_arithmetic.py` |
| D5: Residual nonlinear editor | `run_residual_editor.py` |
| D6: Soft-metric (CLIP-text) baseline | `run_soft_metric_baseline.py` |
| D7: Stable Diffusion cross-attention injection | `run_sd_injection.py` (requires `diffusers`, downloads SD 1.5 weights) |
| D8: Value-augmented plans (`A0 @ V`) | `run_cap_v2.py` |
| D9: Slot attention | `run_slot_cap.py` |
| D10: Token-conditioned suppression | `run_exp3_token_suppression.py` (same script as the 3-strategy comparison above) |

## Notes on reproducibility

- Seeds are set explicitly in each script (see individual `--help` / argparse
  defaults); the paper's tables report mean ± std over the seeds listed in
  each table caption.
- `run_extended_seeds.py`, `run_oracle_distillation.py`, and
  `run_text_cfg_sweep.py` cache intermediate results (oracle plans, CLIP text
  embeddings) keyed by run size, so re-running at a different `--n-eval`
  regenerates rather than silently reusing mismatched cached data.
- Some diagnostics (D7, BLIP-2 replication) download additional pretrained
  weights (Stable Diffusion 1.5, BLIP-2 OPT-2.7B) on first run via
  `transformers`/`diffusers`.

## License

MIT — see `LICENSE`.
