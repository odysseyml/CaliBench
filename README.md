# CaliBench

**Calibration benchmark for image-to-video world models.** CaliBench measures whether a video generator
reproduces the *correct distribution* of physical outcomes — not just plausible motion. Each of nine
stochastic scenes (a die roll, a Galton board, a pendulum released to swing left or right, …) is
image-conditioned from a single frame with a known ground-truth outcome distribution; a model is scored by
how far the distribution of outcomes across many generations departs from that reference, aggregated into a
single headline number, **mnTV** (lower is better).

This repository is the reproducible code-and-data supplement behind the paper: the generation and
outcome-extraction pipeline, the committed extracted outcomes for every model, and the analyses that
produce the paper's tables and figures.

## How it works

For each `(scene, model)` cell the pipeline:

1. **Generates** 32 image-conditioned rollouts from the scene's conditioning frame + prompt
   (`generate_rollouts.py`, via the Replicate API).
2. **Extracts** the discrete outcome of each rollout with a vision-language model (Gemini), using 3-vote
   majority (`extract_bin.py`). Rollouts with no legible outcome are recorded as `null`.
3. **Scores** the cell: **scorability** `ρ` (fraction of rollouts with a legible outcome) and the
   **total-variation** distance between the empirical outcome distribution and the scene's analytic
   reference, corrected by a finite-sample **null band**. The per-cell score combines illegibility and
   excess-TV; **mnTV** is the mean across scenes, with a bootstrap 95% CI. A Monte-Carlo χ² goodness-of-fit
   test and power analysis accompany it.

The scene definitions (conditioning frame + prompt) and the reference distributions are the fixed benchmark
specification; everything else is measurement.

## Installation

```bash
pip install -r requirements.txt
```

Generation and extraction call paid APIs; set the keys before running those stages:

```bash
export REPLICATE_API_TOKEN=...   # video generation
export GEMINI_API_KEY=...        # VLM outcome extraction
```

## Quick start

The extracted outcomes for every `(scene, model)` cell are committed under `analysis/` as
`<scene>_<model>_bins.json`, so the **paper's numbers reproduce with no API calls**:

```bash
python compute_tv_band.py      # per-cell TV + null band + mnTV (the headline metric)
python compute_mntv_ci.py      # bootstrap 95% CI on mnTV
python compute_chi2.py         # per-cell chi^2, p-values, total-variation
python make_scatter_grid.py    # 3x3 scorability-vs-TV figure
```

To run the full pipeline end-to-end (generation → extraction → analysis → figures/tables), including the
appendix ablations and the pendulum reference check:

```bash
./run_pipeline.sh
```

## Repository structure

**Benchmark specification & shared code**
- `scenes.py` — the nine scenes: conditioning-frame stem + generation prompt.
- `constants.py` — reference outcome distributions, per-scene outcome counts, and shared constants.
- `conditioning_frames/` — the single conditioning frame per scene.
- `stats_utils.py` — Monte-Carlo χ² goodness-of-fit utilities.

**Generation** (Replicate API)
- `generate_rollouts.py` — 32 image-to-video rollouts per `(scene, model)`.
- `generate_seedance720.py`, `generate_dur5s.py` — appendix ablations (720p resolution; uniform-5s duration).

**Extraction** (Gemini VLM)
- `extract_bin.py` — 3-vote outcome extraction; writes `analysis/<scene>_<model>_bins.json`.
- `extract_seedance720_gemini.py`, `extract_dur5s_gemini.py`, `extract_dice_studies.py` — ablation extractors.
- `validate_extraction.py` — VLM-vs-human agreement table.

**Analysis & metric**
- `compute_tv_band.py` — per-cell TV, null band, and **mnTV** (headline).
- `compute_mntv_ci.py` — bootstrap 95% CI on mnTV.
- `compute_chi2.py` — χ² test, p-values, total-variation per cell.
- `compute_mcar_sensitivity.py` — best/worst-case null-imputation sensitivity.
- `power_chi2.py` — detectability / power analysis at N=32.
- `analyze_rollouts.py` — per-cell scorability + empirical distributions.
- `verify_pendulum_v3.py` — simulation check that the pendulum frame yields Bernoulli(½).

**Figures & tables**
- `make_scatter_grid.py`, `make_rollout_examples.py` — paper figures.
- `make_config_table.py` — per-model generation-settings table (config disclosure).
- `make_seedance_ablation.py`, `make_dice_ablation_figures.py`, `recompute_dur5s.py` — appendix ablations.

**Data**
- `analysis/` — the committed extracted outcomes (`*_bins.json`) plus human-label validation data and the
  ablation bins; analysis scripts write their JSON/TeX/figure outputs here.

**Cosmos-Predict3 (Cosmos3-Super)**
- `cosmos/cosmos_config.py`, `cosmos/generate_cosmos3_minimal.py` — the standalone config + generator for
  the one non-Replicate model.

`run_pipeline.sh` runs the stages above in order and documents which steps need paid APIs vs. which
reproduce from the committed data.

## Reproducing the paper's results

`run_pipeline.sh` is the end-to-end driver. The **Generation** and **Extraction** stages call paid APIs and
regenerate `analysis/*_bins.json`; skip them to reproduce every table and figure directly from the
committed outcomes via the **Analysis** and **Figures** stages. Appendix ablations (SeeDance 480p-vs-720p,
uniform-5s duration, the dice CFG-sweep and target-face studies) read their own committed bins and are
regenerated only with the noted generation/extraction scripts.

## Citation

```bibtex
@article{calibench,
  title  = {CaliBench: Are the Stochastic Dynamics of Video World Models Physically Calibrated?},
  author = {Sadeghi, Jonathan and Seidenschwarz, Jenny and Allardice, Jesse and
            Srinivasan, Sirish and Graham, Benjamin and Hawke, Jeffrey},
  year   = {2026},
}
```
