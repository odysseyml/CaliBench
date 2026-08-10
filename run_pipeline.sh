#!/bin/bash
# End-to-end pipeline: generate rollouts -> extract outcomes -> analyse -> figures/tables,
# including the appendix ablations (SeeDance 480p-vs-720p resolution, dice CFG sweep +
# explicit target-face) and the pendulum Bernoulli(1/2) reference verification.
#
# Requires two API keys in the environment:
#   export REPLICATE_API_TOKEN=...   # for video generation
#   export GEMINI_API_KEY=...        # for VLM outcome extraction
#
# Note: generation and extraction call paid APIs. To reproduce only the
# figures/tables from the committed analysis JSONs, skip the "Generation" and
# "Extraction" sections and run only "Analysis" onward.
set -e
cd "$(dirname "$0")"

SCENES="galton_board galton_board_real ball cards dice lottery pendulum roulette walking"

echo "=== Generation ==="
for scene in $SCENES; do
  echo "  scene: $scene"
  python generate_rollouts.py --scene "$scene" --model all --count 32
done

echo "=== Extraction (VLM) ==="
python extract_bin.py --scene all --model all --votes 3

echo "=== Analysis ==="
python analyze_rollouts.py --scene all --model all   # per-cell scorability/calibration + comparison plots
python compute_chi2.py                               # chi^2, p-values, TV_total -> chi2_results.json
python compute_tv_band.py                            # per-cell TV + null band + mnTV -> tv_band_results.json
python compute_mntv_ci.py                            # bootstrap 95% CI on mnTV
python compute_mcar_sensitivity.py                   # MCAR best/worst null imputation (needs tv_band_results.json)
python power_chi2.py                                 # detectability / power analysis (appendix)
python verify_pendulum_v3.py                         # pendulum Bernoulli(1/2) reference check (simulation only)
python make_config_table.py > analysis/gen_settings_table.tex   # per-model generation-settings table

echo "=== Appendix ablations (read the committed *_bins.json) ==="
python make_seedance_ablation.py                     # 480p-vs-720p resolution table (analysis/seedance720/ bins)
python make_dice_ablation_figures.py                 # CFG-sweep + target-face figures (dice_cfg_* / dice_face_* bins)
python recompute_dur5s.py                            # uniform-5s duration ablation -> ablation_5s_results.md (analysis/dur5s/ bins)
# The bins the three scripts above consume are committed. To regenerate them (paid
# APIs / GPU):
#   * 720p resolution: generate_seedance720.py + extract_seedance720_gemini.py
#   * uniform-5s duration: generate_dur5s.py + extract_dur5s_gemini.py (WAN/SeeDance/
#     HappyHorse; Veo cannot do 5s, Runway/Cosmos already ~5s)
#   * dice studies: Cosmos3-Super videos via cosmos/generate_cosmos3_minimal.py
#     (--num_frames 61, plus --guidance_scale <v> for the CFG sweep or --target_face <n>)
#     and the Replicate target-face videos via dice_target_face_ablation.py, then
#     extract_dice_studies.py.

echo "=== Figures ==="
python make_scatter_grid.py
python make_rollout_examples.py

echo "=== Done. Outputs in analysis/ ==="
