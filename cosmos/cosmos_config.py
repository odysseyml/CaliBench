#!/usr/bin/env python3
# Cosmos3-Super generation configuration: the arguments used for the paper.

MODEL_REPO = "nvidia/Cosmos3-Super"                          # image-to-video world model
DIFFUSERS_REV = "f34de4330ee0d7b931aaf77b4176886e74cf797f"  # Cosmos3-capable diffusers revision

# Image-to-video generation arguments (9 scenes x num_seeds seeds). Seeds are the
# integers 0..num_seeds-1, each passed to torch.Generator(device).manual_seed(seed).
# Each ablation (CFG-sweep value, target face) is generated as its own independent
# run; the canonical dice cell and the CFG=6.0 sweep point are therefore separate
# generation-and-extraction campaigns (independent draws at N=32), not the same
# videos re-scored.
COSMOS3_SUPER_ARGS = {
    "num_seeds": 32,
    "num_frames": 121,          # 121 @ 24 fps ~= 5.04 s
    "height": 704,
    "width": 1280,
    "steps": 35,                # sampling / inference steps
    "guidance_scale": 6.0,
    "fps": 24.0,
    "attention_backend": "native",   # SDPA
}

# Dice ablation studies on cosmos3_super (appendix figures). These reuse the args
# above EXCEPT num_frames: they were generated at the batch default of 61 frames
# (~2.5 s), i.e. a shorter horizon than the 121-frame main-table cells. Only the
# swept knob / prompt changes within each study, and each guidance value / face is
# generated as its own run and extracted by extract_dice_studies.py. The studies
# are therefore internally-consistent relative comparisons, not horizon-matched
# replicates of the main table.
#
#  - CFG sweep: the dice scene generated at each classifier-free guidance scale
#    below -> analysis/dice_cfg_<v>_cosmos3_super_bins.json.
CFG_SWEEP_GUIDANCE_SCALES = [1.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0]
#  - Target-face prompting: the dice prompt with
#    "The die comes to rest showing the {n}-pip face upward." appended, n in 1..6
#    (same template as dice_target_face_ablation.py)
#    -> analysis/dice_face_<n>_cosmos3_super_bins.json.

NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky "
    "movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, "
    "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
    "Overall, the video is of poor quality."
)
