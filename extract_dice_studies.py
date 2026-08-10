#!/usr/bin/env python3
"""
VLM extraction for the dice CFG sweep (cosmos3 only) and the explicit
target-face ablation (all 6 models). Reuses extract_bin.SCENES['dice']
(same VLM prompt + 6-bin space) and just swaps ROLLOUTS_BASE per scene.

Outputs:
  analysis/dice_cfg_<val>_cosmos3_super_bins.json   (CFG sweep)
  analysis/dice_face_<n>_<model>_bins.json          (target-face study)

Usage:
  export GEMINI_API_KEY=...
  python extract_dice_studies.py
  python extract_dice_studies.py --study cfg     # CFG only
  python extract_dice_studies.py --study target  # target-face only
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import extract_bin as E

REPLICATE_MODELS = ["wan", "seedance", "happyhorse", "veo31", "runway45"]
ALL_MODELS = REPLICATE_MODELS + ["cosmos3_super"]
CFG_VALUES = ["1_0", "1_5", "3_0", "4_5", "6_0", "7_5", "9_0"]
TARGET_FACES = [1, 2, 3, 4, 5, 6]


async def extract_one(scene_name, model_name, base_dir, n_votes=3):
    cfg = E.SCENES["dice"]
    E.ROLLOUTS_BASE = base_dir
    await E.process_model_async(
        scene_name=scene_name,
        model_name=model_name,
        vlm_prompt=cfg["vlm_prompt"],
        num_outcomes=cfg["num_outcomes"],
        outcome_map=cfg["outcome_map"],
        frame_pcts=cfg.get("frame_pcts"),
        use_video=cfg.get("use_video", False),
        n_votes=n_votes,
    )


async def run_cfg_sweep():
    base = Path(__file__).parent / "rollouts_cfg_sweep"
    print("\n=== CFG sweep ===")
    for v in CFG_VALUES:
        await extract_one(f"dice_cfg_{v}", "cosmos3_super", base)


async def run_target_face():
    base = Path(__file__).parent / "rollouts_target_face"
    print("\n=== Target-face ablation ===")
    for n in TARGET_FACES:
        for m in ALL_MODELS:
            # Skip if videos absent (still being generated)
            d = base / f"dice_face_{n}" / m
            if not d.exists() or not any(d.glob("seed_*.mp4")):
                print(f"  skip dice_face_{n}/{m}: no videos yet")
                continue
            await extract_one(f"dice_face_{n}", m, base)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="all",
                        choices=["all", "cfg", "target"])
    args = parser.parse_args()
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY not set")
    E.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    if args.study in ("all", "cfg"):
        await run_cfg_sweep()
    if args.study in ("all", "target"):
        await run_target_face()


if __name__ == "__main__":
    asyncio.run(main())
