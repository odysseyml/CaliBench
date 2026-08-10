#!/usr/bin/env python3
"""
Explicit-target-face ablation on the dice scene.

For each target face N in {1..6}, append " The die comes to rest showing the
N-pip face upward." to the baseline dice prompt (see TARGET_TEMPLATE), generate
5 seeds per (model, N),
and save under rollouts_target_face/dice_face_{N}/<model>/seed_*.mp4.

Covers all 5 Replicate models. Cosmos3-Super is generated separately on EKS.

Usage:
  export REPLICATE_API_TOKEN=...
  python dice_target_face_ablation.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import generate_rollouts as G

BASE_DICE_PROMPT = (
    "A clean 2D physics animation of a single six-sided die being rolled on a flat "
    "surface inside a wooden box with low walls. The die starts tumbling from the "
    "left side and comes to rest showing a single upward face clearly. The animation "
    "is slow, smooth, and physically accurate. The die is white with large black "
    "circular pips, viewed from directly above."
)
TARGET_TEMPLATE = "{base} The die comes to rest showing the {n}-pip face upward."

OUT_BASE = Path(__file__).parent / "rollouts_target_face"


def prompt_for(n: int) -> str:
    return TARGET_TEMPLATE.format(base=BASE_DICE_PROMPT, n=n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        help=f"Model: {', '.join(G.MODELS)} or 'all'")
    parser.add_argument("--count", type=int, default=5,
                        help="Seeds per (model, target face) cell")
    parser.add_argument("--faces", default="1,2,3,4,5,6",
                        help="Comma-separated target faces")
    args = parser.parse_args()

    import os
    if not os.environ.get("REPLICATE_API_TOKEN"):
        raise RuntimeError("REPLICATE_API_TOKEN not set")

    G.ROLLOUTS_BASE = OUT_BASE
    models = list(G.MODELS) if args.model == "all" else [args.model]
    faces = [int(x) for x in args.faces.split(",")]

    frame_path = G.find_conditioning_frame("dice")
    frame_url = G.upload_frame(frame_path)

    for n in faces:
        scene_name = f"dice_face_{n}"
        prompt = prompt_for(n)
        print(f"\n### TARGET FACE {n} — prompt:\n{prompt}\n")
        for m in models:
            G.run_model(scene_name, m, frame_url, prompt, args.count)

    print(f"\nReplicate side complete. Outputs at {OUT_BASE}/dice_face_<N>/<model>/seed_*.mp4")


if __name__ == "__main__":
    main()
