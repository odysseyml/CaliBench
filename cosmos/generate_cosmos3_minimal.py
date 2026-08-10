#!/usr/bin/env python3
# Minimal single-GPU / stock-diffusers reference for generating the Cosmos3-Super
# videos. Config from cosmos_config.py, scenes from scenes.py.
import sys
import time
import argparse
from pathlib import Path

import torch
from diffusers import Cosmos3OmniPipeline, Cosmos3OmniTransformer
from diffusers.utils import export_to_video, load_image
from huggingface_hub import snapshot_download

# scenes.py (conditioning frame + prompt per scene) lives one level up, at the
# repo root; it is the same source generate_rollouts.py uses, so the frame/prompt
# for each scene cannot drift between generators.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scenes import SCENES                                   # noqa: E402
from cosmos_config import MODEL_REPO, COSMOS3_SUPER_ARGS, NEGATIVE_PROMPT  # noqa: E402

# Suffix appended to the dice prompt for the target-face ablation. Matches
# TARGET_TEMPLATE in dice_target_face_ablation.py (verified identical base prompt).
DICE_TARGET_TEMPLATE = "{base} The die comes to rest showing the {n}-pip face upward."


def find_frame(stem: str, frames_dir: Path) -> Path:
    for ext in (".png", ".jpg", ".jpeg", ".JPG", ".webp"):
        p = frames_dir / (stem + ext)
        if p.exists():
            return p
    raise FileNotFoundError(stem)


def load_pipeline(model_repo: str, device_map: str = "balanced"):
    """Stock Cosmos3-Super load. `device_map='balanced'` shards the transformer
    across visible GPUs via accelerate; `device_map='single'` loads on one GPU
    (only viable on a card large enough to hold the model)."""
    model_path = snapshot_download(repo_id=model_repo)
    shard = device_map not in (None, "single", "none")
    transformer = Cosmos3OmniTransformer.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        **({"device_map": device_map} if shard else {}),
    )
    pipe = Cosmos3OmniPipeline.from_pretrained(
        model_path,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        enable_safety_checker=False,
    )
    if shard:
        # accelerate has already placed the transformer; put the remaining
        # (small) components on the first GPU.
        for name, component in pipe.components.items():
            if name == "transformer" or not isinstance(component, torch.nn.Module):
                continue
            component.to("cuda:0")
    else:
        pipe.to("cuda")
    return pipe


def generate(pipe, prompt: str, image_path: Path, output_path: Path, seed: int,
             num_frames: int, height: int, width: int, steps: int,
             guidance_scale: float, fps: float):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    conditioning = load_image(str(image_path))
    print(f"[gen] {output_path.parent.name}/seed_{seed} (frame={image_path.name})", flush=True)
    t0 = time.time()
    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=conditioning,
        num_frames=num_frames,
        height=height,
        width=width,
        fps=fps,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        enable_sound=False,
        generator=generator,
    )
    export_to_video(result.video, str(output_path), fps=int(fps), macro_block_size=1)
    print(f"[gen] -> {output_path}  ({time.time() - t0:.0f}s)", flush=True)


def main():
    d = COSMOS3_SUPER_ARGS
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Minimal reference generator for the Cosmos3-Super videos.")
    parser.add_argument("--model_repo", default=MODEL_REPO)
    parser.add_argument("--conditioning_dir", default=str(repo_root / "conditioning_frames"))
    parser.add_argument("--output_base", default=str(repo_root / "rollouts_cosmos3_super"))
    parser.add_argument("--num_seeds", type=int, default=d["num_seeds"])
    parser.add_argument("--num_frames", type=int, default=d["num_frames"],
                        help="121 for main-table cells; 61 for the dice ablations.")
    parser.add_argument("--height", type=int, default=d["height"])
    parser.add_argument("--width", type=int, default=d["width"])
    parser.add_argument("--steps", type=int, default=d["steps"])
    parser.add_argument("--guidance_scale", type=float, default=d["guidance_scale"])
    parser.add_argument("--fps", type=float, default=d["fps"])
    parser.add_argument("--scenes", default=None, help="comma-separated subset (default: all nine)")
    parser.add_argument("--target_face", type=int, default=None, choices=range(1, 7),
                        help="dice target-face ablation: append the target-face suffix, dice only.")
    parser.add_argument("--label", default=None,
                        help="override the output sub-directory name (single-scene runs only).")
    parser.add_argument("--device_map", default="balanced",
                        help="'balanced' (multi-GPU via accelerate) or 'single'.")
    parser.add_argument("--attention_backend", default="native",
                        help="SDPA ('native') matches the committed run.")
    args = parser.parse_args()

    if args.target_face is not None:
        scene_subset = ["dice"]
    else:
        scene_subset = args.scenes.split(",") if args.scenes else list(SCENES)
    if args.label is not None and len(scene_subset) != 1:
        parser.error("--label only applies when a single scene is generated")

    pipe = load_pipeline(args.model_repo, args.device_map)
    if args.attention_backend:
        pipe.transformer.set_attention_backend(args.attention_backend)

    frames_dir = Path(args.conditioning_dir)
    out_base = Path(args.output_base)
    print(f"[gen] {len(scene_subset)} scene(s) x {args.num_seeds} seeds @ "
          f"{args.num_frames}f, CFG={args.guidance_scale}", flush=True)

    for scene_name in scene_subset:
        frame_stem, prompt = SCENES[scene_name]["frame"], SCENES[scene_name]["prompt"]
        if args.target_face is not None:
            prompt = DICE_TARGET_TEMPLATE.format(base=prompt, n=args.target_face)
            label = f"dice_face_{args.target_face}"
        else:
            label = args.label or scene_name
        frame_path = find_frame(frame_stem, frames_dir)
        out_dir = out_base / label
        out_dir.mkdir(parents=True, exist_ok=True)
        for seed in range(args.num_seeds):
            out_path = out_dir / f"seed_{seed}.mp4"
            if out_path.exists():
                print(f"[gen] skip (exists): {out_path}", flush=True)
                continue
            generate(pipe, prompt, frame_path, out_path, seed,
                     args.num_frames, args.height, args.width, args.steps,
                     args.guidance_scale, args.fps)


if __name__ == "__main__":
    main()
