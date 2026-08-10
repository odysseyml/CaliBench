#!/usr/bin/env python3
"""
Generate 32 I2V rollouts for a given scene and model using Replicate.
A scene = (conditioning frame, prompt) pair.
Saves to rollouts/{scene}/{model}/seed_{i}.mp4

Prediction IDs are written to rollouts/{scene}/{model}/.pending.json on
submission and removed on successful download, so a restart resumes
polling already-submitted predictions instead of resubmitting them.

Requires: REPLICATE_API_TOKEN environment variable
Run with:
  python generate_rollouts.py --scene galton_board --model wan
  python generate_rollouts.py --scene galton_board --model all
  python generate_rollouts.py --scene dice --model all --count 32
"""
import os
import json
import time
import argparse
import requests
import replicate
from pathlib import Path

NUM_ROLLOUTS = 32
POLL_INTERVAL = 5
MAX_WAIT = 1200
MAX_RETRIES = 3

FRAMES_DIR = Path(__file__).parent / "conditioning_frames"
ROLLOUTS_BASE = Path(__file__).parent / "rollouts"

from scenes import SCENES  # canonical scene defs (frame + prompt), shared with cosmos3_super_batch

MODELS = {
    "wan": {
        "id": "wan-video/wan-2.7-i2v",
        "input": lambda url, seed, prompt: {
            "first_frame": url,
            "prompt": prompt,
            "seed": seed,
            "duration": 3,
            "resolution": "720p",
            "negative_prompt": "",
            "enable_prompt_expansion": True,
        },
    },
    "seedance": {
        "id": "bytedance/seedance-2.0",
        "input": lambda url, seed, prompt: {
            "image": url,
            "prompt": prompt,
            "seed": seed,
            "duration": 4,
            "resolution": "480p",
            "generate_audio": False,
        },
    },
    "happyhorse": {
        "id": "alibaba/happyhorse-1.0",
        "input": lambda url, seed, prompt: {
            "image": url,
            "prompt": prompt,
            "seed": seed,
            "duration": 3,
            "resolution": "720p",
        },
    },
    "veo31": {
        "id": "google/veo-3.1",
        "input": lambda url, seed, prompt: {
            "image": url,
            "prompt": prompt,
            "seed": seed,
            "duration": 4,
            "resolution": "720p",
        },
    },
    "runway45": {
        "id": "runwayml/gen-4.5",
        "input": lambda url, seed, prompt: {
            "image": url,
            "prompt": prompt,
            "seed": seed,
            "duration": 5,
        },
    },
}


def find_conditioning_frame(frame_stem):
    for ext in (".png", ".jpg", ".jpeg", ".JPG"):
        p = FRAMES_DIR / (frame_stem + ext)
        if p.exists():
            return p
    raise FileNotFoundError(f"No conditioning frame '{frame_stem}' in {FRAMES_DIR}")


def upload_frame(frame_path):
    print(f"Uploading {frame_path.name}...", end=" ", flush=True)
    with open(frame_path, "rb") as f:
        frame_file = replicate.files.create(f, filename=frame_path.name)
    url = frame_file.urls["get"]
    print("ok")
    return url


def load_pending(pending_path):
    """Load {seed_index: prediction_id} from disk."""
    if pending_path.exists():
        with open(pending_path) as f:
            return {int(k): v for k, v in json.load(f).items()}
    return {}


def save_pending(pending_path, pending_ids):
    """Persist {seed_index: prediction_id} to disk."""
    with open(pending_path, "w") as f:
        json.dump({str(k): v for k, v in pending_ids.items()}, f)


def run_model(scene_name, model_name, frame_url, prompt, count):
    cfg = MODELS[model_name]
    out_dir = ROLLOUTS_BASE / scene_name / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pending_path = out_dir / ".pending.json"

    print(f"\n=== {scene_name} / {model_name} ({cfg['id']}) ===")

    # Recover any predictions submitted in a previous run that weren't downloaded
    saved_ids = load_pending(pending_path)
    pending = {}
    for i, pred_id in saved_ids.items():
        out_path = out_dir / f"seed_{i}.mp4"
        if out_path.exists():
            continue  # already downloaded by some other means
        try:
            pending[i] = replicate.predictions.get(pred_id)
            print(f"  rollout {i:02d}: resuming (id={pred_id})")
        except Exception:
            pass  # prediction expired or invalid; will resubmit below

    # Submit new predictions for seeds not yet on disk and not already pending
    for i in range(count):
        out_path = out_dir / f"seed_{i}.mp4"
        if out_path.exists():
            print(f"  rollout {i:02d}: skip (exists)")
            continue
        if i in pending:
            continue
        for attempt in range(10):
            try:
                pred = replicate.predictions.create(
                    model=cfg["id"],
                    input=cfg["input"](frame_url, i, prompt),
                )
                break
            except replicate.exceptions.ReplicateError as e:
                if e.status == 429:
                    wait = 2 ** attempt
                    print(f"  rollout {i:02d}: rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        else:
            print(f"  rollout {i:02d}: could not submit after retries, skipping")
            continue
        pending[i] = pred
        print(f"  rollout {i:02d}: submitted (id={pred.id})")
        save_pending(pending_path, {k: p.id for k, p in pending.items()})

    if not pending:
        print("  All rollouts already exist.")
        return

    retries = {i: 0 for i in pending}
    print(f"  Polling {len(pending)} predictions...")
    deadline = time.time() + MAX_WAIT

    while pending and time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        done = []
        for i, pred in list(pending.items()):
            pred.reload()
            if pred.status not in ("succeeded", "failed", "canceled"):
                continue  # still running
            saved = False
            if pred.status == "succeeded":
                output = pred.output
                video_url = output[0] if isinstance(output, list) else output
                if video_url:
                    try:
                        data = requests.get(video_url, timeout=120).content
                        with open(out_dir / f"seed_{i}.mp4", "wb") as f:
                            f.write(data)
                        print(f"  rollout {i:02d}: saved → {out_dir / f'seed_{i}.mp4'}")
                        saved = True
                    except Exception as e:
                        print(f"  rollout {i:02d}: download failed — {e}")
                else:
                    print(f"  rollout {i:02d}: succeeded but empty output")
            if saved:
                done.append(i)
                continue
            # failed / canceled / empty-output / download-error -> retry or give up
            if retries[i] < MAX_RETRIES:
                retries[i] += 1
                print(f"  rollout {i:02d}: {pred.status} — retrying ({retries[i]}/{MAX_RETRIES})")
                new_pred = replicate.predictions.create(
                    model=cfg["id"],
                    input=cfg["input"](frame_url, i, prompt),
                )
                pending[i] = new_pred
                save_pending(pending_path, {k: p.id for k, p in pending.items()})
                deadline = time.time() + MAX_WAIT
            else:
                print(f"  rollout {i:02d}: gave up after {MAX_RETRIES} retries — {pred.error}")
                done.append(i)
        for i in done:
            del pending[i]
        if done:
            save_pending(pending_path, {k: p.id for k, p in pending.items()})

    if pending:
        print(f"  WARNING: {len(pending)} predictions did not finish within {MAX_WAIT}s — "
              f"IDs saved to {pending_path}, re-run to resume.")
    else:
        pending_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="galton_board",
                        help=f"Scene to run: {', '.join(SCENES)}")
    parser.add_argument("--model", default="all",
                        help=f"Model to run: {', '.join(MODELS)} or 'all'")
    parser.add_argument("--count", type=int, default=NUM_ROLLOUTS,
                        help="Number of rollouts to generate")
    args = parser.parse_args()

    if not os.environ.get("REPLICATE_API_TOKEN"):
        raise RuntimeError("REPLICATE_API_TOKEN environment variable not set")
    if args.scene not in SCENES:
        raise ValueError(f"Unknown scene '{args.scene}'. Choose from: {', '.join(SCENES)}")

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]
    for m in models_to_run:
        if m not in MODELS:
            raise ValueError(f"Unknown model '{m}'. Choose from: {', '.join(MODELS)}")

    scene_cfg = SCENES[args.scene]
    frame_path = find_conditioning_frame(scene_cfg["frame"])
    frame_url = upload_frame(frame_path)
    prompt = scene_cfg["prompt"]

    for model_name in models_to_run:
        run_model(args.scene, model_name, frame_url, prompt, args.count)

    print("\nAll done.")


if __name__ == "__main__":
    main()
