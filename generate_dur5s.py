#!/usr/bin/env python3
"""Uniform 5-second duration ablation.

Re-generate at duration=5 the models that (a) were not already ~5 s and (b) can
accept 5 s: WAN (3->5), SeeDance (4->5), HappyHorse (3->5). Written to a SEPARATE
tree (rollouts_dur5s/) so the committed mixed-duration bins stay the baseline.
NOT re-run: Runway Gen-4.5 (already 5 s), Cosmos3-Super (~5.04 s / 121 f, cluster-only),
and Veo 3.1 -- Veo's API only accepts duration in {4, 6, 8}, so 5 s is impossible and
it stays at its committed 4 s. The recompute reuses those three models' committed bins.

Requires REPLICATE_API_TOKEN.

Usage:
  python generate_dur5s.py                                     # 4 models x 9 scenes x 32
  python generate_dur5s.py --models wan --scenes ball --count 1   # smoke test (1 video)
"""
import argparse
from pathlib import Path
import generate_rollouts as G

FIVE_S_MODELS = ["wan", "seedance", "happyhorse"]   # Veo excluded: only allows {4,6,8}s
G.ROLLOUTS_BASE = Path(__file__).parent / "rollouts_dur5s"

# Force duration=5 for each re-run model; keep every other input field (seed,
# prompt, resolution, ...) exactly as in the committed run.
for _m in FIVE_S_MODELS:
    _orig = G.MODELS[_m]["input"]
    G.MODELS[_m]["input"] = (
        lambda o: lambda url, seed, prompt: {**o(url, seed, prompt), "duration": 5}
    )(_orig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(FIVE_S_MODELS))
    ap.add_argument("--scenes", default="all")
    ap.add_argument("--count", type=int, default=G.NUM_ROLLOUTS)
    a = ap.parse_args()
    models = a.models.split(",")
    scenes = list(G.SCENES) if a.scenes == "all" else a.scenes.split(",")
    for scene in scenes:
        cfg = G.SCENES[scene]
        frame_url = G.upload_frame(G.find_conditioning_frame(cfg["frame"]))
        for model in models:
            print(f"=== {model}@5s / {scene} (frame={cfg['frame']}) ===", flush=True)
            G.run_model(scene, model, frame_url, cfg["prompt"], a.count)


if __name__ == "__main__":
    main()
