#!/usr/bin/env python3
"""SeeDance-2.0 at 720p across all 9 scenes (resolution ablation).

Reuses generate_rollouts.py wholesale: same conditioning frames, prompts, and 32 fixed
seeds; the ONLY change is resolution 480p -> 720p, written to a SEPARATE tree so the
committed 480p bins stay the comparison baseline. Requires REPLICATE_API_TOKEN.
"""
from pathlib import Path
import generate_rollouts as G

G.ROLLOUTS_BASE = Path(__file__).parent / "rollouts_seedance720"

# override only the resolution in seedance's input builder (keep duration=4, seed, prompt)
_orig = G.MODELS["seedance"]["input"]
G.MODELS["seedance"]["input"] = lambda url, seed, prompt: {**_orig(url, seed, prompt), "resolution": "720p"}

if __name__ == "__main__":
    for scene, cfg in G.SCENES.items():
        frame_url = G.upload_frame(G.find_conditioning_frame(cfg["frame"]))
        print(f"=== seedance@720p / {scene} (frame={cfg['frame']}) ===", flush=True)
        G.run_model(scene, "seedance", frame_url, cfg["prompt"], G.NUM_ROLLOUTS)
