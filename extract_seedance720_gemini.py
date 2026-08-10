#!/usr/bin/env python3
"""Gemini extraction of the SeeDance-2.0 @720p ablation.

Reads rollouts_seedance720/<scene>/seedance/seed_*.mp4, writes to analysis/seedance720/
(kept separate from the committed 480p bins). All 9 scenes, seedance only, votes=3.
Requires GEMINI_API_KEY.
"""
import asyncio
from pathlib import Path
import extract_bin as G

G.ROLLOUTS_BASE = Path(__file__).parent / "rollouts_seedance720"
G.ANALYSIS_DIR = Path(__file__).parent / "analysis" / "seedance720"
G.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

SCENES = ["galton_board", "galton_board_real", "ball", "walking", "pendulum",
          "dice", "cards", "lottery", "roulette"]

asyncio.run(G.main_async(SCENES, ["seedance"], n_votes=3))
