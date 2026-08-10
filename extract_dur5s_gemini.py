#!/usr/bin/env python3
"""Gemini extraction of the uniform-5s duration ablation.

Reads rollouts_dur5s/<scene>/<model>/seed_*.mp4 for the four re-run models and
writes bins to analysis/dur5s/ (kept separate from the committed bins). All nine
scenes, votes=3. Requires GEMINI_API_KEY.
"""
import asyncio
from pathlib import Path
import extract_bin as G

G.ROLLOUTS_BASE = Path(__file__).parent / "rollouts_dur5s"
G.ANALYSIS_DIR = Path(__file__).parent / "analysis" / "dur5s"
G.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["wan", "seedance", "happyhorse"]   # Veo excluded (can't run at 5 s)

if __name__ == "__main__":
    asyncio.run(G.main_async(list(G.SCENES.keys()), MODELS, n_votes=3))
