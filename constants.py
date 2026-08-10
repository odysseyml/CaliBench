#!/usr/bin/env python3
"""Canonical CaliBench constants shared across the analysis pipeline.

Single source of truth for the scene set, the reference outcome distributions,
and the model set, so the compute_* scripts cannot drift out of sync. Import
from here rather than re-declaring these in each script.
"""
import numpy as np
from scipy.stats import binom

# The nine scenes, in the paper's display order.
SCENES = ["galton_board_real", "galton_board", "ball", "walking", "pendulum",
          "dice", "cards", "lottery", "roulette"]

# Number of valid outcome bins (k) per scene.
K_BY_SCENE = {
    "galton_board": 11, "galton_board_real": 11,
    "ball": 2, "walking": 2, "pendulum": 2,
    "dice": 6, "cards": 4, "lottery": 20, "roulette": 3,
}

# Reference outcome distribution p_0 per scene (length K_BY_SCENE[scene], sums to 1).
SCENE_PMF = {
    "galton_board":      np.array([binom.pmf(k, 10, 0.5) for k in range(11)]),
    "galton_board_real": np.array([binom.pmf(k, 10, 0.5) for k in range(11)]),
    "ball":     np.full(2, 0.5),
    "walking":  np.full(2, 0.5),
    "pendulum": np.full(2, 0.5),
    "dice":     np.full(6, 1 / 6),
    "cards":    np.full(4, 0.25),
    "lottery":  np.full(20, 1 / 20),
    "roulette": np.array([18 / 37, 18 / 37, 1 / 37]),
}

# The paper's six benchmarked models, in display order. Only these six enter the
# FDR family; any additional system would change the denominator and shift every
# q-value, so this list must stay exactly the reported set.
MODELS = ["wan", "seedance", "happyhorse", "veo31", "runway45", "cosmos3_super"]

# Canonical display names for the six models (paper spelling).
PRETTY = {
    "wan": "WAN-2.7",
    "seedance": "SeeDance-2.0",
    "happyhorse": "HappyHorse-1.0",
    "veo31": "Veo 3.1",
    "runway45": "Runway Gen-4.5",
    "cosmos3_super": "Cosmos3-Super",
}


if __name__ == "__main__":
    # Invariants: every scene has a matching k and a valid reference pmf, and the
    # model set is exactly the paper's six reported systems.
    assert set(SCENES) == set(K_BY_SCENE) == set(SCENE_PMF), "scene sets disagree"
    for s in SCENES:
        assert len(SCENE_PMF[s]) == K_BY_SCENE[s], f"{s}: pmf length != k"
        assert abs(float(SCENE_PMF[s].sum()) - 1.0) < 1e-12, f"{s}: pmf does not sum to 1"
    assert MODELS == ["wan", "seedance", "happyhorse", "veo31", "runway45", "cosmos3_super"]
    assert set(PRETTY) == set(MODELS), "PRETTY must cover exactly the model set"
    print(f"constants OK: {len(SCENES)} scenes, {len(MODELS)} models, all pmfs valid")
