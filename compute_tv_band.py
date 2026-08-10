#!/usr/bin/env python3
"""
Per-cell TV + null band + mnTV (the headline calibration benchmark).

For each (scene, model) cell:
  TV(p̂_valid, p₀) = ½ Σ |p̂_valid,i − p₀,i|
  band: central-95% range of TV under H₀ from 50,000 multinomial(n_valid, p₀) draws
  E₀[TV]: mean of those draws
  excess = (TV − E₀[TV]) / (1 − E₀[TV])          (signed; no max(0,·))
  score_cell = (1 − ρ) + ρ · excess
  mnTV(model) = mean over scenes of score_cell

The χ² significance verdicts are computed separately in compute_chi2.py; this
script only computes the TV-based magnitude metric.

Writes analysis/tv_band_results.json (mirrors chi2_results.json layout).

Synthetic sanity: because (TV − E₀) is exactly mean-zero under perfect
calibration (E₀ is computed exactly via De Moivre), a perfectly calibrated
model (samples iid from p₀ at ρ=1) yields mnTV → 0 in expectation exactly.
"""
import json
import numpy as np
from pathlib import Path
from scipy.stats import binom as _binom_dist

from constants import K_BY_SCENE, SCENE_PMF, SCENES as SCENE_ORDER, MODELS as MODEL_ORDER

ROOT = Path(__file__).parent
ANALYSIS = ROOT / "analysis"
N_MC = 50_000
SEED = 0


def tv(p_hat, p_0):
    return 0.5 * np.abs(p_hat - p_0).sum()


def _binomial_mad(n, p):
    """E[|X - np|] for X ~ Binomial(n, p), exact via direct summation."""
    if n == 0:
        return 0.0
    xs = np.arange(n + 1)
    return float(np.sum(np.abs(xs - n * p) * _binom_dist.pmf(xs, n, p)))


def analytic_e_tv(n_valid, p_0):
    """E_{H_0}[TV] = (1 / (2 n)) sum_i E[|X_i - n p_i|], via binomial MAD per bin."""
    if n_valid < 1:
        return 0.0
    return sum(_binomial_mad(n_valid, pi) for pi in p_0) / (2 * n_valid)


def null_band(n_valid, p_0, n_mc=N_MC, seed=SEED):
    """Analytic E_H0[TV] + Monte Carlo central-95% range (the band itself is
    not a single expectation, so we still simulate for the quantiles)."""
    if n_valid < 1:
        return None, None, None
    e0 = analytic_e_tv(n_valid, p_0)
    rng = np.random.default_rng(seed)
    sims = rng.multinomial(n_valid, p_0, size=n_mc)
    sim_p = sims / n_valid
    tvs = 0.5 * np.abs(sim_p - p_0[None, :]).sum(axis=1)
    return e0, float(np.quantile(tvs, 0.025)), float(np.quantile(tvs, 0.975))


def per_cell_tv(scene, model, p_0):
    """Load bins JSON, compute (rho, n_valid, tv_valid, e0, band_lo, band_hi)."""
    path = ANALYSIS / f"{scene}_{model}_bins.json"
    if not path.exists():
        return None
    d = json.load(open(path))
    rollouts = d["rollouts"]
    n_total = len(rollouts)
    valid = [r for r in rollouts if r["bin"] is not None]
    n_valid = len(valid)
    rho = n_valid / n_total if n_total else 0.0
    if n_valid < 1:
        return {
            "rho": rho, "n_valid": n_valid,
            "tv_valid": None, "e0": None, "band_lo": None, "band_hi": None,
            "excess": None, "score_cell": 1.0,  # no valid → all mass missing
        }
    counts = np.zeros(len(p_0), dtype=int)
    for r in valid:
        counts[r["bin"] - 1] += 1
    p_hat = counts / n_valid
    tv_obs = tv(p_hat, p_0)
    e0, band_lo, band_hi = null_band(n_valid, p_0)
    excess = (tv_obs - e0) / (1.0 - e0) if e0 is not None else None   # signed; no max(0,·)
    score_cell = (1 - rho) + rho * excess if excess is not None else (1 - rho)
    return {
        "rho": rho, "n_valid": n_valid,
        "tv_valid": float(tv_obs), "e0": e0,
        "band_lo": band_lo, "band_hi": band_hi,
        "excess": excess, "score_cell": float(score_cell),
    }


def main():
    out = {}
    for scene in SCENE_ORDER:
        p_0 = SCENE_PMF[scene]
        out[scene] = {}
        for model in MODEL_ORDER:
            cell = per_cell_tv(scene, model, p_0)
            if cell is not None:
                out[scene][model] = cell

    # Per-model mnTV = mean of score_cell over scenes
    mntv = {}
    for model in MODEL_ORDER:
        scores = []
        for scene in SCENE_ORDER:
            cell = out.get(scene, {}).get(model)
            if cell is not None and cell["score_cell"] is not None:
                scores.append(cell["score_cell"])
        mntv[model] = float(np.mean(scores)) if scores else None
    out["_mntv"] = mntv

    out_path = ANALYSIS / "tv_band_results.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"Saved -> {out_path}\n")

    # Print per-cell summary table
    print(f"{'scene':<22} {'model':<14} {'rho':>5} {'n_v':>3} {'TV':>5} {'E_0':>5} {'band':>13} {'excess':>6} {'score':>5}")
    for scene in SCENE_ORDER:
        for model in MODEL_ORDER:
            v = out.get(scene, {}).get(model)
            if v is None:
                continue
            tv_s = f"{v['tv_valid']:.2f}" if v['tv_valid'] is not None else "—"
            e0_s = f"{v['e0']:.2f}" if v['e0'] is not None else "—"
            band_s = f"[{v['band_lo']:.2f},{v['band_hi']:.2f}]" if v['band_lo'] is not None else "—"
            exc_s = f"{v['excess']:.2f}" if v['excess'] is not None else "—"
            sc_s = f"{v['score_cell']:.2f}"
            print(f"{scene:<22} {model:<14} {v['rho']:>5.2f} {v['n_valid']:>3d} {tv_s:>5} {e0_s:>5} {band_s:>13} {exc_s:>6} {sc_s:>5}")

    print("\n=== mnTV per model (lower = better, 0 = calibrated) ===")
    for m, s in mntv.items():
        print(f"  {m:<16} {s:.3f}")


if __name__ == "__main__":
    main()
