#!/usr/bin/env python3
"""
Bootstrap CI on mnTV.

Scheme:
- Hold the nine scenes FIXED.
- Within each cell, resample all N rollouts per cell (null category +
  valid outcomes) from the empirical multinomial; ρ and n_valid VARY
  across bootstrap iterations.
- Recompute the floor E_0[TV] at each resampled n_valid (function only of
  scene reference and n_valid; recomputed per draw, not frozen).
- Recompute s_cell = (1-ρ) + ρ * (TV - E_0)/(1 - E_0) per cell (signed; no
  max(0,·)).
- mnTV = mean over the nine scenes (pendulum included via its
  (1-ρ) term, which is fixed).

For each bootstrap iteration we get one mnTV per model. Marginal
central-95% intervals are reported for display; paired difference
intervals are also computed because rank claims must be judged on those.
"""
import json
import numpy as np
from pathlib import Path

from constants import K_BY_SCENE, SCENE_PMF, SCENES as SCENE_ORDER, MODELS as MODEL_ORDER, PRETTY

ROOT = Path(__file__).parent
ANALYSIS = ROOT / "analysis"

N_BOOT = 50_000
SEED = 0


def load_outcomes(scene, model):
    """Return list of valid 1-indexed outcomes for this cell, plus n_total."""
    path = ANALYSIS / f"{scene}_{model}_bins.json"
    if not path.exists():
        return None, None
    d = json.load(open(path))
    rollouts = d["rollouts"]
    valid = [r["bin"] for r in rollouts if r["bin"] is not None]
    return valid, len(rollouts)


def _binomial_mad(n, p):
    """E[|X - np|] for X ~ Binomial(n, p), exact via direct summation."""
    if n == 0:
        return 0.0
    from scipy.stats import binom as _b
    xs = np.arange(n + 1)
    return float(np.sum(np.abs(xs - n * p) * _b.pmf(xs, n, p)))


def null_floor(p_0, n):
    """Analytic E_H0[TV] via binomial MAD per bin."""
    if n < 1:
        return 0.0
    return sum(_binomial_mad(n, pi) for pi in p_0) / (2 * n)


# Precompute the null floor E_0[TV] for every (scene, n_valid) once, so it is
# not recomputed inside the 50k × 6 × 9 bootstrap loop.
E0_TABLE = {s: {n: null_floor(SCENE_PMF[s], n) for n in range(0, 33)}
            for s in SCENE_ORDER}


def s_cell(rho, valid_counts, p_0, e0):
    """Given counts vector among valid, compute s_cell."""
    n_valid = valid_counts.sum()
    if n_valid < 1:
        return 1.0 - rho  # equals 1 when rho=0
    p_hat = valid_counts / n_valid
    tv = 0.5 * np.abs(p_hat - p_0).sum()
    excess = (tv - e0) / (1.0 - e0) if e0 < 1.0 else 0.0              # signed; no max(0,·)
    return (1.0 - rho) + rho * excess


def main():
    rng = np.random.default_rng(SEED)

    # Pre-load: for each cell, observed valid outcomes (1-indexed),
    # n_total, rho, and the frozen null floor E_0 at this n_valid.
    cells = {}
    for s in SCENE_ORDER:
        p_0 = SCENE_PMF[s]
        for m in MODEL_ORDER:
            valid, n_total = load_outcomes(s, m)
            if valid is None:
                continue
            n_valid = len(valid)
            rho = n_valid / n_total if n_total > 0 else 0.0
            e0 = null_floor(p_0, n_valid) if n_valid >= 1 else 0.0
            cells[(s, m)] = {
                "rho": rho, "n_valid": n_valid, "valid": np.array(valid, dtype=int),
                "p_0": p_0, "e0": e0, "k": K_BY_SCENE[s], "n_total": n_total,
            }

    # Bootstrap loop. For each iteration, resample all N video generations per
    # cell over the k valid bins + null category (so ρ and n_valid vary),
    # recompute s_cell at the resampled n_valid, aggregate to mnTV.
    boot_mntv = {m: np.zeros(N_BOOT) for m in MODEL_ORDER}
    for b in range(N_BOOT):
        if (b + 1) % 1000 == 0:
            print(f"  bootstrap {b+1}/{N_BOOT}")
        for m in MODEL_ORDER:
            scene_scores = []
            for s in SCENE_ORDER:
                c = cells.get((s, m))
                if c is None:
                    continue
                k, N, n_valid = c["k"], c["n_total"], c["n_valid"]
                counts_obs = np.zeros(k, dtype=int)
                for outc in c["valid"]:
                    counts_obs[outc - 1] += 1
                # empirical pmf over k valid bins + 1 null category
                probs = np.append(counts_obs / N, (N - n_valid) / N)
                draw  = rng.multinomial(N, probs)
                nvb   = int(draw[:k].sum())
                rhob  = nvb / N
                if nvb < 1:
                    scene_scores.append(1.0 - rhob)          # == 1.0
                else:
                    e0b = E0_TABLE[s][nvb]                    # floor at the RESAMPLED n_valid
                    scene_scores.append(s_cell(rhob, draw[:k], c["p_0"], e0b))
            boot_mntv[m][b] = np.mean(scene_scores)

    # Summarize: point estimate (from observed data, not bootstrap mean) and
    # central-95% CI from bootstrap.
    print()
    print(f"{'Model':<16} {'mnTV':>9} {'CI low':>7} {'CI high':>8}")
    out = {"mntv": {}, "ci": {}, "diffs": {}}
    # Recompute point estimate same way
    point = {}
    for m in MODEL_ORDER:
        scene_scores = []
        for s in SCENE_ORDER:
            c = cells.get((s, m))
            if c is None: continue
            n_valid = c["n_valid"]
            if n_valid < 1:
                scene_scores.append(1.0 - c["rho"])
                continue
            k = c["k"]
            counts_obs = np.zeros(k, dtype=int)
            for outc in c["valid"]:
                counts_obs[outc - 1] += 1
            scene_scores.append(s_cell(c["rho"], counts_obs, c["p_0"], c["e0"]))
        point[m] = float(np.mean(scene_scores))
        lo = float(np.quantile(boot_mntv[m], 0.025))
        hi = float(np.quantile(boot_mntv[m], 0.975))
        out["mntv"][m] = point[m]
        out["ci"][m] = [lo, hi]
        print(f"{PRETTY[m]:<16} {point[m]:>9.3f} {lo:>7.3f} {hi:>8.3f}")

    # Paired difference intervals: for each (A, B) pair, CI on A-B
    print("\nPaired difference intervals (rows minus cols, 95% CI):")
    print(f"{'':<14}" + "".join(f"{PRETTY[m]:>16}" for m in MODEL_ORDER))
    for ma in MODEL_ORDER:
        row = [f"{PRETTY[ma]:<14}"]
        for mb in MODEL_ORDER:
            diff = boot_mntv[ma] - boot_mntv[mb]
            mean = diff.mean()
            lo = np.quantile(diff, 0.025)
            hi = np.quantile(diff, 0.975)
            distinguishable = "*" if (lo > 0 or hi < 0) else " "
            row.append(f"{mean:+.02f}{distinguishable}[{lo:+.02f},{hi:+.02f}]")
            out["diffs"][f"{ma}-{mb}"] = {"mean": float(mean), "lo": float(lo), "hi": float(hi)}
        print("".join(f"{c:>16}" for c in row))

    json.dump(out, open(ANALYSIS / "mntv_ci.json", "w"), indent=2)
    print(f"\nSaved -> {ANALYSIS / 'mntv_ci.json'}")


if __name__ == "__main__":
    main()
