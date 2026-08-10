#!/usr/bin/env python3
"""
Monte Carlo chi-squared goodness-of-fit utilities.

Why MC: Cochran's expected-count rule fails for several scenes in this paper
at N=32 (Galton tails, lottery, roulette green), so the asymptotic chi^2
reference distribution is unreliable. We compute critical values and
p-values directly from the null sampling distribution by Monte Carlo.

All routines accept reproducible seeds via numpy.random.default_rng.
"""
from __future__ import annotations

import numpy as np


def chi2_stat(observed: np.ndarray, ref_probs: np.ndarray) -> float:
    """Pearson chi-squared statistic for counts observed under reference probs."""
    observed = np.asarray(observed, dtype=float)
    ref_probs = np.asarray(ref_probs, dtype=float)
    N = observed.sum()
    expected = N * ref_probs
    return float(((observed - expected) ** 2 / expected).sum())


def chi2_mc(observed, ref_probs, n_replicates: int = 50_000, seed: int = 0,
            tol: float = 1e-9):
    """MC chi^2 goodness-of-fit.

    Returns (statistic, p_value, critical_value_05).

    The p-value uses the Hope (1968) add-one estimator  p = (b + 1) / (m + 1),  where
    b is the number of null replicates whose statistic is >= the observed statistic
    (counted within an absolute tolerance `tol` so genuine ties are included robustly
    against floating-point noise; the chi^2 statistic lives on a discrete lattice at
    fixed N, so tol=1e-9 never bridges distinct values) and m = n_replicates. The +1 in
    numerator and denominator gives a strictly positive, (conservatively) unbiased MC
    p-value that can never be exactly 0.
    Ref: Hope, "A simplified Monte Carlo significance test procedure", JRSS-B 30, 582 (1968).
    """
    observed = np.asarray(observed, dtype=float)
    ref_probs = np.asarray(ref_probs, dtype=float)
    N = int(round(observed.sum()))
    expected = N * ref_probs
    obs_stat = float(((observed - expected) ** 2 / expected).sum())
    rng = np.random.default_rng(seed)
    null_counts = rng.multinomial(N, ref_probs, size=n_replicates).astype(float)
    null_stats = ((null_counts - expected) ** 2 / expected).sum(axis=1)
    # --- Tie handling --------------------------------------------------------
    # The chi^2 statistic is DISCRETE: counts are integers, so the statistic lives
    # on a lattice and the null distribution has probability atoms. The observed
    # statistic routinely lands EXACTLY on a populated atom -- e.g. SeeDance/lottery
    # sits at chi^2 = 31.0, an atom carrying ~1.7% of the null mass. Whether those
    # ties are counted swings the p-value materially there: >= (include ties) gives
    # 0.049, > (exclude ties) gives 0.032. We INCLUDE ties, which is the standard,
    # conservative Monte-Carlo p-value (Davison & Hinkley 1997 sec. 4.2; North,
    # Curtis & Sham, Am. J. Hum. Genet. 71:439, 2002).
    #
    # `tol` makes the tie count REPRODUCIBLE ACROSS SYSTEMS: floating-point
    # evaluation of null_stats can differ by ~1e-13 between numpy/BLAS builds, but
    # the lattice spacing (1/min(expected)) is many orders of magnitude larger than
    # tol = 1e-9, so `>= obs_stat - tol` captures every genuine tie without ever
    # merging two distinct lattice values -> identical integer count `b` on any
    # platform. np.random.default_rng (PCG64) is itself a version- and platform-
    # stable stream per numpy's stream-compatibility policy, so the draws match too.
    b = int((null_stats >= obs_stat - tol).sum())
    p_value = (b + 1) / (n_replicates + 1)   # Hope (1968) add-one (see docstring)
    crit_05 = float(np.quantile(null_stats, 0.95))
    return obs_stat, p_value, crit_05


def benjamini_hochberg(pvals) -> np.ndarray:
    """Benjamini-Hochberg step-up adjusted p-values (q-values).

    Controls the false discovery rate. Returns q-values in the SAME order as the
    input, each clipped to [0, 1] and enforced monotone in the p-value ranking
    (so a rejection at rank i never implies non-rejection at a smaller rank).

    Matches scipy.stats.false_discovery_control(pvals, method='bh').
    """
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    # raw BH factor p_(i) * n / i, then enforce monotonicity from the top down
    q_sorted = ranked * n / np.arange(1, n + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    return q


def bonferroni(pvals) -> np.ndarray:
    """Bonferroni-adjusted p-values (controls FWER): min(p * n, 1)."""
    pvals = np.asarray(pvals, dtype=float)
    return np.clip(pvals * pvals.size, 0.0, 1.0)


def chi2_power_mc(ref_probs, alt_probs, N: int, crit_value: float,
                  n_replicates: int = 5000, seed: int = 1) -> float:
    """Power against a specified alternative, using a precomputed MC crit value."""
    ref_probs = np.asarray(ref_probs, dtype=float)
    alt_probs = np.asarray(alt_probs, dtype=float)
    expected = N * ref_probs
    rng = np.random.default_rng(seed)
    alt_counts = rng.multinomial(N, alt_probs, size=n_replicates).astype(float)
    alt_stats = ((alt_counts - expected) ** 2 / expected).sum(axis=1)
    return float((alt_stats >= crit_value).mean())


def min_detectable_epsilon(ref_probs, spike_index: int, N: int,
                           crit_value: float, target_power: float = 0.8,
                           tol: float = 1e-3, n_replicates: int = 5000,
                           seed: int = 1) -> float:
    """
    Binary search for smallest eps in F_eps = (1-eps) F_0 + eps * delta_spike_index
    at which the MC chi^2 test achieves the target power.
    """
    ref_probs = np.asarray(ref_probs, dtype=float)
    spike = np.zeros_like(ref_probs)
    spike[spike_index] = 1.0
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        alt = (1 - mid) * ref_probs + mid * spike
        power = chi2_power_mc(
            ref_probs, alt, N, crit_value,
            n_replicates=n_replicates, seed=seed,
        )
        if power >= target_power:
            hi = mid
        else:
            lo = mid
    return hi


if __name__ == "__main__":
    # Cross-system reproducibility check. The SeeDance/lottery cell is the tie-atom
    # example (observed chi^2 = 31.0 sits on a null atom): with tie-inclusive counting
    # and seed=0 it MUST give p = 0.0490 on any platform/numpy build. If this assert
    # ever fails, the RNG stream or the tie tolerance has drifted.
    _counts = np.array([0, 0, 1, 0, 2, 0, 0, 2, 1, 1, 2, 1, 2, 2, 0, 0, 2, 6, 1, 1], dtype=float)
    _stat, _p, _ = chi2_mc(_counts, np.full(20, 1 / 20), n_replicates=50_000, seed=0)
    assert abs(_stat - 31.0) < 1e-9, f"stat drifted: {_stat}"
    assert abs(_p - 0.0490) < 1e-4, f"p drifted: {_p}"
    print(f"reproducibility OK: SeeDance/lottery chi2={_stat:.4f} p={_p:.4f} "
          f"(expected 31.0000, 0.0490)")
