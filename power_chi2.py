#!/usr/bin/env python3
"""
Monte Carlo power analysis for the chi^2 calibration test at N=32.

For each scene, we use a single-parameter contamination family and report
the smallest contamination level epsilon (and the corresponding total
variation distance) at which the MC chi^2 test rejects with probability >= 0.80.
"""
import argparse
import numpy as np
from scipy.stats import binom as binom_dist

from stats_utils import chi2_mc, chi2_power_mc, min_detectable_epsilon


# Reference distributions and contamination spike index per scene.
SCENES = {
    "Galton board":   (np.array([binom_dist.pmf(i, 10, 0.5) for i in range(11)]), 6),  # spike on bin 7 (index 6 -> outcome 7); the modal outcome is 6 (index 5). Adjust: spike on the modal bin
    "Ball / Walking / Pendulum": (np.array([0.5, 0.5]), 0),
    "Dice":           (np.array([1/6]*6), 0),
    "Cards":          (np.array([0.25]*4), 0),
    "Lottery":        (np.array([1/20]*20), 0),
    "Roulette":       (np.array([18/37, 18/37, 1/37]), 2),   # spike on green (index 2)
}

# Fix: Galton modal bin is bin 6 (outcome 6, i.e., index 5 if outcomes are 1..11)
# The reference array indices are 0..10 corresponding to outcomes 1..11 (bin labels).
# The modal bin for Bin(10, 0.5) is at index 5 (outcome 6).
SCENES["Galton board"] = (SCENES["Galton board"][0], 5)


def compute_for_scene(label, ref, spike_idx, N=32, alpha=0.05,
                      target_power=0.80, seed_crit=0, seed_pow=1):
    crit_value = float(np.quantile(
        # use chi2_mc to get the critical value
        _null_chi2_distribution(ref, N, n_replicates=50_000, seed=seed_crit),
        1 - alpha,
    ))

    eps = min_detectable_epsilon(
        ref, spike_idx, N, crit_value,
        target_power=target_power, n_replicates=5_000, seed=seed_pow,
    )
    spike = np.zeros_like(ref)
    spike[spike_idx] = 1.0
    alt = (1 - eps) * ref + eps * spike
    tv = float(0.5 * np.abs(alt - ref).sum())
    return crit_value, eps, tv


def _null_chi2_distribution(ref, N, n_replicates, seed):
    rng = np.random.default_rng(seed)
    expected = N * ref
    null_counts = rng.multinomial(N, ref, size=n_replicates).astype(float)
    return ((null_counts - expected) ** 2 / expected).sum(axis=1)


def make_table(N=32):
    rows = []
    print(f"\nMC chi^2 power at N={N}, alpha=0.05, target power=0.80")
    print(f"{'Scene':<32}  {'k':>3}  {'c_a':>7}  {'min eps':>8}  {'min TV':>7}")
    print("-" * 65)
    for label, (ref, idx) in SCENES.items():
        crit, eps, tv = compute_for_scene(label, ref, idx, N=N)
        rows.append((label, len(ref), crit, eps, tv))
        print(f"  {label:<30}  {len(ref):>3}  {crit:>7.2f}  {eps:>8.3f}  {tv:>7.3f}")
    return rows


def make_plot(rows_n32):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    n_values = [8, 16, 24, 32]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(n_values)))

    for ax, (label, k, _, _, _) in zip(axes, rows_n32):
        ref, spike_idx = SCENES[label]
        spike = np.zeros_like(ref)
        spike[spike_idx] = 1.0
        # Map epsilon -> TV monotonically for this scene
        eps_grid = np.linspace(0.01, 0.95, 30)
        tvs, alts = [], []
        for e in eps_grid:
            alt = (1 - e) * ref + e * spike
            tvs.append(0.5 * np.abs(alt - ref).sum())
            alts.append(alt)
        for n, col in zip(n_values, colors):
            crit = float(np.quantile(
                _null_chi2_distribution(ref, n, n_replicates=10_000, seed=0),
                0.95,
            ))
            powers = [chi2_power_mc(ref, alt, n, crit, n_replicates=2000, seed=1)
                      for alt in alts]
            ax.plot(tvs, powers, color=col, label=f"N={n}")
        ax.axhline(0.80, color="red", ls="--", lw=0.8, label="80% power")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel(r"$\mathrm{TV}$")
        ax.set_ylabel("Power")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7)

    fig.tight_layout()
    out = "analysis/power_curve.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"\nSaved -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    rows_n32 = make_table(N=32)
    print()
    rows_n16 = make_table(N=16)

    # Combined table for paper
    print("\n\n% LaTeX table")
    print(r"\begin{tabular}{l r r r r r}")
    print(r"\toprule")
    print(r"Scene & $k$ & $c_\alpha$ & Min TV ($N{=}32$) & Min TV ($N{=}16$) \\")
    print(r"\midrule")
    for (label, k, crit, _, tv32), (_, _, _, _, tv16) in zip(rows_n32, rows_n16):
        print(f"  {label} & {k} & {crit:.2f} & {tv32:.3f} & {tv16:.3f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    if args.plot:
        make_plot(rows_n32)


if __name__ == "__main__":
    main()
