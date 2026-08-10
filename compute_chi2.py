#!/usr/bin/env python3
"""
Compute MC chi^2 test results for each (scene, model) cell.

Per cell:
- chi^2 valid: Monte Carlo exact chi^2 goodness-of-fit on the valid (non-null)
  rollouts vs the scene's reference (p-value and critical value taken from the
  null sampling distribution, not the asymptotic chi^2).
- TV_total: null-extended total variation distance,
  (1 - rho) + rho * TV(p_valid, p_ref).

Also applies Benjamini-Hochberg (FDR) and Bonferroni multiple-comparison
corrections across all cells with a finite p-value.

Outputs: analysis/chi2_results.json with per-cell numbers.
"""
import json
import numpy as np
from pathlib import Path

from stats_utils import chi2_mc, benjamini_hochberg, bonferroni
from constants import SCENES, MODELS, SCENE_PMF

ANALYSIS_DIR = Path(__file__).parent / "analysis"

# Reference distribution per scene, as plain lists (from the shared canonical PMFs).
REF = {scene: list(SCENE_PMF[scene]) for scene in SCENES}


def per_cell(bins_path, ref):
    with open(bins_path) as f:
        data = json.load(f)
    rollouts = data["rollouts"]
    N = len(rollouts)
    k = len(ref)

    # --- valid-only chi^2 ---
    valid = [r["bin"] for r in rollouts if r["bin"] is not None]
    n_valid = len(valid)
    rho = n_valid / N
    counts_valid = np.zeros(k)
    for v in valid:
        counts_valid[v - 1] += 1

    # A cell needs >3 valid generations to enter the tested family: a chi^2
    # goodness-of-fit p from <=3 samples is unreliable, so such cells are marked
    # dagger and excluded (matches the Table 2 caption convention).
    if n_valid < 4:
        chi2_v = p_v = crit_v = None
        collapse = False
    else:
        chi2_v, p_v, crit_v = chi2_mc(counts_valid, np.asarray(ref),
                                      n_replicates=50_000, seed=0)
        collapse = bool(counts_valid.max() == n_valid)

    # --- null-extended total variation distance (TV_total) ---
    # TV_total = (1 - rho) + rho * TV(p_valid, p_ref)
    # where TV(p, q) = 0.5 * sum_i |p_i - q_i| in [0, 1].
    # The (1 - rho) term: nulls contribute the maximum mismatch (1) because the
    # reference puts P=0 on the null outcome.
    # Properties:
    #   TV_total in [0, 1]; = 1 when rho = 0; = TV(p_valid, p_ref) when rho = 1.
    if n_valid >= 1:
        p_valid = counts_valid / n_valid
        tv_valid = float(0.5 * np.sum(np.abs(p_valid - np.asarray(ref))))
    else:
        tv_valid = None
    tv_total = (1 - rho) + rho * (tv_valid if tv_valid is not None else 0.0)

    return dict(
        N=N, rho=rho, n_valid=n_valid,
        chi2_valid=chi2_v, p_valid=p_v, crit_valid=crit_v,
        collapse=collapse,
        tv_valid=tv_valid, tv_total=tv_total,
    )


def main():
    print(f"{'Scene':<22} {'Model':<11}  {'rho':>5}  {'chi2':>7}  {'p_MC':>6}  {'TV_tot':>7}  collapse?")
    print("-" * 80)
    table = {}
    for scene in SCENES:
        table[scene] = {}
        for model in MODELS:
            path = ANALYSIS_DIR / f"{scene}_{model}_bins.json"
            if not path.exists():
                table[scene][model] = None
                continue
            r = per_cell(path, REF[scene])
            table[scene][model] = r
            cv = "---" if r["chi2_valid"] is None else f"{r['chi2_valid']:.2f}"
            pv = "---" if r["p_valid"] is None else f"{r['p_valid']:.3f}"
            tt = f"{r['tv_total']:.3f}"
            mark = "*" if r["collapse"] else " "
            print(f"  {scene:<20} {model:<11}  {r['rho']:>5.2f}  "
                  f"{cv:>7}  {pv:>6}  {tt:>7}  {mark}")

    # --- Multiple-comparison correction across all computed tests ---------
    # Family = every cell with a finite p-value (cells with too few valid
    # generations to test have p_valid=None and are excluded: you can only
    # correct tests you actually ran). BH controls the false discovery rate;
    # Bonferroni (FWER) is stored as a conservative reference.
    family = [(s, m) for s in SCENES for m in MODELS
              if table[s].get(m) is not None
              and table[s][m]["p_valid"] is not None]
    pvals = np.array([table[s][m]["p_valid"] for s, m in family])
    q_bh = benjamini_hochberg(pvals)
    p_bonf = bonferroni(pvals)
    alpha = 0.05
    for (s, m), q, pb in zip(family, q_bh, p_bonf):
        table[s][m]["q_bh"] = float(q)
        table[s][m]["p_bonferroni"] = float(pb)
        table[s][m]["sig_bh_05"] = bool(q < alpha)

    n_fam = len(family)
    # BH critical p = largest raw p among rejected cells (q-values are monotone,
    # so {q < alpha} is exactly the BH-rejected set).
    rejected = q_bh < alpha
    bh_threshold = float(pvals[rejected].max()) if rejected.any() else 0.0
    n_sig_bh = int(rejected.sum())
    table["_fdr_meta"] = dict(
        method="benjamini-hochberg", alpha=alpha, family_size=n_fam,
        bh_threshold_p=bh_threshold, bonferroni_threshold_p=alpha / n_fam,
        n_significant_bh=n_sig_bh,
    )

    out = ANALYSIS_DIR / "chi2_results.json"
    with open(out, "w") as f:
        json.dump(table, f, indent=2)
    print(f"\nSaved -> {out}")

    print()
    print("=== Multiple-comparison correction (Benjamini-Hochberg FDR) ===")
    print(f"  family size (finite p-values): {n_fam}")
    print(f"  BH critical p-threshold (alpha={alpha}): {bh_threshold:.5f}")
    print(f"  Bonferroni p-threshold: {alpha / n_fam:.5f}")
    print(f"  significant after BH: {n_sig_bh}/{n_fam}")
    flips = [(s, m, table[s][m]["p_valid"], table[s][m]["q_bh"])
             for s, m in family
             if table[s][m]["p_valid"] < alpha and table[s][m]["q_bh"] >= alpha]
    print(f"  cells significant at raw alpha but NOT after BH: {len(flips)}")
    for s, m, p, q in flips:
        print(f"    {s:<20} {m:<13}  raw p={p:.4f}  ->  q_BH={q:.4f}")

    # The headline benchmark scalar (mnTV, with the per-cell null
    # floor subtracted) is computed in compute_tv_band.py. We deliberately do
    # not print a competing floor-free aggregate here.


if __name__ == "__main__":
    main()
