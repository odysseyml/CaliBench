#!/usr/bin/env python3
"""MCAR null-sensitivity analysis.

For the null-heavy scenes (cards, roulette, lottery, pendulum), treat the N - n_valid null
rollouts as Missing Completely At Random and impute them into the k reference bins two ways,
computing total variation over the FULL N under each:

  best  (min TV): greedy water-fill of the m nulls into the largest-deficit bins.
  worst (max TV): all m nulls on the single bin that maximises TV (a simplex vertex, since
                  TV is convex in the imputation vector -> the max is attained at a vertex).

We report, per cell: rho, the reported null-extended charge tv_total = (1-rho)+rho*TV(p_hat,p0)
(compute_chi2.py's metric), and [TV_best, TV_worst]. The reported charge treats every null as
maximally miscalibrated, so it upper-bounds the MCAR range. We also report each model's mnTV
range by swapping the four null-heavy scenes' score_cell to their best/worst imputation while
holding the other five scenes at their observed score_cell.

Reuses SCENE_PMF / tv / analytic_e_tv / ANALYSIS / MODEL_ORDER from compute_tv_band.py.
Outputs analysis/mcar_sensitivity.json and prints a standalone LaTeX table.
"""
import json
import numpy as np
from compute_tv_band import (SCENE_PMF, tv, analytic_e_tv, ANALYSIS, MODEL_ORDER)
from constants import PRETTY

SCENE_PRETTY = {"cards": "Cards", "roulette": "Roulette", "lottery": "Lottery", "pendulum": "Pendulum"}


def emit_latex(results, mntv_range):
    """Standalone LaTeX table (tab:mcar): per-cell best/worst/reported TV over the
    null-heavy scenes, plus per-model mnTV range. Columns follow TV_best <= TV_worst <= TV_rep."""
    L = [r"\begin{table}[t]\centering",
         r"\caption{MCAR sensitivity for the null-heavy scenes. For each cell we impute the "
         r"$N-n_{\mathrm{valid}}$ nulls to \emph{minimise} (best) and \emph{maximise} (worst) total "
         r"variation over the full $N$; $\mathrm{TV}_{\mathrm{rep}}=(1-\rho)+\rho\,\mathrm{TV}(\hat p,p_0)$ "
         r"is the reported null-extended charge. $\mathrm{TV}_{\mathrm{best}}\le\mathrm{TV}_{\mathrm{worst}}"
         r"\le\mathrm{TV}_{\mathrm{rep}}$ in every cell: the reported metric is the conservative upper bound.}",
         r"\label{tab:mcar}\small",
         r"\begin{tabular}{l l c c c c}", r"\toprule",
         r"Scene & Model & $\rho$ & TV$_{\mathrm{best}}$ & TV$_{\mathrm{worst}}$ & TV$_{\mathrm{rep}}$ \\",
         r"\midrule"]
    for scene in NULL_HEAVY:
        for model in MODEL_ORDER:
            r = results[scene].get(model)
            if r is None:
                continue
            L.append(f"{SCENE_PRETTY[scene]} & {PRETTY[model]} & {r['rho']:.2f} & "
                     f"{r['tv_best']:.3f} & {r['tv_worst']:.3f} & {r['tv_headline']:.3f} \\\\")
        L.append(r"\midrule")
    if L[-1] == r"\midrule":
        L.pop()
    L += [r"\bottomrule", r"\end{tabular}", r"\\[4pt]",
          r"\begin{tabular}{l c c c}", r"\toprule",
          r"Model & mnTV$_{\mathrm{best}}$ & mnTV$_{\mathrm{worst}}$ & mnTV (reported) \\", r"\midrule"]
    for model in MODEL_ORDER:
        v = mntv_range[model]
        L.append(f"{PRETTY[model]} & {v['mntv_best']:.3f} & {v['mntv_worst']:.3f} & {v['mntv_observed']:.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)

NULL_HEAVY = ["cards", "roulette", "lottery", "pendulum"]
ALL_SCENES = ["galton_board_real", "galton_board", "ball", "walking", "pendulum",
              "dice", "cards", "lottery", "roulette"]


def load_counts(scene, model, k):
    path = ANALYSIS / f"{scene}_{model}_bins.json"
    if not path.exists():
        return None
    rollouts = json.load(open(path))["rollouts"]
    n_total = len(rollouts)
    counts = np.zeros(k, dtype=float)
    for r in rollouts:
        if r["bin"] is not None:
            counts[r["bin"] - 1] += 1
    return counts, n_total


def tv_best(counts, n_total, p0):
    """Min full-N TV: greedy integer water-fill of the m nulls into largest-deficit bins."""
    g = counts.copy()
    m = int(round(n_total - counts.sum()))
    for _ in range(m):
        j = int(np.argmax(p0 - g / n_total))   # bin with the largest current deficit
        g[j] += 1
    return float(tv(g / n_total, p0))


def tv_worst(counts, n_total, p0):
    """Max full-N TV: all m nulls on one bin (vertex); brute-force over the k bins."""
    m = int(round(n_total - counts.sum()))
    best = -1.0
    for j in range(len(p0)):
        g = counts.copy(); g[j] += m
        best = max(best, float(tv(g / n_total, p0)))
    return best


def score_from_tv(tv_fullN, n_total, p0):
    """score_cell under full imputation (rho=1): excess against the N-sample null floor."""
    e0 = analytic_e_tv(n_total, p0)
    return (tv_fullN - e0) / (1.0 - e0)


def main():
    tvband_path = ANALYSIS / "tv_band_results.json"
    if not tvband_path.exists():
        raise SystemExit(
            f"{tvband_path} not found. This script depends on the per-scene TV/null-band "
            f"results; run `python compute_tv_band.py` first (it is the preceding step in "
            f"run_pipeline.sh)."
        )
    tvband = json.load(open(tvband_path))
    results = {}
    print(f"{'scene':<10}{'model':<14}{'rho':>6}{'TV_best':>9}{'TV_headline':>12}{'TV_worst':>10}")
    for scene in NULL_HEAVY:
        p0 = SCENE_PMF[scene]; k = len(p0)
        results[scene] = {}
        for model in MODEL_ORDER:
            lc = load_counts(scene, model, k)
            if lc is None:
                continue
            counts, n_total = lc
            n_valid = int(counts.sum()); rho = n_valid / n_total
            tv_valid = float(tv(counts / n_valid, p0)) if n_valid else 0.0
            tv_headline = (1 - rho) + rho * tv_valid
            b = tv_best(counts, n_total, p0)
            w = tv_worst(counts, n_total, p0)
            assert b <= tv_headline + 1e-9 and w <= tv_headline + 1e-9, (scene, model, b, w, tv_headline)
            results[scene][model] = dict(rho=rho, n_valid=n_valid,
                                         tv_best=b, tv_headline=float(tv_headline), tv_worst=w)
            print(f"{scene:<10}{model:<14}{rho:>6.2f}{b:>9.3f}{tv_headline:>12.3f}{w:>10.3f}")

    # --- per-model mnTV range: swap the 4 null-heavy scenes to best/worst score_cell ---
    mntv_range = {}
    for model in MODEL_ORDER:
        obs = [tvband[s][model]["score_cell"] for s in ALL_SCENES
               if tvband[s].get(model) is not None]
        base_other = {s: tvband[s][model]["score_cell"] for s in ALL_SCENES
                      if s not in NULL_HEAVY and tvband[s].get(model) is not None}
        best_scores = dict(base_other); worst_scores = dict(base_other)
        for scene in NULL_HEAVY:
            r = results[scene].get(model)
            if r is None:
                continue
            p0 = SCENE_PMF[scene]
            counts, n_total = load_counts(scene, model, len(p0))
            best_scores[scene] = score_from_tv(r["tv_best"], n_total, p0)
            worst_scores[scene] = score_from_tv(r["tv_worst"], n_total, p0)
        n = len(ALL_SCENES)
        mntv_range[model] = dict(
            mntv_observed=float(np.mean(obs)) if obs else None,
            mntv_best=float(np.mean(list(best_scores.values()))),
            mntv_worst=float(np.mean(list(worst_scores.values()))),
        )
    results["_mntv_range"] = mntv_range

    with open(ANALYSIS / "mcar_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(ANALYSIS / "mcar_sensitivity.tex", "w") as f:
        f.write(emit_latex(results, mntv_range) + "\n")
    print("\nper-model mnTV range [best, worst] (observed):")
    for m, v in mntv_range.items():
        print(f"  {m:<14} [{v['mntv_best']:.3f}, {v['mntv_worst']:.3f}]  (obs {v['mntv_observed']:.3f})")
    print(f"\nSaved -> {ANALYSIS/'mcar_sensitivity.json'}")


if __name__ == "__main__":
    main()
