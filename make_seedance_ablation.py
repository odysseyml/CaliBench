#!/usr/bin/env python3
# SeeDance-2.0 480p-vs-720p resolution ablation: emits JSON + a standalone LaTeX table.
import json, numpy as np
from stats_utils import chi2_mc
from constants import SCENE_PMF

REF = {s: list(SCENE_PMF[s]) for s in SCENE_PMF}
SCENES = ["galton_board_real", "galton_board", "ball", "walking", "pendulum",
          "dice", "cards", "lottery", "roulette"]
PRETTY = {"galton_board_real": "Physical board", "galton_board": "Animated board", "ball": "Ball fork",
          "walking": "Walking", "pendulum": "Pendulum", "dice": "Dice", "cards": "Cards",
          "lottery": "Lottery", "roulette": "Roulette"}


def cell(path, ref):
    d = json.load(open(path))
    v = [r["bin"] for r in d["rollouts"] if r["bin"] is not None]
    n = len(v); k = len(ref); c = np.zeros(k)
    for x in v:
        c[x-1] += 1
    tv = float(0.5*np.abs(c/n - np.array(ref)).sum()) if n >= 1 else None
    p = None
    if n >= 4:   # main-pipeline rule: need >3 valid generations for a chi^2 p-value
        _, p, _ = chi2_mc(c, np.array(ref), n_replicates=50000, seed=0)
    return n, p, tv


rows = {}
for s in SCENES:
    n4, p4, tv4 = cell(f"analysis/{s}_seedance_bins.json", REF[s])
    n7, p7, tv7 = cell(f"analysis/seedance720/{s}_seedance_bins.json", REF[s])
    rows[s] = dict(n480=n4, p480=p4, tv480=tv4, n720=n7, p720=p7, tv720=tv7)
json.dump(rows, open("analysis/seedance_resolution_ablation.json", "w"), indent=2)


def fp(p, n):
    if n == 0:  return "---"
    if n < 4:   return r"$\dagger$"        # <=3 valid: no chi^2 test (dagger rule)
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def ft(tv, n):
    if n == 0:  return "---"
    return (f"{tv:.3f}" + (r"$^\dagger$" if n < 4 else ""))


L = [r"\begin{table}[t]\centering",
     r"\caption{SeeDance-2.0 at its benchmarked 480p vs 720p (same frames, prompts, 32 seeds;"
     r" only resolution changes). $\rho=n_{\mathrm{valid}}/32$; $p$ = MC $\chi^2$; TV vs the scene"
     r" reference. Strongly-miscalibrated scenes ($p<0.001$) are unchanged by resolution."
     r" $\dagger$: $\le 3$ valid generations, so no $\chi^2$ test is reported (main-text dagger rule).}",
     r"\label{tab:seedance_res}\small",
     r"\begin{tabular}{l cc cc cc}",
     r"\toprule",
     r" & \multicolumn{2}{c}{$\rho$} & \multicolumn{2}{c}{$p$} & \multicolumn{2}{c}{TV} \\",
     r"Scene & 480p & 720p & 480p & 720p & 480p & 720p \\ \midrule"]
for s in SCENES:
    r = rows[s]
    L.append(f"{PRETTY[s]} & {r['n480']/32:.2f} & {r['n720']/32:.2f} & "
             f"{fp(r['p480'], r['n480'])} & {fp(r['p720'], r['n720'])} & "
             f"{ft(r['tv480'], r['n480'])} & {ft(r['tv720'], r['n720'])} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open("analysis/seedance_resolution_ablation.tex", "w").write("\n".join(L)+"\n")
print("wrote analysis/seedance_resolution_ablation.json + analysis/seedance_resolution_ablation.tex")
for s in SCENES:
    r = rows[s]
    print(f"  {PRETTY[s]:<15} p: {fp(r['p480'], r['n480']):>10} -> {fp(r['p720'], r['n720']):>10}   "
          f"TV: {ft(r['tv480'], r['n480'])} -> {ft(r['tv720'], r['n720'])}")
