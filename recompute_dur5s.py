#!/usr/bin/env python3
"""Recompute the headline results under a UNIFORM 5-second duration and write a
comparison to ablation_5s_results.md (does not touch main.tex or the committed JSONs).

Builds a combined bins dir: the four re-run models (WAN, SeeDance, HappyHorse, Veo)
from analysis/dur5s/ (generated at duration=5), and Runway Gen-4.5 (already 5 s) +
Cosmos3-Super (~5.04 s) from the committed bins. Then reruns the chi^2/FDR and TV/mnTV
analyses over that combined set and diffs against the committed (mixed-duration) results.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
ANALYSIS = ROOT / "analysis"
DUR5S = ANALYSIS / "dur5s"
COMBINED = ANALYSIS / "dur5s_combined"

RERUN = ["wan", "seedance", "happyhorse"]            # generated at 5 s
# Reused from committed bins: Runway (already 5 s), Cosmos (~5.04 s), and Veo 3.1 --
# Veo's API only accepts duration {4, 6, 8}, so it cannot run at 5 s (stays at 4 s).
REUSED = ["veo31", "runway45", "cosmos3_super"]

from constants import SCENES, MODELS, PRETTY, SCENE_PMF  # noqa: E402
import compute_chi2 as C                                 # noqa: E402
import compute_tv_band as T                              # noqa: E402


def build_combined():
    COMBINED.mkdir(parents=True, exist_ok=True)
    missing = []
    for scene in SCENES:
        for model in MODELS:
            src = (DUR5S if model in RERUN else ANALYSIS) / f"{scene}_{model}_bins.json"
            dst = COMBINED / f"{scene}_{model}_bins.json"
            if src.exists():
                shutil.copy(src, dst)
            else:
                missing.append(src.name)
    if missing:
        print(f"WARNING: {len(missing)} bins missing (cells with no committed baseline "
              f"or not yet generated): {missing[:6]}{'...' if len(missing) > 6 else ''}")


def rerun_chi2():
    C.ANALYSIS_DIR = COMBINED
    C.main()                       # writes COMBINED/chi2_results.json
    return json.load(open(COMBINED / "chi2_results.json"))


def tv_mntv():
    """Per-cell rho/TV and per-model mnTV over the combined bins (reuses compute_tv_band)."""
    T.ANALYSIS = COMBINED
    cells, mntv = {}, {}
    for model in MODELS:
        scores = []
        for scene in T.SCENE_ORDER:
            c = T.per_cell_tv(scene, model, T.SCENE_PMF[scene])
            if c is None:
                continue
            cells[(scene, model)] = c
            if c["score_cell"] is not None:
                scores.append(c["score_cell"])
        mntv[model] = float(sum(scores) / len(scores)) if scores else None
    return cells, mntv


def fmt_p(v):
    if v is None:
        return "---"
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def main():
    build_combined()
    new_chi2 = rerun_chi2()
    old_chi2 = json.load(open(ANALYSIS / "chi2_results.json"))
    new_cells, new_mntv = tv_mntv()
    old_tv = json.load(open(ANALYSIS / "tv_band_results.json"))
    old_mntv = old_tv["_mntv"]

    L = []
    L.append("# Uniform 5-second duration ablation")
    L.append("")

    # --- per-model mnTV ---
    L.append("## Per-model mnTV (lower = better)")
    L.append("")
    L.append("| Model | committed | 5 s | Δ |")
    L.append("|---|---|---|---|")
    for m in MODELS:
        o, n = old_mntv.get(m), new_mntv.get(m)
        if o is None or n is None:
            L.append(f"| {PRETTY[m]} | {o} | {n} | — |")
        else:
            tag = "  *(reused)*" if m in REUSED else ""
            L.append(f"| {PRETTY[m]} | {o:.3f} | {n:.3f} | {n - o:+.3f}{tag} |")
    L.append("")

    # --- FDR family ---
    om, nm = old_chi2["_fdr_meta"], new_chi2["_fdr_meta"]
    L.append("## χ² family / FDR")
    L.append("")
    L.append(f"- Testable cells (family): committed **{om['family_size']}** → 5 s **{nm['family_size']}**")
    L.append(f"- Significant after BH: committed **{om['n_significant_bh']}** → 5 s **{nm['n_significant_bh']}**")
    L.append(f"- BH threshold p: committed {om['bh_threshold_p']:.4f} → 5 s {nm['bh_threshold_p']:.4f}")
    L.append("")

    # --- per-cell for the re-run models ---
    L.append("## Per-cell change (re-run models only)")
    L.append("")
    L.append("ρ = scorability, TV = total variation, p = MC χ² p-value.")
    L.append("")
    L.append("| Scene | Model | ρ (old→5s) | TV (old→5s) | p (old→5s) | verdict flip? |")
    L.append("|---|---|---|---|---|---|")
    for scene in T.SCENE_ORDER:
        for m in RERUN:
            oc = old_tv.get(scene, {}).get(m)
            nc = new_cells.get((scene, m))
            if oc is None or nc is None:
                continue
            op = old_chi2.get(scene, {}).get(m, {}).get("p_valid")
            npv = new_chi2.get(scene, {}).get(m, {}).get("p_valid")
            oq = old_chi2.get(scene, {}).get(m, {}).get("q_bh")
            nq = new_chi2.get(scene, {}).get(m, {}).get("q_bh")
            old_sig = (oq is not None and oq < 0.05)
            new_sig = (nq is not None and nq < 0.05)
            flip = "" if old_sig == new_sig else f"**{'sig→ns' if old_sig else 'ns→sig'}**"
            ov = oc.get("tv_valid"); nv = nc.get("tv_valid")
            tvs = f"{(ov if ov is not None else float('nan')):.2f}→{(nv if nv is not None else float('nan')):.2f}"
            rhos = f"{oc['rho']:.2f}→{nc['rho']:.2f}"
            L.append(f"| {scene} | {PRETTY[m]} | {rhos} | {tvs} | {fmt_p(op)}→{fmt_p(npv)} | {flip} |")
    L.append("")
    L.append("_ρ/TV `nan` = no valid generations. Verdict = significant at BH α=0.05._")

    out = ROOT / "ablation_5s_results.md"
    out.write_text("\n".join(L) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
