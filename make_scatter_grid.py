#!/usr/bin/env python3
"""
Produce a single 3x3 grid figure with scorability vs TV scatter plots,
overlaid with the central-95% null band at the full-N reference level.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from constants import SCENE_PMF, MODELS

ANALYSIS_DIR = Path(__file__).parent / "analysis"
N_FULL = 32
N_MC = 20_000

SCENES = [
    ("galton_board_real", "Physical Galton board"),
    ("galton_board", "Animated Galton board"),
    ("ball", "Ball fork"),
    ("walking", "Walking T-junction"),
    ("pendulum", "Double pendulum"),
    ("dice", "Dice"),
    ("cards", "Cards"),
    ("lottery", "Lottery"),
    ("roulette", "Roulette"),
]

MODEL_LABELS = {
    "wan": "WAN-2.7",
    "seedance": "SeeDance-2.0",
    "happyhorse": "HappyHorse",
    "veo31": "Veo 3.1",
    "runway45": "Runway 4.5",
    "cosmos3_super": "Cosmos3-Super",
}
COLOURS = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))


def load_results(scene, model):
    path = ANALYSIS_DIR / f"{scene}_{model}_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _tv_band_at(p_0, n, n_mc=N_MC, seed=0):
    """Return (lo, hi) central-95% null band for TV at this n."""
    if n < 1:
        return 0.0, 1.0
    rng = np.random.default_rng(seed)
    sims = rng.multinomial(n, p_0, size=n_mc) / n
    tvs = 0.5 * np.abs(sims - p_0[None, :]).sum(axis=1)
    return float(np.quantile(tvs, 0.025)), float(np.quantile(tvs, 0.975))


def _tv(emp, p_0):
    return 0.5 * np.abs(np.asarray(emp) - p_0).sum()


def main():
    fig, axes = plt.subplots(3, 3, figsize=(11, 11), sharex=True)

    for ax, (scene, label) in zip(axes.flat, SCENES):
        ax.set_title(label, fontsize=12)
        p_0 = SCENE_PMF.get(scene)
        lo, hi = _tv_band_at(p_0, N_FULL) if p_0 is not None else (0.0, 1.0)
        ax.axhspan(lo, hi, color="gray", alpha=0.15, zorder=0,
                   label="_nolegend_")
        for model, colour in zip(MODELS, COLOURS):
            r = load_results(scene, model)
            if r is None:
                continue
            rho = r["reliability"]
            emp = r.get("empirical_distribution")
            if emp is None or p_0 is None:
                ax.scatter(rho, 0, marker="x", s=80, color="gray",
                           label="_nolegend_")
                continue
            tv = _tv(emp, p_0)
            ax.scatter(rho, tv, s=70, color=colour, label=MODEL_LABELS[model],
                       edgecolor="black", linewidth=0.5)
        ax.set_xlim(-0.05, 1.1)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    # Common axis labels
    for ax in axes[-1, :]:
        ax.set_xlabel("Scorability $\\rho$", fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\mathrm{TV}$ (lower = better)", fontsize=12)

    # Single legend at the bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=11,
               frameon=False)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path = ANALYSIS_DIR / "scatter_grid.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
