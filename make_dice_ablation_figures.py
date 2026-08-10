#!/usr/bin/env python3
"""
Two figures for the dice ablation appendices:
  - cfg_sweep_histogram.pdf: 8 panels (reference + 7 CFG values), empirical
    dice distribution at each CFG for Cosmos3-Super.
  - target_face_compliance.pdf: 6 panels (one per model), 6x7 confusion matrix
    heatmap (requested face × rolled face + null), with overall compliance.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

ANALYSIS = Path(__file__).parent / "analysis"
OUT_DIR = ANALYSIS

MODELS = ["wan", "seedance", "happyhorse", "veo31", "runway45", "cosmos3_super"]
PRETTY = {"wan": "WAN-2.7", "seedance": "SeeDance-2.0", "happyhorse": "HappyHorse",
          "veo31": "Veo 3.1", "runway45": "Runway 4.5", "cosmos3_super": "Cosmos3-Super"}
CFG_VALUES = ["1_0", "1_5", "3_0", "4_5", "6_0", "7_5", "9_0"]
CFG_LABELS = {"1_0": "1.0", "1_5": "1.5", "3_0": "3.0", "4_5": "4.5",
              "6_0": "6.0", "7_5": "7.5", "9_0": "9.0"}
FACES = [1, 2, 3, 4, 5, 6]


def cfg_sweep_histogram():
    # Reference + 7 CFG panels = 8; lay out 2 rows x 4 cols so each panel is
    # wide enough that titles/ticks stay legible at \linewidth.
    panels = len(CFG_VALUES) + 1
    ncol = 4
    nrow = -(-panels // ncol)  # ceil
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 5.2), sharey=True)
    axes = axes.flatten()
    expected = np.full(6, 1 / 6.0)
    axes[0].bar(np.arange(6), expected, color="orange", alpha=0.85)
    axes[0].set_title("Reference\n(uniform $1/6$)", fontsize=13)
    for i, cfg_id in enumerate(CFG_VALUES):
        ax = axes[i + 1]
        d = json.load(open(ANALYSIS / f"dice_cfg_{cfg_id}_cosmos3_super_bins.json"))
        valid = [r for r in d["rollouts"] if r["bin"] is not None]
        nv = len(valid)
        counts = np.zeros(6, dtype=int)
        for r in valid:
            counts[r["bin"] - 1] += 1
        emp = counts / nv if nv else np.zeros(6)
        ax.bar(np.arange(6), emp, color="steelblue", alpha=0.85)
        cfg_label = CFG_LABELS[cfg_id]
        ax.set_title(f"CFG = {cfg_label}\n($n_{{\\rm valid}}={nv}$)", fontsize=13)
    # y-label on the first column of each row
    for r in range(nrow):
        axes[r * ncol].set_ylabel("Proportion", fontsize=12)
    # hide any unused panels
    for j in range(panels, len(axes)):
        axes[j].set_visible(False)
    for ax in axes[:panels]:
        ax.set_xticks(np.arange(6))
        ax.set_xticklabels(FACES)
        ax.set_xlabel("Face", fontsize=12)
        ax.set_ylim(0, 0.55)
        ax.tick_params(axis="y", labelsize=11)
        ax.tick_params(axis="x", labelsize=11)
    fig.suptitle("Cosmos3-Super dice: empirical distribution by CFG", fontsize=14)
    fig.tight_layout()
    out = OUT_DIR / "cfg_sweep_histogram.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"-> {out}")


def target_face_compliance():
    """One 6×7 heatmap per model (requested face × {1..6, null})."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    cmap = LinearSegmentedColormap.from_list("blueish", ["white", "#1f4e8c"])

    for ax, model in zip(axes.flat, MODELS):
        mat = np.zeros((6, 7), dtype=int)
        for f in FACES:
            d = json.load(open(ANALYSIS / f"dice_face_{f}_{model}_bins.json"))
            for r in d["rollouts"]:
                if r["bin"] is None:
                    mat[f - 1, 6] += 1
                else:
                    mat[f - 1, r["bin"] - 1] += 1
        diag_n = sum(mat[i, i] for i in range(6))
        total_valid = mat[:, :6].sum()
        compliance = diag_n / total_valid if total_valid else 0.0

        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=5)
        for i in range(6):
            for j in range(7):
                v = mat[i, j]
                if v == 0:
                    continue
                colour = "white" if v >= 3 else "black"
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=9, color=colour)

        # diagonal box outline
        for i in range(6):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                        edgecolor="black", lw=1.3))

        ax.set_xticks(range(7))
        ax.set_xticklabels([str(c) for c in FACES] + ["null"], fontsize=8)
        ax.set_yticks(range(6))
        ax.set_yticklabels(FACES, fontsize=8)
        ax.set_xlabel("Rolled face", fontsize=9)
        ax.set_ylabel("Requested face", fontsize=9)
        ax.set_title(f"{PRETTY[model]}  ({compliance:.0%} compliance)", fontsize=10)

    fig.suptitle("Explicit-target-face dice ablation: 5 seeds per (model, requested face)",
                 fontsize=11, y=1.00)
    fig.tight_layout()
    out = OUT_DIR / "target_face_compliance.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"-> {out}")


if __name__ == "__main__":
    cfg_sweep_histogram()
    target_face_compliance()
