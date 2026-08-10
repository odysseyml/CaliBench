#!/usr/bin/env python3
"""
Rollout analysis: scorability + empirical outcome distributions across scenes
and models.

Per model:
  reliability             = fraction of rollouts with a valid outcome
  empirical_distribution  = outcome frequencies over the valid rollouts

Significance verdicts are NOT computed here — the headline statistics (the
Monte Carlo exact χ² test and the TV null-band) live in compute_chi2.py and
compute_tv_band.py. This script produces the per-scene comparison plots.

Run with:
  python analyze_rollouts.py --scene galton_board --model wan
  python analyze_rollouts.py --scene dice --model all
  python analyze_rollouts.py --scene all --model all
"""
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from constants import SCENE_PMF

ANALYSIS_DIR = Path(__file__).parent / "analysis"
KNOWN_MODELS = ["wan", "seedance", "happyhorse", "veo31", "runway45", "cosmos3_super"]
MODEL_LABELS = {
    "wan":        "WAN-2.7",
    "seedance":   "SeeDance-2.0",
    "happyhorse": "HappyHorse",
    "veo31":      "Veo 3.1",
    "runway45":   "Runway Gen-4.5",
    "cosmos3_super": "Cosmos3-Super",
}

# Per-scene labels for the outcome (x) axis on comparison plots. Scenes not
# listed here use the integer outcome index as the label.
OUTCOME_LABELS = {
    "ball":     ["left", "right"],
    "walking":  ["left", "right"],
    "pendulum": ["left", "right"],
    "cards":    ["hearts", "diamonds", "clubs", "spades"],
    "roulette": ["red", "black", "green"],
}


# Reference PMF for each scene (shared source of truth in constants.py).
SCENE_REFERENCE_PMFS = SCENE_PMF


def analyse_model(scene_name, model_name):
    bins_path = ANALYSIS_DIR / f"{scene_name}_{model_name}_bins.json"
    if not bins_path.exists():
        print(f"  No bins file for {model_name} — run extract_bin.py first")
        return None

    with open(bins_path) as f:
        data = json.load(f)

    num_bins = data["num_bins"]
    rollouts = data["rollouts"]
    n_rollouts = len(rollouts)
    valid = [r for r in rollouts if r["bin"] is not None]
    n_valid = len(valid)
    reliability = n_valid / n_rollouts if n_rollouts > 0 else 0.0

    reference_pmf = SCENE_REFERENCE_PMFS.get(scene_name)
    if reference_pmf is not None:
        expected = np.asarray(reference_pmf)
    else:
        expected = None

    print(f"\n=== {model_name} ===")
    print(f"  Reliability: {n_valid}/{n_rollouts} = {reliability:.2f}")

    if n_valid >= 1:
        counts = np.zeros(num_bins, dtype=int)
        for r in valid:
            counts[r["bin"] - 1] += 1
        empirical = counts / n_valid
        outcome_values = np.array([r["bin"] for r in valid], dtype=float)
        empirical_std = float(np.std(outcome_values))
        print(f"  Empirical : {np.round(empirical, 3).tolist()}")
        print(f"  Std(outcomes) = {empirical_std:.3f}")
    else:
        empirical = None
        empirical_std = None

    if n_valid < 2 or expected is None:
        if expected is None:
            print(f"  No reference distribution for {scene_name}")
        else:
            print(f"  Too few valid rollouts to plot a distribution")
        results = {
            "model": model_name,
            "n_rollouts": n_rollouts,
            "n_valid": n_valid,
            "reliability": reliability,
            "empirical_distribution": empirical.tolist() if empirical is not None else None,
            "empirical_std": empirical_std,
            "expected_distribution": expected.tolist() if expected is not None else None,
        }
        results_path = ANALYSIS_DIR / f"{scene_name}_{model_name}_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        return results

    # Per-model histogram
    x = np.arange(num_bins)
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, empirical, width, alpha=0.8, label="Model", color="steelblue")
    ax.bar(x + width / 2, expected, width, alpha=0.8, label="Reference", color="orange")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Proportion")
    ax.set_xticks(x)
    ax.set_xticklabels(range(1, num_bins + 1))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / f"{scene_name}_{model_name}_histogram.pdf")
    plt.close()

    results = {
        "model": model_name,
        "n_rollouts": n_rollouts,
        "n_valid": n_valid,
        "reliability": reliability,
        "empirical_distribution": empirical.tolist(),
        "empirical_std": empirical_std,
        "expected_distribution": expected.tolist(),
    }
    results_path = ANALYSIS_DIR / f"{scene_name}_{model_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def scatter_plot(all_results, scene_name):
    """Reliability vs empirical-outcome spread (higher σ = less mode collapse)."""
    fig, ax = plt.subplots(figsize=(6, 5))

    for r in all_results:
        y = r["empirical_std"] if r["empirical_std"] is not None else 0.0
        ax.scatter(r["reliability"], y, s=60)
        ax.annotate(r["model"], (r["reliability"], y),
                    textcoords="offset points", xytext=(6, 3), fontsize=9)

    ax.set_xlabel("Reliability (fraction valid outputs)")
    ax.set_ylabel("Empirical std of outcome (higher = less collapse)")
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = ANALYSIS_DIR / f"{scene_name}_reliability_vs_calibration.pdf"
    fig.savefig(out_path)
    plt.close()
    print(f"Scatter plot saved → {out_path}")


def comparison_plot(all_results, scene_name):
    """Empirical distributions for each model vs reference, single row.

    Sized to match LaTeX \\linewidth (~6.5in) so fonts at the requested
    point sizes survive the include-at-linewidth scaling.
    """
    valid_results = [r for r in all_results
                     if r["empirical_distribution"] is not None
                     and r["expected_distribution"] is not None]
    if not valid_results:
        return

    num_bins = len(valid_results[0]["expected_distribution"])
    expected = np.array(valid_results[0]["expected_distribution"])
    x = np.arange(num_bins)
    n_panels = len(valid_results) + 1

    fig, axes = plt.subplots(1, n_panels, figsize=(1.4 * n_panels, 2.0),
                             sharey=True)

    panels = [("Reference", expected, "orange")] + [
        (MODEL_LABELS.get(res["model"], res["model"]),
         res["empirical_distribution"], "steelblue")
        for res in valid_results
    ]

    labels = OUTCOME_LABELS.get(scene_name)
    use_named = labels is not None and len(labels) == num_bins
    if use_named:
        tick_positions = x
        tick_labels = labels
    else:
        # round() rather than floor so an 11-bin axis steps by 2 (ticks
        # 1,3,…,11) instead of 1 — the two-digit 10/11 labels were crowding
        # the right edge of each narrow panel.
        step = max(1, round(num_bins / 6))
        tick_positions = x[::step]
        tick_labels = list(range(1, num_bins + 1, step))

    for i, (ax, (title, data, colour)) in enumerate(zip(axes, panels)):
        ax.bar(x, np.array(data), color=colour, alpha=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Outcome", fontsize=9)
        if i == 0:
            ax.set_ylabel("Proportion", fontsize=9)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=8,
                           rotation=30 if use_named else 0,
                           ha="right" if use_named else "center")
        ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    out_path = ANALYSIS_DIR / f"{scene_name}_comparison.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved → {out_path}")


def comparison_plot_no_ref(all_results, scene_name, outcome_labels=None):
    """Empirical-only distribution plot for scenes with no reference distribution."""
    valid_results = [r for r in all_results if r["empirical_distribution"] is not None]
    if not valid_results:
        return

    num_bins = len(valid_results[0]["empirical_distribution"])
    x = np.arange(num_bins)
    n_panels = len(valid_results)

    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, res in zip(axes, valid_results):
        ax.bar(x, np.array(res["empirical_distribution"]), color="steelblue", alpha=0.8)
        ax.set_title(f"{res['model']} (n={res['n_valid']})")
        ax.set_xlabel("Outcome")
        ax.set_xticks(x)
        labels = outcome_labels if outcome_labels else [str(i + 1) for i in range(num_bins)]
        ax.set_xticklabels(labels)

    axes[0].set_ylabel("Proportion")
    fig.tight_layout()
    out_path = ANALYSIS_DIR / f"{scene_name}_comparison.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot (empirical only) saved → {out_path}")


def main():
    all_scene_names = list(SCENE_REFERENCE_PMFS.keys())
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="galton_board",
                        help=f"Scene: {', '.join(all_scene_names)} or 'all'")
    parser.add_argument("--model", default="all",
                        help=f"Model: {', '.join(KNOWN_MODELS)} or 'all'")
    args = parser.parse_args()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    scenes = all_scene_names if args.scene == "all" else [args.scene]
    models = KNOWN_MODELS if args.model == "all" else [args.model]

    for scene_name in scenes:
        print(f"\n{'='*50}")
        print(f"Scene: {scene_name}")
        print(f"{'='*50}")
        all_results = []
        for m in models:
            r = analyse_model(scene_name, m)
            if r:
                all_results.append(r)

        if len(all_results) > 1:
            has_ref = any(r["expected_distribution"] is not None for r in all_results)
            if has_ref:
                comparison_plot(all_results, scene_name)
            else:
                comparison_plot_no_ref(all_results, scene_name)
            scatter_plot(all_results, scene_name)


if __name__ == "__main__":
    main()
