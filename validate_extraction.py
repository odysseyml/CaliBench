#!/usr/bin/env python3
"""
Reproduce the VLM-extraction validation table.

Joins the frozen human labels in analysis/human_labels.json against the
current VLM outcomes in analysis/{scene}_{model}_bins.json (matched by
scene/model/seed) and reports, per scene and overall:

  - Agreement : VLM label == human label (null treated as a distinct value)
  - FN        : false-null  (VLM null, human gave an outcome) / human outcomes
  - MN        : missed-null (VLM gave an outcome, human null)  / human nulls
  - WL        : wrong-label (both gave outcomes, but disagree)

Run:
  python validate_extraction.py
"""
import json
from collections import defaultdict
from pathlib import Path

ANALYSIS_DIR = Path(__file__).parent / "analysis"

SCENES = ["galton_board", "galton_board_real", "ball", "walking", "pendulum",
          "dice", "cards", "lottery", "roulette"]
SCENE_LABELS = {
    "galton_board": "Animated board", "galton_board_real": "Physical board",
    "ball": "Ball fork", "walking": "Walking", "pendulum": "Pendulum",
    "dice": "Dice", "cards": "Cards", "lottery": "Lottery", "roulette": "Roulette",
}
# VLM bins store integers for these scenes; map back to the human label strings.
INT2STR = {
    "ball": {1: "left", 2: "right"},
    "walking": {1: "left", 2: "right"},
    "pendulum": {1: "left", 2: "right"},
    "cards": {1: "hearts", 2: "diamonds", 3: "clubs", 4: "spades"},
    "roulette": {1: "red", 2: "black", 3: "green"},
}


def norm(scene, label):
    """Normalise a label (VLM int or human value) to a comparable string / None."""
    if label is None:
        return None
    if isinstance(label, int):
        return INT2STR.get(scene, {}).get(label, str(label))
    return str(label)


def current_bin(scene, model, seed):
    path = ANALYSIS_DIR / f"{scene}_{model}_bins.json"
    if not path.exists():
        return "MISSING"
    for r in json.load(open(path))["rollouts"]:
        if r["seed"] == seed:
            return r["bin"]
    return "MISSING"


def main():
    human = json.load(open(ANALYSIS_DIR / "human_labels.json"))
    by_scene = defaultdict(list)
    for e in human:
        vlm = current_bin(e["scene"], e["model"], e["seed"])
        if vlm == "MISSING":
            continue
        by_scene[e["scene"]].append((norm(e["scene"], vlm), norm(e["scene"], e["human_label"])))

    print(f"{'Scene':<16} {'n':>3} {'Agree':>9} {'FN':>7} {'MN':>7} {'WL':>3}")
    print("-" * 50)
    tot = dict(n=0, agree=0, fn=0, fn_d=0, mn=0, mn_d=0, wl=0)
    for scene in SCENES:
        rows = by_scene.get(scene, [])
        n = len(rows)
        agree = sum(1 for v, h in rows if v == h)
        fn_d = sum(1 for v, h in rows if h is not None)
        fn = sum(1 for v, h in rows if h is not None and v is None)
        mn_d = sum(1 for v, h in rows if h is None)
        mn = sum(1 for v, h in rows if h is None and v is not None)
        wl = sum(1 for v, h in rows if v is not None and h is not None and v != h)
        print(f"{SCENE_LABELS[scene]:<16} {n:>3} {f'{agree}/{n}':>9} "
              f"{f'{fn}/{fn_d}':>7} {f'{mn}/{mn_d}':>7} {wl:>3}")
        tot["n"] += n; tot["agree"] += agree
        tot["fn"] += fn; tot["fn_d"] += fn_d
        tot["mn"] += mn; tot["mn_d"] += mn_d; tot["wl"] += wl

    print("-" * 50)
    pct = 100 * tot["agree"] / tot["n"] if tot["n"] else 0
    agree_str = f"{tot['agree']}/{tot['n']}"
    fn_str = f"{tot['fn']}/{tot['fn_d']}"
    mn_str = f"{tot['mn']}/{tot['mn_d']}"
    print(f"{'TOTAL':<16} {tot['n']:>3} {agree_str:>9} {fn_str:>7} {mn_str:>7} {tot['wl']:>3}")
    print(f"\nOverall agreement: {agree_str} = {pct:.1f}%")


if __name__ == "__main__":
    main()
