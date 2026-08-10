#!/usr/bin/env python3
"""
Build a 2 x 4 figure showing frames from two physical Galton-board rollouts:
 - Row 1: a scoreable rollout (HappyHorse, seed 0; lands in bin 6)
 - Row 2: an invalid multi-ball rollout (Veo 3.1, seed 29; unscoreable)
Each row samples frames at the timestamps in TIMESTAMPS.
"""
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

ROOT = Path(__file__).parent
ROLLOUTS = ROOT / "rollouts" / "galton_board_real"
OUT_FRAME_DIR = ROOT / "analysis" / "rollout_example_frames"
OUT_FRAME_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = ROOT / "analysis" / "rollout_examples.pdf"

ROWS = [
    {
        "label": "Scoreable",
        "subtitle": "HappyHorse, seed 0\n(lands in bin 6)",
        "video": ROLLOUTS / "happyhorse" / "seed_0.mp4",
        "tag": "scoreable",
    },
    {
        "label": "Invalid",
        "subtitle": "Veo 3.1, seed 29\n(multiple balls; unscoreable)",
        "video": ROLLOUTS / "veo31" / "seed_29.mp4",
        "tag": "invalid",
    },
]

# 4 absolute timestamps in seconds. The shortest video in ROWS is the cap:
# HappyHorse seed 0 is ~3.16 s, so the final sample at 2.9 s is safely inside.
TIMESTAMPS = [0.0, 1.0, 2.0, 2.9]


def _crop_borders(img, tol=20):
    """Trim consecutive near-uniform rows from the top and bottom edges.

    Catches both pure-white whitespace margins and letterbox bands of any
    colour (Veo 3.1 frames, for example, have a 2-row pure-black letterbox
    immediately preceded by a uniform grey row — all three need to go so
    no horizontal band shows up below the apparatus in the assembled figure).
    """
    a = img if img.dtype != np.float32 else (img * 255).astype(np.uint8)
    if a.ndim == 3:
        rows = a[..., :3].reshape(a.shape[0], -1)
    else:
        rows = a.reshape(a.shape[0], -1)
    uniform = (rows.max(axis=1) - rows.min(axis=1)) < tol
    n = len(uniform)
    top = 0
    while top < n and uniform[top]:
        top += 1
    bot = n
    while bot > top and uniform[bot - 1]:
        bot -= 1
    return img[top:bot]


def extract_frame(video_path, t, out_path):
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{t:.2f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-vf", "scale=720:-1",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    if not Path(out_path).exists():
        raise RuntimeError(f"ffmpeg did not produce frame at t={t:.2f} for {video_path}")


def main():
    n_cols = len(TIMESTAMPS)
    n_rows = len(ROWS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 1.55 * n_rows))

    for r, row in enumerate(ROWS):
        for c, t in enumerate(TIMESTAMPS):
            frame_path = OUT_FRAME_DIR / f"{row['tag']}_t{int(t*10):02d}.png"
            extract_frame(row["video"], t, frame_path)
            ax = axes[r, c]
            img = mpimg.imread(frame_path)
            # Strip any near-uniform black or white border rows so different
            # source videos (which have varying letterbox heights) line up
            # and no spurious horizontal band appears in the assembled figure.
            img = _crop_borders(img)
            ax.imshow(img)
            ax.axis("off")
            if r == 0:
                ax.set_title(f"$t = {t:.1f}\\,$s", fontsize=11)
            if c == 0:
                # ax.text instead of set_ylabel because axis("off") hides ylabel
                ax.text(-0.05, 0.5,
                        f"{row['label']}\n{row['subtitle']}",
                        transform=ax.transAxes,
                        fontsize=10, ha="right", va="center")

    fig.subplots_adjust(left=0.13, right=0.99, top=0.92, bottom=0.02,
                        wspace=0.05, hspace=0.04)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {OUT_PDF}")


if __name__ == "__main__":
    main()
