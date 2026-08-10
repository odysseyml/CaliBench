#!/usr/bin/env python3
"""Canonical scene definitions (conditioning frame + generation prompt).

Single source of truth shared by both generators — generate_rollouts.py (the
Replicate models) and cosmos/cosmos3_super_batch.py (Cosmos3-Super) — so the
per-scene frame and prompt cannot drift between them. Dependency-free.

Each entry: {"frame": <stem in conditioning_frames/>, "prompt": <V2 prompt>}.
"""

SCENES = {
    "galton_board": {
        "frame": "galton_board",
        "prompt": (
            "A clean 2D physics animation of a Galton board: a vertical triangular "
            "peg array above a row of collection bins. A single green ball is "
            "released from the top funnel and falls through the pegs, bouncing "
            "left or right at each, until it lands in one of the bins. The "
            "animation is smooth and physically accurate."
        ),
    },
    "galton_board_real": {
        "frame": "galton_board_final",
        "prompt": (
            "A clean physics animation of a Galton board: a vertical triangular "
            "peg array above a row of collection bins. A single metallic ball is "
            "released from the top funnel and falls through the pegs, bouncing "
            "left or right at each, until it lands in one of the bins. The "
            "animation is smooth and physically accurate."
        ),
    },
    "ball": {
        "frame": "ball",
        "prompt": (
            "A clean 2D physics simulation. A single ball rolls down a straight ramp and enters "
            "the top of a perfectly symmetric Y-shaped fork. The fork splits into two equal channels. "
            "The ball travels through one of the two channels and exits at the bottom. "
            "The simulation is smooth and physically accurate, viewed from the front."
        ),
    },
    "cards": {
        "frame": "cards",
        "prompt": (
            "A single playing card is drawn from the top of the face-down deck and flipped "
            "face-up onto the table. The card is shown clearly face-up in the final frame."
        ),
    },
    "dice": {
        "frame": "dice",
        "prompt": (
            "A clean 2D physics animation of a single six-sided die being rolled on a flat surface "
            "inside a wooden box with low walls. The die starts tumbling from the left side and comes "
            "to rest showing a single upward face clearly. The animation is slow, smooth, and "
            "physically accurate. The die is white with large black circular pips, viewed from directly above."
        ),
    },
    "lottery": {
        "frame": "lottery",
        "prompt": (
            "A physics animation of a transparent lottery ball tumbler containing 20 numbered white "
            "balls (1-20). The machine spins and one ball is ejected into a clear tube at the top. "
            "The animation ends with a single ball clearly visible in the tube, its number legible. "
            "Smooth, realistic motion."
        ),
    },
    "pendulum": {
        "frame": "pendulum_v3",
        "prompt": (
            "A clean physics simulation of a double pendulum. Two thin brass-coloured rods "
            "connect three points: a fixed pivot at the top of the post, a brass disk at the "
            "middle joint, and a brass disk at the lower tip. The pendulum is released from rest "
            "and swings freely under gravity. The camera is fixed and does not move."
        ),
    },
    "roulette": {
        "frame": "roulette_v3",
        "prompt": (
            "A clean physics animation of a European roulette wheel spinning rapidly. A single white "
            "ball is released onto the spinning wheel and eventually settles into one of the numbered "
            "pockets. The final frame clearly shows the ball at rest with the pocket number and colour "
            "clearly visible. The animation is smooth, realistic, and viewed from directly above."
        ),
    },
    "walking": {
        "frame": "walking",
        "prompt": (
            "A video of a person walking down a long empty corridor who reaches a perfectly symmetric "
            "T-junction. The left and right paths are identical in appearance, lighting, and length. "
            "The person chooses one direction and walks off screen. "
            "The scene is shot from behind at ground level."
        ),
    },
}
