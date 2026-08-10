#!/usr/bin/env python3
"""
Extract which bin/outcome the subject lands in from I2V rollouts,
using Gemini 3.1 Pro via the official Google Gemini API.

Saves the voted outcomes to analysis/{scene}_{model}_bins.json. When you run this
extractor it ALSO writes the raw Gemini replies + per-vote parsed outcomes to
analysis/vlm_logs/{scene}_{model}.json (for auditing / reproducibility; the bins
schema is unchanged).

Requires: GEMINI_API_KEY environment variable
Run with:
  python extract_bin.py --scene galton_board --model wan
  python extract_bin.py --scene dice --model all
  python extract_bin.py --scene all --model happyhorse
"""
import io
import os
import re
import json
import asyncio
import argparse
import cv2
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types

ROLLOUTS_BASE = Path(__file__).parent / "rollouts"
ANALYSIS_DIR = Path(__file__).parent / "analysis"
NUM_ROLLOUTS = 32

GEMINI_MODEL = "gemini-3.1-pro-preview"

# Each scene: vlm_prompt, num_outcomes, outcome_map (None = parse integer directly)
# outcome_map maps lowercase string in response → integer outcome (1-indexed)
SCENES = {
    "galton_board": {
        "vlm_prompt": (
            "Where is the green ball? If there are multiple green balls in the image, "
            "or if the ball appears to straddle two bins or has not clearly settled in "
            "a single bin, reply with 'null'. "
            "Otherwise reply with only the bin number it has landed in as a single integer."
        ),
        "num_outcomes": 11,
        "outcome_map": None,
        "frame_pcts": [1.0],
    },
    "galton_board_real": {
        "vlm_prompt": (
            "Where is the metallic ball? If there are multiple metallic balls in the image, "
            "or if the ball appears to straddle two bins or has not clearly settled in "
            "a single bin, reply with 'null'. "
            "Otherwise reply with only the bin number it has landed in as a single integer."
        ),
        "num_outcomes": 11,
        "outcome_map": None,
        "frame_pcts": [1.0],
    },
    "ball": {
        "vlm_prompt": (
            "You are shown a video of a ball rolling into a Y-shaped fork. "
            "Through which channel does the ball ultimately exit: left or right? "
            "Only reply 'null' if the ball never clearly exits through either channel. "
            "Otherwise reply with only 'left' or 'right'."
        ),
        "num_outcomes": 2,
        "outcome_map": {"left": 1, "right": 2},
        "use_video": True,
    },
    "cards": {
        "vlm_prompt": (
            "You are shown the final frame of a card-draw video. Look at the playing card "
            "and identify its suit.\n\n"
            "Reply 'null' if any of the following apply:\n"
            "  - the card is face-down, partially obscured, or no card is visible;\n"
            "  - the symbols on the card are deformed, blurry, or distorted to the point "
            "where the suit is ambiguous;\n"
            "  - the card shows different suit symbols (e.g.\\ a heart and a club on the "
            "same card) — a real playing card has only one suit, so inconsistent symbols "
            "indicate a rendering artefact;\n"
            "  - the suit symbols are otherwise unrecognisable as one of the four "
            "standard suits.\n\n"
            "Otherwise reply with only 'hearts', 'diamonds', 'clubs', or 'spades'."
        ),
        "num_outcomes": 4,
        "outcome_map": {"hearts": 1, "diamonds": 2, "clubs": 3, "spades": 4},
        "frame_pcts": [1.0],
    },
    "dice": {
        "vlm_prompt": (
            "Is there one clearly dominant top face on the die? "
            "A slight tilt is acceptable as long as one face is clearly uppermost. "
            "If the die is balanced on an edge with no single face dominant, reply with 'null'. "
            "Otherwise reply with only the number on the top face as a single integer between 1 and 6."
        ),
        "num_outcomes": 6,
        "outcome_map": None,
        "frame_pcts": [1.0],
    },
    "lottery": {
        "vlm_prompt": (
            "Is exactly one lottery ball visible inside the clear display tube? "
            "If no ball has been drawn into the tube, or if multiple balls are visible in the tube, "
            "reply with 'null'. "
            "Otherwise reply with only the number shown on that ball as a single integer between 1 and 20."
        ),
        "num_outcomes": 20,
        "outcome_map": None,
        "frame_pcts": [1.0],
    },
    "pendulum": {
        "vlm_prompt": (
            "You are shown a video of a double pendulum. "
            "Does the pendulum maintain the correct rigid structure of a normal double pendulum "
            "(two rods, three balls) throughout the video? "
            "If the structure is deformed, broken, or grows extra arms or balls at any point, "
            "reply with 'null'. "
            "Otherwise, where is the lowest pendulum weight at the end of the video — "
            "to the left or right of the central pivot? "
            "Reply with only 'left', 'right', or 'null'."
        ),
        "num_outcomes": 2,
        "outcome_map": {"left": 1, "right": 2},
        "use_video": True,
    },
    "roulette": {
        "vlm_prompt": (
            "This is the final frame of a roulette video.\n\n"
            "Step 1 — Find the white ball: Can you see a distinct white roulette ball "
            "resting inside the wheel? If the ball is not visible (the frame is blurry, "
            "the ball has disappeared, or you only see the wheel pattern with no ball), "
            "reply 'null'. If you see multiple balls, reply 'null'.\n\n"
            "Step 2 — Check location: Reply 'null' if the ball is: outside the wheel's "
            "outer pocket ring (bounced off); sitting on the central hub or spinner; or "
            "resting on the inner sloped wooden surface between the hub and the pocket ring "
            "(i.e. it has rolled inward and is not inside any numbered pocket).\n\n"
            "Step 3 — Identify colour: Look at the exact pocket the ball's centre sits in. "
            "Pocket colours are red, black, or green. "
            "Important: these wheels are often filmed from above. From an overhead angle the "
            "ball may appear to sit on or near the outer rim when it is actually resting at "
            "the top edge of a numbered pocket — if a numbered pocket is directly below the "
            "ball, the ball is in that pocket. Report that pocket's colour. "
            "If the ball is equally split between two pockets with no dominant side, reply "
            "'null'. Otherwise report the colour of whichever pocket contains most of the "
            "ball.\n\n"
            "Reply with only 'red', 'black', 'green', or 'null'."
        ),
        "num_outcomes": 3,
        "outcome_map": {"red": 1, "black": 2, "green": 3},
        "frame_pcts": [1.0],
    },
    "walking": {
        "vlm_prompt": (
            "You are shown a video of a person approaching a T-junction. "
            "Has the person clearly chosen to walk left or right? "
            "If the person has not yet reached the junction or has not committed to a "
            "direction, reply with 'null'. "
            "Otherwise reply with only 'left' or 'right'."
        ),
        "num_outcomes": 2,
        "outcome_map": {"left": 1, "right": 2},
        "use_video": True,
    },
}

KNOWN_MODELS = ["wan", "seedance", "happyhorse", "veo31", "runway45", "cosmos3_super"]

_client = None


def get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def extract_frames(video_path, pcts):
    """Extract frames at given fractional positions through the video.
    Falls back to last readable frame if a target position is unreadable."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for pct in pcts:
        target = min(int(total * pct), total - 1)
        frame = None
        for offset in range(0, max(20, total)):
            for idx in [target - offset, target + offset]:
                if 0 <= idx < total:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, f = cap.read()
                    if ok:
                        frame = f
                        break
            if frame is not None:
                break
        if frame is not None:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise RuntimeError(f"Could not read any frame from {video_path}")
    return frames


def to_png_bytes(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def _word(tok):
    """Regex matching `tok` only as a whole word (boundaries against [a-z0-9]),
    so 'red' does not match inside 'scored' and 'left' not inside 'left-to-right'."""
    return re.compile(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])")


def _parse_output(text, num_outcomes, outcome_map):
    """Parse a VLM reply into an outcome (1-indexed) or None (null).

    The prompts request a single bare token ('left', '7', 'null', ...), so on a
    well-formed reply this returns exactly what the original first-match logic
    returned. It is hardened against long / chatty replies:
      - tokens match on WORD boundaries (via _word), fixing substring false hits
        such as 'red' in 'scored' or 'left' in 'left-to-right';
      - a reply naming MORE THAN ONE distinct valid outcome is treated as
        ambiguous (-> None) instead of blindly taking the first one.
    A clean single-token reply yields exactly one candidate and is unaffected.
    The committed *_bins.json were produced by the earlier logic and are
    human-validated (VLM-vs-human agreement 93.8%); they are NOT regenerated, so
    paper results are unchanged. On any clean single-token reply (the case the
    prompts elicit) this parser returns the identical outcome to the original.
    """
    t = text.strip().lower()
    if outcome_map is not None:
        hits = {val for key, val in outcome_map.items() if _word(key).search(t)}
        return next(iter(hits)) if len(hits) == 1 else None
    else:
        if _word("null").search(t):
            return None
        nums = {int(m) for m in re.findall(r"\d+", t) if 1 <= int(m) <= num_outcomes}
        return next(iter(nums)) if len(nums) == 1 else None


# --- API-failure handling ------------------------------------------------
# Previously ANY exception in a seed was swallowed and recorded as bin=null,
# so a transient Gemini/transport error masqueraded as a genuine model "null"
# and deflated rho (= n_valid / N). We now (a) retry the API call with
# exponential backoff, and (b) if it still fails, raise ApiError. This FAILS
# LOUDLY: process_model_async writes no bins file for the cell and the whole run
# aborts, so it is impossible to continue until Gemini has successfully scored
# every seed. A response that is successfully returned but parses to None is a
# genuine model null and is unaffected. NOTE: on the original extraction runs
# every seed returned a Gemini response (no API errors), so this fail-loud path
# never fired and the committed bins are unaffected by it.
API_MAX_RETRIES = 4
API_BACKOFF_S = 2.0
API_ERROR = object()  # sentinel: seed had an unrecoverable API error (not a null)


class ApiError(Exception):
    """Unrecoverable Gemini API / transport error after retries."""


async def _generate_with_retry(make_coro):
    """Await make_coro() with exponential backoff on transient API errors.
    Returns the response, or raises ApiError after API_MAX_RETRIES attempts."""
    delay, last = API_BACKOFF_S, None
    for attempt in range(API_MAX_RETRIES):
        try:
            return await make_coro()
        except Exception as e:  # network / rate-limit / server-side errors
            last = e
            if attempt < API_MAX_RETRIES - 1:
                await asyncio.sleep(delay)
                delay *= 2
    raise ApiError(last)


async def ask_gemini_async(pil_images, vlm_prompt, num_outcomes, outcome_map):
    if isinstance(pil_images, Image.Image):
        pil_images = [pil_images]
    parts = [
        types.Part.from_bytes(data=to_png_bytes(img), mime_type="image/png")
        for img in pil_images
    ]
    parts.append(vlm_prompt)
    response = await _generate_with_retry(lambda: get_client().aio.models.generate_content(
        model=GEMINI_MODEL,
        contents=parts,
    ))
    raw = response.text
    return _parse_output(raw, num_outcomes, outcome_map), raw


async def ask_gemini_video_async(video_path, vlm_prompt, num_outcomes, outcome_map):
    client = get_client()
    video_file = await client.aio.files.upload(
        file=str(video_path),
        config=types.UploadFileConfig(mime_type="video/mp4"),
    )
    while video_file.state.name == "PROCESSING":
        await asyncio.sleep(2)
        video_file = await client.aio.files.get(name=video_file.name)
    try:
        response = await _generate_with_retry(lambda: client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_uri(file_uri=video_file.uri, mime_type="video/mp4"),
                vlm_prompt,
            ],
        ))
        raw = response.text
        return _parse_output(raw, num_outcomes, outcome_map), raw
    finally:
        await client.aio.files.delete(name=video_file.name)


async def _query_once(video_path, vlm_prompt, num_outcomes, outcome_map,
                      frame_pcts, frames_dir, seed, use_video=False, save_frame=False):
    if use_video:
        return await ask_gemini_video_async(video_path, vlm_prompt, num_outcomes, outcome_map)
    else:
        frames = extract_frames(video_path, frame_pcts)
        if save_frame:
            frames[-1].save(frames_dir / f"seed_{seed:02d}.png")
        return await ask_gemini_async(frames, vlm_prompt, num_outcomes, outcome_map)


async def _process_seed_async(i, video_path, vlm_prompt, num_outcomes,
                              outcome_map, frame_pcts, frames_dir, use_video=False,
                              n_votes=3):
    try:
        votes, raws = [], []
        for v in range(n_votes):
            outcome, raw = await _query_once(
                video_path, vlm_prompt, num_outcomes, outcome_map,
                frame_pcts, frames_dir, i, use_video=use_video, save_frame=(v == 0),
            )
            votes.append(outcome)
            raws.append(raw)
    except ApiError as e:
        # Unrecoverable API/transport error. Flag the seed; process_model_async
        # aborts the run (fails loudly) so a Gemini outage can never masquerade as
        # a model null, and no partial/degraded bins file is written.
        print(f"  seed {i:02d}: API ERROR after {API_MAX_RETRIES} tries — {e}; "
              f"aborting (nulls must come from the model, not a failed call)")
        return i, API_ERROR, None
    except Exception as e:
        # Genuine per-seed data problem (e.g. unreadable/empty video): the
        # generation itself failed to yield a scorable outcome -> null.
        print(f"  seed {i:02d}: data ERROR — {e}; recorded as null")
        return i, None, {"seed": i, "outcome": None, "votes": [], "raw": [],
                         "error": str(e)}
    if n_votes == 1:
        outcome = votes[0]
    else:
        # Majority vote: most common answer wins; ties go to None (null)
        counts = {}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1
        best_count = max(counts.values())
        winners = [k for k, c in counts.items() if c == best_count]
        outcome = winners[0] if len(winners) == 1 else None
    label = "null" if outcome is None else f"outcome {outcome}"
    print(f"  seed {i:02d}: {label}" + (f"  (votes: {votes})" if n_votes > 1 else ""))
    # Per-seed log: the parsed per-vote outcomes AND the raw VLM replies, so the
    # extraction is fully auditable / reproducible after the fact (the *_bins.json
    # only records the final voted outcome). Written by process_model_async.
    return i, outcome, {"seed": i, "outcome": outcome, "votes": votes, "raw": raws}


async def process_model_async(scene_name, model_name, vlm_prompt, num_outcomes,
                               outcome_map, frame_pcts=None, use_video=False, n_votes=3):
    rollouts_dir = ROLLOUTS_BASE / scene_name / model_name
    if not rollouts_dir.exists():
        print(f"  No rollouts for {scene_name}/{model_name} — skipping")
        return

    if frame_pcts is None:
        frame_pcts = [0.9]

    frames_dir = ANALYSIS_DIR / "frames" / scene_name / model_name
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {scene_name} / {model_name} ===")

    tasks = [
        _process_seed_async(i, rollouts_dir / f"seed_{i}.mp4", vlm_prompt,
                            num_outcomes, outcome_map, frame_pcts, frames_dir,
                            use_video=use_video, n_votes=n_votes)
        for i in range(NUM_ROLLOUTS)
        if (rollouts_dir / f"seed_{i}.mp4").exists()
    ]

    triples = await asyncio.gather(*tasks)
    results = {i: outcome for i, outcome, _ in triples}
    logs = {i: log for i, _, log in triples if log is not None}
    # An unrecoverable API error must FAIL LOUDLY: we do not write a bins file for
    # this cell (so the committed JSON is never clobbered with degraded data) and
    # we abort the run. Extraction cannot proceed until every seed is scored by a
    # successful Gemini response — re-run the cell once the outage clears.
    api_failed = sorted(i for i, o in results.items() if o is API_ERROR)
    if api_failed:
        raise ApiError(
            f"{scene_name}/{model_name}: {len(api_failed)} seed(s) had unrecoverable "
            f"API errors after {API_MAX_RETRIES} retries each: {api_failed}. "
            f"No bins written; fix the outage and re-run — nulls must come from the "
            f"model, never from a failed API call."
        )
    rollouts = [{"seed": i, "bin": results[i]} for i in sorted(results)]

    out_path = ANALYSIS_DIR / f"{scene_name}_{model_name}_bins.json"
    with open(out_path, "w") as f:
        json.dump({
            "scene": scene_name,
            "model": model_name,
            "num_bins": num_outcomes,
            "rollouts": rollouts,
        }, f, indent=2)
    n_valid = sum(1 for r in rollouts if r["bin"] is not None)
    print(f"  {n_valid}/{len(rollouts)} valid — saved → {out_path}")

    # Also persist the raw VLM replies + per-vote parsed outcomes for auditing /
    # reproducibility. Kept in a separate file so the *_bins.json schema (and thus
    # the paper's committed results) is unchanged.
    logs_dir = ANALYSIS_DIR / "vlm_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{scene_name}_{model_name}.json"
    with open(log_path, "w") as f:
        json.dump({
            "scene": scene_name,
            "model": model_name,
            "vlm_model": GEMINI_MODEL,
            "prompt": vlm_prompt,
            "n_votes": n_votes,
            "responses": [logs[i] for i in sorted(logs)],
        }, f, indent=2)
    print(f"  raw VLM logs → {log_path}")


async def main_async(scenes_to_run, models_to_run, n_votes=3):
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    for scene_name in scenes_to_run:
        cfg = SCENES[scene_name]
        await asyncio.gather(*[
            process_model_async(
                scene_name, model_name,
                cfg["vlm_prompt"], cfg["num_outcomes"], cfg["outcome_map"],
                frame_pcts=cfg.get("frame_pcts"),
                use_video=cfg.get("use_video", False),
                n_votes=n_votes,
            )
            for model_name in models_to_run
        ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="galton_board",
                        help=f"Scene: {', '.join(SCENES)} or 'all'")
    parser.add_argument("--model", default="all",
                        help=f"Model: {', '.join(KNOWN_MODELS)} or 'all'")
    parser.add_argument("--votes", type=int, default=3,
                        help="Number of VLM calls per seed for majority voting "
                             "(default 3, matching the benchmark)")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    scenes_to_run = list(SCENES.keys()) if args.scene == "all" else [args.scene]
    for s in scenes_to_run:
        if s not in SCENES:
            raise ValueError(f"Unknown scene '{s}'. Choose from: {', '.join(SCENES)}")

    models_to_run = KNOWN_MODELS if args.model == "all" else [args.model]
    for m in models_to_run:
        if m not in KNOWN_MODELS:
            raise ValueError(f"Unknown model '{m}'. Choose from: {', '.join(KNOWN_MODELS)}")

    asyncio.run(main_async(scenes_to_run, models_to_run, n_votes=args.votes))


if __name__ == "__main__":
    main()
