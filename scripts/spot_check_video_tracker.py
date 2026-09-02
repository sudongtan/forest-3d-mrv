"""Spot-check `perception/crown_segmenter.py`'s `CrownVideoTracker` against a real camera-trap
clip -- seeds one point on the first frame and propagates through a short sequence, checking that
the mask stays locked on the same animal rather than drifting to background or another animal.

Not a pytest test -- a visual sanity check (per CLAUDE.md Phase 1), meant to be rerun by eye.
Saves a red-mask overlay for a handful of frames across the tracked sequence to
`outputs/perception_spot_checks/video_tracker/`.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE uv run python scripts/spot_check_video_tracker.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image

from src.perception.crown_segmenter import CrownVideoTracker

VIDEO_PATH = Path("data/samples/camera_trap_lila_seattleish/deer/DSCF0008.AVI")
OUTPUT_DIR = Path("outputs/perception_spot_checks/video_tracker")
NUM_FRAMES = 60
# Picked by eye on frame 0 -- lands on the nearer of the two deer in this clip.
SEED_POINT_XY = (500, 330)
FRAMES_TO_SAVE = (0, 15, 30, 45, 59)


def load_frames(video_path: Path, num_frames: int) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for _ in range(num_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = load_frames(VIDEO_PATH, NUM_FRAMES)
    print(f"loaded {len(frames)} frames from {VIDEO_PATH}, size={frames[0].size}")

    tracker = CrownVideoTracker()
    masks_by_frame = tracker.track(frames, point_xy=SEED_POINT_XY)
    print(f"tracked {len(masks_by_frame)} frames")

    for idx in FRAMES_TO_SAVE:
        if idx not in masks_by_frame:
            continue
        mask = masks_by_frame[idx]
        print(f"  frame {idx}: {mask.sum()} px")

        overlay = np.array(frames[idx]).copy()
        overlay[mask] = (0.4 * overlay[mask] + 0.6 * np.array([255, 0, 0])).astype("uint8")
        out_path = OUTPUT_DIR / f"tracked_{idx:03d}.jpg"
        Image.fromarray(overlay).save(out_path)
        print(f"    saved {out_path}")


if __name__ == "__main__":
    main()
