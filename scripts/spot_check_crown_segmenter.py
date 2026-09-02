"""Spot-check `perception/crown_segmenter.py`'s per-frame segmentation against real Open Forest
Observatory sample frames -- chains crown_detector.py's real output straight into SAM2, mirroring
actual pipeline usage rather than testing the segmenter in isolation with hand-made boxes.

Not a pytest test -- a visual sanity check (per CLAUDE.md Phase 1: "visually confirm outputs look
sane"), meant to be rerun by eye whenever the segmenter's model changes. Saves a mask-overlay
image per sample frame to `outputs/perception_spot_checks/crown_segmenter/`.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE uv run python scripts/spot_check_crown_segmenter.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from src.perception.crown_detector import CrownDetector
from src.perception.crown_segmenter import CrownSegmenter

SAMPLE_DIR = Path("data/samples/ofo_mission_000001")
OUTPUT_DIR = Path("outputs/perception_spot_checks/crown_segmenter")
THUMB_WIDTH = 1000
RANDOM_SEED = 0  # only for overlay colors, not model behavior


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detector = CrownDetector()
    segmenter = CrownSegmenter()
    rng = np.random.default_rng(RANDOM_SEED)

    for img_path in sorted(SAMPLE_DIR.glob("*.JPG")):
        image = Image.open(img_path).convert("RGB")
        detections = detector.detect(image)
        masks = segmenter.segment(image, detections)

        print(f"{img_path.name}: {len(masks)} masks")
        for m in masks:
            print(f"    score={m.score:.3f} pixel_count={m.mask.sum()}")

        overlay = np.array(image).copy()
        colors = rng.integers(50, 255, size=(len(masks), 3))
        for m, color in zip(masks, colors):
            overlay[m.mask] = (0.5 * overlay[m.mask] + 0.5 * color).astype("uint8")

        thumb = Image.fromarray(overlay)
        thumb.thumbnail((THUMB_WIDTH, THUMB_WIDTH))
        out_path = OUTPUT_DIR / f"{img_path.stem}_masks.jpg"
        thumb.save(out_path)
        print(f"    saved {out_path}")


if __name__ == "__main__":
    main()
