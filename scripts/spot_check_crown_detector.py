"""Spot-check `perception/crown_detector.py` against real Open Forest Observatory sample frames.

Not a pytest test -- this is a visual sanity check (per CLAUDE.md Phase 1: "visually confirm
outputs look sane"), meant to be rerun by eye whenever the detector's model/thresholds change.
Saves a box-overlay image per sample frame to `outputs/perception_spot_checks/crown_detector/`.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE uv run python scripts/spot_check_crown_detector.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw

from src.perception.crown_detector import CrownDetector

SAMPLE_DIR = Path("data/samples/ofo_mission_000001")
OUTPUT_DIR = Path("outputs/perception_spot_checks/crown_detector")
THUMB_WIDTH = 1000


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detector = CrownDetector()

    for img_path in sorted(SAMPLE_DIR.glob("*.JPG")):
        image = Image.open(img_path).convert("RGB")
        detections = detector.detect(image)

        print(f"{img_path.name}: {len(detections)} detections")
        for d in detections:
            w = d.box_xyxy[2] - d.box_xyxy[0]
            h = d.box_xyxy[3] - d.box_xyxy[1]
            print(f"    score={d.score:.3f} size={w:.0f}x{h:.0f}")

        scale = THUMB_WIDTH / image.size[0]
        thumb = image.resize((THUMB_WIDTH, int(image.size[1] * scale)))
        draw = ImageDraw.Draw(thumb)
        for d in detections:
            box = [v * scale for v in d.box_xyxy]
            draw.rectangle(box, outline="red", width=2)

        out_path = OUTPUT_DIR / f"{img_path.stem}_boxes.jpg"
        thumb.save(out_path)
        print(f"    saved {out_path}")


if __name__ == "__main__":
    main()
