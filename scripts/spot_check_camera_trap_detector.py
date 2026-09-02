"""Spot-check `perception/camera_trap_detector.py` against real camera-trap sample clips
(coyote/deer/heron/empty -- see docs/DATASETS.md for why LILA BC, not MammAlps, is used here).

Not a pytest test -- a visual sanity check (per CLAUDE.md Phase 1), meant to be rerun by eye.
Saves a box+label overlay for the first frame of each clip to
`outputs/perception_spot_checks/camera_trap_detector/`.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE uv run python scripts/spot_check_camera_trap_detector.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from PIL import Image, ImageDraw

from src.perception.camera_trap_detector import CameraTrapDetector

SAMPLE_DIR = Path("data/samples/camera_trap_lila_seattleish")
OUTPUT_DIR = Path("outputs/perception_spot_checks/camera_trap_detector")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detector = CameraTrapDetector()

    for video_path in sorted(SAMPLE_DIR.glob("*/*.AVI")):
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Real trail-cam clips have a lead-in before the animal walks into frame (confirmed on
        # the coyote sample: frame 0 has nothing, the coyote only appears from frame ~60 on) --
        # frame 0 alone is not a representative spot-check. Scan a spread of frames and keep
        # whichever has the most detections. A clip with genuinely zero detections everywhere
        # (the "empty" sample) is a real, correct outcome, not a read failure -- track "was any
        # frame readable at all" separately from "did any frame have a detection" so the two
        # don't get conflated.
        any_frame_read = False
        best_frame_rgb, best_detections = None, []
        for idx in range(0, max(frame_count, 1), 20):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame_bgr = cap.read()
            if not ok:
                continue
            any_frame_read = True
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detections = detector.detect(frame_rgb)
            if best_frame_rgb is None or len(detections) > len(best_detections):
                best_frame_rgb, best_detections = frame_rgb, detections
        cap.release()

        if not any_frame_read:
            print(f"{video_path}: could not read any frame, skipping")
            continue
        frame_rgb, detections = best_frame_rgb, best_detections
        print(f"{video_path.parent.name}/{video_path.name}: {len(detections)} detections")
        for d in detections:
            species_str = f" species={d.species!r} ({d.species_confidence:.2f})" if d.species else ""
            print(f"    {d.category} conf={d.confidence:.2f}{species_str}")

        h, w = frame_rgb.shape[:2]
        image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(image)
        for d in detections:
            x1, y1, x2, y2 = d.box_xyxy_norm
            box = [x1 * w, y1 * h, x2 * w, y2 * h]
            label = d.species or d.category
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], max(0, box[1] - 14)), f"{label} {d.confidence:.2f}", fill="red")

        out_path = OUTPUT_DIR / f"{video_path.parent.name}_{video_path.stem}.jpg"
        image.save(out_path)
        print(f"    saved {out_path}")


if __name__ == "__main__":
    main()
