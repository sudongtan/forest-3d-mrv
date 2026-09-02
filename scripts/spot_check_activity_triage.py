"""Spot-check src/activity/triage.py on real camera-trap footage: runs Phase 1's
CameraTrapDetector on a real clip's frames and confirms animal detections correctly become
wildlife-event candidates, never alerts.

**Real-data coverage note**: a real `vehicle`-labeled LILA BC clip is now included
(camera_trap_lila_seattleish/vehicle/IMG_0096.AVI, found by searching the full bucket listing --
19 real `vehicle` clips exist, 0 `person`/`human` clips do anywhere in this dataset), but it still
produces 0 alerts here -- confirmed by visually inspecting three real `vehicle`-labeled clips
frame-by-frame that none actually show a vehicle in frame (this is an informal/personal dataset;
its `vehicle` label appears to mark trigger events like a distant car on a road, not clips with a
vehicle prominently visible). So the alert-routing path is still only unit-tested
(tests/test_activity_triage.py, using real MegaDetector category strings), not exercised on real
footage that visibly contains what it's meant to catch -- now a documented visual-content gap in
this specific dataset, not an absent-sample gap. See docs/lessons_learnt.md. This script covers
what real footage *can* cover: the animal path (plus, incidentally, a real MegaDetector false
positive on the vehicle clip's background clutter, correctly rejected downstream by SpeciesNet).

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE uv run python scripts/spot_check_activity_triage.py
"""
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.activity.triage import triage
from src.perception.camera_trap_detector import CameraTrapDetector

SAMPLE_DIR = Path("data/samples/camera_trap_lila_seattleish")


def main() -> None:
    detector = CameraTrapDetector()

    for video_path in sorted(SAMPLE_DIR.glob("*/*.AVI")):
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        total_alerts, total_wildlife = 0, 0
        for idx in range(0, max(frame_count, 1), 20):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame_bgr = cap.read()
            if not ok:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detections = detector.detect(frame_rgb)
            alerts, wildlife = triage(detections, timestamp=datetime.now(UTC).isoformat())
            total_alerts += len(alerts)
            total_wildlife += len(wildlife)
        cap.release()

        print(f"{video_path.parent.name}/{video_path.name}: "
              f"{total_wildlife} wildlife candidates, {total_alerts} alerts")


if __name__ == "__main__":
    main()
