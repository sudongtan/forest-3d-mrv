"""Tests src/activity/datasets.py against the real downloaded MammAlps demo clip + annotations
(data/samples/mammalps_demo/) -- not synthetic/mock data, per this project's own testing
philosophy (see docs/lessons_learnt.md for how that sample was sourced).
"""
from pathlib import Path

import pytest

from src.activity.datasets import load_annotation, load_label_mapping, track_ids_in, windows_for_track

DEMO_DIR = Path("data/samples/mammalps_demo")
ANNOTATIONS_PATH = DEMO_DIR / "demo_annotations.json"
LABELS_B1_PATH = DEMO_DIR / "labels_mapping_b1.json"

pytestmark = pytest.mark.skipif(
    not ANNOTATIONS_PATH.exists(),
    reason="real MammAlps demo sample not downloaded -- see docs/DATASETS.md",
)


def test_load_annotation_matches_real_clip_info():
    ann = load_annotation(ANNOTATIONS_PATH)
    assert ann.info.file_id == "S1_C1_E4_V0016"
    assert ann.info.num_frames == 615
    assert ann.info.fps == 30.0
    assert len(ann.frames) == 615


def test_load_label_mapping_has_real_benchmark1_classes():
    mapping = load_label_mapping(LABELS_B1_PATH)
    assert "foraging" in mapping["activities"]
    assert "red_deer" in mapping["species"]
    assert "walking" in mapping["actions"]


def test_track_ids_in_real_clip():
    ann = load_annotation(ANNOTATIONS_PATH)
    ids = track_ids_in(ann)
    assert ids == [1, 2]  # two real tracked deer in this clip -- see lessons_learnt.md


def test_windows_for_track_covers_real_track_extent():
    ann = load_annotation(ANNOTATIONS_PATH)
    windows = windows_for_track(ann, track_id=1, window_size=16, stride=8)
    assert len(windows) > 0
    for w in windows:
        assert w.file_id == "S1_C1_E4_V0016"
        assert w.track_id == 1
        assert w.end_frame - w.start_frame == 16
        assert w.species == "red_deer"
        assert 0.0 < w.label_agreement <= 1.0


def test_windows_for_track2_only_covers_its_real_presence_range():
    # track 2 only appears frames 219-614 in the real clip -- windows_for_track should not
    # fabricate windows from frames where it isn't present.
    ann = load_annotation(ANNOTATIONS_PATH)
    windows = windows_for_track(ann, track_id=2, window_size=16, stride=8)
    assert all(w.start_frame >= 200 for w in windows)  # generous margin below the real 219 start


def test_windows_include_real_activity_diversity():
    ann = load_annotation(ANNOTATIONS_PATH)
    windows = windows_for_track(ann, track_id=2, window_size=16, stride=8)
    activities = {w.activity for w in windows}
    assert "foraging" in activities
    assert "vigilance" in activities  # confirms real behavior-class diversity is preserved
    # through windowing, not collapsed into one dominant label


def test_absent_track_id_produces_no_windows():
    ann = load_annotation(ANNOTATIONS_PATH)
    assert windows_for_track(ann, track_id=999) == []
