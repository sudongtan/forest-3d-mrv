"""Tests src/activity/triage.py's routing logic. Uses manually constructed CameraTrapDetection
instances with the real category strings MegaDetector actually produces ("animal"/"person"/
"vehicle", confirmed in perception/camera_trap_detector.py) -- this is a pure post-processing/
routing function, so it doesn't need a real detector run the way perception wrappers do; the
animal-category path is additionally exercised on real footage in
scripts/spot_check_camera_trap_detector.py (see docs/lessons_learnt.md for why no real
person/vehicle sample clip was available to exercise that path on real footage too).
"""
from src.activity.triage import Alert, WildlifeEventCandidate, triage
from src.perception.camera_trap_detector import CameraTrapDetection


def _det(category, confidence=0.9, species=None, species_confidence=None):
    return CameraTrapDetection(
        box_xyxy_norm=(0.1, 0.1, 0.5, 0.5), category=category, confidence=confidence,
        species=species, species_confidence=species_confidence,
    )


def test_person_detection_becomes_human_intrusion_alert():
    alerts, wildlife = triage([_det("person", confidence=0.87)], timestamp="2026-09-02T10:00:00")
    assert alerts == [Alert(type="human_intrusion", timestamp="2026-09-02T10:00:00", confidence=0.87)]
    assert wildlife == []


def test_vehicle_detection_becomes_vehicle_alert():
    alerts, wildlife = triage([_det("vehicle", confidence=0.75)], timestamp="2026-09-02T10:00:00")
    assert alerts == [Alert(type="vehicle", timestamp="2026-09-02T10:00:00", confidence=0.75)]
    assert wildlife == []


def test_animal_detection_becomes_wildlife_candidate_not_alert():
    alerts, wildlife = triage(
        [_det("animal", confidence=0.9, species="coyote", species_confidence=0.8)],
        timestamp="2026-09-02T10:00:00",
    )
    assert alerts == []
    assert wildlife == [WildlifeEventCandidate(
        species="coyote", species_confidence=0.8, timestamp="2026-09-02T10:00:00",
        detection_confidence=0.9,
    )]


def test_mixed_frame_splits_correctly_and_never_cross_contaminates():
    detections = [
        _det("person", confidence=0.6),
        _det("animal", confidence=0.9, species="deer", species_confidence=0.7),
        _det("vehicle", confidence=0.5),
    ]
    alerts, wildlife = triage(detections, timestamp="t")
    assert len(alerts) == 2
    assert len(wildlife) == 1
    assert {a.type for a in alerts} == {"human_intrusion", "vehicle"}
    assert wildlife[0].species == "deer"


def test_unrecognized_category_raises_rather_than_silently_dropping():
    import pytest
    with pytest.raises(ValueError):
        triage([_det("unknown_category")], timestamp="t")
