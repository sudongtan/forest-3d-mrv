"""Splits Phase 1's MegaDetector/SpeciesNet triage output into scene_state.json's two separate
record types -- `alerts` (person/vehicle) and wildlife-event candidates (animal) -- so human and
vehicle detections never reach species/behavior classification, per CLAUDE.md's Phase 5 item
("human/vehicle detections should populate scene_state.json's alerts, not go through the
species-behavior classifier").

MegaDetector's own 3-class category ("animal"|"person"|"vehicle") already gates SpeciesNet inside
`perception/camera_trap_detector.py` -- species is only populated for "animal" detections, see
that module's `detect()`. This module is the next step: turning that per-detection category into
the two record lists this project's reporting stage (Phase 6's `reporting/scene_state.py`) will
assemble into the final schema.
"""
from dataclasses import dataclass

from src.perception.camera_trap_detector import CameraTrapDetection

ALERT_TYPE_BY_CATEGORY = {"person": "human_intrusion", "vehicle": "vehicle"}


@dataclass(frozen=True)
class Alert:
    type: str  # "human_intrusion" | "vehicle"
    timestamp: str
    confidence: float


@dataclass(frozen=True)
class WildlifeEventCandidate:
    """Not yet a full `scene_state.json` `wildlife_events[]` entry -- `behavior` and `lat`/`lon`
    are populated further downstream: behavior needs `activity/model.py`'s temporal classifier
    run over a multi-frame track (a single triage call only sees one frame), and position needs
    the drone-path's camera-pose/georeferencing machinery (`geometry/georeference.py`), neither
    of which this module has access to. Kept structurally distinct from `Alert` -- a different
    type, not just a different field value -- so the two can never be silently conflated
    downstream.
    """
    species: str | None
    species_confidence: float | None
    timestamp: str
    detection_confidence: float


def triage(
    detections: list[CameraTrapDetection], timestamp: str
) -> tuple[list[Alert], list[WildlifeEventCandidate]]:
    alerts: list[Alert] = []
    wildlife_candidates: list[WildlifeEventCandidate] = []
    for d in detections:
        if d.category in ALERT_TYPE_BY_CATEGORY:
            alerts.append(Alert(
                type=ALERT_TYPE_BY_CATEGORY[d.category], timestamp=timestamp, confidence=d.confidence,
            ))
        elif d.category == "animal":
            wildlife_candidates.append(WildlifeEventCandidate(
                species=d.species, species_confidence=d.species_confidence,
                timestamp=timestamp, detection_confidence=d.confidence,
            ))
        else:
            raise ValueError(f"unrecognized MegaDetector category {d.category!r} -- "
                              "CameraTrapDetection.category should only ever be animal/person/vehicle")
    return alerts, wildlife_candidates
