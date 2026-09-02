"""Assembles `scene_state.json` -- the contract between the geometry/biomass/activity stages and
the reporting stage (see CLAUDE.md). Every upstream module's real output gets turned into this
one structured record here; `reporting/vlm_report.py` is never allowed to read anything else.

**Nullability decision, stated explicitly rather than left as an unexplained inconsistency**:
CLAUDE.md's guardrail says "if a pipeline stage can't populate its part of this schema, that's a
sign the stage needs more work -- not a reason to loosen the schema." Followed literally for
`trees[].lat/lon`: Phase 2/3's drone pipeline *does* real georeferencing (camera-GPS-anchored,
see `geometry/georeference.py`), so those stay required, non-null floats -- no loosening. For
`wildlife_events[].lat/lon`, though: a real production camera-trap deployment has a fixed,
surveyed station GPS, so this isn't a pipeline capability gap the same way an unbuilt stage would
be -- it's that this project's actual real camera-trap sample (the MammAlps demo clip, see
`activity/datasets.py`) doesn't ship site GPS in its metadata. Modeled as nullable here, with the
reasoning kept next to the schema rather than silently assumed, so a future real deployment with
real station coordinates has nothing to change.
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jsonschema

SCENE_STATE_JSON_SCHEMA = {
    "type": "object",
    "required": ["plot_id", "capture_date", "crs", "trees", "canopy_change", "wildlife_events", "alerts"],
    "additionalProperties": False,
    "properties": {
        "plot_id": {"type": "string"},
        "capture_date": {"type": "string"},
        "crs": {"type": "string"},
        "trees": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["tree_id", "lat", "lon", "height_m", "crown_diameter_m",
                             "biomass_kg", "co2e_kg", "height_source", "tag_text"],
                "additionalProperties": False,
                "properties": {
                    "tree_id": {"type": "string"},
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "height_m": {"type": "number"},
                    "crown_diameter_m": {"type": "number"},
                    "biomass_kg": {"type": "number"},
                    "co2e_kg": {"type": "number"},
                    "height_source": {"type": "string", "enum": ["sfm", "depth_fallback"]},
                    "tag_text": {"type": ["string", "null"]},
                },
            },
        },
        "canopy_change": {
            "type": "object",
            "required": ["prior_survey_id", "loss_pct"],
            "additionalProperties": False,
            "properties": {
                "prior_survey_id": {"type": ["string", "null"]},
                "loss_pct": {"type": "number"},
            },
        },
        "wildlife_events": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["timestamp", "species", "behavior", "confidence", "lat", "lon"],
                "additionalProperties": False,
                "properties": {
                    "timestamp": {"type": "string"},
                    "species": {"type": "string"},
                    "behavior": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "lat": {"type": ["number", "null"]},  # see module docstring
                    "lon": {"type": ["number", "null"]},
                },
            },
        },
        "alerts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type", "timestamp", "confidence"],
                "additionalProperties": False,
                "properties": {
                    "type": {"type": "string", "enum": ["human_intrusion", "vehicle"]},
                    "timestamp": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
        },
    },
}


@dataclass
class Tree:
    tree_id: str
    lat: float
    lon: float
    height_m: float
    crown_diameter_m: float
    biomass_kg: float
    co2e_kg: float
    height_source: str  # "sfm" | "depth_fallback"
    tag_text: str | None = None


@dataclass
class CanopyChange:
    prior_survey_id: str | None = None
    loss_pct: float = 0.0  # 0.0 means "no prior survey to compare against", not "confirmed zero
    # loss" -- reporting/vlm_report.py must not present this as a measured result when
    # prior_survey_id is null; see that module's prompt-construction logic.


@dataclass
class WildlifeEvent:
    timestamp: str
    species: str
    behavior: str
    confidence: float
    lat: float | None = None
    lon: float | None = None


@dataclass
class Alert:
    type: str  # "human_intrusion" | "vehicle"
    timestamp: str
    confidence: float


@dataclass
class SceneState:
    plot_id: str
    capture_date: str
    crs: str
    trees: list[Tree] = field(default_factory=list)
    canopy_change: CanopyChange = field(default_factory=CanopyChange)
    wildlife_events: list[WildlifeEvent] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: Path | str | None = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            Path(path).write_text(text)
        return text


def validate_scene_state(data: dict) -> None:
    """Raises `jsonschema.exceptions.ValidationError` if `data` doesn't match the real
    scene_state.json contract from CLAUDE.md. Call this on every assembled SceneState before it
    reaches `vlm_report.py` -- per CLAUDE.md's "How to Work Through This" guidance, schema drift
    should be caught here, not discovered downstream in the report.
    """
    jsonschema.validate(instance=data, schema=SCENE_STATE_JSON_SCHEMA)
