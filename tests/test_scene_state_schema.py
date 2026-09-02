"""Validates src/reporting/scene_state.py's SceneState/to_dict output against the real
scene_state.json contract defined in CLAUDE.md -- catches schema drift between what upstream
modules produce and what the reporting stage expects, per CLAUDE.md's own guidance to write this
test early and run it after every stage that writes to the record.
"""
import jsonschema
import pytest

from src.reporting.scene_state import Alert, CanopyChange, SceneState, Tree, WildlifeEvent, validate_scene_state


def _minimal_valid_scene_state() -> SceneState:
    return SceneState(
        plot_id="ofo_mission_000001",
        capture_date="2024-06-15",
        crs="EPSG:6339",
        trees=[Tree(
            tree_id="tree_0", lat=39.198, lon=-120.904, height_m=10.5, crown_diameter_m=4.2,
            biomass_kg=250.0, co2e_kg=430.0, height_source="sfm", tag_text=None,
        )],
        canopy_change=CanopyChange(prior_survey_id=None, loss_pct=0.0),
        wildlife_events=[WildlifeEvent(
            timestamp="2026-06-01T10:00:00", species="red_deer", behavior="foraging",
            confidence=0.9, lat=None, lon=None,
        )],
        alerts=[Alert(type="human_intrusion", timestamp="2026-06-01T10:05:00", confidence=0.8)],
    )


def test_minimal_valid_scene_state_passes_schema():
    validate_scene_state(_minimal_valid_scene_state().to_dict())


def test_empty_scene_state_passes_schema():
    empty = SceneState(plot_id="p", capture_date="2026-01-01", crs="EPSG:4326")
    validate_scene_state(empty.to_dict())


def test_missing_required_field_fails_schema():
    data = _minimal_valid_scene_state().to_dict()
    del data["crs"]
    with pytest.raises(jsonschema.exceptions.ValidationError):
        validate_scene_state(data)


def test_tree_lat_lon_are_required_not_nullable():
    data = _minimal_valid_scene_state().to_dict()
    data["trees"][0]["lat"] = None
    with pytest.raises(jsonschema.exceptions.ValidationError):
        validate_scene_state(data)


def test_wildlife_event_lat_lon_may_be_null():
    # see scene_state.py's module docstring for why this differs from trees[].lat/lon
    data = _minimal_valid_scene_state().to_dict()
    assert data["wildlife_events"][0]["lat"] is None
    validate_scene_state(data)  # must not raise


def test_invalid_alert_type_fails_schema():
    data = _minimal_valid_scene_state().to_dict()
    data["alerts"][0]["type"] = "not_a_real_type"
    with pytest.raises(jsonschema.exceptions.ValidationError):
        validate_scene_state(data)


def test_invalid_height_source_fails_schema():
    data = _minimal_valid_scene_state().to_dict()
    data["trees"][0]["height_source"] = "made_up_source"
    with pytest.raises(jsonschema.exceptions.ValidationError):
        validate_scene_state(data)


def test_extra_unexpected_field_fails_schema():
    # additionalProperties: False -- a silently-added field should be caught, not ignored
    data = _minimal_valid_scene_state().to_dict()
    data["unexpected_field"] = "should not be here"
    with pytest.raises(jsonschema.exceptions.ValidationError):
        validate_scene_state(data)


def test_confidence_out_of_range_fails_schema():
    data = _minimal_valid_scene_state().to_dict()
    data["alerts"][0]["confidence"] = 1.5
    with pytest.raises(jsonschema.exceptions.ValidationError):
        validate_scene_state(data)
