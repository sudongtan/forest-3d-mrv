"""Real sanity-check tests for src/reporting/vlm_report.py, per CLAUDE.md's Phase 6 checklist:
"deliberately feed a scene_state.json with a known intrusion alert and confirm the generated
report surfaces it prominently, not buried." Runs the real local model (Qwen2.5-1.5B-Instruct,
greedy decoding -- deterministic given a fixed model+prompt, not a coin flip) -- these are real
generations, not mocked, so they're slower than the rest of the suite but exercise the actual
failure mode this check exists to catch.

**These tests exist because the first real run of this exact check found a real bug** (see
vlm_report.py's module docstring): the alert was present in the narrative but buried in
paragraph 3, and a null `prior_survey_id` got misreported as a measured 0% canopy loss. Both are
now fixed with deterministic (non-LLM-dependent) guardrails; these tests lock that in.

**A second, more serious real bug**, found running this on the real 13-tree Phase 2/3 plot (not
a toy example): the model stated total CO2e as "2,007 metric tons" against a real value of 4,896.4
kg (~4.9 metric tons) -- off by ~410x -- and a "mean height" that was actually just one tree's
individual height. `_compute_tree_summary`/`_compute_wildlife_summary` fix this the same way:
precompute every real aggregate in code, never let the model do array arithmetic. Tested here
against the same real plot data, not a synthetic replay.
"""
import json
import re

import pytest

from src.reporting.scene_state import Alert, CanopyChange, SceneState, Tree
from src.reporting.vlm_report import (
    VLMReportGenerator,
    _build_alert_banner,
    _canopy_change_for_prompt,
    _compute_tree_summary,
    _compute_wildlife_summary,
    _session_type_label,
    _strip_stale_title_line,
    _strip_unsupported_area_claims,
)


@pytest.fixture(scope="module")
def generator():
    return VLMReportGenerator()


def _one_tree_scene(**overrides) -> SceneState:
    defaults = dict(
        plot_id="test_plot", capture_date="2026-06-15", crs="EPSG:6339",
        trees=[Tree(tree_id="tree_0", lat=39.2, lon=-120.9, height_m=10.0, crown_diameter_m=3.0,
                     biomass_kg=500.0, co2e_kg=860.0, height_source="sfm")],
    )
    defaults.update(overrides)
    return SceneState(**defaults)


def test_alert_banner_is_deterministic_not_llm_generated():
    # the core guarantee: prominence comes from code, not from the model choosing to comply
    alerts = [{"type": "human_intrusion", "timestamp": "2026-06-15T14:03:00", "confidence": 0.86}]
    banner = _build_alert_banner(alerts)
    assert banner == "ALERT: human_intrusion detected at 2026-06-15T14:03:00 (confidence 86%)."


def test_no_alerts_produces_empty_banner():
    assert _build_alert_banner([]) == ""


def test_canopy_change_rewritten_when_no_prior_survey():
    assert _canopy_change_for_prompt({"prior_survey_id": None, "loss_pct": 0.0}) == \
        "no prior survey available for comparison"


def test_canopy_change_passed_through_when_prior_survey_exists():
    cc = {"prior_survey_id": "survey_2024", "loss_pct": 3.2}
    assert _canopy_change_for_prompt(cc) == cc


def test_real_generation_surfaces_alert_as_first_line(generator):
    scene = _one_tree_scene(alerts=[
        Alert(type="human_intrusion", timestamp="2026-06-15T14:03:00", confidence=0.86)
    ])
    report = generator.generate(scene, max_new_tokens=256)
    first_line = report.strip().splitlines()[0]
    assert first_line.startswith("ALERT: human_intrusion")
    assert "14:03:00" in first_line


def test_real_generation_omits_alert_banner_when_no_alerts(generator):
    # the deterministic banner (the mechanism this test suite actually relies on for prominence)
    # must be absent -- the model incidentally *stating* "no intrusions occurred" in its own
    # narrative is harmless and not what this check is for (found empirically: a real run said
    # exactly that, in the negative, which is fine -- it's not misleading, unlike burying a real
    # alert would be).
    scene = _one_tree_scene(alerts=[])
    report = generator.generate(scene, max_new_tokens=256)
    assert not report.strip().startswith("ALERT:")
    assert "ALERT: human_intrusion" not in report
    assert "ALERT: vehicle" not in report


def test_real_generation_does_not_fabricate_measured_loss_when_no_prior_survey(generator):
    # the meaningful invariant: no fabricated percentage figure -- not exact wording, which a
    # real run showed varies ("no prior survey available for comparison" vs. "due to the lack of
    # prior data for comparison" both correctly avoid a number; only a fabricated percentage
    # would be the actual harm CLAUDE.md's guardrail cares about here).
    scene = _one_tree_scene(canopy_change=CanopyChange(prior_survey_id=None, loss_pct=0.0))
    report = generator.generate(scene, max_new_tokens=256)
    assert re.search(r"\d+(\.\d+)?\s*%", report) is None, (
        f"report stated a percentage despite no prior survey existing: {report!r}"
    )


# The real 13-tree Phase 2/3 drone plot output (scripts/spot_check_scene_state_report.py) that
# first surfaced the ~410x CO2e hallucination -- embedded here (not read from outputs/, which is
# gitignored and not guaranteed to exist) so this regression test is self-contained.
_REAL_13_TREE_HEIGHTS = [9.57, 10.67, 1.57, 2.53, 0.39, 2.35, 1.41, 4.32, 4.58, 0.51, 1.88, 2.03, 0.37]
_REAL_13_TREE_BIOMASS = [753.7, 934.8, 50.4, 116.5, 4.1, 108.1, 51.5, 310.4, 351.6, 6.6, 70.6, 80.7, 4.1]
_REAL_13_TREE_CO2E = [1298.0, 1609.8, 86.9, 200.7, 7.0, 186.2, 88.7, 534.6, 605.6, 11.3, 121.5, 139.0, 7.1]


def _real_13_tree_scene() -> SceneState:
    trees = [
        Tree(tree_id=f"tree_{i}", lat=39.2, lon=-120.9, height_m=h, crown_diameter_m=15.0,
             biomass_kg=b, co2e_kg=c, height_source="sfm")
        for i, (h, b, c) in enumerate(zip(_REAL_13_TREE_HEIGHTS, _REAL_13_TREE_BIOMASS, _REAL_13_TREE_CO2E))
    ]
    return SceneState(plot_id="ofo_mission_000001", capture_date="2023-05-27", crs="EPSG:6339", trees=trees)


def test_compute_tree_summary_matches_real_plot_totals():
    data = json.loads(_real_13_tree_scene().to_json())
    summary = _compute_tree_summary(data["trees"])
    assert summary["num_trees"] == 13
    assert summary["height_m_min"] == 0.37
    assert summary["height_m_max"] == 10.67
    assert summary["height_m_mean"] == pytest.approx(3.24, abs=0.01)
    assert summary["total_biomass_kg"] == pytest.approx(2843.1, abs=0.1)
    assert summary["total_co2e_kg"] == pytest.approx(4896.4, abs=0.1)


def test_compute_tree_summary_empty_is_none():
    assert _compute_tree_summary([]) is None


def test_session_type_label_from_real_shapes():
    assert _session_type_label({"trees": [{"x": 1}], "wildlife_events": []}) == "drone survey plot"
    assert _session_type_label({"trees": [], "wildlife_events": [{"x": 1}]}) == "camera-trap session"
    assert _session_type_label({"trees": [], "wildlife_events": []}) == \
        "survey with no trees or wildlife events recorded"


def test_compute_wildlife_summary_counts_by_species():
    events = [{"species": "red_deer"}, {"species": "red_deer"}, {"species": "roe_deer"}]
    summary = _compute_wildlife_summary(events)
    assert summary["num_events"] == 3
    assert summary["event_count_by_species"] == {"red_deer": 2, "roe_deer": 1}


# Real, saved reports from an actual end-to-end run (scripts/spot_check_scene_state_report.py,
# outputs/perception_spot_checks/scene_state_report/) that first surfaced these two bugs --
# embedded verbatim (not read from outputs/, which is gitignored) so these regression tests don't
# depend on a real generation to reproduce a specific real failure string.
_REAL_STALE_TITLE_REPORT = (
    "**Drone Survey Plot Summary**\n\n"
    "On **2026-01-01**, a camera-trap session recorded 122 instances of red deer behavior within "
    "the plot area. The most frequent behavior observed was foraging, which occurred across "
    "multiple timestamps throughout the day."
)

_REAL_FABRICATED_HECTARE_REPORT = (
    "On May 27, 2023, during the OFO mission, we conducted a drone survey plot covering "
    "approximately 1 hectare. This area contained 13 trees, ranging in height from 0.36 meters "
    "to 10.67 meters. The mean height of these trees was 3.25 meters."
)


def test_strip_stale_title_line_removes_real_wrong_header():
    result = _strip_stale_title_line(_REAL_STALE_TITLE_REPORT)
    assert not result.startswith("**")
    assert "Drone Survey Plot Summary" not in result
    assert result.startswith("On **2026-01-01**, a camera-trap session")


def test_strip_stale_title_line_no_title_present_is_unchanged():
    body = "On 2026-06-15, two red deer were observed foraging."
    assert _strip_stale_title_line(body) == body


def test_strip_unsupported_area_claims_removes_real_fabricated_hectare_sentence():
    result = _strip_unsupported_area_claims(_REAL_FABRICATED_HECTARE_REPORT)
    assert "hectare" not in result.lower()
    assert "acre" not in result.lower()
    # the rest of the paragraph -- real facts, not part of the fabricated sentence -- survives
    assert "13 trees" in result
    assert "3.25 meters" in result


def test_strip_unsupported_area_claims_no_area_mention_is_unchanged():
    body = "This area contained 13 trees, ranging in height from 0.36 to 10.67 meters."
    assert _strip_unsupported_area_claims(body) == body


def test_real_generation_never_states_a_hectare_or_acre_figure(generator):
    # scene_state.json has no plot-area field at all, so this must never appear in real output --
    # the actual regression test for the fabricated "~1 hectare" finding.
    scene = _real_13_tree_scene()
    report = generator.generate(scene, max_new_tokens=400)
    assert "hectare" not in report.lower()
    assert "acre" not in report.lower()


def test_real_generation_has_no_title_or_heading_line(generator):
    scene = _real_13_tree_scene()
    report = generator.generate(scene, max_new_tokens=400)
    first_line = report.strip().splitlines()[0]
    assert not first_line.strip().startswith("#")
    assert not (first_line.strip().startswith("**") and first_line.strip().endswith("**"))


def test_real_generation_states_correct_aggregate_co2e_not_hallucinated(generator):
    # the actual regression test for the ~410x hallucination: the real total (4896.4 kg) must
    # appear, and no other large kg/ton figure that isn't the real one should be presented as a
    # total -- checked by requiring the exact real number appear verbatim in the report.
    scene = _real_13_tree_scene()
    report = generator.generate(scene, max_new_tokens=400)
    assert "4896.4" in report or "4,896.4" in report, (
        f"real total CO2e (4896.4 kg) not found verbatim in report: {report!r}"
    )
