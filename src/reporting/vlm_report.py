"""Generates a natural-language MRV-style report from `scene_state.json` only -- never raw
imagery or point clouds -- so every factual claim in the report is traceable to a specific
upstream pipeline number, not the model's own visual guess (see CLAUDE.md's guardrail:
"`scene_state.json` is the only interface the reporting stage sees").

**Model choice, decided here rather than left as an open TODO**: CLAUDE.md's Tech Stack named
"Qwen2-VL or an API-based model ... decide once cost/local-compute tradeoff is clear on the M1."
Checked both real constraints before choosing: no API credentials are configured in this
environment (no `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`), and -- more fundamentally -- this stage's
own design explicitly never shows the model an image, so a VLM's vision tower would be pure dead
weight here, not a capability this stage uses. Uses **Qwen2.5-1.5B-Instruct** (text-only sibling
of the same model family CLAUDE.md named, via `transformers`, MPS forward pass confirmed) instead
of the full VLM -- smaller, faster, and matched to what this stage's input actually is. Swappable
via `DEFAULT_MODEL_ID` for a larger local model or a real API-based one once credentials exist.

**A real failure found by CLAUDE.md's own sanity-check requirement, and how it's fixed.**
CLAUDE.md's Phase 6 checklist asks for exactly this check: "deliberately feed a scene_state.json
with a known intrusion alert and confirm the generated report surfaces it prominently, not
buried." Ran it for real, prompt-only, before adding any of the code below: the model *did*
mention the alert -- but in paragraph 3, not the first sentence, despite an explicit system-prompt
instruction to lead with it. Worse, on the same run it fabricated a "compared to previous
surveys" claim and reported `loss_pct` as a measured 0% result, even though `canopy_change.
prior_survey_id` was null and the system prompt explicitly said not to do that. A 1.5B model's
zero-shot instruction-following on formatting/prominence and on not over-interpreting an
ambiguous number both turned out unreliable -- confirming CLAUDE.md's own prediction ("this is
the kind of thing that's easy to get subtly wrong: report reads fine but silently drops the one
fact that mattered") almost exactly, just as "buries" rather than "drops."

**Fix: don't trust the LLM for either fact, make both deterministic.** (1) When `alerts` is
non-empty, `generate()` below prepends a plain, data-derived `ALERT:` line built directly from
the scene_state fields -- before the LLM ever runs -- so prominence no longer depends on the
model following an instruction. (2) The JSON shown to the model rewrites `canopy_change` into an
explicit string ("no prior survey available for comparison") when `prior_survey_id` is null,
instead of handing the model a bare `loss_pct: 0.0` it can misread as a measured value. Both are
the same principle CLAUDE.md's own guardrail already states for this stage -- don't let anything
past `scene_state.json` be a fact the report merely *hopes* is right; if a fact needs to be
correct and traceable, make it code, not a prompt instruction the model might not follow.

**A second, more serious real failure, found running this on a real 13-tree plot, not a toy
example.** The report stated total CO2e as "2,007 metric tons." The real number, computed
directly from the same scene_state.json's own `trees[].co2e_kg` values: **4,896.4 kg, i.e. ~4.9
metric tons -- the model's stated figure was wrong by a factor of ~410x.** It also self-
contradicted within one paragraph ("ranging from 0.39 meters to 4.58 meters tall ... The tallest
tree measured 10.67 meters"), stated a mean height (2.03m) that was actually just one specific
tree's height, not a computed average (real mean: 3.24m), and misinterpreted `height_source:
"sfm"` as "the trees were tagged ... previously surveyed" -- a fabricated claim with no basis in
the schema (the field means the height came from photogrammetry, nothing about tagging or prior
surveys). **A 1.5B model cannot reliably sum, average, or take min/max over a JSON array, and
will confidently state a wrong number rather than decline to compute one.** This is exactly the
failure this project's whole credibility argument (CLAUDE.md: "cite every constant ... a number
with no source is indistinguishable from a guess") exists to prevent, now happening inside the
one stage meant to *report* that credibility, not undermine it.

**Fix, same principle again**: `_compute_tree_summary()` and `_compute_wildlife_summary()` below
precompute every real aggregate (tree count, height min/max/mean, total biomass_kg, total co2e_kg,
per-species wildlife event counts) in plain Python from the actual scene_state data, and these are
injected into the prompt as an explicit `precomputed_summary` block the model is instructed to
copy from, never recompute. The model is still responsible for prose/framing -- which trees are
notable, how to phrase the summary -- but never for arithmetic. Also added: an explicit
`session_type` label (the schema has no field distinguishing a drone plot from a camera-trap
session -- a real, separate ambiguity found the same run, where a wildlife-events-only scene_state
got narrated as a "drone survey"), a plain-language clarification of what `height_source` means,
and `lat`/`lon` values of `None` are omitted from the prompt copy entirely rather than passed
through as raw `null` (which produced literal `[latitude]` placeholder text in the same run).
"""
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.reporting.scene_state import SceneState, validate_scene_state

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = """You are an MRV (Monitoring, Reporting, Verification) report generator for a \
forest carbon and biodiversity monitoring system. You will be given a structured JSON record \
(scene_state.json) describing one drone survey plot or camera-trap session: recovered trees \
(height, biomass, CO2e), any detected canopy change, wildlife activity events, and any security \
alerts (human intrusion / vehicle detections). The record also includes a `session_type` label \
and a `precomputed_summary` block -- read both before writing.

Rules, follow exactly:
1. Use ONLY the numbers and facts present in the JSON -- never invent a tree, species, count, or \
number not present in the input.
2. For ANY count, sum, average, minimum, or maximum (total biomass, total CO2e, height range, \
number of trees, etc.), you MUST use the exact value already given in `precomputed_summary` -- \
do NOT compute your own by reading through the `trees` or `wildlife_events` list yourself, and \
do NOT state a single tree's individual value as if it were an aggregate.
3. `height_source: "sfm"` means the height was measured via drone photogrammetry (structure-from-\
motion). It says nothing about whether the tree was tagged or previously surveyed -- never claim \
either of those.
4. If `alerts` is non-empty, mention each alert naturally in your narrative (an ALERT line is \
already prepended before your report by the calling system, so you do not need to lead with it \
yourself -- just don't omit it from the body).
5. If `alerts` is empty, do not mention alerts, intrusions, or vehicles at all.
6. Report `canopy_change` exactly as given to you in the JSON -- do not add your own \
interpretation of what a null value means.
7. Write `session_type` as given -- do not call a camera-trap session a "drone survey" or vice \
versa.
8. Keep the report concise: a short paragraph, not a bulleted data dump."""


class VLMReportGenerator:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, scene_state: SceneState, max_new_tokens: int = 512) -> str:
        data = scene_state.to_dict()
        validate_scene_state(data)  # never let a malformed record reach the model -- see
        # scene_state.py's own guardrail about catching schema drift before it reaches reporting

        alert_banner = _build_alert_banner(data["alerts"])
        data["canopy_change"] = _canopy_change_for_prompt(data["canopy_change"])
        data["session_type"] = _session_type_label(data)
        data["precomputed_summary"] = {
            "trees": _compute_tree_summary(data["trees"]),
            "wildlife_events": _compute_wildlife_summary(data["wildlife_events"]),
        }
        for event in data["wildlife_events"]:
            if event["lat"] is None:
                del event["lat"]
            if event["lon"] is None:
                del event["lon"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(data, indent=2)},
        ]
        # `return_dict=True` + `model.generate(**inputs)` -- passing apply_chat_template's output
        # directly as a positional arg to `generate()` fails in this transformers version
        # (5.16.1): the returned object isn't the raw tensor `generate()`'s positional-arg path
        # expects. Found via a real MPS smoke test before wiring this in, per CLAUDE.md's
        # guidance to test each model's forward pass in isolation first.
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        # do_sample=False (greedy) -- deterministic given the same model+prompt, so the alert-
        # prominence sanity check (tests/test_vlm_report.py) is reproducible, not a coin flip.
        narrative = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return f"{alert_banner}\n\n{narrative}" if alert_banner else narrative


def _build_alert_banner(alerts: list[dict]) -> str:
    """A plain, data-derived ALERT line per real alert -- built directly from scene_state fields,
    not generated text -- so prominence is guaranteed by construction, not by the model choosing
    to follow an instruction. See module docstring for the real failure this replaced.
    """
    if not alerts:
        return ""
    lines = [
        f"ALERT: {a['type']} detected at {a['timestamp']} (confidence {a['confidence']:.0%})."
        for a in alerts
    ]
    return "\n".join(lines)


def _canopy_change_for_prompt(canopy_change: dict) -> dict | str:
    """Rewrites `canopy_change` into an unambiguous string when there's no prior survey, instead
    of handing the model a bare `loss_pct: 0.0` it can misread as a measured value -- see module
    docstring for the real fabricated-comparison failure this replaced.
    """
    if canopy_change["prior_survey_id"] is None:
        return "no prior survey available for comparison"
    return canopy_change


def _session_type_label(data: dict) -> str:
    """The schema itself has no field distinguishing a drone plot from a camera-trap session --
    derived here from which lists are populated, since a real run showed the model guessing wrong
    (a wildlife-events-only record narrated as a "drone survey") without an explicit label.
    """
    has_trees, has_wildlife = bool(data["trees"]), bool(data["wildlife_events"])
    if has_trees and has_wildlife:
        return "combined drone survey and camera-trap session"
    if has_trees:
        return "drone survey plot"
    if has_wildlife:
        return "camera-trap session"
    return "survey with no trees or wildlife events recorded"


def _compute_tree_summary(trees: list[dict]) -> dict | None:
    """Real aggregate statistics over `trees`, computed in plain Python -- never left for the
    model to compute itself. See module docstring: a real run had the model state a total CO2e
    off by ~410x and a "mean height" that was actually just one tree's individual height.
    """
    if not trees:
        return None
    heights = [t["height_m"] for t in trees]
    return {
        "num_trees": len(trees),
        "height_m_min": round(min(heights), 2),
        "height_m_max": round(max(heights), 2),
        "height_m_mean": round(sum(heights) / len(heights), 2),
        "total_biomass_kg": round(sum(t["biomass_kg"] for t in trees), 1),
        "total_co2e_kg": round(sum(t["co2e_kg"] for t in trees), 1),
    }


def _compute_wildlife_summary(wildlife_events: list[dict]) -> dict | None:
    if not wildlife_events:
        return None
    species_counts: dict[str, int] = {}
    for event in wildlife_events:
        species_counts[event["species"]] = species_counts.get(event["species"], 0) + 1
    return {
        "num_events": len(wildlife_events),
        "event_count_by_species": species_counts,
    }
