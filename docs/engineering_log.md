# Engineering Log

A chronological record of the real errors, crashes, and design decisions hit while building this
pipeline — the "why" behind choices that aren't obvious from the code alone. Full technical
detail for toolchain/environment issues lives in `CLAUDE.md`'s Gotchas section; full detail for
model-behavior-on-real-data bugs lives in `docs/lessons_learnt.md`. This file is the index that
ties them together in the order they actually happened, plus the design discussions that aren't
written down as a "bug" anywhere else.

---

## Phase 0 — environment setup

**Error — `pycolmap` + `torch` co-import SIGABRT.** Both bundle their own copy of macOS's
`libomp.dylib`; loading both in one process aborts with `OMP: Error #15`. Fixed with
`KMP_DUPLICATE_LIB_OK=TRUE`, set in `tests/conftest.py` before either import happens. Full
detail: CLAUDE.md Gotchas.

**Error — `pycolmap.Database(path)` doesn't exist.** No public constructor; use the
`Database.open(path)` factory method instead (pycolmap 4.1.1). Full detail: CLAUDE.md Gotchas.

**Discussion — dataset access audit.** Before building anything, checked whether every dataset
named in the plan (Open Forest Observatory, NEON, MammAlps, Dronescape, ForestSeg, SelvaBox,
Rolandseck, SA-FARI, PanAf20K) is actually freely obtainable. Both primaries (OFO, NEON) confirmed
accessible; NEON now requires a free API token (a mid-2026 change); Dronescape is paywalled;
ForestSeg/Rolandseck have no confirmed self-serve download. None of the non-primary gaps forced a
plan change. Full detail: `docs/DATASETS.md`.

---

## Phase 1 — perception wrappers

**Discussion — does this project need classification, and of what kind?** Two separate needs,
resolved differently: (1) wildlife species ID — MegaDetector only gives 3-class triage
(animal/person/vehicle), so **SpeciesNet** gets chained onto its "animal" detections. (2) tree
species for biomass's wood-density lookup — confirmed via Open Forest Observatory's own
`multiview-tree-species-classification-experiments` repo that classifying tree species from
aerial crown imagery is still an open research problem even for OFO's own team, not something to
build in this project. Resolution: join against OFO's ground-reference field-survey data
(species is present there) for validation plots; fall back to a regional mean wood density
elsewhere, propagating the added uncertainty rather than silently guessing one species value.

**Discussion — where does cross-frame tree identity get resolved: Phase 1 (2D) or Phase 2 (3D)?**
Originally planned as part of `crown_segmenter.py`'s SAM2 video-tracking mode. Reconsidered: drone
photogrammetry missions fly grid/cross-hatch patterns, so chronologically-adjacent frames can have
large viewpoint jumps between flight lines — exactly the case a 2D video tracker (built for small
continuous motion) is prone to mis-associate. Moved the authoritative merge to `canopy_height.py`,
done geometrically via COLMAP poses once they exist; kept SAM2 tracking in Phase 1 only as a
capability spot-check (verified working on a real camera-trap clip, 60 frames, full pose change,
no drift).

**Error — Grounding DINO's default threshold hides every real crown detection.** On dense-canopy
drone imagery, the single highest-confidence match for "tree crown" is the whole image, not any
one tree; real per-crown boxes exist in the raw output but score lower. Fixed by lowering the
threshold and adding an explicit max-area-fraction filter to reject the degenerate whole-image
box. Full detail: `docs/lessons_learnt.md`.

**Error — PytorchWildlife (MegaDetector) redownloads its ~50MB checkpoint on every process
launch.** A filename mismatch in the library's own cache-existence check (`MDV6b-yolov9-c.pt`
checked for, `MDV6-yolov9-c.pt` actually saved) means the check always fails. Fixed by passing
the real cached path explicitly via `weights=`. Full detail: `docs/lessons_learnt.md`.

**Error — SpeciesNet classifies only the first box when given a list of boxes.** Not documented
in the README. Fixed by calling `preprocess`/`predict` once per detected animal. Also surfaced a
real (not a bug) finding: species ID can flip between visually-similar species (coyote/grey fox)
frame-to-frame for the same real animal — worth remembering for `scene_state.json`'s
per-event confidence and any future clip-level aggregation. Full detail: `docs/lessons_learnt.md`.

---

## Phase 2 — geometry

**Discussion — where to get a real overlapping frame sequence without downloading multi-GB
zips.** The OFO Phase-1 sample frames are curated non-sequential examples (confirmed: 0 SfM
matches). The full per-mission `images.zip` is 1.8-3.5GB. Used `remotezip` (HTTP range requests
against the zip's central directory, since the host supports `Accept-Ranges`) to pull just 13
consecutive, real, sequentially-numbered frames — ~106MB instead of 3.1GB. Confirmed working:
13/13 frames registered, 4,653 points. Full detail: `docs/DATASETS.md`.

**Finding — GPS/scale agreement was much tighter than the early risk assessment predicted.**
This project's own planning flagged GPS altitude noise (±3-10m) as a likely error-floor risk for
scale resolution. Measured on real data instead of assuming: max 0.66% relative error on
completely held-out camera pairs, ~0.5% coefficient of variation across 78 pairs. A genuine
positive surprise, caveated as one sample, not yet a settled finding. Full detail:
`docs/lessons_learnt.md`.

**Error — `pycolmap` used, then a `transformers` model loaded in the same process: SIGSEGV.** A
second, more severe crash than the Phase 0 libomp issue — no environment variable fixes it; only
import order does (`torch` before `pycolmap`, literally). Found while wiring
`canopy_height.py`'s end-to-end spot-check. Caught mid-fix that guarding it inside
`src/geometry/*.py` modules alone wasn't sufficient — the calling script's own top-level
`import pycolmap` (ahead of its `from src.geometry...` imports) defeated it, so every entry point
needs the guard too, not just library modules. Two regression tests lock this in
(`test_pycolmap_before_transformers_model_load_crashes_process`,
`test_torch_before_pycolmap_avoids_the_crash`), each run in an isolated subprocess since a real
SIGSEGV would kill the whole pytest run if reproduced in-process. Full detail: CLAUDE.md Gotchas.

**Error — `pycolmap._core.Point3DMap` has no `.get()` method.** Dict-like but not a real dict;
use `in` + `[]` indexing instead of `.get()`. Minor, fixed inline in `canopy_height.py`.

**Discussion — Depth Anything V2's "metric" checkpoint turned out not to be metric here.**
Tested rather than trusted: the HF "metric outdoor" checkpoint (fine-tuned on Virtual KITTI, a
ground-level driving dataset) predicts 6.4-17.2m depth on a real drone frame actually flown at
~60-100m altitude. Decision: don't engineer a recalibration for what's already meant to be the
pipeline's lowest-confidence fallback path; expose the raw depth map and mark
`metric_scale_confirmed=False` rather than presenting the numbers as trustworthy. Full detail:
`docs/lessons_learnt.md`.

**Resolved — `canopy_height.py`'s ground-elevation heuristic, and three more real findings on
top of it, in order:**

1. **Error — world Z axis wasn't actually vertical.** Every reconstructed 3D point came out
   *above* every camera on a nadir shot — physically impossible. Root cause: COLMAP's per-frame
   gravity prior is stored in each camera's local frame, not world frame; having it available
   during bundle adjustment doesn't mean the resulting world frame ends up Z-up. Fixed with
   `sfm.py`'s new `align_gravity()`: transforms each camera's local gravity into world coordinates
   via that camera's own pose, robustly averages (excluding outliers >30° from the median), and
   rotates the reconstruction so that average becomes true "up".

2. **Discussion — a 13-frame single-flight-line sample is too geometrically degenerate to
   precisely fit horizontal alignment on top of that fix.** Camera positions are near-colinear
   (SVD spread ratio ~0.004); combined with real per-shot gravity noise (std ~0.26 in raw
   readings), the horizontal georeference RMSE went from ~0.4m to ~40m after the gravity fix —
   not a regression, a real small-sample precision limit newly exposed by fixing the bigger bug.

3. **Discussion — a larger 51-frame multi-line sample (downloaded specifically to fix #2) instead
   revealed a *different*, deeper issue: orientation drift between flight lines.** Per-camera
   gravity vectors cluster by which flight-line segment a camera belongs to, not around one
   consistent world direction — impossible if gravity were the only thing varying. This is
   incremental SfM's known "doming" failure mode: weaker cross-line matching constraints let each
   line's pose chain drift slightly out of rotational alignment with the others. Decided not to
   fix the underlying drift (real scope — loop-closure-aware matching or constrained bundle
   adjustment); used the single-line sample instead, which is immune to it by construction.

4. **Error — residual orientation imprecision was corrupting height itself, not just absolute
   position.** Re-running height estimation on the (gravity-fixed) single-line sample gave 56-112m
   trees — impossible for this site. Swept `dbscan_eps_m` x `ground_search_radius_m` and found a
   clean, monotonic pattern: small values (0.5m / 3.0m) gave plausible heights (0.1-10.7m), larger
   values reliably gave impossible ones. Confirms the mechanism directly: residual tilt leaks real
   elevation into apparent horizontal position (and vice versa) more the further apart two points
   are, so any height computation pulling from a wide neighborhood inherits that error. New
   defaults set in `canopy_height.py`, justified in its own docstring. Tree-to-LiDAR horizontal
   matching was fixed the same way — anchor each tree to its contributing camera(s)' own GPS
   (already <1% accurate) rather than any whole-reconstruction rigid fit.

**Final real result** (`scripts/spot_check_lidar_validation.py`, 13-frame single-line sample vs.
real USGS 3DEP LiDAR CHM, n=13 trees): **RMSE 6.74m, AbsRel 67.9%, mean signed diff −6.00m,
systematically low on every single tree.** This is the exact failure mode the original
ground-heuristic docstring predicted, now empirically confirmed rather than just hypothesized.
Caveats stated plainly in `CLAUDE.md`: small n, several trees sharing one anchored LiDAR cell
(position-anchoring isn't precise enough to separate some nearby trees), LiDAR/imagery
capture-date gap not fully ruled out. Full blow-by-blow: `docs/lessons_learnt.md`.

**`notebooks/00_sfm_scale_validation.ipynb` built and executed for real** (`jupyter nbconvert
--execute`, not hand-typed output) — the polished Phase 2 deliverable, narrating the whole
bug/finding chain inline. Its live run landed close but not identical to the script run above:
n=14, RMSE 6.52m, AbsRel 65.18%, mean signed diff −5.43m, 13/14 trees low (one +2.0m outlier).
**New finding surfaced by having two independent runs to compare**: Grounding DINO/SAM2 aren't
seeded here, so the exact resolved-tree set — and every metric's decimal digits — shifts slightly
run to run. The qualitative result (large, systematic, mechanistically-explained underestimate) is
stable across both runs; the precise number is not, and neither run should be quoted as more exact
than that. Run also logged to MLflow (`src/mlflow_utils.py`, experiment
`phase2_geometry_validation`) per the MLflow guardrail in `CLAUDE.md`, which had not been wired up
for Phase 2 before this. **Phase 2 is now complete.**

`scripts/spot_check_canopy_height.py`'s missing-`align_gravity()` staleness (caught during a
pre-compression documentation audit) is fixed — it now matches `spot_check_lidar_validation.py`'s
correct pipeline order.

**New — synthetic ground-truth check isolates step 7's own math from steps 3-5's real-data
issues.** The real run's height/diameter numbers are known-corrupted by upstream causes (orientation
imprecision, multi-tree crown masks) — which left it unclear whether `canopy_height.py`'s own
clustering/height/diameter logic was independently correct. Built
`scripts/spot_check_canopy_height_synthetic.py`: a hand-constructed, correctly-wired
`pycolmap.Reconstruction` (real `Rig`/`Frame`/`Camera`/`Image`/`Point3D`/`Track` objects, not a
mock) with two trees at known ground-truth height/diameter, and crown masks sized by inverting
`_crown_diameter_from_masks`'s own area formula against each frame's real camera distance. Run for
real: recovered height within 0.2-0.6% of ground truth, recovered diameter within 0.1-0.4% —
confirms the module's math is correct, and pins the real run's bad numbers on upstream data
quality, not this module. Full detail: `docs/lessons_learnt.md`.

---

## Phase 3 — biomass & carbon

**Discussion — which allometric model, given this pipeline can't measure DBH.** CLAUDE.md's Tech
Stack named Chave et al. (2014) as the default equation. Implemented it (`biomass/allometry.py`,
`estimate_agb_from_dbh`), with its real coefficients/error stats verified against the actual
published PDF (not assumed from memory): AGB = 0.0673 x (rho D^2 H)^0.976, mean per-tree CV
56.5%, fit on 4004 tropical/subtropical harvested trees. But Chave needs trunk diameter (DBH) --
not observable from nadir drone imagery, since the canopy occludes the trunk from above. Added a
second, independently cited model, Jucker et al. (2017) (`estimate_agb_from_crown`), which needs
only height and crown diameter -- both of which `geometry/canopy_height.py` already produces.
Coefficients confirmed against the `itcSegment` R package's real source (which cites the same
paper), not guessed: AGB = (0.016+a) x (H x CD)^(2.013+b) x exp(0.204^2/2), CV ~=20.6% (derived
from the paper's own log-scale sigma via the standard lognormal identity). **This is the model
this project's own drone-derived trees actually use.**

**Discussion — wood specific gravity / functional-type table for this project's real site.**
Needed real, cited values for the 5 species dominant at the Phase 2 validation site (Sierra
Nevada mixed conifer/oak) -- pulled from USDA Forest Service Res. Note NRS-38 (Miles & Smith,
2009), Table 1A, green-volume-basis specific gravity column (verified against the real PDF, not
assumed). Regional fallback WSG (for trees without a species ID) is the unweighted mean of just
those 5 species, explicitly *not* a global/tropical mean (inappropriate for a temperate conifer-
dominated site) and explicitly not basal-area-weighted (no real stand-composition data
available) -- a narrow, honestly-labeled stand-in, not a general constant.

**Discussion — carbon fraction and CO2 conversion, both verified against source, not assumed.**
IPCC (2006) GL Vol. 4 Ch. 4 Table 4.3: default CF=0.47 (matches this project's Temperate/Boreal
domain too), with functional-type-specific overrides available (conifer 0.51, broadleaf 0.48,
Lamlom & Savidge 2003) reusing the same species table. CO2:C ratio = 44.01/12.011, standard
stoichiometry.

**Real bugs found building crown-diameter estimation (needed for the Jucker path) -- three of
them, in sequence, each found by actually running the pipeline end-to-end on real data rather
than trusting the math on paper:**

1. Crown diameter derived from the tree's own resolved 3D point cluster's horizontal spread came
   out exactly 0.00m for 10 of 13 real trees -- sparse SfM resolves most crowns to one duplicated
   3D point (no spread by definition), not a real point cloud spread across the crown.
2. The fix (mask pixel area -> real-world area, needing a camera-to-crown viewing distance)
   first computed that distance as the axis-aligned `camera_Z - crown_top_Z`, which came out
   negative for most observations -- the same residual-orientation-tilt problem Phase 2 already
   found for height, here biting harder over the longer camera-to-tree ranges involved. Fixed by
   switching to 3D Euclidean distance (rotation-invariant, confirmed directly: 74.5-78.0m,
   consistent with this site's known ~60-100m flight altitude, vs. the corrupted -25.8 to -2.4m
   axis-aligned figures).
3. **Not fixed, left open and flagged**: even after both fixes, all 13 of 13 real trees get
   implausibly large crown diameters (14.5-21.9m) because some of Phase 1's real crown masks on
   this mission's wide, high-altitude frames span several adjacent trees' combined canopy, not
   one tree -- a segmentation-granularity gap, not something Phase 3's equations can fix.
   `scripts/spot_check_biomass.py` flags every affected tree rather than hiding the problem.

Full diagnosis of all three: `docs/lessons_learnt.md`. `src/geometry/canopy_height.py`'s
`TreeHeightEstimate.crown_diameter_m` field was added in Phase 3 (wasn't needed until now).

**Real, tested deliverables**: `biomass/allometry.py` and `biomass/carbon.py`, both with every
constant cited and unit tests checking hand-computed values (`tests/test_allometry.py`,
`tests/test_carbon.py`, 16 tests, all passing) -- these are correct and trustworthy on their own
terms. `scripts/spot_check_biomass.py` runs the full pipeline end-to-end on the real Phase 2 plot
and populates a scene_state.json-shaped `trees[]` list with `biomass_kg`/`co2e_kg` and an
uncertainty range that combines each model's own reported error with Phase 2's measured ~65%
height AbsRel in quadrature, per CLAUDE.md's guardrail. **What's not yet a validated Phase 3
result the way Phase 2's height was**: bug 3 above means these specific real numbers carry a
real, systematic, unquantified extra bias from oversized crown masks -- Phase 3's modules are
done and correct; this particular real-data run is an honest, flagged, open finding, not a clean
validated number.

---

## Phase 4 — deferred

**Discussion — checked for real usable data before writing any code.** This project's only real
footage so far (the OFO drone sequence) is pure nadir canopy imagery from ~60-100m altitude --
confirmed directly on a real frame that no plot marker or tree tag would ever be legible in it.
No existing public dataset of real tagged-tree photos was found either (NEON, ForestGEO,
Wikimedia Commons all checked). Asked the user how to proceed rather than build
`ocr/tag_reader.py` against a synthetic image and call Phase 4's real-footage requirement met;
decided to defer to Phase 5. Full record: `docs/lessons_learnt.md`, `CLAUDE.md` Phase 4 section.

---

## Phase 5 — camera-trap activity recognition

**Discussion — Rolandseck and full MammAlps both turned out impractical, found a real
substitute.** Rolandseck (CLAUDE.md's named architecture-sanity-check set) confirmed via search
to be a self-created research dataset with no self-serve download. Full MammAlps confirmed via a
direct range-request test to be a single 87.9GB zip on a Zenodo file server that doesn't support
HTTP range requests -- Phase 2's `remotezip` trick (used for OFO's 3GB `images.zip`) doesn't
apply here, and there's no per-file download alternative. Found a real substitute instead of
stalling: MammAlps' own GitHub repo bundles one real, fully-annotated ~12.6MB demo clip in git
for its demo notebook -- downloaded and used as the real (if small) architecture sanity-check set
in Rolandseck's place.

**Real finding -- MammAlps' actual schema is dense per-frame/per-track, not clip-level.**
Confirmed by inspecting the real downloaded annotation JSON: every frame has its own
bbox+action+activity+species per tracked individual, not one label per clip. `activity/
datasets.py`'s `windows_for_track` derives fixed-length, majority-labeled windows from this --
the manifest granularity CLAUDE.md's plan wanted, constructed rather than given directly.

**Built**: `activity/datasets.py` (real-schema annotation/label-mapping loader + windowing, 7
tests against the real downloaded demo clip), `activity/features.py` (frozen MobileNetV3-Small
per-frame embeddings -- no keypoints, since MammAlps' real schema doesn't provide them),
`activity/model.py` (small 1D-CNN temporal classifier), `activity/evaluate.py` (accuracy/F1/
confusion matrix, MLflow logging), `activity/train.py` (real end-to-end run). `activity/
triage.py` (new, not originally a separate planned file) implements the human/vehicle-vs-animal
routing CLAUDE.md's Phase 5 checklist asked for -- MegaDetector's category already gates
SpeciesNet inside `camera_trap_detector.py`; this module turns that into the two scene_state.json
-shaped record lists. 5 unit tests (real MegaDetector category strings, synthetic detection
objects) plus a real-footage spot-check (`scripts/spot_check_activity_triage.py`) on the existing
Phase 1 camera-trap clips: 6/22/0/17 wildlife candidates for coyote/deer/empty/heron, 0 alerts
for all four (correct -- none of these clips contain a person or vehicle). The alert-routing path
itself is therefore only unit-tested, not exercised on real footage -- no real person/vehicle
clip was available in this project's samples.

**Real result**: `activity/train.py` on the real demo clip -- 98.4% accuracy training+evaluating
on all 122 real windows from both tracked individuals (the actual architecture-sanity-check this
step is meant to be, not a generalization claim -- one clip/one camera can't support a real
site-disjoint split, per CLAUDE.md's own guardrail). A bonus cross-track check (train on
individual 1, eval on individual 2) gives 60.4%, cleanly explained by individual 1's real windows
never containing a "vigilance" example -- illustrates directly why that guardrail exists, using
real data, not a hypothetical.

**Not done, and honestly out of reach with only one real clip**: Phase 5's actual "Done when" --
a documented accuracy/F1 on a real MammAlps *eval split* -- needs multiple real cameras/clips for
a genuine site-disjoint split, which requires either the full 87.9GB download or another data
source. Left open rather than dressed up as done.

**New -- searched for and tested real vehicle/person footage for step 15's alert path.** Searched
the full real LILA "Seattle(ish)" bucket listing (4,464 objects, not a sample) by keyword: found
19 real `vehicle`-category clips, 0 `person`/`human` clips anywhere in the dataset. Downloaded and
ran the real pipeline on one (`IMG_0096.AVI`, kept in `data/samples/.../vehicle/`) -- 0 alerts.
Visually inspected it and two more real `vehicle` clips frame-by-frame before accepting the
result: none show a vehicle in frame; one triggered a real MegaDetector false positive (0.57-0.8
"animal" on a tree stump), correctly rejected downstream by SpeciesNet as `blank`. Conclusion:
this informal/personal dataset's `vehicle` label most likely marks trigger events (a car on a
distant road), not clips with a vehicle prominently in frame -- a real, more specific
characterization of the gap, not a fix for it. Full detail: `docs/lessons_learnt.md`.

---

## Phase 6 — multimodal reporting

**Discussion -- model choice, decided rather than deferred.** CLAUDE.md's Tech Stack named
"Qwen2-VL or an API-based model." No API credentials are configured (checked: no
`ANTHROPIC_API_KEY`/`OPENAI_API_KEY`), and this stage's own design (per CLAUDE.md's guardrail)
never shows the model raw imagery -- only `scene_state.json` -- so a VLM's vision tower would be
unused weight. Used Qwen2.5-1.5B-Instruct (text-only sibling, MPS forward pass confirmed via a
real smoke test first) instead of the full VLM.

**Built**: `reporting/scene_state.py` (SceneState/Tree/WildlifeEvent/Alert/CanopyChange
dataclasses + a real jsonschema-based `validate_scene_state()` against CLAUDE.md's own contract,
9 tests), `reporting/vlm_report.py` (report generation), `tests/test_scene_state_schema.py` (per
CLAUDE.md's "How to Work Through This" guidance to write this test early -- done at Phase 6 rather
than earlier, but before any upstream drift went unnoticed).

**Real bugs found by running CLAUDE.md's own required sanity check** (full diagnosis:
`docs/lessons_learnt.md`) -- three of them, escalating in severity, each found by actually running
the check on real data rather than trusting the design on paper:

1. A known intrusion alert was present in the generated report but buried in paragraph 3, despite
   an explicit system-prompt instruction to lead with it.
2. The same run fabricated a "compared to previous surveys" claim and reported a null
   `prior_survey_id`'s `loss_pct: 0.0` as a measured result.
3. **The serious one, found once tested on the actual real 13-tree Phase 2/3 plot instead of a
   toy example**: the report stated total CO2e as "2,007 metric tons" against a real summed value
   of 4,896.4 kg (~4.9 metric tons) -- off by ~410x -- plus a self-contradicted height range and a
   "mean height" that was actually one tree's individual value. A 1.5B model cannot reliably do
   array arithmetic and states a wrong number with the same confidence as a correct one -- exactly
   the failure this project's "cite every constant" credibility argument exists to prevent.

**Fix, one principle for all three**: never let the model compute or place a fact that needs to be
correct -- alerts get a deterministic, data-derived banner prepended before generation; null
`canopy_change` gets rewritten to an unambiguous string; every real aggregate (count, height
min/max/mean, total biomass, total CO2e, per-species wildlife counts) is precomputed in plain
Python and handed to the model as a `precomputed_summary` block it's instructed to copy from, not
recompute; an explicit `session_type` label disambiguates drone-plot vs. camera-trap sessions
(the schema itself has no such field); null `lat`/`lon` are omitted from the prompt rather than
passed through as raw JSON `null` (which produced literal `[latitude]` placeholder text). Verified
fixed on the real 13-tree plot, on repeated real reruns: reported totals now match the real summed
values exactly.

**Real end-to-end deliverable** (`scripts/spot_check_scene_state_report.py`, satisfies Phase 6's
"Done when"): real footage -> `scene_state.json` -> readable report, for one real drone plot (the
Phase 2/3 validated OFO plot, real camera-GPS-anchored lat/lon, real EXIF capture date) and one
real camera-trap session (the MammAlps demo clip -- species from a *fresh* real MegaDetector+
SpeciesNet inference testing whether Phase 1's wrapper generalizes to Alpine red-deer footage,
behavior from Phase 5's real trained temporal classifier). Both scene_state.json files pass real
schema validation; both reports state correct real numbers.

**New -- the two remaining low-stakes cosmetic issues are now fixed too.** Reproduced both from
real saved output first (`outputs/.../drone_plot_report.txt`'s invented "approximately 1 hectare";
`outputs/.../camera_trap_report.txt`'s "**Drone Survey Plot Summary**" title on a camera-trap
session). Since `scene_state.json` has no area field at all, any hectare/acre mention is
guaranteed fabricated -- and rule 8 never actually asked for a title, so there's no correct one to
substitute. Fixed with the same principle as every other bug in this module: reworded system
prompt + a deterministic post-generation guardrail that doesn't depend on the model complying
(`_strip_unsupported_area_claims`, `_strip_stale_title_line`). Verified against the exact real
captured failure strings and with 2 new real-generation regression tests
(`tests/test_vlm_report.py`, 18 tests total, all passing).
