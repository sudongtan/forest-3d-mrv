# Forest Carbon & Biodiversity Monitoring

A pipeline that recovers metrically-correct 3D tree structure from ordinary drone video,
validates that geometry against real independent LiDAR ground truth, and layers biomass/carbon
estimation, camera-trap wildlife activity recognition, and multimodal (VLM-based) reporting on
top — an end-to-end MRV (Monitoring, Reporting, Verification) system built and checked against
real footage and real datasets at every stage, not a synthetic demo.

**The core claim this repo backs up**: recovered canopy height was validated against real
airborne LiDAR (USGS 3DEP) and found to be honestly, systematically low — by ~6.5m RMSE, ~65%
AbsRel — with the root cause diagnosed and explained, not hidden. That validation discipline, and
the habit of running every later stage against real data and documenting what actually happened
(three separate report-generation hallucination bugs were found and fixed in the final stage
alone), is what makes this a credible geometry+ML pipeline rather than a pretty demo. See
`CLAUDE.md` for the full project plan and architecture rationale, and `docs/lessons_learnt.md` /
`docs/engineering_log.md` for the complete, chronological account of every real bug found and
fixed (or honestly left open) building it.

**Status** (2026-09-02): Phases 0–3 and 6 are complete, each validated against real data. Phase 5
(camera-trap activity recognition) has a real, working, tested architecture, validated on a small
real sample — but not yet a full held-out evaluation (that needs more real multi-camera data than
this project could practically obtain, see step 14). Phase 4 (OCR) is deliberately deferred — a
real blocker was found before writing any code (step 11). Phase 7 (Streamlit demo) is not
started. Nothing below is aspirational — every "done" step has real code, a real test or
spot-check, and (where relevant) a real number attached.

## Pipeline

Each step: what it's for, what goes in and out, what model/algorithm does the work (and where the
code or notebook lives), and which real dataset it was built and checked against.

### Track A — Drone footage → 3D trees → carbon numbers

**1. Crown detection**
- **Purpose**: locate individual tree crowns in a drone photo, upstream of segmentation.
- **Input**: one drone RGB photo.
- **Output**: a bounding box per detected crown.
- **Model/code**: Grounding DINO, open-vocabulary ("tree crown" text prompt),
  `IDEA-Research/grounding-dino-tiny` via Hugging Face `transformers`. `src/perception/
  crown_detector.py`. Real bug found and fixed: the default confidence threshold hid every real
  detection on dense canopy imagery (the highest-confidence match was the whole image, not any
  one crown) — fixed with a lower threshold + a max-area-fraction filter rejecting that
  degenerate box (`docs/lessons_learnt.md`).
- **Dataset**: Open Forest Observatory (OFO) sample drone frames.

**2. Crown segmentation**
- **Purpose**: turn each crown box into a precise pixel mask.
- **Input**: drone photo + boxes from step 1.
- **Output**: one boolean pixel mask per crown, per frame.
- **Model/code**: SAM2.1 (`facebook/sam2.1-hiera-tiny`, HF `transformers`
  `Sam2Model`/`Sam2Processor`), box-prompted. `src/perception/crown_segmenter.py`. SAM2's video
  *tracking* mode (`Sam2VideoModel`) is also exercised here as a capability spot-check, verified
  clean on a real camera-trap clip — but it is **not** the source of cross-frame tree identity for
  the drone/SfM path (see step 7): drone missions fly grid patterns with large viewpoint jumps
  between chronologically-adjacent frames, exactly the case a continuous-motion 2D tracker is
  prone to mis-associate.
- **Dataset**: OFO drone frames (per-frame masks); a LILA BC camera-trap clip (tracking
  capability spot-check only).

**3. Structure-from-motion (SfM)**
- **Purpose**: recover where each photo was taken from, plus a 3D point cloud of the scene —
  scale and orientation both still arbitrary at this point.
- **Input**: a sequence of overlapping drone photos.
- **Output**: per-image camera poses + a sparse 3D point cloud (unitless scale, arbitrary
  orientation).
- **Model/code**: COLMAP (extract_features → match_exhaustive → incremental_mapping) via
  `pycolmap`. `src/geometry/sfm.py` (`run_sfm`). Real result: 13/13 frames registered, 4,653 3D
  points on the primary validation sequence.
- **Dataset**: OFO mission `000001`, a real 13-frame single-flight-line sequence and a 51-frame
  multi-line sequence, both pulled via HTTP range requests (`remotezip`) against the mission's
  full `images.zip` rather than downloading the whole 1.8–3.5GB archive (`docs/DATASETS.md`).

**4. Gravity alignment**
- **Purpose**: fix the reconstruction's world Z axis to actually point up.
- **Input**: the raw SfM reconstruction (step 3) + each frame's stored gravity prior (from EXIF).
- **Output**: the same reconstruction, rotated so a robust average of every camera's real-world
  "down" direction becomes true down.
- **Model/code**: a geometric fix, not an ML model — `align_gravity()` in `src/geometry/sfm.py`
  (robust-averages per-camera gravity priors, transformed into world coordinates via each
  camera's own pose, then applies a single Rodrigues-formula rotation to the whole
  reconstruction). **Real bug found and fixed**: without this step, every reconstructed 3D point
  came out *above* every camera on a nadir shot — physically impossible. COLMAP's own gravity
  priors are stored per-camera-local, not world-frame, so having them available during bundle
  adjustment doesn't by itself make the resulting world frame Z-up.
- **Dataset**: same OFO sequences as step 3.

**5. Scale resolution**
- **Purpose**: turn COLMAP's arbitrary unitless scale into real meters.
- **Input**: the gravity-aligned reconstruction + each frame's GPS (EXIF).
- **Output**: the same reconstruction, metrically scaled (a single scale factor applied).
- **Model/code**: median pairwise camera-distance ratio (GPS distance, projected to a local
  metric frame via `pyproj`'s Azimuthal Equidistant projection, vs. SfM's own arbitrary
  distance) — a robust-statistics method, not ML. `src/geometry/scale_resolution.py`. Real,
  measured result (`tests/test_scale_resolution.py`, a real leave-3-out validation, not a mock):
  **max 0.66% relative error** on completely held-out camera pairs, ~0.5% coefficient of
  variation across all 78 pairs — much tighter than this project's own pre-registered risk
  estimate (±3–10m of GPS/altitude noise).
- **Dataset**: OFO frames' own GPS EXIF.

**6. Depth fallback**
- **Purpose**: a lower-confidence height estimate for single/non-overlapping frames where SfM
  isn't viable at all.
- **Input**: one 2D photo.
- **Output**: a per-pixel depth map, explicitly flagged as relative-structure-only.
- **Model/code**: Depth Anything V2 (`Depth-Anything-V2-Metric-Outdoor-Small-hf`, HF
  `transformers`). `src/geometry/depth_fallback.py`. Real finding: this checkpoint's "metric"
  calibration (fine-tuned on Virtual KITTI, a ground-level driving dataset) does **not** transfer
  to drone altitude — it predicted 6.4–17.2m depth on a real OFO frame actually flown at
  ~60–100m. Decided not to build a custom recalibration for this already-lowest-confidence
  fallback; the wrapper exposes the raw depth map and an explicit `metric_scale_confirmed=False`
  flag rather than presenting it as trustworthy meters.
- **Dataset**: one real OFO sample frame (capability check only — not used in the validated
  height pipeline, which had real overlapping frames and didn't need this fallback).

**7. Canopy height + crown diameter**
- **Purpose**: one height and one crown-diameter estimate per physical tree — and, in the same
  step, resolving *which* points across every frame belong to the same physical tree.
- **Input**: the metric, gravity-aligned point cloud (step 5) + every frame's crown masks (step
  2).
- **Output**: per-tree `height_m`, `crown_diameter_m`, 3D centroid, and which frames contributed.
- **Model/code**: back-projects each frame's 2D crown mask into 3D via that frame's COLMAP pose,
  then merges same-tree observations across frames by 3D spatial proximity (Open3D
  `remove_statistical_outlier` + `cluster_dbscan`) — geometric clustering, not a learned model,
  and *not* inherited from step 2's 2D tracking IDs. `src/geometry/canopy_height.py`
  (`estimate_tree_heights`). **Three real bugs found building crown diameter specifically**: (1)
  deriving it from the tree's own sparse SfM point cluster gave exactly 0.00m for 10 of 13 real
  trees, because sparse SfM resolves most crowns to a single duplicated 3D point with no spread —
  fixed by deriving diameter from each contributing crown mask's real-world pixel area instead;
  (2) that fix's camera-to-crown distance, first computed as an axis-aligned Z-difference, came
  out negative for most observations (the same residual-orientation-tilt problem noted below,
  biting harder over long range) — fixed by switching to rotation-invariant 3D Euclidean
  distance; (3) **still open**: even fixed, all 13 of 13 real trees get implausibly large crown
  diameters (14.5–21.9m) because some real crown masks on this mission's frames span several
  adjacent trees' combined canopy, not one — a Phase 1/2 segmentation-granularity gap, flagged by
  `scripts/spot_check_biomass.py` rather than hidden. Height itself required its own earlier fix:
  residual imprecision in step 4's gravity fit was found to corrupt height (56–112m, impossible,
  before a parameter sweep on the clustering/ground-search radii fixed it — see
  `docs/lessons_learnt.md`).
- **Dataset**: OFO 13-frame sequence.

**8. Georeference**
- **Purpose**: real-world latitude/longitude per tree.
- **Input**: each tree's local 3D position + the GPS of whichever camera(s) actually saw it.
- **Output**: real lat/lon (WGS84) per tree.
- **Model/code**: each tree is anchored to the mean UTM position of its contributing camera(s)'
  own GPS (reprojected via `pyproj`), not a whole-reconstruction rigid-body fit — a global fit's
  error grows with distance from wherever it was anchored, and this project's own reconstruction
  has documented internal orientation drift (see step 3's multi-line finding) that a global fit
  would inherit. `src/geometry/georeference.py` (a horizontal-only 2D similarity fit is also
  implemented, for cases where camera-anchoring isn't available) plus the anchoring logic in
  `scripts/spot_check_lidar_validation.py`/`scripts/spot_check_scene_state_report.py`.
- **Dataset**: OFO frames' GPS EXIF (already validated to <1% relative error, step 5).

**9. LiDAR validation** — *the credibility anchor for everything above it*
- **Purpose**: check recovered tree height against independent, real ground truth.
- **Input**: recovered heights (step 7) + a real LiDAR canopy-height model of the same plot.
- **Output**: RMSE / AbsRel / mean-signed-difference, with error analysis.
- **Notebook/code**: `notebooks/00_sfm_scale_validation.ipynb` — built with heavy inline
  commentary and **executed live** (`jupyter nbconvert --execute`; every number in it is real
  output, not pasted in), plus `scripts/spot_check_lidar_validation.py`. Real, live-executed
  result: **n=14 trees matched, RMSE 6.52m, AbsRel 65.18%, mean signed diff −5.43m, 13 of 14
  trees low** (a separate script run got close-but-not-identical numbers — n=13, RMSE 6.74m,
  AbsRel 67.9% — because Grounding DINO/SAM2 aren't seeded, a real, documented finding in its own
  right). The systematic underestimate matches a mechanism `canopy_height.py`'s own docstring
  predicted *before* this validation ran: the ground-elevation heuristic likely picks up
  near-ground vegetation instead of true bare earth under canopy closure.
- **Dataset**: OFO drone imagery + real USGS 3DEP airborne LiDAR (`CA_SierraNevada_B22`
  acquisition, the tile actually covering this site — NEON's AOP network doesn't reach it, so
  USGS 3DEP was used instead, see `docs/DATASETS.md`).

**10. Biomass & carbon**
- **Purpose**: tree geometry → above-ground biomass (kg) → CO2-equivalent (kg), with an
  uncertainty range.
- **Input**: `height_m` + `crown_diameter_m` (+ species, when known) from step 7.
- **Output**: `biomass_kg`, `co2e_kg`, each with a lower/upper uncertainty bound.
- **Model/code**: **two independently cited allometric models**, because this pipeline's own
  measurements can't feed the Tech Stack's originally-named default. Chave et al. (2014) needs
  trunk diameter (DBH), which is occluded from nadir drone imagery by the canopy — implemented
  (`estimate_agb_from_dbh`) for a future ground-truthed plot, coefficients (0.0673, 0.976) and
  error (56.5% mean per-tree CV) verified against the real published paper. **The model this
  pipeline's own trees actually use** is Jucker et al. (2017) (`estimate_agb_from_crown`), which
  needs only height + crown diameter — coefficients confirmed against the real `itcSegment` R
  package source, CV ≈20.6%. Wood specific gravity / functional type comes from a small, real,
  cited species table (5 Sierra Nevada species, USDA Forest Service Res. Note NRS-38) with an
  honest regional-mean fallback when species is unknown. Carbon: IPCC (2006) default carbon
  fraction (0.47, verified against the real table) × the CO2:C molar-mass ratio.
  `src/biomass/allometry.py`, `src/biomass/carbon.py`, 16 unit tests against hand-computed
  values. Uncertainty combines each model's own reported error with step 9's measured ~65%
  height AbsRel, in quadrature (`scripts/spot_check_biomass.py`). **Caveat carried over from step
  7**: the still-open oversized-crown-mask bug means these specific real numbers aren't yet a
  validated result the way step 9's height was — the equations themselves are correct and
  tested; this particular real run is a flagged, open finding.
- **Dataset**: OFO drone plot (same trees as steps 7–9); no external field-measured-biomass
  dataset was used (none was available/needed for the equations themselves, which are validated
  in their own source papers).

**11. Tag OCR — deferred**
- **Purpose**: read a plot marker or tree ID tag, geolocate it via camera poses, cross-check
  auto-detected tree positions against a ground-truth plot record.
- **Status**: **not built.** A real blocker was found and checked before writing any code: this
  project's only real footage (the OFO nadir drone sequence, ~60–100m altitude) was visually
  confirmed to never show a legible tag or marker at all, and a real search for an existing
  public dataset of legible tree-tag photos (NEON, ForestGEO, Wikimedia Commons) found nothing
  self-serve and ready to use. Decided (with the user) to defer rather than build
  `ocr/tag_reader.py` against a synthetic image and call the real-footage requirement met.
- **Dataset**: none found.

### Track B — Camera-trap footage → wildlife events

**12. Triage**
- **Purpose**: cheap animal/person/vehicle classification on every camera-trap frame, gating
  which frames go on to species/behavior classification vs. straight to a security alert.
- **Input**: one camera-trap frame.
- **Output**: a box + category (`animal`/`person`/`vehicle`) + confidence.
- **Model/code**: MegaDetector v6 (`MDV6-yolov9-c`) via `PytorchWildlife`.
  `src/perception/camera_trap_detector.py`. Real bug found and fixed: the library's own
  local-cache existence check looked for the wrong filename, causing a ~50MB checkpoint
  re-download on every process launch — fixed by passing the real cached path directly.
- **Dataset**: LILA BC "Seattle(ish) Camera Traps" (coyote/deer/heron/empty real clips, used for
  wrapper validation since MammAlps has no small-sample path); the real MammAlps demo clip for
  the Phase 6 end-to-end camera-trap session.

**13. Species ID**
- **Purpose**: species-level identification, for `animal`-category detections only.
- **Input**: a cropped animal detection (step 12).
- **Output**: species name + confidence.
- **Model/code**: SpeciesNet (Google/CameraTrapAI), `SpeciesNetClassifier`. Same file as step 12.
  Real bug found and fixed: needs one classification call per detected animal, not one call for a
  whole frame (a multi-box call silently only classified the first box). A real, separate finding
  in Phase 6: run fresh (not reused/cached) on the MammAlps demo clip's real Alpine red deer
  footage, as a genuine check of whether a wrapper built and validated on North American species
  generalizes.
- **Dataset**: LILA BC clips; MammAlps demo clip.

**14. Behavior classification**
- **Purpose**: classify what a tracked animal is doing (foraging, vigilance, ...) over a short
  time window.
- **Input**: per-frame appearance features over a fixed 16-frame window, for one tracked
  individual.
- **Output**: one behavior label per window.
- **Model/code**: a frozen, ImageNet-pretrained MobileNetV3-Small (`torchvision`) embeds each
  frame's bbox crop (`src/activity/features.py`); a small 1D-CNN temporal classifier, trained
  from scratch on real windows, classifies the sequence (`src/activity/model.py`,
  `src/activity/train.py`). MammAlps' real annotation schema is dense per-frame/per-track (not
  clip-level as originally planned) — `src/activity/datasets.py`'s `windows_for_track` derives
  fixed-length, majority-labeled windows from it. Real result: **98.4% accuracy** training and
  evaluating on all 122 real windows from both of the demo clip's tracked individuals (the actual
  architecture-sanity-check goal of this step — not a generalization claim, which one clip/one
  camera can't honestly support). A bonus cross-individual check (train on one deer, evaluate on
  the other) gets 60.4%, cleanly explained by one individual's real windows never containing a
  "vigilance" example.
- **Dataset**: the real MammAlps demo clip (`github.com/eceo-epfl/MammAlps`, bundled in the
  dataset's own repo for its demo notebook) — used in place of both Rolandseck (turned out not to
  be a confirmed self-serve download) and the full MammAlps release (87.9GB, single Zenodo zip,
  confirmed via a real range-request test to support no partial download, unlike step 3's OFO
  source). **Not done**: a real held-out MammAlps eval split — needs multiple real cameras/sites
  for a genuine split, which needs the full 87.9GB download or another source.

**15. Alerts**
- **Purpose**: route `person`/`vehicle` detections straight to a security alert — never through
  species/behavior classification.
- **Input**: a `person`/`vehicle` category detection (step 12).
- **Output**: an alert record (`type`, `timestamp`, `confidence`).
- **Model/code**: `src/activity/triage.py` — deterministic routing logic, no model. Real-footage
  spot-check (`scripts/spot_check_activity_triage.py`): 0 alerts, correctly, on all 4 real LILA
  BC clips (none contain a person or vehicle). The alert path itself is therefore unit-tested
  with realistic constructed detections, not exercised on real footage — no real person/vehicle
  camera-trap clip was available in this project's samples.
- **Dataset**: LILA BC clips (animal path only, on real footage).

### Where the two tracks meet

**16. `scene_state.json`**
- **Purpose**: the one structured record the reporting stage is allowed to read — never raw
  imagery or point clouds.
- **Input**: real tree records (steps 7/8/10), real wildlife events (steps 12–14), real alerts
  (step 15).
- **Output**: one schema-validated JSON record.
- **Code**: `src/reporting/scene_state.py` (`SceneState`/`Tree`/`WildlifeEvent`/`Alert`/
  `CanopyChange` dataclasses + a real `jsonschema`-based `validate_scene_state()` against
  CLAUDE.md's own contract). `tests/test_scene_state_schema.py`, 9 tests.
- **Dataset**: n/a — assembly code, not a model.

**17. VLM report**
- **Purpose**: a natural-language MRV-style report, generated from `scene_state.json` only —
  never raw photos or point clouds, so every claim traces back to a specific upstream number.
- **Input**: one validated `scene_state.json`.
- **Output**: report text.
- **Model/code**: Qwen2.5-1.5B-Instruct (local, via `transformers`) — a deliberate change from
  the Tech Stack's originally-named Qwen2-VL: no API credentials are configured in this
  environment, and this stage's own design never shows the model an image, so a VLM's vision
  tower would be unused weight. `src/reporting/vlm_report.py`. **Three real bugs found by running
  the project's own required sanity check, each fixed with a deterministic guardrail rather than
  a prompt instruction**: (1) a known alert was present in the report but buried in paragraph 3 —
  fixed with a data-derived `ALERT:` banner prepended before generation; (2) a null prior-survey
  ID got fabricated into a measured "0% loss compared to previous surveys" claim — fixed by
  rewriting `canopy_change` to an explicit string when there's no prior survey; (3) **the serious
  one, found once tested on the real 13-tree plot instead of a toy example**: total CO2e was
  reported as "2,007 metric tons" against a real value of 4,896.4 kg (~4.9 metric tons) — off by
  **~410x**, because a 1.5B model cannot reliably sum/average/min/max a JSON array and states a
  wrong number as confidently as a correct one — fixed by precomputing every real aggregate in
  plain Python and handing the model a block it's instructed to copy from, never recompute.
  Verified fixed across repeated real reruns. Two low-stakes cosmetic issues were left honestly
  open rather than further chased (an occasionally-invented "~1 hectare" plot-area detail, and an
  inconsistent report title on the camera-trap case) — see `docs/lessons_learnt.md`.
- **Dataset**: n/a — a pretrained instruction-tuned checkpoint, no project-specific training
  data. Real end-to-end deliverable: `scripts/spot_check_scene_state_report.py` — one real drone
  plot + one real camera-trap session, both schema-valid, both reports checked against the real
  underlying numbers.

**18. Streamlit demo**
- **Status**: not started (Phase 7). Planned: upload footage (or pick a bundled sample) → run
  the above end-to-end → final report + a rendered 3D/height view + activity timeline.

---

MLflow logs real numeric results alongside steps 3–5 (`phase0_environment_smoke`), step 9
(`phase2_geometry_validation`), and step 14 (`phase5_activity_sanity_check`) — local, SQLite-
backed store (`sqlite:///mlruns/mlflow.db`); `mlflow ui --backend-store-uri sqlite:///mlruns/
mlflow.db` to browse runs. It isn't part of the data flow itself.

## Datasets & licenses

See [docs/DATASETS.md](docs/DATASETS.md) for exact access/download instructions per dataset.
Primary sources (Open Forest Observatory, NEON, SelvaBox) are CC BY 4.0. **MammAlps (camera-trap
activity) and SA-FARI/PanAf20K are non-commercial licenses (CC BY-NC 4.0 or equivalent)** — fine
for this portfolio project, called out here since the repo is public.

## Setup

```bash
uv sync
uv run pytest tests/
```

`pycolmap` and `torch` cannot both be imported in the same process on macOS without
`KMP_DUPLICATE_LIB_OK=TRUE` set first (duplicate bundled `libomp.dylib` — see `CLAUDE.md`
Gotchas). Tests set this in `tests/conftest.py`; any script or entry point importing both needs
to set it too, e.g.:

```bash
KMP_DUPLICATE_LIB_OK=TRUE uv run python scripts/spot_check_scene_state_report.py
```

`scripts/` holds one real, runnable spot-check per pipeline stage (all read real sample data
under `data/samples/`, gitignored — see `docs/DATASETS.md` to reproduce them).
