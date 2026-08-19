# Forest Carbon & Biodiversity Monitoring — 3D Perception + Activity Recognition Portfolio Project
Whenever facing a problem, find out if it's already solved by an established package first.
Don't reinvent things.

language: python, use uv to manage dependencies
develop environment: Apple M1 Pro, 32G, need to run inference + light fine-tuning of CV/3D models locally
in the end, pack everything in docker so it can be replicated on a GPU machine

**The core portfolio claim for this repo**: "I built a pipeline that recovers
metrically-correct 3D structure from ordinary drone/camera-trap video and validated it against
real ground truth (LiDAR canopy height), then layered temporal activity recognition and
multimodal reporting on top" — not "I ran a bunch of pretrained models on forest videos."
The geometry-vs-ground-truth validation step is the part that makes this credible; don't skip it
to get to the demo faster.

---

## Tech Stack

- **Detection / segmentation (reused as pretrained components, not retrained from scratch
  unless a validation gap demands it)**: Grounding DINO or YOLOv8/RT-DETR for tree crown /
  wildlife detection, SAM2 for crown segmentation + video object tracking, MegaDetector for
  camera-trap human/vehicle/animal triage
- **Depth & geometry (the load-bearing module)**:
  - `COLMAP` (via `pycolmap`) for structure-from-motion on overlapping drone frames — recovers
    camera poses + a metric-scale-ambiguous point cloud
  - Depth Anything V2 or Metric3D as a monocular depth fallback for single-frame / non-overlapping
    footage where SfM isn't viable
  - Scale resolution: GPS/altitude from drone flight logs (EXIF or a paired `.srt`/telemetry file)
    to convert SfM's arbitrary scale to metric scale — **do not skip this**, an unscaled point
    cloud produces meaningless height/biomass numbers
  - `rasterio` + `pyproj` for georeferencing the recovered geometry into a real CRS
- **Biomass / carbon**: standard published allometric equations (Chave et al. 2014 pantropical
  equation as default, since it's well-documented and widely cited in the MRV literature) taking
  crown diameter + estimated height → above-ground biomass → tCO2e. Treat the equation choice and
  its stated error bounds as an explicit, cited config value — not a magic number buried in code.
- **OCR**: PaddleOCR (or `easyocr` as a lighter fallback) for plot markers / GPS stake tags /
  tree ID tags visible in frames
- **Activity recognition (camera-trap module)**:
  - Pose/keypoint extraction where applicable (RTMPose or a species-agnostic keypoint model if
    available; many camera-trap behavior datasets instead provide direct clip-level labels —
    check per-dataset before assuming keypoints are extractable)
  - A lightweight temporal classifier (small transformer or 1D-CNN over per-clip features, or
    ST-GCN if keypoints are available) fine-tuned on a real labeled behavior dataset
    (MammAlps or the Rolandseck action set as the primary small, clean options — see Datasets)
- **Multimodal reporting**: a VLM (Qwen2-VL or an API-based model — decide once cost/local-compute
  tradeoff is clear on the M1) consuming structured scene output (species, counts, 3D positions,
  activity labels, biomass estimates) to generate a natural-language MRV-style report
- **Experiment tracking**: MLflow, local SQLite-backed store (`sqlite:///mlruns/mlflow.db` —
  MLflow 3.x deprecated the plain bare-filesystem `./mlruns` store, so don't rely on that)
- **Config management**: Hydra or plain YAML, one schema per pipeline stage
- **GIS**: `rasterio`, `pyproj`, `geopandas`, `xarray`
- **Demo**: Streamlit — upload flight footage (or point at a sample), get a report

---

## Repo Structure

```
forest-carbon-3d-perception/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── base.yaml
│   ├── flight/                      # per-dataset camera/scale/CRS config
│   │   └── open_forest_observatory_<site>.yaml
│   └── camera_trap/
│       └── mammalps.yaml
├── src/
│   ├── perception/                  # 2D detection/segmentation — thin wrappers around
│   │   │                            # pretrained models, not a training framework
│   │   ├── crown_detector.py        # Grounding DINO / YOLO wrapper
│   │   ├── crown_segmenter.py       # SAM2 wrapper, also used for video tracking
│   │   └── camera_trap_detector.py  # MegaDetector wrapper
│   ├── geometry/                    # the core differentiator — keep this module's
│   │   │                            # correctness the top priority
│   │   ├── sfm.py                   # pycolmap wrapper: frames -> poses + point cloud
│   │   ├── scale_resolution.py      # GPS/altitude telemetry -> metric scale factor
│   │   ├── depth_fallback.py        # Depth Anything V2 / Metric3D for non-SfM cases
│   │   ├── canopy_height.py         # point cloud + crown masks -> per-tree height
│   │   └── georeference.py          # local 3D frame -> real-world CRS (rasterio/pyproj)
│   ├── biomass/
│   │   ├── allometry.py             # cited equation(s), explicit uncertainty bounds
│   │   └── carbon.py                # biomass -> tCO2e conversion, constants documented
│   ├── ocr/
│   │   └── tag_reader.py            # plot marker / tree tag OCR -> geolocated tag records
│   ├── activity/
│   │   ├── datasets.py              # clip-level manifest reader for behavior datasets
│   │   ├── features.py              # keypoint or frame-feature extraction
│   │   ├── model.py                 # temporal classifier (ST-GCN / small transformer)
│   │   ├── train.py
│   │   └── evaluate.py              # accuracy, per-class F1, confusion matrix -> MLflow
│   ├── reporting/
│   │   ├── scene_state.py           # assembles structured JSON: trees, biomass, species,
│   │   │                            # activity, alerts, tag reads — the VLM's input contract
│   │   └── vlm_report.py            # scene_state.json -> natural-language MRV report
│   ├── mlflow_utils.py              # experiment/run setup, logging helpers
│   └── cli.py
├── notebooks/
│   ├── 00_sfm_scale_validation.ipynb    # SfM height vs. NEON/Open Forest Observatory LiDAR CHM
│   └── 01_activity_model_sanity_check.ipynb
├── app/
│   └── streamlit_app.py
├── mlruns/
└── tests/
    ├── test_scale_resolution.py
    ├── test_georeference_roundtrip.py
    └── test_scene_state_schema.py
```

**`scene_state.json` is the contract between the geometry/biomass/activity stages and the
reporting stage** — every upstream module produces (or updates) a piece of this structured
record; the VLM never touches raw pixels or point clouds, only this summary:

```jsonc
{
  "plot_id": "string",
  "capture_date": "ISO date",
  "crs": "EPSG code",
  "trees": [
    {"tree_id": "string", "lat": 0.0, "lon": 0.0, "height_m": 0.0,
     "crown_diameter_m": 0.0, "biomass_kg": 0.0, "co2e_kg": 0.0,
     "height_source": "sfm|depth_fallback", "tag_text": "string|null"}
  ],
  "canopy_change": {"prior_survey_id": "string|null", "loss_pct": 0.0},
  "wildlife_events": [
    {"timestamp": "ISO", "species": "string", "behavior": "string",
     "confidence": 0.0, "lat": 0.0, "lon": 0.0}
  ],
  "alerts": [{"type": "human_intrusion|vehicle", "timestamp": "ISO", "confidence": 0.0}]
}
```

If a pipeline stage can't populate its part of this schema, that's a sign the stage needs more
work — not a reason to loosen the schema.

---

## Datasets

- **Drone / canopy geometry validation**: Open Forest Observatory (raw imagery + orthomosaics +
  canopy height models + ground inventory across many sites) as the primary source; NEON sites
  where available as the gold-standard LiDAR CHM cross-check. Dronescape as a secondary source
  for segmentation/tracking practice (SAM-generated masks — weak labels, treat accordingly).
- **Crown detection/segmentation training**, if the pretrained Grounding DINO/SAM2 combo
  under-performs on real footage: ForestSeg (instance-level crowns) or SelvaBox (tropical crowns).
- **Camera-trap activity**: MammAlps as primary (real behavior labels, segmentation maps,
  multi-view) — Rolandseck's 7-class action set as a small, clean sanity-check set to validate
  the temporal model architecture before scaling up. SA-FARI or PanAf20K if a bigger, more
  diverse species-detection set is needed for the detection stage specifically.
- **Split discipline**: geographic-block splits for drone plots (never split adjacent tiles/trees
  across train/val/test), and camera-site splits for activity data (a clip from a camera the
  model has already seen during training leaks background/lighting information).
- **Access/license status (checked 2026-08-19)** — both primaries are confirmed freely available,
  so Phase 2's credibility anchor is not at access risk:
  - **Open Forest Observatory**: available now, no registration, CC BY 4.0. Portal:
    openforestobservatory.org/data (HTML map, STAC catalog, or Python). Example concrete download:
    zenodo.org/records/8136161 (~10.9GB, GeoTIFF orthomosaics + GeoPackage crown boundaries).
  - **NEON**: available now (DP3.30015.001, 1m GeoTIFF CHM, CC BY 4.0), but as of mid-2026
    downloading requires a free NEON account + API token (previously anonymous) — add this as a
    one-line Phase 0 setup step, not a blocker. data.neonscience.org/data-products/DP3.30015.001.
  - **MammAlps**: available now, no registration, Zenodo DOI 10.5281/zenodo.15040901, ~88GB
    zipped. **License is CC BY-NC 4.0 (non-commercial)** — fine for a portfolio project, state
    this explicitly in the README since the repo is public.
  - **SelvaBox**: available now, no gating, CC BY 4.0, huggingface.co/datasets/CanopyRS/SelvaBox.
  - **Dronescape**: gated behind an IEEE DataPort subscription ($40/mo, free only for IEEE Society
    members) for only 48MB of video. Given it's already scoped as a secondary/weak-label set,
    treat as skippable unless the subscription is otherwise justified.
  - **ForestSeg**: **unconfirmed** — could not locate a self-serve data-hosting URL (Zenodo/GitHub/
    Figshare) as of the last check; the Scientific Reports paper's data-availability statement
    needs a manual read before relying on this one. Not a blocker since it's already scoped as a
    conditional fallback, not primary.
  - **Rolandseck**: **unconfirmed / possibly request-based**, not a self-serve public download as
    of the last check — may require emailing the authors. Since it's only used as a small
    architecture sanity-check before MammAlps, have a substitute small labeled set in mind in case
    it doesn't pan out.
  - **SA-FARI**: available, CC BY-NC 4.0, annotations gated behind a Hugging Face click-through
    (account + accept terms, not an approval process); raw video on a public GCS bucket.
  - **PanAf20K**: available now, direct zip download (42GB), Non-Commercial Government Licence,
    requires citing the dataset + IJCV paper.

---

## Implementation Steps

### Phase 0 — Environment & smoke tests
- [x] `uv` project set up, `pycolmap` installed and confirmed working on the M1 (COLMAP's Python
      bindings can be finicky on Apple Silicon — verify with a trivial 3-image SfM run before
      relying on it for anything real). Done via `tests/test_environment_smoke.py`
      (`test_pycolmap_trivial_sfm_roundtrip`): synthesizes a 5-image scene, renders real images,
      runs the actual extract→match→incremental_mapping codepath, confirms >=3/5 frames register.
      Hit and fixed a real macOS libomp conflict along the way — see Gotchas.
- [x] Confirm MPS availability/limits for each pretrained model you plan to use (SAM2, Grounding
      DINO, MegaDetector, Depth Anything V2, the VLM) — note any that silently need CPU fallback
      and record it here once found. Base `torch` MPS backend confirmed working
      (`test_mps_available`); per-model checks deferred to Phase 1/2/6 when each wrapper is built,
      per guardrail #4 — see Gotchas.
- [x] MLflow local tracking store set up, one smoke-test run logged. `sqlite:///mlruns/mlflow.db`,
      experiment `phase0_environment_smoke`, run logs torch/pycolmap versions + MPS/SfM smoke
      results as params/metrics — verified queryable via `mlflow.search_runs`.
- [x] check the availability of the datasets that are used in this project. Both primaries
      confirmed freely accessible (Open Forest Observatory, NEON w/ new free API-token step);
      MammAlps confirmed accessible (CC BY-NC 4.0). Dronescape is paywalled, ForestSeg/Rolandseck
      unconfirmed — all three are already non-primary/fallback in the plan, so no plan change
      needed. Full detail in the Datasets section above.

### Phase 1 — Perception: crown detection/segmentation + camera-trap triage
Everything downstream depends on this stage's output (crown masks feed Phase 2's height
calculation, MegaDetector's triage feeds Phase 5's alerts), so wire it up and sanity-check it
against real footage before building geometry on top of it.
- [ ] `perception/crown_detector.py`: Grounding DINO (open-vocabulary — start here, since it
      needs no fine-tuning) or YOLOv8/RT-DETR wrapper for tree crown detection on drone frames
- [ ] `perception/crown_segmenter.py`: SAM2 wrapper, prompted from the detector's boxes, producing
      per-tree crown masks; also exercise SAM2's video tracking mode on a short clip so crown IDs
      stay consistent across frames, not just per-frame masks
- [ ] `perception/camera_trap_detector.py`: MegaDetector wrapper for camera-trap frames —
      animal/human/vehicle triage, the upstream gate before species/behavior classification
- [ ] Run each wrapper against a handful of real frames (not synthetic/toy images) from the actual
      datasets you'll use later (Open Forest Observatory for crowns, MammAlps or SA-FARI for
      camera-trap) and visually confirm outputs look sane — catches an obviously-broken wrapper
      before it silently corrupts every downstream stage
- [ ] **Done when:** crown detection + segmentation produces per-tree masks on a real drone frame
      sequence, and MegaDetector produces correct animal/human/vehicle triage on a real
      camera-trap clip, both spot-checked visually and logged as sample artifacts.

### Phase 2 — Geometry: the credibility anchor
- [ ] `geometry/sfm.py`: run COLMAP on a small overlapping-frame sequence from Open Forest
      Observatory, recover camera poses + point cloud
- [ ] `geometry/scale_resolution.py`: pull GPS/altitude from flight telemetry, resolve SfM's
      unitless scale to metric scale — write a test that round-trips a known ground-control
      distance and checks the recovered metric distance against it
- [ ] `geometry/depth_fallback.py`: wire up Depth Anything V2 / Metric3D for single-frame cases,
      documented as strictly a fallback (lower confidence than SfM) in `scene_state.json`'s
      `height_source` field
- [ ] `geometry/canopy_height.py`: combine point cloud + crown segmentation masks (from Phase 1's
      `perception/crown_segmenter.py`) into per-tree height estimates
- [ ] `notebooks/00_sfm_scale_validation.ipynb`: **the single most important deliverable in this
      repo** — plot your recovered canopy heights against NEON/Open Forest Observatory LiDAR CHM
      for the same plot, report AbsRel/RMSE, discuss where and why it diverges (occlusion, dense
      canopy closure, GPS telemetry noise, etc.). A model that's honestly wrong by a documented,
      explained amount is a stronger portfolio artifact than an unvalidated pretty point cloud.
- [ ] **Done when:** you have a numeric height-accuracy result against real LiDAR ground truth,
      written up with error analysis, for at least one real Open Forest Observatory site.

### Phase 3 — Biomass & carbon
- [ ] `biomass/allometry.py`: implement the chosen published equation, with the citation and its
      stated error range as an explicit docstring/config comment, not just a formula
- [ ] `biomass/carbon.py`: biomass → tCO2e, again with the conversion constant's source cited
- [ ] Propagate Phase 2's height error into a biomass uncertainty range, even a rough one —
      an MRV-flavored project that reports a bare point estimate with no error bar undersells
      the validation work already done in Phase 2
- [ ] **Done when:** `scene_state.json`'s `trees[].biomass_kg`/`co2e_kg` are populated for a real
      validated plot, with an uncertainty figure traceable back to Phase 2's measured error.

### Phase 4 — OCR
- [ ] `ocr/tag_reader.py`: PaddleOCR over frames containing plot markers/tree tags, cross-referenced
      against SfM camera poses to geolocate each read tag
- [ ] Confirm tag reads populate `scene_state.json`'s `trees[].tag_text` and can be used to
      cross-check auto-detected tree positions against a ground-truth plot record where available
- [ ] **Done when:** at least one real footage sample with visible tags produces correctly
      geolocated tag reads, with a documented false-read/miss rate (OCR on natural, angled,
      partially-occluded outdoor tags is genuinely hard — report this honestly, don't cherry-pick).

### Phase 5 — Camera-trap activity recognition
- [ ] `activity/datasets.py`: manifest reader for MammAlps (or Rolandseck for the initial sanity
      check) — clip path, species, behavior label, split, camera/site id
- [ ] `activity/model.py` + `train.py`: start with the small, clean Rolandseck set to validate the
      temporal model architecture trains and overfits a tiny subset correctly, *then* move to
      MammAlps for the real result
- [ ] `activity/evaluate.py`: per-class accuracy/F1, confusion matrix, logged to MLflow
- [ ] Wire Phase 1's `perception/camera_trap_detector.py` (MegaDetector) as the upstream triage
      step — human/vehicle detections should populate `scene_state.json`'s `alerts`, not go
      through the species-behavior classifier
- [ ] **Done when:** a real MammAlps eval split produces a documented accuracy/F1 per behavior
      class, and human/vehicle intrusion alerts are demonstrably distinguished from wildlife
      activity in `scene_state.json` output.

### Phase 6 — Multimodal reporting
- [ ] `reporting/scene_state.py`: assembles the full structured record from all upstream stages
      for one plot/session
- [ ] `reporting/vlm_report.py`: prompts a VLM with `scene_state.json` (not raw imagery, to keep
      this stage fast/cheap and to keep the report's factual claims traceable to upstream
      pipeline output rather than the VLM's own visual guesses) to produce a natural-language
      MRV-style summary
- [ ] Sanity-check: deliberately feed a `scene_state.json` with a known intrusion alert and confirm
      the generated report surfaces it prominently, not buried — this is the kind of thing that's
      easy to get subtly wrong (report reads fine but silently drops the one fact that mattered)
- [ ] **Done when:** end-to-end, real footage → `scene_state.json` → readable report, for at least
      one drone plot and one camera-trap session.

### Phase 7 — Demo & polish
- [ ] `app/streamlit_app.py`: upload footage (or pick a bundled sample) → progress through stages
      → final report + a rendered 3D/height visualization + activity timeline
- [ ] README: lead with the Phase 2 validation plot (recovered height vs. LiDAR ground truth) —
      that's the evidence this is real geometric CV, not a pretty-video demo
- [ ] Honest limitations section: SfM failure modes (fast motion, low texture, poor overlap),
      allometric equation's known error range and its own validation domain (don't imply it's
      more universal than the source paper claims), activity model's dataset-specific
      generalization limits, OCR's real-world failure rate from Phase 4
- [ ] Docker packaging: confirm `pycolmap` and any MPS-specific fallbacks (see Phase 0) behave
      correctly in a Linux/GPU container, not just on the M1 dev machine — this is exactly the
      kind of thing likely to silently break on the platform switch

---

## Guardrails

- **Geometry correctness is the top priority — validate before building on top of it.** Every
  later stage (biomass, reporting) inherits Phase 2's error. Don't let schedule pressure push you
  to skip the LiDAR validation notebook to get to the flashier demo stages sooner.
- **Cite every constant that isn't obviously self-evident** (allometric equation, carbon
  conversion factor, any confidence threshold used to gate an alert) — a number with no source is
  indistinguishable from a guess to a reviewer, and this project's entire credibility rests on not
  looking like a guess.
- **`scene_state.json` is the only interface the reporting stage sees.** Don't let
  `reporting/vlm_report.py` reach back into raw imagery or point clouds directly — if the report
  needs a fact, that fact belongs in the schema, added explicitly, not fetched ad hoc.
- **Prefer pretrained/off-the-shelf models for the 2D perception stage** (crown detection/seg,
  MegaDetector) unless real-footage validation shows a specific, documented gap. This project's
  differentiator is the geometry + fusion + reporting layers, not re-proving you can fine-tune a
  detector — don't burn time there unless the numbers say you have to.
- **Report uncertainty, not just point estimates**, anywhere error compounds across stages
  (height → biomass → carbon). A single tCO2e number with no error bar is a weaker, less honest
  artifact than a number with one.
- **Geographic/site splits, never random**, for both the drone plots and camera-trap data — a
  plot/camera the model has already seen during training leaks background, lighting, and terrain
  information into val/test.
- **Log real experiments to MLflow from Phase 2 onward**, including the geometry validation run
  (log the AbsRel/RMSE numbers as MLflow metrics, not just a notebook plot) so later changes to
  the SfM/depth pipeline are comparable over time.

---

## Gotchas & Lessons Learned

- **`pycolmap` + `torch` imported in the same process crashes on macOS/M1** with
  `OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized` (SIGABRT).
  Each wheel bundles its own copy of LLVM's OpenMP runtime (`pycolmap`'s under
  `.venv/.../pycolmap/.dylibs/libomp.dylib`, `torch`'s under `.venv/.../torch/lib/`), and macOS
  aborts the process when both load. Workaround: set `KMP_DUPLICATE_LIB_OK=TRUE` in the
  environment *before* either import happens (see `tests/conftest.py`) — this is the standard,
  widely-used pragmatic fix for this exact conflict, not a project-specific hack, but it is
  officially unsupported per the OMP runtime's own warning, so don't reach for it to silence a
  *different* crash without checking it's this same duplicate-runtime cause. Any future pipeline
  entry point that imports both `pycolmap` and `torch`-based models (which will be every stage
  from Phase 2 onward once `depth_fallback.py`, the perception wrappers, etc. are wired together)
  needs this set first — add it to `src/cli.py` and `app/streamlit_app.py` when those are built.
  Re-check whether this reproduces on the Linux/GPU Docker target in Phase 7 — different wheel
  builds there may not conflict the same way, so don't assume the workaround is still needed
  without re-testing.
- **`pycolmap.Database` has no public constructor** — use the `Database.open(path)` factory
  method, not `Database(path)` (confirmed on pycolmap 4.1.1). `Database.open` also works as a
  context manager (`with pycolmap.Database.open(path) as db:`).
- **MPS backend itself (matmul, basic tensor ops) works fine on this M1 Pro** — `torch==2.13.0`,
  `torch.backends.mps.is_available()` is `True` and a basic matmul smoke test passes. Per-model
  MPS forward-pass checks (SAM2, Grounding DINO, MegaDetector, Depth Anything V2, the VLM) are
  deferred to when each wrapper is built in Phase 1/2/6, per guardrail #4 below — this entry
  covers only the base `torch` MPS backend, confirmed working in Phase 0.

---

## How to Work Through This With Claude Code

1. Do not start Phase 5/6/7 before Phase 2's LiDAR validation notebook produces a real,
   documented number. That number is this project's entire credibility anchor — everything
   downstream is easier to build than it is to make trustworthy without it.
2. Get Phase 1's perception wrappers spot-checked against real footage before Phase 2 — a silently
   broken crown mask (empty, mis-registered, wrong scale) will corrupt every height/biomass number
   downstream in a way that's much harder to debug once it's buried under SfM output.
3. Within Phase 2, get a single COLMAP run working end-to-end on a tiny 3-5 image subset before
   scaling up to a full flight sequence — SfM failures are much easier to debug small.
4. When wiring each pretrained model (Grounding DINO, SAM2, MegaDetector, Depth Anything V2, the
   VLM), test its forward pass on MPS immediately in isolation, before integrating it into the
   pipeline — dtype/precision bugs specific to Apple Silicon are much cheaper to find as a
   standalone smoke test than to trace back through pipeline code later.
5. Ask Claude Code to write the `scene_state.json` schema validation test
   (`tests/test_scene_state_schema.py`) early and run it after every stage that writes to the
   record — this catches silent schema drift before it reaches the reporting stage.