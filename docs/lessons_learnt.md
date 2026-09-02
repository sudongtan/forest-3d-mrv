# Lessons Learned

Running log of real bugs found and fixed while building this pipeline, and non-obvious behavior
discovered by testing against real footage/data rather than assumed. Toolchain/environment-level
issues (macOS library conflicts, API quirks) stay in `CLAUDE.md`'s Gotchas section since they're
project-setup concerns; this file is for bugs in *how a model behaves on real data*.

---

## Phase 1 — Grounding DINO's default threshold hides every real crown detection

**Where**: `src/perception/crown_detector.py`

**What happened**: Ran the crown detector on real Open Forest Observatory sample frames
(`data/samples/ofo_mission_000001/`) with a standard confidence threshold (~0.25, the value
commonly used in Grounding DINO examples/tutorials). Result: exactly 1 detection per image, and
that one box covered 90%+ of the entire frame — clearly not "one tree crown," but the whole
photo.

**Why**: Dug into the raw, pre-threshold model output (all 900 object queries, before any
filtering) instead of assuming the model was simply wrong. Found that the single *highest*
confidence match for the text prompt `"tree crown."` is consistently the entire image
(score ≈0.6) — plausible in hindsight, since a wide shot of dense canopy genuinely does look
like "a tree crown" at whole-image scale to a model not calibrated for aerial/nadir imagery. Real,
correctly-sized individual-crown boxes (roughly 250-700px on a 5472x3648 image — a plausible size
for one tree crown at typical survey altitude) were present in the same raw output, just scored
lower, around 0.11-0.25. A standard threshold keeps only the degenerate whole-image box and
silently discards every real detection — the model wasn't "not finding" trees, its real
detections were just being thrown away by a threshold tuned for a different kind of photo.

**Fix**: In `CrownDetector.detect()`:
1. Lowered `box_threshold`/`text_threshold` from ~0.25 to 0.12, to let the real per-crown boxes
   through.
2. Added an explicit `max_area_fraction` filter (default 0.3) that rejects any box covering more
   than 30% of the frame — this is what actually gets rid of the degenerate whole-image match
   that the lowered threshold now also admits.

**Verified**: Visually, on all 4 downloaded sample frames — boxes overlaid on each image and
manually inspected. Boxes land on individually-distinguishable tree crowns; the detector
correctly stays sparse over continuous, unseparable canopy (no boxes forced onto regions where
individual trees genuinely can't be told apart). Not verified against a labeled ground-truth set
(none exists for these spot-check samples) — this is a "does it look sane" check, not a
precision/recall number; that level of rigor belongs to whichever phase does quantitative
evaluation against a labeled set, if one gets added later.

**Takeaway for future wrappers**: when a pretrained model's first real-data result looks
suspiciously degenerate (one giant box, all-empty output, a single implausible detection),
inspect the *raw, pre-threshold* output before concluding the model doesn't work. A working model
with a badly-calibrated threshold and a broken model can look identical from the outside.

---

## Phase 1 — SAM2: no bugs, both capabilities verified clean on real data

**Where**: `src/perception/crown_segmenter.py`

**Per-frame segmentation** (`CrownSegmenter.segment()`): fed real Grounding DINO boxes (from the
bug above, post-fix) into SAM2 on the same OFO sample frames. High confidence scores (0.89-0.98)
and, visually, precise crown-shaped mask outlines — not just filled-in boxes — on both an open
area with isolated trees and a denser mixed-canopy frame. No calibration issues, unlike the
detector; used as-is with `multimask_output=False`.

**Video tracking spot-check** (`CrownVideoTracker.track()`): seeded with a single point on the
first frame of a real camera-trap clip
(`data/samples/camera_trap_lila_seattleish/deer/DSCF0008.AVI`, 2 deer in frame), propagated
through 60 frames (2 seconds at 30fps). Correctly stayed locked on the same deer through a full
pose change (side-on → turning and walking away), with no drift onto the second deer or the
background. Confirms this capability genuinely works, for whichever part of the project ends up
using continuous video (not the main drone/SfM path — see `CLAUDE.md`'s Phase 1/2 split on why).

**Model/library notes**: both via HF `transformers` (`facebook/sam2.1-hiera-tiny`) —
`Sam2Model`/`Sam2Processor` for per-frame masks, `Sam2VideoModel`/`Sam2VideoProcessor` for
tracking. Needed `torchvision` as an extra dependency (SAM2's image processor requires it; not
needed for Grounding DINO). A harmless warning appears on every load ("You are using a model of
type `sam2_video` to instantiate a model of type `sam2`") — the checkpoint's config is tagged for
the more general video model, loading fine into the narrower image-only class regardless.

---

## Phase 1 — MegaDetector: real upstream caching bug, and how it was found

**Where**: `src/perception/camera_trap_detector.py`

**What happened**: First attempts to load `PytorchWildlife.models.detection.MegaDetectorV6`
intermittently failed with `HTTPError: HTTP Error 504: Gateway Time-out` partway through
downloading the ~50MB checkpoint from Zenodo — and, more importantly, kept re-downloading the
full checkpoint on *every single process launch*, even after a successful download, rather than
reusing the local cache (`~/.cache/torch/hub/checkpoints/`).

**Why**: Read the actual `_load_model` source in the installed package (not just the docs) rather
than assuming caching should "just work." Found the real cause: for `version="MDV6-yolov9-c"`,
the library's own local-cache existence check looks for a file named `MDV6b-yolov9-c.pt` (note
the extra `b`), but the file that actually gets downloaded — its name comes from the download
URL, `.../files/MDV6-yolov9-c.pt` — is `MDV6-yolov9-c.pt`, no `b`. The exists-check therefore
never matches the real cached file and always falls through to a fresh download, hitting Zenodo's
occasionally-flaky host every time. Confirmed directly: found both `MDV6-yolov9-c.pt` (real,
downloaded) and a `MDV6-yolov9-c (1).pt` (wget's auto-renamed duplicate from the second forced
redownload) sitting in the cache dir side by side.

**Fix**: In `CameraTrapDetector.__init__`, check for the real cached filename
(`f"{version}.pt"`, matching the URL, not the library's internal `MODEL_NAME`) ourselves before
constructing `MegaDetectorV6`, and pass it explicitly via `weights=<path>` when it exists —
this branch in the upstream code skips the buggy URL/cache-check path entirely.
Verified: cold load with the bug present took 787ms-plus-a-flaky-network-download; load via the
explicit `weights=` workaround took 0.07s, no network call at all.

**Also required**: `PytorchWildlife`'s top-level `__init__.py` unconditionally imports its
bioacoustics submodule (unrelated to camera-trap image detection), which pulled in `soundfile`
and `librosa` as hard import-time dependencies just to use the image detector at all — added both
even though this project has no use for bioacoustics.

---

## Phase 1 — SpeciesNet: per-animal classification needs one call per box, not one call for all

**Where**: `src/perception/camera_trap_detector.py`

**What happened**: On a real frame with 2 deer, calling `SpeciesNetClassifier.preprocess()` with
both MegaDetector boxes passed together in one `bboxes` list, followed by one `predict()` call,
returned only a single classification result for the whole frame — not one per box. Not
documented explicitly in the README (which shows the CLI-only `run_md_and_speciesnet` workflow
for multi-detection cases, not the underlying Python API).

**Fix**: loop over each MegaDetector detection individually — one `preprocess(image,
bboxes=[single_box])` + `predict()` call per animal. Verified against the real 2-deer frame: the
two boxes, classified separately, correctly returned two different top-1 predictions (mule deer
0.77, white-tailed deer 0.55) rather than the same single result repeated.

**Real finding, not a bug**: species classification is genuinely uncertain frame-to-frame for
some animals. Sampling the same real coyote across multiple frames of one clip, the top-1
species prediction flipped between "coyote" and "grey fox" depending on the frame (both plausible
similar-looking canids at trail-cam image quality) — worth remembering when Phase 5's activity
recognition or `scene_state.json`'s `wildlife_events[].confidence` field get built: a single
frame's species call is not necessarily stable, and aggregating across a clip (majority vote,
highest-confidence frame, etc.) may be worth doing rather than trusting one frame's answer.

**A working example of the two-stage design catching a weak detection**: on the heron sample
frame, MegaDetector found 2 boxes — a real heron (0.90 confidence, correctly classified "bird",
0.90) and a second, much weaker box (0.22 confidence) on a plant stalk sticking out of the grass.
SpeciesNet correctly classified that second crop as "blank" (1.00 confidence) rather than forcing
a species guess onto it — exactly the kind of marginal false-positive the two-stage
detector-then-classifier design is meant to catch, seen working on real, unstaged footage.

**Also required**: LILA BC's "empty" sample clip genuinely has zero detections in every sampled
frame — the correct, desired outcome for a labeled true-negative clip. The first version of
`scripts/spot_check_camera_trap_detector.py` conflated "no frame could be read" with "no frame
had a detection" (both looked like "nothing found"), misreporting this clip as unreadable. Fixed
by tracking whether any frame was successfully decoded separately from whether any frame produced
a detection.

---

## Phase 2 — Depth Anything V2's "metric" checkpoint is not metric for drone altitude

**Where**: `src/geometry/depth_fallback.py`

**What happened**: tested `Depth-Anything-V2-Metric-Outdoor-Small-hf` (the HF-hosted metric
outdoor variant, meant to output real depth in meters rather than merely relative/ordinal depth)
against a real Open Forest Observatory nadir drone frame. Predicted depths: 6.4-17.2m across the
whole frame. The drone's own GPS/altitude EXIF for this mission puts these frames at roughly
60-100m above the canopy — the model's output is off by close to an order of magnitude.

**Why**: this checkpoint is fine-tuned on Virtual KITTI, a synthetic *ground-level driving*
dataset — car-mounted camera, road-scene depths of a few to tens of meters, forward-facing view.
None of that transfers to a drone shooting straight down from 60-100m: different altitude range,
different viewing angle, different scene content entirely. "Metric" here means "calibrated to
real units in the training domain," not "calibrated to real units in general" — a distinction
that's easy to miss from the checkpoint name alone.

**Decision, not a fix**: did not attempt a custom recalibration (e.g. rescaling the depth map
using the frame's own GPS altitude as an anchor point). This is explicitly the pipeline's
lowest-confidence fallback path already (per `CLAUDE.md`'s `height_source` field), used only when
SfM has no second viewpoint to triangulate from at all — engineering a correction for a
fundamentally out-of-domain pretrained model would be a lot of effort for a path that's meant to
be a documented-worse option, not a competitive one. `depth_fallback.py` exposes the raw depth
map (its *relative* structure within one frame — which pixels are nearer/farther — is still
usable) and explicitly marks `metric_scale_confirmed=False` rather than silently presenting the
model's numbers as trustworthy meters.

---

## Phase 2 — scale resolution: real GPS/scale agreement was much tighter than expected

**Where**: `src/geometry/scale_resolution.py`

**What happened**: this project's early planning flagged a real risk — consumer/prosumer drone
GPS altitude is typically noisy (±3-10m), which could dominate the height error budget more than
SfM itself. Tested this directly rather than assuming it: on the real 13-frame OFO sequence,
fitting the SfM-to-metric scale factor from 10 cameras' pairwise GPS-vs-SfM distance ratios, then
checking it against the 3 held-out cameras' distances, gave a **max 0.66% relative error** on
completely unseen pairs, and a coefficient of variation of ~0.5% across all 78 pairs (distances
ranging 9.5-138m). Much tighter than the ±3-10m altitude-noise hypothesis would suggest.

**Caveat, not a retraction**: one site/one flight sequence isn't enough to conclude the original
concern was wrong — this could be a well-calibrated GPS unit, favorable satellite geometry that
day, or a real-time-kinematic-corrected system, none of which is confirmed. Worth re-running the
same round-trip check on a second, independent OFO site before treating "GPS scale is precise
enough" as a settled finding rather than a promising first data point.

---

## Phase 2 — a second, more severe pycolmap/torch crash: import order, not just co-import

**Where**: `src/geometry/sfm.py`, `scale_resolution.py`, `canopy_height.py`,
`scripts/spot_check_canopy_height.py`, `tests/conftest.py`

Full writeup in `CLAUDE.md`'s Gotchas (this one is a toolchain/environment issue, not a
model-behavior-on-data finding, so it lives there per this file's own scope note at the top).
Short version: `pycolmap.Database.open()` followed by loading any `transformers` model in the
same process SIGSEGVs — a different, more severe failure than the known libomp co-import abort,
found while wiring `canopy_height.py`'s end-to-end spot-check together for the first time. Fixed
by import order (`torch` before `pycolmap`, in every module *and* every entry point that uses
both), not an environment variable. Caught a real instance of the fix being incomplete: adding it
to the `src/geometry/*.py` modules alone didn't stop the crash, because
`spot_check_canopy_height.py` itself had its own top-level `import pycolmap` ahead of its
`from src.geometry...` imports — worth remembering for any future script/entry point that touches
both `pycolmap` and a `transformers` model.

---

## Phase 2 — orientation drift between flight lines in multi-line SfM reconstructions

**Where**: `src/geometry/sfm.py` (`align_gravity`), `src/geometry/georeference.py`, the whole
Phase 2 validation effort against real LiDAR.

**Timeline of what was actually found, in order** (worth keeping, since each step's fix revealed
the next, deeper issue):

1. First real bug: on the original 13-frame single-flight-line sample, every single reconstructed
   3D point came out with a *larger* Z than every camera center. For a nadir drone shot this is
   physically impossible (points must be below the camera). Root cause: COLMAP's per-frame
   gravity prior (read from EXIF/telemetry) is stored in each camera's *local* frame, not world
   frame -- having gravity priors available during bundle adjustment does not mean the resulting
   world frame's Z axis ends up vertical. Fix: `align_gravity()` transforms each camera's local
   gravity vector into world coordinates (via that camera's own pose) and rotates the whole
   reconstruction so the robust average of those vectors becomes true "down". Confirmed working
   on the 13-frame set: point Z range moved from being disjoint-above camera Z to properly
   overlapping/bracketing it.

2. With that fixed, `georeference()`'s horizontal (XY) alignment fit got *worse*, not better
   (RMSE ~0.4m before the gravity fix, ~40m after, on the same 13-frame set). Investigated rather
   than reverted: per-camera world-frame gravity vectors on this set have real spread (std ~0.26
   in the raw per-shot readings before outlier rejection) -- individual EXIF/telemetry gravity
   readings are noisy by a few degrees, and with only 13 near-colinear cameras there isn't enough
   independent geometric constraint to average that noise down further. A real, inherent
   small-sample limitation, not a new bug.

3. Downloaded a larger, multi-flight-line sample (51 frames spanning a real turn between lines --
   see `docs/DATASETS.md` for how the flight pattern was mapped and the frames pulled) expecting
   more geometric diversity to fix the precision problem. It didn't -- `align_gravity`'s 30-degree
   inlier filter started rejecting 46 of 51 cameras as "outliers." Printed every camera's
   individual world-frame gravity vector to find out why, and found the real explanation: they
   don't scatter noisily around one true direction, they **cluster into distinct groups that line
   up exactly with which flight-line segment each camera belongs to** (frames 1-4 point one way,
   frames 5-27 a clearly different way, frames 29-48 yet another). Real gravity cannot depend on
   which direction the drone is flying -- this means the reconstruction itself has accumulated
   **orientation drift between flight lines**, i.e. it is not one consistent rigid 3D structure
   across the whole sequence, just internally consistent *within* each line.

**Why this happens** (a known photogrammetry failure mode, not specific to this codebase):
incremental SfM adds one image at a time, most tightly constrained by its overlap with
already-registered *neighboring* images. Consecutive frames along the same flight line have
strong, direct overlap; frames on an adjacent line (connected only through weaker overlap near a
turn) have fewer/noisier correspondences. With a simple camera model (`SIMPLE_RADIAL`, COLMAP's
default here) and limited viewing-angle diversity, that weaker inter-line constraint lets each
line's chain of poses drift slightly out of true rotational alignment with the others -- commonly
called "doming"/systematic deformation in the photogrammetry literature. More lines or more
frames does not fix this by itself; it needs either stronger inter-line matching (e.g.
loop-closure-aware matching, not just exhaustive pairwise) or a bundle adjustment with additional
constraints (e.g. enforcing consistent gravity across the whole reconstruction, not just
measuring it after the fact the way `align_gravity` does).

**Decision**: did not attempt to fix the underlying drift (would mean building custom
loop-closure/constrained bundle adjustment -- real scope, disproportionate to what this
validation notebook needs). Instead: use the single-flight-line 13-frame sample (no cross-line
drift by construction, since there's only one line) as the actual basis for the LiDAR height
comparison, and report the multi-line finding as an explicit, explained limitation -- relevant to
anyone scaling this pipeline to a full multi-line mission later, not a blocker for this
notebook's narrower claim.

---

## Phase 2 — residual orientation imprecision corrupts height, not just absolute position

**Where**: `src/geometry/canopy_height.py`, following directly from the orientation-drift finding
above.

**What happened**: after fixing the Z-sign bug (`align_gravity`) and re-running height estimation
on the single-flight-line 13-frame sample (the one chosen specifically to avoid the cross-line
drift issue), resolved tree heights came out as **56m to 112m** — physically impossible for this
site, a modest mixed conifer/oak stand visibly nowhere near that scale in the source photos
(compare: the tallest trees on Earth, coast redwoods, top out around 115m; this is not that
forest).

**Why, most likely** (not yet fully isolated — flagged honestly as unresolved, not asserted as
certain): `align_gravity`'s fitted "up" direction is a robust average, not an exact one — the
per-camera world-frame gravity vectors that go into it have real spread even among inliers (see
the finding above). A residual tilt of even a few degrees, applied to a reconstruction with ~150m
of horizontal extent, does not just distort *absolute* position (already documented) — it also
means real elevation (true Z) increasingly leaks into apparent horizontal position, and vice
versa, the further a point is from wherever the rotation's effective pivot sits. `canopy_height.py`
was written assuming Z is precisely vertical everywhere in the cloud; `_estimate_ground_z`'s
fixed-radius XY neighborhood search and the DBSCAN clustering step (`eps` tuned in meters along
Z) both implicitly depend on that no longer holding at longer range. A secondary compounding
possibility not ruled out: `dbscan_eps_m` was chosen before this fix, against a Z axis that
wasn't actually vertical -- what "1.5m along Z" means changed once the axis was corrected, which
could also be merging points from different trees into one cluster now.

**Status: resolved via candidate (2)/(3) above.** Swept `dbscan_eps_m` in {0.5, 1.0, 1.5} x
`ground_search_radius_m` in {2.0, 3.0, 5.0} on this exact reconstruction and printed the
resulting heights for each combination. The pattern was completely clean: small values
(`eps=0.5`, `radius=2.0-3.0`) gave physically plausible heights (0.1-10.7m); any larger value in
either parameter reliably produced impossible heights (up to 80m), monotonically worse as either
parameter grew. Confirms the diagnosis directly rather than just plausibly explaining it: the
larger the neighborhood a height calculation pulls from, the more residual tilt error it
accumulates. New defaults (`eps=0.5`, `radius=3.0`) are set in `canopy_height.py`, justified in
its own docstring, not just here. Tree-to-LiDAR horizontal matching was separately fixed the same
way -- anchoring each tree's position to its contributing camera(s)' own GPS (already <1%
accurate, see the scale-resolution finding above) instead of any whole-reconstruction rotation
fit. Final validated result against real LiDAR, with these fixes: RMSE 6.74m, AbsRel 67.9%, mean
signed diff -6.00m, systematically low -- see `CLAUDE.md`'s Phase 2 section for the full number
and its own caveats (small n, position-anchoring not precise enough to separate some nearby
trees, LiDAR/imagery capture-date gap not fully ruled out as a contributor).

---

## Phase 2 — synthetic ground-truth check: is step 7's own math correct, isolated from steps 3-5?

**Where**: `scripts/spot_check_canopy_height_synthetic.py` (new), exercising
`src/geometry/canopy_height.py`'s real `estimate_tree_heights`.

**Why this was needed**: the real 13-frame run's height/diameter numbers are corrupted by known
upstream issues (step 4's residual orientation imprecision; step 1/2's crown masks sometimes
spanning multiple trees — see above and the Phase 3 crown-diameter entry below). That makes it
impossible to tell, from the real run alone, whether `canopy_height.py`'s own clustering/height/
diameter *logic* is correct, or whether it's just faithfully propagating bad input.

**What was built**: a hand-constructed `pycolmap.Reconstruction` — a real, correctly-wired object
graph (`Rig`/`Frame`/`Camera`/`Image`/`Point3D`/`Track`, the same API real SfM output uses, just
assembled directly rather than produced by `incremental_mapping`) — with six nadir camera frames
at a known 80m altitude and two trees placed at known ground-truth height (8.00m, 15.00m) and
known real-world crown diameter (4.00m, 6.00m). Crown masks were sized by *inverting*
`_crown_diameter_from_masks`'s own formula (`area_px = pi*(D/2)^2 * (focal_length/distance_m)^2`)
against each frame's real camera-to-crown distance, so a correct recovery is a genuine check, not
circular. This is deliberately a proof-of-concept script, not a unit test — it prints recovered
vs. ground truth rather than asserting pass/fail.

**Result**: recovered height 8.05m (0.6% error) / 15.03m (0.2% error); recovered diameter 4.02m
(0.4% error) / 6.01m (0.1% error) — both trees, run for real. This confirms the DBSCAN clustering,
`_estimate_ground_z` heuristic, and the mask-area diameter formula are each implemented correctly
when given clean, already-metric-scaled input, and pins the real run's bad numbers on upstream
data quality (steps 1/2/4), not on a bug in step 7 itself.

**Caveat, stated plainly**: this is a best-case synthetic scene (denser, tighter point clusters
than the real sparse-SfM case — deliberately, since the sparse-cluster degeneracy is a separate,
already-documented issue the mask-area fix already addresses) and a single, simple two-tree
layout. It demonstrates the math is correct, not that step 7 is robust to every real-world
degeneracy — those remain the Phase 3 crown-diameter entry's problem, not this one's.

---

## Phase 2 — the validation numbers aren't perfectly reproducible run-to-run, and that's expected

**Where**: `notebooks/00_sfm_scale_validation.ipynb` vs. `scripts/spot_check_lidar_validation.py`
— nominally the exact same pipeline, run twice.

**What happened**: the script run (documented above) got n=13 trees matched, RMSE 6.74m, AbsRel
67.9%, mean signed diff −6.00m, with every single matched tree low. Building the notebook meant
re-running the identical pipeline end-to-end a second time, and it resolved a *different* tree
count — n=14 — with RMSE 6.52m, AbsRel 65.18%, mean signed diff −5.43m, and one tree (of 14) came
out *higher* via SfM than LiDAR, not lower.

**Why**: `CrownDetector`/`CrownSegmenter` (Grounding DINO + SAM2, both from Phase 1) are not given
a fixed random seed anywhere in this pipeline. Small differences in which detection boxes clear
the confidence threshold, or where exactly SAM2 draws a mask boundary, are enough to shift
`canopy_height.py`'s DBSCAN clustering step to resolve one extra (or fewer) tree cluster, which
then shifts every downstream metric.

**Why this isn't a bug worth fixing**: the two runs agree on everything that actually matters for
this project's claim — a large, systematic, mechanistically-explained height underestimate, of
similar magnitude, on nearly every real tree. Seeding the perception models would make the exact
decimal digits reproducible but wouldn't change what's true about the pipeline's accuracy; it's
not on the critical path for anything in Phases 3+. Documented instead so that neither run's exact
numbers get quoted as more precise than they are — both `CLAUDE.md` and `docs/engineering_log.md`
now state both runs' numbers side by side for this reason, and the notebook itself says so inline
rather than presenting its own numbers as the single canonical result.

---

## Phase 3 — crown diameter: three real bugs found building `_crown_diameter_from_masks`

**Where**: `src/geometry/canopy_height.py` (new in Phase 3, needed by `biomass/allometry.py`'s
Jucker et al. 2017 crown-diameter AGB path, since DBH isn't observable from this project's
nadir drone imagery).

**Bug 1 — crown diameter from the tree's own 3D point cluster gave exactly 0.00m for 10 of 13
real trees.** First attempt: `2 * mean(horizontal distance from each cluster point to the
centroid)`, reusing the same cluster `canopy_height.py` already resolves for height. Root cause,
confirmed by inspecting the raw clusters directly: sparse SfM (COLMAP matches discrete SIFT-like
features, not a dense per-pixel depth map) finds very few visually distinctive, repeatably-
matchable features per tree crown — foliage is self-similar texture, which feature matching
struggles with — so most crown clusters here resolve to *one real, unique 3D point*, observed
from several frames (hence 5-15 "points" in a cluster that are almost all exact duplicates of
that one coordinate; confirmed via a direct `unique_xyz` count on the real data: 10 of 13
clusters had `unique_xyz=1`). One point has zero horizontal spread by definition. Height still
works fine off the same cluster (it only needs that single point's Z), which is exactly why this
didn't surface during Phase 2's validation — height and diameter turned out to need different
information from the same sparse points.

**Bug 2 — the mask-area-based fix's viewing-distance term came out negative for most
observations.** Fix for bug 1: a crown mask's *pixel area* carries real crown-size information
regardless of how many distinct 3D features SfM found inside it, so convert mask pixel area to a
real-world area via the pinhole relationship `real_area_m2 = mask_area_px * (distance_m /
focal_length_px)^2`. First attempt at `distance_m`: the axis-aligned drop `camera_Z -
crown_top_Z`. On real data this came out *negative* (camera appearing below the crown) for most
observations, silently discarding them via a `distance_m <= 0` guard. Root cause: the exact same
residual-orientation-tilt problem already documented for `_estimate_ground_z`/height (`
align_gravity`'s fitted "up" is only approximately correct, so real elevation increasingly leaks
into apparent horizontal position, and vice versa, the further two points are apart) — but here
it bit much harder, because a camera can be tens of meters horizontally from the tree it
photographed, far outside the small local neighborhoods `_estimate_ground_z` already restricts
itself to. Confirmed directly on one real tree: `camera_Z - crown_top_Z` ranged from -25.8 to
-2.4 across its 5 contributing frames (all negative, all discarded), while the same 5 frames'
camera-to-tree **3D Euclidean distance** was a tight 74.5-78.0m — consistent with this site's
documented ~60-100m flight altitude. Euclidean distance is rotation-invariant (a rigid rotation
preserves all pairwise distances exactly), so it isn't corrupted by residual tilt the way a
single axis's difference is. Fixed by switching to it.

**Bug 3 — still open, not fixed: some real crown masks span multiple trees, not one.** With
bugs 1 and 2 fixed, **all 13 of 13** trees in the real sample get *implausibly large* crown
diameters (14.5-21.9m — several times a real conifer's actual crown width; the two tallest,
best-resolved trees get the smallest of the bunch at 14.5-14.7m, everything else 19-22m).
Diagnosis: this mission's real
crown masks span a huge size range — as small as ~30K px (implying a plausible ~6-7m diameter at
this altitude) up to 2.5M+ px (implying an obviously-wrong 20m+ diameter). The small masks look
like genuine single-tree crowns; the large ones are almost certainly `perception/crown_detector.py`
+ `crown_segmenter.py` (Grounding DINO + SAM2) detecting one big blob covering several adjacent
trees' combined canopy on this specific mission's wide, high-altitude nadir frames — a different
failure mode than Phase 1's own already-documented threshold issue (which hid detections
entirely; this one merges several real trees into one detection instead). Which mask a given
tree's observations happen to draw from is essentially luck of which detection box the crown
segmenter drew around it, and `canopy_height.py` has no way to tell "one real crown" from "a
multi-tree blob" from inside the biomass step. **Left honest and open, not silently patched**:
`scripts/spot_check_biomass.py` flags any tree whose `crown_diameter_m` exceeds a generous 10m
sanity bound rather than hiding the problem, and its own docstring states plainly that AGB/CO2e
numbers from this real run are not yet trustworthy point estimates the way Phase 2's LiDAR-
validated heights are. Fixing this for real means revisiting Phase 1's crown detection/
segmentation granularity for high-altitude wide-shot frames specifically (tighter box
proposals, or a different per-tree delineation strategy) — real scope, not something Phase 3's
allometric equations can correct for after the fact.

**Why this matters for how much to trust `SPECIES_TABLE`'s regional WSG/functional-type
fallback uncertainty numbers (see `biomass/allometry.py`)**: those numbers were derived assuming
the *input* (height, crown diameter) carries only the stated model-level error. Bug 3 means the
crown-diameter input itself can carry a large, systematic, as-yet-unquantified extra error on
top of that for affected trees — `scripts/spot_check_biomass.py`'s plausibility flag exists
specifically so this doesn't get silently absorbed into a number that looks precise.

---

## Phase 4 — OCR deferred: no real tagged footage exists in this project yet

**Where**: would-be `src/ocr/tag_reader.py`, not written.

**What happened**: before writing any code, checked whether this project's real footage could
even exercise a tag-OCR module. This project's only real footage so far (the OFO drone sequence
used throughout Phases 2-3) is pure nadir canopy imagery shot from ~60-100m altitude — visually
confirmed directly on a real sample frame (canopy and bare ground only visible, at a scale where
a physical tree tag or plot marker would be far smaller than one pixel). A real search for an
existing public dataset of legible tree-tag/plot-marker photos (NEON, ForestGEO, Wikimedia
Commons) turned up nothing self-serve and ready to use either. Asked the user how to proceed;
decided to defer Phase 4 to Phase 5 rather than build the wrapper against a synthetic image and
call Phase 4's real-footage "Done when" satisfied when it wasn't. Full decision record:
`CLAUDE.md`'s Phase 4 section.

---

## Phase 5 — MammAlps' full release doesn't support partial download; Rolandseck isn't self-serve

**Where**: `src/activity/datasets.py`, `src/activity/train.py`.

**What happened**: CLAUDE.md's plan called for validating the temporal-model architecture on
Rolandseck (a small, clean sanity-check set) before scaling to MammAlps. Checked both for real
before writing code, rather than assuming either was available:

- **Rolandseck**: confirmed via a real search that it's a self-created research dataset (from the
  SWIFT/MAROON paper) with no confirmed self-serve public download — matches the "unconfirmed"
  status CLAUDE.md's Datasets section already flagged. Not pursued further (would need emailing
  the authors, out of scope for this session).
- **MammAlps full release**: confirmed via a direct `curl -I`/range-request test that Zenodo's
  file server for this record does *not* support HTTP range requests (no `Accept-Ranges` header;
  a `Range:` request still returns a full `200 OK`, not `206 Partial Content`) — so Phase 2's
  `remotezip` trick (used to pull 13 frames out of OFO's 3GB `images.zip` without downloading the
  whole thing) doesn't apply here. The dataset is a single, monolithic 87.9GB zip with no
  alternative per-file download option (confirmed by inspecting the Zenodo record directly) —
  impractical for this project's local M1 Pro dev environment.

**The fix**: MammAlps' own GitHub repo (`github.com/eceo-epfl/MammAlps`) bundles one real,
fully-annotated demo clip directly in git for its own demo notebook —
`resources/demo_video.mp4` + `resources/demo_annotations.json`, ~12.6MB total, plus the real
Benchmark I/II label-taxonomy JSON files. Downloaded and used as the real, small architecture
sanity-check set in Rolandseck's place (`data/samples/mammalps_demo/`). Real content confirmed by
inspection, not assumed: 615 frames (matches the annotation's own `num_frames`), two real tracked
red deer (`track_id` 1 and 2, track 2 only present frames 219-614), three real activity classes
with meaningful representation (foraging, vigilance, unknown) and real frame-to-frame label noise
near behavior transitions (the ground truth genuinely flickers between labels every few frames at
some transitions — a real property of the annotation, not a bug in how it was parsed).

**A real, non-obvious finding surfaced by this**: MammAlps' Benchmark I schema is *dense
per-frame, per-track* (bbox + action/activity/species attributes on every frame), not a
clip-level single label the way CLAUDE.md's plan phrasing ("clip path, species, behavior label")
assumed before the real schema was inspected. `datasets.py`'s `windows_for_track` derives
fixed-length, majority-labeled windows from this instead — the manifest granularity the plan
wanted, just constructed rather than given directly.

**Real result** (`src/activity/train.py`, run on the real demo clip): a small 1D-CNN over frozen
MobileNetV3-Small per-frame features reaches **98.4% accuracy training and evaluating on all 122
real windows from both tracked individuals** — confirms the architecture can learn to discriminate
real behavior classes from real appearance features, which is exactly the sanity-check role this
step is meant to play, not a generalization claim (impossible to make honestly with one clip/one
camera — see CLAUDE.md's own site-disjoint-split guardrail). As a bonus, training on track 1 only
and evaluating on track 2 (a different real individual, same camera/scene) gives **60.4%
accuracy** — a real, well-explained number, not a mystery: track 1's real windows never contain a
single "vigilance" example (it's a foraging/unknown-only individual in this clip), so a model
trained only on track 1 structurally cannot predict "vigilance" correctly on track 2, which does
contain it. This is a real illustration of exactly why CLAUDE.md's guardrail insists on
camera/site-disjoint splits for a genuine generalization claim — a single individual's real
behavior distribution is not representative, even within the same clip.

---

## Phase 5 — a real vehicle-labeled clip exists, but doesn't visibly show a vehicle

**Where**: `src/activity/triage.py`'s real-footage coverage (`scripts/spot_check_activity_triage.py`).

**What happened**: step 15 (alerts)'s real-footage gap — none of the 4 original LILA BC sample
clips (coyote/deer/empty/heron) contain a person or vehicle, so the alert-routing path was only
unit-tested. Searched the full "Seattle(ish) Camera Traps" bucket listing (4,464 objects, public
GCS, no gate) by keyword for `vehicle`/`human`/`person`/`car`: found **19 real `vehicle`-category
clips, 0 `person`/`human` clips anywhere in the dataset**.

Downloaded and ran the real pipeline on one (`IMG_0096.AVI`) — 0 alerts, same as before. Before
concluding anything, visually inspected it frame-by-frame across its full length: **no vehicle is
visible anywhere in the clip** — just a static backyard scene. Checked two more real `vehicle`
clips the same way before accepting this as a pattern, not a fluke: same result on one
(`DSCF0020.AVI`, not kept), and on a third (`DSCF0032.AVI`, not kept) a real MegaDetector
detection did fire — 0.57-0.8 confidence "animal" — but visually it's a false positive on a broken
tree stump/branch in the foreground, not a vehicle, and SpeciesNet correctly rejected it as
`blank` downstream (the same guardrail-worked-as-intended pattern already documented for a real
heron clip in Phase 1).

**Conclusion**: this is an informal/personal dataset (already caveated in `docs/DATASETS.md` as
"not a scientific benchmark"), and its `vehicle` label most likely marks *trigger events* — e.g. a
car passing on a distant road/driveway that triggered this backyard-facing camera — not clips
where a vehicle is prominently in frame. Kept `IMG_0096.AVI` in
`data/samples/camera_trap_lila_seattleish/vehicle/` as real, honest evidence of this finding
rather than discarding it. **The alert path's real-footage gap is now more precisely
characterized** (a visual-content limitation of this specific dataset, confirmed by direct
inspection) but not fully closed — still 0 real alerts ever produced end-to-end, and no real
person-category clip exists in this dataset at all to even attempt that branch.

---

## Phase 6 — three real report-generation bugs, found by running CLAUDE.md's own sanity check

**Where**: `src/reporting/vlm_report.py`.

**Model choice, decided not deferred**: CLAUDE.md's Tech Stack named "Qwen2-VL or an API-based
model." No API credentials are configured in this environment, and this stage's own design
(`scene_state.json` only, never raw imagery — CLAUDE.md's own guardrail) means a VLM's vision
tower is never used. Used Qwen2.5-1.5B-Instruct (the text-only sibling) instead — smaller, and
matched to what this stage's input actually is.

**Bug 1 — a real alert was present in the report but buried in paragraph 3.** CLAUDE.md's Phase 6
checklist asks for exactly this check: feed a scene_state with a known intrusion alert, confirm
it's surfaced prominently. First real run, prompt-only: the model *did* mention the alert, but
three paragraphs in, despite an explicit system-prompt instruction to lead with it. Same run also
fabricated a "compared to previous surveys" claim and reported `loss_pct` as a measured 0% result,
even though `canopy_change.prior_survey_id` was null and the system prompt said not to do that.

**Bug 2 — a real 13-tree plot's real total CO2e was reported as ~410x too high.** Running the fix
for bug 1 on the *actual* Phase 2/3 validated plot (not a toy 1-2 tree example) surfaced a much
more serious failure: the report stated total CO2e as "2,007 metric tons." The real number,
summed directly from that same scene_state.json's own thirteen `co2e_kg` values: **4,896.4 kg —
about 4.9 metric tons.** The model also self-contradicted within one paragraph on the height range,
and stated a "mean height" that was actually just one specific tree's individual height, not a
computed average. Root cause: a 1.5B-parameter model cannot reliably sum, average, or take min/max
over a JSON array of more than a couple of items — and, worse, it doesn't decline to answer when
it can't; it states a wrong number with the same confident phrasing as a correct one. This is
precisely the failure this project's whole credibility argument (CLAUDE.md: "cite every constant
... a number with no source is indistinguishable from a guess") exists to prevent, now happening
inside the stage meant to *report* that credibility.

**Bug 3 — a camera-trap-only scene_state (no trees) got narrated as a "drone survey."** The real
`scene_state.json` schema has no field distinguishing a drone plot from a camera-trap session —
CLAUDE.md's own contract doesn't ask for one, since the two cases are told apart by which of
`trees`/`wildlife_events` is populated. The model guessed wrong anyway. Same run, `lat: null`/
`lon: null` (real, honest nulls — this project's real camera-trap sample has no site GPS, see the
Phase 5 entry above) got echoed as literal `[latitude]`/`[longitude]` placeholder text rather than
omitted or described in words.

**Fix, one consistent principle for all three**: don't trust the model for anything that needs to
be *correct*, only for prose/framing. (1) A plain, data-derived `ALERT:` line is prepended before
the model ever runs, guaranteeing prominence by construction. (2) `canopy_change` is rewritten to
an explicit string when there's no prior survey, instead of a bare `loss_pct: 0.0` the model can
misread. (3) `_compute_tree_summary()`/`_compute_wildlife_summary()` precompute every real
aggregate (count, height min/max/mean, total biomass, total CO2e, per-species counts) in plain
Python and inject them as a `precomputed_summary` block the model is instructed to copy from, never
recompute. (4) An explicit `session_type` label is derived from which lists are populated and
injected into the prompt. (5) Null `lat`/`lon` are dropped from the prompt copy entirely rather
than passed through as raw JSON `null`. Verified against the real 13-tree plot after the fix: the
report's stated total CO2e now matches the real summed value exactly, on repeated real reruns.

**Two more low-stakes cosmetic issues, left open for a while, now fixed the same way.** Real,
saved output (`outputs/perception_spot_checks/scene_state_report/`) captured both: the drone
report stated an invented "approximately 1 hectare" plot area (not present anywhere in the input
— `scene_state.json` has no area field, so this is guaranteed fabricated, not just a plausible
guess), and the camera-trap session report's own generated title header read
"**Drone Survey Plot Summary**" even though the body text correctly said "camera-trap session"
throughout (the `session_type` instruction reached the body but not a title the system prompt
never actually told the model to write).

**Fix, same principle as the rest of this module**: don't trust the wording alone — a reworded
system prompt (rule 1 now explicitly names hectares/acres as unsupported; a new rule 9 forbids any
title/heading line) is backed by two deterministic post-generation guardrails,
`_strip_unsupported_area_claims()` (drops any sentence containing "hectare"/"acre") and
`_strip_stale_title_line()` (drops any leading bolded/markdown-heading line regardless of
content). Verified directly against the exact real captured failure strings above (both now
produce clean output with the rest of the paragraph intact) plus two new real-generation
regression tests in `tests/test_vlm_report.py` (18 tests total, all passing) asserting neither
ever recurs.
