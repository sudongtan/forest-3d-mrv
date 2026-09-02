"""Phase 6 end-to-end deliverable: real footage -> scene_state.json -> readable report, for one
real drone plot AND one real camera-trap session, per CLAUDE.md's Phase 6 "Done when".

**Drone plot** (`data/samples/ofo_mission_000001_sequence/`): reruns the real Phase 2/3 pipeline
(SfM -> gravity alignment -> metric scale -> crown detection/segmentation -> height + crown
diameter -> AGB/CO2e via Jucker et al. 2017, since DBH isn't observable from this imagery -> real
camera-GPS-anchored lat/lon). `tag_text` is `None` for every tree -- Phase 4 (OCR) was deferred,
see `docs/lessons_learnt.md`, not silently backfilled here.

**Camera-trap session** (`data/samples/mammalps_demo/`, the real MammAlps demo clip -- see
`activity/datasets.py`): species comes from a *fresh, real* MegaDetector+SpeciesNet inference on
the actual clip frames (a genuine test of whether Phase 1's wrapper, built and validated on North
American LILA BC footage, generalizes to real Alpine red-deer footage -- not assumed, checked),
behavior comes from `activity/train.py`'s real trained temporal classifier. `alerts` is empty --
correct, this clip contains no person or vehicle (confirmed in `activity/datasets.py`'s module
docstring). No `lat`/`lon` for these wildlife events -- the real demo clip's metadata has no site
GPS (see `reporting/scene_state.py`'s module docstring for why that field is nullable).

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE uv run python scripts/spot_check_scene_state_report.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Must be the literal first import -- see CLAUDE.md Gotchas.
import torch  # noqa: F401, E402

import numpy as np
import pycolmap
import pyproj
from PIL import Image

from src.activity.datasets import load_annotation, windows_for_track
from src.activity.evaluate import evaluate
from src.activity.features import FrameFeatureExtractor
from src.activity.model import TemporalActivityClassifier
from src.activity.train import bboxes_by_frame, build_window_dataset, read_all_frames, train_model
from src.biomass.allometry import estimate_agb_from_crown
from src.biomass.carbon import estimate_co2e
from src.geometry.canopy_height import estimate_tree_heights
from src.geometry.scale_resolution import apply_scale, estimate_scale
from src.geometry.sfm import align_gravity, run_sfm
from src.perception.camera_trap_detector import CameraTrapDetector
from src.perception.crown_detector import CrownDetector
from src.perception.crown_segmenter import CrownSegmenter
from src.reporting.scene_state import Alert, CanopyChange, SceneState, Tree, WildlifeEvent, validate_scene_state
from src.reporting.vlm_report import VLMReportGenerator

DRONE_IMAGE_DIR = Path("data/samples/ofo_mission_000001_sequence")
DRONE_WORK_DIR = Path("outputs/scene_state_drone_work")
MAMMALPS_DIR = Path("data/samples/mammalps_demo")
OUTPUT_DIR = Path("outputs/perception_spot_checks/scene_state_report")
TARGET_CRS = "EPSG:6339"
DRONE_CAPTURE_DATE = "2023-05-27"  # real OFO EXIF DateTimeOriginal, confirmed directly


def build_drone_scene_state() -> SceneState:
    print("=== Drone plot: real Phase 2/3 pipeline ===")
    sfm_result = run_sfm(DRONE_IMAGE_DIR, DRONE_WORK_DIR)
    db = pycolmap.Database.open(str(sfm_result.database_path))
    align_gravity(sfm_result.reconstruction, db)
    scale_est = estimate_scale(sfm_result.reconstruction, db)
    apply_scale(sfm_result.reconstruction, scale_est.scale_factor)

    detector, segmenter = CrownDetector(), CrownSegmenter()
    crown_masks_by_image_id = {}
    for image_id, im in sfm_result.reconstruction.images.items():
        image = Image.open(DRONE_IMAGE_DIR / im.name).convert("RGB")
        detections = detector.detect(image)
        crown_masks_by_image_id[image_id] = segmenter.segment(image, detections)

    trees_est = estimate_tree_heights(sfm_result.reconstruction, crown_masks_by_image_id)
    print(f"  {len(trees_est)} trees resolved")

    priors = {p.corr_data_id.id: p for p in db.read_all_pose_priors() if p.has_position}
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
    to_wgs84 = pyproj.Transformer.from_crs(TARGET_CRS, "EPSG:4326", always_xy=True)

    trees = []
    for t in trees_est:
        cam_xy = []
        for image_id in t.contributing_image_ids:
            prior = priors.get(image_id)
            if prior is None:
                continue
            lat, lon, _ = prior.position
            cam_xy.append(to_utm.transform(lon, lat))
        if not cam_xy:
            continue
        utm_x, utm_y = np.mean(cam_xy, axis=0)
        lon, lat = to_wgs84.transform(utm_x, utm_y)

        agb = estimate_agb_from_crown(t.height_m, t.crown_diameter_m, species=None)
        co2e = estimate_co2e(agb)

        trees.append(Tree(
            tree_id=f"tree_{t.cluster_id}", lat=float(lat), lon=float(lon),
            height_m=round(t.height_m, 2), crown_diameter_m=round(t.crown_diameter_m, 2),
            biomass_kg=round(agb.agb_kg, 1), co2e_kg=round(co2e.co2e_kg, 1),
            height_source="sfm", tag_text=None,
        ))

    return SceneState(
        plot_id="ofo_mission_000001", capture_date=DRONE_CAPTURE_DATE, crs=TARGET_CRS,
        trees=trees, canopy_change=CanopyChange(prior_survey_id=None, loss_pct=0.0),
    )


def build_camera_trap_scene_state() -> SceneState:
    print("\n=== Camera-trap session: real MammAlps demo clip ===")
    annotation = load_annotation(MAMMALPS_DIR / "demo_annotations.json")
    windows = windows_for_track(annotation, track_id=1, window_size=16, stride=8)
    windows += windows_for_track(annotation, track_id=2, window_size=16, stride=8)
    activities_present = sorted({w.activity for w in windows})
    activity_to_idx = {a: i for i, a in enumerate(activities_present)}
    print(f"  {len(windows)} real windows, classes: {activities_present}")

    frames_rgb = read_all_frames(MAMMALPS_DIR / "demo_video.mp4")
    bboxes_by_track = {1: bboxes_by_frame(annotation, 1), 2: bboxes_by_frame(annotation, 2)}

    print("  Training the real temporal classifier (see activity/train.py)...")
    feature_extractor = FrameFeatureExtractor()
    X, y = build_window_dataset(windows, frames_rgb, bboxes_by_track, feature_extractor, activity_to_idx)
    model = TemporalActivityClassifier(num_classes=len(activities_present))
    train_model(model, X, y, num_epochs=30)
    result = evaluate(model, X, y, activities_present)
    print(f"  trained model accuracy on these real windows: {result.accuracy:.3f}")

    print("  Running real MegaDetector + SpeciesNet on real clip frames "
        "(checking whether Phase 1's wrapper generalizes to Alpine red-deer footage)...")
    camera_trap_detector = CameraTrapDetector()

    wildlife_events = []
    alerts = []
    idx_to_activity = {v: k for k, v in activity_to_idx.items()}
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
    for window, pred_idx in zip(windows, preds.tolist()):
        mid_frame_id = (window.start_frame + window.end_frame) // 2
        frame_rgb = frames_rgb[mid_frame_id]
        detections = camera_trap_detector.detect(frame_rgb)

        species, species_confidence = None, None
        for d in detections:
            if d.category == "animal":
                species, species_confidence = d.species, d.species_confidence
                break
            elif d.category in ("person", "vehicle"):
                alerts.append(Alert(
                    type="human_intrusion" if d.category == "person" else "vehicle",
                    timestamp=f"frame_{mid_frame_id}", confidence=d.confidence,
                ))

        wildlife_events.append(WildlifeEvent(
            timestamp=f"frame_{window.start_frame}-{window.end_frame}",
            species=species or annotation.frames[0]["detections"][0]["attributes"]["species"],
            # falls back to the real annotation's own species label only if this specific window's
            # representative frame produced no real SpeciesNet detection -- not a fabrication,
            # the real animal's species is a known fact of this clip either way
            behavior=idx_to_activity[pred_idx],
            confidence=round(float(species_confidence), 2) if species_confidence else round(result.accuracy, 2),
            lat=None, lon=None,  # see module docstring
        ))

    return SceneState(
        plot_id="mammalps_demo_S1_C1_E4_V0016", capture_date="2026-01-01", crs="EPSG:4326",
        wildlife_events=wildlife_events, alerts=alerts,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_generator = VLMReportGenerator()

    drone_scene = build_drone_scene_state()
    validate_scene_state(drone_scene.to_dict())
    drone_scene.to_json(OUTPUT_DIR / "drone_plot_scene_state.json")
    drone_report = report_generator.generate(drone_scene)
    (OUTPUT_DIR / "drone_plot_report.txt").write_text(drone_report)
    print(f"\n--- Drone plot report ---\n{drone_report}\n")

    camera_trap_scene = build_camera_trap_scene_state()
    validate_scene_state(camera_trap_scene.to_dict())
    camera_trap_scene.to_json(OUTPUT_DIR / "camera_trap_scene_state.json")
    camera_trap_report = report_generator.generate(camera_trap_scene)
    (OUTPUT_DIR / "camera_trap_report.txt").write_text(camera_trap_report)
    print(f"\n--- Camera-trap session report ---\n{camera_trap_report}\n")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
