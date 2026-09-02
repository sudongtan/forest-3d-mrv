"""End-to-end Phase 2 validation spot-check: real drone frames -> SfM -> gravity alignment ->
metric scale -> crown detection/segmentation -> tree heights -> compare against real LiDAR CHM.

Tree real-world XY position is anchored to the GPS of the camera(s) that actually observed each
tree (projected straight to UTM via pyproj), not derived from a whole-reconstruction rigid-fit
georeference -- see docs/lessons_learnt.md ("orientation drift" / "residual orientation
imprecision corrupts height"): a global rotation fit's error grows with distance from wherever it
was fit, but individual camera GPS is already validated to <1% accuracy (test_scale_resolution.py)
and each tree's contributing camera(s) are always close to it (nadir imagery).

This is the script version of notebooks/00_sfm_scale_validation.ipynb's core computation.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE uv run python scripts/spot_check_lidar_validation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Must be the literal first import -- see CLAUDE.md Gotchas.
import torch  # noqa: F401, E402

import laspy
import matplotlib.pyplot as plt
import numpy as np
import pycolmap
import pyproj
from PIL import Image
from scipy.interpolate import griddata

from src.geometry.canopy_height import estimate_tree_heights
from src.geometry.scale_resolution import apply_scale, estimate_scale
from src.geometry.sfm import align_gravity, run_sfm
from src.perception.crown_detector import CrownDetector
from src.perception.crown_segmenter import CrownSegmenter

IMAGE_DIR = Path("data/samples/ofo_mission_000001_sequence")
WORK_DIR = Path("outputs/lidar_validation_work")
LIDAR_PATH = Path("data/lidar/USGS_LPC_CA_SierraNevada_B22_10SFJ8040.laz")
OUTPUT_DIR = Path("outputs/perception_spot_checks/lidar_validation")
TARGET_CRS = "EPSG:6339"  # NAD83(2011) UTM Zone 10N -- matches the LAZ header, see georeference.py
GRID_RESOLUTION_M = 1.0
MATCH_RADIUS_M = 5.0  # camera-GPS-anchored positions are much tighter than the whole-reconstruction
# fit was, but a nadir camera isn't pinned exactly above the tree it saw -- a modest, not huge,
# allowance


def tree_utm_positions(reconstruction, database, trees) -> dict[int, tuple[float, float]]:
    """Each tree's real-world XY = the mean UTM position of the camera(s) that observed it,
    projected directly from their own GPS EXIF -- see module docstring for why this, not a
    whole-reconstruction rigid transform.
    """
    priors = {p.corr_data_id.id: p for p in database.read_all_pose_priors() if p.has_position}
    transformer = pyproj.Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)

    positions = {}
    for tree in trees:
        cam_xy = []
        for image_id in tree.contributing_image_ids:
            prior = priors.get(image_id)
            if prior is None:
                continue
            lat, lon, _ = prior.position
            x, y = transformer.transform(lon, lat)
            cam_xy.append((x, y))
        if cam_xy:
            positions[tree.cluster_id] = tuple(np.mean(cam_xy, axis=0))
    return positions


def build_lidar_chm(lidar_path: Path, bounds_xy: tuple[float, float, float, float]) -> dict:
    las = laspy.read(str(lidar_path))
    x, y, z = np.array(las.x), np.array(las.y), np.array(las.z)
    classification = np.array(las.classification)

    min_x, min_y, max_x, max_y = bounds_xy
    in_bounds = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)
    x, y, z, classification = x[in_bounds], y[in_bounds], z[in_bounds], classification[in_bounds]
    print(f"  {in_bounds.sum()} LiDAR points in bounds")

    ground_mask = classification == 2  # ASPRS standard: 2 = ground
    print(f"  {ground_mask.sum()} ground points, {(~ground_mask).sum()} non-ground")

    grid_x = np.arange(min_x, max_x, GRID_RESOLUTION_M)
    grid_y = np.arange(min_y, max_y, GRID_RESOLUTION_M)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    dtm = griddata((x[ground_mask], y[ground_mask]), z[ground_mask], (grid_xx, grid_yy),
                    method="linear")
    dsm = np.full(grid_xx.shape, np.nan)
    col = ((x - min_x) / GRID_RESOLUTION_M).astype(int).clip(0, grid_xx.shape[1] - 1)
    row = ((y - min_y) / GRID_RESOLUTION_M).astype(int).clip(0, grid_xx.shape[0] - 1)
    for r, c, zi in zip(row, col, z):
        if np.isnan(dsm[r, c]) or zi > dsm[r, c]:
            dsm[r, c] = zi

    chm = dsm - dtm
    return {"dtm": dtm, "dsm": dsm, "chm": chm, "grid_x": grid_x, "grid_y": grid_y}


def sample_chm_near(chm_data: dict, x: float, y: float, radius_m: float) -> float | None:
    grid_x, grid_y, chm = chm_data["grid_x"], chm_data["grid_y"], chm_data["chm"]
    col_mask = np.abs(grid_x - x) <= radius_m
    row_mask = np.abs(grid_y - y) <= radius_m
    window = chm[np.ix_(row_mask, col_mask)]
    window = window[~np.isnan(window)]
    if len(window) == 0:
        return None
    return float(np.nanmax(window))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Running SfM...")
    sfm_result = run_sfm(IMAGE_DIR, WORK_DIR)
    print(f"  {sfm_result.num_registered}/{sfm_result.num_total} registered")

    db = pycolmap.Database.open(str(sfm_result.database_path))

    print("Aligning to gravity...")
    grav = align_gravity(sfm_result.reconstruction, db)
    print(f"  used {grav.num_cameras_used} cameras, excluded {grav.num_outliers_excluded} outlier(s)")

    print("Resolving scale...")
    scale_est = estimate_scale(sfm_result.reconstruction, db)
    print(f"  scale={scale_est.scale_factor:.4f}, cv={scale_est.ratio_cv:.4f}")
    apply_scale(sfm_result.reconstruction, scale_est.scale_factor)

    print("Running crown detection + segmentation...")
    detector = CrownDetector()
    segmenter = CrownSegmenter()
    crown_masks_by_image_id = {}
    for image_id, im in sfm_result.reconstruction.images.items():
        image = Image.open(IMAGE_DIR / im.name).convert("RGB")
        detections = detector.detect(image)
        masks = segmenter.segment(image, detections)
        crown_masks_by_image_id[image_id] = masks

    print("Estimating tree heights (local, tilt-robust parameters -- see canopy_height.py)...")
    trees = estimate_tree_heights(sfm_result.reconstruction, crown_masks_by_image_id)
    print(f"  {len(trees)} trees resolved")

    print("Anchoring tree positions to contributing cameras' own GPS...")
    utm_positions = tree_utm_positions(sfm_result.reconstruction, db, trees)

    camera_utm = np.array(list(utm_positions.values()))
    margin = 15.0
    bounds = (
        camera_utm[:, 0].min() - margin, camera_utm[:, 1].min() - margin,
        camera_utm[:, 0].max() + margin, camera_utm[:, 1].max() + margin,
    )
    print(f"Building LiDAR CHM over bounds {bounds}...")
    chm_data = build_lidar_chm(LIDAR_PATH, bounds)

    print("\nComparing SfM-derived heights against LiDAR CHM:")
    results = []
    for t in trees:
        if t.cluster_id not in utm_positions:
            continue
        x, y = utm_positions[t.cluster_id]
        lidar_h = sample_chm_near(chm_data, x, y, MATCH_RADIUS_M)
        if lidar_h is None:
            continue
        results.append((t, lidar_h, (x, y)))
        print(f"  tree {t.cluster_id}: SfM={t.height_m:.2f}m, LiDAR={lidar_h:.2f}m, "
              f"diff={t.height_m - lidar_h:+.2f}m (n_obs={t.num_observations})")

    if not results:
        print("No trees matched to LiDAR CHM coverage.")
        return

    sfm_heights = np.array([r[0].height_m for r in results])
    lidar_heights = np.array([r[1] for r in results])
    abs_err = np.abs(sfm_heights - lidar_heights)
    rmse = np.sqrt(np.mean((sfm_heights - lidar_heights) ** 2))
    valid = lidar_heights > 0.5
    abs_rel = np.mean(abs_err[valid] / lidar_heights[valid]) if valid.any() else float("nan")

    print(f"\nn={len(results)} trees matched")
    print(f"RMSE: {rmse:.2f}m")
    print(f"AbsRel: {abs_rel:.2%}")
    print(f"Mean signed diff (SfM - LiDAR): {(sfm_heights - lidar_heights).mean():+.2f}m")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].imshow(chm_data["chm"], origin="lower", cmap="viridis",
                    extent=[chm_data["grid_x"].min(), chm_data["grid_x"].max(),
                            chm_data["grid_y"].min(), chm_data["grid_y"].max()])
    axes[0].scatter([r[2][0] for r in results], [r[2][1] for r in results],
                     c="red", s=30, label="SfM-resolved trees (camera-GPS-anchored)")
    axes[0].set_title("LiDAR CHM + SfM tree positions")
    axes[0].set_xlabel("UTM Easting (m)")
    axes[0].set_ylabel("UTM Northing (m)")
    axes[0].legend(fontsize=8)

    max_h = max(sfm_heights.max(), lidar_heights.max()) + 1
    axes[1].plot([0, max_h], [0, max_h], "k--", alpha=0.5, label="1:1")
    axes[1].scatter(lidar_heights, sfm_heights, c="steelblue")
    axes[1].set_xlabel("LiDAR CHM height (m)")
    axes[1].set_ylabel("SfM-derived height (m)")
    axes[1].set_title(f"n={len(results)}, RMSE={rmse:.2f}m, AbsRel={abs_rel:.1%}")
    axes[1].legend()
    axes[1].set_xlim(0, max_h)
    axes[1].set_ylim(0, max_h)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "lidar_comparison.jpg"
    plt.savefig(out_path, dpi=120)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
