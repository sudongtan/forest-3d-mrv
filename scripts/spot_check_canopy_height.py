"""End-to-end Phase 2 spot-check: real drone frames -> SfM -> gravity alignment -> metric scale ->
crown detection/segmentation on every frame -> cross-frame tree height estimation.

Not a pytest test -- a visual/numeric sanity check, meant to be rerun by eye. Prints per-tree
height estimates and saves a 3D scatter (crown-observation points, colored by resolved tree
cluster) to `outputs/perception_spot_checks/canopy_height/`.

Does not include the real-LiDAR comparison -- see `scripts/spot_check_lidar_validation.py` for
the actual validated end-to-end pipeline (this script is the lighter-weight "just the heights"
version of the same thing, kept for quick iteration on the perception/geometry wiring alone).
Must still call `align_gravity()`, though: without it, the reconstruction's world Z axis isn't
reliably vertical and every height number this script prints is meaningless (a real bug found
and fixed the hard way -- see docs/lessons_learnt.md).

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE uv run python scripts/spot_check_canopy_height.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Must be the literal first import in this file, before `pycolmap` -- pycolmap-then-torch (in
# that order, anywhere earlier in the process) crashes with SIGSEGV on this machine when a
# transformers model gets loaded later. Each src/geometry/*.py module already guards itself the
# same way, but that only helps if nothing *else* in this script imports pycolmap first -- an
# entry point's own top-level imports can still defeat it, as this script did before this line
# was added. See CLAUDE.md Gotchas.
import torch  # noqa: F401, E402

import matplotlib.pyplot as plt
import numpy as np
import pycolmap
from PIL import Image

from src.geometry.canopy_height import estimate_tree_heights
from src.geometry.scale_resolution import apply_scale, estimate_scale
from src.geometry.sfm import align_gravity, run_sfm
from src.perception.crown_detector import CrownDetector
from src.perception.crown_segmenter import CrownSegmenter

IMAGE_DIR = Path("data/samples/ofo_mission_000001_sequence")
WORK_DIR = Path("outputs/canopy_height_work")
OUTPUT_DIR = Path("outputs/perception_spot_checks/canopy_height")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Running SfM...")
    sfm_result = run_sfm(IMAGE_DIR, WORK_DIR)
    print(f"  {sfm_result.num_registered}/{sfm_result.num_total} registered, "
          f"{sfm_result.num_points3D} points")

    db = pycolmap.Database.open(str(sfm_result.database_path))

    print("Aligning to gravity (fixing world Z to actually be vertical)...")
    grav = align_gravity(sfm_result.reconstruction, db)
    print(f"  used {grav.num_cameras_used} cameras, excluded {grav.num_outliers_excluded} outlier(s)")

    print("Resolving scale...")
    scale_estimate = estimate_scale(sfm_result.reconstruction, db)
    print(f"  scale factor={scale_estimate.scale_factor:.4f}, cv={scale_estimate.ratio_cv:.4f}, "
          f"n_pairs={scale_estimate.num_pairs}")
    apply_scale(sfm_result.reconstruction, scale_estimate.scale_factor)

    print("Running crown detection + segmentation on every registered frame...")
    detector = CrownDetector()
    segmenter = CrownSegmenter()

    id_by_name = {im.name: image_id for image_id, im in sfm_result.reconstruction.images.items()}
    crown_masks_by_image_id = {}
    for name, image_id in id_by_name.items():
        image = Image.open(IMAGE_DIR / name).convert("RGB")
        detections = detector.detect(image)
        masks = segmenter.segment(image, detections)
        crown_masks_by_image_id[image_id] = masks
        print(f"  {name}: {len(masks)} crown masks")

    print("Estimating tree heights...")
    heights = estimate_tree_heights(sfm_result.reconstruction, crown_masks_by_image_id)
    print(f"\n{len(heights)} trees resolved:")
    for t in sorted(heights, key=lambda t: -t.height_m):
        print(f"  cluster {t.cluster_id}: height={t.height_m:.2f}m "
              f"(crown_top={t.crown_top_z:.2f}, ground={t.ground_z:.2f}) "
              f"from {t.num_observations} frame(s), {t.num_points} points")

    if not heights:
        print("No trees resolved -- nothing to plot.")
        return

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.tab20(np.linspace(0, 1, len(heights)))
    for t, color in zip(heights, colors):
        ax.scatter(t.centroid_xy[0], t.centroid_xy[1], t.crown_top_z, color=color, s=80,
                   label=f"tree {t.cluster_id}: {t.height_m:.1f}m")
        ax.plot([t.centroid_xy[0]] * 2, [t.centroid_xy[1]] * 2, [t.ground_z, t.crown_top_z],
                color=color, alpha=0.6)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{len(heights)} resolved trees, metric scale")
    ax.legend(loc="upper left", fontsize=7)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "tree_heights_3d.jpg"
    plt.savefig(out_path, dpi=120)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
