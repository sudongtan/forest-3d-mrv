"""Synthetic proof-of-concept for `geometry/canopy_height.py` (step 7), decoupled from real
step 3 (SfM) / step 5 (scale resolution) data quality.

Why this script exists: on this project's real 13-frame OFO sequence, steps 3 and 5 are
themselves good (13/13 registered, <1% scale error -- see README step 3/5), but step 4's
residual orientation imprecision and step 1/2's crown-mask granularity corrupt step 7's real
numbers (impossible 56-112m heights before a parameter sweep; implausible 14.5-21.9m crown
diameters, still open -- see README step 7 discussion). That makes it hard to tell, from the real
run alone, whether step 7's *own* clustering/height/diameter logic is correct, or whether it's
just compounding upstream error.

This script isolates step 7 by hand-building a `pycolmap.Reconstruction` with camera poses and a
point cloud that are already correct and already metric-scaled -- i.e. dummy, known-good
stand-ins for step 3+5's output -- plus hand-built crown masks standing in for step 1+2's output.
Two trees are placed at known height and known real-world crown diameter; if step 7 recovers
those numbers back out, that's a real (not hand-waved) confirmation that the clustering,
ground-height heuristic, and mask-area diameter formula are each implemented correctly, and that
the real run's bad numbers trace to upstream data quality, not to this module's own math.

This is a proof-of-concept, not a unit test: it doesn't assert pass/fail, it prints recovered vs.
ground-truth numbers so the comparison can be read directly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: F401  -- import before pycolmap, see CLAUDE.md Gotchas
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

from src.geometry.canopy_height import estimate_tree_heights
from src.perception.crown_detector import CrownDetection
from src.perception.crown_segmenter import CrownMask

CAMERA_ALTITUDE_M = 80.0  # matches this project's real ~60-100m OFO flight altitude
FOCAL_LENGTH_PX = 1400.0
IMAGE_WIDTH, IMAGE_HEIGHT = 1920, 1080
FRAME_X_POSITIONS_M = [-25.0, -15.0, -5.0, 5.0, 15.0, 25.0]  # one flight line, y=0

TREES_GROUND_TRUTH = [
    {"name": "tree_A", "xy": (0.0, 0.0), "height_m": 8.0, "diameter_m": 4.0},
    {"name": "tree_B", "xy": (18.0, 0.0), "height_m": 15.0, "diameter_m": 6.0},
]

RNG = np.random.default_rng(0)


def build_nadir_camera_and_frames(recon: pycolmap.Reconstruction) -> dict:
    camera = pycolmap.Camera.create_from_model_id(
        1, pycolmap.CameraModelId.SIMPLE_PINHOLE, FOCAL_LENGTH_PX, IMAGE_WIDTH, IMAGE_HEIGHT
    )
    recon.add_camera(camera)
    rig = pycolmap.Rig(rig_id=1)
    rig.add_ref_sensor(camera.sensor_id)
    recon.add_rig(rig)

    # nadir: camera looks straight down at world -Z, i.e. rotated 180deg about X from identity
    rotation = pycolmap.Rotation3d(Rotation.from_euler("x", 180, degrees=True).as_quat())

    image_ids_by_x = {}
    for i, x in enumerate(FRAME_X_POSITIONS_M):
        image_id = i + 1
        cam_pos = np.array([x, 0.0, CAMERA_ALTITUDE_M])
        cam_from_world = pycolmap.Rigid3d(
            rotation=rotation, translation=-(rotation.matrix() @ cam_pos)
        )
        image = pycolmap.Image(name=f"frame_{image_id}.png", camera_id=camera.camera_id, image_id=image_id)
        frame = pycolmap.Frame()
        frame.frame_id = image_id
        frame.rig_id = rig.rig_id
        frame.add_data_id(image.data_id)
        frame.rig_from_world = cam_from_world
        recon.add_frame(frame)
        image.frame_id = frame.frame_id
        recon.add_image(image)
        image_ids_by_x[x] = image_id
    return image_ids_by_x


def in_bounds(px: np.ndarray, margin_px: float = 0.0) -> bool:
    return (
        margin_px <= px[0] <= IMAGE_WIDTH - margin_px
        and margin_px <= px[1] <= IMAGE_HEIGHT - margin_px
    )


def add_observed_point(recon: pycolmap.Reconstruction, image_id: int, xyz: np.ndarray) -> bool:
    """Projects xyz into image_id's real camera pose; if in-bounds, registers it as a real
    triangulated point3D observed by that image. Returns whether it was added."""
    image = recon.images[image_id]
    px = image.project_point(xyz)
    if px is None or not in_bounds(px):
        return False
    image.points2D.append(pycolmap.Point2D(xy=px))
    idx = len(image.points2D) - 1
    track = pycolmap.Track()
    track.add_element(image_id, idx)
    point3D_id = recon.add_point3D(xyz, track)
    image.set_point3D_for_point2D(idx, point3D_id)
    return True


def add_ground_plane(recon: pycolmap.Reconstruction, image_ids_by_x: dict) -> int:
    """Scatters real ground-level (z~0) points3D across the flight area, each attached to
    whichever frame is closest in x -- `_estimate_ground_z` reads the whole point cloud, not
    just a tree's own crown points, so these just need to exist and be near z=0."""
    n_added = 0
    xs = np.arange(-30.0, 46.0, 3.0)
    ys = np.arange(-12.0, 13.0, 4.0)
    for gx in xs:
        for gy in ys:
            nearest_frame_x = min(FRAME_X_POSITIONS_M, key=lambda fx: abs(fx - gx))
            image_id = image_ids_by_x[nearest_frame_x]
            z = float(RNG.normal(0.0, 0.05))
            if add_observed_point(recon, image_id, np.array([gx, gy, z])):
                n_added += 1
    # extra dense ground points right under each tree, so the 3.0m ground-search radius has
    # real nearby ground even though the broader grid above is spaced 3-4m apart
    for tree in TREES_GROUND_TRUTH:
        cx, cy = tree["xy"]
        for _ in range(10):
            gx = cx + RNG.uniform(-1.5, 1.5)
            gy = cy + RNG.uniform(-1.5, 1.5)
            nearest_frame_x = min(FRAME_X_POSITIONS_M, key=lambda fx: abs(fx - gx))
            image_id = image_ids_by_x[nearest_frame_x]
            z = float(RNG.normal(0.0, 0.05))
            if add_observed_point(recon, image_id, np.array([gx, gy, z])):
                n_added += 1
    return n_added


def add_crown_observations(
    recon: pycolmap.Reconstruction, image_ids_by_x: dict
) -> dict[int, list[CrownMask]]:
    crown_masks_by_image_id: dict[int, list[CrownMask]] = {}
    for tree in TREES_GROUND_TRUTH:
        cx, cy = tree["xy"]
        height = tree["height_m"]
        diameter = tree["diameter_m"]
        crown_top_xyz = np.array([cx, cy, height])

        for x in FRAME_X_POSITIONS_M:
            image_id = image_ids_by_x[x]
            image = recon.images[image_id]
            center_px = image.project_point(crown_top_xyz)
            if center_px is None or not in_bounds(center_px, margin_px=80.0):
                continue  # tree too close to this frame's edge to fit a whole crown mask

            # scatter a handful of real 3D points inside the crown volume, near its known top,
            # standing in for a dense-enough SfM cluster (the real 13-frame sequence's sparse
            # cluster is exactly the degenerate case this script is deliberately avoiding, to
            # isolate step 7's math from that separate, already-documented input-quality issue)
            n_added_for_this_frame = 0
            for _ in range(6):
                r = RNG.uniform(0.0, 0.25)
                theta = RNG.uniform(0.0, 2 * np.pi)
                pt = np.array(
                    [cx + r * np.cos(theta), cy + r * np.sin(theta), height - RNG.uniform(0.0, 0.3)]
                )
                if add_observed_point(recon, image_id, pt):
                    n_added_for_this_frame += 1
            if n_added_for_this_frame == 0:
                continue

            # reverse the diameter formula from `_crown_diameter_from_masks` to build a mask
            # whose pixel area encodes exactly `diameter`, given this frame's real camera
            # distance to the crown -- so a correct recovery is a real check, not circular
            distance_m = float(np.linalg.norm(image.projection_center() - crown_top_xyz))
            real_area_m2 = np.pi * (diameter / 2.0) ** 2
            area_px = real_area_m2 * (FOCAL_LENGTH_PX / distance_m) ** 2
            radius_px = int(round(np.sqrt(area_px / np.pi)))

            mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=bool)
            yy, xx = np.ogrid[:IMAGE_HEIGHT, :IMAGE_WIDTH]
            cx_px, cy_px = center_px
            mask[(xx - cx_px) ** 2 + (yy - cy_px) ** 2 <= radius_px**2] = True

            crown_masks_by_image_id.setdefault(image_id, []).append(
                CrownMask(
                    mask=mask,
                    score=0.9,
                    source_detection=CrownDetection(
                        box_xyxy=(
                            cx_px - radius_px,
                            cy_px - radius_px,
                            cx_px + radius_px,
                            cy_px + radius_px,
                        ),
                        score=0.9,
                        label="tree",
                    ),
                )
            )
    return crown_masks_by_image_id


def main() -> None:
    recon = pycolmap.Reconstruction()
    image_ids_by_x = build_nadir_camera_and_frames(recon)
    n_ground = add_ground_plane(recon, image_ids_by_x)
    crown_masks_by_image_id = add_crown_observations(recon, image_ids_by_x)
    n_crown_masks = sum(len(v) for v in crown_masks_by_image_id.values())

    print(f"Synthetic scene: {len(FRAME_X_POSITIONS_M)} frames, {n_ground} ground points, "
          f"{n_crown_masks} crown-mask observations across {len(TREES_GROUND_TRUTH)} trees.")
    print()

    estimates = estimate_tree_heights(recon, crown_masks_by_image_id)
    print(f"step 7 recovered {len(estimates)} tree cluster(s).\n")

    for tree in TREES_GROUND_TRUTH:
        cx, cy = tree["xy"]
        # match by centroid proximity, since cluster_id is arbitrary
        closest = min(estimates, key=lambda e: np.hypot(e.centroid_xy[0] - cx, e.centroid_xy[1] - cy))
        print(f"{tree['name']} (ground truth: height={tree['height_m']:.2f}m, "
              f"diameter={tree['diameter_m']:.2f}m, center=({cx:.1f},{cy:.1f}))")
        print(f"  recovered: height={closest.height_m:.2f}m "
              f"(crown_top_z={closest.crown_top_z:.2f}, ground_z={closest.ground_z:.2f}), "
              f"diameter={closest.crown_diameter_m:.2f}m, "
              f"centroid=({closest.centroid_xy[0]:.2f},{closest.centroid_xy[1]:.2f}), "
              f"num_points={closest.num_points}, num_observations={closest.num_observations}")
        height_err_pct = 100 * abs(closest.height_m - tree["height_m"]) / tree["height_m"]
        diam_err_pct = 100 * abs(closest.crown_diameter_m - tree["diameter_m"]) / tree["diameter_m"]
        print(f"  error: height {height_err_pct:.1f}%, diameter {diam_err_pct:.1f}%")
        print()


if __name__ == "__main__":
    main()
