"""Converts a metric-scaled local reconstruction (`sfm.py` + `scale_resolution.py`) into a real
projected CRS (e.g. UTM), so its coordinates can be directly overlaid against other georeferenced
data -- a LiDAR canopy height model, an orthomosaic, a ground-inventory shapefile.

**Deliberately horizontal-only: yaw rotation (about the vertical axis) + XY translation + a
simple Z offset -- not a general 3D rotation.** Tried the general approach first
(`pycolmap.align_reconstruction_to_locations`, a RANSAC-robust full Sim3d fit) and it produced a
badly wrong result: fitted a rotation that mapped "up" to mostly sideways (confirmed by
inspecting the rotation matrix -- the row that should read close to [0,0,1] read
[-0.98,-0.05,-0.20]). Root cause, confirmed via SVD of the camera positions: a single drone
flight line's cameras are nearly colinear (measured spread ratio ~0.004 between the smallest and
largest principal axis on the real sequence this was built against) -- a severely
under-constrained configuration for fitting a full 3D rotation from point correspondences alone,
even with RANSAC. Sound for a multi-line grid survey with real 3D spread; not sound for a single
strip, which is exactly what small validation sequences like this project's sample data are.

Solving only for yaw + XY translation + Z offset sidesteps the degeneracy entirely, and leans on
something already true rather than re-deriving it from noisy data: `sfm.py`'s reconstruction
already has a Z axis reasonably close to true vertical, because COLMAP's bundle adjustment used
each frame's gravity prior (confirmed present in the feature-extraction logs) -- so there's no
need to re-fit vertical orientation from GPS at all, only horizontal position and heading.
"""
from dataclasses import dataclass

# Must import before pycolmap -- see CLAUDE.md Gotchas (pycolmap-then-torch crashes with SIGSEGV
# on this machine, order-dependent). See sfm.py for the full explanation.
import torch  # noqa: F401

import numpy as np
import pycolmap
import pyproj


@dataclass
class GeoreferenceResult:
    transform: pycolmap.Sim3d
    crs: str  # e.g. "EPSG:6339"
    horizontal_rmse_m: float  # residual after fitting yaw + XY translation -- fit quality
    z_offset_std_m: float  # spread of the per-camera Z offsets the single Z translation summarizes
    num_images_used: int


def _fit_horizontal_similarity(
    src_xy: np.ndarray, tgt_xy: np.ndarray
) -> tuple[float, np.ndarray]:
    """2D Kabsch/Procrustes: fits a rotation angle (about Z) + XY translation, scale fixed at 1
    (already resolved by scale_resolution.py). Returns (angle_radians, translation_xy).
    """
    src_centroid, tgt_centroid = src_xy.mean(axis=0), tgt_xy.mean(axis=0)
    src_centered, tgt_centered = src_xy - src_centroid, tgt_xy - tgt_centroid

    h = src_centered.T @ tgt_centered
    u, _, vt = np.linalg.svd(h)
    rotation_2d = vt.T @ u.T
    if np.linalg.det(rotation_2d) < 0:  # reflection, not a rotation -- flip to the closest proper one
        vt[-1, :] *= -1
        rotation_2d = vt.T @ u.T

    angle = float(np.arctan2(rotation_2d[1, 0], rotation_2d[0, 0]))
    translation_xy = tgt_centroid - rotation_2d @ src_centroid
    return angle, translation_xy


def georeference(
    reconstruction: pycolmap.Reconstruction,
    database: pycolmap.Database,
    target_crs: str = "EPSG:6339",  # NAD83(2011) UTM Zone 10N -- confirmed from the actual
    # downloaded USGS 3DEP LAZ header for this site (data/lidar/), not assumed
    min_images: int = 3,
) -> GeoreferenceResult:
    """Aligns `reconstruction` in place to `target_crs` using each image's own GPS prior.
    `reconstruction` should already be metric-scaled (see scale_resolution.py).
    """
    priors = database.read_all_pose_priors()
    id_to_name = {im.image_id: im.name for im in database.read_all_images()}
    name_to_sfm_center = {
        im.name: im.cam_from_world().inverse().translation for im in reconstruction.images.values()
    }

    transformer = pyproj.Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

    src_xyz, tgt_xyz = [], []
    for prior in priors:
        if not prior.has_position:
            continue
        name = id_to_name.get(prior.corr_data_id.id)
        if name is None or name not in name_to_sfm_center:
            continue
        lat, lon, alt = prior.position
        x, y = transformer.transform(lon, lat)
        tgt_xyz.append([x, y, alt])
        src_xyz.append(name_to_sfm_center[name])

    if len(src_xyz) < min_images:
        raise ValueError(f"Only {len(src_xyz)} images have both GPS priors and a pose; need at "
                          f"least {min_images}")

    src_xyz, tgt_xyz = np.array(src_xyz), np.array(tgt_xyz)

    angle, translation_xy = _fit_horizontal_similarity(src_xyz[:, :2], tgt_xyz[:, :2])
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_3d = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

    predicted_xy = (rotation_3d[:2, :2] @ src_xyz[:, :2].T).T + translation_xy
    horizontal_rmse = float(np.sqrt(np.mean(np.sum((predicted_xy - tgt_xyz[:, :2]) ** 2, axis=1))))

    per_camera_z_offset = tgt_xyz[:, 2] - src_xyz[:, 2]
    z_offset = float(per_camera_z_offset.mean())
    z_offset_std = float(per_camera_z_offset.std())

    transform = pycolmap.Sim3d(
        scale=1.0,
        rotation=pycolmap.Rotation3d(rotation_3d),
        translation=np.array([translation_xy[0], translation_xy[1], z_offset]),
    )
    reconstruction.transform(transform)

    return GeoreferenceResult(
        transform=transform,
        crs=target_crs,
        horizontal_rmse_m=horizontal_rmse,
        z_offset_std_m=z_offset_std,
        num_images_used=len(src_xyz),
    )
