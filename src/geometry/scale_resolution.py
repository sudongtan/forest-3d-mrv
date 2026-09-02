"""Resolves SfM's arbitrary/unitless scale to real metric scale, using GPS/altitude priors
pycolmap already reads from each frame's EXIF during feature extraction.

**Do not skip this step.** `sfm.py`'s output is only correct up to an unknown similarity
transform -- a distance of "1.0" in its reconstruction could be 1 meter or 1 kilometer. Every
downstream measurement (canopy height, crown diameter, biomass) is meaningless without it.

Method: every registered image has both a GPS position (from EXIF, WGS84 lat/lon/alt) and an
SfM-recovered camera center (in COLMAP's arbitrary local frame). Project the GPS positions into
a local metric frame (an Azimuthal Equidistant projection centered on the data, valid for
small-area drone surveys), compute the distance between every pair of cameras in both the metric
GPS frame and the arbitrary SfM frame, and fit the ratio between them. That ratio is the scale
factor: multiply every SfM coordinate by it and the reconstruction is in real meters.
"""
from dataclasses import dataclass
from itertools import combinations

# Must import before pycolmap -- see CLAUDE.md Gotchas (pycolmap-then-torch crashes with SIGSEGV
# on this machine, order-dependent). See sfm.py for the full explanation.
import torch  # noqa: F401

import numpy as np
import pycolmap
import pyproj


@dataclass
class ScaleEstimate:
    scale_factor: float  # multiply SfM coordinates by this to get real meters
    num_pairs: int  # number of camera pairs the estimate was fit from
    ratio_std: float  # spread of per-pair ratios -- a rough precision indicator
    ratio_cv: float  # ratio_std / scale_factor -- coefficient of variation, scale-independent


def _camera_gps_positions(database: pycolmap.Database) -> dict[int, np.ndarray]:
    """Maps image_id -> (lat, lon, alt) as read from each frame's EXIF GPS tags."""
    priors = database.read_all_pose_priors()
    return {p.corr_data_id.id: np.array(p.position) for p in priors if p.has_position}


def _gps_to_local_metric(gps_by_id: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """WGS84 (lat, lon, alt) -> local (x, y, z) in meters, via an Azimuthal Equidistant
    projection centered on the data's centroid. Valid for the scale of one drone survey area;
    not intended as a general-purpose global projection.
    """
    lats = [g[0] for g in gps_by_id.values()]
    lons = [g[1] for g in gps_by_id.values()]
    lat0, lon0 = float(np.mean(lats)), float(np.mean(lons))

    aeqd = pyproj.CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +ellps=WGS84")
    transformer = pyproj.Transformer.from_crs("EPSG:4326", aeqd, always_xy=True)

    local = {}
    for image_id, (lat, lon, alt) in gps_by_id.items():
        x, y = transformer.transform(lon, lat)
        local[image_id] = np.array([x, y, alt])
    return local


def _pairwise_distances(
    reconstruction: pycolmap.Reconstruction,
    metric_pos_by_id: dict[int, np.ndarray],
    image_ids: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (sfm_distances, metric_distances) arrays, one entry per camera pair, restricted
    to `image_ids` if given (used to hold out a subset for validation).
    """
    sfm_center_by_id = {
        image_id: image.cam_from_world().inverse().translation
        for image_id, image in reconstruction.images.items()
    }
    ids = image_ids if image_ids is not None else list(sfm_center_by_id.keys())
    ids = [i for i in ids if i in sfm_center_by_id and i in metric_pos_by_id]

    sfm_dists, metric_dists = [], []
    for id_a, id_b in combinations(ids, 2):
        sfm_dists.append(np.linalg.norm(sfm_center_by_id[id_a] - sfm_center_by_id[id_b]))
        metric_dists.append(np.linalg.norm(metric_pos_by_id[id_a] - metric_pos_by_id[id_b]))
    return np.array(sfm_dists), np.array(metric_dists)


def estimate_scale(
    reconstruction: pycolmap.Reconstruction,
    database: pycolmap.Database,
    image_ids: list[int] | None = None,
) -> ScaleEstimate:
    """Estimates the SfM-arbitrary-units -> meters scale factor from GPS priors.

    `image_ids`, if given, restricts the fit to only those images' pairwise distances -- used by
    the round-trip test to fit on a subset and validate against the rest.
    """
    gps_by_id = _camera_gps_positions(database)
    if len(gps_by_id) < 2:
        raise ValueError(
            f"Need at least 2 images with GPS priors to estimate scale, got {len(gps_by_id)}"
        )
    metric_pos_by_id = _gps_to_local_metric(gps_by_id)
    sfm_dists, metric_dists = _pairwise_distances(reconstruction, metric_pos_by_id, image_ids)
    if len(sfm_dists) == 0:
        raise ValueError("No overlapping camera pairs between the reconstruction and GPS priors")

    ratios = metric_dists / sfm_dists
    scale_factor = float(np.median(ratios))
    ratio_std = float(ratios.std())
    return ScaleEstimate(
        scale_factor=scale_factor,
        num_pairs=len(ratios),
        ratio_std=ratio_std,
        ratio_cv=ratio_std / scale_factor if scale_factor else float("nan"),
    )


def apply_scale(reconstruction: pycolmap.Reconstruction, scale_factor: float) -> None:
    """Rescales `reconstruction`'s camera poses and point cloud in place -- pure scale, no
    rotation or translation (that's `georeference.py`'s job, once a target CRS is chosen).
    """
    transform = pycolmap.Sim3d(
        scale=scale_factor, rotation=pycolmap.Rotation3d(), translation=np.zeros(3)
    )
    reconstruction.transform(transform)
