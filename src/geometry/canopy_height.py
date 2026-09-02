"""Combines a metric-scale point cloud (`sfm.py` + `scale_resolution.py`) with per-frame crown
masks (`perception/crown_segmenter.py`) into one height estimate per physical tree.

Cross-frame crown identity is resolved *here*, geometrically -- not by inheriting SAM2's 2D
tracking IDs from Phase 1 (see CLAUDE.md's Phase 1/2 split and why). For every frame's crown
mask, this module looks up which of that frame's already-triangulated 2D keypoints fall inside
the mask, and takes their corresponding 3D points as "this crown observation's points". The same
physical tree, seen and masked in several different frames, produces several such observations
whose 3D points sit in the same place in space -- so pooling every observation's points together
and spatially clustering (Open3D DBSCAN) naturally merges them into one cluster per tree, with no
need to match crown IDs across frames at all.

Ground elevation is estimated with a simple local-minimum heuristic (nearest points to a tree's
footprint, in the full point cloud, not just its own crown points) -- not a proper DTM/ground
classification. How good or bad that heuristic actually is is exactly what
`notebooks/00_sfm_scale_validation.ipynb`'s comparison against real LiDAR CHM is for; this module
doesn't try to pre-judge that, per CLAUDE.md's guardrail to validate before building further on
top of an unvalidated heuristic.
"""
from dataclasses import dataclass

# Must import before pycolmap -- importing pycolmap first, then later using a torch-based model
# in the same process, crashes with SIGSEGV on this machine (order-dependent, not merely a
# co-import issue -- see CLAUDE.md Gotchas). `CrownMask` itself pulls in torch transitively via
# crown_segmenter.py, but that happens too late if pycolmap is imported above it, so torch is
# imported directly and first, here, rather than relying on import order got right by accident.
import torch  # noqa: F401

import numpy as np
import open3d as o3d
import pycolmap

from src.perception.crown_segmenter import CrownMask


@dataclass
class TreeHeightEstimate:
    cluster_id: int
    num_points: int
    num_observations: int  # how many (frame, crown mask) pairs contributed to this tree
    contributing_image_ids: list[int]  # which frames saw this tree -- lets a caller anchor its
    # real-world position to those cameras' own GPS rather than trust a whole-reconstruction
    # rigid-fit alignment over long range (see docs/lessons_learnt.md)
    centroid_xy: tuple[float, float]
    crown_top_z: float
    ground_z: float
    height_m: float
    crown_diameter_m: float  # median, across this tree's contributing (frame, mask)
    # observations, of a mask-area-based diameter estimate -- see `_crown_diameter_from_masks`
    # docstring for the two real bugs found and fixed getting here (sparse-3D-point clusters,
    # then Z-axis tilt corruption). **A third, still-open issue found while validating this on
    # real data, not yet fixed**: on this project's actual 13-frame mission, some of
    # `perception/crown_detector.py`/`crown_segmenter.py`'s real detected "crown" masks are
    # themselves too large -- spanning several adjacent trees' combined canopy rather than one
    # tree (mask sizes on this mission range from ~30K px, giving plausible ~6-7m diameters, up
    # to 2.5M+ px, giving obviously-wrong 20m+ diameters). Whichever mask a tree's observations
    # happen to draw from, this function has no way to tell "one real crown" from "a multi-tree
    # blob" -- see docs/lessons_learnt.md's biomass entry. Values from this field should be
    # treated the same way Phase 2 initially treated pre-LiDAR-validation heights: runs without
    # error, not yet trustworthy as an accurate per-tree number.


def _crown_observation_points(
    reconstruction: pycolmap.Reconstruction,
    crown_masks_by_image_id: dict[int, list[CrownMask]],
) -> tuple[np.ndarray, np.ndarray, dict[int, int], dict[int, int]]:
    """For every (frame, crown mask) pair, finds the frame's already-triangulated 2D keypoints
    that fall inside the mask, and returns their corresponding 3D points.

    Returns (points_xyz, observation_id_per_point, image_id_by_observation_id,
    mask_area_px_by_observation_id) -- the second array lets you trace each pooled point back to
    which single (frame, crown) observation it came from; the third maps that observation back to
    which frame it came from (used to anchor a tree's position to its contributing cameras' own
    GPS -- see module docstring); the fourth is that observation's own crown mask's pixel area
    (used by `_crown_diameter_from_masks`, since the mask's real-world extent -- not the sparse
    3D points inside it -- is what actually carries crown-diameter information; see that
    function's docstring for why).
    """
    all_points = []
    observation_ids = []
    image_id_by_observation_id = {}
    mask_area_px_by_observation_id = {}
    observation_id = 0

    for image_id, crown_masks in crown_masks_by_image_id.items():
        image = reconstruction.images[image_id]
        triangulated = [p2d for p2d in image.points2D if p2d.has_point3D()]

        for crown_mask in crown_masks:
            h, w = crown_mask.mask.shape
            for p2d in triangulated:
                x, y = int(round(p2d.xy[0])), int(round(p2d.xy[1]))
                if 0 <= x < w and 0 <= y < h and crown_mask.mask[y, x]:
                    if p2d.point3D_id in reconstruction.points3D:
                        point3D = reconstruction.points3D[p2d.point3D_id]
                        all_points.append(point3D.xyz)
                        observation_ids.append(observation_id)
            image_id_by_observation_id[observation_id] = image_id
            mask_area_px_by_observation_id[observation_id] = int(crown_mask.mask.sum())
            observation_id += 1

    if not all_points:
        return np.empty((0, 3)), np.empty((0,), dtype=int), {}, mask_area_px_by_observation_id
    return (
        np.array(all_points),
        np.array(observation_ids),
        image_id_by_observation_id,
        mask_area_px_by_observation_id,
    )


def _crown_diameter_from_masks(
    reconstruction: pycolmap.Reconstruction,
    observation_ids: set[int],
    image_id_by_observation_id: dict[int, int],
    mask_area_px_by_observation_id: dict[int, int],
    crown_point_xyz: np.ndarray,
) -> float:
    """Real-world crown diameter, estimated from each contributing observation's 2D mask area --
    not from the tree's own resolved 3D point cluster.

    **Why not the 3D points**: an earlier version of this function used `2 * mean(horizontal
    distance from each cluster point to the centroid)`. On this project's real 13-frame
    validation sequence, that gave a crown diameter of *exactly 0.00m* for 10 of 13 trees. Root
    cause, confirmed by inspecting the raw clusters: sparse SfM (COLMAP extracts and matches
    discrete SIFT-like features, not a dense per-pixel depth map) finds very few visually
    distinctive, repeatably-matchable features per tree crown -- foliage is self-similar texture,
    which SIFT-style matching struggles with -- so most crown clusters here resolve to *one*
    real, unique 3D point, observed from several frames (hence 5-15 "points" in a cluster that
    are almost all exact duplicates of that single coordinate). One point has zero spread by
    definition; the height computation (which only needs that one point's Z) still works, but a
    diameter computed from its horizontal spread cannot. See docs/lessons_learnt.md for the full
    diagnosis (`unique_xyz` counts from the real data).

    **The fix, and a second real bug found while building it**: a crown mask's *pixel area*
    carries real information about the crown's true size regardless of how many distinct 3D
    features SfM found inside it. Converting that pixel area to a real-world area needs the
    camera-to-crown viewing distance and the camera's focal length:
    real_area_m2 = mask_area_px * (distance_m / focal_length_px)^2, then
    diameter = 2 * sqrt(real_area_m2 / pi) (treating the mask footprint as circular). The first
    attempt at `distance_m` used the axis-aligned drop `camera_Z - crown_top_z` -- and on this
    project's real data, that came out *negative* (camera appearing below the crown) for most
    observations, silently discarding them. Root cause: this is the exact same residual-tilt
    problem `canopy_height.py`'s module docstring already documents for ground_z/height
    (`align_gravity`'s fitted "up" is only approximately correct, so real elevation increasingly
    leaks into apparent horizontal position -- and vice versa -- the further two points are
    apart) -- but here it bit harder, because a camera can be tens of meters horizontally from
    the tree it photographed, far outside the small local neighborhoods `_estimate_ground_z`
    already restricts itself to. Confirmed directly: for one real tree, `camera_Z - crown_top_z`
    ranged from -25.8 to -2.4 (all negative, all discarded) across its 5 contributing frames,
    while the camera-to-tree **3D Euclidean distance** for those same 5 frames was a tight,
    physically-plausible 74.5-78.0m -- consistent with this site's documented ~60-100m flight
    altitude (see `geometry/depth_fallback.py`'s lessons-learnt entry). Euclidean distance is
    rotation-invariant (a rigid rotation preserves all pairwise distances exactly), so it isn't
    corrupted by residual tilt the way a single axis's difference is -- **use it instead of the
    Z-only drop**, even though it's a mild over-approximation of true viewing distance for a
    slightly-oblique (non-exactly-nadir) shot. Takes the median across all of this tree's
    contributing observations, for robustness to any one frame's mask being an unusually
    oblique or partially-occluded view of the crown.
    """
    diameters = []
    for obs_id in observation_ids:
        image_id = image_id_by_observation_id[obs_id]
        image = reconstruction.images[image_id]
        camera = image.camera
        distance_m = float(np.linalg.norm(np.array(image.projection_center()) - crown_point_xyz))
        if distance_m < 1.0:
            continue  # a real drone shot is never this close -- treat as a degenerate pose/point
            # rather than let it blow up the area estimate
        area_px = mask_area_px_by_observation_id[obs_id]
        real_area_m2 = area_px * (distance_m / camera.focal_length) ** 2
        diameters.append(2 * (real_area_m2 / np.pi) ** 0.5)

    if not diameters:
        return 0.0
    return float(np.median(diameters))


def _estimate_ground_z(
    full_point_cloud_xyz: np.ndarray, query_xy: np.ndarray, radius_m: float
) -> float:
    """Ground-height heuristic: the lowest z among all reconstructed points (not just this
    tree's own crown points -- a tree's crown observations rarely include ground-level points at
    all) within `radius_m` of `query_xy`. A real DTM/ground-classification step would do better;
    this is deliberately simple, see module docstring.
    """
    xy = full_point_cloud_xyz[:, :2]
    dists = np.linalg.norm(xy - query_xy, axis=1)
    nearby_z = full_point_cloud_xyz[dists < radius_m, 2]
    if len(nearby_z) == 0:
        # fall back to the whole cloud if nothing is within radius (e.g. a tree at the edge
        # of the reconstruction) -- better than crashing, but this tree's height should be
        # treated as lower-confidence.
        nearby_z = full_point_cloud_xyz[:, 2]
    return float(np.percentile(nearby_z, 5))  # 5th percentile, not min: robust to a stray
    # low outlier point that survived cleaning


def estimate_tree_heights(
    reconstruction: pycolmap.Reconstruction,
    crown_masks_by_image_id: dict[int, list[CrownMask]],
    dbscan_eps_m: float = 0.5,
    dbscan_min_points: int = 5,
    ground_search_radius_m: float = 3.0,
) -> list[TreeHeightEstimate]:
    """`reconstruction` must already be metric-scaled (see scale_resolution.apply_scale) --
    `dbscan_eps_m` and `ground_search_radius_m` are meaningless otherwise.

    **These defaults are small on purpose, not arbitrary.** `align_gravity` (sfm.py) fixes world
    Z to be *approximately* vertical, from a robust average of noisy per-shot gravity priors --
    it is not exact, and the residual tilt means real elevation increasingly leaks into apparent
    horizontal position (and vice versa) the further two points are apart. Swept both parameters
    on this project's real 13-frame validation sequence: `eps=0.5`/`radius=3.0` gave physically
    plausible heights (0.1-10.7m); `eps=1.5`/`radius=5.0` gave 3-80m, including obviously
    impossible values for this site. Larger values aren't "more thorough", they pull in points
    from far enough away that residual tilt has accumulated into real error. See
    `docs/lessons_learnt.md` ("residual orientation imprecision corrupts height") for the full
    sweep and diagnosis. Revisit these defaults if `align_gravity`'s fit precision improves.
    """
    obs_points, observation_ids, image_id_by_obs, mask_area_px_by_obs = _crown_observation_points(
        reconstruction, crown_masks_by_image_id
    )
    if len(obs_points) == 0:
        return []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obs_points)
    cleaned_pcd, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    cleaned_points = np.asarray(cleaned_pcd.points)
    cleaned_observation_ids = observation_ids[inlier_idx]

    labels = np.array(cleaned_pcd.cluster_dbscan(eps=dbscan_eps_m, min_points=dbscan_min_points))

    full_cloud_xyz = np.array([p.xyz for p in reconstruction.points3D.values()])

    estimates = []
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:  # DBSCAN noise label, not a real tree
            continue
        cluster_mask = labels == cluster_id
        cluster_points = cleaned_points[cluster_mask]
        centroid_xy = cluster_points[:, :2].mean(axis=0)
        crown_top_z = float(cluster_points[:, 2].max())
        ground_z = _estimate_ground_z(full_cloud_xyz, centroid_xy, ground_search_radius_m)
        cluster_obs_ids = set(cleaned_observation_ids[cluster_mask].tolist())
        crown_point_xyz = np.array([centroid_xy[0], centroid_xy[1], crown_top_z])
        crown_diameter_m = _crown_diameter_from_masks(
            reconstruction, cluster_obs_ids, image_id_by_obs, mask_area_px_by_obs, crown_point_xyz
        )

        estimates.append(
            TreeHeightEstimate(
                cluster_id=int(cluster_id),
                num_points=int(cluster_mask.sum()),
                num_observations=len(cluster_obs_ids),
                contributing_image_ids=sorted({image_id_by_obs[o] for o in cluster_obs_ids}),
                centroid_xy=(float(centroid_xy[0]), float(centroid_xy[1])),
                crown_top_z=crown_top_z,
                ground_z=ground_z,
                height_m=crown_top_z - ground_z,
                crown_diameter_m=crown_diameter_m,
            )
        )
    return estimates
