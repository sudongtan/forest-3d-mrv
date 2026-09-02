"""Structure-from-motion: overlapping drone frames -> camera poses + a 3D point cloud.

Output is in COLMAP's own arbitrary local reference frame: unitless scale (a distance of "1.0"
is not 1 meter), arbitrary origin, arbitrary axis orientation -- see `scale_resolution.py` for
turning this into real metric scale, and `georeference.py` for turning that into a real-world CRS.
Don't use this reconstruction's raw coordinates for anything that needs to be a real measurement.
"""
from dataclasses import dataclass
from pathlib import Path

# Must import before pycolmap -- see CLAUDE.md Gotchas: pycolmap-then-torch (in that order, in
# the same process) crashes with SIGSEGV on this machine. Importing torch first, even unused,
# avoids it. Every module in this package that imports pycolmap does this for the same reason,
# so whichever one a caller happens to import first still gets the safe order.
import torch  # noqa: F401

import numpy as np
import pycolmap

# COLMAP's C++ core logs one line per SIFT thread + per-image at the default INFO level --
# useful on a terminal, overwhelming as library output. See notebooks/00_explore_colmap_open3d.
pycolmap.logging.minloglevel = 2  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL


@dataclass
class SfMResult:
    reconstruction: pycolmap.Reconstruction
    database_path: Path
    num_registered: int
    num_total: int
    num_points3D: int


def run_sfm(image_dir: Path | str, work_dir: Path | str) -> SfMResult:
    """Runs the real extract -> match -> incremental_mapping pipeline on `image_dir`'s frames.

    `image_dir` should hold a set of frames with genuine visual overlap (e.g. a consecutive
    run from one flight line) -- curated/non-sequential example frames will not register (see
    docs/lessons_learnt.md and docs/DATASETS.md for a confirmed real case of this failing).
    """
    image_dir = Path(image_dir)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(image_dir.glob("*.JPG")) + list(image_dir.glob("*.jpg"))
    if not image_paths:
        raise ValueError(f"No .JPG/.jpg images found in {image_dir}")

    db_path = work_dir / "database.db"
    pycolmap.extract_features(str(db_path), str(image_dir))
    pycolmap.match_exhaustive(str(db_path))

    sparse_dir = work_dir / "sparse"
    maps = pycolmap.incremental_mapping(str(db_path), str(image_dir), str(sparse_dir))
    if not maps:
        raise RuntimeError(
            f"COLMAP registered 0 images from {image_dir} ({len(image_paths)} input frames) -- "
            "check the frames actually have visual overlap (see docs/lessons_learnt.md for a "
            "real example of curated/non-sequential frames failing this way)."
        )

    reconstruction = max(maps.values(), key=lambda r: r.num_reg_images())
    return SfMResult(
        reconstruction=reconstruction,
        database_path=db_path,
        num_registered=reconstruction.num_reg_images(),
        num_total=len(image_paths),
        num_points3D=reconstruction.num_points3D(),
    )


@dataclass
class GravityAlignmentResult:
    up_direction_before: tuple[float, float, float]  # world-frame "up", before this ran
    num_cameras_used: int
    num_outliers_excluded: int


def align_gravity(reconstruction: pycolmap.Reconstruction, database: pycolmap.Database) -> GravityAlignmentResult:
    """Rotates `reconstruction` in place so world +Z is actually vertical ("up").

    **Do not assume `run_sfm`'s raw output already has a vertical Z axis just because each
    frame's EXIF/telemetry gravity prior was available during feature extraction.** Confirmed
    real, not hypothetical: on this project's own sample sequence, every single reconstructed 3D
    point came out with a *larger* Z than every camera center -- for a nadir (straight-down)
    drone shot, real canopy/ground points must be *below* the camera, so this was impossible, and
    tracing it back (transforming each image's own local gravity prior into the world frame via
    that image's pose) showed world "down" was actually close to world **+Y**, not +Z -- COLMAP's
    gravity priors refine bundle adjustment quality, they do not automatically orient the world
    frame's axes to match true vertical. `georeference.py`'s horizontal-only alignment strategy
    (deliberately not re-fitting a 3D rotation from GPS, since a single flight line is too
    close-to-colinear to fit one reliably) depends on Z already being vertical -- so this has to
    run first, on every reconstruction, not assumed away.

    Robust to a bad individual camera pose: excludes any camera whose local-gravity-in-world
    disagrees with the majority by more than 30 degrees before averaging (one such outlier was
    present in this project's own sample sequence).
    """
    priors = {p.corr_data_id.id: p for p in database.read_all_pose_priors() if p.has_gravity}

    world_down_vectors = []
    for image_id, image in reconstruction.images.items():
        prior = priors.get(image_id)
        if prior is None:
            continue
        local_gravity = np.array(prior.gravity)
        world_gravity = image.cam_from_world().rotation.inverse() * local_gravity
        world_down_vectors.append(world_gravity)

    if len(world_down_vectors) < 2:
        raise ValueError(
            f"Need at least 2 images with gravity priors to align orientation, got "
            f"{len(world_down_vectors)}"
        )
    world_down_vectors = np.array(world_down_vectors)

    median_down = np.median(world_down_vectors, axis=0)
    median_down /= np.linalg.norm(median_down)
    cos_sim = world_down_vectors @ median_down
    inliers = cos_sim > np.cos(np.radians(30))

    robust_down = world_down_vectors[inliers].mean(axis=0)
    robust_down /= np.linalg.norm(robust_down)
    up_direction = -robust_down

    rotation = _rotation_aligning_vectors(up_direction, np.array([0.0, 0.0, 1.0]))
    transform = pycolmap.Sim3d(
        scale=1.0, rotation=pycolmap.Rotation3d(rotation), translation=np.zeros(3)
    )
    reconstruction.transform(transform)

    return GravityAlignmentResult(
        up_direction_before=tuple(up_direction.tolist()),
        num_cameras_used=int(inliers.sum()),
        num_outliers_excluded=int((~inliers).sum()),
    )


def _rotation_aligning_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Returns the rotation matrix R such that R @ a == b, for unit vectors a, b (Rodrigues'
    rotation formula applied to the axis/angle between them).
    """
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    if s < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
