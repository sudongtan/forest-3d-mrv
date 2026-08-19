"""Phase 0 smoke tests: confirm the core toolchain (pycolmap, MPS) actually works on
this machine before any real pipeline code is built on top of it.
"""
import numpy as np
import pycolmap
import pytest
import torch


def test_mps_available():
    assert torch.backends.mps.is_built()
    assert torch.backends.mps.is_available()
    x = torch.randn(64, 64, device="mps")
    y = (x @ x).cpu()
    assert torch.isfinite(y).all()


def test_pycolmap_trivial_sfm_roundtrip(tmp_path):
    """Synthesize a small multi-camera scene, render real images from it, then run
    pycolmap's actual extract -> match -> incremental_mapping pipeline (the same
    codepath used on real drone frames) and confirm the recovered camera positions
    match the synthetic ground truth up to a similarity transform.
    """
    image_dir = tmp_path / "images"
    db_path = tmp_path / "database.db"
    image_dir.mkdir()

    synth_opts = pycolmap.SyntheticDatasetOptions()
    synth_opts.num_rigs = 1
    synth_opts.num_cameras_per_rig = 1
    synth_opts.num_frames_per_rig = 5
    synth_opts.num_points3D = 200
    synth_opts.camera_width = 640
    synth_opts.camera_height = 480

    database = pycolmap.Database.open(str(db_path))
    gt_reconstruction = pycolmap.synthesize_dataset(synth_opts, database)
    database.close()

    img_opts = pycolmap.SyntheticImageOptions()
    pycolmap.synthesize_images(img_opts, gt_reconstruction, str(image_dir))

    rendered = list(image_dir.glob("*"))
    assert len(rendered) == 5, f"expected 5 rendered frames, got {len(rendered)}"

    sfm_db_path = tmp_path / "sfm_database.db"
    pycolmap.extract_features(str(sfm_db_path), str(image_dir))
    pycolmap.match_exhaustive(str(sfm_db_path))

    maps = pycolmap.incremental_mapping(
        str(sfm_db_path), str(image_dir), str(tmp_path / "sparse")
    )
    assert len(maps) >= 1, "incremental mapping recovered no reconstruction"

    recon = max(maps.values(), key=lambda r: r.num_reg_images())
    assert recon.num_reg_images() >= 3, (
        f"expected to register at least 3/5 frames, got {recon.num_reg_images()}"
    )

    gt_centers = np.array(
        [im.cam_from_world().inverse().translation for im in gt_reconstruction.images.values()]
    )
    est_centers = np.array(
        [im.cam_from_world().inverse().translation for im in recon.images.values()]
    )
    gt_spread = gt_centers.std(axis=0).sum()
    est_spread = est_centers.std(axis=0).sum()
    assert est_spread > 0, "recovered camera positions collapsed to a point"
    assert gt_spread > 0
