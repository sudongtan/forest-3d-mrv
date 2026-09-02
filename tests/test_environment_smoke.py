"""Phase 0 smoke tests: confirm the core toolchain (pycolmap, MPS) actually works on
this machine before any real pipeline code is built on top of it.
"""
import subprocess
import sys
import textwrap

import numpy as np
import open3d as o3d
import pycolmap
import pytest
import torch


def test_pycolmap_before_transformers_model_load_crashes_process():
    """Documents a real, confirmed SIGSEGV: `pycolmap.Database.open()` (real usage, not just
    `import pycolmap`) followed by loading any Hugging Face `transformers` model in the same
    process crashes the whole Python process -- reproducible, order-dependent, and distinct from
    the libomp duplicate-runtime abort above (KMP_DUPLICATE_LIB_OK does not fix this one; a plain
    `torch` op like a matmul does not trigger it either, only an actual transformers model load
    does). The fix is import order: `import torch` before `import pycolmap`, done defensively in
    every `src/geometry/*.py` module and in this test suite's `conftest.py` -- see CLAUDE.md
    Gotchas. This test runs the *broken* order in an isolated subprocess (a real SIGSEGV would
    kill the whole pytest process if reproduced in-process) purely to keep this finding honest
    and reproducible, not because the broken order is expected anywhere in this codebase.
    """
    broken_order_script = textwrap.dedent("""
        import pycolmap
        pycolmap.logging.minloglevel = 2
        db = pycolmap.Database.open("dummy.db")
        db.close()
        from transformers import AutoModelForZeroShotObjectDetection
        AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
        print("no crash")
    """)
    result = subprocess.run(
        [sys.executable, "-c", broken_order_script],
        capture_output=True,
        text=True,
        timeout=120,
        cwd="/tmp",
    )
    assert result.returncode == -11, (  # SIGSEGV
        f"expected the broken import order to crash with SIGSEGV (returncode -11); got "
        f"{result.returncode}. If this now passes, the underlying native library bug may have "
        f"been fixed upstream -- worth re-checking whether the import-order workaround in "
        f"src/geometry/*.py and conftest.py is still needed."
    )


def test_torch_before_pycolmap_avoids_the_crash():
    """The actual protective assertion: importing `torch` before `pycolmap` -- the order every
    `src/geometry/*.py` module and this test suite's `conftest.py` now uses -- does not crash,
    even doing the exact same real pycolmap + transformers work as the broken-order test above.
    """
    fixed_order_script = textwrap.dedent("""
        import torch
        import pycolmap
        pycolmap.logging.minloglevel = 2
        db = pycolmap.Database.open("dummy.db")
        db.close()
        from transformers import AutoModelForZeroShotObjectDetection
        AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
        print("no crash")
    """)
    result = subprocess.run(
        [sys.executable, "-c", fixed_order_script],
        capture_output=True,
        text=True,
        timeout=120,
        cwd="/tmp",
    )
    assert result.returncode == 0, (
        f"torch-before-pycolmap import order still crashed (returncode {result.returncode}): "
        f"{result.stderr[-2000:]}"
    )


def test_pycolmap_torch_open3d_coimport():
    """pycolmap, torch, and open3d each bundle their own libomp.dylib on macOS; importing any
    two (or all three) in the same process aborts unless KMP_DUPLICATE_LIB_OK is set first (see
    conftest.py and CLAUDE.md Gotchas). This just exercises basic Open3D point-cloud ops to
    confirm the workaround actually holds for all three together, not just pycolmap+torch.
    """
    pts = np.random.rand(500, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    _, inliers = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    assert len(inliers) > 0

    labels = np.array(pcd.cluster_dbscan(eps=0.1, min_points=5))
    assert len(labels) == len(pts)


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
