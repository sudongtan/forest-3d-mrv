"""Round-trip test for geometry/scale_resolution.py: fit the SfM-to-metric scale factor on most
of a real reconstruction's camera pairs, then check it correctly predicts the real-world (GPS)
distance between camera pairs it never saw -- the guardrail this project's plan explicitly asks
for, since an unvalidated scale factor makes every downstream height/biomass number meaningless.

Needs a real overlapping frame sequence, which is a gitignored sample (~106MB, not committed) --
see docs/DATASETS.md for how to reproduce it. Skips (not fails) if that data isn't present, so a
fresh clone or CI run doesn't break without ever being asked to fetch drone imagery.
"""
from pathlib import Path

import numpy as np
import pycolmap
import pytest

from src.geometry.scale_resolution import (
    _camera_gps_positions,
    _gps_to_local_metric,
    _pairwise_distances,
    estimate_scale,
)
from src.geometry.sfm import run_sfm

SAMPLE_DIR = Path("data/samples/ofo_mission_000001_sequence")

pytestmark = pytest.mark.skipif(
    not SAMPLE_DIR.exists(),
    reason=f"real drone frame sample not present at {SAMPLE_DIR} -- see docs/DATASETS.md",
)


@pytest.fixture(scope="module")
def sfm_result(tmp_path_factory):
    work_dir = tmp_path_factory.mktemp("sfm_scale_test")
    return run_sfm(SAMPLE_DIR, work_dir)


def test_scale_estimate_is_precise_and_consistent(sfm_result):
    db = pycolmap.Database.open(str(sfm_result.database_path))
    estimate = estimate_scale(sfm_result.reconstruction, db)

    assert estimate.num_pairs > 10
    # coefficient of variation across all camera-pair ratios should be tight if the
    # reconstruction and GPS priors genuinely agree on relative geometry -- loose on real
    # OFO data (frames 100-112, mission 000001): observed ~0.6%, so 5% leaves real margin
    # while still catching a badly broken estimate.
    assert estimate.ratio_cv < 0.05


def test_scale_round_trips_to_held_out_camera_pairs(sfm_result):
    """Fit the scale factor on 10 of 13 cameras, then verify it correctly predicts real-world
    (GPS-derived) distances for pairs involving the 3 held-out cameras it never saw.
    """
    db = pycolmap.Database.open(str(sfm_result.database_path))
    all_ids = list(sfm_result.reconstruction.images.keys())
    assert len(all_ids) >= 6, "need enough registered images for a meaningful held-out split"

    rng = np.random.default_rng(0)
    held_out_ids = sorted(rng.choice(all_ids, size=3, replace=False).tolist())
    fit_ids = [i for i in all_ids if i not in held_out_ids]

    estimate = estimate_scale(sfm_result.reconstruction, db, image_ids=fit_ids)

    gps_by_id = _camera_gps_positions(db)
    metric_pos_by_id = _gps_to_local_metric(gps_by_id)
    sfm_dists, metric_dists = _pairwise_distances(
        sfm_result.reconstruction, metric_pos_by_id, image_ids=held_out_ids
    )
    assert len(sfm_dists) > 0, "no held-out pairs to validate against"

    predicted_metric_dists = sfm_dists * estimate.scale_factor
    relative_error = np.abs(predicted_metric_dists - metric_dists) / metric_dists

    # observed max ~0.66% on real OFO data; 5% is a real, meaningful bound, not a rubber stamp
    assert relative_error.max() < 0.05, (
        f"scale factor fit on {len(fit_ids)} cameras predicted held-out camera-pair distances "
        f"with {relative_error.max():.1%} max relative error (expected < 5%)"
    )
