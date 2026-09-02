"""Phase 3 end-to-end spot-check: real drone frames -> SfM -> gravity alignment -> metric scale ->
crown detection/segmentation -> tree heights + crown diameters -> AGB -> CO2e, on the same real
site used for Phase 2's LiDAR validation (data/samples/ofo_mission_000001_sequence/).

This is the "does this actually run on real pipeline output" check for `src/biomass/allometry.py`
and `src/biomass/carbon.py` -- not a pytest test (those live in tests/test_allometry.py and
tests/test_carbon.py and check the equations themselves against hand-computed values). Produces a
scene_state.json-shaped `trees[]` list with biomass_kg/co2e_kg populated, per Phase 3's "Done
when" criterion in CLAUDE.md -- the full reporting/scene_state.py assembly module is Phase 6's
job, not built here.

No tree in this sample has a species-survey record, so every tree uses allometry.py's regional
fallback (WSG for Chave, functional type for Jucker) -- and every tree uses the crown-diameter
path (Jucker et al. 2017), not the DBH path (Chave et al. 2014), because DBH isn't observable
from this nadir drone imagery. See allometry.py's module docstring for why both models exist.

**Read this before citing any number this script prints.** Building and running this end-to-end
surfaced three real bugs in `geometry/canopy_height.py`'s crown-diameter estimation, in order --
full diagnosis of each in that module's docstrings and docs/lessons_learnt.md:
  1. An initial attempt derived crown diameter from the tree's own resolved 3D point cluster's
     horizontal spread -- came out *exactly 0.00m* for 10 of 13 real trees, because sparse SfM
     resolves most crowns to a single duplicated 3D point, which has no spread by definition.
  2. The fix (deriving diameter from each contributing crown mask's real-world pixel area
     instead) initially computed camera-to-crown viewing distance as the axis-aligned
     `camera_Z - crown_top_Z` -- came out *negative* (camera below the crown) for most
     observations on real data, because of the exact same residual-orientation-tilt problem
     Phase 2 already documented for height/ground_z, here biting harder since camera-to-tree
     pairs can be tens of meters apart. Fixed by switching to 3D Euclidean distance (rotation-
     invariant, so immune to this).
  3. **Still open, not fixed here**: even with (1) and (2) fixed, several trees in this real
     sample get implausibly large crown diameters (~19-22m -- several times a real conifer's
     actual crown width) because some of Phase 1's real crown masks on this specific mission's
     frames span multiple adjacent trees' combined canopy, not one tree. This script has no way
     to detect that from within the biomass step -- it's a Phase 1/2 segmentation-granularity
     gap, not something an allometric equation can correct for. **AGB/CO2e numbers for any tree
     this script flags below are not yet trustworthy point estimates** -- the pipeline runs
     end-to-end and the *equations* are real, cited, and unit-tested (see
     tests/test_allometry.py, tests/test_carbon.py), but this specific real-data run is not
     Phase 3's equivalent of Phase 2's LiDAR-validated result. Left honest and open rather than
     quietly patched, per this project's own guardrails.

Per CLAUDE.md's guardrail ("Propagate Phase 2's height error into a biomass uncertainty range"),
each tree's final uncertainty combines two independent sources in quadrature: Jucker's own
20.6% CV, and Phase 2's measured LiDAR validation error (AbsRel ~65%, see
notebooks/00_sfm_scale_validation.ipynb) as a proxy for how wrong the height input itself could
be -- not just how wrong the AGB model is conditional on a correct height.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE uv run python scripts/spot_check_biomass.py
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Must be the literal first import -- see CLAUDE.md Gotchas.
import torch  # noqa: F401, E402

import pycolmap
from PIL import Image

from src.biomass import allometry, carbon
from src.geometry.canopy_height import estimate_tree_heights
from src.geometry.scale_resolution import apply_scale, estimate_scale
from src.geometry.sfm import align_gravity, run_sfm
from src.perception.crown_detector import CrownDetector
from src.perception.crown_segmenter import CrownSegmenter

IMAGE_DIR = Path("data/samples/ofo_mission_000001_sequence")
WORK_DIR = Path("outputs/biomass_work")
OUTPUT_PATH = Path("outputs/perception_spot_checks/biomass/scene_state_trees.json")

# Phase 2's measured height AbsRel against real LiDAR (notebook run, n=14 -- see
# notebooks/00_sfm_scale_validation.ipynb and docs/lessons_learnt.md's non-determinism entry for
# why this isn't quoted to more decimal places than that).
PHASE2_HEIGHT_ABS_REL = 0.65

# A generous upper bound on real crown diameter for the species in allometry.py's SPECIES_TABLE
# (all modest Sierra Nevada conifers/oak, not sequoias) -- used only to flag, not silently drop,
# trees whose crown_diameter_m is implausible (see module docstring, bug #3).
PLAUSIBLE_CROWN_DIAMETER_M = 10.0


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Running SfM...")
    sfm_result = run_sfm(IMAGE_DIR, WORK_DIR)
    print(f"  {sfm_result.num_registered}/{sfm_result.num_total} registered")

    db = pycolmap.Database.open(str(sfm_result.database_path))

    print("Aligning to gravity...")
    align_gravity(sfm_result.reconstruction, db)

    print("Resolving scale...")
    scale_est = estimate_scale(sfm_result.reconstruction, db)
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

    print("Estimating tree heights + crown diameters...")
    trees = estimate_tree_heights(sfm_result.reconstruction, crown_masks_by_image_id)
    print(f"  {len(trees)} trees resolved")

    print("\nEstimating biomass (Jucker et al. 2017 -- no DBH available from aerial imagery) "
          "and CO2e for each tree:")
    scene_state_trees = []
    for t in trees:
        agb = allometry.estimate_agb_from_crown(t.height_m, t.crown_diameter_m, species=None)

        # Combine Jucker's own model CV with Phase 2's measured height error in quadrature --
        # an approximation (assumes independence), documented as such, not a rigorous joint
        # error propagation. See module docstring.
        combined_cv = math.sqrt(agb.relative_uncertainty**2 + PHASE2_HEIGHT_ABS_REL**2)
        agb_with_height_error = allometry.AGBEstimate(
            agb_kg=agb.agb_kg,
            agb_kg_lower=agb.agb_kg * (1 - combined_cv),
            agb_kg_upper=agb.agb_kg * (1 + combined_cv),
            relative_uncertainty=combined_cv,
            model=agb.model,
            wsg=agb.wsg,
            domain_caveat=agb.domain_caveat,
        )
        co2e = carbon.estimate_co2e(agb_with_height_error)

        plausible = t.crown_diameter_m <= PLAUSIBLE_CROWN_DIAMETER_M
        flag = "" if plausible else "  [!] crown_diameter implausibly large -- see script docstring, bug #3"
        print(f"  tree {t.cluster_id}: height={t.height_m:.2f}m, crown_diameter={t.crown_diameter_m:.2f}m "
              f"-> AGB={agb.agb_kg:.1f}kg [{agb_with_height_error.agb_kg_lower:.1f}, "
              f"{agb_with_height_error.agb_kg_upper:.1f}], CO2e={co2e.co2e_kg:.1f}kg "
              f"[{co2e.co2e_kg_lower:.1f}, {co2e.co2e_kg_upper:.1f}]{flag}")

        scene_state_trees.append({
            "tree_id": f"tree_{t.cluster_id}",
            "lat": None,  # georeferencing to lat/lon is spot_check_lidar_validation.py's job;
            "lon": None,  # this script focuses on the biomass step in isolation
            "height_m": round(t.height_m, 2),
            "crown_diameter_m": round(t.crown_diameter_m, 2),
            "biomass_kg": round(agb.agb_kg, 1),
            "biomass_kg_lower": round(agb_with_height_error.agb_kg_lower, 1),
            "biomass_kg_upper": round(agb_with_height_error.agb_kg_upper, 1),
            "co2e_kg": round(co2e.co2e_kg, 1),
            "co2e_kg_lower": round(co2e.co2e_kg_lower, 1),
            "co2e_kg_upper": round(co2e.co2e_kg_upper, 1),
            "height_source": "sfm",
            "biomass_model": agb.model,
            "wsg_source": agb.wsg.source if agb.wsg else None,
            "tag_text": None,
            "crown_diameter_plausible": plausible,
        })

    OUTPUT_PATH.write_text(json.dumps(scene_state_trees, indent=2))
    print(f"\nwrote {OUTPUT_PATH}")

    total_co2e = sum(t["co2e_kg"] for t in scene_state_trees)
    print(f"\nTotal CO2e across {len(scene_state_trees)} trees: {total_co2e:.1f} kg "
          f"(point estimate -- per-tree uncertainty ranges are wide, see above; do not sum the "
          f"lower/upper bounds directly, since tree-level errors aren't independent of each "
          f"other in the way a naive sum would assume)")


if __name__ == "__main__":
    main()
