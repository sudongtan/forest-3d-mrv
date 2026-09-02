"""MegaDetector triage + SpeciesNet species ID for camera-trap frames.

Two-stage pipeline: MegaDetector does cheap animal/person/vehicle triage on every frame (via
PytorchWildlife); only "animal" detections get cropped and passed on to SpeciesNet for
species-level classification. "person"/"vehicle" detections skip species/behavior classification
entirely -- those go straight to `scene_state.json`'s `alerts`, not `wildlife_events` (see
CLAUDE.md). MegaDetector alone cannot tell you species, only the 3-class category -- that's the
whole reason SpeciesNet is chained on afterward instead of relying on MegaDetector by itself.
"""
import os
from dataclasses import dataclass

import numpy as np
import speciesnet
import torch
from PIL import Image
from PytorchWildlife.models import detection as pw_detection

DEFAULT_MEGADETECTOR_VERSION = "MDV6-yolov9-c"
DEFAULT_SPECIESNET_MODEL = speciesnet.DEFAULT_MODEL


@dataclass
class CameraTrapDetection:
    box_xyxy_norm: tuple[float, float, float, float]  # normalized 0-1 coords
    category: str  # "animal" | "person" | "vehicle"
    confidence: float
    species: str | None = None  # populated only for "animal" detections
    species_confidence: float | None = None


class CameraTrapDetector:
    def __init__(
        self,
        device: str | None = None,
        megadetector_version: str = DEFAULT_MEGADETECTOR_VERSION,
        speciesnet_model: str = DEFAULT_SPECIESNET_MODEL,
    ):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

        # PytorchWildlife 1.3.0's own local-cache check for this version string looks for a
        # filename ("MDV6b-yolov9-c.pt") that never actually gets written -- the real downloaded
        # file (matching the download URL) is "MDV6-yolov9-c.pt", no "b". That mismatch means the
        # exists-check always fails and it redownloads the ~50MB checkpoint from Zenodo on every
        # process launch, which is sometimes flaky (hit a 504 mid-download once while building
        # this wrapper). See docs/lessons_learnt.md. Workaround: check for the real cached file
        # ourselves and pass its path directly via `weights=`, which skips the buggy check
        # entirely. First run on a clean machine still goes through the (occasionally flaky, but
        # functional) download path once.
        cached_weights = os.path.join(
            torch.hub.get_dir(), "checkpoints", f"{megadetector_version}.pt"
        )
        weights_arg = cached_weights if os.path.exists(cached_weights) else None
        self.detector = pw_detection.MegaDetectorV6(
            weights=weights_arg, device=self.device, version=megadetector_version
        )
        self.classifier = speciesnet.SpeciesNetClassifier(speciesnet_model)

    def detect(
        self, image: np.ndarray, det_conf_thres: float = 0.2
    ) -> list[CameraTrapDetection]:
        """`image`: RGB numpy array (H, W, 3), e.g. one video frame."""
        result = self.detector.single_image_detection(image, det_conf_thres=det_conf_thres)
        pil_image = Image.fromarray(image)

        detections = []
        for i, (x1, y1, x2, y2) in enumerate(result["normalized_coords"]):
            class_id = int(result["detections"].class_id[i])
            category = self.detector.CLASS_NAMES[class_id]
            confidence = float(result["detections"].confidence[i])

            species = None
            species_confidence = None
            if category == "animal":
                # SpeciesNetClassifier.predict returns one classification set per call, so
                # animals sharing a frame must be classified one at a time, each cropped to its
                # own box -- passing every box from the frame in one call silently only uses the
                # first (verified empirically while building this wrapper, not documented).
                bbox = speciesnet.BBox(
                    xmin=float(x1), ymin=float(y1), width=float(x2 - x1), height=float(y2 - y1)
                )
                preprocessed = self.classifier.preprocess(pil_image, bboxes=[bbox])
                pred = self.classifier.predict(f"detection_{i}", preprocessed)
                classifications = pred.get("classifications")
                if classifications and classifications["classes"]:
                    # class string is "uuid;class;order;family;genus;species;common_name" --
                    # scene_state.json wants the common name.
                    species = classifications["classes"][0].split(";")[-1]
                    species_confidence = float(classifications["scores"][0])

            detections.append(
                CameraTrapDetection(
                    box_xyxy_norm=(float(x1), float(y1), float(x2), float(y2)),
                    category=category,
                    confidence=confidence,
                    species=species,
                    species_confidence=species_confidence,
                )
            )
        return detections
