"""Monocular depth estimation, for single-frame / non-overlapping footage where SfM (`sfm.py`)
isn't viable -- no second viewpoint to triangulate from.

**Strictly a fallback, lower confidence than SfM.** Any tree height derived from this must be
marked `height_source: "depth_fallback"` in `scene_state.json`, never treated as equal-confidence
to an SfM-derived height (see CLAUDE.md).

Confirmed real limitation, not a hypothetical: tested the "metric" checkpoint
(`Depth-Anything-V2-Metric-Outdoor-*`, fine-tuned on Virtual KITTI -- a ground-level driving
dataset) against a real Open Forest Observatory nadir drone frame. It predicted depths of
6.4-17.2m, when the drone's own GPS/altitude EXIF for frames from this mission puts it at roughly
60-100m above the canopy. The model's "metric" calibration does not transfer to this altitude/
viewing-angle domain -- its absolute depth values are not trustworthy for aerial drone imagery,
even though the checkpoint is nominally metric. Documented here rather than silently used as if
accurate, and rather than over-engineering a custom recalibration for what's already meant to be
this pipeline's lowest-confidence path (see docs/lessons_learnt.md for the full test).

Given that, this wrapper deliberately does NOT claim its output is meters-accurate. It exposes
the model's raw depth map (relative structure within one frame is still useful -- closer/farther
pixels are ordered correctly even if the absolute scale is off) and leaves any attempt at
calibrating it to real altitude to whoever consumes it, with the mismatch explicit rather than
hidden.
"""
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Fine-tuned on Virtual KITTI for metric *driving-scene* depth -- confirmed NOT metrically
# accurate for nadir drone altitude (see module docstring). Kept because relative depth structure
# is still useful, and it's the best off-the-shelf outdoor-domain option available.
DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"


@dataclass
class DepthEstimate:
    depth: np.ndarray  # (H, W) float array, same size as the input image
    metric_scale_confirmed: bool = False  # always False -- see module docstring


class DepthFallbackEstimator:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def estimate(self, image: Image.Image) -> DepthEstimate:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        result = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[(image.size[1], image.size[0])]
        )[0]
        depth = result["predicted_depth"].cpu().numpy()
        return DepthEstimate(depth=depth, metric_scale_confirmed=False)
