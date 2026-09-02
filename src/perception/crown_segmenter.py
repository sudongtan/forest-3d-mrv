"""Per-frame tree crown segmentation via SAM2, prompted from crown_detector.py's boxes.

Produces one pixel-precise mask per detected crown, per frame. SAM2's video tracking mode is a
separate capability, exercised in `track_video()` purely as a spot-check (useful for continuous
footage elsewhere in this project, e.g. a handheld walkthrough) -- per CLAUDE.md, it is NOT the
source of cross-frame tree identity for the main drone/SfM path. That's resolved geometrically in
`geometry/canopy_height.py`, using COLMAP poses once they exist, not by inheriting 2D tracking
IDs from this module.
"""
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor, Sam2VideoModel, Sam2VideoProcessor

from src.perception.crown_detector import CrownDetection

DEFAULT_MODEL_ID = "facebook/sam2.1-hiera-tiny"


@dataclass
class CrownMask:
    mask: np.ndarray  # boolean array, shape (H, W) -- same size as the input image
    score: float  # SAM2's own predicted IoU confidence for this mask
    source_detection: CrownDetection


class CrownSegmenter:
    """Segments one pixel-precise crown mask per box, for a single frame."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def segment(self, image: Image.Image, detections: list[CrownDetection]) -> list[CrownMask]:
        if not detections:
            return []

        # All boxes for one image go in a single inner list -- SAM2 batches multiple
        # box-prompts against the same image encoding rather than re-encoding per box.
        boxes = [[list(d.box_xyxy) for d in detections]]
        inputs = self.processor(images=image, input_boxes=boxes, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)

        # post_process_masks upscales SAM2's native 256x256 mask back to the original image
        # resolution -- masks[0] because there's one image in this batch.
        masks = self.processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"])[
            0
        ]

        results = []
        for i, det in enumerate(detections):
            mask = masks[i, 0].cpu().numpy().astype(bool)
            score = float(outputs.iou_scores[0, i, 0])
            results.append(CrownMask(mask=mask, score=score, source_detection=det))
        return results


class CrownVideoTracker:
    """SAM2's video tracking mode -- a capability spot-check only (see module docstring), not
    used by the main drone/SfM pipeline. Verified against a real camera-trap clip
    (`data/samples/camera_trap_lila_seattleish/deer/DSCF0008.AVI`): given one point on the first
    frame, correctly tracked the same animal through 60 frames including a full pose change
    (side-on to walking away), with no drift onto background or the second deer in frame.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = Sam2VideoProcessor.from_pretrained(model_id)
        self.model = Sam2VideoModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def track(
        self, frames: list[Image.Image], point_xy: tuple[int, int], start_frame_idx: int = 0
    ) -> dict[int, np.ndarray]:
        """Track one object, seeded by a single point on `frames[start_frame_idx]`, through the
        rest of `frames`. Returns {frame_idx: boolean mask array} for every propagated frame.
        """
        session = self.processor.init_video_session(video=frames, inference_device=self.device)

        self.processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=start_frame_idx,
            obj_ids=1,
            input_points=[[[list(point_xy)]]],
            input_labels=[[[1]]],  # 1 = positive (foreground) click
        )
        with torch.no_grad():
            self.model(inference_session=session, frame_idx=start_frame_idx)

            masks_by_frame = {}
            for output in self.model.propagate_in_video_iterator(session):
                masks = self.processor.post_process_masks(
                    [output.pred_masks],
                    original_sizes=[[session.video_height, session.video_width]],
                    binarize=True,
                )[0]
                masks_by_frame[output.frame_idx] = masks[0, 0].cpu().numpy().astype(bool)
        return masks_by_frame
