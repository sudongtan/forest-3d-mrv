"""Open-vocabulary tree crown detection on drone frames.

Wraps Hugging Face's Grounding DINO. No fine-tuning needed for this project -- per CLAUDE.md's
guardrail to prefer pretrained/off-the-shelf models for the 2D perception stage, this is used
as-is with a text prompt ("tree crown.") rather than trained on a fixed class list. Output boxes
feed SAM2 in `crown_segmenter.py` for per-tree pixel masks.
"""
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
# Grounding DINO expects a lowercase, period-separated list of phrases to detect.
DEFAULT_PROMPT = "tree crown."
# Calibrated against real Open Forest Observatory frames (see CLAUDE.md Gotchas), not a
# textbook default: on a dense-canopy aerial photo, the single *highest*-confidence match for
# "tree crown" is consistently the whole image (score ~0.6), not any individual tree -- real,
# well-formed individual-crown-sized boxes exist in the same output but score lower (~0.1-0.25).
# The usual 0.25+ threshold silently keeps only the degenerate whole-image box and discards every
# real detection. Lowering the threshold recovers them, at the cost of also admitting the
# whole-image box itself, which MAX_AREA_FRACTION then explicitly rejects below.
DEFAULT_BOX_THRESHOLD = 0.12
DEFAULT_TEXT_THRESHOLD = 0.12
# Reject boxes covering an implausibly large fraction of the frame -- a single tree crown at
# typical drone survey altitude should occupy a small portion of a multi-megapixel frame; a box
# near full-frame size is the degenerate "whole scene matches the prompt" failure mode above, not
# a real crown.
DEFAULT_MAX_AREA_FRACTION = 0.3


@dataclass
class CrownDetection:
    box_xyxy: tuple[float, float, float, float]  # pixel coords in the original image
    score: float
    label: str


class CrownDetector:
    """Detects tree crowns in a single drone frame via open-vocabulary text prompting."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def detect(
        self,
        image: Image.Image,
        prompt: str = DEFAULT_PROMPT,
        box_threshold: float = DEFAULT_BOX_THRESHOLD,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
        max_area_fraction: float = DEFAULT_MAX_AREA_FRACTION,
    ) -> list[CrownDetection]:
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],  # processor wants (height, width)
        )[0]

        image_area = image.size[0] * image.size[1]
        detections = [
            CrownDetection(box_xyxy=tuple(box.tolist()), score=float(score), label=str(label))
            for box, score, label in zip(
                result["boxes"], result["scores"], result["text_labels"]
            )
        ]
        return [
            d for d in detections
            if (d.box_xyxy[2] - d.box_xyxy[0]) * (d.box_xyxy[3] - d.box_xyxy[1])
            < max_area_fraction * image_area
        ]
