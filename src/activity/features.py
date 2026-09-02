"""Per-frame appearance features for `activity/model.py`'s temporal classifier.

MammAlps' real annotation schema (see `datasets.py`) gives bounding boxes + attributes per frame,
not skeleton keypoints -- so this project's feature path is frame-appearance based, not
pose-based, per CLAUDE.md's own explicit caveat that many camera-trap behavior datasets provide
direct labels rather than extractable keypoints; confirmed true for MammAlps specifically by
inspecting its real downloaded annotation JSON, not assumed from the paper text.

Uses a frozen, ImageNet-pretrained MobileNetV3-Small (torchvision) to embed each frame's bbox
crop -- per CLAUDE.md's guardrail to prefer pretrained/off-the-shelf models for perception unless
real-footage validation shows a specific gap; this project's differentiator is the geometry +
fusion + reporting layers, not proving a backbone can be fine-tuned from a single demo clip.
Chosen over a larger backbone (e.g. ResNet50) specifically for CPU/MPS speed on an M1 -- a
per-clip windowing scheme (see datasets.py) already means many small forward passes per training
step.
"""
import numpy as np
import torch
import torchvision
from torchvision.models import MobileNet_V3_Small_Weights

FEATURE_DIM = 576  # MobileNetV3-Small's pooled feature dim before its classification head --
# confirmed by inspection (removing the classifier head and checking the real output shape), not
# assumed from documentation.


class FrameFeatureExtractor:
    def __init__(self, device: str | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        backbone = torchvision.models.mobilenet_v3_small(weights=weights)
        backbone.classifier = torch.nn.Identity()  # keep the pooled 576-dim feature, drop the
        # 1000-way ImageNet classification head -- this project only needs the embedding
        self.model = backbone.to(self.device).eval()
        self.preprocess = weights.transforms()

    @torch.no_grad()
    def extract_crop(self, frame_rgb: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray:
        """One frame, cropped to `bbox_xyxy` (the track's real detection box for that frame) ->
        one 576-dim feature vector.
        """
        x1, y1, x2, y2 = bbox_xyxy
        h, w = frame_rgb.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(FEATURE_DIM, dtype=np.float32)  # degenerate bbox (real data can have
            # a box that's clipped to zero width/height at a frame boundary) -- a zero feature is
            # a defensible neutral fallback, not a crash, for one frame inside a longer window
        tensor = self.preprocess(torch.from_numpy(crop).permute(2, 0, 1)).unsqueeze(0).to(self.device)
        out = self.model(tensor)
        return out.squeeze(0).cpu().numpy()

    def extract_window(
        self, frames_rgb: dict[int, np.ndarray], bboxes_by_frame: dict[int, tuple[int, int, int, int]],
        start_frame: int, end_frame: int,
    ) -> np.ndarray:
        """A whole window (see `datasets.ActivityWindow`) -> a (window_size, 576) feature array,
        one row per frame. Frames where the track has no box in this window (shouldn't happen for
        frames windows_for_track already selected, but real data can still be missing a frame)
        reuse the nearest available frame's feature via forward-fill, rather than a zero vector,
        so the temporal model doesn't see a spurious discontinuity mid-window.
        """
        feats = []
        last_valid = None
        for frame_id in range(start_frame, end_frame):
            if frame_id in bboxes_by_frame and frame_id in frames_rgb:
                feat = self.extract_crop(frames_rgb[frame_id], bboxes_by_frame[frame_id])
                last_valid = feat
            elif last_valid is not None:
                feat = last_valid
            else:
                feat = np.zeros(FEATURE_DIM, dtype=np.float32)
            feats.append(feat)
        return np.stack(feats)
