"""Trains `activity/model.py`'s temporal classifier on real MammAlps demo-clip windows
(`datasets.py`) and confirms it can learn to discriminate the real behavior classes present --
the architecture-sanity-check step CLAUDE.md's plan asks for before scaling to the full dataset.

**What this script can and can't claim, stated plainly up front**: with only one real annotated
clip available (see `datasets.py`'s module docstring for why -- MammAlps' full ~88GB release has
no partial-download path, and Rolandseck turned out not to be a confirmed self-serve download
either), there is no way to build a genuine camera/site-disjoint train/val split, which
CLAUDE.md's own guardrail requires for any real generalization claim. This script instead does
exactly what CLAUDE.md's plan says this step is *for*: confirms the architecture trains and
overfits correctly on a small real labeled set (all windows, from both of the clip's two real
tracked individuals). As a bonus -- not a substitute for a real held-out eval -- it also reports
accuracy training on one individual (track 1) and evaluating on the other (track 2, a different
animal, same camera/scene) as a strictly weaker "quasi-generalization" signal, clearly logged
under a different metric name so it can never be confused with a real site-disjoint result.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE uv run python -m src.activity.train
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import mlflow
import numpy as np
import torch
from torch import nn

from src.activity.datasets import ActivityWindow, load_annotation, load_label_mapping, windows_for_track
from src.activity.evaluate import evaluate, log_to_mlflow
from src.activity.features import FrameFeatureExtractor
from src.activity.model import TemporalActivityClassifier
from src.mlflow_utils import set_experiment

DEMO_DIR = Path("data/samples/mammalps_demo")
WINDOW_SIZE = 16
STRIDE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3


def read_all_frames(video_path: Path) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = {}
    frame_id = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frames[frame_id] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_id += 1
    cap.release()
    return frames


def bboxes_by_frame(annotation, track_id: int) -> dict[int, tuple[int, int, int, int]]:
    result = {}
    for f in annotation.frames:
        for det in f["detections"]:
            if det["track_id"] == track_id:
                result[f["frame_id"]] = tuple(det["bbox"])
    return result


def build_window_dataset(
    windows: list[ActivityWindow], frames_rgb: dict[int, np.ndarray],
    bboxes_by_track: dict[int, dict[int, tuple[int, int, int, int]]],
    extractor: FrameFeatureExtractor, activity_to_idx: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    features, labels = [], []
    for w in windows:
        feat = extractor.extract_window(
            frames_rgb, bboxes_by_track[w.track_id], w.start_frame, w.end_frame
        )
        features.append(feat)
        labels.append(activity_to_idx[w.activity])
    X = torch.tensor(np.stack(features), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return X, y


def train_model(model: TemporalActivityClassifier, X: torch.Tensor, y: torch.Tensor, num_epochs: int) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            acc = (logits.argmax(dim=1) == y).float().mean().item()
            print(f"  epoch {epoch}: loss={loss.item():.4f}, train_acc={acc:.3f}")


def main() -> None:
    annotation = load_annotation(DEMO_DIR / "demo_annotations.json")
    label_mapping = load_label_mapping(DEMO_DIR / "labels_mapping_b1.json")

    print("Extracting real annotation windows...")
    windows_t1 = windows_for_track(annotation, track_id=1, window_size=WINDOW_SIZE, stride=STRIDE)
    windows_t2 = windows_for_track(annotation, track_id=2, window_size=WINDOW_SIZE, stride=STRIDE)
    all_windows = windows_t1 + windows_t2
    activities_present = sorted({w.activity for w in all_windows})
    activity_to_idx = {a: i for i, a in enumerate(activities_present)}
    print(f"  track 1: {len(windows_t1)} windows, track 2: {len(windows_t2)} windows")
    print(f"  real activity classes present: {activities_present}")

    print("Reading real video frames...")
    frames_rgb = read_all_frames(DEMO_DIR / "demo_video.mp4")
    bboxes_by_track = {
        1: bboxes_by_frame(annotation, 1),
        2: bboxes_by_frame(annotation, 2),
    }

    print("Extracting per-frame features (frozen MobileNetV3-Small)...")
    extractor = FrameFeatureExtractor()
    X_all, y_all = build_window_dataset(all_windows, frames_rgb, bboxes_by_track, extractor, activity_to_idx)
    X_t1, y_t1 = build_window_dataset(windows_t1, frames_rgb, bboxes_by_track, extractor, activity_to_idx)
    X_t2, y_t2 = build_window_dataset(windows_t2, frames_rgb, bboxes_by_track, extractor, activity_to_idx)

    set_experiment("phase5_activity_sanity_check")
    with mlflow.start_run(run_name="mammalps_demo_clip_overfit_check"):
        mlflow.log_params({
            "window_size": WINDOW_SIZE, "stride": STRIDE, "num_epochs": NUM_EPOCHS,
            "num_windows_total": len(all_windows), "num_classes": len(activities_present),
            "classes": ",".join(activities_present), "file_id": annotation.info.file_id,
        })

        print(f"\n[1] Overfit check: train and evaluate on all {len(all_windows)} real windows "
              "(both tracked individuals) -- confirms the architecture can learn to discriminate "
              "the real classes present. NOT a generalization claim.")
        model = TemporalActivityClassifier(num_classes=len(activities_present))
        train_model(model, X_all, y_all, NUM_EPOCHS)
        result = evaluate(model, X_all, y_all, activities_present)
        print(f"  overfit accuracy={result.accuracy:.3f}, per-class F1={result.per_class_f1}")
        log_to_mlflow(result, prefix="overfit_")

        print(f"\n[2] Quasi-generalization check: train on track 1 ({len(windows_t1)} windows), "
              f"evaluate on track 2 ({len(windows_t2)} windows) -- a different real individual, "
              "same camera/scene. Weaker than a real site-disjoint split (CLAUDE.md's guardrail) "
              "-- reported separately so it's never confused with one.")
        model2 = TemporalActivityClassifier(num_classes=len(activities_present))
        train_model(model2, X_t1, y_t1, NUM_EPOCHS)
        cross_track_result = evaluate(model2, X_t2, y_t2, activities_present)
        print(f"  cross-track accuracy={cross_track_result.accuracy:.3f}, "
              f"per-class F1={cross_track_result.per_class_f1}")
        log_to_mlflow(cross_track_result, prefix="cross_track_")

    print("\nLogged to MLflow experiment 'phase5_activity_sanity_check'.")


if __name__ == "__main__":
    main()
