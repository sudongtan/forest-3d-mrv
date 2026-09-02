"""Manifest reader for MammAlps' real Benchmark I annotation schema: dense, per-frame, per-track
detections with action/activity/species attributes -- verified against a real downloaded sample
(`data/samples/mammalps_demo/`), not assumed from the paper text alone.

**Why a demo clip, not the full dataset**: MammAlps' full release is a single ~87.9GB zip on
Zenodo with no partial-download option (confirmed: the file server doesn't support HTTP range
requests, so Phase 2's `remotezip` trick for OFO's images.zip doesn't apply here) -- impractical
for this project's local M1 Pro dev environment. The dataset's own GitHub repo
(github.com/eceo-epfl/MammAlps) bundles one real, fully-annotated demo clip directly in git
(`resources/demo_video.mp4` + `resources/demo_annotations.json`, ~12.6MB total) for its own demo
notebook -- used here as a real, small, immediately-usable substitute for Rolandseck (the
architecture sanity-check set CLAUDE.md's plan originally named), since Rolandseck itself turned
out not to be a confirmed self-serve download either. One real clip, two real tracked individuals
(two red deer), three real activity classes with meaningful representation (foraging, vigilance,
unknown) -- enough to sanity-check that `activity/model.py`'s architecture can learn to
discriminate real behavior windows, **not** enough for a genuine held-out generalization claim
(that needs multiple cameras/sites, per CLAUDE.md's own site-disjoint-split guardrail, which this
one-clip sample cannot satisfy by construction). See docs/lessons_learnt.md for the full account
of why the full dataset and Rolandseck were both ruled out.

**Schema, confirmed from the real downloaded JSON** (not the shape a clip-level reader might
assume): each annotation file covers one clip and has dense *per-frame* detections, each with a
`track_id` (one real individual, trackable across frames), a `bbox`, and `attributes` with
`action` (fine-grained, e.g. "walking", "sniffing"), `activity` (coarser, e.g. "foraging",
"vigilance") and `species`. There is no clip-level single label -- `windows_for_track` below
turns this into fixed-length, majority-labeled windows, which *is* the manifest granularity
CLAUDE.md's plan wanted (one label per training sample), just derived rather than given directly.
"""
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClipInfo:
    site_id: str
    cam_id: str
    event_id: str
    file_id: str
    file_path: str
    num_frames: int
    duration_s: float
    fps: float
    resolution: str


@dataclass(frozen=True)
class ClipAnnotation:
    info: ClipInfo
    frames: list[dict]  # kept close to the real JSON shape (list of {frame_id, detections})
    # rather than re-modeled into a deeper dataclass tree -- there's exactly one consumer
    # (windows_for_track) and re-modeling would just be indirection with no real payoff here.


def load_annotation(path: Path | str) -> ClipAnnotation:
    data = json.loads(Path(path).read_text())
    info = ClipInfo(
        site_id=data["info"]["site_id"],
        cam_id=data["info"]["cam_id"],
        event_id=data["info"]["event_id"],
        file_id=data["info"]["file_id"],
        file_path=data["info"]["file_path"],
        num_frames=data["info"]["num_frames"],
        duration_s=data["info"]["duration_s"],
        fps=data["info"]["fps"],
        resolution=data["info"]["resolution"],
    )
    return ClipAnnotation(info=info, frames=data["frames"])


def load_label_mapping(path: Path | str) -> dict[str, dict[str, int]]:
    """Loads a `labels_mapping_b{1,2}.json` file -- e.g. `{"activities": {"foraging": 4, ...},
    "actions": {...}, "species": {...}}`. Benchmark I (`labels_mapping_b1.json`) is what
    `windows_for_track` below is built against.
    """
    return json.loads(Path(path).read_text())


@dataclass(frozen=True)
class ActivityWindow:
    file_id: str
    track_id: int
    start_frame: int
    end_frame: int  # exclusive
    activity: str
    action: str
    species: str
    site_id: str
    cam_id: str
    label_agreement: float  # fraction of this window's frames whose activity label matched the
    # majority-vote label -- a low value means the window straddles a real behavior transition
    # (this clip's real labels flip frame-to-frame near transitions, sometimes every few frames)
    # rather than sitting cleanly inside one behavior; low-agreement windows are real, harder
    # training examples, not a data-quality bug, but worth being able to filter/inspect.


def windows_for_track(
    annotation: ClipAnnotation, track_id: int, window_size: int = 16, stride: int = 8,
    min_presence: float = 0.5,
) -> list[ActivityWindow]:
    """Turns one track's dense per-frame labels into fixed-length windows with one majority-vote
    label each -- MammAlps' real schema has no clip/window-level label directly (see module
    docstring), so this derives one.

    A window is only emitted if `track_id` actually appears in at least `min_presence` of its
    frames (real tracks don't span every frame of a clip -- e.g. this project's real demo clip's
    second track only appears in frames 219-614, not from frame 0).
    """
    frames_by_id = {f["frame_id"]: f["detections"] for f in annotation.frames}
    max_frame = annotation.info.num_frames

    windows = []
    for start in range(0, max_frame - window_size + 1, stride):
        end = start + window_size
        activities, actions, species = [], [], []
        for frame_id in range(start, end):
            for det in frames_by_id.get(frame_id, []):
                if det["track_id"] == track_id:
                    activities.append(det["attributes"]["activity"])
                    actions.append(det["attributes"]["action"])
                    species.append(det["attributes"]["species"])

        if len(activities) < min_presence * window_size:
            continue

        activity_counts = Counter(activities)
        majority_activity, majority_count = activity_counts.most_common(1)[0]
        windows.append(ActivityWindow(
            file_id=annotation.info.file_id,
            track_id=track_id,
            start_frame=start,
            end_frame=end,
            activity=majority_activity,
            action=Counter(actions).most_common(1)[0][0],
            species=Counter(species).most_common(1)[0][0],
            site_id=annotation.info.site_id,
            cam_id=annotation.info.cam_id,
            label_agreement=majority_count / len(activities),
        ))
    return windows


def track_ids_in(annotation: ClipAnnotation) -> list[int]:
    ids = set()
    for f in annotation.frames:
        for det in f["detections"]:
            ids.add(det["track_id"])
    return sorted(ids)
