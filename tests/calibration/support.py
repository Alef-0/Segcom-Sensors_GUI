"""Stable image and journal fixtures shared by focused calibration tests."""

import json
from pathlib import Path

import cv2
import numpy as np

FIXTURE_DIR = Path(__file__).parent / "fixtures"
IMAGE_FIXTURE_DIR = FIXTURE_DIR / "images"
RECORDING_FRAME_FIXTURE_DIR = FIXTURE_DIR / "recording_frames"


def image_fixture(name: str) -> np.ndarray:
    """Load a deterministic synthetic image fixture."""
    path = IMAGE_FIXTURE_DIR / name
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Missing or unreadable test image: {path}")
    return image


def recording_frame_fixture(name: str) -> np.ndarray:
    """Load a checked-in frame copied from a project recording."""
    path = RECORDING_FRAME_FIXTURE_DIR / name
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Missing or unreadable recording frame fixture: {path}")
    return image


def build_calibration_analysis_recording(root: Path, *, legacy=False, resets=False,
                                         count=180) -> Path:
    """Create a deterministic analysis fixture for calibration timing tests."""
    from calibration.evidence import detection_evidence

    folder = root / "recording"
    folder.mkdir()
    analysis = root / "recording_analysis"
    analysis.mkdir()
    zero = 10_000_000_000
    display = [{"kind": "session", "grid_qrs": 4, "visible_qrs": 1}]
    for index in range(count * 2 + 20):
        stamp = zero + index * 10_000_000
        display.append({"kind": "frame", "index": index, "cell": index % 4,
                        "marker_ns": stamp, "presentation_return_ns": stamp})
    cameras, decoded = [], []
    for index in range(count):
        latest = 2 * index + 10
        pts = latest * 10_000_000 + 85_000_000 + (index % 2) * 1_000_000
        filename = f"images/camera_{index + 1:06d}.jpg"
        cameras.append({"frame": filename, "stream_epoch": 1, "segment_epoch": 1,
                        "mapping_revision": index if resets else 0,
                        "running_time_ns": pts, "pts_ns": pts,
                        "media_monotonic_ns": zero + pts,
                        "application_arrival_monotonic_ns": zero + pts + 50_000_000})
        values = [None] * 4
        values[latest % 4] = f"{(zero + latest * 10_000_000) // 1_000_000:012d}"
        observation = {"raw": str(latest), "bbox": (10, 10, 40, 40), "cell": 0,
                       "confidence": 0.9, "marker": {"cell": latest % 4},
                       "display_index": latest, "original_points": None}
        row = {"filename": filename, "qr_values_ms": values}
        if not legacy:
            row["qr_evidence"] = detection_evidence(
                [observation], (100, 100), {"status": "unavailable"}
            )
        decoded.append(row)
    for name, rows in (("display_timestamps.jsonl", display),
                       ("camera_timestamps.jsonl", cameras)):
        (folder / name).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    (folder / "camera_timing_session.json").write_text(json.dumps({"epochs": [
        {"stream_epoch": 1, "mapping_revision": 0, "segment_epoch": 1,
         "pipeline_zero_monotonic_ns": zero}]}), encoding="utf-8")
    (analysis / "calibration_analysis.json").write_text(json.dumps({
        "recording_directory": str(folder), "frames": decoded}), encoding="utf-8")
    return analysis


class FakeReader:
    def __init__(self, decoded, boxes):
        self.decoded = tuple(decoded)
        self.boxes = boxes

    def detect_and_decode(self, image, return_detections=False, is_bgr=False):
        if image.shape[:2] != (100, 100):
            return ((), ()) if return_detections else ()
        detections = tuple({"bbox_xyxy": np.asarray(box, dtype=np.float32), "confidence": 0.9}
                           for box in self.boxes)
        return (self.decoded, detections) if return_detections else self.decoded


def display_rows(count=5, period=16_666_667):
    frames = []
    for index in range(count):
        marker = 10_000_000_000 + index * period
        frames.append({
            "kind": "frame", "index": index, "corner": index % 4,
            "marker_ns": marker, "deadline_ns": marker + 1_000_000,
            "submit_ns": marker + 500_000, "flip_return_ns": marker + 1_000_000,
            "frame_period_ns": period, "interval_ns": period if index else None,
            "late_submit": False, "skipped_periods": 0, "irregular_interval": False,
            "resumed_after_pause": False,
        })
    return [{"kind": "session", "format": "segcom-qr-display-v1"}, *frames]


class RecordingFixtureMixin:
    def fixture(
        self,
        folder: Path,
        count=5,
        legacy=False,
        grid_qrs=4,
        visible_qrs=2,
        period=16_666_667,
    ):
        rows = display_rows(count, period)
        rows[0].update({"grid_qrs": grid_qrs, "visible_qrs": visible_qrs})
        for index, row in enumerate(rows[1:]):
            row["cell"] = index % grid_qrs
            row["corner"] = row["cell"]
        (folder / "display_timestamps.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        image_folder = folder if legacy else folder / "images"
        image_folder.mkdir(exist_ok=True)
        image_name = "camera_000001.jpg" if legacy else "images/camera_000001.jpg"
        if not cv2.imwrite(str(folder / image_name), image_fixture("empty_screen.png")):
            raise RuntimeError("Could not create camera fixture image")
        (folder / "camera_timestamps.jsonl").write_text(json.dumps({
            "frame": image_name, "stream_epoch": 1,
            "pts_ns": 100_000_000, "reference_ntp_ns": 1_700_000_000_000_000_000,
        }) + "\n", encoding="utf-8")
        (folder / "camera_timing_session.json").write_text(json.dumps({
            "epochs": [{"stream_epoch": 1, "pipeline_zero_monotonic_ns": 9_950_000_000}]
        }), encoding="utf-8")
        return rows

    def four_value_model(self, folder: Path, count=5):
        from calibration.qr import timestamp_payload
        from calibration.recording_display import DEFAULT_INTRINSICS, RecordingAnalyzer

        rows = self.fixture(folder, count)
        frames = rows[1:]
        selected = (frames[4], frames[1], frames[2], frames[3])
        boxes = ([10, 10, 30, 30], [60, 10, 80, 30],
                 [60, 60, 80, 80], [10, 60, 30, 80])
        reader = FakeReader([timestamp_payload(row["marker_ns"]) for row in selected], boxes)
        return RecordingAnalyzer(folder, DEFAULT_INTRINSICS, reader=reader), frames
