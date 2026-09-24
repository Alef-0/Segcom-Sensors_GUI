"""Timestamp QR creation, decoding, and grid classification."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps

import numpy as np
import qrcode
from qrcode.constants import ERROR_CORRECT_L

PAYLOAD_DIGITS = 12
PAYLOAD_MODULUS_MS = 10 ** PAYLOAD_DIGITS
QUIET_ZONE_MODULES = 2
QR_MASK_PATTERNS = tuple(range(8))
DETECTION_BATCH_SIZE = 4
GRID_LAYOUTS = {4: (2, 2), 6: (2, 3), 8: (2, 4), 9: (3, 3), 10: (2, 5), 12: (2, 6)}


def timestamp_payload(timestamp_ns: int) -> str:
    return f"{int(timestamp_ns) // 1_000_000 % PAYLOAD_MODULUS_MS:012d}"


def qr_matrix(payload: str, border: int = QUIET_ZONE_MODULES,
              mask_pattern: int | None = None) -> np.ndarray:
    if mask_pattern is not None and mask_pattern not in QR_MASK_PATTERNS:
        raise ValueError("QR mask pattern must be from 0 to 7")
    qr = qrcode.QRCode(version=1, error_correction=ERROR_CORRECT_L, box_size=1,
                       border=border, mask_pattern=mask_pattern)
    qr.add_data(payload, optimize=0)
    qr.make(fit=True)
    return np.asarray(qr.get_matrix(), dtype=np.uint8)


def decode_qrs(reader, image: np.ndarray) -> list[dict]:
    decoded, detections = reader.detect_and_decode(image=image, return_detections=True, is_bgr=True)
    results = []
    for raw, detection in zip(decoded, detections or ()):
        box = np.asarray(detection["bbox_xyxy"], dtype=float).reshape(4)
        x1, y1, x2, y2 = box
        results.append({"raw": raw, "bbox": box,
                        "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                        "confidence": float(detection.get("confidence", 0.0))})
    return results


def decode_qrs_batch(reader, images: list[np.ndarray]) -> list[list[dict]]:
    """Batch QReader detection when supported, preserving input order."""
    if not images:
        return []
    detector = getattr(reader, "detector", None)
    model = getattr(detector, "model", None)
    if len(images) == 1 or model is None or not hasattr(reader, "decode"):
        return [decode_qrs(reader, image) for image in images]

    try:
        from qrdet import _prepare_input, _yolo_v8_results_to_dict
    except ImportError:
        return [decode_qrs(reader, image) for image in images]

    prepared = [_prepare_input(source=image, is_bgr=True) for image in images]
    predictions = model.predict(
        source=prepared,
        conf=detector._conf_th,
        iou=detector._nms_iou,
        device=None,
        max_det=100,
        augment=False,
        agnostic_nms=True,
        classes=None,
        verbose=False,
    )
    if len(predictions) != len(images):
        raise RuntimeError(
            f"QR detector returned {len(predictions)} results for {len(images)} images"
        )

    batches = []
    for image, prepared_image, prediction in zip(images, prepared, predictions):
        detections = _yolo_v8_results_to_dict(results=prediction, image=prepared_image)
        decoded = [reader.decode(image=image, detection_result=item) for item in detections]
        batch = []
        for raw, detection in zip(decoded, detections):
            box = np.asarray(detection["bbox_xyxy"], dtype=float).reshape(4)
            x1, y1, x2, y2 = box
            batch.append({
                "raw": raw,
                "bbox": box,
                "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                "confidence": float(detection.get("confidence", 0.0)),
            })
        batches.append(batch)
    return batches


def grid_shape(grid_qrs: int) -> tuple[int, int]:
    try:
        return GRID_LAYOUTS[int(grid_qrs)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"QR grid must contain one of: {', '.join(map(str, GRID_LAYOUTS))}") from error


def grid_positions(grid_qrs: int) -> tuple[tuple[int, int], ...]:
    rows, columns = grid_shape(grid_qrs)
    return tuple(
        (row, column)
        for row in range(rows)
        for column in (range(columns) if row % 2 == 0 else range(columns - 1, -1, -1))
    )


def grid_cell_names(grid_qrs: int) -> tuple[str, ...]:
    if grid_qrs == 4:
        return ("Top-left", "Top-right", "Bottom-right", "Bottom-left")
    return tuple(f"Row {row + 1}, column {column + 1}" for row, column in grid_positions(grid_qrs))


def grid_bounds(width: int, height: int, grid_qrs: int) -> tuple[tuple[int, int, int, int], ...]:
    rows, columns = grid_shape(grid_qrs)
    return tuple((width * col // columns, height * row // rows,
                  width * (col + 1) // columns, height * (row + 1) // rows)
                 for row, col in grid_positions(grid_qrs))


def cell_index_for(center: tuple[float, float], size: tuple[int, int], grid_qrs: int) -> int:
    x, y = center
    width, height = size
    rows, columns = grid_shape(grid_qrs)
    row = min(rows - 1, max(0, int(y * rows / max(1, height))))
    column = min(columns - 1, max(0, int(x * columns / max(1, width))))
    return grid_positions(grid_qrs).index((row, column))


def detect_contrast_cells(image: np.ndarray, grid_qrs: int) -> list[dict]:
    """Find display cells with bright, high-contrast content on the dark screen."""
    pixels = np.asarray(image)
    if pixels.ndim == 2:
        gray = pixels.astype(np.float32)
    elif pixels.ndim == 3 and pixels.shape[2] == 1:
        gray = pixels[:, :, 0].astype(np.float32)
    elif pixels.ndim == 3 and pixels.shape[2] >= 3:
        bgr = pixels[:, :, :3].astype(np.float32)
        gray = 0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2]
    else:
        raise ValueError("Contrast detection expects a grayscale or BGR image")
    height, width = gray.shape
    detections = []
    for cell, (left, top, right, bottom) in enumerate(grid_bounds(width, height, grid_qrs)):
        region = gray[top:bottom, left:right]
        if region.size == 0:
            continue
        background = float(np.percentile(region, 50))
        threshold = max(110.0, min(245.0, background + 50.0))
        bright = region > threshold
        bright_count = int(np.count_nonzero(bright))
        minimum_count = max(6, int(np.ceil(region.size * 0.005)))
        if bright_count < minimum_count:
            continue
        ys, xs = np.nonzero(bright)
        x1, y1 = left + int(xs.min()), top + int(ys.min())
        x2, y2 = left + int(xs.max()) + 1, top + int(ys.max()) + 1
        bbox = np.asarray((x1, y1, x2, y2), dtype=float)
        detections.append({
            "raw": None,
            "bbox": bbox,
            "center": ((x1 + x2) / 2, (y1 + y2) / 2),
            "confidence": bright_count / region.size,
            "cell": cell,
            "detection_method": "screen_contrast",
        })
    return detections


def decode_qrs_with_grid_retries(reader, image: np.ndarray, grid_qrs: int) -> list[dict]:
    return decode_qrs_with_grid_retries_batch(reader, [image], grid_qrs)[0]


def decode_qrs_with_grid_retries_batch(reader, images: list[np.ndarray], grid_qrs: int,
                                       batch_size: int = DETECTION_BATCH_SIZE) -> list[list[dict]]:
    if batch_size < 1:
        raise ValueError("QR detection batch size must be positive")
    results = []
    for start in range(0, len(images), batch_size):
        results.extend(decode_qrs_batch(reader, images[start:start + batch_size]))
    jobs = []
    for image_index, (image, detections) in enumerate(zip(images, results)):
        height, width = image.shape[:2]
        seen = {cell_index_for(item["center"], (width, height), grid_qrs)
                for item in detections if item.get("raw") is not None}
        for cell, (left, top, right, bottom) in enumerate(grid_bounds(width, height, grid_qrs)):
            if cell not in seen:
                jobs.append((image_index, cell, left, top, image[top:bottom, left:right]))
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start:start + batch_size]
        decoded = decode_qrs_batch(reader, [job[4] for job in chunk])
        for (image_index, cell, left, top, _crop), detections in zip(chunk, decoded):
            image = images[image_index]
            height, width = image.shape[:2]
            for item in detections:
                item["bbox"] = np.asarray(item["bbox"], dtype=float) + (left, top, left, top)
                item["center"] = (item["center"][0] + left, item["center"][1] + top)
                if cell_index_for(item["center"], (width, height), grid_qrs) == cell:
                    results[image_index].append(item)
    return results


def decode_qrs_with_quadrant_retries(reader, image: np.ndarray) -> list[dict]:
    return decode_qrs_with_grid_retries(reader, image, 4)


def create_qreader():
    from qreader import QReader
    reader = QReader(min_confidence=0.3)
    predict = reader.detector.model.predict

    @wraps(predict)
    def compatible_predict(*args, **kwargs):
        kwargs.pop("half", None)
        kwargs.setdefault("quantize", None)
        return predict(*args, **kwargs)

    reader.detector.model.predict = compatible_predict
    return reader


@dataclass(frozen=True, slots=True)
class Quadrant:
    index: int
    name: str


QUADRANTS = tuple(Quadrant(i, name) for i, name in enumerate(
    ("Top-left", "Top-right", "Bottom-right", "Bottom-left")
))


def quadrant_for(center: tuple[float, float], size: tuple[int, int]) -> Quadrant:
    x, y = center
    width, height = size
    if y < height / 2:
        return QUADRANTS[0 if x < width / 2 else 1]
    return QUADRANTS[3 if x < width / 2 else 2]


def order_by_quadrant(detections: list[dict], size: tuple[int, int]) -> list[dict]:
    rows = [{**item, "quadrant": q.index, "quadrant_name": q.name}
            for item in detections if (q := quadrant_for(item["center"], size))]
    return sorted(rows, key=lambda item: (item["quadrant"], item["center"][1], item["center"][0]))


def order_by_cell(detections: list[dict], size: tuple[int, int], grid_qrs: int) -> list[dict]:
    names = grid_cell_names(grid_qrs)
    rows = []
    for item in detections:
        cell = cell_index_for(item["center"], size, grid_qrs)
        rows.append({**item, "cell": cell, "cell_name": names[cell],
                     "quadrant": cell, "quadrant_name": names[cell]})
    return sorted(rows, key=lambda item: (item["cell"], item["center"][1], item["center"][0]))
