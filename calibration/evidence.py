"""Pixel evidence and optional screen geometry, independent of fitted timing."""

from __future__ import annotations

import numpy as np


EVIDENCE_VERSION = 1


def screen_geometry(config: dict | None, filename: str, size: tuple[int, int],
                    alpha: float) -> dict:
    """Map undistorted image pixels to the screen unit square.

    Corners are TL, TR, BR, BL in the undistorted image. Frame overrides allow
    explicit recalibration after movement; no timing model chooses geometry.
    """
    if config is None:
        return {"status": "unavailable"}
    if not isinstance(config, dict) or not isinstance(config.get("frames", {}), dict):
        return {"status": "invalid", "reason": "geometry_config_and_frames_must_be_objects"}
    entry = config.get("frames", {}).get(filename, config.get("default"))
    if entry is None:
        return {"status": "unavailable"}
    try:
        if list(entry["image_size"]) != list(size):
            raise ValueError("image_size_mismatch")
        if abs(float(entry["undistortion_alpha"]) - alpha) > 1e-9:
            raise ValueError("undistortion_alpha_mismatch")
        corners = np.asarray(entry["corners"], dtype=float)
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            raise ValueError("invalid_corners")
        edges = np.roll(corners, -1, axis=0) - corners
        following = np.roll(edges, -1, axis=0)
        cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
        if not np.all(cross > 1):
            raise ValueError("corners_must_be_convex_TL_TR_BR_BL")
        equations, values = [], []
        for (x, y), (u, v) in zip(corners, ((0, 0), (1, 0), (1, 1), (0, 1))):
            equations.extend(((x, y, 1, 0, 0, 0, -u*x, -u*y),
                              (0, 0, 0, x, y, 1, -v*x, -v*y)))
            values.extend((u, v))
        matrix = np.append(np.linalg.solve(equations, values), 1).reshape(3, 3)
        width, height = size
        clipped = bool(np.any(corners < 2) or np.any(corners[:, 0] >= width - 2)
                       or np.any(corners[:, 1] >= height - 2))
        return {"status": "clipped" if clipped else "valid",
                "corners": corners.tolist(), "image_size": list(size),
                "undistortion_alpha": alpha, "matrix": matrix.tolist()}
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as error:
        return {"status": "invalid", "reason": str(error)}


def screen_cell(center, geometry: dict, positions) -> int | None:
    if geometry.get("status") not in ("valid", "clipped"):
        return None
    point = np.asarray(geometry["matrix"]) @ np.array([*center, 1.0])
    if not np.isfinite(point).all() or abs(point[2]) < 1e-12:
        return None
    u, v = point[:2] / point[2]
    if not (0 <= u < 1 and 0 <= v < 1):
        return None
    rows = max(row for row, _ in positions) + 1
    columns = max(column for _, column in positions) + 1
    return list(positions).index((int(v * rows), int(u * columns)))


def _overlap(a, b) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right-left) * max(0, bottom-top)
    area = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / area if area > 0 else 0.0


def detection_evidence(observations: list[dict], size: tuple[int, int],
                       geometry: dict) -> dict:
    """Keep every detection, grouping overlapping full-frame/crop retries."""
    width, height = size
    detections, groups = [], []
    for observation in observations:
        box = np.asarray(observation["bbox"], dtype=float).tolist()
        raw = observation.get("raw")
        # Overlap defines a physical region; conflicting payloads are retained.
        group = next((index for index, members in enumerate(groups)
                      if any(_overlap(box, detections[m]["bbox"]) >= 0.5 for m in members)), None)
        if group is None:
            group = len(groups)
            groups.append([])
        clipped = box[0] <= 2 or box[1] <= 2 or box[2] >= width-2 or box[3] >= height-2
        original = observation.get("original_points")
        if original is not None:
            points = np.asarray(original)
            clipped = clipped or bool(np.any(points < 2) or np.any(points[:, 0] >= width-2)
                                      or np.any(points[:, 1] >= height-2))
        detections.append({
            "group": group, "raw": raw, "bbox": box,
            "original_points": None if original is None else np.asarray(original).tolist(),
            "confidence": float(observation.get("confidence", 0)),
            "detected_cell": int(observation["cell"]),
            "position_basis": observation.get("position_basis", "camera_image_grid"),
            "screen_cell": observation.get("screen_cell"),
            "journal_cell": (observation.get("marker") or {}).get("cell"),
            "display_index": observation.get("display_index"),
            "clipped": bool(clipped),
        })
        groups[group].append(len(detections)-1)
    unreadable, conflicts = [], []
    for index, members in enumerate(groups):
        identities = {detections[m]["display_index"] for m in members
                      if detections[m]["display_index"] is not None}
        if not identities:
            unreadable.append(index)
        if len(identities) > 1:
            conflicts.append(index)
    return {"version": EVIDENCE_VERSION, "image_size": list(size),
            "geometry": geometry, "detections": detections,
            "physical_detection_groups": len(groups), "unreadable_groups": unreadable,
            "conflicting_groups": conflicts}


def repeated_state_checks(frames: list[dict], presentation_times: list[int],
                          paused_indices=()) -> list[dict | None]:
    """Flag sustained newest-QR repeats without fitting a camera correction.

    Inputs use latest_display_index, media_reference_monotonic_ns, and segment.
    A repeat must exceed the local journal interval plus one median display
    period (at least two periods). This tolerates normal camera oversampling.
    All members are suspect: a repeated old readable QR cannot establish which
    image, if any, represents its original presentation interval. This is an
    offline exclusion, not proof of a display freeze or a runtime timing input.
    """
    checks = [None] * len(frames)
    if len(presentation_times) < 2:
        return checks
    periods = np.diff(np.asarray(presentation_times, dtype=np.int64))
    if np.any(periods <= 0):
        return checks
    period = float(np.median(periods))
    paused = set(paused_indices)
    run = []

    def finish():
        if len(run) < 2:
            return
        latest = frames[run[0]]["latest_display_index"]
        if latest in paused or not 0 <= latest < len(presentation_times) - 1:
            return
        span = (frames[run[-1]]["media_reference_monotonic_ns"]
                - frames[run[0]]["media_reference_monotonic_ns"])
        interval = presentation_times[latest + 1] - presentation_times[latest]
        threshold = max(2 * period, interval + period)
        if span <= threshold:
            return
        detail = {"status": "suspected_stale_visual_state",
                  "reason": "newest_qr_repeated_beyond_presentation_interval",
                  "latest_display_index": latest,
                  "camera_span_ms": span / 1e6,
                  "threshold_ms": threshold / 1e6,
                  "run_frames": len(run),
                  "cause": "display_hold_camera_repeat_or_missing_newer_qr_unresolved"}
        for position in run:
            checks[position] = dict(detail)

    for position, frame in enumerate(frames):
        latest = frame.get("latest_display_index")
        reference = frame.get("media_reference_monotonic_ns")
        if latest is None or reference is None or reference < 0:
            finish()
            run = []
            continue
        if run:
            previous = frames[run[-1]]
            if (latest != previous["latest_display_index"]
                    or frame.get("segment") != previous.get("segment")
                    or not 0 < reference - previous["media_reference_monotonic_ns"] <= 1_000_000_000):
                finish()
                run = []
        run.append(position)
    finish()
    return checks


def assess_evidence(frame: dict, indices: list[int], *, transition: bool = False) -> dict:
    """Score the newest readable QR; retain older/partial artifacts as warnings."""
    evidence = frame.get("qr_evidence")
    reasons = []
    warnings = []
    if not indices:
        return {"status": "no_reference", "reasons": ["no_matched_qr"], "primary_usable": False}
    if not isinstance(evidence, dict) or evidence.get("version") != EVIDENCE_VERSION:
        reasons.append("pixel_evidence_unavailable_rerun_decoder")
    else:
        detections = evidence.get("detections", [])
        matched = {d.get("display_index") for d in detections if d.get("display_index") is not None}
        if matched != set(indices):
            reasons.append("saved_identities_disagree_with_pixel_evidence")
        if evidence.get("unreadable_groups"):
            warnings.append("unreadable_regions_may_contain_newer_qr")
        if evidence.get("conflicting_groups"):
            reasons.append("conflicting_payloads_in_one_region")
        if any(d.get("clipped") for d in detections):
            warnings.append("clipped_qr_regions")
        geometry = evidence.get("geometry", {})
        if geometry.get("status") in ("invalid", "clipped"):
            reasons.append("screen_geometry_" + geometry["status"])
        if geometry.get("status") == "valid" and any(
            d.get("display_index") is not None and d.get("screen_cell") != d.get("journal_cell")
            for d in detections
        ):
            reasons.append("screen_geometry_disagrees_with_journal")
    temporal = frame.get("temporal_evidence") or {}
    if temporal.get("status") == "suspected_stale_visual_state":
        reasons.append("newest_qr_repeated_beyond_presentation_interval")
    if frame.get("manual_values"):
        reasons.append("manual_identity_requires_separate_review")
    if transition:
        warnings.append("multiple_generations_without_common_visibility")
    if temporal.get("status") == "suspected_stale_visual_state":
        status = "suspected_stale_visual_state"
    elif reasons:
        status = "potentially_missing_newer_generation" if evidence else "unknown_pixel_evidence"
    elif warnings:
        status = "usable_newest_readable_with_artifacts"
    else:
        status = "usable_conditional"
    return {"status": status, "reasons": reasons, "primary_usable": not reasons,
            "warnings": warnings,
            "geometry_verified": bool(evidence and evidence.get("geometry", {}).get("status") == "valid")}
