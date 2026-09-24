"""Pixel evidence and geometry checks, independent of timing-model fitting."""

import numpy as np

EVIDENCE_VERSION = 1


def screen_geometry(config: dict | None, filename: str, size: tuple[int, int], alpha: float) -> dict:
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
        next_edges = np.roll(edges, -1, axis=0)
        cross = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
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
        return {"status": "clipped" if clipped else "valid", "corners": corners.tolist(),
                "image_size": list(size), "undistortion_alpha": alpha, "matrix": matrix.tolist()}
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as error:
        return {"status": "invalid", "reason": str(error)}


def screen_cell(center, geometry: dict, positions) -> int | None:
    if geometry.get("status") not in ("valid", "clipped"):
        return None
    point = np.asarray(geometry["matrix"], dtype=float) @ np.array([*center, 1.0])
    if not np.isfinite(point).all() or abs(point[2]) < 1e-12:
        return None
    u, v = point[:2] / point[2]
    if not 0 <= u < 1 or not 0 <= v < 1:
        return None
    rows = max(row for row, _ in positions) + 1
    columns = max(column for _, column in positions) + 1
    return list(positions).index((int(v * rows), int(u * columns)))


def _overlap(first, second) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = ((first[2] - first[0]) * (first[3] - first[1])
             + (second[2] - second[0]) * (second[3] - second[1]) - intersection)
    return intersection / union if union > 0 else 0.0


def detection_evidence(observations: list[dict], size: tuple[int, int], geometry: dict) -> dict:
    width, height = size
    detections, groups = [], []
    for row in observations:
        box = np.asarray(row["bbox"], dtype=float).reshape(4).tolist()
        group = next((i for i, members in enumerate(groups)
                      if any(_overlap(box, detections[j]["bbox"]) >= 0.5 for j in members)), None)
        if group is None:
            group = len(groups)
            groups.append([])
        points = row.get("original_points")
        clipped = box[0] <= 2 or box[1] <= 2 or box[2] >= width - 2 or box[3] >= height - 2
        if points is not None:
            points = np.asarray(points)
            clipped |= bool(np.any(points < 2) or np.any(points[:, 0] >= width - 2)
                            or np.any(points[:, 1] >= height - 2))
        detections.append({
            "group": group, "raw": row.get("raw"), "bbox": box,
            "original_points": None if points is None else points.tolist(),
            "confidence": float(row.get("confidence", 0)),
            "detection_method": row.get("detection_method", "qr_reader"),
            "detected_cell": int(row["cell"]),
            "position_basis": row.get("position_basis", "camera_image_grid"),
            "screen_cell": row.get("screen_cell"),
            "journal_cell": (row.get("marker") or {}).get("cell"),
            "display_index": row.get("display_index"), "clipped": bool(clipped),
        })
        groups[group].append(len(detections) - 1)
    unreadable, conflicts = [], []
    for index, members in enumerate(groups):
        identities = {detections[n]["display_index"] for n in members
                      if detections[n]["display_index"] is not None}
        if not identities:
            unreadable.append(index)
        if len(identities) > 1:
            conflicts.append(index)
    return {"version": EVIDENCE_VERSION, "image_size": list(size), "geometry": geometry,
            "detections": detections, "physical_detection_groups": len(groups),
            "unreadable_groups": unreadable, "conflicting_groups": conflicts}


def repeated_state_checks(frames: list[dict], presentation_times: list[int], paused_indices=()) -> list[dict | None]:
    checks = [None] * len(frames)
    if len(presentation_times) < 2:
        return checks
    periods = np.diff(np.asarray(presentation_times, dtype=np.int64))
    if np.any(periods <= 0):
        return checks
    period = float(np.median(periods))
    paused, run = set(paused_indices), []

    def finish():
        if len(run) < 2:
            return
        first = frames[run[0]]
        latest = first.get("latest_display_index")
        if latest in paused or latest is None or not 0 <= latest < len(presentation_times) - 1:
            return
        start = first.get("media_reference_monotonic_ns")
        end = frames[run[-1]].get("media_reference_monotonic_ns")
        if start is None or end is None:
            return
        span = end - start
        interval = presentation_times[latest + 1] - presentation_times[latest]
        threshold = max(2 * period, interval + period)
        if span <= threshold:
            return
        detail = {"status": "suspected_stale_visual_state",
                  "reason": "newest_qr_repeated_beyond_presentation_interval",
                  "latest_display_index": latest, "camera_span_ms": span / 1e6,
                  "threshold_ms": threshold / 1e6, "run_frames": len(run),
                  "cause": "display_hold_camera_repeat_or_missing_newer_qr_unresolved"}
        for position in run:
            checks[position] = dict(detail)

    for index, frame in enumerate(frames):
        latest, reference = frame.get("latest_display_index"), frame.get("media_reference_monotonic_ns")
        if latest is None or reference is None or reference < 0:
            finish()
            run = []
            continue
        if run:
            previous = frames[run[-1]]
            old_reference = previous.get("media_reference_monotonic_ns")
            if (latest != previous.get("latest_display_index")
                    or frame.get("segment") != previous.get("segment")
                    or old_reference is None or not 0 < reference - old_reference <= 1_000_000_000):
                finish()
                run = []
        run.append(index)
    finish()
    return checks


def assess_evidence(frame: dict, indices: list[int], *, transition: bool = False) -> dict:
    if not indices:
        return {"status": "no_reference", "reasons": ["no_matched_qr"], "primary_usable": False}
    evidence = frame.get("qr_evidence")
    reasons, warnings = [], []
    if not isinstance(evidence, dict) or evidence.get("version") != EVIDENCE_VERSION:
        reasons.append("pixel_evidence_unavailable_rerun_decoder")
    else:
        detections = evidence.get("detections", [])
        matched = {row.get("display_index") for row in detections if row.get("display_index") is not None}
        if matched != set(indices):
            reasons.append("saved_identities_disagree_with_pixel_evidence")
        if evidence.get("unreadable_groups"):
            warnings.append("unreadable_regions_may_contain_newer_qr")
        if evidence.get("conflicting_groups"):
            reasons.append("conflicting_payloads_in_one_region")
        if any(row.get("clipped") for row in detections):
            warnings.append("clipped_qr_regions")
        geometry = evidence.get("geometry", {})
        if geometry.get("status") in ("invalid", "clipped"):
            reasons.append("screen_geometry_" + geometry["status"])
        if geometry.get("status") == "valid" and any(
            row.get("display_index") is not None and row.get("screen_cell") != row.get("journal_cell")
            for row in detections
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
    else:
        status = "usable_newest_readable_with_artifacts" if warnings else "usable_conditional"
    return {"status": status, "reasons": reasons, "primary_usable": not reasons,
            "warnings": warnings,
            "geometry_verified": bool(evidence and evidence.get("geometry", {}).get("status") == "valid")}
