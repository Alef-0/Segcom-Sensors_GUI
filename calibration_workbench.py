from __future__ import annotations

import argparse
from bisect import bisect_left
import math
from pathlib import Path
import re
from typing import Any

import cv2 as cv
import FreeSimpleGUI as sg
import numpy as np

from processing.playback.playback import PlaybackEntry, load_recording_entries
from processing.recording.point_cloud_reader import PointCloudReader
from processing.visualization.extrinsic_calibrator import (
    CommonPointCorrespondence,
    ExtrinsicCalibrationResult,
    auto_match_sift_overlap,
    scale_pixel_to_native,
    solve_lateral_camera_extrinsic,
)
from processing.visualization.graph_draw import Graph_radar
from processing.visualization.multi_radar_fusion import (
    CHANNEL_TO_POD,
    DEFAULT_CAM_B_MATRIX_PATH,
    DEFAULT_EXTRINSICS_PATH,
    POD_COLORS_BGR,
    POD_TO_CHANNEL,
    MultiSensorRegistry,
)
from processing.visualization.radar_camera_transformer import RadarCameraTransformer

CAM_VIEW_W = 512
CAM_VIEW_H = 288
RADAR_VIEW_SIZE = 288


def _encode_png(image_bgr: np.ndarray) -> bytes:
    ok, buf = cv.imencode(".png", image_bgr)
    return buf.tobytes() if ok else b""


def discover_paired_recording_entries(
    selected_folder: str | Path,
    target_channel: int,
    max_sync_diff_s: float = 0.50,
) -> list[dict[str, Any]]:
    """Find time-paired (Group B, Group A/C) recording entries from a session or parent folder."""
    root = Path(selected_folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Select an existing recording directory")

    target_pod = CHANNEL_TO_POD[int(target_channel)]
    folder_b: Path | None = None
    folder_s: Path | None = None

    match = re.match(r"^recording_([ABC])_(.+)$", root.name)
    if match:
        suffix = match.group(2)
        cand_b = root.parent / f"recording_B_{suffix}"
        cand_s = root.parent / f"recording_{target_pod}_{suffix}"
        if cand_b.is_dir() and cand_s.is_dir():
            folder_b, folder_s = cand_b, cand_s

    if folder_b is None or folder_s is None:
        b_dirs = sorted(root.glob("recording_B_*"), reverse=True)
        for cand_b in b_dirs:
            suffix = cand_b.name[len("recording_B_"):]
            cand_s = root / f"recording_{target_pod}_{suffix}"
            if cand_s.is_dir():
                folder_b, folder_s = cand_b, cand_s
                break

    if folder_b is None or folder_s is None:
        raise ValueError(
            f"Could not find paired recording_B_<timestamp> and recording_{target_pod}_<timestamp> folders"
        )

    entries_b = [e for e in load_recording_entries(folder_b) if e.camera_frame is not None]
    entries_s = [e for e in load_recording_entries(folder_s) if e.camera_frame is not None]
    if not entries_b or not entries_s:
        raise ValueError("Both paired recording folders must contain recorded camera frames")

    times_s = [e.recorded_at.timestamp() for e in entries_s]
    pairs: list[dict[str, Any]] = []
    for eb in entries_b:
        tb = eb.recorded_at.timestamp()
        pos = bisect_left(times_s, tb)
        candidates = [entries_s[i] for i in (pos - 1, pos) if 0 <= i < len(entries_s)]
        if not candidates:
            continue
        best_s = min(candidates, key=lambda es: abs(es.recorded_at.timestamp() - tb))
        if abs(best_s.recorded_at.timestamp() - tb) <= max_sync_diff_s:
            pairs.append({"entry_b": eb, "entry_s": best_s})

    if not pairs:
        raise ValueError(f"No synchronized camera frames within {max_sync_diff_s * 1000:.0f} ms were found")
    return pairs


def _points_from_pcd_entry(
    entry: PlaybackEntry,
    channel: int,
    registry: MultiSensorRegistry,
    distance_cutoff: float = 25.0,
) -> tuple[dict[str, Any], ...]:
    """Load PCD points for an offline PlaybackEntry and map them to unified B coordinates."""
    if entry.point_cloud is None or not entry.point_cloud.is_file():
        return ()
    reader = PointCloudReader(entry.point_cloud)
    pose = registry.get_radar_pose(channel)
    color = POD_COLORS_BGR[int(channel)]
    out: list[dict[str, Any]] = []
    for pt in reader.points():
        lat = getattr(pt, "dist_latitude", None)
        lon = getattr(pt, "dist_long", None)
        if lat is None or lon is None or not (math.isfinite(lat) and math.isfinite(lon)):
            continue
        if math.hypot(lat, lon) > distance_cutoff:
            continue
        uni = pose.transform_xy([[lat, lon]])[0]
        out.append({
            "channel": int(channel),
            "local_lat": float(lat),
            "local_long": float(lon),
            "unified_lat_b": float(uni[0]),
            "unified_long_b": float(uni[1]),
            "rcs": getattr(pt, "rcs", None),
            "color": color,
        })
    return tuple(out)


class CalibrationWorkbench:
    """Non-blocking FreeSimpleGUI workbench for common-point lateral camera calibration."""

    def __init__(
        self,
        registry: MultiSensorRegistry,
        *,
        target_channel: int = 3,
        request_live_pair_cb=None,
        on_saved_cb=None,
    ) -> None:
        self.registry = registry
        self.target_channel = 1 if int(target_channel) == 1 else 3
        self.request_live_pair_cb = request_live_pair_cb
        self.on_saved_cb = on_saved_cb
        self.cam_b: RadarCameraTransformer = self.registry.load_camera_b()
        self.frame_b: np.ndarray | None = None
        self.frame_s: np.ndarray | None = None
        self.radar_points_b: tuple[dict[str, Any], ...] = ()
        self.radar_points_s: tuple[dict[str, Any], ...] = ()
        self.correspondences: list[CommonPointCorrespondence] = []
        self._next_id = 1
        self.pending_pixel_b: tuple[float, float] | None = None
        self.pending_radar_b: tuple[float, float] | None = None
        self.last_result: ExtrinsicCalibrationResult | None = None
        self.offline_pairs: list[dict[str, Any]] = []
        self.offline_idx = 0
        self.radar_canvas = Graph_radar(25.0, RADAR_VIEW_SIZE, RADAR_VIEW_SIZE, 15.0, 25.0)
        self._displayed_radar_targets: list[dict[str, Any]] = []
        self.window = self._build_window()
        self._sync_shift_inputs_from_registry()
        self._redraw_all()

    def _build_window(self) -> sg.Window:
        pod = CHANNEL_TO_POD[self.target_channel]
        top_bar = [
            sg.Text("Target Pod:"),
            sg.Radio("Left A", "wb_pod", key="wb_pod_1", default=self.target_channel == 1, enable_events=True),
            sg.Radio("Right C", "wb_pod", key="wb_pod_3", default=self.target_channel == 3, enable_events=True),
            sg.VSep(),
            sg.Button("GRAB LIVE PAIR (B + Target)", key="wb_grab_live", button_color=("white", "green")),
            sg.VSep(),
            sg.Text("Session Folder:"),
            sg.Input(str(Path.cwd() / "recordings"), key="wb_folder", size=(28, 1)),
            sg.FolderBrowse("SELECT", key="wb_browse", target="wb_folder"),
            sg.Button("LOAD", key="wb_load_folder"),
            sg.Button("<", key="wb_prev_frame", disabled=True),
            sg.Text("0/0", key="wb_frame_idx", size=(7, 1), justification="center"),
            sg.Button(">", key="wb_next_frame", disabled=True),
        ]
        canvases = [
            sg.Column([
                [sg.Text("Middle Camera B (Calibrated Reference)", justification="center", expand_x=True)],
                [sg.Graph((CAM_VIEW_W, CAM_VIEW_H), (0, CAM_VIEW_H), (CAM_VIEW_W, 0), key="wb_graph_b", enable_events=True)],
            ]),
            sg.Column([
                [sg.Text(f"Lateral Camera {pod} (Target + Live Overlay)", key="wb_title_s", justification="center", expand_x=True)],
                [sg.Graph((CAM_VIEW_W, CAM_VIEW_H), (0, CAM_VIEW_H), (CAM_VIEW_W, 0), key="wb_graph_s", enable_events=True)],
            ]),
            sg.Column([
                [sg.Text("Unified Radar Picker", justification="center", expand_x=True)],
                [sg.Graph((RADAR_VIEW_SIZE, RADAR_VIEW_SIZE), (0, RADAR_VIEW_SIZE), (RADAR_VIEW_SIZE, 0), key="wb_graph_radar", enable_events=True)],
            ]),
        ]
        mode_row = [
            sg.Text("Point Mode:"),
            sg.Radio("Ground Z (CamB+CamS)", "wb_mode", key="wb_mode_ground", default=True, enable_events=True),
            sg.Radio("Radar-Anchored 3D (Rad+CamB+CamS)", "wb_mode", key="wb_mode_anchored", enable_events=True),
            sg.Radio("Radar->CamS (Outer FoV)", "wb_mode", key="wb_mode_rad2cam", enable_events=True),
            sg.Radio("Visual Epipolar (CamB+CamS)", "wb_mode", key="wb_mode_epipolar", enable_events=True),
            sg.Text("Z_down (m):"),
            sg.Input("0.00", key="wb_z_down", size=(6, 1), justification="right"),
            sg.Button("AUTO-MATCH SIFT", key="wb_auto_sift"),
        ]
        solver_row = [
            sg.Text("Solver:"),
            sg.Radio("3-DoF Rotation + Prior", "wb_solver", key="wb_solver_3dof", default=True, enable_events=True),
            sg.Radio("Full 6-DoF Hybrid", "wb_solver", key="wb_solver_6dof", enable_events=True),
            sg.VSep(),
            sg.Text("Prior:"),
            sg.Radio("Cam B Shift", "wb_prior", key="wb_prior_shift", default=True),
            sg.Radio("Pod Bracket", "wb_prior", key="wb_prior_bracket"),
            sg.Text("dLong:"),
            sg.Input("0.00", key="wb_dlong", size=(6, 1), justification="right"),
            sg.Text("dLat:"),
            sg.Input("0.25", key="wb_dlat", size=(6, 1), justification="right"),
            sg.Text("dDown:"),
            sg.Input("0.00", key="wb_ddown", size=(6, 1), justification="right"),
            sg.Text("m"),
        ]
        list_row = [
            sg.Listbox(values=[], size=(95, 5), key="wb_corr_list", expand_x=True),
            sg.Column([
                [sg.Button("TOGGLE ON/OFF", key="wb_corr_toggle", expand_x=True)],
                [sg.Button("DELETE", key="wb_corr_delete", expand_x=True)],
                [sg.Button("CLEAR ALL", key="wb_corr_clear", expand_x=True)],
            ]),
            sg.Column([
                [sg.Button("SOLVE & PREVIEW", key="wb_solve", button_color=("white", "blue"), expand_x=True)],
                [sg.Button("SAVE EXTRINSICS", key="wb_save", button_color=("white", "green"), expand_x=True, disabled=True)],
                [sg.Button("CLOSE", key="wb_close", expand_x=True)],
            ]),
        ]
        status_rows = [
            [sg.Text("Step: Load or grab a synchronized frame pair, then click common points.", key="wb_prompt", expand_x=True)],
            [sg.Text("Diagnostics: uncalibrated", key="wb_diagnostics", expand_x=True)],
        ]
        layout = [top_bar, canvases, mode_row, solver_row, list_row, *status_rows]
        return sg.Window("Lateral Camera Extrinsic Calibration Workbench", layout, finalize=True)

    @property
    def closed(self) -> bool:
        return self.window is None

    def close(self) -> None:
        if self.window is not None:
            self.window.close()
            self.window = None

    def _sync_shift_inputs_from_registry(self) -> None:
        if self.window is None:
            return
        entry = self.registry.lateral_cameras.get(self.target_channel, {})
        shift = entry.get("camera_shift_from_b_m") or {}
        default_lat = -0.25 if self.target_channel == 1 else 0.25
        self.window["wb_dlong"].update(f"{float(shift.get('d_long_m', 0.0)):.2f}")
        self.window["wb_dlat"].update(f"{float(shift.get('d_lat_m', default_lat)):.2f}")
        self.window["wb_ddown"].update(f"{float(shift.get('d_down_m', 0.0)):.2f}")
        pod = CHANNEL_TO_POD[self.target_channel]
        self.window["wb_title_s"].update(f"Lateral Camera {pod} (Target + Live Overlay)")

    def receive_live_pair(self, payload: dict[str, Any]) -> None:
        """Accept synchronized JPEG + radar frame payload from the camera/radar workers."""
        if self.window is None:
            return
        arr_b = np.frombuffer(payload["image_b_bytes"], dtype=np.uint8)
        arr_s = np.frombuffer(payload["image_s_bytes"], dtype=np.uint8)
        self.frame_b = cv.imdecode(arr_b, cv.IMREAD_COLOR)
        self.frame_s = cv.imdecode(arr_s, cv.IMREAD_COLOR)
        self.radar_points_b = tuple(payload.get("radar_points_b", ()))
        self.radar_points_s = tuple(payload.get("radar_points_s", ()))
        self.pending_pixel_b = None
        self.pending_radar_b = None
        pod = CHANNEL_TO_POD[self.target_channel]
        self.window["wb_prompt"].update(
            f"Captured live B + {pod} pair "
            f"(Radar B: {len(self.radar_points_b)} pts, Radar {pod}: {len(self.radar_points_s)} pts). "
            "Click common points or run AUTO-MATCH SIFT."
        )
        self._redraw_all()

    def report_error(self, message: str) -> None:
        if self.window is not None:
            self.window["wb_prompt"].update(f"ERROR: {message}")

    def _active_corr_mode(self, values: dict[str, Any]) -> str:
        if values.get("wb_mode_anchored"):
            return "radar_anchored"
        if values.get("wb_mode_rad2cam"):
            return "radar_to_cam"
        if values.get("wb_mode_epipolar"):
            return "epipolar"
        return "ground_plane"

    def _load_offline_index(self, idx: int) -> None:
        if not self.offline_pairs or self.window is None:
            return
        self.offline_idx = max(0, min(int(idx), len(self.offline_pairs) - 1))
        pair = self.offline_pairs[self.offline_idx]
        eb: PlaybackEntry = pair["entry_b"]
        es: PlaybackEntry = pair["entry_s"]
        self.frame_b = cv.imread(str(eb.camera_frame), cv.IMREAD_COLOR)
        self.frame_s = cv.imread(str(es.camera_frame), cv.IMREAD_COLOR)
        self.radar_points_b = _points_from_pcd_entry(eb, 2, self.registry)
        self.radar_points_s = _points_from_pcd_entry(es, self.target_channel, self.registry)
        self.pending_pixel_b = None
        self.pending_radar_b = None
        self.window["wb_frame_idx"].update(f"{self.offline_idx + 1}/{len(self.offline_pairs)}")
        self.window["wb_prev_frame"].update(disabled=self.offline_idx <= 0)
        self.window["wb_next_frame"].update(disabled=self.offline_idx >= len(self.offline_pairs) - 1)
        self._redraw_all()

    def _current_target_transformer(self) -> RadarCameraTransformer:
        if self.last_result is not None:
            return RadarCameraTransformer(
                self.cam_b.intrinsic,
                self.cam_b.distortion,
                self.last_result.extrinsic_from_radar_b,
            )
        transformer, _ = self.registry.get_camera_transformer(self.target_channel, radar_source="global")
        return transformer

    def _draw_camera_view(self, frame: np.ndarray | None, *, is_cam_b: bool) -> np.ndarray:
        if frame is None:
            blank = np.full((CAM_VIEW_H, CAM_VIEW_W, 3), 38, dtype=np.uint8)
            label = "No Camera B Frame" if is_cam_b else f"No Camera {CHANNEL_TO_POD[self.target_channel]} Frame"
            cv.putText(blank, label, (CAM_VIEW_W // 2 - 95, CAM_VIEW_H // 2), cv.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 1)
            return blank

        nh, nw = frame.shape[:2]
        canvas = cv.resize(frame, (CAM_VIEW_W, CAM_VIEW_H), interpolation=cv.INTER_AREA)
        sx, sy = CAM_VIEW_W / float(nw), CAM_VIEW_H / float(nh)

        # Overlay radar points in unified B frame onto camera view
        transformer = self.cam_b if is_cam_b else self._current_target_transformer()
        all_rad = (*self.radar_points_b, *self.radar_points_s) if not is_cam_b else self.radar_points_b
        if all_rad:
            pts_3d = np.array([[p["unified_long_b"], p["unified_lat_b"], 0.0] for p in all_rad], dtype=np.float64)
            cam_pts = transformer.radar_to_camera(pts_3d)
            vis = cam_pts[:, 2] > 0.1
            if np.any(vis):
                proj = transformer.radar_to_image(pts_3d[vis], distorted=True)
                for px, item in zip(proj, [p for p, k in zip(all_rad, vis) if k]):
                    vx, vy = int(round(px[0] * sx)), int(round(px[1] * sy))
                    if 0 <= vx < CAM_VIEW_W and 0 <= vy < CAM_VIEW_H:
                        col = POD_COLORS_BGR.get(item["channel"], (0, 255, 0))
                        cv.circle(canvas, (vx, vy), 5, (255, 255, 255), 1)
                        cv.circle(canvas, (vx, vy), 3, col, -1)

        # Draw correspondence markers
        for corr in self.correspondences:
            px_nat = corr.pixel_b if is_cam_b else corr.pixel_s
            if px_nat is None:
                continue
            vx, vy = int(round(px_nat[0] * sx)), int(round(px_nat[1] * sy))
            col = (0, 230, 255) if corr.enabled else (140, 140, 140)
            cv.drawMarker(canvas, (vx, vy), col, cv.MARKER_CROSS, 12, 2)
            cv.putText(canvas, f"#{corr.corr_id}", (vx + 5, vy - 5), cv.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

        if is_cam_b and self.pending_pixel_b is not None:
            vx = int(round(self.pending_pixel_b[0] * sx))
            vy = int(round(self.pending_pixel_b[1] * sy))
            cv.circle(canvas, (vx, vy), 7, (0, 255, 0), 2)
        return canvas

    def _draw_radar_picker(self) -> np.ndarray:
        canvas = self.radar_canvas.create_base_image()
        self._displayed_radar_targets = []
        for item in (*self.radar_points_b, *self.radar_points_s):
            ux, uy = item["unified_lat_b"], item["unified_long_b"]
            px = self.radar_canvas.graph_to_pixel(ux, uy)
            col = POD_COLORS_BGR.get(item["channel"], (0, 160, 0))
            cv.circle(canvas, px, 5, col, -1)
            self._displayed_radar_targets.append({"pixel": px, "long_b": uy, "lat_b": ux, "channel": item["channel"]})

        if self.pending_radar_b is not None:
            sel_px = self.radar_canvas.graph_to_pixel(self.pending_radar_b[1], self.pending_radar_b[0])
            cv.circle(canvas, sel_px, 9, (0, 220, 0), 2)
        return canvas

    def _refresh_corr_listbox(self) -> None:
        if self.window is None:
            return
        errs = self.last_result.per_point_errors_px if self.last_result else {}
        lines = []
        for c in self.correspondences:
            state = "ON " if c.enabled else "OFF"
            err_txt = f"{errs[c.corr_id]:.2f} px" if c.corr_id in errs else "--"
            rad_txt = (
                f"Rad({c.radar_b_long_lat[0]:.1f}m,{c.radar_b_long_lat[1]:+.1f}m)"
                if c.radar_b_long_lat is not None
                else f"Z={c.plane_z_down_m:+.2f}m"
            )
            lines.append(f"#{c.corr_id:02d} [{state}] {c.mode:<14} | {rad_txt} | err={err_txt} {c.label}")
        self.window["wb_corr_list"].update(values=lines)

    def _redraw_all(self) -> None:
        if self.window is None:
            return
        for key, img in (
            ("wb_graph_b", self._draw_camera_view(self.frame_b, is_cam_b=True)),
            ("wb_graph_s", self._draw_camera_view(self.frame_s, is_cam_b=False)),
            ("wb_graph_radar", self._draw_radar_picker()),
        ):
            graph_elem = self.window[key]
            graph_elem.erase()
            graph_elem.draw_image(data=_encode_png(img), location=(0, 0))
        self._refresh_corr_listbox()

    def _parse_floats(self, values: dict[str, Any]) -> tuple[float, tuple[float, float, float]]:
        z_down = float(str(values.get("wb_z_down", "0.0")).strip())
        dlong = float(str(values.get("wb_dlong", "0.0")).strip())
        dlat = float(str(values.get("wb_dlat", "0.0")).strip())
        ddown = float(str(values.get("wb_ddown", "0.0")).strip())
        return z_down, (dlong, dlat, ddown)

    def _run_solve(self, values: dict[str, Any]) -> None:
        if self.window is None:
            return
        try:
            _, shift_vec = self._parse_floats(values)
            solver_mode = "full_6dof" if values.get("wb_solver_6dof") else "rotation_3dof"
            prior_mode = "pod_bracket" if values.get("wb_prior_bracket") else "camera_b_shift"
            res = solve_lateral_camera_extrinsic(
                self.correspondences,
                self.cam_b,
                self.registry.get_radar_pose(self.target_channel),
                solver_mode=solver_mode,
                prior_mode=prior_mode,
                camera_shift_from_b_m=shift_vec,
            )
            self.last_result = res
            self.window["wb_save"].update(disabled=False)
            sh = res.shift_from_cam_b_m
            warn = f" | WARN: {res.warnings[0]}" if res.warnings else ""
            self.window["wb_diagnostics"].update(
                f"RMSE={res.rmse_px:.2f} px | Shift vs Cam B: dLong={sh[0]:+.2f}m, dLat={sh[1]:+.2f}m, dDown={sh[2]:+.2f}m | "
                f"Yaw={res.boresight_yaw_deg:+.2f}°, Pitch={res.boresight_pitch_deg:+.2f}°, Roll={res.boresight_roll_deg:+.2f}°{warn}"
            )
            self._redraw_all()
        except Exception as error:
            self.window["wb_diagnostics"].update(f"Solve error: {error}")

    def _selected_corr_index(self, values: dict[str, Any]) -> int | None:
        selected = values.get("wb_corr_list") or []
        if not selected:
            return None
        match = re.match(r"^#(\d+)", str(selected[0]))
        if not match:
            return None
        cid = int(match.group(1))
        return next((i for i, c in enumerate(self.correspondences) if c.corr_id == cid), None)

    def poll(self) -> None:
        """Process one non-blocking GUI event tick for the Calibration Workbench."""
        if self.window is None:
            return
        event, values = self.window.read(timeout=0)
        if event in (None, sg.WIN_CLOSED, "wb_close"):
            self.close()
            return
        if event == "__TIMEOUT__":
            return

        if event in ("wb_pod_1", "wb_pod_3"):
            new_ch = 1 if event == "wb_pod_1" else 3
            if new_ch != self.target_channel:
                self.target_channel = new_ch
                self.correspondences.clear()
                self.last_result = None
                self.window["wb_save"].update(disabled=True)
                self._sync_shift_inputs_from_registry()
                self._redraw_all()
            return

        if event == "wb_grab_live":
            if self.request_live_pair_cb is not None:
                self.window["wb_prompt"].update("Requesting live synchronized frame pair from Camera B + Target...")
                self.request_live_pair_cb(self.target_channel)
            return

        if event == "wb_load_folder":
            try:
                self.offline_pairs = discover_paired_recording_entries(values.get("wb_folder", ""), self.target_channel)
                self._load_offline_index(0)
                self.window["wb_prompt"].update(f"Loaded {len(self.offline_pairs)} synchronized frame pairs.")
            except Exception as error:
                self.report_error(str(error))
            return

        if event == "wb_prev_frame":
            self._load_offline_index(self.offline_idx - 1)
            return
        if event == "wb_next_frame":
            self._load_offline_index(self.offline_idx + 1)
            return

        if event == "wb_graph_radar" and self._displayed_radar_targets:
            click = values.get("wb_graph_radar")
            if click and None not in click:
                nearest = min(
                    self._displayed_radar_targets,
                    key=lambda t: (t["pixel"][0] - click[0]) ** 2 + (t["pixel"][1] - click[1]) ** 2,
                )
                self.pending_radar_b = (nearest["long_b"], nearest["lat_b"])
                self.window["wb_prompt"].update(
                    f"Selected Radar target (long={nearest['long_b']:.2f}m, lat={nearest['lat_b']:+.2f}m). "
                    "Now click the target in Camera B (for Radar-Anchored) or Camera S (for Radar->CamS)."
                )
                self._redraw_all()
            return

        if event == "wb_graph_b" and self.frame_b is not None:
            click = values.get("wb_graph_b")
            if click and None not in click:
                nh, nw = self.frame_b.shape[:2]
                self.pending_pixel_b = scale_pixel_to_native(click[0], click[1], (CAM_VIEW_W, CAM_VIEW_H), (nw, nh))
                self.window["wb_prompt"].update(
                    f"Selected Camera B pixel ({self.pending_pixel_b[0]:.0f}, {self.pending_pixel_b[1]:.0f}). "
                    f"Now click the matching point in Lateral Camera {CHANNEL_TO_POD[self.target_channel]}."
                )
                self._redraw_all()
            return

        if event == "wb_graph_s" and self.frame_s is not None:
            click = values.get("wb_graph_s")
            if not click or None in click:
                return
            try:
                z_down, _ = self._parse_floats(values)
            except ValueError:
                self.report_error("Z_down and shift fields must be numeric")
                return
            nh, nw = self.frame_s.shape[:2]
            px_s = scale_pixel_to_native(click[0], click[1], (CAM_VIEW_W, CAM_VIEW_H), (nw, nh))
            mode = self._active_corr_mode(values)
            if mode in ("ground_plane", "epipolar") and self.pending_pixel_b is None:
                self.report_error("Click the point in Middle Camera B first")
                return
            if mode == "radar_anchored" and (self.pending_pixel_b is None or self.pending_radar_b is None):
                self.report_error("Select a Radar target on the right AND click the object in Camera B first")
                return
            if mode == "radar_to_cam" and self.pending_radar_b is None:
                self.report_error("Select a Radar target on the Unified Radar Picker first")
                return

            self.correspondences.append(CommonPointCorrespondence(
                corr_id=self._next_id,
                mode=mode,
                pixel_s=px_s,
                pixel_b=self.pending_pixel_b if mode != "radar_to_cam" else None,
                radar_b_long_lat=self.pending_radar_b if mode in ("radar_anchored", "radar_to_cam") else None,
                plane_z_down_m=z_down,
            ))
            self._next_id += 1
            self.pending_pixel_b = None
            self.window["wb_prompt"].update(f"Added correspondence #{self._next_id - 1} ({mode}).")
            if sum(1 for c in self.correspondences if c.enabled) >= 2:
                self._run_solve(values)
            else:
                self._redraw_all()
            return

        if event == "wb_auto_sift":
            if self.frame_b is None or self.frame_s is None:
                self.report_error("Load or grab both Camera B and Lateral Camera frames before running SIFT")
                return
            matches = auto_match_sift_overlap(
                self.frame_b,
                self.frame_s,
                self.cam_b,
                start_id=self._next_id,
            )
            if not matches:
                self.report_error("No reliable SIFT overlap inliers found; click manual ground/radar points instead")
                return
            self.correspondences.extend(matches)
            self._next_id += len(matches)
            self.window["wb_prompt"].update(f"Added {len(matches)} RANSAC-verified SIFT overlap points.")
            self._run_solve(values)
            return

        if event == "wb_corr_toggle":
            idx = self._selected_corr_index(values)
            if idx is not None:
                self.correspondences[idx].enabled = not self.correspondences[idx].enabled
                self._redraw_all()
            return

        if event == "wb_corr_delete":
            idx = self._selected_corr_index(values)
            if idx is not None:
                self.correspondences.pop(idx)
                self._redraw_all()
            return

        if event == "wb_corr_clear":
            self.correspondences.clear()
            self.last_result = None
            self.window["wb_save"].update(disabled=True)
            self._redraw_all()
            return

        if event == "wb_solve":
            self._run_solve(values)
            return

        if event == "wb_save" and self.last_result is not None:
            try:
                _, shift_vec = self._parse_floats(values)
                used_shift = self.last_result.shift_from_cam_b_m if self.last_result.solver_mode == "full_6dof" else shift_vec
                self.registry.set_lateral_camera_calibration(
                    self.target_channel,
                    self.last_result.extrinsic_from_radar_b,
                    solver_mode=self.last_result.solver_mode,
                    prior_mode=self.last_result.prior_mode,
                    camera_shift_from_b_m=used_shift,
                    rmse_px=self.last_result.rmse_px,
                )
                self.registry.save()
                pod = CHANNEL_TO_POD[self.target_channel]
                self.window["wb_prompt"].update(
                    f"Saved Camera {pod} extrinsics to {self.registry.extrinsics_path.name} "
                    f"(RMSE {self.last_result.rmse_px:.2f} px)."
                )
                if self.on_saved_cb is not None:
                    self.on_saved_cb(self.target_channel, self.last_result)
            except Exception as error:
                self.report_error(str(error))


def parse_arguments(argv=None):
    """Parse CLI flags for standalone Calibration Workbench execution."""
    parser = argparse.ArgumentParser(description="Lateral Camera Extrinsic Calibration Workbench")
    parser.add_argument("--pod", choices=("A", "C"), default="C", help="Target lateral pod (A or C)")
    parser.add_argument("--folder", type=str, default=None, help="Optional recording session folder to preload")
    parser.add_argument("--extrinsics", type=str, default=str(DEFAULT_EXTRINSICS_PATH), help="Output extrinsics JSON")
    parser.add_argument("--cam-b", type=str, default=str(DEFAULT_CAM_B_MATRIX_PATH), help="Reference Camera B JSON")
    return parser.parse_args(argv)


def main(argv=None):
    """Run the Calibration Workbench as a standalone desktop utility."""
    args = parse_arguments(argv)
    sg.set_options(font=("Helvetica", 11))
    registry = MultiSensorRegistry(args.extrinsics, args.cam_b)
    wb = CalibrationWorkbench(registry, target_channel=POD_TO_CHANNEL[args.pod])
    if args.folder:
        try:
            wb.offline_pairs = discover_paired_recording_entries(args.folder, wb.target_channel)
            wb._load_offline_index(0)
        except Exception as error:
            wb.report_error(str(error))
    while not wb.closed:
        wb.poll()
        if not wb.closed:
            sg.Window.read_all_windows(timeout=15)


if __name__ == "__main__":
    main()
