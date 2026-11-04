from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from processing.visualization.radar_camera_transformer import RadarCameraTransformer

DEFAULT_EXTRINSICS_PATH = Path(__file__).resolve().parents[2] / "multi_sensor_extrinsics.json"
DEFAULT_CAM_B_MATRIX_PATH = Path(__file__).resolve().parents[2] / "camera_matrixes.json"
RADAR_CHANNELS = (1, 2, 3)
CHANNEL_TO_POD = {1: "A", 2: "B", 3: "C"}
POD_TO_CHANNEL = {"A": 1, "B": 2, "C": 3}
POD_COLORS_BGR = {
    1: (230, 140, 0),   # Pod A (Left): cyan-blue
    2: (40, 170, 40),   # Pod B (Middle): green
    3: (30, 80, 230),   # Pod C (Right): orange-red
}
DEFAULT_STALE_SECONDS = 0.50
DEFAULT_FOV_HALF_DEG = 60.0


@dataclass(frozen=True)
class RadarExtrinsic:
    """Planar and 3D pose of a radar pod relative to Middle Radar B (-Left, +Right)."""
    channel: int
    offset_x_m: float = 0.0   # Lateral offset relative to B (-Left, +Right)
    offset_y_m: float = 0.0   # Longitudinal offset relative to B (+Forward)
    offset_z_m: float = 0.0   # Vertical offset relative to B (+Down)
    yaw_deg: float = 0.0      # Azimuth yaw relative to B (-Left, +Right)

    @property
    def pod(self) -> str:
        return CHANNEL_TO_POD.get(self.channel, str(self.channel))

    def rotation_2d(self) -> np.ndarray:
        """Return 2x2 rotation mapping local [x_lat, y_long] to unified [X_lat, Y_long]."""
        rad = math.radians(float(self.yaw_deg))
        c, s = math.cos(rad), math.sin(rad)
        return np.array([[c, s], [-s, c]], dtype=np.float64)

    def transform_xy(self, local_xy: Any) -> np.ndarray:
        """Transform local (N, 2) [dist_latitude, dist_long] to unified B coordinates."""
        pts = np.asarray(local_xy, dtype=np.float64).reshape(-1, 2)
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        rot = self.rotation_2d()
        trans = np.array([self.offset_x_m, self.offset_y_m], dtype=np.float64)
        return (pts @ rot.T) + trans

    def inverse_transform_xy(self, unified_xy: Any) -> np.ndarray:
        """Transform unified B (N, 2) [X_lat, Y_long] back into local sensor coordinates."""
        pts = np.asarray(unified_xy, dtype=np.float64).reshape(-1, 2)
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        rot = self.rotation_2d()
        trans = np.array([self.offset_x_m, self.offset_y_m], dtype=np.float64)
        return (pts - trans) @ rot

    def transform_velocity(self, local_vel_xy: Any) -> np.ndarray:
        """Rotate local (N, 2) [vel_lat, vel_long] into unified B orientation."""
        vel = np.asarray(local_vel_xy, dtype=np.float64).reshape(-1, 2)
        if len(vel) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        return vel @ self.rotation_2d().T

    def matrix_radar3d_to_ref_b(self) -> np.ndarray:
        """Return 4x4 SE(3) matrix in 3D radar order [dist_long, dist_latitude, z_down, 1]."""
        rad = math.radians(float(self.yaw_deg))
        c, s = math.cos(rad), math.sin(rad)
        return np.array([
            [c, -s, 0.0, float(self.offset_y_m)],
            [s, c, 0.0, float(self.offset_x_m)],
            [0.0, 0.0, 1.0, float(self.offset_z_m)],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def matrix_ref_b_to_radar3d(self) -> np.ndarray:
        """Return inverse 4x4 SE(3) matrix from Reference B to this radar's 3D frame."""
        return np.linalg.inv(self.matrix_radar3d_to_ref_b())


def compute_prior_lateral_camera_extrinsic(
    cam_b: RadarCameraTransformer,
    radar_pose: RadarExtrinsic,
    *,
    prior_mode: str = "camera_b_shift",
    camera_shift_from_b_m: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Build prior (T_{cam_S<-radar_B}, T_{cam_S<-radar_S}) from Camera B and shift/pod prior."""
    r_b = cam_b.rotation_matrix
    t_b = cam_b.translation_vector.reshape(3)
    c_b = -(r_b.T @ t_b)  # Camera B optical center in Radar B [long, lat, down]
    t_b_from_s = radar_pose.matrix_radar3d_to_ref_b()
    r_pod = t_b_from_s[:3, :3]  # 3D yaw rotation around +Z_down

    if prior_mode == "pod_bracket":
        t_cam_from_local = cam_b.extrinsic.copy()
        t_cam_from_b = t_cam_from_local @ radar_pose.matrix_ref_b_to_radar3d()
        return t_cam_from_b, t_cam_from_local

    shift = np.asarray(camera_shift_from_b_m, dtype=np.float64).reshape(3)
    c_s = c_b + shift  # Shifted optical center in Radar B [long, lat, down]
    r_s = r_b @ r_pod.T  # Align camera boresight with lateral pod yaw
    t_s = -(r_s @ c_s)
    t_cam_from_b = np.eye(4, dtype=np.float64)
    t_cam_from_b[:3, :3] = r_s
    t_cam_from_b[:3, 3] = t_s
    t_cam_from_local = t_cam_from_b @ t_b_from_s
    return t_cam_from_b, t_cam_from_local


class MultiSensorRegistry:
    """Manages radar SE(2)/SE(3) poses and lateral camera extrinsics without touching Camera B JSON."""

    def __init__(
        self,
        extrinsics_path: str | Path = DEFAULT_EXTRINSICS_PATH,
        cam_b_path: str | Path = DEFAULT_CAM_B_MATRIX_PATH,
    ) -> None:
        self.extrinsics_path = Path(extrinsics_path)
        self.cam_b_path = Path(cam_b_path)
        self.radars: dict[int, RadarExtrinsic] = {
            1: RadarExtrinsic(1, offset_x_m=-0.25, offset_y_m=0.0, yaw_deg=0.0),
            2: RadarExtrinsic(2, offset_x_m=0.0, offset_y_m=0.0, yaw_deg=0.0),
            3: RadarExtrinsic(3, offset_x_m=0.25, offset_y_m=0.0, yaw_deg=0.0),
        }
        self.lateral_cameras: dict[int, dict[str, Any]] = {
            1: self._default_lateral_cam(-0.25),
            3: self._default_lateral_cam(0.25),
        }
        self.reload()

    @staticmethod
    def _default_lateral_cam(default_lat: float) -> dict[str, Any]:
        return {
            "calibrated": False,
            "solver_mode": "prior",
            "prior_mode": "camera_b_shift",
            "camera_shift_from_b_m": {
                "d_long_m": 0.0,
                "d_lat_m": float(default_lat),
                "d_down_m": 0.0,
            },
            "rmse_px": None,
            "updated_at": None,
            "extrinsic_from_radar_b": None,
            "extrinsic_from_local_radar": None,
        }

    def reload(self) -> None:
        """Reload radar and lateral camera extrinsics from multi_sensor_extrinsics.json if present."""
        if not self.extrinsics_path.is_file():
            return
        data = json.loads(self.extrinsics_path.read_text(encoding="utf-8"))
        radars_raw = data.get("radars") or {}
        for pod, ch in POD_TO_CHANNEL.items():
            entry = radars_raw.get(pod)
            if not isinstance(entry, dict):
                continue
            if ch == 2:
                self.radars[2] = RadarExtrinsic(2, 0.0, 0.0, 0.0, 0.0)
                continue
            self.radars[ch] = RadarExtrinsic(
                channel=ch,
                offset_x_m=float(entry.get("offset_x_m", 0.0)),
                offset_y_m=float(entry.get("offset_y_m", 0.0)),
                offset_z_m=float(entry.get("offset_z_m", 0.0)),
                yaw_deg=float(entry.get("yaw_deg", 0.0)),
            )
        cams_raw = data.get("lateral_cameras") or {}
        for pod, ch in (("A", 1), ("C", 3)):
            entry = cams_raw.get(pod)
            if isinstance(entry, dict):
                merged = self._default_lateral_cam(-0.25 if ch == 1 else 0.25)
                merged.update(entry)
                self.lateral_cameras[ch] = merged

    def save(self) -> None:
        """Persist current radar poses and lateral camera matrices to multi_sensor_extrinsics.json."""
        payload = {
            "schema_version": 1,
            "reference_pod": "B",
            "angle_convention": "yaw_deg: negative is left (-X), positive is right (+X)",
            "radars": {
                CHANNEL_TO_POD[ch]: {
                    "offset_x_m": float(pose.offset_x_m),
                    "offset_y_m": float(pose.offset_y_m),
                    "offset_z_m": float(pose.offset_z_m),
                    "yaw_deg": float(pose.yaw_deg),
                    "rotation_matrix_2x2": pose.rotation_2d().tolist(),
                }
                for ch, pose in sorted(self.radars.items())
            },
            "lateral_cameras": {
                CHANNEL_TO_POD[ch]: self.lateral_cameras[ch]
                for ch in (1, 3)
            },
        }
        self.extrinsics_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    def load_camera_b(self) -> RadarCameraTransformer:
        """Load shared intrinsics, distortion, and Group B extrinsic from camera_matrixes.json."""
        return RadarCameraTransformer.from_json(self.cam_b_path)

    def get_radar_pose(self, channel: int) -> RadarExtrinsic:
        return self.radars.get(int(channel), RadarExtrinsic(int(channel)))

    def set_radar_pose(
        self,
        channel: int,
        offset_x_m: float,
        offset_y_m: float,
        yaw_deg: float,
        offset_z_m: float = 0.0,
    ) -> RadarExtrinsic:
        """Update lateral radar pose and refresh dependent local-pod camera matrix if calibrated."""
        ch = int(channel)
        if ch == 2:
            self.radars[2] = RadarExtrinsic(2, 0.0, 0.0, 0.0, 0.0)
            return self.radars[2]
        pose = RadarExtrinsic(
            channel=ch,
            offset_x_m=float(offset_x_m),
            offset_y_m=float(offset_y_m),
            offset_z_m=float(offset_z_m),
            yaw_deg=float(yaw_deg),
        )
        self.radars[ch] = pose
        cam_entry = self.lateral_cameras.get(ch)
        if cam_entry and cam_entry.get("calibrated") and cam_entry.get("extrinsic_from_radar_b") is not None:
            t_cam_b = np.asarray(cam_entry["extrinsic_from_radar_b"], dtype=np.float64)
            t_cam_local = t_cam_b @ pose.matrix_radar3d_to_ref_b()
            cam_entry["extrinsic_from_local_radar"] = t_cam_local.tolist()
        return pose

    def set_lateral_camera_calibration(
        self,
        channel: int,
        extrinsic_from_radar_b: np.ndarray,
        *,
        solver_mode: str,
        prior_mode: str = "camera_b_shift",
        camera_shift_from_b_m: Sequence[float] = (0.0, 0.0, 0.0),
        rmse_px: float | None = None,
    ) -> None:
        """Store solved lateral camera extrinsics (both global B and standalone local pod)."""
        ch = int(channel)
        if ch not in (1, 3):
            raise ValueError(f"Lateral camera calibration only applies to A (1) or C (3), got {ch}")
        t_cam_b = np.asarray(extrinsic_from_radar_b, dtype=np.float64).reshape(4, 4)
        t_cam_local = t_cam_b @ self.get_radar_pose(ch).matrix_radar3d_to_ref_b()
        shift = np.asarray(camera_shift_from_b_m, dtype=np.float64).reshape(3)
        self.lateral_cameras[ch] = {
            "calibrated": True,
            "solver_mode": str(solver_mode),
            "prior_mode": str(prior_mode),
            "camera_shift_from_b_m": {
                "d_long_m": float(shift[0]),
                "d_lat_m": float(shift[1]),
                "d_down_m": float(shift[2]),
            },
            "rmse_px": None if rmse_px is None else round(float(rmse_px), 4),
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "extrinsic_from_radar_b": t_cam_b.tolist(),
            "extrinsic_from_local_radar": t_cam_local.tolist(),
        }

    def get_camera_transformer(
        self,
        channel: int,
        *,
        radar_source: str = "local",
    ) -> tuple[RadarCameraTransformer, dict[str, Any]]:
        """Return (RadarCameraTransformer, info) for channel 1/2/3 and 'local' or 'global' radar coords."""
        ch = int(channel)
        cam_b = self.load_camera_b()
        if ch == 2:
            return cam_b, {"calibrated": True, "solver_mode": "reference_b", "rmse_px": None}
        if ch not in (1, 3):
            raise ValueError(f"Unsupported camera channel: {ch}")

        entry = self.lateral_cameras[ch]
        pose = self.get_radar_pose(ch)
        shift_dict = entry.get("camera_shift_from_b_m") or {}
        shift_vec = (
            float(shift_dict.get("d_long_m", 0.0)),
            float(shift_dict.get("d_lat_m", -0.25 if ch == 1 else 0.25)),
            float(shift_dict.get("d_down_m", 0.0)),
        )
        if entry.get("calibrated") and entry.get("extrinsic_from_radar_b") is not None:
            t_from_b = np.asarray(entry["extrinsic_from_radar_b"], dtype=np.float64)
            if entry.get("extrinsic_from_local_radar") is not None:
                t_from_local = np.asarray(entry["extrinsic_from_local_radar"], dtype=np.float64)
            else:
                t_from_local = t_from_b @ pose.matrix_radar3d_to_ref_b()
        else:
            t_from_b, t_from_local = compute_prior_lateral_camera_extrinsic(
                cam_b,
                pose,
                prior_mode=str(entry.get("prior_mode", "camera_b_shift")),
                camera_shift_from_b_m=shift_vec,
            )
        matrix = t_from_b if radar_source == "global" else t_from_local
        transformer = RadarCameraTransformer(cam_b.intrinsic, cam_b.distortion, matrix)
        info = {
            "calibrated": bool(entry.get("calibrated", False)),
            "solver_mode": str(entry.get("solver_mode", "prior")),
            "rmse_px": entry.get("rmse_px"),
        }
        return transformer, info

    def camera_status_summary(self, channel: int) -> str:
        """Return compact human-readable calibration status for GUI badges."""
        if int(channel) == 2:
            return "B: Reference (camera_matrixes.json)"
        pod = CHANNEL_TO_POD[int(channel)]
        entry = self.lateral_cameras[int(channel)]
        if entry.get("calibrated"):
            rmse = entry.get("rmse_px")
            rmse_txt = f"RMSE {rmse:.2f} px" if isinstance(rmse, (int, float)) else "Calibrated"
            return f"{pod}: Calibrated ({entry.get('solver_mode', 'solved')}, {rmse_txt})"
        return f"{pod}: Prior (uncalibrated)"


def check_points_in_fov(
    unified_xy: Any,
    pose: RadarExtrinsic,
    *,
    fov_half_deg: float = DEFAULT_FOV_HALF_DEG,
    max_range_m: float = 100.0,
) -> np.ndarray:
    """Return boolean mask of unified [X_lat, Y_long] points inside sensor's angular/range cone."""
    pts = np.asarray(unified_xy, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    local = pose.inverse_transform_xy(pts)
    x_lat, y_fwd = local[:, 0], local[:, 1]
    az_deg = np.degrees(np.arctan2(x_lat, y_fwd))  # -Left, +Right from +Y boresight
    ranges = np.hypot(x_lat, y_fwd)
    return (y_fwd > 0.0) & (ranges <= float(max_range_m)) & (np.abs(az_deg) <= float(fov_half_deg))


def find_co_detections(
    ref_xy_unified: np.ndarray,
    tgt_xy_unified: np.ndarray,
    *,
    ref_vel_unified: np.ndarray | None = None,
    tgt_vel_unified: np.ndarray | None = None,
    dist_gate_m: float = 0.50,
    vel_gate_mps: float | None = 1.00,
) -> list[dict[str, Any]]:
    """Solve one-to-one spatial and Doppler gated co-detection associations via Hungarian assignment."""
    if len(ref_xy_unified) == 0 or len(tgt_xy_unified) == 0:
        return []
    dist_mat = cdist(ref_xy_unified, tgt_xy_unified)  # Pairwise Euclidean distance in meters
    impossible = 1e6
    cost = np.where(dist_mat <= float(dist_gate_m), dist_mat, impossible)
    if (
        vel_gate_mps is not None
        and ref_vel_unified is not None
        and tgt_vel_unified is not None
        and len(ref_vel_unified) == len(ref_xy_unified)
        and len(tgt_vel_unified) == len(tgt_xy_unified)
    ):
        valid_ref = np.isfinite(ref_vel_unified).all(axis=1)
        valid_tgt = np.isfinite(tgt_vel_unified).all(axis=1)
        if np.any(valid_ref) and np.any(valid_tgt):
            vel_diff = cdist(
                np.nan_to_num(ref_vel_unified, nan=0.0),
                np.nan_to_num(tgt_vel_unified, nan=0.0),
            )
            both_valid = np.outer(valid_ref, valid_tgt)
            cost = np.where(both_valid & (vel_diff > float(vel_gate_mps)), impossible, cost)

    row_ind, col_ind = linear_sum_assignment(cost)
    matches: list[dict[str, Any]] = []
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] >= impossible:
            continue
        matches.append({
            "ref_idx": int(i),
            "tgt_idx": int(j),
            "ref_xy": (float(ref_xy_unified[i, 0]), float(ref_xy_unified[i, 1])),
            "tgt_xy": (float(tgt_xy_unified[j, 0]), float(tgt_xy_unified[j, 1])),
            "distance_m": float(dist_mat[i, j]),
        })
    return matches


def estimate_radar_se2_kabsch(
    ref_xy: Any,
    local_target_xy: Any,
    *,
    channel: int = 3,
    offset_z_m: float = 0.0,
) -> tuple[RadarExtrinsic, float, np.ndarray]:
    """Solve optimal 2D rigid alignment (offset_x, offset_y, yaw_deg) from >=2 common radar pairs."""
    p_ref = np.asarray(ref_xy, dtype=np.float64).reshape(-1, 2)
    p_src = np.asarray(local_target_xy, dtype=np.float64).reshape(-1, 2)
    if len(p_ref) != len(p_src) or len(p_ref) < 2:
        raise ValueError("At least 2 matched common radar target pairs are required")
    if not (np.isfinite(p_ref).all() and np.isfinite(p_src).all()):
        raise ValueError("Radar point coordinates must be finite")

    mean_ref = p_ref.mean(axis=0)
    mean_src = p_src.mean(axis=0)
    centered_ref = p_ref - mean_ref
    centered_src = p_src - mean_src
    if np.linalg.norm(centered_src) < 1e-4 or np.linalg.norm(centered_ref) < 1e-4:
        raise ValueError("Matched radar points are coincident; spread targets apart to observe angle")

    # Solve min || (R @ p_src + t) - p_ref ||^2 via 2x2 SVD
    cov = centered_src.T @ centered_ref  # 2x2 cross-covariance
    u_mat, _, vt_mat = np.linalg.svd(cov)
    rot = vt_mat.T @ u_mat.T
    if np.linalg.det(rot) < 0:
        vt_mat[-1, :] *= -1.0
        rot = vt_mat.T @ u_mat.T

    trans = mean_ref - (rot @ mean_src)
    # R(psi) = [[cos(psi), sin(psi)], [-sin(psi), cos(psi)]] => atan2(R[0,1], R[0,0])
    yaw_deg = math.degrees(math.atan2(float(rot[0, 1]), float(rot[0, 0])))
    pose = RadarExtrinsic(
        channel=int(channel),
        offset_x_m=float(trans[0]),
        offset_y_m=float(trans[1]),
        offset_z_m=float(offset_z_m),
        yaw_deg=yaw_deg,
    )
    transformed = pose.transform_xy(p_src)
    residuals = np.linalg.norm(transformed - p_ref, axis=1)
    rmse_m = float(np.sqrt(np.mean(residuals ** 2)))
    return pose, rmse_m, residuals


@dataclass
class _ChannelEntry:
    local_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    local_vel: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    colors: tuple[tuple[int, int, int], ...] = ()
    points: tuple[Any, ...] = ()
    frame_type: str = "cluster"
    updated_monotonic: float | None = None
    recorded_at: datetime | None = None


@dataclass(frozen=True)
class GlobalRadarSnapshot:
    """Fused multi-radar state at an instant, resilient to missing/offline channels."""
    unified_xy: np.ndarray
    local_xy: np.ndarray
    colors: tuple[tuple[int, int, int], ...]
    channels: tuple[int, ...]
    points: tuple[Any, ...]
    channel_status: dict[int, dict[str, Any]]
    co_detections: dict[str, list[dict[str, Any]]]
    poses: dict[int, RadarExtrinsic]


class ChannelFrameCache:
    """Stores per-channel radar frames with automatic staleness expiry for fault-tolerant fusion."""

    def __init__(self, stale_seconds: float = DEFAULT_STALE_SECONDS) -> None:
        self.stale_seconds = float(stale_seconds)
        self._entries: dict[int, _ChannelEntry] = {ch: _ChannelEntry() for ch in RADAR_CHANNELS}
        self._co_detection_accumulator: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {1: [], 3: []}

    def clear(self) -> None:
        """Reset all channels to offline/empty state (e.g., when radar socket disconnects)."""
        self._entries = {ch: _ChannelEntry() for ch in RADAR_CHANNELS}

    def clear_accumulated_pairs(self, channel: int | None = None) -> None:
        if channel is None:
            self._co_detection_accumulator = {1: [], 3: []}
        elif int(channel) in self._co_detection_accumulator:
            self._co_detection_accumulator[int(channel)] = []

    def update_channel(
        self,
        channel: int,
        filtered_points: Sequence[Any],
        colors: Sequence[tuple[int, int, int]],
        *,
        frame_type: str = "cluster",
        recorded_at: datetime | None = None,
        now_monotonic: float | None = None,
    ) -> None:
        """Record a completed frame (including 0-target empty frames) as live for `channel`."""
        ch = int(channel)
        if ch not in self._entries:
            return
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        xy_list: list[tuple[float, float]] = []
        vel_list: list[tuple[float, float]] = []
        valid_colors: list[tuple[int, int, int]] = []
        valid_points: list[Any] = []

        for pt, col in zip(filtered_points, colors):
            lat = getattr(pt, "dist_latitude", None)
            lon = getattr(pt, "dist_long", None)
            if lat is None or lon is None or not (math.isfinite(lat) and math.isfinite(lon)):
                continue
            v_lat = getattr(pt, "velocity_latitude", None)
            v_lon = getattr(pt, "velocity_longitude", None)
            v_lat_f = float(v_lat) if v_lat is not None and math.isfinite(v_lat) else float("nan")
            v_lon_f = float(v_lon) if v_lon is not None and math.isfinite(v_lon) else float("nan")
            xy_list.append((float(lat), float(lon)))
            vel_list.append((v_lat_f, v_lon_f))
            valid_colors.append((int(col[0]), int(col[1]), int(col[2])))
            valid_points.append(pt)

        self._entries[ch] = _ChannelEntry(
            local_xy=np.asarray(xy_list, dtype=np.float64).reshape(-1, 2),
            local_vel=np.asarray(vel_list, dtype=np.float64).reshape(-1, 2),
            colors=tuple(valid_colors),
            points=tuple(valid_points),
            frame_type=str(frame_type),
            updated_monotonic=now,
            recorded_at=recorded_at,
        )

    def build_snapshot(
        self,
        registry: MultiSensorRegistry,
        *,
        distance_cutoff: float = 15.0,
        color_by_pod: bool = True,
        compute_overlap: bool = True,
        dist_gate_m: float = 0.50,
        vel_gate_mps: float | None = 1.00,
        accumulate_pairs: bool = False,
        now_monotonic: float | None = None,
    ) -> GlobalRadarSnapshot:
        """Fuse all currently non-stale radars into Middle B's frame without blocking on offline pods."""
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        cutoff = float(distance_cutoff)

        per_ch_unified: dict[int, np.ndarray] = {}
        per_ch_local: dict[int, np.ndarray] = {}
        per_ch_vel: dict[int, np.ndarray] = {}
        status: dict[int, dict[str, Any]] = {}
        poses: dict[int, RadarExtrinsic] = {}

        merged_unified: list[np.ndarray] = []
        merged_local: list[np.ndarray] = []
        merged_colors: list[tuple[int, int, int]] = []
        merged_channels: list[int] = []
        merged_points: list[Any] = []

        for ch in RADAR_CHANNELS:
            entry = self._entries[ch]
            pose = registry.get_radar_pose(ch)
            poses[ch] = pose
            online = (
                entry.updated_monotonic is not None
                and (now - entry.updated_monotonic) <= self.stale_seconds
            )
            if not online:
                status[ch] = {"online": False, "count": 0, "pod": CHANNEL_TO_POD[ch]}
                continue

            if len(entry.local_xy) > 0:
                ranges = np.hypot(entry.local_xy[:, 0], entry.local_xy[:, 1])
                keep = ranges <= cutoff
                loc_xy = entry.local_xy[keep]
                loc_vel = entry.local_vel[keep]
                pts_kept = [pt for pt, k in zip(entry.points, keep) if k]
                cols_kept = [
                    POD_COLORS_BGR[ch] if color_by_pod else col
                    for col, k in zip(entry.colors, keep) if k
                ]
            else:
                loc_xy = np.zeros((0, 2), dtype=np.float64)
                loc_vel = np.zeros((0, 2), dtype=np.float64)
                pts_kept = []
                cols_kept = []

            uni_xy = pose.transform_xy(loc_xy)
            uni_vel = pose.transform_velocity(loc_vel)
            per_ch_local[ch] = loc_xy
            per_ch_unified[ch] = uni_xy
            per_ch_vel[ch] = uni_vel
            status[ch] = {"online": True, "count": len(uni_xy), "pod": CHANNEL_TO_POD[ch]}

            if len(uni_xy) > 0:
                merged_unified.append(uni_xy)
                merged_local.append(loc_xy)
                merged_colors.extend(cols_kept)
                merged_channels.extend([ch] * len(uni_xy))
                merged_points.extend(pts_kept)

        co_detections: dict[str, list[dict[str, Any]]] = {"AB": [], "CB": []}
        if compute_overlap and 2 in per_ch_unified:
            for lat_ch, key in ((1, "AB"), (3, "CB")):
                if lat_ch not in per_ch_unified:
                    continue
                matches = find_co_detections(
                    per_ch_unified[2],
                    per_ch_unified[lat_ch],
                    ref_vel_unified=per_ch_vel[2],
                    tgt_vel_unified=per_ch_vel[lat_ch],
                    dist_gate_m=dist_gate_m,
                    vel_gate_mps=vel_gate_mps,
                )
                co_detections[key] = matches
                if accumulate_pairs and matches:
                    bucket = self._co_detection_accumulator[lat_ch]
                    for m in matches:
                        bucket.append((
                            per_ch_unified[2][m["ref_idx"]].copy(),
                            per_ch_local[lat_ch][m["tgt_idx"]].copy(),
                        ))
                    if len(bucket) > 300:
                        del bucket[:-300]

        all_uni = np.vstack(merged_unified) if merged_unified else np.zeros((0, 2), dtype=np.float64)
        all_loc = np.vstack(merged_local) if merged_local else np.zeros((0, 2), dtype=np.float64)
        return GlobalRadarSnapshot(
            unified_xy=all_uni,
            local_xy=all_loc,
            colors=tuple(merged_colors),
            channels=tuple(merged_channels),
            points=tuple(merged_points),
            channel_status=status,
            co_detections=co_detections,
            poses=poses,
        )

    def solve_accumulated_se2(self, channel: int) -> tuple[RadarExtrinsic, float, int]:
        """Fit optimal (offset_x, offset_y, yaw_deg) from accumulated co-detection pairs for `channel`."""
        ch = int(channel)
        pairs = self._co_detection_accumulator.get(ch, [])
        if len(pairs) < 2:
            raise ValueError(
                f"Need at least 2 co-detected pairs with Pod B to auto-fit Pod {CHANNEL_TO_POD.get(ch, ch)} "
                f"(currently have {len(pairs)})"
            )
        ref_pts = np.asarray([p[0] for p in pairs], dtype=np.float64)
        src_pts = np.asarray([p[1] for p in pairs], dtype=np.float64)
        pose, rmse_m, _ = estimate_radar_se2_kabsch(ref_pts, src_pts, channel=ch)
        return pose, rmse_m, len(pairs)
