from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares

from processing.visualization.multi_radar_fusion import RadarExtrinsic
from processing.visualization.radar_camera_transformer import RadarCameraTransformer

CORRESPONDENCE_MODES = ("radar_anchored", "ground_plane", "radar_to_cam", "epipolar")
SOLVER_MODES = ("rotation_3dof", "full_6dof")
PRIOR_MODES = ("camera_b_shift", "pod_bracket")


@dataclass
class CommonPointCorrespondence:
    """Single calibration observation linking Middle B / Radar coordinates to Lateral Camera S."""
    corr_id: int
    mode: str                                     # One of CORRESPONDENCE_MODES
    pixel_s: tuple[float, float]                  # Native-resolution (u_s, v_s) in Lateral Camera S
    pixel_b: tuple[float, float] | None = None    # Native-resolution (u_b, v_b) in Middle Camera B
    radar_b_long_lat: tuple[float, float] | None = None  # Unified B (dist_long, dist_latitude) in m
    plane_z_down_m: float = 0.0                   # Elevation Z in Radar B frame (+Down, 0=radar plane)
    enabled: bool = True
    label: str = ""


@dataclass(frozen=True)
class ExtrinsicCalibrationResult:
    """Solved 4x4 lateral camera extrinsics, reprojection residuals, and physical diagnostics."""
    extrinsic_from_radar_b: np.ndarray
    extrinsic_from_local_radar: np.ndarray
    solver_mode: str
    prior_mode: str
    rmse_px: float
    per_point_errors_px: dict[int, float]
    camera_center_in_b_m: tuple[float, float, float]
    shift_from_cam_b_m: tuple[float, float, float]
    boresight_yaw_deg: float
    boresight_pitch_deg: float
    boresight_roll_deg: float
    condition_number: float
    translation_std_m: float | None
    warnings: tuple[str, ...]


def scale_pixel_to_native(
    u_view: float,
    v_view: float,
    view_size: tuple[int, int],
    native_size: tuple[int, int],
) -> tuple[float, float]:
    """Convert a click from resized display coordinates back to native sensor resolution."""
    vw, vh = int(view_size[0]), int(view_size[1])
    nw, nh = int(native_size[0]), int(native_size[1])
    if vw <= 0 or vh <= 0 or nw <= 0 or nh <= 0:
        raise ValueError("View and native dimensions must be positive")
    return float(u_view) * (nw / float(vw)), float(v_view) * (nh / float(vh))


def undistort_normalized_rays(
    pixels: np.ndarray,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized homogeneous coords (N, 3) [x_n, y_n, 1] and unit 3D rays (N, 3)."""
    pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    if len(pts) == 0:
        empty = np.zeros((0, 3), dtype=np.float64)
        return empty, empty
    undist = cv2.undistortPoints(pts, intrinsic, distortion).reshape(-1, 2)
    homog = np.column_stack([undist, np.ones(len(undist), dtype=np.float64)])
    norms = np.linalg.norm(homog, axis=1, keepdims=True)
    return homog, homog / np.maximum(norms, 1e-12)


def camera_center_and_ray_in_radar_b(
    u_b: float,
    v_b: float,
    cam_b: RadarCameraTransformer,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Camera B optical center C_B and world unit ray d_B in Radar B [long, lat, down]."""
    r_b = cam_b.rotation_matrix
    t_b = cam_b.translation_vector.reshape(3)
    c_b = -(r_b.T @ t_b)
    _, unit_cam = undistort_normalized_rays(
        np.array([[u_b, v_b]], dtype=np.float64),
        cam_b.intrinsic,
        cam_b.distortion,
    )
    d_b = r_b.T @ unit_cam[0]
    return c_b, d_b / max(float(np.linalg.norm(d_b)), 1e-12)


def reconstruct_radar_anchored_3d(
    u_b: float,
    v_b: float,
    rad_long_b: float,
    rad_lat_b: float,
    cam_b: RadarCameraTransformer,
) -> tuple[np.ndarray, float]:
    """Intersect Camera B pixel ray with vertical target line at (rad_long_b, rad_lat_b)."""
    c_b, d_b = camera_center_and_ray_in_radar_b(u_b, v_b, cam_b)
    horiz_norm_sq = float(d_b[0] ** 2 + d_b[1] ** 2)
    if horiz_norm_sq < 1e-6:
        raise ValueError("Camera B ray is nearly vertical and cannot intersect horizontal radar range")
    delta_long = float(rad_long_b) - float(c_b[0])
    delta_lat = float(rad_lat_b) - float(c_b[1])
    ray_depth = (delta_long * float(d_b[0]) + delta_lat * float(d_b[1])) / horiz_norm_sq
    if ray_depth <= 0.2:
        raise ValueError("Radar target lies behind Camera B along the selected ray")
    point_3d = c_b + ray_depth * d_b
    horiz_err = math.hypot(float(point_3d[0]) - float(rad_long_b), float(point_3d[1]) - float(rad_lat_b))
    return point_3d, horiz_err


def reconstruct_ground_plane_3d(
    u_b: float,
    v_b: float,
    cam_b: RadarCameraTransformer,
    plane_z_down_m: float = 0.0,
) -> np.ndarray:
    """Intersect Camera B pixel ray with horizontal plane Z_down = plane_z_down_m."""
    c_b, d_b = camera_center_and_ray_in_radar_b(u_b, v_b, cam_b)
    if abs(float(d_b[2])) < 1e-5:
        raise ValueError("Selected Camera B point is on the horizon and does not intersect the ground plane")
    ray_depth = (float(plane_z_down_m) - float(c_b[2])) / float(d_b[2])
    if ray_depth <= 0.2:
        raise ValueError("Selected Camera B point is above the horizon relative to the target plane")
    return c_b + ray_depth * d_b


def resolve_correspondence_3d(
    corr: CommonPointCorrespondence,
    cam_b: RadarCameraTransformer,
) -> np.ndarray | None:
    """Resolve metric 3D point [long, lat, down] in Radar B frame for metric correspondence modes."""
    if corr.mode == "radar_anchored":
        if corr.pixel_b is None or corr.radar_b_long_lat is None:
            raise ValueError(f"Correspondence #{corr.corr_id} requires both pixel_b and radar_b_long_lat")
        pt_3d, _ = reconstruct_radar_anchored_3d(
            corr.pixel_b[0],
            corr.pixel_b[1],
            corr.radar_b_long_lat[0],
            corr.radar_b_long_lat[1],
            cam_b,
        )
        return pt_3d
    if corr.mode == "ground_plane":
        if corr.pixel_b is None:
            raise ValueError(f"Correspondence #{corr.corr_id} requires pixel_b for ground_plane mode")
        return reconstruct_ground_plane_3d(
            corr.pixel_b[0],
            corr.pixel_b[1],
            cam_b,
            plane_z_down_m=corr.plane_z_down_m,
        )
    if corr.mode == "radar_to_cam":
        if corr.radar_b_long_lat is None:
            raise ValueError(f"Correspondence #{corr.corr_id} requires radar_b_long_lat for radar_to_cam mode")
        return np.array([
            float(corr.radar_b_long_lat[0]),
            float(corr.radar_b_long_lat[1]),
            float(corr.plane_z_down_m),
        ], dtype=np.float64)
    if corr.mode == "epipolar":
        if corr.pixel_b is None:
            raise ValueError(f"Correspondence #{corr.corr_id} requires pixel_b for epipolar mode")
        return None
    raise ValueError(f"Unsupported correspondence mode: {corr.mode!r}")


def auto_match_sift_overlap(
    frame_b_bgr: np.ndarray,
    frame_s_bgr: np.ndarray,
    cam_b: RadarCameraTransformer,
    *,
    max_matches: int = 50,
    start_id: int = 1,
) -> list[CommonPointCorrespondence]:
    """Detect and RANSAC-filter common visual keypoints in the overlap between Camera B and Camera S."""
    gray_b = cv2.cvtColor(frame_b_bgr, cv2.COLOR_BGR2GRAY) if frame_b_bgr.ndim == 3 else frame_b_bgr
    gray_s = cv2.cvtColor(frame_s_bgr, cv2.COLOR_BGR2GRAY) if frame_s_bgr.ndim == 3 else frame_s_bgr
    sift = cv2.SIFT_create(nfeatures=2500)
    kp_b, desc_b = sift.detectAndCompute(gray_b, None)
    kp_s, desc_s = sift.detectAndCompute(gray_s, None)
    if desc_b is None or desc_s is None or len(kp_b) < 8 or len(kp_s) < 8:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_knn = matcher.knnMatch(desc_b, desc_s, k=2)
    good = [m for pair in raw_knn if len(pair) == 2 for m, n in (pair,) if m.distance < 0.72 * n.distance]
    if len(good) < 8:
        return []

    pts_b = np.array([kp_b[m.queryIdx].pt for m in good], dtype=np.float64)
    pts_s = np.array([kp_s[m.trainIdx].pt for m in good], dtype=np.float64)
    homog_b, _ = undistort_normalized_rays(pts_b, cam_b.intrinsic, cam_b.distortion)
    homog_s, _ = undistort_normalized_rays(pts_s, cam_b.intrinsic, cam_b.distortion)

    _, mask = cv2.findEssentialMat(
        homog_b[:, :2],
        homog_s[:, :2],
        np.eye(3, dtype=np.float64),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=0.0025,
    )
    if mask is None:
        return []

    inlier_idx = [i for i, keep in enumerate(mask.ravel()) if keep]
    if len(inlier_idx) < 5:
        return []
    # Sort by match quality and enforce spatial grid diversity so one texture patch doesn't dominate
    inlier_idx.sort(key=lambda idx: good[idx].distance)
    occupied_cells: set[tuple[int, int]] = set()
    selected: list[int] = []
    cell_px = 45.0
    for idx in inlier_idx:
        cell = (int(pts_s[idx, 0] // cell_px), int(pts_s[idx, 1] // cell_px))
        if cell in occupied_cells:
            continue
        occupied_cells.add(cell)
        selected.append(idx)
        if len(selected) >= int(max_matches):
            break

    return [
        CommonPointCorrespondence(
            corr_id=start_id + k,
            mode="epipolar",
            pixel_b=(float(pts_b[idx, 0]), float(pts_b[idx, 1])),
            pixel_s=(float(pts_s[idx, 0]), float(pts_s[idx, 1])),
            label=f"SIFT #{k + 1}",
        )
        for k, idx in enumerate(selected)
    ]


def _skew(vec: np.ndarray) -> np.ndarray:
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _epipolar_residuals_px(
    r_s: np.ndarray,
    t_s: np.ndarray,
    r_b: np.ndarray,
    t_b: np.ndarray,
    homog_b: np.ndarray,
    homog_s: np.ndarray,
    focal_px: float,
) -> np.ndarray:
    """Compute baseline-safe epipolar/rotational transfer residuals in pixel units."""
    if len(homog_b) == 0:
        return np.zeros(0, dtype=np.float64)
    r_rel = r_s @ r_b.T
    t_rel = t_s - (r_rel @ t_b)
    baseline = float(np.linalg.norm(t_rel))
    rotated_b = (r_rel @ homog_b.T).T  # (N, 3)

    if baseline <= 0.02:
        # Pure rotational transfer when cameras share nearly identical optical center
        z_vals = np.where(rotated_b[:, 2:3] > 0.05, rotated_b[:, 2:3], 0.05)
        proj = rotated_b[:, :2] / z_vals
        cheirality_pen = np.where(rotated_b[:, 2] <= 0.0, 500.0, 0.0)
        diff = np.linalg.norm(proj - homog_s[:, :2], axis=1) * focal_px + cheirality_pen
        return diff

    t_hat = t_rel / baseline
    e_mat = _skew(t_hat) @ r_rel
    ex_b = (e_mat @ homog_b.T).T
    etx_s = (e_mat.T @ homog_s.T).T
    num = np.sum(homog_s * ex_b, axis=1)
    denom = np.sqrt(ex_b[:, 0] ** 2 + ex_b[:, 1] ** 2 + etx_s[:, 0] ** 2 + etx_s[:, 1] ** 2 + 1e-12)
    # Guard against 180-degree twisted-pair rotation by checking forward ray alignment
    cheirality_pen = np.where(rotated_b[:, 2] <= -0.2, 500.0, 0.0)
    return (num / denom) * focal_px + cheirality_pen


def _solve_wahba_rotation(world_unit_rays: np.ndarray, cam_unit_rays: np.ndarray) -> np.ndarray:
    """Solve R_S in min || R_S @ u_i - v_i ||^2 via 3x3 SVD."""
    cov = cam_unit_rays.T @ world_unit_rays  # 3x3 matrix
    u_mat, _, vt_mat = np.linalg.svd(cov)
    det_sign = 1.0 if np.linalg.det(u_mat @ vt_mat) >= 0.0 else -1.0
    return u_mat @ np.diag([1.0, 1.0, det_sign]) @ vt_mat


def _extract_boresight_angles_deg(r_s: np.ndarray) -> tuple[float, float, float]:
    """Extract (yaw_right_deg, pitch_up_deg, roll_deg) of Camera S in Radar B [Forward, Right, Down]."""
    fwd = r_s[2, :]   # +Z_cam optical axis expressed in [Forward, Right, Down]
    down = r_s[1, :]  # +Y_cam image-down axis expressed in [Forward, Right, Down]
    yaw_deg = math.degrees(math.atan2(float(fwd[1]), float(fwd[0])))
    pitch_deg = math.degrees(-math.asin(max(-1.0, min(1.0, float(fwd[2])))))
    # Build zero-roll camera down vector for comparison
    yaw_rad = math.radians(yaw_deg)
    right_horiz = np.array([-math.sin(yaw_rad), math.cos(yaw_rad), 0.0], dtype=np.float64)
    down_zero_roll = np.cross(fwd, right_horiz)
    norm_dz = float(np.linalg.norm(down_zero_roll))
    if norm_dz < 1e-6:
        roll_deg = 0.0
    else:
        down_zero_roll /= norm_dz
        sin_r = float(np.dot(np.cross(down_zero_roll, down), fwd))
        cos_r = float(np.dot(down_zero_roll, down))
        roll_deg = math.degrees(math.atan2(sin_r, cos_r))
    return yaw_deg, pitch_deg, roll_deg


def _compute_prior_center_in_b(
    cam_b: RadarCameraTransformer,
    radar_pose_s: RadarExtrinsic,
    prior_mode: str,
    camera_shift_from_b_m: Sequence[float],
) -> np.ndarray:
    c_b = -(cam_b.rotation_matrix.T @ cam_b.translation_vector.reshape(3))
    if prior_mode == "pod_bracket":
        homog = np.append(c_b, 1.0)
        return (radar_pose_s.matrix_radar3d_to_ref_b() @ homog)[:3]
    shift = np.asarray(camera_shift_from_b_m, dtype=np.float64).reshape(3)
    return c_b + shift


def solve_lateral_camera_extrinsic(
    correspondences: Sequence[CommonPointCorrespondence],
    cam_b: RadarCameraTransformer,
    radar_pose_s: RadarExtrinsic,
    *,
    solver_mode: str = "rotation_3dof",
    prior_mode: str = "camera_b_shift",
    camera_shift_from_b_m: Sequence[float] = (0.0, 0.0, 0.0),
    prior_regularization: float = 0.0,
) -> ExtrinsicCalibrationResult:
    """Estimate Lateral Camera S 4x4 extrinsic matrix using shared Middle Camera B intrinsics."""
    if solver_mode not in SOLVER_MODES:
        raise ValueError(f"Unsupported solver_mode: {solver_mode!r}")
    if prior_mode not in PRIOR_MODES:
        raise ValueError(f"Unsupported prior_mode: {prior_mode!r}")

    active = [c for c in correspondences if c.enabled]
    if len(active) < 2:
        raise ValueError("At least 2 enabled common-point correspondences are required")

    k_mat = cam_b.intrinsic
    d_vec = cam_b.distortion
    r_b = cam_b.rotation_matrix
    t_b = cam_b.translation_vector.reshape(3)
    c_b = -(r_b.T @ t_b)
    focal_px = 0.5 * float(k_mat[0, 0] + k_mat[1, 1])
    c_prior = _compute_prior_center_in_b(cam_b, radar_pose_s, prior_mode, camera_shift_from_b_m)

    metric_ids: list[int] = []
    metric_pts_3d: list[np.ndarray] = []
    metric_px_s: list[tuple[float, float]] = []
    epipolar_ids: list[int] = []
    epipolar_px_b: list[tuple[float, float]] = []
    epipolar_px_s: list[tuple[float, float]] = []

    for corr in active:
        pt_3d = resolve_correspondence_3d(corr, cam_b)
        if pt_3d is not None:
            metric_ids.append(corr.corr_id)
            metric_pts_3d.append(pt_3d)
            metric_px_s.append(corr.pixel_s)
        else:
            assert corr.pixel_b is not None
            epipolar_ids.append(corr.corr_id)
            epipolar_px_b.append(corr.pixel_b)
            epipolar_px_s.append(corr.pixel_s)

    pts_3d_arr = np.asarray(metric_pts_3d, dtype=np.float64).reshape(-1, 3)
    px_s_metric_arr = np.asarray(metric_px_s, dtype=np.float64).reshape(-1, 2)
    px_b_epi_arr = np.asarray(epipolar_px_b, dtype=np.float64).reshape(-1, 2)
    px_s_epi_arr = np.asarray(epipolar_px_s, dtype=np.float64).reshape(-1, 2)
    homog_b_epi, rays_b_epi = undistort_normalized_rays(px_b_epi_arr, k_mat, d_vec)
    homog_s_epi, rays_s_epi = undistort_normalized_rays(px_s_epi_arr, k_mat, d_vec)
    _, rays_s_metric = undistort_normalized_rays(px_s_metric_arr, k_mat, d_vec)

    if solver_mode == "full_6dof" and len(pts_3d_arr) < 4:
        raise ValueError(
            f"Full 6-DoF mode requires at least 4 metric points (Sources 1-3); got {len(pts_3d_arr)}. "
            "Add metric/ground points or use 'rotation_3dof' mode."
        )

    # Step 1: Closed-form 3-DoF Wahba rotation around c_prior (always available as clean baseline)
    world_rays_list: list[np.ndarray] = []
    cam_rays_list: list[np.ndarray] = []
    if len(pts_3d_arr) > 0:
        vecs = pts_3d_arr - c_prior
        norms = np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
        world_rays_list.append(vecs / norms)
        cam_rays_list.append(rays_s_metric)
    if len(rays_b_epi) > 0:
        world_dirs_b = (r_b.T @ rays_b_epi.T).T
        approx_pts = c_b + 15.0 * world_dirs_b  # Reference depth for initial ray direction
        vecs_epi = approx_pts - c_prior
        norms_epi = np.maximum(np.linalg.norm(vecs_epi, axis=1, keepdims=True), 1e-9)
        world_rays_list.append(vecs_epi / norms_epi)
        cam_rays_list.append(rays_s_epi)

    r_init_3dof = _solve_wahba_rotation(np.vstack(world_rays_list), np.vstack(cam_rays_list))
    rvec_3dof, _ = cv2.Rodrigues(r_init_3dof)
    rvec_3dof = rvec_3dof.reshape(3)

    def _project_residuals_px(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        if len(pts_3d_arr) == 0:
            return np.zeros(0, dtype=np.float64)
        r_mat, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        cam_pts = (r_mat @ pts_3d_arr.T).T + tvec.reshape(1, 3)
        proj, _ = cv2.projectPoints(
            pts_3d_arr.reshape(-1, 1, 3),
            rvec.reshape(3, 1),
            tvec.reshape(3, 1),
            k_mat,
            d_vec,
        )
        diff = (proj.reshape(-1, 2) - px_s_metric_arr).reshape(-1)
        behind = cam_pts[:, 2] <= 0.05
        if np.any(behind):
            penalty = np.repeat(behind, 2)
            diff = np.where(penalty, diff + np.sign(diff + 1e-6) * 500.0, diff)
        return diff

    warnings: list[str] = []
    trans_std_m: float | None = None

    if solver_mode == "rotation_3dof":
        def _res_3dof(rvec_param: np.ndarray) -> np.ndarray:
            r_mat, _ = cv2.Rodrigues(rvec_param.reshape(3, 1))
            t_vec = -(r_mat @ c_prior)
            parts = [_project_residuals_px(rvec_param, t_vec)]
            if len(homog_b_epi) > 0:
                parts.append(_epipolar_residuals_px(r_mat, t_vec, r_b, t_b, homog_b_epi, homog_s_epi, focal_px))
            return np.concatenate(parts)

        opt = least_squares(_res_3dof, rvec_3dof, method="trf", loss="huber", f_scale=3.0)
        r_sol, _ = cv2.Rodrigues(opt.x.reshape(3, 1))
        t_sol = -(r_sol @ c_prior)
        jac = opt.jac
        svals = np.linalg.svd(jac, compute_uv=False)
        cond_num = float(svals[0] / max(float(svals[-1]), 1e-12)) if len(svals) else float("inf")
    else:
        # Multi-hypothesis initialization for full 6-DoF
        candidates: list[tuple[np.ndarray, np.ndarray]] = [(rvec_3dof, -(r_init_3dof @ c_prior))]
        z_spread = float(np.ptp(pts_3d_arr[:, 2])) if len(pts_3d_arr) > 0 else 0.0
        if z_spread < 0.15:
            warnings.append(
                f"Metric 3D points have narrow height spread ({z_spread:.2f} m); "
                "add a Radar-Anchored elevated point (e.g., person head/pole top) or check camera position sanity."
            )

        for flag in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_IPPE):
            try:
                count, rvecs_cand, tvecs_cand, _ = cv2.solvePnPGeneric(
                    pts_3d_arr.reshape(-1, 1, 3),
                    px_s_metric_arr.reshape(-1, 1, 2),
                    k_mat,
                    d_vec,
                    flags=flag,
                )
                if count:
                    for rv, tv in zip(rvecs_cand, tvecs_cand):
                        r_c, _ = cv2.Rodrigues(rv)
                        c_cand = -(r_c.T @ tv.reshape(3))
                        # Reject physically flipped poses underground when Cam B is above ground
                        if c_b[2] < -0.2 and c_cand[2] > 0.3:
                            continue
                        cam_z = ((r_c @ pts_3d_arr.T).T + tv.reshape(1, 3))[:, 2]
                        if np.all(cam_z > 0.1):
                            candidates.append((rv.reshape(3), tv.reshape(3)))
            except cv2.error:
                pass

        def _res_6dof(params: np.ndarray) -> np.ndarray:
            rv = params[:3]
            tv = params[3:6]
            r_mat, _ = cv2.Rodrigues(rv.reshape(3, 1))
            parts = [_project_residuals_px(rv, tv)]
            if len(homog_b_epi) > 0:
                parts.append(_epipolar_residuals_px(r_mat, tv, r_b, t_b, homog_b_epi, homog_s_epi, focal_px))
            if prior_regularization > 0.0:
                c_curr = -(r_mat.T @ tv)
                parts.append(math.sqrt(float(prior_regularization)) * (c_curr - c_prior))
            return np.concatenate(parts)

        best_x = np.concatenate([candidates[0][0], candidates[0][1]])
        best_cost = float(np.sum(_res_6dof(best_x) ** 2))
        for rv_c, tv_c in candidates[1:]:
            cand_x = np.concatenate([rv_c, tv_c])
            cost_c = float(np.sum(_res_6dof(cand_x) ** 2))
            if cost_c < best_cost:
                best_cost = cost_c
                best_x = cand_x

        opt = least_squares(_res_6dof, best_x, method="trf", loss="huber", f_scale=3.0)
        r_sol, _ = cv2.Rodrigues(opt.x[:3].reshape(3, 1))
        t_sol = opt.x[3:6].reshape(3)

        jac = opt.jac
        svals = np.linalg.svd(jac, compute_uv=False)
        cond_num = float(svals[0] / max(float(svals[-1]), 1e-12)) if len(svals) else float("inf")
        dof = max(len(opt.fun) - 6, 1)
        sigma2 = float(np.sum(opt.fun ** 2)) / dof
        try:
            cov_6x6 = sigma2 * np.linalg.pinv(jac.T @ jac)
            trans_std_m = float(np.sqrt(np.max(np.diag(cov_6x6)[3:6])))
            if trans_std_m > 0.25:
                warnings.append(
                    f"Estimated translation uncertainty is high (±{trans_std_m:.2f} m); "
                    "consider 3-DoF + Offset Prior mode or adding near/elevated points."
                )
        except np.linalg.LinAlgError:
            warnings.append("Jacobian is near-singular; translation may be weakly constrained.")

    # Compute final clean per-correspondence pixel errors
    rvec_sol, _ = cv2.Rodrigues(r_sol)
    per_point_err: dict[int, float] = {}
    sq_errors: list[float] = []

    if len(pts_3d_arr) > 0:
        proj_s, _ = cv2.projectPoints(
            pts_3d_arr.reshape(-1, 1, 3),
            rvec_sol,
            t_sol.reshape(3, 1),
            k_mat,
            d_vec,
        )
        errs_metric = np.linalg.norm(proj_s.reshape(-1, 2) - px_s_metric_arr, axis=1)
        for cid, err in zip(metric_ids, errs_metric):
            per_point_err[cid] = float(err)
            sq_errors.append(float(err) ** 2)

    if len(homog_b_epi) > 0:
        errs_epi = np.abs(_epipolar_residuals_px(r_sol, t_sol, r_b, t_b, homog_b_epi, homog_s_epi, focal_px))
        for cid, err in zip(epipolar_ids, errs_epi):
            per_point_err[cid] = float(err)
            sq_errors.append(float(err) ** 2)

    rmse_px = math.sqrt(sum(sq_errors) / max(len(sq_errors), 1))
    c_sol = -(r_sol.T @ t_sol)
    shift_from_b = c_sol - c_b
    if solver_mode == "full_6dof" and float(np.linalg.norm(shift_from_b)) > 2.5:
        warnings.append(
            f"Solved camera center is {np.linalg.norm(shift_from_b):.2f} m away from Camera B; "
            "verify physical plausibility before saving."
        )

    ext_from_b = np.eye(4, dtype=np.float64)
    ext_from_b[:3, :3] = r_sol
    ext_from_b[:3, 3] = t_sol
    ext_from_local = ext_from_b @ radar_pose_s.matrix_radar3d_to_ref_b()
    yaw_deg, pitch_deg, roll_deg = _extract_boresight_angles_deg(r_sol)

    return ExtrinsicCalibrationResult(
        extrinsic_from_radar_b=ext_from_b,
        extrinsic_from_local_radar=ext_from_local,
        solver_mode=solver_mode,
        prior_mode=prior_mode,
        rmse_px=float(rmse_px),
        per_point_errors_px=per_point_err,
        camera_center_in_b_m=(float(c_sol[0]), float(c_sol[1]), float(c_sol[2])),
        shift_from_cam_b_m=(float(shift_from_b[0]), float(shift_from_b[1]), float(shift_from_b[2])),
        boresight_yaw_deg=float(yaw_deg),
        boresight_pitch_deg=float(pitch_deg),
        boresight_roll_deg=float(roll_deg),
        condition_number=float(cond_num),
        translation_std_m=trans_std_m,
        warnings=tuple(warnings),
    )
