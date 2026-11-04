from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

import cv2 as cv
import numpy as np

import tests  # Ensure headless FreeSimpleGUI shims are active
from calibration_workbench import CalibrationWorkbench, discover_paired_recording_entries
import main as main_module
from menu_configurations import Configurations
from processing.visualization.extrinsic_calibrator import (
    CommonPointCorrespondence,
    auto_match_sift_overlap,
    reconstruct_ground_plane_3d,
    reconstruct_radar_anchored_3d,
    solve_lateral_camera_extrinsic,
)
from processing.visualization.graph_draw import Graph_radar
from processing.visualization.multi_radar_fusion import (
    ChannelFrameCache,
    MultiSensorRegistry,
    RadarExtrinsic,
    check_points_in_fov,
    estimate_radar_se2_kabsch,
    find_co_detections,
)
from processing.visualization.radar_camera_transformer import RadarCameraTransformer
from processing.visualization.transposition import (
    MultiGroupCameraOverlay,
    global_transposition_payload,
    transposition_payload,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CAM_B_JSON = REPO_ROOT / "camera_matrixes.json"


def _build_ground_truth_camera(
    cam_b: RadarCameraTransformer,
    shift_from_b_m: tuple[float, float, float],
    *,
    yaw_right_deg: float,
    pitch_up_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> RadarCameraTransformer:
    """Construct a synthetic lateral camera sharing Cam B intrinsics/distortion at a known 6-DoF pose."""
    r_b = cam_b.rotation_matrix
    t_b = cam_b.translation_vector.reshape(3)
    c_b = -(r_b.T @ t_b)
    c_s = c_b + np.asarray(shift_from_b_m, dtype=np.float64)

    # Yaw around +Z_down (+Right), pitch around camera +X_right, roll around camera +Z_fwd
    psi = math.radians(yaw_right_deg)
    r_yaw_world = np.array([
        [math.cos(psi), -math.sin(psi), 0.0],
        [math.sin(psi), math.cos(psi), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    r_pitch_cam, _ = cv.Rodrigues(np.array([math.radians(pitch_up_deg), 0.0, 0.0], dtype=np.float64))
    r_roll_cam, _ = cv.Rodrigues(np.array([0.0, 0.0, math.radians(roll_deg)], dtype=np.float64))

    r_s = r_roll_cam @ r_pitch_cam @ r_b @ r_yaw_world.T
    t_s = -(r_s @ c_s)
    ext = np.eye(4, dtype=np.float64)
    ext[:3, :3] = r_s
    ext[:3, 3] = t_s
    return RadarCameraTransformer(cam_b.intrinsic, cam_b.distortion, ext)


def _project_single(cam: RadarCameraTransformer, pt_3d: np.ndarray) -> tuple[float, float]:
    pts = np.asarray(pt_3d, dtype=np.float64).reshape(1, 3)
    cam_xyz = cam.radar_to_camera(pts)
    assert float(cam_xyz[0, 2]) > 0.1, f"Point {pt_3d} projected behind synthetic camera"
    uv = cam.radar_to_image(pts, distorted=True)
    return float(uv[0, 0]), float(uv[0, 1])


class MultiRadarGeometryAndFaultToleranceTests(unittest.TestCase):
    """Verify (-Left, +Right) angle conventions, SE(2) Kabsch fitting, and offline radar resilience."""

    def test_yaw_sign_convention_left_negative_right_positive(self):
        pose_left = RadarExtrinsic(1, offset_x_m=-0.30, offset_y_m=0.0, yaw_deg=-30.0)
        pose_right = RadarExtrinsic(3, offset_x_m=0.30, offset_y_m=0.0, yaw_deg=30.0)

        ahead_local = np.array([[0.0, 10.0]], dtype=np.float64)  # x_lat=0, y_long=10m
        uni_left = pose_left.transform_xy(ahead_local)[0]
        uni_right = pose_right.transform_xy(ahead_local)[0]

        self.assertAlmostEqual(uni_left[0], -0.30 - 5.0, places=6)
        self.assertAlmostEqual(uni_left[1], 10.0 * math.cos(math.radians(30.0)), places=6)
        self.assertAlmostEqual(uni_right[0], 0.30 + 5.0, places=6)
        self.assertAlmostEqual(uni_right[1], 10.0 * math.cos(math.radians(30.0)), places=6)

        # Round-trip inverse must be exact
        np.testing.assert_allclose(pose_left.inverse_transform_xy([uni_left])[0], ahead_local[0], atol=1e-9)
        np.testing.assert_allclose(pose_right.inverse_transform_xy([uni_right])[0], ahead_local[0], atol=1e-9)

    def test_2d_and_3d_radar_transforms_are_identical(self):
        pose = RadarExtrinsic(3, offset_x_m=0.42, offset_y_m=-0.11, offset_z_m=0.05, yaw_deg=27.5)
        local_lat, local_long, local_z = -1.8, 9.4, -0.6
        uni_2d = pose.transform_xy([[local_lat, local_long]])[0]
        vec_3d = pose.matrix_radar3d_to_ref_b() @ np.array([local_long, local_lat, local_z, 1.0])

        self.assertAlmostEqual(vec_3d[0], uni_2d[1], places=9)  # dist_long == Y_long
        self.assertAlmostEqual(vec_3d[1], uni_2d[0], places=9)  # dist_latitude == X_lat
        self.assertAlmostEqual(vec_3d[2], local_z + 0.05, places=9)

    def test_fault_tolerant_cache_handles_offline_radars_and_empty_active_frames(self):
        with TemporaryDirectory() as tmp:
            reg = MultiSensorRegistry(Path(tmp) / "ext.json", CAM_B_JSON)
            reg.set_radar_pose(1, -0.30, 0.0, -25.0)
            reg.set_radar_pose(3, 0.30, 0.0, 25.0)
            cache = ChannelFrameCache(stale_seconds=0.50)

            pt_a = SimpleNamespace(dist_latitude=0.0, dist_long=8.0, velocity_latitude=0.0, velocity_longitude=-1.0)
            pt_b = SimpleNamespace(dist_latitude=0.5, dist_long=9.0, velocity_latitude=0.0, velocity_longitude=0.0)
            pt_c = SimpleNamespace(dist_latitude=0.0, dist_long=7.0, velocity_latitude=0.0, velocity_longitude=0.5)

            # 1) Only Left Radar A online (B and C completely unplugged)
            cache.update_channel(1, (pt_a,), ((255, 0, 0),), now_monotonic=10.0)
            snap_1 = cache.build_snapshot(reg, now_monotonic=10.1)
            self.assertTrue(snap_1.channel_status[1]["online"])
            self.assertFalse(snap_1.channel_status[2]["online"])
            self.assertFalse(snap_1.channel_status[3]["online"])
            self.assertEqual(len(snap_1.unified_xy), 1)

            # 2) B comes online with 0 targets (empty scene), C comes online with 1 target
            cache.update_channel(2, (), (), now_monotonic=10.2)
            cache.update_channel(3, (pt_c,), ((0, 0, 255),), now_monotonic=10.2)
            snap_2 = cache.build_snapshot(reg, now_monotonic=10.3)
            self.assertTrue(snap_2.channel_status[2]["online"])
            self.assertEqual(snap_2.channel_status[2]["count"], 0)
            self.assertTrue(snap_2.channel_status[3]["online"])
            self.assertEqual(len(snap_2.unified_xy), 2)

            # 3) Middle Radar B has a target at t=10.3, then unplugs; A & C update at t=11.0
            cache.update_channel(2, (pt_b,), ((0, 255, 0),), now_monotonic=10.3)
            cache.update_channel(1, (pt_a,), ((255, 0, 0),), now_monotonic=11.0)
            cache.update_channel(3, (pt_c,), ((0, 0, 255),), now_monotonic=11.0)
            snap_3 = cache.build_snapshot(reg, now_monotonic=11.0)
            self.assertFalse(snap_3.channel_status[2]["online"])  # Stale B automatically purged
            self.assertTrue(snap_3.channel_status[1]["online"])
            self.assertTrue(snap_3.channel_status[3]["online"])
            self.assertEqual(snap_3.channels, (1, 3))

            # Global map rendering succeeds in every state without cv.imshow
            graph = Graph_radar(20.0, 640, 480, 15.0, 20.0)
            img = graph.render_global_image(snap_3, show_cones=True, show_co_detections=True)
            self.assertEqual(img.shape, (480, 640, 3))

    def test_co_detection_doppler_gating_and_kabsch_se2_autofit(self):
        true_pose_c = RadarExtrinsic(3, offset_x_m=0.44, offset_y_m=-0.09, yaw_deg=31.5)
        world_targets_b = np.array([
            [1.2, 6.0],
            [2.8, 9.5],
            [0.5, 12.0],
            [4.1, 14.2],
        ], dtype=np.float64)
        local_c = true_pose_c.inverse_transform_xy(world_targets_b)

        recovered_pose, rmse_m, _ = estimate_radar_se2_kabsch(world_targets_b, local_c, channel=3)
        self.assertAlmostEqual(recovered_pose.offset_x_m, 0.44, places=6)
        self.assertAlmostEqual(recovered_pose.offset_y_m, -0.09, places=6)
        self.assertAlmostEqual(recovered_pose.yaw_deg, 31.5, places=6)
        self.assertLess(rmse_m, 1e-9)

        # Verify FOV cone check and Doppler gating reject opposing velocities
        in_fov = check_points_in_fov(world_targets_b, true_pose_c, fov_half_deg=55.0, max_range_m=20.0)
        self.assertTrue(np.all(in_fov))

        vel_ref = np.array([[0.0, -3.0], [0.0, 2.5]], dtype=np.float64)
        vel_tgt_bad_second = np.array([[0.1, -2.9], [0.0, -4.0]], dtype=np.float64)
        matches = find_co_detections(
            world_targets_b[:2],
            world_targets_b[:2] + 0.05,
            ref_vel_unified=vel_ref,
            tgt_vel_unified=vel_tgt_bad_second,
            dist_gate_m=0.50,
            vel_gate_mps=1.0,
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["ref_idx"], 0)


class LateralCameraExtrinsicCalibrationTests(unittest.TestCase):
    """Verify lateral camera calibration from common points while keeping camera_matrixes.json untouched."""

    def setUp(self):
        self.cam_b_before = CAM_B_JSON.read_text(encoding="utf-8")
        self.cam_b = RadarCameraTransformer.from_json(CAM_B_JSON)

    def tearDown(self):
        self.assertEqual(
            CAM_B_JSON.read_text(encoding="utf-8"),
            self.cam_b_before,
            "camera_matrixes.json must remain strictly untouched for Middle Camera B!",
        )

    def test_rotation_3dof_recovers_shifted_lateral_camera_from_3_ground_points(self):
        shift_c = (0.0, 0.25, 0.0)  # +25 cm right of Camera B
        gt_cam_c = _build_ground_truth_camera(
            self.cam_b,
            shift_c,
            yaw_right_deg=28.0,
            pitch_up_deg=-2.0,
            roll_deg=1.2,
        )
        radar_pose_c = RadarExtrinsic(3, offset_x_m=0.25, offset_y_m=0.0, yaw_deg=30.0)

        # Common ground points (Z_down = 0.0) visible in both Middle Camera B and Right Camera C
        ground_pts = [
            np.array([7.0, 1.2, 0.0]),
            np.array([10.5, 2.4, 0.0]),
            np.array([14.0, 1.8, 0.0]),
        ]
        correspondences = [
            CommonPointCorrespondence(
                corr_id=idx + 1,
                mode="ground_plane",
                pixel_b=_project_single(self.cam_b, pt),
                pixel_s=_project_single(gt_cam_c, pt),
                plane_z_down_m=0.0,
            )
            for idx, pt in enumerate(ground_pts)
        ]

        res = solve_lateral_camera_extrinsic(
            correspondences,
            self.cam_b,
            radar_pose_c,
            solver_mode="rotation_3dof",
            prior_mode="camera_b_shift",
            camera_shift_from_b_m=shift_c,
        )

        self.assertLess(res.rmse_px, 0.05)
        np.testing.assert_allclose(res.extrinsic_from_radar_b, gt_cam_c.extrinsic, atol=1e-4)

        # Standalone Pod C projection via extrinsic_from_local_radar must match unified B projection
        local_target_c = np.array([8.5, 1.5, 0.0])  # [dist_long, dist_lat, z_down] in Radar C
        global_target_b = (radar_pose_c.matrix_radar3d_to_ref_b() @ np.append(local_target_c, 1.0))[:3]
        cam_c_local = RadarCameraTransformer(self.cam_b.intrinsic, self.cam_b.distortion, res.extrinsic_from_local_radar)
        cam_c_global = RadarCameraTransformer(self.cam_b.intrinsic, self.cam_b.distortion, res.extrinsic_from_radar_b)

        uv_local = cam_c_local.radar_to_image(local_target_c.reshape(1, 3), distorted=True)
        uv_global = cam_c_global.radar_to_image(global_target_b.reshape(1, 3), distorted=True)
        np.testing.assert_allclose(uv_local, uv_global, atol=1e-6)

    def test_full_6dof_recovers_diverse_mount_across_all_4_correspondence_modes(self):
        # Future-proof non-standard mounting: forward +0.35m, left -0.58m, 0.32m higher (-down)
        diverse_shift_a = (0.35, -0.58, -0.32)
        gt_cam_a = _build_ground_truth_camera(
            self.cam_b,
            diverse_shift_a,
            yaw_right_deg=-32.0,
            pitch_up_deg=-3.5,
            roll_deg=-1.5,
        )
        radar_pose_a = RadarExtrinsic(1, offset_x_m=-0.55, offset_y_m=0.30, yaw_deg=-30.0)

        corrs: list[CommonPointCorrespondence] = []
        cid = 1

        # 1) Radar-Anchored 3D points at two heights (feet Z=0 and head/pole top Z=-1.6m)
        for long_b, lat_b, z_down in (
            (8.0, -1.2, 0.0),
            (8.0, -1.2, -1.65),
            (12.5, -2.2, 0.0),
            (12.5, -2.2, -1.80),
        ):
            pt = np.array([long_b, lat_b, z_down], dtype=np.float64)
            u_b, v_b = _project_single(self.cam_b, pt)
            recon_3d, horiz_err = reconstruct_radar_anchored_3d(u_b, v_b, long_b, lat_b, self.cam_b)
            self.assertLess(horiz_err, 1e-6)
            np.testing.assert_allclose(recon_3d, pt, atol=1e-5)

            corrs.append(CommonPointCorrespondence(
                corr_id=cid,
                mode="radar_anchored",
                pixel_b=(u_b, v_b),
                pixel_s=_project_single(gt_cam_a, pt),
                radar_b_long_lat=(long_b, lat_b),
            ))
            cid += 1

        # 2) Ground-plane points in B ∩ A overlap
        for pt in (np.array([6.5, -0.8, 0.0]), np.array([15.0, -2.8, 0.0])):
            u_b, v_b = _project_single(self.cam_b, pt)
            np.testing.assert_allclose(reconstruct_ground_plane_3d(u_b, v_b, self.cam_b, 0.0), pt, atol=1e-5)
            corrs.append(CommonPointCorrespondence(
                corr_id=cid,
                mode="ground_plane",
                pixel_b=(u_b, v_b),
                pixel_s=_project_single(gt_cam_a, pt),
                plane_z_down_m=0.0,
            ))
            cid += 1

        # 3) Outer-FoV Radar -> Cam A points (far left at lat = -6.0 m, outside Cam B FoV)
        for pt in (np.array([9.0, -5.8, 0.0]), np.array([13.0, -7.5, 0.0])):
            corrs.append(CommonPointCorrespondence(
                corr_id=cid,
                mode="radar_to_cam",
                pixel_s=_project_single(gt_cam_a, pt),
                radar_b_long_lat=(float(pt[0]), float(pt[1])),
                plane_z_down_m=0.0,
            ))
            cid += 1

        # 4) Pure visual epipolar correspondences on distant elevated facade features
        for pt in (np.array([22.0, -3.5, -2.4]), np.array([25.0, -4.8, -1.2])):
            corrs.append(CommonPointCorrespondence(
                corr_id=cid,
                mode="epipolar",
                pixel_b=_project_single(self.cam_b, pt),
                pixel_s=_project_single(gt_cam_a, pt),
            ))
            cid += 1

        res = solve_lateral_camera_extrinsic(
            corrs,
            self.cam_b,
            radar_pose_a,
            solver_mode="full_6dof",
            prior_mode="camera_b_shift",
            camera_shift_from_b_m=(0.0, -0.25, 0.0),  # Deliberately rough prior; 6-DoF must find true shift!
        )

        self.assertLess(res.rmse_px, 0.05)
        np.testing.assert_allclose(res.shift_from_cam_b_m, diverse_shift_a, atol=3e-3)
        np.testing.assert_allclose(res.extrinsic_from_radar_b, gt_cam_a.extrinsic, atol=2e-3)

    def test_zero_baseline_epipolar_transfer_converges_without_nan(self):
        gt_cam_c = _build_ground_truth_camera(
            self.cam_b,
            (0.0, 0.0, 0.0),
            yaw_right_deg=24.0,
        )
        radar_pose_c = RadarExtrinsic(3, offset_x_m=0.0, offset_y_m=0.0, yaw_deg=25.0)
        pts = [
            np.array([10.0, 1.5, -0.5]),
            np.array([14.0, 2.8, -1.8]),
            np.array([18.0, 3.2, 0.0]),
        ]
        corrs = [
            CommonPointCorrespondence(
                corr_id=i + 1,
                mode="epipolar",
                pixel_b=_project_single(self.cam_b, p),
                pixel_s=_project_single(gt_cam_c, p),
            )
            for i, p in enumerate(pts)
        ]
        res = solve_lateral_camera_extrinsic(
            corrs,
            self.cam_b,
            radar_pose_c,
            solver_mode="rotation_3dof",
            prior_mode="camera_b_shift",
            camera_shift_from_b_m=(0.0, 0.0, 0.0),
        )
        self.assertTrue(math.isfinite(res.rmse_px))
        self.assertLess(res.rmse_px, 0.05)
        np.testing.assert_allclose(res.extrinsic_from_radar_b[:3, :3], gt_cam_c.rotation_matrix, atol=1e-4)

    def test_multi_group_camera_overlay_projects_local_and_global_sources(self):
        with TemporaryDirectory() as tmp:
            ext_path = Path(tmp) / "multi_sensor_extrinsics.json"
            reg = MultiSensorRegistry(ext_path, CAM_B_JSON)
            reg.set_radar_pose(3, 0.30, 0.0, 25.0)
            reg.save()

            overlay = MultiGroupCameraOverlay(ext_path, CAM_B_JSON)
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)

            # Local Pod C target straight ahead of Radar C
            pt_c = SimpleNamespace(dist_long=9.0, dist_latitude=0.0, velocity_longitude=0.0, velocity_latitude=0.0)
            local_payload = transposition_payload(
                (pt_c,),
                ((20, 200, 240),),
                frame_type="cluster",
                recorded_at=datetime.now(timezone.utc),
                distance_cutoff=20.0,
                group=3,
            )
            drawn_local = overlay.draw(3, frame, local_payload, source_size=(1920, 1080))
            self.assertTrue(np.any(drawn_local))

            # Global fused snapshot projected onto Camera C
            cache = ChannelFrameCache()
            cache.update_channel(3, (pt_c,), ((20, 200, 240),), now_monotonic=5.0)
            snap = cache.build_snapshot(reg, distance_cutoff=20.0, color_by_pod=False, now_monotonic=5.0)
            glob_payload = global_transposition_payload(
                snap,
                recorded_at=datetime.now(timezone.utc),
            )
            drawn_global = overlay.draw(3, frame, glob_payload, source_size=(1920, 1080))
            np.testing.assert_array_equal(drawn_local, drawn_global)


class MenuFrame2AndEventRoutingTests(unittest.TestCase):
    """Verify all new tabs reside on Frame 2 and main.py routes multi-group events cleanly."""

    @staticmethod
    def _walk_elements(rows):
        for row in rows:
            items = row if isinstance(row, (list, tuple)) else (row,)
            for elem in items:
                yield elem
                sub_rows = getattr(elem, "Rows", None)
                if sub_rows:
                    yield from MenuFrame2AndEventRoutingTests._walk_elements(sub_rows)

    def test_new_tabs_are_exclusively_on_frame_2_general_control(self):
        cfg = Configurations()
        frame2_tabs = [
            el.Title
            for el in self._walk_elements(cfg.radar_control.Rows)
            if type(el).__name__ == "Tab"
        ]
        frame3_tabs = [
            el.Title
            for el in self._walk_elements(cfg.filters.Rows)
            if type(el).__name__ == "Tab"
        ]
        self.assertIn("Global Map", frame2_tabs)
        self.assertIn("Calibration", frame2_tabs)
        self.assertEqual(frame3_tabs, ["Basic", "Cluster Options", "Transposition"])

    def test_switching_groups_keeps_transposition_active_and_routes_workbench(self):
        cfg = Configurations()
        runtime = main_module.RuntimeState()
        send_radar = MagicMock()
        send_cam = MagicMock()
        send_gps = MagicMock()
        send_playback = MagicMock()
        send_snapshot_playback = MagicMock()
        shutdown = MagicMock()

        values = {
            "transposition_src_local": True,
            "transposition_src_global": False,
            "global_codet_gate": "0.50",
            "global_color_pod": True,
            "global_color_dyn": False,
            "global_show_cones": True,
            "global_show_codet": True,
            "global_accumulate_pairs": False,
            "cal_pod_1": True,
            "cal_pod_3": False,
        }

        # Enable transposition while on Group B, then switch to Group A (choose_1)
        main_module._handle_gui_event(
            "transposition_toggle", values, cfg, runtime,
            send_radar, send_cam, send_gps, send_playback, send_snapshot_playback, shutdown,
        )
        self.assertTrue(cfg.transposition)

        main_module._handle_gui_event(
            "choose_1", values, cfg, runtime,
            send_radar, send_cam, send_gps, send_playback, send_snapshot_playback, shutdown,
        )
        self.assertEqual(runtime.active_channel, 1)
        self.assertTrue(cfg.transposition)  # Must stay enabled for lateral Pod A!

        # Open Calibration Workbench from Frame 2 Calibration tab
        main_module._handle_gui_event(
            "calibration_open_workbench", values, cfg, runtime,
            send_radar, send_cam, send_gps, send_playback, send_snapshot_playback, shutdown,
        )
        self.assertIsInstance(runtime.workbench, CalibrationWorkbench)
        self.assertEqual(runtime.workbench.target_channel, 1)
        runtime.workbench.close()
        self.assertTrue(runtime.workbench.closed)

    def test_discover_paired_recording_entries_matches_b_and_lateral_sessions(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for pod, img_name in (("B", "camera_b.jpg"), ("C", "camera_c.jpg")):
                folder = root / f"recording_{pod}_20261104_120000"
                (folder / "images").mkdir(parents=True)
                img = np.zeros((32, 32, 3), dtype=np.uint8)
                cv.imwrite(str(folder / "images" / img_name), img)
                (folder / "recording.json").write_text(
                    json.dumps([{
                        "point_cloud": None,
                        "camera_frame": f"images/{img_name}",
                        "recorded_at": "2026-11-04T12:00:01.000000+00:00",
                    }]),
                    encoding="utf-8",
                )
            pairs = discover_paired_recording_entries(root, target_channel=3)
            self.assertEqual(len(pairs), 1)
            self.assertTrue(str(pairs[0]["entry_b"].camera_frame).endswith("camera_b.jpg"))
            self.assertTrue(str(pairs[0]["entry_s"].camera_frame).endswith("camera_c.jpg"))

    def test_accumulated_co_detections_autofit_workflow(self):
        with TemporaryDirectory() as tmp:
            reg = MultiSensorRegistry(Path(tmp) / "ext.json", CAM_B_JSON)
            # Initial rough guess for Pod A (-25 deg vs true -27 deg)
            reg.set_radar_pose(1, -0.25, 0.0, -25.0)
            true_pose_a = RadarExtrinsic(1, offset_x_m=-0.32, offset_y_m=-0.04, yaw_deg=-27.0)
            cache = ChannelFrameCache()

            for step, (bx, by) in enumerate(((-1.5, 6.0), (-2.8, 9.0), (-0.8, 11.5))):
                ax, ay = true_pose_a.inverse_transform_xy([[bx, by]])[0]
                pt_b = SimpleNamespace(dist_latitude=bx, dist_long=by, velocity_latitude=0.0, velocity_longitude=0.0)
                pt_a = SimpleNamespace(dist_latitude=float(ax), dist_long=float(ay), velocity_latitude=0.0, velocity_longitude=0.0)
                t_now = 10.0 + step * 0.1
                cache.update_channel(2, (pt_b,), ((0, 255, 0),), now_monotonic=t_now)
                cache.update_channel(1, (pt_a,), ((255, 0, 0),), now_monotonic=t_now)
                cache.build_snapshot(reg, dist_gate_m=1.0, accumulate_pairs=True, now_monotonic=t_now)

            fitted_pose, rmse_m, n_pairs = cache.solve_accumulated_se2(1)
            self.assertEqual(n_pairs, 3)
            self.assertLess(rmse_m, 1e-6)
            self.assertAlmostEqual(fitted_pose.offset_x_m, -0.32, places=5)
            self.assertAlmostEqual(fitted_pose.offset_y_m, -0.04, places=5)
            self.assertAlmostEqual(fitted_pose.yaw_deg, -27.0, places=5)

    def test_sift_overlap_automatch_and_workbench_save_workflow(self):
        cam_b = RadarCameraTransformer.from_json(CAM_B_JSON)
        gt_cam_c = _build_ground_truth_camera(cam_b, (0.0, 0.25, 0.0), yaw_right_deg=18.0)

        # Render rich textured billboard at X_long = 12 m in Radar B into both distorted cameras
        rng = np.random.default_rng(42)
        tex = rng.integers(30, 230, size=(240, 360, 3), dtype=np.uint8)
        tex = cv.GaussianBlur(tex, (5, 5), 1.1)
        for k in range(25):
            cx, cy = int(rng.integers(20, 340)), int(rng.integers(20, 220))
            cv.circle(tex, (cx, cy), int(rng.integers(5, 16)), (255, 255, 255), 2)
            cv.rectangle(tex, (cx - 8, cy - 8), (cx + 8, cy + 8), (15, 15, 15), -1)

        def _render_billboard(cam: RadarCameraTransformer) -> np.ndarray:
            uu, vv = np.meshgrid(np.arange(1920, dtype=np.float32), np.arange(1080, dtype=np.float32))
            undist = cv.undistortPoints(
                np.stack([uu, vv], axis=-1).reshape(-1, 1, 2),
                cam.intrinsic,
                cam.distortion,
            ).reshape(1080, 1920, 2)
            rays_cam = np.dstack([undist, np.ones((1080, 1920), dtype=np.float64)])
            r_mat = cam.rotation_matrix
            c_pos = -(r_mat.T @ cam.translation_vector.reshape(3))
            rays_w = rays_cam @ r_mat  # equivalent to (r_mat.T @ ray) per pixel
            depth = np.where(rays_w[:, :, 0] > 0.05, (12.0 - c_pos[0]) / np.maximum(rays_w[:, :, 0], 0.05), -1.0)
            hit_lat = c_pos[1] + depth * rays_w[:, :, 1]
            hit_down = c_pos[2] + depth * rays_w[:, :, 2]
            map_x = ((hit_lat - (-2.0)) / 8.0 * 360.0).astype(np.float32)
            map_y = ((hit_down - (-3.5)) / 5.0 * 240.0).astype(np.float32)
            valid = (depth > 0) & (map_x >= 0) & (map_x < 359) & (map_y >= 0) & (map_y < 239)
            map_x = np.where(valid, map_x, -1.0)
            map_y = np.where(valid, map_y, -1.0)
            return cv.remap(tex, map_x, map_y, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(20, 20, 20))

        img_b = _render_billboard(cam_b)
        img_c = _render_billboard(gt_cam_c)
        matches = auto_match_sift_overlap(img_b, img_c, cam_b, max_matches=30)
        self.assertGreaterEqual(len(matches), 8)

        with TemporaryDirectory() as tmp:
            ext_path = Path(tmp) / "multi_sensor_extrinsics.json"
            reg = MultiSensorRegistry(ext_path, CAM_B_JSON)
            reg.set_radar_pose(3, 0.25, 0.0, 18.0)
            saved_events = []
            wb = CalibrationWorkbench(reg, target_channel=3, on_saved_cb=lambda ch, res: saved_events.append((ch, res)))
            _, jpg_b = cv.imencode(".jpg", img_b)
            _, jpg_c = cv.imencode(".jpg", img_c)
            wb.receive_live_pair({
                "image_b_bytes": jpg_b.tobytes(),
                "image_s_bytes": jpg_c.tobytes(),
                "radar_points_b": (),
                "radar_points_s": (),
            })
            wb.correspondences.extend(matches)
            vals = {
                "wb_z_down": "0.0",
                "wb_dlong": "0.0",
                "wb_dlat": "0.25",
                "wb_ddown": "0.0",
                "wb_solver_6dof": False,
                "wb_prior_bracket": False,
            }
            wb._run_solve(vals)
            self.assertIsNotNone(wb.last_result)
            np.testing.assert_allclose(
                wb.last_result.extrinsic_from_radar_b[:3, :3],
                gt_cam_c.rotation_matrix,
                atol=0.015,
            )
            reg.set_lateral_camera_calibration(
                3,
                wb.last_result.extrinsic_from_radar_b,
                solver_mode="rotation_3dof",
                camera_shift_from_b_m=(0.0, 0.25, 0.0),
                rmse_px=wb.last_result.rmse_px,
            )
            reg.save()
            wb.close()
            self.assertTrue(ext_path.is_file())
            reloaded = MultiSensorRegistry(ext_path, CAM_B_JSON)
            self.assertTrue(reloaded.lateral_cameras[3]["calibrated"])


if __name__ == "__main__":
    unittest.main()
