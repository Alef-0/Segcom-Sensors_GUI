"""Exercise GUI orchestration with fake processes; never connect/open a UI."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch

import main
from calibration.display_qt import QRClockWindow


class CalibrationWorkflowTests(unittest.TestCase):
    def setUp(self):
        validation = patch.object(
            main,
            "_validate_calibration_decoder",
            return_value="rtx",
        )
        validation.start()
        self.addCleanup(validation.stop)

    def fixture(self):
        process = Mock()
        process.is_alive.return_value = True
        process.exitcode = 0
        context = Mock()
        context.Process.return_value = process
        first_qr_event = Mock()
        first_qr_event.is_set.return_value = False
        context.Event.side_effect = [Mock(), Mock(), first_qr_event]
        runtime = main.RuntimeState(process_context=context)
        config = SimpleNamespace(calibration_camera=True, connected_cam=True,
                                 calibration_recording=False,
                                 show_calibration_error=Mock(), change_calibration_clock=Mock(),
                                 change_calibration_recording=Mock(),
                                 window={"calibration_status": Mock()})
        return config, runtime, process

    def test_display_journal_destination_precedes_delayed_camera_recording(self):
        config, runtime, process = self.fixture()
        camera = Mock()
        with TemporaryDirectory() as folder, patch.object(main.time, "monotonic", return_value=100):
            main._start_calibration_clock({"record_folder": folder}, config, runtime)
            destination = Path(runtime.calibration_prepared_folder)
            self.assertTrue(destination.is_dir())
            arguments = runtime.process_context.Process.call_args.kwargs
            display_options = arguments["args"][2]
            self.assertEqual(display_options["journal_path"], str(destination / "display_timestamps.jsonl"))
            self.assertEqual(display_options["screen_index"], 0)
            self.assertEqual(display_options["visible_qrs"], 2)
            self.assertEqual(display_options["grid_qrs"], 4)
            self.assertEqual(
                display_options["qr_mask_pattern"],
                main.CALIBRATION_QR_MASK_PATTERN,
            )
            self.assertEqual(display_options["display_backend"], "qt")
            wait_event = display_options["recording_wait_event"]
            first_qr_event = display_options["first_qr_event"]
            self.assertIs(wait_event, runtime.calibration_recording_wait_event)
            self.assertIs(first_qr_event, runtime.calibration_first_qr_event)
            wait_event.set.assert_called_once_with()
            self.assertNotIn("visible_frames", display_options)
            self.assertIsNone(runtime.calibration_recording_deadline)
            config.window["calibration_status"].update.assert_called_with(
                "QR ACTIVE — WAITING FOR FIRST QR FRAME"
            )
            main._service_calibration(config, runtime, camera)
            camera.send.assert_not_called()
            with patch.object(main.time, "monotonic", return_value=200):
                main._service_calibration(config, runtime, camera)
            camera.send.assert_not_called()
            first_qr_event.is_set.return_value = True
            with patch.object(main.time, "monotonic", return_value=200):
                main._service_calibration(config, runtime, camera)
            self.assertEqual(runtime.calibration_recording_deadline, 207)
            config.window["calibration_status"].update.assert_called_with(
                "QR ACTIVE — RECORDING IN 7 SECONDS"
            )
            with patch.object(main.time, "monotonic", return_value=206.9):
                main._service_calibration(config, runtime, camera)
            camera.send.assert_not_called()
            wait_event.clear.assert_not_called()
            with patch.object(main.time, "monotonic", return_value=207):
                main._service_calibration(config, runtime, camera)
            camera.send.assert_called_once_with(("record_start", {
                "folders": {4: str(destination)}, "calibration": True,
                "display_journal": "display_timestamps.jsonl"}))
            wait_event.clear.assert_not_called()
            main._apply_status_message(
                "calibration_recording_state", {"active": True}, config, runtime,
                Mock(), camera, Mock(), Mock(),
            )
            wait_event.clear.assert_called_once_with()
            self.assertEqual(runtime.calibration_recording_folder, str(destination))
            config.calibration_recording = True
            process.is_alive.return_value = False
            main._service_calibration(config, runtime, camera)
            camera.send.assert_called_with(("record_stop", None))

    def test_closing_before_deadline_cancels_capture_but_preserves_destination(self):
        config, runtime, process = self.fixture()
        with TemporaryDirectory() as folder:
            main._start_calibration_clock({"record_folder": folder}, config, runtime)
            destination = Path(runtime.calibration_prepared_folder)
            process.is_alive.return_value = False
            camera = Mock()
            main._service_calibration(config, runtime, camera)
            camera.send.assert_not_called()
            self.assertIsNone(runtime.calibration_recording_deadline)
            self.assertIsNone(runtime.calibration_prepared_folder)
            self.assertTrue(destination.exists())

    def test_qr_display_without_camera_shows_red_strip_without_scheduling_capture(self):
        for calibration_camera, connected_cam in ((False, False), (True, False)):
            with self.subTest(calibration_camera=calibration_camera, connected_cam=connected_cam):
                config, runtime, process = self.fixture()
                config.calibration_camera = calibration_camera
                config.connected_cam = connected_cam
                runtime.process_context.Event.side_effect = threading.Event
                main._start_calibration_clock({}, config, runtime)
                process.start.assert_called_once_with()
                display_options = runtime.process_context.Process.call_args.kwargs["args"][2]
                wait_event = display_options["recording_wait_event"]
                self.assertIs(wait_event, runtime.calibration_recording_wait_event)
                self.assertTrue(wait_event.is_set())
                first_qr_event = display_options["first_qr_event"]
                self.assertIsNone(display_options["journal_path"])
                self.assertIsNone(runtime.calibration_recording_deadline)
                camera = Mock()
                main._service_calibration(config, runtime, camera)
                camera.send.assert_not_called()
                self.assertTrue(wait_event.is_set())
                self.assertIsNone(runtime.calibration_recording_deadline)
                first_qr_event.set()
                with patch.object(main.time, "monotonic", return_value=100):
                    main._service_calibration(config, runtime, camera)
                self.assertEqual(runtime.calibration_recording_deadline, 107)
                with patch.object(main.time, "monotonic", return_value=106.9):
                    main._service_calibration(config, runtime, camera)
                self.assertTrue(wait_event.is_set())
                with patch.object(main.time, "monotonic", return_value=107):
                    main._service_calibration(config, runtime, camera)
                self.assertFalse(wait_event.is_set())
                self.assertIsNone(runtime.calibration_recording_deadline)
                camera.send.assert_not_called()
                config.show_calibration_error.assert_not_called()

    def test_qr_display_started_while_recording_still_runs_red_strip_countdown(self):
        config, runtime, _ = self.fixture()
        config.calibration_recording = True
        runtime.process_context.Event.side_effect = threading.Event
        main._start_calibration_clock({}, config, runtime)
        self.assertTrue(runtime.calibration_recording_wait_event.is_set())
        self.assertIsNone(runtime.calibration_recording_deadline)
        runtime.calibration_first_qr_event.set()
        camera = Mock()
        with patch.object(main.time, "monotonic", return_value=100):
            main._service_calibration(config, runtime, camera)
        with patch.object(main.time, "monotonic", return_value=107):
            main._service_calibration(config, runtime, camera)
        self.assertFalse(runtime.calibration_recording_wait_event.is_set())
        camera.send.assert_not_called()
        config.show_calibration_error.assert_not_called()

    def test_camera_closing_does_not_cancel_red_strip_countdown(self):
        for close_before_first_frame in (True, False):
            with self.subTest(close_before_first_frame=close_before_first_frame):
                config, runtime, _ = self.fixture()
                config.change_calibration_camera = Mock()
                runtime.process_context.Event.side_effect = threading.Event
                camera = Mock()
                with TemporaryDirectory() as folder:
                    main._start_calibration_clock({"record_folder": folder}, config, runtime)
                    first_qr_event = runtime.calibration_first_qr_event
                    if not close_before_first_frame:
                        first_qr_event.set()
                        with patch.object(main.time, "monotonic", return_value=100):
                            main._service_calibration(config, runtime, camera)
                    main._apply_status_message(
                        "calibration_camera_state", {"active": False}, config, runtime,
                        Mock(), camera, Mock(), Mock(),
                    )
                    config.calibration_camera = False
                    if close_before_first_frame:
                        first_qr_event.set()
                        with patch.object(main.time, "monotonic", return_value=100):
                            main._service_calibration(config, runtime, camera)
                    with patch.object(main.time, "monotonic", return_value=106.9):
                        main._service_calibration(config, runtime, camera)
                    self.assertTrue(runtime.calibration_recording_wait_event.is_set())
                    with patch.object(main.time, "monotonic", return_value=107):
                        main._service_calibration(config, runtime, camera)
                    self.assertFalse(runtime.calibration_recording_wait_event.is_set())
                    camera.send.assert_not_called()
                    config.show_calibration_error.assert_not_called()

    def test_first_qr_swap_signals_recording_countdown(self):
        first_qr_event = threading.Event()
        monitor = Mock()
        monitor.observe.return_value = {}
        window = SimpleNamespace(
            pending_frame={
                "cell": 0, "display_index": 0, "marker_ns": 1,
                "submit_ns": 2, "paint_start_ns": 1,
                "recording_wait_active": True,
            },
            monitor=monitor,
            timestamp_mode="paint-start",
            resumed_after_pause=False,
            journal=Mock(),
            first_qr_event=first_qr_event,
            next_display_index=0,
            paused=True,
        )
        QRClockWindow._frame_swapped(window)
        self.assertTrue(first_qr_event.is_set())
        self.assertEqual(window.next_display_index, 1)

    def test_failed_start_does_not_schedule_recording(self):
        config, runtime, process = self.fixture()
        process.start.side_effect = OSError("process start failed")
        with TemporaryDirectory() as folder:
            main._start_calibration_clock({"record_folder": folder}, config, runtime)
        self.assertIsNone(runtime.calibration_clock_process)
        self.assertIsNone(runtime.calibration_recording_deadline)
        config.show_calibration_error.assert_called_once()

    def test_pygame_selection_reuses_grid_monitor_and_reported_refresh(self):
        config, runtime, _ = self.fixture()
        with TemporaryDirectory() as folder:
            main._start_calibration_clock({
                "record_folder": folder,
                "calibration_display": "Pygame / SDL",
                "calibration_screen": "1: HDMI-1 @ 144.000 Hz",
                "calibration_grid_qrs": 12,
                "calibration_visible_qrs": 8,
            }, config, runtime)

        display_options = runtime.process_context.Process.call_args.kwargs["args"][2]
        self.assertEqual(display_options["display_backend"], "pygame")
        self.assertEqual(display_options["screen_index"], 1)
        self.assertEqual(display_options["grid_qrs"], 12)
        self.assertEqual(display_options["visible_qrs"], 8)
        self.assertEqual(
            display_options["qr_mask_pattern"],
            main.CALIBRATION_QR_MASK_PATTERN,
        )
        self.assertEqual(display_options["refresh_hz"], 144.0)

    def test_calibration_layout_has_fixed_qr_mode_without_amount_control(self):
        def elements(rows):
            for row in rows:
                for element in row:
                    yield element
                    nested = getattr(element, "Rows", None)
                    if nested:
                        yield from elements(nested)
        controls = list(elements(main.Configurations._create_calibration_layout()))
        self.assertTrue(any(
            "fullscreen QR view records after 7 seconds" in getattr(element, "DisplayText", "")
            for element in controls
        ))
        self.assertFalse(any(getattr(element, "Key", None) == "calibration_visible_frames"
                             for element in controls))
        button = next(element for element in controls
                      if getattr(element, "Key", None) == "calibration_clock_start")
        self.assertEqual(button.ButtonText, "START QR CALIBRATION")
        self.assertTrue(any(getattr(element, "Key", None) == "calibration_decoder"
                            for element in controls))
        self.assertTrue(any(getattr(element, "Key", None) == "calibration_display"
                            for element in controls))
        self.assertTrue(any(getattr(element, "Key", None) == "calibration_screen"
                            for element in controls))
        self.assertTrue(any(getattr(element, "Key", None) == "calibration_visible_qrs"
                            for element in controls))
        self.assertTrue(any(getattr(element, "Key", None) == "calibration_grid_qrs"
                            for element in controls))
        self.assertFalse(any(type(element).__name__ == "VerticalSeparator"
                             for element in controls))

    def test_camera_timestamp_correction_uses_provisional_default(self):
        self.assertEqual(main._camera_latency_settings({}), (145, 87.348))

    def test_calibration_settings_use_narrow_grouped_rows(self):
        layout = main.Configurations._create_calibration_layout()
        first_row_keys = {getattr(element, "Key", None) for element in layout[0]}
        second_row_keys = {getattr(element, "Key", None) for element in layout[1]}
        third_row_keys = {getattr(element, "Key", None) for element in layout[2]}

        self.assertTrue({
            "calibration_latency_apply",
        }.issubset(first_row_keys))
        self.assertTrue({
            "calibration_latency_status",
            "calibration_decoder",
            "calibration_display",
        }.issubset(second_row_keys))
        self.assertTrue({
            "calibration_screen",
            "calibration_grid_qrs",
            "calibration_visible_qrs",
        }.issubset(third_row_keys))

    def test_transposition_control_selects_group_b_and_notifies_both_workers(self):
        config = SimpleNamespace(
            window={"choose_2": Mock()},
            change_transposition=Mock(),
        )
        radar = Mock()
        camera = Mock()

        main._set_transposition(True, config, radar, camera)

        config.window["choose_2"].update.assert_called_once_with(value=True)
        self.assertEqual(radar.send.call_args_list[0].args[0], ("choose", 2))
        self.assertEqual(camera.send.call_args_list[0].args[0], ("choose", 2))
        radar.send.assert_called_with(("transposition", {"active": True}))
        camera.send.assert_called_with(("transposition", {"active": True}))
        config.change_transposition.assert_called_once_with(True, None)

    def test_radar_controls_include_transposition_tab(self):
        config = main.Configurations.__new__(main.Configurations)
        config.create_filters()

        self.assertEqual(config.filters.Title, "Radar controls")
        controls = [
            element
            for row in config._create_transposition_layout()
            for element in row
        ]
        self.assertTrue(any(
            getattr(element, "Key", None) == "transposition_toggle"
            for element in controls
        ))

    def test_monitor_choices_only_show_index_name_and_refresh_rate(self):
        catalog = [{
            "index": 1,
            "name": "HDMI-1",
            "width": 1920,
            "height": 1080,
            "refresh_hz": 144.0,
            "primary": True,
        }]
        completed = SimpleNamespace(
            returncode=0,
            stdout=f"{main.json.dumps(catalog)}\n",
        )

        with patch.object(main.subprocess, "run", return_value=completed):
            self.assertEqual(
                main._qt_screen_choices(),
                ["1: HDMI-1 @ 144.000 Hz"],
            )

    def test_selected_decoder_is_sent_before_opening_calibration_camera(self):
        config = SimpleNamespace(
            connected_radar=False,
            recording=False,
            recording_pending=False,
            playback=False,
            playback_pending=False,
            snapshot_playback=False,
            snapshot_playback_pending=False,
        )
        runtime = main.RuntimeState(pending_calibration_camera={
            "pipeline_latency_ms": 145,
            "latency_adjustment_ms": 109.0,
            "recording_frames_per_30": 30,
            "decoder_backend": "orin",
        })
        camera = Mock()

        main._maybe_open_calibration_camera(config, runtime, camera)

        self.assertEqual(camera.send.call_args_list[0].args[0], (
            "camera_decoder_backend", {"backend": "orin"}
        ))
        self.assertEqual(camera.send.call_args_list[-1].args[0], (
            "calibration_camera", {"active": True}
        ))

    def test_monitor_label_is_converted_to_qt_screen_index(self):
        self.assertEqual(main._calibration_screen_index({
            "calibration_screen": "1: HDMI-1 — 1920×1080 @ 144.000 Hz"
        }), 1)
        self.assertEqual(main._calibration_screen_refresh_hz({
            "calibration_screen": "1: HDMI-1 — 1920×1080 @ 144.000 Hz"
        }), 144.0)
        self.assertIsNone(main._calibration_screen_refresh_hz({
            "calibration_screen": "0: Primary @ unknown Hz"
        }))

    def test_display_backend_selection_is_validated(self):
        self.assertEqual(main._calibration_display_backend({}), "qt")
        self.assertEqual(main._calibration_display_backend({
            "calibration_display": "Pygame / SDL",
        }), "pygame")
        with self.assertRaisesRegex(ValueError, "Select a QR display"):
            main._calibration_display_backend({"calibration_display": "Other"})

    def test_pygame_backend_is_dispatched_without_backend_only_option(self):
        stop_event = Mock()
        error_queue = Mock()
        options = {
            "display_backend": "pygame",
            "screen_index": 1,
            "grid_qrs": 12,
            "visible_qrs": 8,
        }

        with (
            patch(
                "calibration.display.run_calibration_display",
                return_value=None,
            ) as run_display,
            patch.object(main, "CalibrationSchedulerPriority") as priority,
        ):
            main._run_calibration_clock_process(stop_event, error_queue, options)

        run_display.assert_called_once_with(
            stop_event,
            screen_index=1,
            grid_qrs=12,
            visible_qrs=8,
        )
        error_queue.put.assert_not_called()
        priority.assert_called_once_with("QR calibration display")
        priority.return_value.enable.assert_called_once_with()

    def test_grid_and_visible_qr_selections_are_validated_together(self):
        self.assertEqual(main._calibration_grid_qrs({
            "calibration_grid_qrs": 12,
        }), 12)
        self.assertEqual(main._calibration_visible_qrs({
            "calibration_visible_qrs": 10,
        }, 12), 10)
        with self.assertRaisesRegex(ValueError, "from 1 to 6"):
            main._calibration_visible_qrs({"calibration_visible_qrs": 7}, 6)
        with self.assertRaisesRegex(ValueError, "must be one of"):
            main._calibration_grid_qrs({"calibration_grid_qrs": 5})

    def test_grid_change_limits_the_visible_qr_choices(self):
        config = main.Configurations.__new__(main.Configurations)
        visible = Mock()
        config.window = {"calibration_visible_qrs": visible}

        config.change_calibration_qr_grid(6, 10)

        visible.update.assert_called_once_with(
            values=(1, 2, 3, 4, 5, 6),
            value=6,
        )

    def test_visualization_launches_single_analyzer_with_only_the_folder(self):
        with TemporaryDirectory(prefix="qr calibration ") as folder:
            path = Path(folder)
            (path / "camera_timestamps.jsonl").write_text("{}\n")
            runtime = main.RuntimeState()
            config = SimpleNamespace(window={"visualization_open": Mock(), "visualization_status": Mock()})
            with patch.object(main.subprocess, "Popen") as launch:
                main._start_calibration_visualization({"visualization_folder": folder}, config, runtime)
            command = launch.call_args.args[0]
            self.assertEqual(Path(command[1]).name, "analyze_calibration_recording.py")
            self.assertEqual(command[2:], [str(path.resolve())])


if __name__ == "__main__":
    unittest.main()
