"""Exercise GUI orchestration with fake processes; never connect/open a UI."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import main


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
        runtime = main.RuntimeState(process_context=context)
        config = SimpleNamespace(calibration_camera=True, connected_cam=True,
                                 calibration_recording=False,
                                 show_calibration_error=Mock(), change_calibration_clock=Mock(),
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
            self.assertEqual(display_options["display_backend"], "qt")
            self.assertNotIn("visible_frames", display_options)
            main._service_calibration(config, runtime, camera)
            camera.send.assert_not_called()
            with patch.object(main.time, "monotonic", return_value=103):
                main._service_calibration(config, runtime, camera)
            camera.send.assert_called_once_with(("record_start", {
                "folders": {4: str(destination)}, "calibration": True,
                "display_journal": "display_timestamps.jsonl"}))
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

        with patch(
            "calibration.display.run_calibration_display",
            return_value=None,
        ) as run_display:
            main._run_calibration_clock_process(stop_event, error_queue, options)

        run_display.assert_called_once_with(
            stop_event,
            screen_index=1,
            grid_qrs=12,
            visible_qrs=8,
        )
        error_queue.put.assert_not_called()

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
