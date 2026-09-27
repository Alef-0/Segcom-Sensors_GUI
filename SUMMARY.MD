# Root Directory Summary

This folder contains the application entry points, core GUI interface, configuration definitions, and environment specifications for the SEGCOM Sensors GUI system.

## Files

### [main.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/main.py)
Main application entry point. Parses CLI flags with `argparse`, sets up multiprocessing pipelines (`spawn`), creates worker communication channels (`WorkerPipes`), launches background worker processes (radar, camera, GPS, playback), runs the GUI event loop, and coordinates graceful process shutdown.
- `parse_arguments()`: Parses command line parameters (`--font-size`, `--font-family`).
- `main()`: Orchestrates IPC channels, process lifecycles, and event loop execution.

### [menu_layout.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/menu_layout.py)
Pure declarative layout generation using `FreeSimpleGUI`. Defines tabs, frames, buttons, sliders, and real-time input fields without mixing state or event logic.
- `build_radar_columns()`: Builds telemetry columns for radar channels A, B, and C.
- `build_telemetry_frame()`: Builds top telemetry display with received CAN IDs and camera NTP time.
- `build_radar_options()`, `build_record_tab()`: Builds configuration tabs with recording, snapshot button, and unified playback controls.
- `build_graph_tab()`: Builds point cutoff distance, graph resolution, and graph range tab.
- `build_video_tab()`: Builds camera resolution, recorded FPS, jitter latency, and offset correction tab.
- `build_filter_tab()`, `build_cluster_filter_tab()`, `build_transposition_tab()`: Builds filter tabs.
- `build_main_layout()`: Assembles the root window layout.

### [menu_controls.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/menu_controls.py)
Window and widget state controller. Owns the GUI window instance, handles input validation and real-time character filtering, and updates UI labels and button states.
- `MenuControls`: Controller class wrapping `sg.Window` and managing operational states (live, recording, playback, snapshot).
- `sanitize_numeric_input()`: Strips non-numeric characters from input fields in real time on keystroke events.
- `validate_playback_resolution()`, `validate_point_cutoff()`, `validate_graph_settings()`, `validate_recording_rate()`: Validates input parameters.
- `refresh_mode_controls()`: Coordinates enabling/disabling widgets based on active sensor/playback modes.
- `update_radar_connection()`, `update_cam_connection()`, `update_recording_state()`, `update_playback_state()`, `update_playback_progress()`: UI state updaters.

### [menu_functions.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/menu_functions.py)
Application event dispatcher and IPC bridge. Bundles worker communication pipes, manages process lifecycle, coordinates background worker messages, and handles relative path normalization.
- `WorkerPipes`: Dataclass bundling worker communication channels (radar, camera, GPS, playback).
- `RuntimeState`: Dataclass tracking pending playback and recording states.
- `format_relative_path()`: Normalizes paths within the current workspace to relative `./` paths.
- `handle_gui_event()`: Dispatches window events to appropriate worker pipes and controls.
- `apply_status_message()`: Drains and applies worker status notifications to the GUI.
- `run_event_loop()`: Event polling and worker queue drain loop.
- `start_recording()`, `request_recording_stop()`, `start_snapshot()`: Sensor recording and snapshot triggers.
- `request_playback()`: Unified playback state management with snapshot stepping/pause logic.
- `shutdown_workers()`, `join_processes()`: Graceful worker shutdown and termination routines.


### [convert_to_csv.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/convert_to_csv.py)
Offline data conversion utility converting radar PCD files, recordings, and CAN logs into CSV files.

### [camera_matrixes.json](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/camera_matrixes.json)
JSON configuration holding camera intrinsic, distortion, and extrinsic projection matrices.

### [fix_date.sh](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/fix_date.sh)
Shell script synchronizing remote sensor host date/time over SSH.

### [requirements.txt](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/requirements.txt)
Python runtime dependencies for the project.

## Subdirectories

### [sensors/](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors)
Sensor drivers, background worker integrations for Camera, GPS, and Radar, unified filtering (`sensors/filter`), recording, playback, snapshot, and auxiliary processing utilities. See [sensors/SUMMARY.MD](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/SUMMARY.MD).


