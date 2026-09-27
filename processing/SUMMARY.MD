# Processing Package Summary

This package groups radar and camera playback and visualization services used by `main.py` and the sensor processing pipeline. Recording services have been unified under the [sensors/](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors) package.

## Package Root

### [__init__.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/processing/__init__.py)
Exposes backward-compatibility aliases and re-exports for recording, snapshot, and playback services:
- `CameraRecorder`, `CameraSnapshotRecorder`, `RadarRecorder`, `RadarRecordingSession` from [sensors.recording](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/recording.py).
- `SnapshotWriter`, `ManualSnapshotWriter` from [sensors.snapshot](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/snapshot.py).
- `PlaybackController`, `SnapshotPlaybackController`, `PlaybackEntry`, `load_recording_entries`, `load_snapshot_entries` from [sensors.playback](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/playback.py).
- `PointCloudReader` from [sensors.auxiliary](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/auxiliary.py).

## Subdirectories

### [visualization/](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/processing/visualization)
Top-down Cartesian radar plotting, radar point filtering, radar-camera projection, and transposition overlays.
- See detailed documentation in [processing/visualization/SUMMARY.MD](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/processing/visualization/SUMMARY.MD).
