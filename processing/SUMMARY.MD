# Processing Package Summary

This package groups radar and camera playback and visualization services used by `main.py` and the sensor processing pipeline. Recording services have been unified under the [sensors/](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors) package.

## Package Root

### [__init__.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/processing/__init__.py)
Exposes backward-compatibility aliases and re-exports for recording and snapshot services:
- `CameraRecorder`, `CameraSnapshotRecorder`, `RadarRecorder`, `RadarRecordingSession` from [sensors.recording](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/recording.py).
- `SnapshotWriter`, `ManualSnapshotWriter` from [sensors.snapshot](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/snapshot.py).
- `PointCloudReader` from [sensors.auxiliary](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/auxiliary.py).

## Subdirectories

### [playback/](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/processing/playback)
Loads multi-sensor recordings and saved snapshot pairs for review.
- See detailed documentation in [processing/playback/SUMMARY.MD](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/processing/playback/SUMMARY.MD).

### [visualization/](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/processing/visualization)
Top-down Cartesian radar plotting, radar point filtering, radar-camera projection, and transposition overlays.
- See detailed documentation in [processing/visualization/SUMMARY.MD](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/processing/visualization/SUMMARY.MD).
