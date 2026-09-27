# Sensors Package Summary

This package manages the hardware and network interface boundaries of the SEGCOM Sensors GUI application, structured into sensor subpackages (Camera, Radar, GPS) along with unified recording, snapshot, and auxiliary processing utilities.

## Top-Level Modules

### [sensors/recording.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/recording.py)
Unified session and continuous recording management for both camera and radar streams.
- [CameraRecorder](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/recording.py#L35): Manages off-thread disk serialization of camera JPEG frames and outputs the per-session `camera_timestamps.csv` index with PTS, camera NTP, and monotonic arrival clocks (emitting `-1,-1,-1,monotonic` for invalid PTS frames).
- [RadarRecorder](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/recording.py#L225) (alias `RadarRecordingSession`): Coordinates multi-channel radar point-cloud folder generation, background PCD queue writing, and `recording.json` / `timestamps.json` metadata index generation.
- [save_point_cloud](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/recording.py#L31): Writes PCD files from structured point cloud objects using the module's configured PointCloud writer class.

### [sensors/snapshot.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/snapshot.py)
Unified single-frame manual snapshot writer for radar point clouds and camera images.
- [SnapshotWriter](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/snapshot.py#L32) (alias `ManualSnapshotWriter`): Pairs a single camera JPEG frame with the corresponding radar PCD frame, maintains sequential indexing across snapshots, and updates `recording.json` and `timestamps.json`.

### [sensors/auxiliary.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/auxiliary.py)
Core utilities for PCD schemas, file path resolution, image encoding/decoding, and point cloud filtering.
- [save_point_cloud](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/auxiliary.py#L145): Encodes `RadarPoint` or `RadarObject` sequences to PCD files using cluster or extended object schemas.
- [PointCloudReader](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/auxiliary.py#L164): Parses saved PCD files back into `RadarPoint` and `RadarObject` collections, detecting schema variations automatically.
- [image_path](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/auxiliary.py#L52), [point_cloud_path](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/auxiliary.py#L56), [resolve_recording_file](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/auxiliary.py#L68): Safe relative/absolute path generators and legacy layout resolvers.
- [encode_jpeg](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/auxiliary.py#L86), [decode_jpeg](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/auxiliary.py#L93): Lightweight image format conversion helpers.
- [filter_point_cutoff](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/auxiliary.py#L100), [filter_rcs](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/auxiliary.py#L104): Range and radar cross-section (RCS) threshold filters.

## Subdirectories

### [camera/](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/camera)
RTSP camera integration, dynamic GStreamer pipeline construction, clock synchronization, NTP extraction, and radar-to-camera transposition.
- See detailed documentation in [sensors/camera/SUMMARY.MD](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/camera/SUMMARY.MD).

### [gps/](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/gps)
Dahua DVR HTTP digest GPS integration. Polls device position status, parses DMS coordinates into decimal degrees with hemisphere detection, and emits formatted UI coordinates and Google Maps search links.
- See detailed documentation in [sensors/gps/SUMMARY.MD](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/gps/SUMMARY.MD).

### [radar/](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar)
Continental ARS40X radar sensor integration. Contains TCP socket transport to the CAN gateway, unified CAN payload parsing and generation, per-channel point/object accumulation, and the radar background worker loop.
- See detailed documentation in [sensors/radar/SUMMARY.MD](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/SUMMARY.MD).
