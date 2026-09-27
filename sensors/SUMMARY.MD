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

### [sensors/playback.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/playback.py)
Unified session and snapshot playback engine for recorded radar and camera datasets.
- [PlaybackController](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/playback.py): Unified controller handling recorded session and snapshot playback with frame-by-frame stepping, pause, and on-demand snapshot capture.
- [load_recording_entries](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/playback.py), [load_snapshot_entries](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/playback.py): Metadata and disk readers loading sorted `PlaybackEntry` lists from root or session subdirectories.
- [detect_playback_structure](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/playback.py): Identifies snapshot vs continuous recording datasets based on folder contents.
- [playback_main](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/playback.py), [snapshot_playback_main](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/playback.py): Multiprocessing worker process entry points.

### [sensors/auxiliary.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/auxiliary.py)
Core utilities for PCD schemas, file path resolution, image encoding/decoding, and vectorized point cloud filtering.
- [filter_radar_points](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/auxiliary.py#L117): Unified vectorized multi-attribute filter using NumPy arrays for coordinates, RCS, PDH, dynamic properties, ambiguity states, invalid flags, distance cutoff, and BGR color mapping.
- [filter_point_cutoff](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/auxiliary.py#L225), [filter_rcs](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/auxiliary.py#L231): Backward-compatible threshold filter aliases delegating to `filter_radar_points`.
- [save_point_cloud](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/auxiliary.py#L277): Encodes `RadarPoint` or `RadarObject` sequences to PCD files using cluster or extended object schemas.
- [PointCloudReader](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/auxiliary.py#L298): Parses saved PCD files back into `RadarPoint` and `RadarObject` collections, detecting schema variations automatically.
- [image_path](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/auxiliary.py#L52), [point_cloud_path](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/auxiliary.py#L56), [resolve_recording_file](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/auxiliary.py#L68): Safe relative/absolute path generators and legacy layout resolvers.
- [encode_jpeg](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/auxiliary.py#L90), [decode_jpeg](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/auxiliary.py#L97): Lightweight image format conversion helpers.

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
