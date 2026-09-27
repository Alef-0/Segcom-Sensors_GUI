# Camera Package Summary

The camera package connects to Dahua DVR RTSP video streams, handles hardware/software decoding, manages timestamps, produces frames for live preview and recording, and integrates radar-to-camera transposition overlays.

## Files

### [camera_pipeline.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/camera/camera_pipeline.py)
Hardware decoder backend discovery and GStreamer pipeline construction. Supports NVIDIA GPU (`rtx` using `nvh264dec`), NVIDIA Jetson (`orin` using `nvv4l2decoder`), and CPU fallback (`cpu` using `avdec_h264`). Constructs the pipeline with optional display appsink (`include_display_sink=True/False`) and full-resolution capture appsink.
- [`is_jetson_platform()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/camera/camera_pipeline.py#L53): Detects Jetson board via `/etc/nv_tegra_release` or `/proc/device-tree`.
- [`available_decoder_backends()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/camera/camera_pipeline.py#L86): Finds usable H.264 decoders based on installed GStreamer plugins.
- [`build_camera_pipeline()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/camera/camera_pipeline.py#L132): Builds the GStreamer launch string for capture and optional live display sinks.

### [camera_gstreamer.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/camera/camera_gstreamer.py)
Background camera worker process owning RTSP connection state, GStreamer lifecycle, live display rendering in OpenCV, recording hooks, channel switching, and radar transposition overlay.
- [`GStreamerPipeline`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/camera/camera_gstreamer.py#L48): Worker class managing GStreamer pipeline and GLib main loop.
- [`display_latest_frame()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/camera/camera_gstreamer.py#L368): Renders live camera frame and draws radar transposition points when enabled.
- [`on_new_display_sample()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/camera/camera_gstreamer.py#L274) / [`on_new_capture_sample()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/camera/camera_gstreamer.py#L289): Appsink callbacks for live display and recording.
- [`process_commands()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/camera/camera_gstreamer.py#L324): Handles IPC messages for channel switching, recording, snapshots, and latency settings.
- [`gstreamer_main()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/camera/camera_gstreamer.py#L461): Entry point for the camera background worker process.
