# Sensors Package Summary

This package manages the hardware and network interface boundaries of the SEGCOM Sensors GUI application, structured into three sensor subpackages: Camera (RTSP/GStreamer), Radar (Continental ARS40X CAN gateway), and GPS (Dahua DVR position polling).

## Subdirectories

### [camera/](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/camera)
RTSP camera integration, dynamic GStreamer pipeline construction (supporting hardware VA-API, NVIDIA NVCODEC, Apple VideoToolbox, and CPU backends), clock synchronization, NTP epoch extrapolation, and timestamp adjustment.

### [gps/](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/gps)
Dahua DVR HTTP digest GPS integration. Polls device position status, parses DMS coordinates into decimal degrees with hemisphere detection, and emits formatted UI coordinates and Google Maps search links.
- See detailed documentation in [sensors/gps/SUMMARY.MD](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/gps/SUMMARY.MD).

### [radar/](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar)
Continental ARS40X radar sensor integration. Contains TCP socket transport to the CAN gateway, unified CAN payload parsing and generation, per-channel point/object accumulation, and the radar background worker loop.
- See detailed documentation in [sensors/radar/SUMMARY.MD](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/SUMMARY.MD).
