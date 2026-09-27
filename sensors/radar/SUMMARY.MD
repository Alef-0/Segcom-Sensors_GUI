# Radar Integration Directory Summary

This folder contains the radar integration modules for Continental ARS40X radar sensors, including TCP network gateway communication, CAN message decoding, per-channel point/object accumulation, sensor configuration encoding, and background worker lifecycle management.

## Transport Specification

The radar subsystem connects to the network CAN gateway at `192.168.1.101:2323` over TCP. The gateway stream is split into fixed 23-byte records using the little-endian packed layout `<BBIQ8sB>`:

| Field | Meaning |
| --- | --- |
| byte 0 | CAN data length (DLC) |
| byte 1 | Gateway flags |
| bytes 2-5 | CAN ID (32-bit uint) |
| bytes 6-13 | Gateway timestamp (64-bit uint) |
| bytes 14-21 | 8-byte CAN payload |
| byte 22 | Radar channel (1-3 map to groups A-C) |

## Frame Assembly & Lifecycle

- `0x600` marks the start of a cluster frame and `0x60A` marks the start of an object frame.
- When the next start marker arrives for a channel, the previous frame is considered complete and dispatched for visualization, snapshot history buffer, and PCD recording.
- Cluster quality packets (`0x702`) and object detail packets (`0x60C`-`0x60E`) may arrive separately from general data (`0x701` / `0x60B`); their accumulator classes merge all fields by ID before `snapshot()` generates an immutable frame tuple. Points/objects missing general position data are filtered out.
- There is a deliberate 500 ms pause after sending 0x200 configuration messages to give the Vector gateway interface time to register the packet.

## Files

### [connection_packages.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/connection_packages.py)
Unified CAN message module for Continental ARS40X radar data. Contains payload validation, dataclasses for points and objects, message accumulators, packet decoders, and sensor configuration encoders.
- [`check_payload()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/connection_packages.py#L23-L26): Validates that a CAN payload is exactly 8 bytes.
- [`ObjectStatus`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/connection_packages.py#L32-L37): Dataclass representing 0x60A object status counters and interface version.
- [`RadarPoint`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/connection_packages.py#L40-L57): Dataclass representing a single cluster detection point with kinematics and quality fields.
- [`RadarObject`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/connection_packages.py#L60-L95): Dataclass representing a tracked obstacle object with kinematics, dimensions, and classification.
- [`Clusters_messages`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/connection_packages.py#L98-L122): Accumulator merging 0x701 (general) and 0x702 (quality) cluster frames by cluster ID.
- [`Objects_messages`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/connection_packages.py#L125-L181): Accumulator merging 0x60A-0x60E object packets by object ID.
- [`read_701_cluster_list()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/connection_packages.py#L184-L200): Decodes 0x701 cluster position, velocity, dynamic property, and RCS.
- [`read_702_quality_info()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/connection_packages.py#L203-L206): Decodes 0x702 cluster PDH, ambiguity state, and invalid flags.
- [`read_60a_object_status()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/connection_packages.py#L209-L216): Decodes 0x60A object count, measurement counter, and interface version.
- [`read_60b_object_general()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/connection_packages.py#L219-L235): Decodes 0x60B object distance, velocity, dynamic property, and RCS.
- [`read_60c_object_quality()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/connection_packages.py#L238-L260): Decodes 0x60C RMS variances, measurement state, and probability of existence.
- [`read_60d_object_extended()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/connection_packages.py#L263-L279): Decodes 0x60D acceleration, classification, orientation angle, length, and width.
- [`read_60e_object_warning()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/connection_packages.py#L282-L285): Decodes 0x60E collision warning region flags.
- [`create_200_radar_configuration()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/connection_packages.py#L288-L313): Encodes 0x200 configuration message payload.
- [`read_201_radar_state_extended()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/connection_packages.py#L316-L339): Decodes 0x201 radar state with extended flags.
- [`read_201_radar_state()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/connection_packages.py#L342-L354): Decodes 0x201 radar state returning standard 6-tuple.

### [connection_communication.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/connection_communication.py)
TCP socket transport layer communicating with the CAN-Ethernet gateway (`192.168.1.101:2323`). Buffers stream bytes and slices out 23-byte CAN records.
- [`Can_Connection`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/connection_communication.py#L11-L60): Manages non-blocking TCP socket connection, stream reading, buffering, and message sending.
- [`can_data`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/connection_communication.py#L63-L79): Unpacks 23-byte binary struct into DLC, flags, CAN ID, timestamp, payload bytes, and CAN channel.

### [connection_main.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/connection_main.py)
Background multiprocessing worker process for radar operations. Manages the TCP gateway connection, decodes incoming packets, detects frame boundaries, updates OpenCV real-time visualizer, coordinates PCD recording sessions, and pairs snapshot captures.
- [`put_status()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/connection_main.py#L50-L58): Thread-safe queue dispatcher for status messages to the GUI.
- [`configuration_label()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/connection_main.py#L61-L66): Formats configuration option labels.
- [`treat_201_message()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/connection_main.py#L69-L94): Decodes 0x201 state response and posts telemetry to UI.
- [`send_configuration_message()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/connection_main.py#L97-L113): Encodes and dispatches 0x200 configuration frames.
- [`stop_recording()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/connection_main.py#L116-L135): Stops PCD recording session and notifies GUI.
- [`create_connection_communication()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/radar/connection_main.py#L138-L434): Radar worker process entry point and main event loop.
