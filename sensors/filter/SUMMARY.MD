# Filter Package Summary

This package provides unified filtering, 2D top-down Cartesian radar plotting, coordinate transformations, and real-time radar-to-camera image transposition overlays.

## Modules

### [filter.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/filter/filter.py)
Consolidated radar filtering engine, GUI filter schema, option descriptors, and dynamic color mappings.
- [FilterOption](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/filter/filter.py#L21): Descriptor for UI filter toggle controls and keys.
- [parse_filter_key](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/filter/filter.py#L60): Parses GUI event keys into filter field and option choices.
- [filter_radar_points](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/filter/filter.py#L76): Fast NumPy vectorized filtering across coordinates, distance cutoff, RCS, PDH quality, dynamic property, ambiguity state, and invalid flags.
- [filter_point_cutoff](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/filter/filter.py#L122), [filter_rcs](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/filter/filter.py#L128): Threshold-based filter helpers.
- [Filter_graph](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/filter/filter.py#L134): State controller managing enabled filter sets from GUI events and executing point/object filtering.

### [graph_draw.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/filter/graph_draw.py)
OpenCV-based 2D top-down Cartesian radar plotting canvas.
- [Graph_radar](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/filter/graph_draw.py#L15): Canvas renderer with dynamic grid scaling, distance arcs, cutoff boundary, mouse click inspection, and window event handling.

### [transposition.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/filter/transposition.py)
Radar-to-camera 3D projection, camera frame transposition overlay, and inter-process communication queue helpers.
- [RadarCameraTransformer](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/filter/transposition.py#L79): Validates camera intrinsics, distortion coefficients, and 4x4 extrinsic transformation matrices with SVD normalization, projecting 3D radar coordinates to camera space and 2D pixel coordinates.
- [RadarCameraOverlay](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/filter/transposition.py#L148): High-performance frame overlay drawing projected radar detections on live camera frames with latency checks and scaling.
- [transposition_payload](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/filter/transposition.py#L55): Prepares filtered points within cutoff distance for inter-process transposition.
- [put_latest](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/filter/transposition.py#L29), [get_latest](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/filter/transposition.py#L44), [clear_latest](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/filter/transposition.py#L19): Non-blocking IPC queue utilities.
