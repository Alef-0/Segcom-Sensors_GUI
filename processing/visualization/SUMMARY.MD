# Processing Visualization Package Summary

This package provides radar point cloud filtering, 2D top-down Cartesian radar plotting, coordinate transformations, and radar-to-camera image overlay rendering.

## Modules

### [filter_schema.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/processing/visualization/filter_schema.py)
Declares filter identifiers, types, default limits, and validation rules for radar point/object filters.

### [graph_filter.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/processing/visualization/graph_filter.py)
Applies state, quality, RCS, velocity, and distance filters to radar detections.
- [filter_points](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/processing/visualization/graph_filter.py#L32): Filters an iterable of radar points or objects according to current UI filter limits.

### [graph_draw.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/processing/visualization/graph_draw.py)
Matplotlib-based Cartesian top-down radar display canvas.
- [Graph_radar](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/processing/visualization/graph_draw.py#L25): Manages the drawing canvas, dynamic grid scaling, and interactive point selection/inspection.

### [radar_camera_transformer.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/processing/visualization/radar_camera_transformer.py)
Projects 3D radar coordinates into 2D camera pixel coordinates using camera intrinsics and extrinsic calibration.
- [RadarCameraTransformer](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/processing/visualization/radar_camera_transformer.py#L18): Validates calibration matrices and projects radar detections into pixel bounding boxes or circles.

### [transposition.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/processing/visualization/transposition.py)
Coordinates live transposition overlays and drops stale frames before GUI rendering.
- [RadarTranspositionOverlay](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/processing/visualization/transposition.py#L20): Combines video frames with radar projections, managing timestamp correlation and visual rendering.
