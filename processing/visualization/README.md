# Visualization

- `filter_schema.py` defines the shared radar filter names and values.
- `graph_filter.py` applies state, quality, RCS, and coordinate filters.
- `graph_draw.py` draws the top-down radar view and point inspection behavior.
- `radar_camera_transformer.py` validates calibration coefficients and projects radar points into the camera image.
- `transposition.py` creates radar overlay payloads and drops stale frames before drawing.
