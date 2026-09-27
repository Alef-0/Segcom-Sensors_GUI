# GPS Integration Directory Summary

This folder contains the GPS receiver interface module that connects to the DVR position endpoint (`http://192.168.1.108/cgi-bin/positionManager.cgi?action=getStatus`), parses coordinates, and provides Google Maps telemetry links.

## Operational Overview

`gps_connection.py` polls the DVR position endpoint once per second over HTTP digest authentication while GPS monitoring is enabled. It converts the DVR's offset coordinate representation (90 degree latitude offset, 180 degree longitude offset) to signed decimal coordinates, formats degrees/minutes/seconds for the GUI, and maintains a Google Maps URL for the latest position. Network failures are logged and polling automatically retries on the next interval.

## Files

### [gps_connection.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/gps/gps_connection.py)
Background worker process polling DVR GPS status via HTTP digest authentication. Parses raw position strings into signed decimal coordinates, formats human-readable degrees/minutes/seconds strings, and feeds coordinate updates and map links to the GUI.
- [`get_gps()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/gps/gps_connection.py#L16-L24): Fetches raw position status text from the DVR endpoint using digest authentication.
- [`dms_to_dd()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/gps/gps_connection.py#L27-L29): Converts degrees, minutes, and seconds into decimal degrees.
- [`dd_to_dms()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/gps/gps_connection.py#L32-L36): Converts decimal degrees into integer degrees, minutes, and floating-point seconds.
- [`parse_coordinate()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/gps/gps_connection.py#L39-L46): Parses coordinate string, subtracts hemisphere offset, and returns decimal degrees and formatted DMS string.
- [`transform_into_coordinates()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/gps/gps_connection.py#L49-L57): Parses full DVR position text into formatted latitude/longitude and Google Maps search link.
- [`main()`](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/sensors/gps/gps_connection.py#L60-L93): Background process loop polling GPS on timer, reacting to GUI events, and opening browser maps.
