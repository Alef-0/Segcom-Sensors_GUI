import signal
import time
import webbrowser

import requests
import urllib3
from requests.auth import HTTPDigestAuth

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
DVR_IP = "192.168.1.108"
USERNAME = "admin"
PASSWORD = "l1v3user5"
URL = f"http://{DVR_IP}/cgi-bin/positionManager.cgi?action=getStatus"


def get_gps(session=None):
    """Fetch position status text from DVR."""
    client = session or requests
    kwargs = {"verify": False, "timeout": 3}
    if session is None:
        kwargs["auth"] = HTTPDigestAuth(USERNAME, PASSWORD)
    response = client.get(URL, **kwargs)
    response.raise_for_status()
    return response.text


def dms_to_dd(degrees, minutes, seconds):
    """Convert degrees, minutes, seconds to decimal degrees."""
    return degrees + minutes / 60.0 + seconds / 3600.0


def dd_to_dms(decimal):
    """Convert decimal degrees to degrees, minutes, seconds."""
    deg = int(decimal)
    mins = int((decimal - deg) * 60)
    secs = ((decimal - deg) * 60 - mins) * 60
    return deg, mins, secs


def parse_coordinate(raw_line: str, offset: float, neg_hemi: str, pos_hemi: str):
    """Parse coordinate string and return decimal degrees and formatted DMS string."""
    raw_val = raw_line.split("=", 1)[1].strip("()") if "=" in raw_line else raw_line.strip("()")
    parts = [float(x) for x in raw_val.split(",")]
    dd_val = dms_to_dd(*parts[:3]) - offset
    deg, mins, secs = dd_to_dms(abs(dd_val))
    hemi = neg_hemi if dd_val < 0 else pos_hemi
    return dd_val, f"{deg}° {mins}' {secs:.3f}'' {hemi}"


def transform_into_coordinates(text):
    """Convert DVR position status text to display coordinates and maps link."""
    lat_text = lon_text = ""
    lat_val = lon_val = 0.0
    for line in text.split():
        if "Latitude" in line:
            lat_val, lat_text = parse_coordinate(line, 90.0, "S", "N")
        elif "Longitude" in line:
            lon_val, lon_text = parse_coordinate(line, 180.0, "W", "E")
    return f"{lat_text}, {lon_text}", f"https://www.google.com/maps/search/?api=1&query={lat_val},{lon_val}"


def main(connection, pool, shutdown_event):
    """Poll GPS coordinates from DVR and dispatch events to GUI."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: shutdown_event.set())
    session = requests.Session()
    session.auth = HTTPDigestAuth(USERNAME, PASSWORD)
    reading = False
    maps_link = "https://www.google.com/maps/search/?api=1&query=0,0"
    next_read = 0.0

    try:
        while not shutdown_event.is_set():
            while connection.poll():
                try:
                    event, _ = connection.recv()
                except (EOFError, OSError):
                    shutdown_event.set()
                    break
                if event == "STOP":
                    shutdown_event.set()
                elif event == "conn_gps":
                    reading = not reading
                    pool.put((event, reading))
                elif event == "gps_maps":
                    webbrowser.open(maps_link)
            now = time.monotonic()
            if reading and now >= next_read:
                try:
                    text, maps_link = transform_into_coordinates(get_gps(session))
                    pool.put(("gps_text", text))
                except requests.RequestException as exc:
                    print(f"GPS request failed: {exc}")
                next_read = now + 1.0
            shutdown_event.wait(0.05)
    finally:
        session.close()
