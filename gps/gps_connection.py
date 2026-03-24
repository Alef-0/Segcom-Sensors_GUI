import requests
from requests.auth import HTTPDigestAuth
import urllib3
from multiprocessing.connection import Connection
from multiprocess.queues import Queue
from multiprocessing import Pipe
import time
import webbrowser

import zenoh


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
DVR_IP = "192.168.1.108"
USERNAME = "admin"
PASSWORD = "l1v3user5"
URL = f"http://{DVR_IP}/cgi-bin/positionManager.cgi?action=getStatus"

def get_gps():
    try:
        response = requests.get(URL, auth=HTTPDigestAuth(USERNAME, PASSWORD), verify=False, timeout=1)
        response.raise_for_status()
        return response.text
    except requests.exceptions.Timeout:
        return "Error: The request timed out."
    except requests.exceptions.HTTPError as err:
        return f"Error occurred: {err}"
    except requests.exceptions.RequestException as e:
        # This catches any other requests-related issues
        return f"Error occurred: {e}"
    

def dms_to_dd(degrees, minutes, seconds):
    """Converts Degrees, Minutes, Seconds to Decimal Degrees."""
    return degrees + minutes / 60.0 + seconds / 3600.0

def dd_to_dms(decimal):
    deg = int(decimal)
    min = int((decimal - deg) * 60)
    sec = ((decimal - deg) * 60 - min) * 60
    return deg, min, sec

def transform_into_coordinates(text : str):
    lines = text.split()
    longitude_text = ""
    latitude_text = ""
    lon_value = 0; lat_value = 0

    for line in lines:
        if "Latitude" in line:
            numbers = [float(x) for x in line[17:-1].split(",")]
            lat_value = dms_to_dd(*numbers) - 90.0
            corrected = dd_to_dms(abs(lat_value))
            latitude_text = f"{corrected[0]}° {corrected[1]}' {corrected[2]:.3}'' {'S' if lat_value < 0 else 'W'}"

        if "Longitude" in line:
            numbers = [float(x) for x in line[18:-1].split(",")]
            lon_value = dms_to_dd(*numbers) - 180.0
            corrected = dd_to_dms(abs(lon_value))
            longitude_text = f"{corrected[0]}° {corrected[1]}' {corrected[2]:.3}'' {'W' if lon_value < 0 else 'E'}"
        
            
    
    return f"{latitude_text}, {longitude_text}", f"https://www.google.com/maps/search/?api=1&query={lat_value},{lon_value}"
