import requests
from requests.auth import HTTPDigestAuth
import urllib3
from multiprocessing.connection import Connection
from multiprocess.queues import Queue
from multiprocessing import Pipe
import time
import webbrowser
from time import sleep


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
DVR_IP = "192.168.1.108"
USERNAME = "admin"
PASSWORD = "l1v3user5"
URL = f"http://{DVR_IP}/cgi-bin/positionManager.cgi?action=getStatus"

def get_gps():
    response = requests.get(URL, auth=HTTPDigestAuth(USERNAME, PASSWORD), verify=False)
    return response.text


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

def main(connection : Connection, pool : Queue):
    read = False
    maps_link = f"https://www.google.com/maps/search/?api=1&query={0},{0}"

    while True:
        sleep(0.01)
        if read: 
            all_values = get_gps()
            text, maps_link = transform_into_coordinates(all_values)
            pool.put(("gps_text", text))
            time.sleep(1)
        
        # Tratar dos callbacks
        if connection.poll():
            event, value = connection.recv()
            match event:
                case "conn_gps": 
                    read = not read;
                    pool.put((event, read))
                case "gps_maps":
                    webbrowser.open(maps_link)
                


if __name__ == "__main__": 
    send, receive = Pipe()
    q = Queue(5)
    main()
