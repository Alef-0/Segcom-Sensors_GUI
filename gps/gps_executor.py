import zenoh
from time import sleep
from gps.gps_connection import get_gps, transform_into_coordinates, webbrowser

read = False
maps_link = f"https://www.google.com/maps/search/?api=1&query={0},{0}"

def threat_message(sample : zenoh.Sample, session):
    load = sample.payload.to_string()
    
    if load == "conn_gps": 
        global read; read = not read;
        session.put("GUI/MAIN/conn_gps", f"{read}")
    elif load == "gps_maps":
        webbrowser.open(maps_link)

def gps_main():
    print("GPS Initialized")
    global read
    pub_sub = zenoh.open(zenoh.Config())
    sub = pub_sub.declare_subscriber("GUI/GPS", lambda x: threat_message(x, pub_sub))
    
    try:
        while True:
            if read: 
                all_values = get_gps()
                if all_values.startswith("Error"): 
                    read = not read; 
                    pub_sub.put("GUI/MAIN/conn_gps", f"{read}")
                else:
                    global maps_link
                    text, maps_link = transform_into_coordinates(all_values)
                    pub_sub.put("GUI/MAIN/gps_text", text)
            sleep(1)
    except: pass
        
if __name__ == "__main__": gps_main()