from menu_configurations import Configurations
from connection.connection_communication import Can_Connection, can_data
import FreeSimpleGUI as sg
from time import sleep
import re
import signal

from connection.connection_packages import read_201_radar_state as r201
from connection.connection_packages import read_701_cluster_list as r701
from connection.connection_packages import read_702_quality_info as r702
from connection.connection_packages import create_200_radar_configuration as c200
from connection.connection_packages import Clusters_messages

from graph.graph_filter import Filter_graph
from graph.graph_draw import Graph_radar

from camera.camera_webcam import camera_start
from camera.camera_gstreamer import gstreamer_main

from multiprocessing import Process, Queue, Pipe
from multiprocessing.connection import Connection

from connection.connection_main import create_connection_communication
from gps.gps_connection import main as gps_main
loop =  True

def check_popup():
    layout = [
        [sg.Text("Digite [Alohomora] para confirmar salvar permanentemente nos radares!", justification="center")],
        [sg.Input("", key="passwd", expand_x=True, justification="center")],
        [sg.Push(),sg.Ok(), sg.Cancel(), sg.Push()]
    ]
    window = sg.Window("PASSWORD", layout)
    result = False
    while True:
        events, values = window.read()
        match events:
            case sg.WIN_CLOSED: return False
            case "Ok": result = (values['passwd'] == "Alohomora"); break
            case "Cancel": break
    
    window.close()
    return result

def signal_handler(sig, frame):
    print("Finalizing menu by interruption")
    global loop
    loop = False

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    font = ("Helvetica", 12) 
    sg.set_options(font=font)

    all_queue = Queue(5)
    receive_radar, send_radar = Pipe()
    receive_cam, send_cam = Pipe()
    receive_gps, send_gps = Pipe()

    config = Configurations()
    event, values = config.read()
    
    conn_process = Process(target=create_connection_communication, args=(values, receive_radar, all_queue), daemon=True)
    conn_process.start()

    camera_process = Process(target=gstreamer_main, args=(receive_cam, all_queue), daemon=True)
    camera_process.start()

    gps_process = Process(target=gps_main, args = (receive_gps, all_queue),daemon=True)
    gps_process.start()
    

    while loop:
        sleep(0.01)
        event, values = config.read()
        if event == sg.WINDOW_CLOSED: send_cam.send(("STOP", None)); break
        
        # Parte do Menu
        match event:
            case "Send":
                if config.connected_radar: send_radar.send((event, values))
                config.window["save_nvm"].update(button_color=("black", "white"))
            case "save_nvm": 
                if config.connected:
                    result = check_popup()
                    if result:  config.window["save_nvm"].update(button_color=("white", "green"))
                    else:       config.window["save_nvm"].update(button_color=("white", "red"))
                    send_radar.send((event, values))
            case s if re.match(r"^filter", s): send_radar.send((event, values))
            case s if re.match(r"^choose_", s):
                choice = int(event[-1])
                send_radar.send( ("choose", choice))
                send_cam.send(  ("choose", choice))
            case s if re.match(r"^conn_", s):
                if event.endswith("radar"): send_radar.send((event, None))
                elif event.endswith("cam"): send_cam.send((event, None))
                elif event.endswith("gps"): send_gps.send((event, None))
            case "gps_maps":
                send_gps.send((event, None))
            case sg.TIMEOUT_EVENT: pass
            case _: print(event)        
        if event != sg.TIMEOUT_EVENT: print(event)

        # See events
        if not all_queue.empty():
            message, values = all_queue.get()
            # print("VALUES ON POOL:", message, values)
            match message:
                case "message_201": config.change_radar(values)
                case "change_radar": config.change_connection_radar(values)
                case "change_cam": config.change_connection_cam(values)
                case "gps_text": config.window[message].update(values)
                case "conn_gps": config.change_connection_gps(values)
                case _: print("UNKNOWN MESSAGE", message, values)