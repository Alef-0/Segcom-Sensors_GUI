from menu_configurations import Configurations
import FreeSimpleGUI as sg
import signal
from camera.camera_gstreamer import gstreamer_main
from multiprocessing import Process, Queue, Pipe

from connection.connection_main import create_connection_communication
from gps.gps_executor import gps_main
import zenoh

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
    pub_sub = zenoh.open(zenoh.Config())

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

    gps_process = Process(target=gps_main, daemon=True)
    gps_process.start()

    while loop:
        # print(gps_process.is_alive())
        event, values = config.read()
        if event == sg.WINDOW_CLOSED: send_cam.send(("STOP", None)); break
        
        # Parte do Menu
        if event == "Send":
            if config.connected_radar: send_radar.send((event, values))
            config.window["save_nvm"].update(button_color=("black", "white"))
        elif event == "save_nvm": 
            if config.connected:
                result = check_popup()
                if result:  config.window["save_nvm"].update(button_color=("white", "green"))
                else:       config.window["save_nvm"].update(button_color=("white", "red"))
                send_radar.send((event, values))
        elif event.startswith("filter"): send_radar.send((event, values))
        elif event.startswith("choose_"):
            choice = int(event[-1])
            send_radar.send( ("choose", choice))
            send_cam.send(  ("choose", choice))
        elif event.startswith("conn_"):
            if event.endswith("radar"): send_radar.send((event, None))
            elif event.endswith("cam"): send_cam.send((event, None))
            elif event.endswith("gps"): 
                print("Sent in here")
                pub_sub.put("GUI/GPS", event)
                
        elif event == "gps_maps": pub_sub.put("GUI/GPS", event)
        
        if event == sg.TIMEOUT_EVENT: pass        
        else: print(event)

        # See events        
        for message in pub_sub.get("GUI/MAIN/**"):
            if not message.ok: print("Something wrong with message"); continue;
            topic, values = str(message.ok.key_expr), message.ok.payload.to_string()
            # print("VALUES ON POOL:", message, values)
            if topic == "message_201": config.change_radar(values)
            elif topic == "change_radar": config.change_connection_radar(values)
            elif topic == "change_cam": config.change_connection_cam(values)
            elif topic == "gps_text": config.window[message].update(values)
            elif topic == "conn_gps": config.change_connection_gps(values)
            else: print("UNKNOWN MESSAGE", message, values)