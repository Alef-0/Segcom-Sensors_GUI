import FreeSimpleGUI as sg

class Configurations:
    POWER = ["STANDARD", "-3dB Tx gain", "-6dB Tx gain", "-9dB Tx gain"]
    OUTPUT = ["NONE", "OBJECT", "CLUSTERS"]
    RCS = ["STANDARD", "HIGH SENSITIVITY"]
    FONT = ("Helvetica", 12) 

    def __init__(self):
        # Tem que colocar as configurações primeiro
        sg.set_options(font=self.FONT)
        sg.theme("SystemDefaultForReal")
        
        # Criar divisões especiais
        self.real_time_status()
        self.radar_configurations()
        self.filter_configurations()

        self.layout = [ [self.FRAME], [sg.Push(), self.options, sg.Push()], [self.filter]]
        self.window = sg.Window("Configurations Menu", self.layout, finalize=True)
        
        #Configurações adicionais
        self.centralize_combos()
        self.connected_radar = False
        self.connected_cam = False

    def real_time_status(self):
        # O GPS foi colocado aqui porque era Melhor
        GPS = sg.Frame("", 
                       [[sg.Text("0° 0' 0\" N, 0° 0' 0\" E", expand_x=True, key="gps_text"), 
                         sg.Button("OPEN GPS", key="conn_gps", button_color=("white", "green")), 
                         sg.Button("MAPS", key="gps_maps")]], 
                       title_location=sg.TITLE_LOCATION_RIGHT)
        warning_text = [
            sg.Push(), sg.Text("GRAPHS NEEDS CLUSTER + QUALITY"), sg.Push(), 
            GPS, sg.Push()
        ]
        separation = []
        names = ["", "LEFT", "MIDDLE", "RIGHT"]
        letter = ["NULL", "A", "B", "C"]
        for i in range(1,4):
            curr_tab = sg.Column([
                [sg.Text(f"{names[i]} {letter[i]}", justification="center", expand_x=True)],
                [sg.Text("Distance", expand_x=True, justification="left"),    sg.Text("XXX", key=f"DISTANCE_{i}", expand_x=True, justification="right")],
                [sg.Text("Radar", expand_x=True, justification="left"),       sg.Text("XXX", key=f"RPW_{i}", expand_x=True, justification="right")],
                [sg.Text("Output", expand_x=True, justification="left"),      sg.Text("XXX", key=f"OUT_{i}", expand_x=True, justification="right")],
                [sg.Text("RCS", expand_x=True, justification="left"),         sg.Text("XXX", key=f"RCS_{i}", expand_x=True, justification="right")],
                [sg.Text("Extra Info", expand_x=True, justification="left"),    sg.Text("XXX", key=f"EXT_{i}", expand_x=True, justification="right")],
                [sg.Radio(f"Visualizar Grupo {letter[i]}", "visu_radar", key=f"choose_{i}", default= True if i == 2 else False, enable_events=True)]
            ]
            )
            separation.extend([sg.Push(), curr_tab, sg.Push(), sg.VSep()])
        separation.pop()

        
        self.FRAME = [sg.Frame("Radar Status", [separation, warning_text], expand_x=True, title_location=sg.TITLE_LOCATION_TOP)]

    def radar_configurations(self):
        self.choices = sg.Frame("Radar", [[
            sg.Radio("1", "choose", key="send_1"), sg.Radio("2", "choose", key="send_2"), 
            sg.Radio("3", "choose", key="send_3"),  sg.Radio("all", "choose", key="send_all", default=True)
        ]], 
        title_location=sg.TITLE_LOCATION_LEFT)

        self.column1 = sg.Column([
            [sg.Checkbox("Radar Power", expand_x=True, key="CHECK_RPW",default=True), sg.Push(), sg.Combo(self.POWER, self.POWER[3], font=self.FONT, size=15, key="RPW", readonly=True), sg.Push()],
            [sg.Checkbox("RCS Treshold", expand_x=True, key="CHECK_RCS",default=True),  sg.Combo(self.RCS, self.RCS[1], font=self.FONT, size=16, key="RCS", readonly=True), sg.Push()],
        ])
        self.column2 = sg.Column([
            [sg.Checkbox("Output Power", expand_x=True, key="CHECK_OUT",default=True),sg.Push(), sg.Combo(self.OUTPUT, self.OUTPUT[2], font=self.FONT, size=15, key="OUT", readonly=True), sg.Push()],
            [sg.Push(), sg.Checkbox("Send Quality", key="CHECK_QUALITY",default=True), sg.Push()]
        ])

        self.options = sg.Frame("Radar Configurations", [
            # Distancia baseada no que o radar no modo consegue alcançar
            [sg.Checkbox("Max Distance", expand_x=True, key="CHECK_DISTANCE",default=True),  sg.Push(), sg.Slider((196, 260), 100, orientation="h", resolution=1, key="DISTANCE", size=(40, 20)), sg.Push()],
            [self.column1, sg.VerticalSeparator(), self.column2],
            [sg.Button("Send"), self.choices,  sg.VerticalSeparator(), sg.Button("OPEN RADAR", key="conn_radar", button_color=("white", "green")), sg.Button("OPEN CAM", key="conn_cam", button_color=("white", "green"), size=(10)),],
            [sg.Button("SAVE in Non Volatile Memory", key="save_nvm", expand_x=True, button_color=("black", "white"))]
        ], title_location=sg.TITLE_LOCATION_TOP)

    def filter_configurations(self):
        self.dynprop = sg.Column([
            [sg.Push(), 
                sg.Checkbox("Moving", default=True, enable_events=True, key="filter_dyn_move"),                     sg.Button("", button_color="#FF0000", disabled=True), 
                sg.Checkbox("Stationary", default=True, enable_events=True, key="filter_dyn_stationary"),           sg.Button("", button_color="#FF7B00", disabled=True), 
                sg.Checkbox("Oncoming", default=True, enable_events=True, key="filter_dyn_oncoming"),               sg.Button("", button_color="#FFE600", disabled=True), 
                sg.Checkbox("Stationary Candidate",default=False, enable_events=True, key="filter_dyn_candidate"),   sg.Button("", button_color="#00FF00", disabled=True), 
            sg.Push()],
            [sg.Push(), 
                sg.Checkbox("Unknown",default=True, enable_events=True, key="filter_dyn_unknown"),                  sg.Button("", button_color="#0000FF", disabled=True), 
                sg.Checkbox("Crossing Stationary", default=True, enable_events=True, key="filter_dyn_crossstat"),   sg.Button("", button_color="#00FFFF", disabled=True), 
                sg.Checkbox("Crossing Moving", default=True, enable_events=True, key="filter_dyn_crossmove"),       sg.Button("", button_color="#8400FF", disabled=True), 
                sg.Checkbox("Stopped", default=True, enable_events=True, key="filter_dyn_stopped"),                 sg.Button("", button_color="#000000", disabled=True), 
            sg.Push()],
        ], justification="center")
        
        self.slider_pdh = sg.Column([
            [sg.Push(), sg.Text("PDH - False Alarm Probability", justification="center"), sg.Push()],
            [sg.Slider((1, 7),disable_number_display=True, default_value=3, orientation='h', tick_interval=1, expand_x=True, enable_events=True, key="filter_phd")],
            [sg.Text("[0, 1 = 25, 2 = 50, 3 = 75, 4 = 90, 5 = 99...]", expand_x=True, justification="center")]
        ], justification="center")

        self.ambig_state = sg.Column([
            [sg.Push(), sg.Text("Ambiguitity State"), sg.Push()],
            [
                sg.Push(), sg.Checkbox("Ambiguous", enable_events=True, key="filter_ambg_ambig") , 
                sg.Checkbox("Staggered Ramp", enable_events=True, key="filter_ambg_staggered"), sg.Push()
            ],
            [
                sg.Push(), sg.Checkbox("Unambiguous", enable_events=True, default=True, key="filter_ambg_unambig"), 
                sg.Checkbox("Stationary Candidates", default=True, enable_events=True, key="filter_ambg_stat") ,sg.Push()
            ]
        ], justification="center")

        self.cluster_state = sg.Column([
            [sg.Text("Cluster State", justification="center", expand_x=True)],
            [sg.Push(), 
                sg.Checkbox("0x0", default=True, enable_events=True,    key="filter_inv_00") , 
                sg.Checkbox("0x1", enable_events=True,                  key="filter_inv_01"), 
                sg.Checkbox("0x2", enable_events=True,                  key="filter_inv_02"), 
                sg.Checkbox("0x3", enable_events=True,                  key="filter_inv_03"), 
                sg.Checkbox("0x4", enable_events=True, default=True,    key="filter_inv_04") , 
                sg.Checkbox("0x5", enable_events=True, disabled=True,   key="filter_inv_05"), 
                sg.Checkbox("0x6", enable_events=True,                  key="filter_inv_06"), 
                sg.Checkbox("0x7", enable_events=True,                  key="filter_inv_07"), 
                sg.Checkbox("0x8", enable_events=True, default=True,    key="filter_inv_08") , 
            sg.Push()],
            [sg.Push(), 
                sg.Checkbox("0x9", enable_events=True, default=True,    key="filter_inv_09"), 
                sg.Checkbox("0xA", enable_events=True, default=True,    key="filter_inv_0A"), 
                sg.Checkbox("0xB", enable_events=True, default=True,    key="filter_inv_0B"), 
                sg.Checkbox("0xC", enable_events=True, default=True,    key="filter_inv_0C") , 
                sg.Checkbox("0xD", enable_events=True, disabled=True,   key="filter_inv_0D"), 
                sg.Checkbox("0xE", enable_events=True,                  key="filter_inv_0E"), 
                sg.Checkbox("0xF", enable_events=True, default=True,    key="filter_inv_0F"),
                sg.Checkbox("0x10", enable_events=True, default=True,   key="filter_inv_10") , 
                sg.Checkbox("0x11", enable_events=True, default=True,   key="filter_inv_11"),  
            sg.Push()],
        ], expand_x=True)

        self.rcs_limit = sg.Column([
            [sg.Text("RCS Filtering\nMinimum Value", justification="center"),
             sg.Slider((-64, 64), default_value=-20, orientation='h', tick_interval=10, expand_x=True, enable_events=True, key="filter_rcs")]
        ], expand_x=True)

        self.filter = sg.Frame("Filters for points", [
                [self.dynprop], 
                [sg.HorizontalSeparator()], 
                [self.rcs_limit],
                [sg.HorizontalSeparator()], 
                [self.slider_pdh, sg.VerticalSeparator(), self.ambig_state],
                [sg.HorizontalSeparator()],
                [self.cluster_state]
            ], title_location=sg.TITLE_LOCATION_TOP, expand_x=True)

    def centralize_combos(self):
        """Just Centralizing things messing with tkinter under the hood"""
        self.window["RPW"].Widget.configure(justify="center")
        self.window["OUT"].Widget.configure(justify="center")
        self.window["RCS"].Widget.configure(justify="center")

    def read(self):
        return self.window.read(1) # milliseconds
    
    def change_connection_radar(self, connection):
        self.connected_radar =  connection
        if self.connected_radar:
                self.window['conn_radar'].update("CLOSE RADAR", button_color=("white", "red"))
        else:   self.window['conn_radar'].update("OPEN RADAR", button_color=("white", "green"))

        # Clear radar
        def create_dict(channel):
            dummy = {}
            dummy[f'DISTANCE_{channel}'] = "XXX"
            dummy[f'RPW_{channel}'] = "XXX"
            dummy[f'OUT_{channel}'] = "XXX"
            dummy[f'RCS_{channel}'] = "XXX"
            dummy[f'EXT_{channel}'] = "XXX"
            return dummy
        
        self.change_radar(create_dict(1))
        self.change_radar(create_dict(2))
        self.change_radar(create_dict(3))
    
    def change_connection_cam(self, connection):
        self.connected_cam =  connection
        if self.connected_cam:
                self.window['conn_cam'].update("CLOSE CAM", button_color=("white", "red"))
        else:   self.window['conn_cam'].update("OPEN CAM", button_color=("white", "green"))
    
    def change_connection_gps(self, connection):
        if connection:  self.window['conn_gps'].update("CLOSE GPS", button_color=("white", "red"))
        else:           self.window['conn_gps'].update("OPEN GPS", button_color=("white", "green"))

    def change_radar(self, dicio : dict):
        for key, item in dicio.items():
            self.window[key].update(item)
            

    def __del__(self): 
        self.window.close()
        print("DELETED WINDOW CONFIGURATION")