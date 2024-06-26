import customtkinter as ctk

GREEN_COLOR = "#00FF00"
GREY_COLOR = "#808080"

def create_checkboxes(root, config_manager, update_callback):
    checkboxes = {}
    checkbox_configs = [
        ("auto_aim", "Auto aim", 10, 10),
        ("trigger_bot", "Trigger bot", 10, 40),
        ("toggle", "Auto aim toggle", 110, 10),
        ("recoil", "Recoil control", 110, 40),
        ("aim_shake", "Aim shake", 10, 70),
        ("overlay", "Overlay", 110, 70),
        ("preview", "Preview", 320, 10),
        ("mask_left", "Mask left", 240, 290),
        ("mask_right", "Mask right", 380, 290),
        ("fov", "Enable FOV", 10, 510),
        ("show_fov", "Show FOV", 320, 280),  # New checkbox for showing FOV
    ]

    for key, text, x, y in checkbox_configs:
        var = ctk.StringVar(value=config_manager.get_setting(key))
        checkbox = ctk.CTkCheckBox(
            root,
            text=text,
            variable=var,
            onvalue="on",
            offvalue="off",
            command=lambda k=key, v=var: update_callback(k, v.get()),
            fg_color=GREEN_COLOR,
            hover_color="#0F0"
        )
        checkbox.place(x=x, y=y)
        checkboxes[key] = checkbox

    return checkboxes

def create_sliders(root, config_manager, update_callback):
    sliders = {}
    slider_configs = [
        ("sensitivity", "Sensitivity", 0, 100, 10, 100, 125),
        ("headshot", "Headshot offset", 0, 100, 10, 150, 175),
        ("trigger_bot_distance", "Trigger bot distance", 0, 100, 10, 200, 225),
        ("confidence", "Confidence", 0, 100, 10, 250, 275),
        ("recoil_strength", "Recoil control strength", 0, 100, 10, 300, 325),
        ("aim_shake_strength", "Aim shake strength", 0, 100, 10, 350, 375),
        ("max_move", "Max move speed", 0, 100, 10, 400, 425),
        ("mask_width", "Mask width", 0, 640, 240, 320, 345),
        ("mask_height", "Mask height", 0, 640, 240, 370, 395),
        ("fov_size", "FOV Size", 0, 200, 10, 540, 565),
    ]

    for key, text, from_, to, x, label_y, slider_y in slider_configs:
        label = ctk.CTkLabel(root, text=f"{text}: {config_manager.get_setting(key)}", text_color=GREY_COLOR)
        label.place(x=x, y=label_y)
        
        slider = ctk.CTkSlider(
            root,
            from_=from_,
            to=to,
            command=lambda value, k=key, l=label, t=text: slider_event(value, k, l, t, update_callback),
            fg_color=GREEN_COLOR
        )
        slider.set(config_manager.get_setting(key))
        slider.place(x=x, y=slider_y)
        sliders[key] = slider

    return sliders

def slider_event(value, key, label, text, update_callback):
    label.configure(text=f"{text}: {round(float(value))}")
    update_callback(key, float(value))

def create_comboboxes(root, config_manager, update_callback):
    comboboxes = {}
    combobox_configs = [
        ("fps", "FPS:", ["30", "60", "90", "120", "144", "165", "180"], 10, 720, 750),
        ("yolo_version", "Yolo version:", ["v8", "v5"], 10, 600, 630),
        ("yolo_model", "Yolo model:", ["Default"], 10, 660, 690),
        ("yolo_mode", "Inference mode:", ["pytorch", "onnx", "tensorrt"], 160, 600, 630),
        ("yolo_device", "Device:", ["cpu", "amd", "nvidia"], 160, 660, 690),
        ("yolo_model_size", "Model size:", ["160x160", "320x320", "480x480", "640x640"], 310, 600, 630),
        ("mouse_input", "Mouse input method:", ["default", "arduino"], 310, 420, 450),
        ("arduino", "Arduino port:", ["COM1"], 310, 480, 510),
    ]

    for key, text, values, x, label_y, combo_y in combobox_configs:
        label = ctk.CTkLabel(root, text=text)
        label.place(x=x, y=label_y)
        
        combobox = ctk.CTkComboBox(
            root,
            values=values,
            command=lambda choice, k=key: update_callback(k, choice),
            state="readonly"
        )
        combobox.set(str(config_manager.get_setting(key)))
        combobox.place(x=x, y=combo_y)
        comboboxes[key] = combobox

    return comboboxes

def create_buttons(root, main_window):
    buttons = {}
    
    reload_button = ctk.CTkButton(root, text="Reload model", command=main_window.reload_model)
    reload_button.place(x=310, y=690)
    buttons['reload'] = reload_button
    
    keybindings_button = ctk.CTkButton(root, text="Configure keybindings", command=main_window.show_keybindings)
    keybindings_button.place(x=10, y=575)
    buttons['keybindings'] = keybindings_button
    
    return buttons

def create_labels(root):
    labels = {}
    
    fps_label = ctk.CTkLabel(root, text="Fps:", text_color=GREY_COLOR)
    fps_label.place(x=240, y=10)
    labels['fps'] = fps_label
    
    activation_key_label = ctk.CTkLabel(root, text="Activation key: None")
    activation_key_label.place(x=10, y=455)
    labels['activation_key'] = activation_key_label
    
    quit_key_label = ctk.CTkLabel(root, text="Quit key: None", text_color=GREY_COLOR)
    quit_key_label.place(x=10, y=485)
    labels['quit_key'] = quit_key_label
    
    return labels
