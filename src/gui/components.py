import customtkinter as ctk

def create_checkboxes(root, config_manager, update_callback):
    checkboxes = {}
    checkbox_configs = [
        ("auto_aim", "Auto aim"),
        ("auto_aim_toggle", "Auto aim toggle"),
        ("preview", "Preview"),
        ("trigger_bot", "Trigger bot"),
        ("recoil_control", "Recoil control"),
        ("aim_shake", "Aim shake"),
        ("overlay", "Overlay"),
        ("mask_left", "Mask left"),
        ("mask_right", "Mask right"),
    ]

    for i, (key, text) in enumerate(checkbox_configs):
        var = ctk.BooleanVar(value=config_manager.get_setting(key))
        checkbox = ctk.CTkCheckBox(
            root,
            text=text,
            variable=var,
            command=lambda k=key, v=var: update_callback(k, v.get()),
        )
        checkbox.grid(row=i//3, column=i%3, padx=5, pady=2, sticky="w")
        checkboxes[key] = checkbox

    return checkboxes

# In components.py

def create_sliders(root, config_manager, update_callback, slider_configs=None):
    sliders = {}
    if slider_configs is None:
        slider_configs = [
            ("sensitivity", "Sensitivity", 0, 100),
            ("smoothing_factor", "Smoothing factor", 0, 100),
            ("confidence", "Confidence threshold", 0, 100),
            ("headshot", "Headshot offset", 0, 100),
            ("trigger_bot_distance", "Trigger bot distance", 0, 100),
            ("recoil_strength", "Recoil control strength", 0, 100),
            ("aim_shake_strength", "Aim shake strength", 0, 100),
            ("max_move", "Max move speed", 0, 100),
            ("mask_width", "Mask width", 0, 640),
            ("mask_height", "Mask height", 0, 640),
            ("fov_size", "FOV Size", 0, 200),
        ]

    for i, (key, text, from_, to) in enumerate(slider_configs):
        frame = ctk.CTkFrame(root)
        frame.pack(fill="x", padx=10, pady=5)
        
        current_value = config_manager.get_setting(key)
        if not isinstance(current_value, (int, float)):
            current_value = (from_ + to) // 2
            config_manager.update_setting(key, current_value)
        
        label = ctk.CTkLabel(frame, text=f"{text}: {current_value:.1f}")
        label.pack(side="left")
        
        slider = ctk.CTkSlider(
            frame,
            from_=from_,
            to=to,
            command=lambda value, k=key, l=label, t=text: slider_event(value, k, l, t, update_callback),
        )
        slider.set(current_value)
        slider.pack(side="right", expand=True, fill="x", padx=10)
        sliders[key] = slider

    return sliders

def slider_event(value, key, label, text, update_callback):
    rounded_value = round(float(value), 1)
    label.configure(text=f"{text}: {rounded_value:.1f}")
    update_callback(key, rounded_value)

def create_comboboxes(root, config_manager, update_callback):
    comboboxes = {}
    combobox_configs = [
        ("yolo_version", "YOLO version:", ["v5", "v8"]),
        ("yolo_model", "YOLO model:", ["v5_Fortnite_taipei"]),
        ("yolo_mode", "Inference mode:", ["tensorrt", "onnx", "pytorch"]),
        ("yolo_device", "Device:", ["nvidia", "cpu", "amd"]),
        ("mouse_input", "Mouse input method:", ["default", "arduino"]),
        ("arduino", "Arduino port:", ["COM1", "COM2", "COM3", "COM4"]),
    ]

    for i, (key, text, values) in enumerate(combobox_configs):
        frame = ctk.CTkFrame(root)
        frame.pack(fill="x", padx=10, pady=5)
        
        label = ctk.CTkLabel(frame, text=text)
        label.pack(side="left")
        
        combobox = ctk.CTkComboBox(
            frame,
            values=values,
            command=lambda choice, k=key: update_callback(k, choice),
            state="readonly"
        )
        combobox.set(str(config_manager.get_setting(key)))
        combobox.pack(side="right")
        comboboxes[key] = combobox

    return comboboxes

def create_buttons(root, main_window):
    buttons = {}
    
    reload_button = ctk.CTkButton(root, text="Reload model", command=main_window.reload_model)
    reload_button.pack(pady=10)
    buttons['reload'] = reload_button
    
    keybindings_button = ctk.CTkButton(root, text="Configure keybindings", command=main_window.show_keybindings)
    keybindings_button.pack(pady=10)
    buttons['keybindings'] = keybindings_button
    
    return buttons

def create_labels(root):
    labels = {}
    
    fps_label = ctk.CTkLabel(root, text="FPS: 0")
    labels['fps'] = fps_label
    
    activation_key_label = ctk.CTkLabel(root, text="Activation key: None")
    labels['activation_key'] = activation_key_label
    
    quit_key_label = ctk.CTkLabel(root, text="Quit key: None")
    labels['quit_key'] = quit_key_label
    
    return labels