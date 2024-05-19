def checkbox_auto_aim_event():
    settings['auto_aim'] = var_auto_aim.get()

def checkbox_trigger_bot_event():
    settings['trigger_bot'] = var_trigger_bot.get()

def checkbox_toggle_event():
    settings['toggle'] = var_toggle.get()

def checkbox_recoil_event():
    settings['recoil'] = var_recoil.get()

def checkbox_aim_shake_event():
    settings['aim_shake'] = var_aim_shake.get()

def checkbox_overlay_event():
    toggle_overlay()
    settings['overlay'] = var_overlay.get()

def checkbox_preview_event():
    settings['preview'] = var_preview.get()
    if settings['preview'] == "off":
        image_label_preview.configure(image=image_preview)

def checkbox_mask_left_event():
    settings['mask_left'] = var_mask_left.get()

def checkbox_mask_right_event():
    settings['mask_right'] = var_mask_right.get()

def slider_sensitivity_event(value):
    label_sensitivity.configure(text=f"Sensitivity: {round(value)}%")
    settings['sensitivity'] = value

def slider_headshot_event(value):
    label_headshot.configure(text=f"Headshot offset: {round(value)}%")
    settings['headshot'] = value

def slider_trigger_bot_event(value):
    label_trigger_bot.configure(text=f"Trigger bot distance: {round(value)} px")
    settings['trigger_bot_distance'] = value

def slider_confidence_event(value):
    label_confidence.configure(text=f"Confidence: {round(value)}%")
    settings['confidence'] = value

def slider_recoil_strength_event(value):
    label_recoil_strength.configure(text=f"Recoil control strength: {round(value)}%")
    settings['recoil_strength'] = value

def slider_aim_shake_strength_event(value):
    label_aim_shake_strength.configure(text=f"Aim shake strength: {round(value)}%")
    settings['aim_shake_strength'] = value

def slider_max_move_event(value):
    label_max_move.configure(text=f"Max move speed: {round(value)} px")
    settings['max_move'] = value
def combobox_fps_callback(choice):
    settings['max_fps'] = int(choice)
    with open(f"{script_directory}/configuration/config.json", 'w') as json_file:
        json.dump(settings, json_file, indent=4)
    button_reload_event()

def slider_mask_width_event(value):
    label_mask_width.configure(text=f"Mask width: {round(value)} px")
    settings['mask_width'] = value

def slider_mask_height_event(value):
    label_mask_height.configure(text=f"Mask height: {round(value)} px")
    settings['mask_height'] = value

def combobox_yolo_version_callback(choice):
    models = [yolo_model for yolo_model in launcher_models if yolo_model.startswith(combobox_yolo_version.get())]
    combobox_yolo_model.configure(values=models)
    combobox_yolo_model.set(models[0])

def combobox_yolo_model_callback(choice):
    return

def combobox_yolo_model_size_callback(choice):
    return

def combobox_yolo_mode_callback(choice):
    return

def combobox_yolo_device_callback(choice):
    return

def combobox_mouse_input_callback(choice):
    settings['mouse_input'] = choice

def combobox_arduino_callback(choice):
    global arduino
    settings['arduino'] = choice
    arduino = serial.Serial(choice, 9600, timeout=1)

def combobox_mouse_activation_bind_callback(choice):
    label_activation_bind.configure(text=f"Activation key: {choice}")
    settings['activation_key_string'] = choice
    activation_key = get_keycode(settings['activation_key_string'])
    settings['activation_key'] = activation_key
    label_activation_key.configure(text=f"Activation key: {choice}")

def combobox_mouse_quit_bind_callback(choice):
    label_quit_bind.configure(text=f"Quit key: {choice}")
    settings['quit_key_string'] = choice
    quit_key = get_keycode(settings['quit_key_string'])
    settings['quit_key'] = quit_key
    label_quit_key.configure(text=f"Quit key: {choice}")

def button_activation_bind_event():
    activation_key = keyboard.read_event(suppress=True).name
    label_activation_bind.configure(text=f"Activation key: {activation_key}")
    settings['activation_key_string'] = activation_key
    activation_key = get_keycode(activation_key)
    settings['activation_key'] = activation_key
    label_activation_key.configure(text=f"Activation key: {settings['activation_key_string']}")

def button_quit_bind_event():
    quit_key = keyboard.read_event(suppress=True).name
    label_quit_bind.configure(text=f"Quit key: {quit_key}")
    settings['quit_key_string'] = quit_key
    quit_key = get_keycode(quit_key)
    settings['quit_key'] = quit_key
    label_quit_key.configure(text=f"Quit key: {settings['quit_key_string']}")

def button_keybindings_event():
    if keybindings.state() == 'withdrawn':
        keybindings.deiconify()
        keybindings.geometry(f"400x160+{root.winfo_x() + 40}+{root.winfo_y() + 100}")
        keybindings.focus()
        keybindings.attributes('-topmost', True)
    else:
        keybindings.withdraw()

def button_reload_event():
    global screen, overlay
    settings['yolo_version'] = combobox_yolo_version.get()
    settings['yolo_model'] = combobox_yolo_model.get()
    settings['yolo_mode'] = combobox_yolo_mode.get()
    settings['yolo_device'] = combobox_yolo_device.get()
    settings['height'], settings['width'] = map(int, combobox_yolo_model_size.get().split('x'))
    slider_mask_width.configure(to=settings['width'])
    slider_mask_height.configure(to=settings['height'])
    combobox_fps.set(str(settings['max_fps']))
    slider_mask_width.set(0)
    slider_mask_height.set(0)
    if settings['mask_width'] <= settings['width']:
        slider_mask_width.set(settings['mask_width'])
    else:
        slider_mask_width.set(settings['width'])
        slider_mask_width_event(settings['width'])
    if settings['mask_height'] <= settings['height']:
        slider_mask_height.set(settings['mask_height'])
    else:
        slider_mask_height.set(settings['height'])
        slider_mask_height_event(settings['height'])
    left = int(win32api.GetSystemMetrics(0) / 2 - settings['width'] / 2)
    top = int(win32api.GetSystemMetrics(1) / 2 - settings['height'] / 2)
    right = int(left + settings['width'])
    bottom = int(top + settings['height'])
    if screen is None:
        screen = bettercam.create(output_color="BGRA", max_buffer_len=512)
    else:
        screen.stop()
    screen.start(region=(left, top, right, bottom), target_fps=int(settings['max_fps']), video_mode=True)
    load_model()
    if overlay is not None:
        toggle_overlay()
        toggle_overlay()
