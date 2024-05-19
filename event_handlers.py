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
from event_handlers import combobox_fps_callback, slider_mask_width_event, slider_mask_height_event, combobox_yolo_version_callback, combobox_yolo_model_callback, combobox_yolo_model_size_callback, combobox_yolo_mode_callback, combobox_yolo_device_callback, combobox_mouse_input_callback, combobox_arduino_callback, combobox_mouse_activation_bind_callback, combobox_mouse_quit_bind_callback, button_activation_bind_event, button_quit_bind_event, button_keybindings_event, button_reload_event
