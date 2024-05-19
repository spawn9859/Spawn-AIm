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
