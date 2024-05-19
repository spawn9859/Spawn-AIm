from Main.shared import (
    label_sensitivity,
    label_headshot,
    label_trigger_bot,
    label_confidence,
    label_recoil_strength,
    label_aim_shake_strength,
    label_max_move,
    label_mask_width,
    label_mask_height,
    settings,
    label_activation_bind,
    label_quit_bind,
    label_activation_key,
    label_quit_key,
    image_label_preview
)
from main import toggle_overlay


def checkbox_auto_aim_event(var_auto_aim, settings):
    settings['auto_aim'] = var_auto_aim.get()

def checkbox_trigger_bot_event(var_trigger_bot, settings):
    settings['trigger_bot'] = var_trigger_bot.get()

def checkbox_toggle_event(var_toggle, settings):
    settings['toggle'] = var_toggle.get()

def checkbox_recoil_event(var_recoil, settings):
    settings['recoil'] = var_recoil.get()

def checkbox_aim_shake_event(var_aim_shake, settings):
    settings['aim_shake'] = var_aim_shake.get()

def checkbox_overlay_event(var_overlay, settings, toggle_overlay):
    toggle_overlay()
    settings['overlay'] = var_overlay.get()

def checkbox_preview_event(var_preview, settings, image_label_preview, image_preview):
    settings['preview'] = var_preview.get()
    if settings['preview'] == "off":
        image_label_preview.configure(image=image_preview)

def checkbox_mask_left_event(var_mask_left, settings):
    settings['mask_left'] = var_mask_left.get()

def checkbox_mask_right_event(var_mask_right, settings):
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

def slider_mask_width_event(value):
    label_mask_width.configure(text=f"Mask width: {round(value)} px")
    settings['mask_width'] = value

def slider_mask_height_event(value):
    label_mask_height.configure(text=f"Mask height: {round(value)} px")
    settings['mask_height'] = value
