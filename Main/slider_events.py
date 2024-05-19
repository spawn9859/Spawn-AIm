from main import (
    label_sensitivity,
    label_headshot,
    label_trigger_bot,
    label_confidence,
    label_recoil_strength,
    label_aim_shake_strength,
    label_max_move,
    label_mask_width,
    label_mask_height,
    settings
)

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
