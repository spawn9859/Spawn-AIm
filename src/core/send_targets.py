import win32api
import win32con
import numpy as np
import time

def send_targets(controller, settings, targets, distances, random_x, random_y, get_left_trigger, get_right_trigger):
    if distances.size == 0:
        return

    # Apply settings
    sensitivity_factor = settings["sensitivity"] / 20 
    max_move = settings["max_move"]
    aim_shake_strength = settings["aim_shake_strength"]
    
    # Find closest target
    min_index = np.argmin(distances)
    target_x, target_y = targets[min_index]

    # Apply aim shake
    target_x += random_x * aim_shake_strength
    target_y += random_y * aim_shake_strength

    # Limit target movement speed
    target_x = max(-max_move, min(target_x, max_move))
    target_y = max(-max_move, min(target_y, max_move))

    # Check for trigger activation (either held or toggled)
    trigger_pressed = get_left_trigger(controller) > 0.5
    auto_aim_active = settings["auto_aim"] == "on" and (
        (trigger_pressed and settings["toggle"] == "off") or settings["toggle"] == "on"
    )

    # Check for recoil control activation using the right trigger
    recoil_control_active = settings["recoil"] == "on" and get_right_trigger(controller) > 0.5

    if auto_aim_active:
        # Apply recoil control if enabled and right trigger is pressed
        recoil = int(settings["recoil_strength"]) if recoil_control_active else 0 

        # Calculate mouse movement with improved smoothing
        mouse_move_x, mouse_move_y = smooth_aim(target_x, target_y, sensitivity_factor, settings)

        # Apply recoil
        mouse_move_y += recoil

        # Move mouse
        _move_mouse(mouse_move_x, mouse_move_y, settings)

        # Implement trigger bot
        if settings["trigger_bot"] == "on":
            trigger_bot(target_x, target_y, settings)

def smooth_aim(target_x, target_y, sensitivity_factor, settings):
    # Implement a simple acceleration curve
    distance = np.sqrt(target_x**2 + target_y**2)
    acceleration = min(distance / 100, 1.5)  # Max acceleration of 1.5x
    
    mouse_move_x = int(target_x * sensitivity_factor * acceleration)
    mouse_move_y = int(target_y * sensitivity_factor * acceleration)
    
    # Apply additional smoothing
    smoothing_factor = settings.get("smoothing_factor", 0.5)
    mouse_move_x = int(mouse_move_x * smoothing_factor)
    mouse_move_y = int(mouse_move_y * smoothing_factor)
    
    return mouse_move_x, mouse_move_y

def _move_mouse(move_x, move_y, settings):
    if settings["mouse_input"] == "arduino":
        settings["arduino"].write(f"{move_x}:{move_y}:0x".encode())
    else:
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, move_x, move_y, 0, 0)

def trigger_bot(target_x, target_y, settings):
    trigger_distance = round(settings["trigger_bot_distance"])
    if abs(target_x) <= trigger_distance and abs(target_y) <= trigger_distance:
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(0.01)  # Short delay to simulate a quick click
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)