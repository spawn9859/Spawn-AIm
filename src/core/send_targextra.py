import win32api
import win32con
import numpy as np

def send_targets(controller, settings, targets, distances, random_x, random_y, get_left_trigger, get_right_trigger):
    """Sends mouse movements based on detected targets and settings.

    Args:
        controller: The connected controller object.
        settings: A dictionary containing the current application settings.
        targets: A list of (x, y) tuples representing the calculated aim points for each target.
        distances: A list of distances corresponding to each target in 'targets'.
        random_x: Random horizontal offset for aim shake.
        random_y: Random vertical offset for aim shake.
        get_left_trigger: Function to get the value of the left trigger from the controller. 
        get_right_trigger: Function to get the value of the right trigger from the controller.
    """

    if distances.size == 0:
        return

    # Apply settings
    trigger_distance = round(settings["trigger_bot_distance"])
    sensitivity_factor = settings["sensitivity"] / 20 
    
    # Find closest target
    min_index = np.argmin(distances)
    target_x, target_y = targets[min_index]

    # Limit target movement speed
    target_x = max(-settings["max_move"], min(target_x, settings["max_move"]))
    target_y = max(-settings["max_move"], min(target_y, settings["max_move"]))

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

        # Calculate mouse movement with aim shake
        mouse_move_x = int((target_x + random_x) * sensitivity_factor)
        mouse_move_y = int((target_y + random_y + recoil) * sensitivity_factor)

        # Determine if trigger bot should activate
        click = 1 if (
            settings["trigger_bot"] == "on"
            and abs(target_x) <= trigger_distance
            and abs(target_y) <= trigger_distance
        ) else 0

        _move_mouse(mouse_move_x, mouse_move_y, click, settings)

def _move_mouse(move_x, move_y, click, settings):
    """Moves the mouse using either default OS functions or Arduino communication."""
    if settings["mouse_input"] == "arduino":
        settings["arduino"].write(f"{move_x}:{move_y}:{click}x".encode()) 
    else:
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, move_x, move_y, 0, 0)
        if click == 1:
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)