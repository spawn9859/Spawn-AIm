import numpy as np
import ctypes
import time
import win32con

user32 = ctypes.windll.user32
user32.mouse_event.argtypes = [ctypes.c_uint, ctypes.c_long, ctypes.c_long, ctypes.c_uint, ctypes.c_void_p]

last_move_time = 0
MOVE_INTERVAL = 0.01  # 10ms between moves

def send_targets(controller, settings, targets, distances, random_x, random_y, get_left_trigger, get_right_trigger):
    global last_move_time
    current_time = time.time()
    if current_time - last_move_time < MOVE_INTERVAL:
        return
    last_move_time = current_time

    if distances.size == 0:
        return

    sensitivity_factor = settings["sensitivity"] / 10
    max_move = settings["max_move"]
    aim_shake_strength = settings["aim_shake_strength"]
    
    min_index = np.argmin(distances)
    target_x, target_y = targets[min_index]

    target_x = np.clip(target_x + random_x * aim_shake_strength, -max_move, max_move)
    target_y = np.clip(target_y + random_y * aim_shake_strength, -max_move, max_move)

    # Get input based on device type
    trigger_pressed = get_left_trigger(controller) > 0.5
    auto_aim_active = settings["auto_aim"] and (
        (trigger_pressed and not settings["toggle"]) or settings["toggle"]
    )

    if auto_aim_active:
        mouse_move_x = int(target_x * sensitivity_factor)
        mouse_move_y = int(target_y * sensitivity_factor)

        if settings["recoil"] and get_right_trigger(controller) > 0.5:
            mouse_move_y += int(settings["recoil_strength"])

        _move_mouse(mouse_move_x, mouse_move_y, settings)

        if settings["trigger_bot"]:
            trigger_bot(target_x, target_y, settings)

def _move_mouse(move_x, move_y, settings):
    if settings["mouse_input"] == "arduino":
        settings["arduino"].write(f"{move_x}:{move_y}:0x".encode())
    else:
        user32.mouse_event(win32con.MOUSEEVENTF_MOVE, move_x, move_y, 0, None)

def trigger_bot(target_x: int, target_y: int, settings: dict) -> None:
    """
    Trigger the bot action if the target is within the specified distance.

    Args:
        target_x (int): Target's x-coordinate.
        target_y (int): Target's y-coordinate.
        settings (dict): Configuration settings.
    """
    trigger_distance = round(settings["trigger_bot_distance"])
    if abs(target_x) <= trigger_distance and abs(target_y) <= trigger_distance:
        user32.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, None)
        time.sleep(0.01)
        user32.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, None)
