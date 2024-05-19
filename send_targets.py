import math
import win32api
import win32con
from main import get_left_trigger

def send_targets(controller, settings, targets, distances, random_x, random_y):
    def mouse_click():
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def mouse_move(move_x, move_y, click):
        if settings['mouse_input'] == "arduino":
            arduino.write("{}:{}:{}x".format(move_x, move_y, click).encode())
        else:
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, move_x, move_y, 0, 0)
        if not settings['mouse_input'] == "arduino" and click == 1:
            mouse_click()

    if not distances:
        return

    trigger_distance = round(settings['trigger_bot_distance'])
    sensitivity_factor = settings['sensitivity'] / 20
    click = 0

    target_x, target_y = targets[distances.index(min(distances))]
    target_x = min(max(target_x, -settings['max_move']), settings['max_move'])
    target_y = min(max(target_y, -settings['max_move']), settings['max_move'])

    trigger_pressed = get_left_trigger(controller) > 0.5  # Assuming half-press as activation threshold

    if settings['auto_aim'] == "on" and (trigger_pressed and settings['toggle'] == "off" or settings['toggle'] == "on"):
        recoil = int(settings['recoil_strength']) if settings['recoil'] == "on" and trigger_pressed else 0
        mouse_move_x = int((target_x + random_x) * sensitivity_factor)
        mouse_move_y = int((target_y + random_y + recoil) * sensitivity_factor)

        if settings['trigger_bot'] == "on" and target_x <= trigger_distance and target_y <= trigger_distance and target_x >= -trigger_distance and target_y >= -trigger_distance:
            click = 1

        mouse_move(mouse_move_x, mouse_move_y, click)
