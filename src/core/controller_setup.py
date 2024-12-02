import pygame
import logging  # Added for logging errors
import win32api
import win32con
import time

class ControllerInitializationError(Exception):
    pass

def initialize_input_device():
    """Initialize either controller or mouse input"""
    pygame.init()
    pygame.joystick.init()
    
    try:
        controller = pygame.joystick.Joystick(0)
        controller.init()
        return {"type": "controller", "device": controller, "last_poll": 0}
    except pygame.error as e:
        logging.info("No Xbox controller detected, defaulting to mouse input")
        return {"type": "mouse", "device": None, "last_poll": 0}

def get_left_trigger(input_device):
    """Get activation input value from either controller or mouse"""
    current_time = time.time()
    if current_time - input_device["last_poll"] < 0.01:  # Reduce polling interval
        return input_device.get("last_left_trigger", 0.0)
    input_device["last_poll"] = current_time
    if input_device["type"] == "controller":
        value = (input_device["device"].get_axis(4) + 1) / 2
        input_device["last_left_trigger"] = value
        return value
    else:
        value = 1.0 if win32api.GetKeyState(win32con.VK_RBUTTON) < 0 else 0.0
        input_device["last_left_trigger"] = value
        return value

def get_right_trigger(input_device):
    """Get secondary input value from either controller or mouse"""
    current_time = time.time()
    if current_time - input_device["last_poll"] < 0.05:
        return input_device.get("last_right_trigger", 0.0)
    input_device["last_poll"] = current_time
    if input_device["type"] == "controller":
        value = (input_device["device"].get_axis(5) + 1) / 2
        input_device["last_right_trigger"] = value
        return value
    else:
        return 0.0
