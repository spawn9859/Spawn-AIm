import pygame
import logging  # Added for logging errors
import win32api
import win32con

class ControllerInitializationError(Exception):
    pass

def initialize_input_device():
    """Initialize either controller or mouse input"""
    pygame.init()
    pygame.joystick.init()
    
    try:
        controller = pygame.joystick.Joystick(0)
        controller.init()
        return {"type": "controller", "device": controller}
    except pygame.error as e:
        logging.info("No Xbox controller detected, defaulting to mouse input")
        return {"type": "mouse", "device": None}

def get_left_trigger(input_device):
    """Get activation input value from either controller or mouse"""
    if input_device["type"] == "controller":
        return (input_device["device"].get_axis(4) + 1) / 2
    else:
        # Check if right mouse button is pressed
        return 1.0 if win32api.GetKeyState(win32con.VK_RBUTTON) < 0 else 0.0

def get_right_trigger(input_device):
    """Get secondary input value from either controller or mouse"""
    if input_device["type"] == "controller":
        return (input_device["device"].get_axis(5) + 1) / 2
    else:
        # Could map to another mouse button or key if needed
        return 0.0
