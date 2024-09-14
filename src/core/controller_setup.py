import pygame
import logging  # Added for logging errors

class ControllerInitializationError(Exception):
    pass

def initialize_pygame_and_controller():
    pygame.init()
    pygame.joystick.init()
    try:
        controller = pygame.joystick.Joystick(0)
        controller.init()
        return controller
    except pygame.error as e:
        logging.error("Unable to initialize the Xbox Controller")
        raise ControllerInitializationError("Controller initialization failed") from e

def get_left_trigger(controller):
    return (
        controller.get_axis(4) + 1
    ) / 2  # Normalize to 0 (unpressed) to 1 (fully pressed)

def get_right_trigger(controller):
    return (
        controller.get_axis(5) + 1
    ) / 2  # Normalize to 0 (unpressed) to 1 (fully pressed)