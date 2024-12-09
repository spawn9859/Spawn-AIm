# controller_setup.py

import pygame

# This is for Xbox One controllers!

def initialize_pygame_and_controller():
    """
    Initializes Pygame and attempts to connect to an Xbox controller.

    Returns:
        pygame.joystick.Joystick or None: The connected controller instance or None if no controller is found.
    """
    pygame.init()
    pygame.joystick.init()
    try:
        controller = pygame.joystick.Joystick(
            0
        )  # Assumes the controller is the first joystick
        controller.init()
        return controller
    except pygame.error:
        print("Unable to initialize the Xbox Controller")
        return None

def get_left_trigger(controller):
    """
    Gets the normalized value of the left trigger.

    Args:
        controller (pygame.joystick.Joystick): The controller instance.

    Returns:
        float: Normalized value of the left trigger (0.0 to 1.0).
    """
    return (
        controller.get_axis(4) + 1
    ) / 2  # Normalize to 0 (unpressed) to 1 (fully pressed)

def get_right_trigger(controller):
    """
    Gets the normalized value of the right trigger.

    Args:
        controller (pygame.joystick.Joystick): The controller instance.

    Returns:
        float: Normalized value of the right trigger (0.0 to 1.0).
    """
    return (
        controller.get_axis(5) + 1
    ) / 2  # Normalize to 0 (unpressed) to 1 (fully pressed)