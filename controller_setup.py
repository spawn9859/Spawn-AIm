import pygame


def initialize_pygame_and_controller():
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
    return (
        controller.get_axis(4) + 1
    ) / 2  # Normalize to 0 (unpressed) to 1 (fully pressed)
