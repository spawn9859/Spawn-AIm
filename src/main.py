# main.py

from line_profiler import profile
import os
import sys
import json
import time
import random
import numba
import threading
import torch
import numpy as np
import pygame
import serial
import onnxruntime as ort
import serial.tools.list_ports
from colorama import Fore, Style
import win32api
from utils.general import non_max_suppression
from ultralytics import YOLO
from ultralytics.utils import ops
import mss
from controller_setup import initialize_pygame_and_controller, get_left_trigger, get_right_trigger
from core.send_targets import send_targets
from gui.main_window import MainWindow
from spawn_utils.config_manager import ConfigManager
from spawn_utils.yolo_handler import YOLOHandler
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Use CuPy for GPU acceleration if available
if torch.cuda.is_available():
    import cupy as cp

# --- Global Variables ---
MODELS_PATH = os.path.join(SCRIPT_DIR, "models")

screen = None
random_x, random_y, arduino = 0, 0, None

@numba.jit(nopython=True, fastmath=True)
def calculate_targets_numba(boxes, width_half, height_half, headshot_percent):
    """
    Calculates target coordinates and distances using Numba for performance.

    Args:
        boxes (np.ndarray): Array of bounding boxes.
        width_half (float): Half the width of the screen.
        height_half (float): Half the height of the screen.
        headshot_percent (float): Percentage adjustment for headshots.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Target coordinates and distances.
    """
    
    x = ((boxes[:, 0] + boxes[:, 2]) / 2) - width_half
    y = ((boxes[:, 1] + boxes[:, 3]) / 2) + headshot_percent * (boxes[:, 1] - ((boxes[:, 1] + boxes[:, 3]) / 2)) - height_half
    
    targets = np.column_stack((x, y))
    distances = np.sqrt(np.sum(targets**2, axis=1))
    
    return targets, distances

def preprocess_frame(np_frame):
    """
    Preprocesses the frame by removing the alpha channel if present.

    Args:
        np_frame (np.ndarray): Input frame.

    Returns:
        np.ndarray: Processed frame.
    """
    if np_frame.shape[2] == 4:
        return np_frame[:, :, :3]
    return np_frame

# --- Utility Functions ---
def pr_red(text):
    print(Fore.RED + text, Style.RESET_ALL)

def pr_green(text):
    print(Fore.GREEN + text + Style.RESET_ALL)

def pr_yellow(text):
    print(Fore.YELLOW + text + Style.RESET_ALL)

def pr_blue(text):
    print(Fore.BLUE + text + Style.RESET_ALL)

def pr_purple(text):
    print(Fore.MAGENTA + text + Style.RESET_ALL)

def pr_cyan(text):
    print(Fore.CYAN + text + Style.RESET_ALL)

def get_keycode(key):
    """Gets the virtual key code for a given key."""
    return config_manager.get_key_code(key) or win32api.VkKeyScan(key)

def update_aim_shake():
    """Updates random aim shake offsets."""
    global random_x, random_y
    if config_manager.get_setting("aim_shake") == "on":
        aim_shake_strength = int(config_manager.get_setting("aim_shake_strength"))
        random_x = random.randint(-aim_shake_strength, aim_shake_strength)
        random_y = random.randint(-aim_shake_strength, aim_shake_strength)
    else:
        random_x = 0
        random_y = 0

def mask_frame(frame, config_manager):
    """
    Masks out specified regions of the frame using vectorized operations.

    Args:
        frame (np.ndarray): Input frame.
        config_manager (ConfigManager): Configuration manager instance.

    Returns:
        np.ndarray: Masked frame.
    """
    if config_manager.get_setting("mask_left") == "on":
        mask_height = int(config_manager.get_setting("mask_height"))
        mask_width = int(config_manager.get_setting("mask_width"))
        height = int(config_manager.get_setting("height"))
        frame[height - mask_height:height, 0:mask_width, :] = 0
    if config_manager.get_setting("mask_right") == "on":
        mask_height = int(config_manager.get_setting("mask_height"))
        mask_width = int(config_manager.get_setting("mask_width"))
        height = int(config_manager.get_setting("height"))
        width = int(config_manager.get_setting("width"))
        frame[height - mask_height:height, width - mask_width:width, :] = 0
    return frame

def initialize_game_window(config_manager):
    """
    Initializes the game window and returns the screen capture region.

    Args:
        config_manager (ConfigManager): Configuration manager instance.

    Returns:
        dict: Screen capture region.
    """
    global screen
    width = int(config_manager.get_setting("width"))
    height = int(config_manager.get_setting("height"))
    left = int(win32api.GetSystemMetrics(0) / 2 - width / 2)
    top = int(win32api.GetSystemMetrics(1) / 2 - height / 2)

    screen = mss.mss()
    return {"top": top, "left": left, "width": width, "height": height}

@profile
def process_detections(detections, config_manager):
    """
    Processes YOLO detections, calculates targets, and filters by FOV.

    Args:
        detections (np.ndarray): YOLO detection results.
        config_manager (ConfigManager): Configuration manager instance.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Targets, distances, and coordinates.
    """
    targets = np.empty((0, 2), dtype=np.float32)
    distances = np.empty(0, dtype=np.float32)
    coordinates = np.empty((0, 4), dtype=np.float32)

    if detections.shape[0] > 0:
        confidence_threshold = config_manager.get_setting("confidence") / 100
        valid_detections = detections[:, 4] > confidence_threshold
        boxes = detections[valid_detections, :4]
        
        if boxes.shape[0] > 0:
            width_half = config_manager.get_setting("width") / 2
            height_half = config_manager.get_setting("height") / 2
            headshot_percent = config_manager.get_setting("headshot") / 100

            targets, distances = calculate_targets_numba(
                boxes,
                width_half,
                height_half,
                headshot_percent,
            )
            coordinates = boxes

    if config_manager.get_setting("fov_enabled") == "on":
        fov_size = config_manager.get_setting("fov_size")
        fov_mask = np.sum(targets**2, axis=1) <= fov_size**2
        targets = targets[fov_mask]
        distances = distances[fov_mask]
        coordinates = coordinates[fov_mask]

    return targets, distances, coordinates

@profile
def main_loop(controller, main_window, yolo_handler, monitor, config_manager):
    """
    Main processing loop for handling frame capture, detection, and UI updates.

    Args:
        controller: Game controller instance.
        main_window (MainWindow): Main window instance.
        yolo_handler (YOLOHandler): YOLO handler instance.
        monitor (dict): Screen capture region.
        config_manager (ConfigManager): Configuration manager instance.
    """
    start_time = time.time()
    frame_count = 0
    pressing = False

    @profile  # Uncomment this decorator
    def process_frame():
        nonlocal frame_count, start_time, pressing

        iteration_start_time = time.time()

        frame_count += 1
        np_frame = preprocess_frame(np.array(screen.grab(monitor)))
        pygame.event.pump()
        
        frame = cv2.resize(np_frame, (config_manager.get_setting("width"), config_manager.get_setting("height")))

        with torch.no_grad():
            frame = mask_frame(frame, config_manager)
            detections = yolo_handler.detect(frame)
            targets, distances, coordinates = process_detections(detections, config_manager)

        send_targets(
            controller,
            config_manager.settings,
            targets,
            distances,
            random_x,
            random_y,
            get_left_trigger,
            get_right_trigger,
        )

        main_window.update_preview(frame, coordinates, targets, distances)

        if config_manager.get_setting("overlay") == "on":
           main_window.update_overlay(coordinates)

        elapsed_time = time.time() - start_time
        if elapsed_time >= 0.2:
           main_window.update_fps_label(round(frame_count / elapsed_time))
           frame_count = 0
           start_time = time.time()
           update_aim_shake()

        if config_manager.get_setting("toggle") == "on":
           trigger_value = get_left_trigger(controller)
           if trigger_value > 0.5 and not pressing:
               pressing = True
               main_window.toggle_auto_aim()
           elif trigger_value <= 0.5 and pressing:
               pressing = False

        if win32api.GetKeyState(config_manager.get_setting("quit_key")) in (-127, -128):
            main_window.on_closing()
            return

        iteration_end_time = time.time()
        iteration_duration = iteration_end_time - iteration_start_time
        print(f"Iteration Time: {iteration_duration:.4f} seconds")

        if main_window.running:
            main_window.root.after(1, process_frame)

    main_window.root.after(1, process_frame)
    main_window.run()

#@profile
def main(**argv):
    """
    Main function to initialize settings, create the main window, and start the processing loop.

    Args:
        **argv: Command-line arguments.
    """
    global config_manager

    # Initialize ConfigManager
    config_manager = ConfigManager(argv['settingsProfile'])

    # Print welcome message and instructions
    pr_purple(
        """
  ██████  ██▓███   ▄▄▄      █     █░ ███▄    █      ▄▄▄       ██▓ ███▄ ▄███▓ ▄▄▄▄    ▒█████  ▄▄▄█████▓
▒██    ▒ ▓██░  ██ ▒████▄   ▓█░ █ ░█░ ██ ▀█   █     ▒████▄   ▒▓██▒▓██▒▀█▀ ██▒▓█████▄ ▒██▒  ██▒▓  ██▒ ▓▒
░ ▓██▄   ▓██░ ██▓▒▒██  ▀█▄ ▒█░ █ ░█ ▓██  ▀█ ██▒    ▒██  ▀█▄ ▒▒██▒▓██    ▓██░▒██▒ ▄██▒██░  ██▒▒ ▓██░ ▒░
  ▒   ██▒▒██▄█▓▒ ▒░██▄▄▄▄██░█░ █ ░█ ▓██▒  ▐▌██▒    ░██▄▄▄▄██░░██░▒██    ▒██ ▒██░█▀  ▒██   ██░░ ▓██▓ ░ 
▒██████▒▒▒██▒ ░  ░▒▓█   ▓██░░██▒██▓ ▒██░   ▓██░     ▓█   ▓██░░██░▒██▒   ░██▒░▓█  ▀█▓░ ████▓▒░  ▒██▒ ░ 
▒ ▒▓▒ ▒ ░▒▓▒░ ░  ░░▒▒   ▓▒█░ ▓░▒ ▒  ░ ▒░   ▒ ▒      ░   ▒▒ ░ ▒ ░░ ▒░   ░  ░░▒▓███▀▒░ ▒░▒░▒░   ▒ ░░   
░ ░▒  ░  ░▒ ░     ░ ░   ▒▒   ▒ ░ ░  ░ ░░   ░ ▒░      ░    ░    ░         ░    ░    ░ ░ ░ ░ ▒    ░      
░  ░  ░  ░░         ░   ▒    ░   ░     ░   ░ ░       ░    ░    ░         ░    ░          ░ ░"""
    )
    print("https://github.com/spawn9859/Spawn-Aim")
    pr_yellow("\nMake sure your game is in the center of your screen!")

    # Load launcher settings from JSON file
    with open(
        os.path.join(
            SCRIPT_DIR,
            "configuration",
            f"{argv['settingsProfile'].lower()}.json",
        ),
        "r",
    ) as f:
        launcher_settings = json.load(f)

    # Detect available serial ports
    ports = [port[0] for port in serial.tools.list_ports.comports()]
    default_port = next(
        (port[0] for port in ports if "Arduino" in port[1]), "COM1"
    )

    # Initialize Pygame and controller
    controller = initialize_pygame_and_controller()

    # Get activation and quit key codes
    activation_key = get_keycode(launcher_settings["activationKey"])
    quit_key = get_keycode(launcher_settings["quitKey"])

    # Initialize settings from launcher settings
    config_manager.update_setting("auto_aim", "on")
    config_manager.update_setting("trigger_bot", "on" if launcher_settings["autoFire"] else "off")
    config_manager.update_setting("toggle", "on" if launcher_settings["toggleable"] else "off")
    config_manager.update_setting("recoil", "off")
    config_manager.update_setting("aim_shake", "on" if launcher_settings["aimShakey"] else "off")
    config_manager.update_setting("overlay", "off")
    config_manager.update_setting("preview", "on" if launcher_settings["visuals"] else "off")
    config_manager.update_setting("mask_left", "on" if launcher_settings["maskLeft"] and launcher_settings["useMask"] else "off")
    config_manager.update_setting("mask_right", "on" if not launcher_settings["maskLeft"] and launcher_settings["useMask"] else "off")
    config_manager.update_setting("sensitivity", launcher_settings["movementAmp"] * 100)
    config_manager.update_setting("headshot", launcher_settings["headshotDistanceModifier"] * 100 if launcher_settings["headshotMode"] else 40)
    config_manager.update_setting("trigger_bot_distance", launcher_settings["autoFireActivationDistance"])
    config_manager.update_setting("confidence", launcher_settings["confidence"])
    config_manager.update_setting("recoil_strength", 0)
    config_manager.update_setting("aim_shake_strength", launcher_settings["aimShakeyStrength"])
    config_manager.update_setting("max_move", 100)
    config_manager.update_setting("height", launcher_settings["screenShotHeight"])
    config_manager.update_setting("width", launcher_settings["screenShotHeight"])
    config_manager.update_setting("mask_width", launcher_settings["maskWidth"])
    config_manager.update_setting("mask_height", launcher_settings["maskHeight"])
    config_manager.update_setting("yolo_version", f"v{argv['yoloVersion']}")
    config_manager.update_setting("yolo_model", "v5_Fortnite_taipeiuser")
    config_manager.update_setting("yolo_mode", "tensorrt")
    config_manager.update_setting("yolo_device", {1: "cpu", 2: "amd", 3: "nvidia"}.get(launcher_settings["onnxChoice"]))
    config_manager.update_setting("activation_key", activation_key)
    config_manager.update_setting("quit_key", quit_key)
    config_manager.update_setting("activation_key_string", launcher_settings["activationKey"])
    config_manager.update_setting("quit_key_string", launcher_settings["quitKey"])
    config_manager.update_setting("mouse_input", "default")
    config_manager.update_setting("arduino", default_port)
    config_manager.update_setting("fov_enabled", "on" if launcher_settings["fovToggle"] else "off")
    config_manager.update_setting("fov_size", launcher_settings["fovSize"])

    # Create main window
    main_window = MainWindow(config_manager)

    # Initialize YOLOHandler
    yolo_handler = YOLOHandler(config_manager, MODELS_PATH)

    if config_manager.get_setting("overlay") == "on":
        main_window.toggle_overlay()

    try:
        # Initialize screen capture
        monitor = initialize_game_window(config_manager)

        main_loop(controller, main_window, yolo_handler, monitor, config_manager)

    except Exception as e:
        pr_red(f"An error occurred: {str(e)}")
    finally:
        pr_green("Goodbye!")

if __name__ == "__main__":
   main(settingsProfile="config", yoloVersion=5, version=0)