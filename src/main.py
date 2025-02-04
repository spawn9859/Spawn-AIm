from line_profiler import profile
import os
import sys
import json
import time
import random
import numba
import torch
import numpy as np
import pygame
import serial
import serial.tools.list_ports
from colorama import Fore, Style
import win32api
import mss
from controller_setup import initialize_pygame_and_controller, get_left_trigger, get_right_trigger
from core.send_targets import send_targets
from gui.main_window import MainWindow
from spawn_utils.config_manager import ConfigManager
from spawn_utils.yolo_handler import YOLOHandler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Use CuPy for GPU acceleration if available
if torch.cuda.is_available():
    import cupy as cp

# --- Global Variables ---
MODELS_PATH = os.path.join(SCRIPT_DIR, "models")

screen = None
arduino = None

@numba.njit(parallel=True)
def calculate_targets_numba(boxes, width, height, headshot_percent):
    width_half = width / 2
    height_half = height / 2
    
    x = ((boxes[:, 0] + boxes[:, 2]) / 2) - width_half
    y = ((boxes[:, 1] + boxes[:, 3]) / 2) + headshot_percent * (boxes[:, 1] - ((boxes[:, 1] + boxes[:, 3]) / 2)) - height_half
    
    targets = np.column_stack((x, y))
    distances = np.sqrt(np.sum(targets**2, axis=1))
    
    return targets, distances

def preprocess_frame(np_frame):
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

def mask_frame(frame):
    """Masks out specified regions of the frame."""
    if config_manager.get_setting("mask_left") == "on":
        frame[
            int(config_manager.get_setting("height") - config_manager.get_setting("mask_height")) : config_manager.get_setting("height"),
            0 : int(config_manager.get_setting("mask_width")),
            :,
        ] = 0
    if config_manager.get_setting("mask_right") == "on":
        frame[
            int(config_manager.get_setting("height") - config_manager.get_setting("mask_height")) : config_manager.get_setting("height"),
            int(config_manager.get_setting("width") - config_manager.get_setting("mask_width")) : config_manager.get_setting("width"),
            :,
        ] = 0
    return frame

def initialize_game_window():
    global screen
    left = int(win32api.GetSystemMetrics(0) / 2 - config_manager.get_setting("width") / 2)
    top = int(win32api.GetSystemMetrics(1) / 2 - config_manager.get_setting("height") / 2)
    right = left + config_manager.get_setting("width")
    bottom = top + config_manager.get_setting("height")

    screen = mss.mss()
    return {"top": top, "left": left, "width": config_manager.get_setting("width"), "height": config_manager.get_setting("height")}

@profile
def process_detections(detections):
    # Cache configuration values
    conf_threshold = config_manager.get_setting("confidence") / 100
    width = config_manager.get_setting("width")
    height = config_manager.get_setting("height")
    headshot = config_manager.get_setting("headshot") / 100
    
    targets = np.empty((0, 2), dtype=np.float32)
    distances = np.empty(0, dtype=np.float32)
    coordinates = np.empty((0, 4), dtype=np.float32)

    if detections.shape[0] > 0:
        # Process detections
        valid_detections = detections[:, 4] > conf_threshold
        boxes = detections[valid_detections, :4]
        confs = detections[valid_detections, 4]
        classes = detections[valid_detections, 5]

        if boxes.shape[0] > 0:
            targets, distances = calculate_targets_numba(
                boxes,
                width,
                height,
                headshot,
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
def main_loop(controller, main_window, yolo_handler, monitor):
    start_time = time.time()
    frame_count = 0
    pressing = False
    aim_shake = (0, 0)  # Local tuple for (random_x, random_y)

    def process_frame():
        nonlocal frame_count, start_time, pressing, aim_shake

        frame_count += 1
        np_frame = preprocess_frame(np.array(screen.grab(monitor)))
        pygame.event.pump()

        with torch.no_grad():
            frame = mask_frame(np_frame)
            detections = yolo_handler.detect(frame)
            targets, distances, coordinates = process_detections(detections)

        # Use local aim_shake values when sending targets.
        send_targets(
            controller,
            config_manager.settings,
            targets,
            distances,
            aim_shake[0],
            aim_shake[1],
            get_left_trigger,
            get_right_trigger,
        )

        main_window.update_preview(np_frame, coordinates, targets, distances)

        if config_manager.get_setting("overlay") == "on":
           main_window.update_overlay(coordinates)

        elapsed_time = time.time() - start_time
        if elapsed_time >= 0.2:
           main_window.update_fps_label(round(frame_count / elapsed_time))
           frame_count = 0
           start_time = time.time()
           # Update aim shake offsets locally rather than using a separate function.
           if config_manager.get_setting("aim_shake") == "on":
               strength = int(config_manager.get_setting("aim_shake_strength"))
               aim_shake = (random.randint(-strength, strength), random.randint(-strength, strength))
           else:
               aim_shake = (0, 0)

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

        if main_window.running:
            main_window.root.after(1, process_frame)

    main_window.root.after(1, process_frame)
    main_window.run()

@profile
def main(**argv):
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
        monitor = initialize_game_window()

        main_loop(controller, main_window, yolo_handler, monitor)

    except Exception as e:
        pr_red(f"An error occurred: {str(e)}")
    finally:
        pr_green("Goodbye!")

if __name__ == "__main__":
   main(settingsProfile="config", yoloVersion=5, version=0)