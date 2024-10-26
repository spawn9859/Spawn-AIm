import asyncio
import json
import os
import random
import sys
import threading
import time
import mss
import numpy as np
import pygame
import torch
import win32api
import logging
import winloop
from concurrent.futures import ThreadPoolExecutor  # {{ edit_6 }} Replace ProcessPoolExecutor with ThreadPoolExecutor
from line_profiler import profile
from serial.tools import list_ports
from core.controller_setup import get_left_trigger, get_right_trigger, ControllerInitializationError, initialize_input_device
from core.send_targets import send_targets
from gui.main_window import MainWindow
from spawn_utils.config_manager import ConfigManager
from spawn_utils.yolo_handler import YOLOHandler

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(SCRIPT_DIR, "models")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


class AimAssistant:
    """
    AimAssistant manages the core functionality of the aim assist tool,
    including configuration, controller input, frame processing, and GUI interaction.
    """

    def __init__(self, settings_profile: str, yolo_version: int, version: int):
        """
        Initialize the AimAssistant with the given settings profile, YOLO version, and application version.

        Args:
            settings_profile (str): The profile name for settings configuration.
            yolo_version (int): The version of YOLO to use for object detection.
            version (int): The application version number.
        """
        self.config_manager = ConfigManager(settings_profile)
        self.models_path = MODELS_PATH
        self.screen = None
        self.random_x = 0
        self.random_y = 0
        self.arduino = self.detect_arduino()

        self.load_settings(settings_profile, yolo_version)
        self.main_window = MainWindow(self.config_manager)
        self.yolo_handler = YOLOHandler(self.config_manager, self.models_path)
        if self.config_manager.get_setting("overlay"):
            self.main_window.toggle_overlay()

        # Configure asyncio to use winloop for better performance
        asyncio.set_event_loop_policy(winloop.EventLoopPolicy())  # {{ edit_2 }}
        self.loop = asyncio.new_event_loop()
        self.shutdown_event = threading.Event()

        # Initialize ThreadPoolExecutor for CPU-bound tasks  # {{ edit_3 }}
        self.process_pool = ThreadPoolExecutor()  # {{ edit_3 }}

    def detect_arduino(self) -> str:
        ports = [port.device for port in list_ports.comports()]
        return next((port for port in ports if "Arduino" in port.description), "COM1")

    def load_settings(self, settings_profile: str, yolo_version: int) -> None:
        settings_path = os.path.join(SCRIPT_DIR, "configuration", f"{settings_profile.lower()}.json")
        with open(settings_path, "r") as f:
            launcher_settings = json.load(f)

        # Initialize input device (controller or mouse)
        input_device = initialize_input_device()
        self.controller = input_device
        
        if input_device["type"] == "mouse":
            logging.info("Using mouse input - Right click to activate aim assist")

        # Get key codes
        activation_key = self.get_keycode(launcher_settings.get("activationKey", "F1"))
        quit_key = self.get_keycode(launcher_settings.get("quitKey", "F2"))

        # Update settings with defaults
        defaults = {
            "auto_aim": False,
            "trigger_bot": launcher_settings.get("autoFire", False),
            "toggle": launcher_settings.get("toggleable", False),
            "recoil": False,
            "aim_shake": launcher_settings.get("aimShakey", False),
            "overlay": False,
            "preview": launcher_settings.get("visuals", False),
            "mask_left": launcher_settings.get("maskLeft", False) and launcher_settings.get("useMask", False),
            "mask_right": not launcher_settings.get("maskLeft", False) and launcher_settings.get("useMask", False),
            "sensitivity": launcher_settings.get("movementAmp", 1) * 100,
            "headshot": launcher_settings.get("headshotDistanceModifier", 0.4) * 100 if launcher_settings.get("headshotMode", False) else 40,
            "trigger_bot_distance": launcher_settings.get("autoFireActivationDistance", 0),
            "confidence": launcher_settings.get("confidence", 50),
            "recoil_strength": 0,
            "aim_shake_strength": launcher_settings.get("aimShakeyStrength", 0),
            "max_move": 100,
            "height": launcher_settings.get("screenShotHeight", 600),
            "width": launcher_settings.get("screenShotWidth", 800),
            "mask_width": launcher_settings.get("maskWidth", 100),
            "mask_height": launcher_settings.get("maskHeight", 100),
            "yolo_version": f"v{yolo_version}",
            "yolo_model": "v5_Fortnite_taipeiuser",
            "yolo_mode": "tensorrt",
            "yolo_device": {1: "cpu", 2: "amd", 3: "nvidia"}.get(launcher_settings.get("onnxChoice", 1), "cpu"),
            "activation_key": activation_key,
            "quit_key": quit_key,
            "activation_key_string": launcher_settings.get("activationKey", "F1"),
            "quit_key_string": launcher_settings.get("quitKey", "F2"),
            "mouse_input": "default",
            "arduino": self.arduino,
            "fov_enabled": launcher_settings.get("fovToggle", False),
            "fov_size": launcher_settings.get("fovSize", 100),
            "input_type": input_device["type"],  # Add this line
        }

        for key, value in defaults.items():
            self.config_manager.update_setting(key, value)

    def get_keycode(self, key: str) -> int:
        return self.config_manager.get_key_code(key) or win32api.VkKeyScan(key)

    @staticmethod
    @profile
    def calculate_targets_numba(boxes: np.ndarray, width: int, height: int, headshot_percent: float) -> tuple:
        width_half = width / 2
        height_half = height / 2

        x = ((boxes[:, 0] + boxes[:, 2]) / 2) - width_half
        y = ((boxes[:, 1] + boxes[:, 3]) / 2 + headshot_percent * (boxes[:, 1] - ((boxes[:, 1] + boxes[:, 3]) / 2))) - height_half

        targets = np.column_stack((x, y))
        distances = np.sqrt((targets ** 2).sum(axis=1))

        return targets, distances

    def preprocess_frame(self, np_frame: np.ndarray) -> np.ndarray:
        return np_frame[:, :, :3] if np_frame.shape[2] == 4 else np_frame

    def update_aim_shake(self) -> None:
        if self.config_manager.get_setting("aim_shake"):
            strength = int(self.config_manager.get_setting("aim_shake_strength"))
            self.random_x = random.randint(-strength, strength)
            self.random_y = random.randint(-strength, strength)
        else:
            self.random_x, self.random_y = 0, 0

    def mask_frame(self, frame: np.ndarray) -> np.ndarray:
        settings = self.config_manager.settings
        if settings.get("mask_left"):
            frame[-settings["mask_height"]:, :settings["mask_width"], :] = 0
        if settings.get("mask_right"):
            frame[-settings["mask_height"]:, -settings["mask_width"]:, :] = 0
        return frame

    @profile
    def process_detections(self, detections: np.ndarray) -> tuple:
        """
        Process YOLO detections to determine targets and their distances.

        Args:
            detections (np.ndarray): Array of detection results from YOLO.

        Returns:
            tuple: A tuple containing targets (np.ndarray), distances (np.ndarray), and boxes (np.ndarray).
        """
        settings = self.config_manager.settings
        if detections.size == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.float32), np.empty((0, 4), dtype=np.float32)

        valid = detections[:, 4] > settings["confidence"] / 100
        boxes = detections[valid, :4]
        if boxes.size == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.float32), np.empty((0, 4), dtype=np.float32)

        targets, distances = self.calculate_targets_numba(
            boxes,
            settings["width"],
            settings["height"],
            settings["headshot"] / 100,
        )

        if settings["fov_enabled"]:
            fov_size_sq = settings["fov_size"] ** 2
            fov_mask = (targets ** 2).sum(axis=1) <= fov_size_sq
            targets, distances, boxes = targets[fov_mask], distances[fov_mask], boxes[fov_mask]

        return targets, distances, boxes

    async def process_frame_async(self, monitor: dict) -> None:
        try:
            start_time = time.time()
            frame_count = 0
            pressing = False

            while self.main_window.running and not self.shutdown_event.is_set():
                frame_count += 1
                np_frame = np.array(self.screen.grab(monitor))
                pygame.event.pump()

                # Offload CPU-bound tasks to the ThreadPoolExecutor  # {{ edit_4 }}
                frame = await self.loop.run_in_executor(self.process_pool, self.preprocess_frame, np_frame)
                detections = await self.loop.run_in_executor(self.process_pool, self.yolo_handler.detect, frame)
                targets, distances, coordinates = await self.loop.run_in_executor(self.process_pool, self.process_detections, detections)

                send_targets(
                    self.controller,
                    self.config_manager.settings,
                    targets,
                    distances,
                    self.random_x,
                    self.random_y,
                    get_left_trigger,
                    get_right_trigger,
                )

                self.main_window.root.after(
                    0, self.main_window.update_preview, np_frame, coordinates, targets, distances
                )

                if self.config_manager.get_setting("overlay"):
                    self.main_window.root.after(0, self.main_window.update_overlay, coordinates)

                elapsed = time.time() - start_time
                if elapsed >= 0.2:
                    fps = round(frame_count / elapsed)
                    self.main_window.root.after(0, self.main_window.update_fps_label, fps)
                    frame_count = 0
                    start_time = time.time()
                    self.update_aim_shake()

                if self.config_manager.get_setting("toggle"):
                    trigger_value = get_left_trigger(self.controller)
                    if trigger_value > 0.5 and not pressing:
                        pressing = True
                        self.main_window.root.after(0, self.main_window.toggle_auto_aim)
                    elif trigger_value <= 0.5 and pressing:
                        pressing = False

                if win32api.GetKeyState(self.config_manager.get_setting("quit_key")) < 0:
                    self.main_window.root.after(0, self.main_window.on_closing)
                    break

                await asyncio.sleep(0)  # Yield control to the event loop

        except Exception as e:
            logging.error(f"An error occurred: {e}")
        finally:
            logging.info("Goodbye!")

    async def main_loop_async(self, monitor: dict) -> None:
        await self.process_frame_async(monitor)

    def display_welcome_message(self) -> None:
        welcome_art = """
          ██████  ██▓███   ▄▄▄      █     █░ ███▄    █      ▄▄▄       ██▓ ███▄ ▄███▓ ▄▄▄▄    ▒█████  ▄▄▄█████▓
        ▒██    ▒ ▓██░  ██ ▒████▄   ▓█░ █ ░█░ ██ ▀█   █     ▒████▄   ▒▓██▒▓██▒▀█▀ ██▒▓█████▄ ▒██▒  ██▒▓  ██▒ ▓▒
        ░ ▓██▄   ▓██░ ██▓▒▒██  ▀█▄ ▒█░ █ ░█ ▓██  ▀█ ██▒    ▒██  ▀█▄ ▒▒██▒▓██    ▓██░▒██▒ ▄██▒██░  ██▒▒ ▓██░ ▒░
          ▒   ██▒▒██▄█▓▒ ▒░██▄▄▄▄██░█░ █ ░█ ▓██▒  ▐▌██▒    ░██▄▄▄▄██░░██░▒██    ▒██ ▒██░█▀  ▒██   ██░░ ▓██▓ ░ 
        ▒██████▒▒▒██▒ ░  ░▒▓█   ▓██░░██▒██▓ ▒██░   ▓██░     ▓█   ▓██░ ▒ ░░ ▒░   ░██▒░▓█  ▀█▓░ ████▓▒░  ▒██▒ ░ 
        ▒ ▒▓▒ ▒ ░▒▓▒░ ░  ░░▒▒   ▓▒█░ ▓░▒ ▒  ░ ▒░   ▒ ▒      ░   ▒▒ ░ ▒ ░░ ▒░   ░  ░░▒▓███▀▒░ ▒░▒░▒░   ▒ ░░   
        ░ ░▒  ░  ░▒ ░     ░ ░   ▒▒   ▒ ░ ░  ░ ░░   ░ ▒░      ░    ░    ░         ░    ░    ░ ░ ░ ░ ▒    ░      
        ░  ░  ░  ░░         ░   ▒    ░   ░     ░   ░ ░       ░    ░    ░         ░    ░          ░ ░"""
        logging.info(welcome_art)
        logging.info("https://github.com/spawn9859/Spawn-Aim")
        logging.warning("\nMake sure your game is in the center of your screen!")

    def initialize_game_window(self) -> dict:
        settings = self.config_manager.settings
        screen_width, screen_height = win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)
        left = screen_width // 2 - settings["width"] // 2
        top = screen_height // 2 - settings["height"] // 2
        return {"top": top, "left": left, "width": settings["width"], "height": settings["height"]}

    async def run_async(self) -> None:
        self.display_welcome_message()
        try:
            self.screen = mss.mss()
            monitor = self.initialize_game_window()
            await self.main_loop_async(monitor)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
        finally:
            logging.info("Goodbye!")

    def start_async_loop(self):
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self.run_async())
        finally:
            self.process_pool.shutdown()  # {{ edit_5 }} Shutdown ThreadPoolExecutor
            self.loop.close()

    def run(self) -> None:
        processing_thread = threading.Thread(target=self.start_async_loop, daemon=True)
        processing_thread.start()
        self.main_window.run()
        self.shutdown_event.set()
        processing_thread.join()


def main(settingsProfile: str = "config", yoloVersion: int = 5, version: int = 0) -> None:
    assistant = AimAssistant(settingsProfile, yoloVersion, version)
    assistant.run()


if __name__ == "__main__":
    main()
