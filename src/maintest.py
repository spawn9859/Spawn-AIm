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
import win32api
import logging
from concurrent.futures import ThreadPoolExecutor
from line_profiler import profile
from serial.tools import list_ports
from core.controller_setup import get_left_trigger, get_right_trigger, initialize_input_device
from core.send_targets import send_targets
from gui.main_window import MainWindow
from spawn_utils.config_manager import ConfigManager
from spawn_utils.yolo_handler import YOLOHandler
from typing import Dict
import numba  # Add Numba for just-in-time compilation
from numba import njit

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(SCRIPT_DIR, "models")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class AimAssistant:
    def __init__(self, settings_profile: str, yolo_version: int, version: int):
        self.config_manager = ConfigManager(settings_profile)
        self.models_path = MODELS_PATH
        self.screen = None
        self.random_x = 0
        self.random_y = 0
        self.arduino = self.detect_arduino()

        self.load_settings(settings_profile, yolo_version)
        self.main_window = MainWindow(self.config_manager)
        try:
            self.yolo_handler = YOLOHandler(self.config_manager, self.models_path)
        except Exception as e:
            logging.error(f"Failed to initialize YOLOHandler: {e}")
            # Handle exception or set default

        if self.config_manager.get_setting("overlay"):
            self.main_window.toggle_overlay()

        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        self.loop = asyncio.new_event_loop()
        self.shutdown_event = threading.Event()
        self.process_pool = ThreadPoolExecutor()

    def detect_arduino(self) -> str:
        ports = [port.device for port in list_ports.comports()]
        return next((port for port in ports if "Arduino" in port.description), "COM1")

    def load_settings(self, settings_profile: str, yolo_version: int) -> None:
        settings_path = os.path.join(SCRIPT_DIR, "configuration", f"{settings_profile.lower()}.json")
        with open(settings_path, "r") as f:
            launcher_settings = json.load(f)

        input_device = initialize_input_device()
        self.controller = input_device
        
        if input_device["type"] == "mouse":
            logging.info("Using mouse input - Right click to activate aim assist")

        activation_key = self.get_keycode(launcher_settings.get("activationKey", "F1"))
        quit_key = self.get_keycode(launcher_settings.get("quitKey", "F2"))

        defaults = {
            # ... (unchanged)
            "yolo_device": {1: "cpu", 2: "amd", 3: "nvidia"}.get(launcher_settings.get("onnxChoice", 1), "cpu"),
            "activation_key": activation_key,
            "quit_key": quit_key,
            # ... (unchanged)
        }

        for key, value in defaults.items():
            self.config_manager.update_setting(key, value)

    def get_keycode(self, key: str) -> int:
        return self.config_manager.get_key_code(key) or win32api.VkKeyScan(key)

    @staticmethod
    @profile
    @njit  # Use Numba to speed up this function
    def calculate_targets_numba(boxes, width, height, headshot_percent):
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
            self.random_x, self.random_y = random.randint(-strength, strength), random.randint(-strength, strength)
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
        settings = self.config_manager.settings
        if detections.size == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.float32), np.empty((0, 4), dtype=np.float32)

        valid = detections[:, 4] > settings["confidence"] / 100
        boxes = detections[valid, :4]
        if boxes.size == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.float32), np.empty((0, 4), dtype=np.float32)

        # Call the Numba-optimized function
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

    async def process_frame_async(self, monitor: Dict[str, int]) -> None:
        try:
            start_time = time.time()
            frame_count = 0
            pressing = False

            while self.main_window.running and not self.shutdown_event.is_set():
                frame_count += 1

                np_frame = np.array(self.screen.grab(monitor))
                pygame.event.pump()

                # Inline preprocessing to reduce overhead
                frame = np_frame[:, :, :3] if np_frame.shape[2] == 4 else np_frame

                # Directly call detection without run_in_executor
                detections = self.yolo_handler.detect(frame)
                targets, distances, coordinates = self.process_detections(detections)

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

                if frame_count % 2 == 0:
                    self.main_window.root.after(
                        0, self.main_window.update_preview, np_frame, coordinates, targets, distances
                    )
                    if self.config_manager.get_setting("overlay"):
                        self.main_window.root.after(0, self.main_window.update_overlay, coordinates)

                elapsed = time.time() - start_time
                if elapsed >= 0.2:
                    fps = round(frame_count / elapsed)
                    self.main_window.root.after(0, self.main_window.update_fps_label, fps)
                    frame_count, start_time = 0, time.time()
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

                await asyncio.sleep(0)

        except Exception as e:
            logging.error(f"An error occurred in process_frame_async: {e}")
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
        ░  ��  ░  ░░         ░   ▒    ░   ���     ░   ░ ░       ░    ░    ░         ░    ░          ░ ░"""
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
            self.process_pool.shutdown()
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
