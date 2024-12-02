import json
import os
import logging
from typing import Any, Dict

class ConfigManager:
    def __init__(self, settings_profile: str):
        self.script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = os.path.join(self.script_dir, "configuration", f"{settings_profile.lower()}.json")
        self.key_mapping_path = os.path.join(self.script_dir, "configuration", "key_mapping.json")
        self.settings_cache: Dict[str, Any] = {}
        self.settings = self.load_settings()
        self.key_mapping = self.load_key_mapping()
        self.settings["show_fov"] = False
        # Removed redundant load_settings call
        

    def load_settings(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, "r") as f:
                self.settings = json.load(f)
            self.validate_settings()
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"Failed to load settings: {e}")
            self.settings = {}
        return self.settings

    def validate_settings(self):
        default_values = {
            "auto_aim": False,
            "trigger_bot": False,
            "toggle": False,
            "recoil": False,
            "aim_shake": False,
            "overlay": False,
            "preview": False,
            "mask_left": False,
            "mask_right": False,
            "sensitivity": 14.0,
            "headshot": 40.0,
            "trigger_bot_distance": 8,
            "recoil_strength": 0,
            "aim_shake_strength": 5,
            "max_move": 100,
            "height": 320,
            "width": 320,
            "mask_width": 65,
            "mask_height": 145,
            "yolo_version": "v5",
            "yolo_model": "best",
            "yolo_mode": "tensorrt",
            "yolo_device": "nvidia",
            "activation_key": 164,
            "quit_key": 337,
            "activation_key_string": "Alt",
            "quit_key_string": "Q",
            "mouse_input": "default",
            "arduino": "COM1",
            "max_fps": 180,
            "fov_size": 160,
            "fov_enabled": False,
            "show_fov": False,
            "confidence": 0.55,  # Changed from string to float
        }
        
        for key, default_value in default_values.items():
            if key not in self.settings:
                self.settings[key] = default_value
                continue

            if isinstance(default_value, bool):
                self.settings[key] = bool(self.settings[key])
            elif isinstance(default_value, (int, float)):
                try:
                    self.settings[key] = type(default_value)(self.settings[key])
                except (ValueError, TypeError):
                    self.settings[key] = default_value
            elif isinstance(default_value, str):
                self.settings[key] = str(self.settings[key])
            # Add more type checks as necessary
        

    def load_key_mapping(self):
        with open(self.key_mapping_path, "r") as f:
            return json.load(f)

    def save_settings(self):
        with open(self.config_path, "w") as f:
            json.dump(self.settings, f, indent=4)

    def get_setting(self, key):
        if key in self.settings_cache:
            return self.settings_cache[key]
        value = self.settings.get(key)
        self.settings_cache[key] = value
        return value

    def update_setting(self, key, value):
        self.settings[key] = value
        self.settings_cache[key] = value
        self.save_settings()

    def get_key_code(self, key):
        return self.key_mapping.get(key)
