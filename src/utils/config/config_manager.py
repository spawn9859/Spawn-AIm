import json
import os

class ConfigManager:
    def __init__(self, settings_profile):
        self.script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.config_path = os.path.join(self.script_dir, "configuration", f"{settings_profile.lower()}.json")
        self.key_mapping_path = os.path.join(self.script_dir, "configuration", "key_mapping.json")
        self.settings = self.load_settings()
        self.key_mapping = self.load_key_mapping()
        self.settings["show_fov"] = "off"
        self.load_settings()
        

    def load_settings(self):
        with open(self.config_path, "r") as f:
            return json.load(f)
        self.validate_settings()

    def validate_settings(self):
        default_values = {
            "auto_aim": "off",
            "trigger_bot": "off",
            "toggle": "off",
            "recoil": "off",
            "aim_shake": "off",
            "overlay": "off",
            "preview": "off",
            "mask_left": "off",
            "mask_right": "off",
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
            "fov_enabled": "off",
            "show_fov": "off",
            "confidence": 0.55,  # Changed from string to float
            "calibration_mode": "off",
        }
        
        for key, default_value in default_values.items():
            if key not in self.settings or not isinstance(self.settings[key], type(default_value)):
                self.settings[key] = default_value
        

    def load_key_mapping(self):
        with open(self.key_mapping_path, "r") as f:
            return json.load(f)

    def save_settings(self):
        with open(self.config_path, "w") as f:
            json.dump(self.settings, f, indent=4)

    def get_setting(self, key):
        return self.settings.get(key)

    def update_setting(self, key, value):
        self.settings[key] = value
        self.save_settings()

    def get_key_code(self, key):
        return self.key_mapping.get(key)
