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
        

    def load_settings(self):
        with open(self.config_path, "r") as f:
            return json.load(f)

    def load_key_mapping(self):
        with open(self.key_mapping_path, "r") as f:
            return json.load(f)

    def save_settings(self):
        with open(self.config_path, "w") as f:
            json.dump(self.settings, f, indent=4)

    def update_setting(self, key, value):
        self.settings[key] = value
        self.save_settings()

    def get_setting(self, key):
        return self.settings.get(key)

    def get_key_code(self, key):
        return self.key_mapping.get(key)
