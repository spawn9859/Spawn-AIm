import json
import os

def load_configuration(script_directory):
    with open(f"{script_directory}/configuration/key_mapping.json", 'r') as json_file:
        key_mapping = json.load(json_file)

    with open(f"{script_directory}/configuration/config.json", 'r') as json_file:
        settings = json.load(json_file)
    
    return key_mapping, settings
