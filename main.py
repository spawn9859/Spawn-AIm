import os
import sys
import json
import time
import math
import random
import shutil
import re

import torch
import numpy as np
import cv2
import pygame
import serial
import keyboard
import onnxruntime as ort
import serial.tools.list_ports
from PIL import Image
from colorama import Fore, Style
import customtkinter as ctk
import win32api
import win32con
import win32gui
from utils.general import non_max_suppression
from ultralytics import YOLO
from ultralytics.utils import ops
import bettercam
from controller_setup import initialize_pygame_and_controller, get_left_trigger, get_right_trigger
from send_targets import send_targets

# Define script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Color Definitions
GREEN_COLOR = "#00FF00"  # Bright green
GREY_COLOR = "#808080"  # Grey

# Clone YOLOv5 repository and install requirements if not already present
if not os.path.exists(f"{SCRIPT_DIR}/yolov5"):
    os.system(f"git clone https://github.com/ultralytics/yolov5 {SCRIPT_DIR}/yolov5")
    os.system(f"pip install -r {SCRIPT_DIR}/yolov5/requirements.txt")
sys.path.append(f"{SCRIPT_DIR}/yolov5")

# Use CuPy for GPU acceleration if available
if torch.cuda.is_available():
    import cupy as cp

# --- Global Variables ---
MODELS_PATH = os.path.join(SCRIPT_DIR, "models")
LAUNCHER_MODELS = [
    os.path.splitext(file)[0]
    for file in os.listdir(MODELS_PATH) if file.endswith(".pt")
]

targets = np.empty((0, 2), dtype=np.float32)
distances = np.empty(0, dtype=np.float32)
coordinates = np.empty((0, 4), dtype=np.float32)

model, screen, overlay, canvas = None, None, None, None
random_x, random_y, arduino = 0, 0, None

def load_configuration(SCRIPT_DIR):
    with open(f"{SCRIPT_DIR}/configuration/key_mapping.json", "r") as json_file:
        key_mapping = json.load(json_file)

    with open(f"{SCRIPT_DIR}/configuration/config.json", "r") as json_file:
        settings = json.load(json_file)

    return key_mapping, settings

def calculate_targets_vectorized(boxes):
    width_half = settings["width"] / 2
    height_half = settings["height"] / 2
    headshot_percent = settings["headshot"] / 100
    
    x = ((boxes[:, 0] + boxes[:, 2]) / 2) - width_half
    y = ((boxes[:, 1] + boxes[:, 3]) / 2) + headshot_percent * (boxes[:, 1] - ((boxes[:, 1] + boxes[:, 3]) / 2)) - height_half
    
    targets = np.column_stack((x, y))
    distances = np.sqrt(np.sum(targets**2, axis=1))
    
    return targets, distances

# Load key mapping and user settings from JSON files
key_mapping, settings = load_configuration(SCRIPT_DIR)


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


def pr_light_gray(text):
    print(Fore.WHITE + text + Style.RESET_ALL)


def pr_black(text):
    print(Fore.BLACK + text + Style.RESET_ALL)


def load_configuration(script_directory):
    """Loads key mapping and user settings from JSON files."""
    with open(f"{script_directory}/configuration/key_mapping.json", "r") as f:
        key_mapping = json.load(f)

    with open(f"{script_directory}/configuration/config.json", "r") as f:
        settings = json.load(f)

    return key_mapping, settings


# --- Checkbox Event Handlers ---
def checkbox_auto_aim_event():
    settings["auto_aim"] = var_auto_aim.get()


def checkbox_trigger_bot_event():
    settings["trigger_bot"] = var_trigger_bot.get()


def checkbox_toggle_event():
    settings["toggle"] = var_toggle.get()


def checkbox_recoil_event():
    settings["recoil"] = var_recoil.get()


def checkbox_aim_shake_event():
    settings["aim_shake"] = var_aim_shake.get()


def checkbox_overlay_event():
    toggle_overlay()
    settings["overlay"] = var_overlay.get()


def checkbox_preview_event():
    settings["preview"] = var_preview.get()
    if settings["preview"] == "off":
        image_label_preview.configure(image=image_preview)


def checkbox_mask_left_event():
    settings["mask_left"] = var_mask_left.get()


def checkbox_mask_right_event():
    settings["mask_right"] = var_mask_right.get()

def checkbox_fov_event():
    settings["fov_enabled"] = var_fov.get()

# --- Slider Event Handlers ---
def slider_sensitivity_event(value):
    label_sensitivity.configure(text=f"Sensitivity: {round(float(value))}%")
    settings["sensitivity"] = float(value)


def slider_headshot_event(value):
    label_headshot.configure(text=f"Headshot offset: {round(float(value))}%")
    settings["headshot"] = float(value)


def slider_trigger_bot_event(value):
    label_trigger_bot.configure(
        text=f"Trigger bot distance: {round(float(value))} px"
    )
    settings["trigger_bot_distance"] = float(value)


def slider_confidence_event(value):
    label_confidence.configure(text=f"Confidence: {round(float(value))}%")
    settings["confidence"] = float(value)


def slider_recoil_strength_event(value):
    label_recoil_strength.configure(
        text=f"Recoil control strength: {round(float(value))}%"
    )
    settings["recoil_strength"] = float(value)


def slider_aim_shake_strength_event(value):
    label_aim_shake_strength.configure(
        text=f"Aim shake strength: {round(float(value))}%"
    )
    settings["aim_shake_strength"] = float(value)


def slider_max_move_event(value):
    label_max_move.configure(text=f"Max move speed: {round(float(value))} px")
    settings["max_move"] = float(value)


def slider_mask_width_event(value):
    label_mask_width.configure(text=f"Mask width: {round(float(value))} px")
    settings["mask_width"] = float(value)


def slider_mask_height_event(value):
    label_mask_height.configure(text=f"Mask height: {round(float(value))} px")
    settings["mask_height"] = float(value)

def slider_fov_size_event(value):
    label_fov_size.configure(text=f"FOV Size: {round(float(value))} px")
    settings["fov_size"] = float(value)


# --- Combobox Event Handlers ---
def combobox_fps_callback(choice):
    settings["max_fps"] = int(choice)
    with open(f"{SCRIPT_DIR}/configuration/config.json", "w") as json_file:
        json.dump(settings, json_file, indent=4)
    button_reload_event()


def combobox_yolo_version_callback(choice):
    models = [
        yolo_model
        for yolo_model in LAUNCHER_MODELS
        if yolo_model.startswith(combobox_yolo_version.get())
    ]
    combobox_yolo_model.configure(values=models)
    combobox_yolo_model.set(models[0])


def combobox_yolo_model_callback(choice):
    return


def combobox_yolo_model_size_callback(choice):
    return


def combobox_yolo_mode_callback(choice):
    return


def combobox_yolo_device_callback(choice):
    return


def combobox_mouse_input_callback(choice):
    settings["mouse_input"] = choice


def combobox_arduino_callback(choice):
    global arduino
    settings["arduino"] = choice
    arduino = serial.Serial(choice, 9600, timeout=1)


def combobox_mouse_activation_bind_callback(choice):
    label_activation_bind.configure(text=f"Activation key: {choice}")
    settings["activation_key_string"] = choice
    activation_key = get_keycode(settings["activation_key_string"])
    settings["activation_key"] = activation_key
    label_activation_key.configure(text=f"Activation key: {choice}")


def combobox_mouse_quit_bind_callback(choice):
    label_quit_bind.configure(text=f"Quit key: {choice}")
    settings["quit_key_string"] = choice
    quit_key = get_keycode(settings["quit_key_string"])
    settings["quit_key"] = quit_key
    label_quit_key.configure(text=f"Quit key: {settings['quit_key_string']}")


# --- Button Event Handlers ---
def button_activation_bind_event():
    activation_key = keyboard.read_event(suppress=True).name
    label_activation_bind.configure(text=f"Activation key: {activation_key}")
    settings["activation_key_string"] = activation_key
    activation_key = get_keycode(activation_key)
    settings["activation_key"] = activation_key
    label_activation_key.configure(
        text=f"Activation key: {settings['activation_key_string']}"
    )


def button_quit_bind_event():
    quit_key = keyboard.read_event(suppress=True).name
    label_quit_bind.configure(text=f"Quit key: {quit_key}")
    settings["quit_key_string"] = quit_key
    quit_key = get_keycode(quit_key)
    settings["quit_key"] = quit_key
    label_quit_key.configure(text=f"Quit key: {settings['quit_key_string']}")


def button_reload_event():
    """Reloads the YOLO model and updates UI elements based on settings."""
    global screen, overlay

    

    # Update settings from UI elements
    settings["yolo_version"] = combobox_yolo_version.get()
    settings["yolo_model"] = combobox_yolo_model.get()
    settings["yolo_mode"] = combobox_yolo_mode.get()
    settings["yolo_device"] = combobox_yolo_device.get()
    settings["height"], settings["width"] = map(
        int, combobox_yolo_model_size.get().split("x")
    )

    # Update UI elements based on settings
    slider_mask_width.configure(to=settings["width"])
    slider_mask_height.configure(to=settings["height"])
    combobox_fps.set(str(settings["max_fps"]))

    # Reset mask sliders
    slider_mask_width.set(0)
    slider_mask_height.set(0)

    # Set mask slider values, ensuring they are within bounds
    slider_mask_width.set(min(settings["mask_width"], settings["width"]))
    slider_mask_height.set(min(settings["mask_height"], settings["height"]))

    # Calculate screen capture region based on selected model size
    left = int(win32api.GetSystemMetrics(0) / 2 - settings["width"] / 2)
    top = int(win32api.GetSystemMetrics(1) / 2 - settings["height"] / 2)
    right = left + settings["width"]
    bottom = top + settings["height"]

    # Restart screen capture with updated region and FPS
    if screen is None:
        screen = bettercam.create(output_color="BGRA", max_buffer_len=512)
    else:
        screen.stop()
    screen.start(
        region=(left, top, right, bottom),
        target_fps=settings["max_fps"],
        video_mode=True,
    )

    # Load the selected YOLO model
    load_model()

    # Refresh the overlay if it's active
    if overlay is not None:
        toggle_overlay()
        toggle_overlay()


def button_keybindings_event():
    """Toggles the visibility of the keybindings window."""
    if keybindings.state() == "withdrawn":
        keybindings.deiconify()
        keybindings.geometry(f"400x160+{root.winfo_x() + 40}+{root.winfo_y() + 100}")
        keybindings.focus()
        keybindings.attributes("-topmost", True)
    else:
        keybindings.withdraw()


# --- Helper Functions ---
def toggle_overlay():
    """Toggles the visibility of the overlay."""
    global overlay, canvas
    if overlay is None:
        left = int(win32api.GetSystemMetrics(0) / 2 - settings["width"] / 2)
        top = int(win32api.GetSystemMetrics(1) / 2 - settings["height"] / 2)
        overlay = ctk.CTkToplevel()
        overlay.geometry(f"{settings['width']}x{settings['height']}+{left}+{top}")
        overlay.overrideredirect(True)
        overlay.config(bg="#000000")
        overlay.attributes("-alpha", 0.75)
        overlay.wm_attributes("-topmost", 1)
        overlay.attributes("-transparentcolor", "#000000", "-topmost", 1)
        overlay.resizable(False, False)
        set_clickthrough(overlay.winfo_id())
        canvas = ctk.CTkCanvas(
            overlay, width=640, height=640, bg="black", highlightbackground="white"
        )
        canvas.pack()
    else:
        overlay.destroy()
        canvas.destroy()
        overlay = None
        canvas = None


def update_overlay():
    """Updates the overlay with detected object bounding boxes."""
    overlay.update()
    if coordinates and canvas:
        canvas.delete("all")
        for coord in coordinates:
            x_min, y_min, x_max, y_max = map(int, coord)
            canvas.create_rectangle(
                x_min, y_min, x_max, y_max, outline="white", width=2
            )


def update_preview(frame):
    """Updates the preview window with detected objects and aim points."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if distances.size > 0:
        min_distance_index = np.argmin(distances)
        half_width = settings["width"] // 2
        half_height = settings["height"] // 2
        for i, coord in enumerate(coordinates):
            x_min, y_min, x_max, y_max = map(int, coord)
            color = (255, 255, 0) if i == min_distance_index else (0, 0, 255)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        
        for i, target in enumerate(targets):
            target_x, target_y = target
            color = (255, 255, 0) if i == min_distance_index else (0, 0, 255)
            cv2.line(frame, 
                     (half_width, half_height),
                     (int(target_x + half_width), int(target_y + half_height)),
                     color, 4)
            cv2.circle(frame,
                       (int(target_x + half_width), int(target_y + half_height)),
                       8, (0, 255, 0), -1)

    frame = Image.fromarray(cv2.resize(frame, (240, 240), interpolation=cv2.INTER_NEAREST))
    ctkframe = ctk.CTkImage(size=(240, 240), dark_image=frame, light_image=frame)
    image_label_preview.configure(image=ctkframe)


def set_clickthrough(hwnd):
    """Makes a window click-through."""
    styles = win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, styles)
    win32gui.SetLayeredWindowAttributes(hwnd, 0, 255, 0x00000001)


def get_keycode(key):
    """Gets the virtual key code for a given key."""
    return key_mapping.get(key) or win32api.VkKeyScan(key)


def update_aim_shake():
    """Updates random aim shake offsets."""
    global random_x, random_y
    if settings["aim_shake"] == "on":
        aim_shake_strength = int(settings["aim_shake_strength"])
        random_x = random.randint(-aim_shake_strength, aim_shake_strength)
        random_y = random.randint(-aim_shake_strength, aim_shake_strength)
    else:
        random_x = 0
        random_y = 0


def get_mode():
    """Returns the YOLO model inference mode based on settings."""
    mode = settings["yolo_mode"]
    return "engine" if mode in ("pytorch", "onnx", "tensorrt") else None


def get_model_name():
    """Returns the filename of the YOLO model based on settings."""
    if settings["yolo_mode"] == "pytorch":
        return f"{settings['yolo_model']}.pt"
    return f"{settings['yolo_model']}{settings['yolo_version']}{settings['height']}{settings['width']}Half.{get_mode()}"


def model_existence_check():
    """Checks if the selected YOLO model exists and exports it if not."""
    if settings["yolo_mode"] != "pytorch" and not os.path.exists(
        f"{MODELS_PATH}/{get_model_name()}"
    ):
        pr_cyan(
            f"Exporting {settings['yolo_model']}.pt to {get_model_name()} with yolo{settings['yolo_version']}."
        )
        export_model()


def load_model():
    """Loads the selected YOLO model for inference."""
    global model
    model_existence_check()
    pr_blue(
        f"Loading {get_model_name()} with yolo{settings['yolo_version']} for {settings['yolo_mode']} inference."
    )

    if settings["yolo_mode"] == "onnx":
        onnx_provider = {
            "cpu": "CPUExecutionProvider",
            "amd": "DmlExecutionProvider",
            "nvidia": "CUDAExecutionProvider",
        }.get(settings["yolo_device"])

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        model = ort.InferenceSession(
            f"{MODELS_PATH}/{get_model_name()}",
            sess_options=so,
            providers=[onnx_provider],
        )

    else:  # PyTorch or TensorRT
        model_path = f"{MODELS_PATH}/{get_model_name()}"
        if settings["yolo_version"] == "v8":
            model = YOLO(model_path, task="detect")
        elif settings["yolo_version"] == "v5":
            model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=model_path,
                verbose=False,
                trust_repo=True,
                force_reload=True,
            )

    pr_green("Model loaded.")


def export_model():
    """Exports the selected YOLO model to the specified inference mode."""
    temp_model_path = f"{SCRIPT_DIR}/temp.pt"
    shutil.copy(f"{MODELS_PATH}/{settings['yolo_model']}.pt", temp_model_path)
    height = int(settings["height"])
    width = int(settings["width"])
    if settings["yolo_version"] == "v8":
        x_model = YOLO(temp_model_path)
        x_model.export(
            format=get_mode(), imgsz=[height, width], half=True, device=0
        )
    elif settings["yolo_version"] == "v5":
        os.system(
            f"python {SCRIPT_DIR}/yolov5/export.py --weights {temp_model_path} --include {get_mode()} --imgsz {height} {width} --half --device 0"
        )
    os.remove(temp_model_path)
    if settings["yolo_mode"] == "tensorrt":
        os.remove(f"{SCRIPT_DIR}/temp.onnx")
        shutil.move(
            f"{SCRIPT_DIR}/temp.engine", f"{MODELS_PATH}/{get_model_name()}"
        )
    else:
        shutil.move(
            f"{SCRIPT_DIR}/temp.onnx", f"{MODELS_PATH}/{get_model_name()}"
        )
    pr_green("Export complete.")


def mask_frame(frame):
    """Masks out specified regions of the frame."""
    if settings["mask_left"] == "on":
        frame[
            int(settings["height"] - settings["mask_height"]) : settings["height"],
            0 : int(settings["mask_width"]),
            :,
        ] = 0
    if settings["mask_right"] == "on":
        frame[
            int(settings["height"] - settings["mask_height"]) : settings["height"],
            int(settings["width"] - settings["mask_width"]) : settings["width"],
            :,
        ] = 0
    return frame


def extract_original_name(name):
    """Extracts the original model name from a formatted filename."""
    match = re.match(r"^(.+?)v\d{7}Half$", name)
    return match.group(1) if match else name


def mouse_click():
    """Simulates a mouse click."""
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def mouse_move(move_x, move_y, click):
    """Moves the mouse or sends movement commands to Arduino."""
    if settings["mouse_input"] == "arduino":
        arduino.write(f"{move_x}:{move_y}:{click}x".encode())
    else:
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, move_x, move_y, 0, 0)
        if click == 1:
            mouse_click()


def calculate_targets(x1, y1, x2, y2):
    """Calculates aim targets and distances from bounding box coordinates."""
    width_half = settings["width"] / 2
    height_half = settings["height"] / 2
    headshot_percent = settings["headshot"] / 100
    x = int(((x1 + x2) / 2) - width_half)
    y = int(
        ((y1 + y2) / 2)
        + headshot_percent * (y1 - ((y1 + y2) / 2))
        - height_half
    )
    distance = math.sqrt(x**2 + y**2)
    return (x, y), distance


# --- UI Setup ---
root = ctk.CTk()
root.title("Spawn-Aim")
root.geometry("600x850+40+40")
root.resizable(width=False, height=False)

# --- UI Element Creation and Placement ---
# Auto aim checkbox
var_auto_aim = ctk.StringVar(value="off")
var_trigger_bot = ctk.StringVar(value="off")
var_toggle = ctk.StringVar(value="off")
var_recoil = ctk.StringVar(value="off")
var_aim_shake = ctk.StringVar(value="off")
var_overlay = ctk.StringVar(value="off")
var_preview = ctk.StringVar(value="off")
var_mask_left = ctk.StringVar(value="off")
var_mask_right = ctk.StringVar(value="off")
var_fov = ctk.StringVar(value="off")


checkbox_auto_aim = ctk.CTkCheckBox(
    root,
    text="Auto aim",
    variable=var_auto_aim,
    onvalue="on",
    offvalue="off",
    command=checkbox_auto_aim_event,
    fg_color=GREEN_COLOR,
    hover_color="#0F0"
)
checkbox_auto_aim.place(x=10, y=10)

# Trigger bot checkbox
var_trigger_bot = ctk.StringVar(value="off")
checkbox_trigger_bot = ctk.CTkCheckBox(
    root,
    text="Trigger bot",
    variable=var_trigger_bot,
    onvalue="on",
    offvalue="off",
    command=checkbox_trigger_bot_event,
    fg_color=GREEN_COLOR,
    hover_color="#0F0"
)
checkbox_trigger_bot.place(x=10, y=40)
# Auto aim toggle checkbox
var_toggle = ctk.StringVar(value="off")
checkbox_toggle = ctk.CTkCheckBox(
    root,
    text="Auto aim toggle",
    variable=var_toggle,
    onvalue="on",
    offvalue="off",
    command=checkbox_toggle_event,
    fg_color=GREEN_COLOR,
    hover_color="#0F0"
)
checkbox_toggle.place(x=110, y=10)

# Recoil control checkbox
checkbox_recoil = ctk.CTkCheckBox(
    root,
    text="Recoil control",
    variable=var_recoil,
    onvalue="on",
    offvalue="off",
    command=checkbox_recoil_event,
    fg_color=GREEN_COLOR,
    hover_color="#0F0"
)
checkbox_recoil.place(x=110, y=40)

# Aim shake checkbox
checkbox_aim_shake = ctk.CTkCheckBox(
    root,
    text="Aim shake",
    variable=var_aim_shake,
    onvalue="on",
    offvalue="off",
    command=checkbox_aim_shake_event,
    fg_color=GREEN_COLOR,
    hover_color="#0F0"
)
checkbox_aim_shake.place(x=10, y=70)

# Overlay checkbox
checkbox_overlay = ctk.CTkCheckBox(
    root,
    text="Overlay",
    variable=var_overlay,
    onvalue="on",
    offvalue="off",
    command=checkbox_overlay_event,
    fg_color=GREEN_COLOR,
    hover_color="#0F0"
)
checkbox_overlay.place(x=110, y=70)

# Sensitivity slider
label_sensitivity = ctk.CTkLabel(root, text="Sensitivity: 0%", text_color=GREY_COLOR)
label_sensitivity.place(x=10, y=100)
slider_sensitivity = ctk.CTkSlider(
    root, from_=0, to=100, command=slider_sensitivity_event, fg_color=GREEN_COLOR
)
slider_sensitivity.place(x=10, y=125)

# Headshot offset slider
label_headshot = ctk.CTkLabel(root, text="Headshot offset: 0%", text_color=GREY_COLOR)
label_headshot.place(x=10, y=150)
slider_headshot = ctk.CTkSlider(root, from_=0, to=100, command=slider_headshot_event, fg_color=GREEN_COLOR)
slider_headshot.place(x=10, y=175)

# Trigger bot distance slider
label_trigger_bot = ctk.CTkLabel(root, text="Trigger bot distance: 0 px", text_color=GREY_COLOR)
label_trigger_bot.place(x=10, y=200)
slider_trigger_bot = ctk.CTkSlider(
    root, from_=0, to=100, command=slider_trigger_bot_event, fg_color=GREEN_COLOR
)
slider_trigger_bot.place(x=10, y=225)

# Confidence slider
label_confidence = ctk.CTkLabel(root, text="Confidence: 0%", text_color=GREY_COLOR)
label_confidence.place(x=10, y=250)
slider_confidence = ctk.CTkSlider(
    root, from_=0, to=100, command=slider_confidence_event, fg_color=GREEN_COLOR
)
slider_confidence.place(x=10, y=275)

# Recoil control strength slider
label_recoil_strength = ctk.CTkLabel(root, text="Recoil control strength: 0%", text_color=GREY_COLOR)
label_recoil_strength.place(x=10, y=300)
slider_recoil_strength = ctk.CTkSlider(
    root, from_=0, to=100, command=slider_recoil_strength_event, fg_color=GREEN_COLOR
)
slider_recoil_strength.place(x=10, y=325)

# Aim shake strength slider
label_aim_shake_strength = ctk.CTkLabel(root, text="Aim shake strength: 0%", text_color=GREY_COLOR)
label_aim_shake_strength.place(x=10, y=350)
slider_aim_shake_strength = ctk.CTkSlider(
    root, from_=0, to=100, command=slider_aim_shake_strength_event, fg_color=GREEN_COLOR
)
slider_aim_shake_strength.place(x=10, y=375)

# Max move speed slider
label_max_move = ctk.CTkLabel(root, text="Max move speed: 0 px", text_color=GREY_COLOR)
label_max_move.place(x=10, y=400)
slider_max_move = ctk.CTkSlider(root, from_=0, to=100, command=slider_max_move_event, fg_color=GREEN_COLOR)
slider_max_move.place(x=10, y=425)

# FPS combobox
label_fps = ctk.CTkLabel(root, text="FPS:")
label_fps.place(x=10, y=720)
combobox_fps = ctk.CTkComboBox(
    root,
    values=["30", "60", "90", "120", "144", "165", "180"],
    command=combobox_fps_callback,
    state="readonly",
)
combobox_fps.place(x=10, y=750)
combobox_fps.set(str(settings["max_fps"]))

# YOLO version combobox
label_yolo_version = ctk.CTkLabel(root, text="Yolo version:")
label_yolo_version.place(x=10, y=600)
combobox_yolo_version = ctk.CTkComboBox(
    root, values=["v8", "v5"], command=combobox_yolo_version_callback, state="readonly"
)
combobox_yolo_version.place(x=10, y=630)

# YOLO model combobox
label_yolo_model = ctk.CTkLabel(root, text="Yolo model:")
label_yolo_model.place(x=10, y=660)
combobox_yolo_model = ctk.CTkComboBox(
    root, values=["Default"], command=combobox_yolo_model_callback, state="readonly"
)
combobox_yolo_model.place(x=10, y=690)

# Inference mode combobox
label_yolo_mode = ctk.CTkLabel(root, text="Inference mode:")
label_yolo_mode.place(x=160, y=600)
combobox_yolo_mode = ctk.CTkComboBox(
    root,
    values=["pytorch", "onnx", "tensorrt"],
    command=combobox_yolo_mode_callback,
    state="readonly"
)
combobox_yolo_mode.place(x=160, y=630)

# Device combobox
label_yolo_device = ctk.CTkLabel(root, text="Device:")
label_yolo_device.place(x=160, y=660)
combobox_yolo_device = ctk.CTkComboBox(
    root,
    values=["cpu", "amd", "nvidia"],
    command=combobox_yolo_device_callback,
    state="readonly",
)
combobox_yolo_device.place(x=160, y=690)

# Model size combobox
label_model_size = ctk.CTkLabel(root, text="Model size:")
label_model_size.place(x=310, y=600)
combobox_yolo_model_size = ctk.CTkComboBox(
    root,
    values=["160x160", "320x320", "480x480", "640x640"],
    command=combobox_yolo_model_size_callback,
    state="readonly",
)
combobox_yolo_model_size.place(x=310, y=630)

# Reload model button
button_reload = ctk.CTkButton(root, text="Reload model", command=button_reload_event)
button_reload.place(x=310, y=690)

# Configure keybindings button
button_keybindings = ctk.CTkButton(
    root, text="Configure keybindings", command=button_keybindings_event
)
button_keybindings.place(x=10, y=575)

# Activation key label
label_activation_key = ctk.CTkLabel(root, text="Activation key: None")
label_activation_key.place(x=10, y=455)

# Quit key label
label_quit_key = ctk.CTkLabel(root, text="Quit key: None", text_color=GREY_COLOR)
label_quit_key.place(x=10, y=485)

# Preview checkbox
checkbox_preview = ctk.CTkCheckBox(
    root,
    text="Preview",
    variable=var_preview,
    onvalue="on",
    offvalue="off",
    command=checkbox_preview_event,
    fg_color=GREEN_COLOR,
    hover_color="#0F0"
)
checkbox_preview.place(x=320, y=10)

# FPS label
label_fps = ctk.CTkLabel(root, text="Fps:", text_color=GREY_COLOR)
label_fps.place(x=240, y=10)

# Preview image label
image_preview = ctk.CTkImage(
    size=(240, 240),
    dark_image=Image.open(f"{SCRIPT_DIR}/preview.png"),
    light_image=Image.open(f"{SCRIPT_DIR}/preview.png"),
)
image_label_preview = ctk.CTkLabel(root, image=image_preview, text="", text_color=GREY_COLOR)
image_label_preview.place(x=240, y=40)

# Mask left checkbox
checkbox_mask_left = ctk.CTkCheckBox(
    root,
    text="Mask left",
    variable=var_mask_left,
    onvalue="on",
    offvalue="off",
    command=checkbox_mask_left_event,
    fg_color=GREEN_COLOR,
    hover_color="#0F0"
)
checkbox_mask_left.place(x=240, y=290)

# Mask right checkbox
var_mask_right = ctk.StringVar(value="off")
checkbox_mask_right = ctk.CTkCheckBox(
    root,
    text="Mask right",
    variable=var_mask_right,
    onvalue="on",
    offvalue="off",
    command=checkbox_mask_right_event,
    fg_color=GREEN_COLOR,
    hover_color="#0F0"
)
checkbox_mask_right.place(x=380, y=290)

# Mask width slider
label_mask_width = ctk.CTkLabel(root, text="Mask width: 0 px")
label_mask_width.place(x=240, y=320)
slider_mask_width = ctk.CTkSlider(
    root, from_=0, to=640, command=slider_mask_width_event, fg_color=GREEN_COLOR
)
slider_mask_width.place(x=240, y=345)

# Mask height slider
label_mask_height = ctk.CTkLabel(root, text="Mask height: 0 px")
label_mask_height.place(x=240, y=370)
slider_mask_height = ctk.CTkSlider(
    root, from_=0, to=640, command=slider_mask_height_event, fg_color=GREEN_COLOR
)
slider_mask_height.place(x=240, y=395)

# Mouse input method combobox
label_mouse_input = ctk.CTkLabel(root, text="Mouse input method:")
label_mouse_input.place(x=310, y=420)
combobox_mouse_input = ctk.CTkComboBox(
    root, values=["default"], command=combobox_mouse_input_callback, state="readonly"
)
combobox_mouse_input.place(x=310, y=450)

# Arduino port combobox
label_mouse_input = ctk.CTkLabel(root, text="Arduino port:")
label_mouse_input.place(x=310, y=480)
combobox_arduino = ctk.CTkComboBox(
    root, values=["COM1"], command=combobox_arduino_callback, state="readonly"
)
combobox_arduino.place(x=310, y=510)

# FOV Checkbox
checkbox_fov = ctk.CTkCheckBox(
    root,
    text="Enable FOV",
    variable=var_fov,
    onvalue="on",
    offvalue="off",
    command=checkbox_fov_event,
    fg_color=GREEN_COLOR,
    hover_color="#0F0",
)
checkbox_fov.place(x=10, y=510)

# FOV Size slider
label_fov_size = ctk.CTkLabel(root, text="FOV Size: 0 px", text_color=GREY_COLOR)
label_fov_size.place(x=10, y=540)
slider_fov_size = ctk.CTkSlider(
    root, from_=0, to=200, command=slider_fov_size_event, fg_color=GREEN_COLOR
)
slider_fov_size.place(x=10, y=565)



# --- Keybindings Window ---
keybindings = ctk.CTkToplevel()
keybindings.title("Spawn-Aim Keybinder")
keybindings.geometry(f"400x160+{root.winfo_x() + 40}+{root.winfo_y() + 100}")
keybindings.resizable(width=False, height=False)
keybindings.withdraw()
keybindings.protocol("WM_DELETE_WINDOW", lambda: keybindings.withdraw())
keybindings.bind("<Unmap>", lambda event: keybindings.withdraw())

# Configure activation key button
button_activation_bind = ctk.CTkButton(
    keybindings, text="Configure activation key", command=button_activation_bind_event
)
button_activation_bind.place(x=10, y=10)

# Configure quit key button
button_quit_bind = ctk.CTkButton(
    keybindings, text="Configure quit key", command=button_quit_bind_event
)
button_quit_bind.place(x=10, y=50)

# Activation key label
label_activation_bind = ctk.CTkLabel(
    keybindings, text=f"Activation key: {settings['activation_key_string']}", text_color=GREY_COLOR
)
label_activation_bind.place(x=10, y=90)

# Quit key label
label_quit_bind = ctk.CTkLabel(
    keybindings, text=f"Quit key: {settings['quit_key_string']}", text_color=GREY_COLOR
)
label_quit_bind.place(x=10, y=120)

# Mouse activation key combobox
combobox_mouse_activation_bind = ctk.CTkComboBox(
    keybindings,
    values=list(key_mapping.keys()),
    command=combobox_mouse_activation_bind_callback,
    state="readonly",
)
combobox_mouse_activation_bind.place(x=200, y=10)

# Mouse quit key combobox
combobox_mouse_quit_bind = ctk.CTkComboBox(
    keybindings,
    values=list(key_mapping.keys()),
    command=combobox_mouse_quit_bind_callback,
    state="readonly",
)
combobox_mouse_quit_bind.place(x=200, y=50)

combobox_mouse_activation_bind.set("Select special")
combobox_mouse_quit_bind.set("Select special")


def main(**argv):
    """Main function for the Spawn-Aim program."""
    global model, screen, settings, overlay, canvas

    # Print welcome message and instructions
    pr_purple(
        """
  ██████  ██▓███   ▄▄▄      █     █░ ███▄    █      ▄▄▄       ██▓ ███▄ ▄███▓ ▄▄▄▄    ▒█████  ▄▄▄█████▓
▒██    ▒ ▓██░  ██ ▒████▄   ▓█░ █ ░█░ ██ ▀█   █     ▒████▄   ▒▓██▒▓██▒▀█▀ ██▒▓█████▄ ▒██▒  ██▒▓  ██▒ ▓▒
░ ▓██▄   ▓██░ ██▓▒▒██  ▀█▄ ▒█░ █ ░█ ▓██  ▀█ ██▒    ▒██  ▀█▄ ▒▒██▒▓██    ▓██░▒██▒ ▄██▒██░  ██▒▒ ▓██░ ▒░
  ▒   ██▒▒██▄█▓▒ ▒░██▄▄▄▄██░█░ █ ░█ ▓██▒  ▐▌██▒    ░██▄▄▄▄██░░██░▒██    ▒██ ▒██░█▀  ▒██   ██░░ ▓██▓ ░ 
▒██████▒▒▒██▒ ░  ░▒▓█   ▓██░░██▒██▓ ▒██░   ▓██░     ▓█   ▓██░░██░▒██▒   ░██▒░▓█  ▀█▓░ ████▓▒░  ▒██▒ ░ 
▒ ▒▓▒ ▒ ░▒▓▒░ ░  ░░▒▒   ▓▒█░ ▓░▒ ▒  ░ ▒░   ▒ ▒      ▒▒   ▓▒█ ░▓  ░ ▒░   ░  ░░▒▓███▀▒░ ▒░▒░▒░   ▒ ░░   
░ ░▒  ░  ░▒ ░     ░ ░   ▒▒   ▒ ░ ░  ░ ░░   ░ ▒░      ░   ▒▒ ░ ▒ ░░  ░      ░▒░▒   ░   ░ ▒ ▒░     ░    
░  ░  ░  ░░         ░   ▒    ░   ░     ░   ░ ░       ░   ▒  ░ ▒ ░░      ░    ░    ░ ░ ░ ░ ▒    ░      
      ░                 ░      ░             ░           ░    ░         ░    ░          ░ ░"""
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

    # Define available mouse input methods
    mouse_inputs = ["default", "arduino"]

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
    settings.update(
        {
            "auto_aim": "on",
            "trigger_bot": "on" if launcher_settings["autoFire"] else "off",
            "toggle": "on" if launcher_settings["toggleable"] else "off",
            "recoil": "off",
            "aim_shake": "on" if launcher_settings["aimShakey"] else "off",
            "overlay": "off",
            "preview": "on" if launcher_settings["visuals"] else "off",
            "mask_left": "on" if launcher_settings["maskLeft"] and launcher_settings["useMask"] else "off",
            "mask_right": "on" if not launcher_settings["maskLeft"] and launcher_settings["useMask"] else "off",
            "sensitivity": launcher_settings["movementAmp"] * 100,
            "headshot": launcher_settings["headshotDistanceModifier"] * 100
            if launcher_settings["headshotMode"]
            else 40,
            "trigger_bot_distance": launcher_settings[
                "autoFireActivationDistance"
            ],
            "confidence": launcher_settings["confidence"] * 100,
            "recoil_strength": 0,
            "aim_shake_strength": launcher_settings["aimShakeyStrength"],
            "max_move": 100,
            "height": launcher_settings["screenShotHeight"],
            "width": launcher_settings["screenShotHeight"],
            "mask_width": launcher_settings["maskWidth"],
            "mask_height": launcher_settings["maskHeight"],
            "yolo_version": f"v{argv['yoloVersion']}",
            "yolo_model": "v5_Fortnite_taipeiuser",
            "yolo_mode": "tensorrt",
            "yolo_device": {1: "cpu", 2: "amd", 3: "nvidia"}.get(
                launcher_settings["onnxChoice"]
            ),
            "activation_key": activation_key,
            "quit_key": quit_key,
            "activation_key_string": launcher_settings["activationKey"],
            "quit_key_string": launcher_settings["quitKey"],
            "mouse_input": "default",
            "arduino": default_port,
            "fov_enabled": "on" if launcher_settings["fovToggle"] else "off",
            "fov_size": launcher_settings["fovSize"],
        }
    )

    # Update UI elements based on settings
    var_auto_aim.set(settings["auto_aim"])
    var_trigger_bot.set(settings["trigger_bot"])
    var_toggle.set(settings["toggle"])
    var_recoil.set(settings["recoil"])
    var_aim_shake.set(settings["aim_shake"])
    var_overlay.set(settings["overlay"])
    var_preview.set(settings["preview"])
    var_mask_left.set(settings["mask_left"])
    var_mask_right.set(settings["mask_right"])
    var_fov.set(settings["fov_enabled"])
    label_sensitivity.configure(
        text=f"Sensitivity: {round(settings['sensitivity'])}%"
    )
    slider_sensitivity.set(settings["sensitivity"])
    label_headshot.configure(text=f"Headshot offset: {round(settings['headshot'])}%")
    slider_headshot.set(settings["headshot"])
    label_trigger_bot.configure(
        text=f"Trigger bot distance: {round(settings['trigger_bot_distance'])} px"
    )
    slider_trigger_bot.set(settings["trigger_bot_distance"])
    label_confidence.configure(
        text=f"Confidence: {round(settings['confidence'])}%"
    )
    slider_confidence.set(settings["confidence"])
    label_recoil_strength.configure(
        text=f"Recoil control strength: {round(settings['recoil_strength'])}%"
    )
    slider_recoil_strength.set(settings["recoil_strength"])
    label_aim_shake_strength.configure(
        text=f"Aim shake strength: {round(settings['aim_shake_strength'])}%"
    )
    slider_aim_shake_strength.set(settings["aim_shake_strength"])
    label_max_move.configure(
        text=f"Max move speed: {round(settings['max_move'])} px"
    )
    slider_max_move.set(settings["max_move"])
    combobox_yolo_model_size.set(f"{settings['height']}x{settings['width']}")
    label_mask_width.configure(
        text=f"Mask width: {round(settings['mask_width'])} px"
    )
    slider_mask_width.set(settings["mask_width"])
    label_mask_height.configure(
        text=f"Mask height: {round(settings['mask_height'])} px"
    )
    slider_mask_height.set(settings["mask_height"])
    combobox_yolo_version.set(settings["yolo_version"])
    combobox_yolo_model.configure(
        values=[
            yolo_model
            for yolo_model in LAUNCHER_MODELS
            if yolo_model.startswith(combobox_yolo_version.get())
        ]
    )
    combobox_yolo_model.set(settings["yolo_model"])
    combobox_yolo_mode.set(settings["yolo_mode"])
    combobox_yolo_device.set(settings["yolo_device"])
    label_activation_key.configure(
        text=f"Activation key: {settings['activation_key_string']}"
    )
    label_quit_key.configure(text=f"Quit key: {settings['quit_key_string']}")
    combobox_fps.set(str(settings["max_fps"]))
    combobox_mouse_input.configure(values=mouse_inputs)
    combobox_mouse_input.set(settings["mouse_input"])
    combobox_arduino.configure(values=ports)
    combobox_arduino.set(settings["arduino"])
    label_fov_size.configure(text=f"FOV Size: {settings['fov_size']} px")
    slider_fov_size.set(settings["fov_size"])

    # Enable UI elements
    for widget in (
        checkbox_trigger_bot,
        checkbox_toggle,
        checkbox_recoil,
        checkbox_aim_shake,
        checkbox_overlay,
        checkbox_preview,
        checkbox_mask_left,
        checkbox_mask_right,
        checkbox_fov,
        slider_sensitivity,
        slider_headshot,
        slider_trigger_bot,
        slider_confidence,
        slider_recoil_strength,
        slider_aim_shake_strength,
        slider_max_move,
        slider_mask_width,
        slider_mask_height,
        slider_fov_size,
        button_keybindings,
        button_reload,
        combobox_mouse_input,
        combobox_arduino,
        combobox_yolo_version,
        combobox_yolo_model,
        combobox_yolo_mode,
        combobox_yolo_device,
        combobox_yolo_model_size,
    ):
        widget.configure(state="normal")

    # Load the YOLO model and initialize overlay if enabled
    button_reload_event()

    if settings["overlay"] == "on":
        toggle_overlay()

    # Initialize variables for FPS calculation and auto aim toggle
    start_time = time.time()
    frame_count = 0
    pressing = False
    # Pre-allocate arrays for object detection results
    max_detections = 4  # Set this to the maximum number of detections expected
    boxes = np.zeros((max_detections, 4))
    confs = np.zeros((max_detections,))
    classes = np.zeros((max_detections,))

    # --- Main Loop ---
    while True:
        # Update UI and get the latest frame from screen capture
        root.update()
        frame_count += 1
        np_frame = np.array(screen.get_latest_frame())
        pygame.event.pump()

        # Initialize arrays for each iteration
        targets = np.empty((0, 2), dtype=np.float32)
        distances = np.empty(0, dtype=np.float32)
        coordinates = np.empty((0, 4), dtype=np.float32)

        # Convert frame to RGB format if necessary
        if np_frame.shape[2] == 4:
            np_frame = np_frame[:, :, :3]

        # --- Object Detection and Aim Assist ---
        with torch.no_grad():
            # Apply frame masking if enabled
            frame = mask_frame(np_frame)

            # Perform object detection using the selected YOLO model and mode
            if settings["yolo_mode"] in ("pytorch", "tensorrt"):
                if settings["yolo_version"] == "v5":
                    model.conf = settings["confidence"] / 100
                    model.iou = settings["confidence"] / 100
                    results = model(frame, size=[settings["height"], settings["width"]])
                    if len(results.xyxy[0]) != 0:
                        num_detections = len(results.xyxy[0])
                        # Check if the number of detections exceeds the pre-allocated size
                        if num_detections > max_detections:
                            # Resize the arrays if needed
                            max_detections = num_detections
                            boxes = np.resize(boxes, (max_detections, 4))
                            confs = np.resize(confs, max_detections)
                            classes = np.resize(classes, max_detections)
                        # Copy the detection results to the pre-allocated arrays
                        boxes[:num_detections] = results.xyxy[0][:, :4].cpu().numpy()
                        confs[:num_detections] = results.xyxy[0][:, 4].cpu().numpy()
                        classes[:num_detections] = results.xyxy[0][:, 5].cpu().numpy()
                        coordinates = boxes[:num_detections]
                        targets, distances = calculate_targets_vectorized(boxes[:num_detections])

                        if settings["fov_enabled"] == "on":
                            fov_mask = np.sum(targets**2, axis=1) <= settings["fov_size"]**2
                            targets = targets[fov_mask]
                            distances = distances[fov_mask]
                            coordinates = coordinates[fov_mask]

                elif settings["yolo_version"] == "v8":
                    results = model.predict(
                        frame,
                        verbose=False,
                        conf=settings["confidence"] / 100,
                        iou=settings["confidence"] / 100,
                        half=False,
                        imgsz=[settings["height"], settings["width"]],
                    )
                    for result in results:
                        if len(result.boxes.xyxy) != 0:
                            num_detections = len(result.boxes.xyxy)
                            # Check if the number of detections exceeds the pre-allocated size
                            if num_detections > max_detections:
                                # Resize the arrays if needed
                                max_detections = num_detections
                                boxes = np.resize(boxes, (max_detections, 4))
                                confs = np.resize(confs, max_detections)
                                classes = np.resize(classes, max_detections)
                            # Copy the detection results to the pre-allocated arrays
                            boxes[:num_detections] = result.boxes.xyxy.cpu().numpy()
                            coordinates = boxes[:num_detections]
                            targets, distances = calculate_targets_vectorized(boxes[:num_detections])

                            if settings["fov_enabled"] == "on":
                                fov_mask = np.sum(targets**2, axis=1) <= settings["fov_size"]**2
                                targets = targets[fov_mask]
                                distances = distances[fov_mask]
                                coordinates = coordinates[fov_mask]

            elif settings["yolo_mode"] == "onnx":
                if settings["yolo_device"] == "nvidia":
                    frame = torch.from_numpy(frame).to("cuda")

                    frame = torch.movedim(frame, 2, 0)
                    frame = frame.half()
                    frame /= 255
                    if len(frame.shape) == 3:
                        frame = frame[None]
                else:
                    frame = frame / 255
                    frame = frame.astype(np.half)
                    frame = np.moveaxis(frame, 3, 1)

                if settings["yolo_device"] == "nvidia":
                    outputs = model.run(None, {"images": cp.asnumpy(frame)})
                else:
                    outputs = model.run(None, {"images": np.array(frame)})

                frame = torch.from_numpy(outputs[0])

                if settings["yolo_version"] == "v5":
                    predictions = non_max_suppression(
                        frame,
                        settings["confidence"] / 100,
                        settings["confidence"] / 100,
                        0,
                        False,
                        max_det=4,
                    )

                elif settings["yolo_version"] == "v8":
                    predictions = ops.non_max_suppression(
                        frame,
                        settings["confidence"] / 100,
                        settings["confidence"] / 100,
                        0,
                        False,
                        max_det=4,
                    )

                for i, det in enumerate(predictions):
                    if len(det):
                        num_detections = len(det)
                        # Check if the number of detections exceeds the pre-allocated size
                        if num_detections > max_detections:
                            # Resize the arrays if needed
                            max_detections = num_detections
                            boxes = np.resize(boxes, (max_detections, 4))
                            confs = np.resize(confs, max_detections)
                            classes = np.resize(classes, max_detections)
                        # Copy the detection results to the pre-allocated arrays
                        boxes[:num_detections] = det[:, :4].cpu().numpy()
                        confs[:num_detections] = det[:, 4].cpu().numpy()
                        classes[:num_detections] = det[:, 5].cpu().numpy()
                        coordinates = boxes[:num_detections]
                        targets, distances = calculate_targets_vectorized(boxes[:num_detections])

                        if settings["fov_enabled"] == "on":
                            fov_mask = np.sum(targets**2, axis=1) <= settings["fov_size"]**2
                            targets = targets[fov_mask]
                            distances = distances[fov_mask]
                            coordinates = coordinates[fov_mask]
            # Send aim assist commands based on detected targets
            send_targets(
                controller,
                settings,
                targets,
                distances,
                random_x,
                random_y,
                get_left_trigger,
                get_right_trigger,
            )

        # Update preview window if enabled
        if settings["preview"] == "on":
            update_preview(np_frame)

        # Update overlay if enabled
        if settings["overlay"] == "on":
            update_overlay()

        # Calculate and update FPS label
        elapsed_time = time.time() - start_time
        if elapsed_time >= 0.2:
            label_fps.configure(text=f"Fps: {round(frame_count / elapsed_time)}")
            frame_count = 0
            start_time = time.time()
            update_aim_shake()

        # Toggle auto aim based on the left trigger of the Xbox controller
        if settings["toggle"] == "on":
            trigger_value = get_left_trigger()
            if trigger_value > 0.5 and not pressing:
                pressing = True
                checkbox_auto_aim.toggle()
            elif trigger_value <= 0.5 and pressing:
                pressing = False

        # Quit the program if the quit key is pressed
        if win32api.GetKeyState(settings["quit_key"]) in (-127, -128):
            pr_green("Goodbye!")
            screen.stop()
            screen.release()
            quit()

        # Clear arrays efficiently
        targets = targets[:0]
        distances = distances[:0] 
        coordinates = coordinates[:0]


if __name__ == "__main__":
    main(settingsProfile="config", yoloVersion=5, version=0)