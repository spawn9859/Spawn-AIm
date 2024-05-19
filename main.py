from colorama import Fore, Back, Style
from ultralytics.utils import ops
from PIL import Image, ImageTk
import serial.tools.list_ports
from ultralytics import YOLO
import customtkinter as ctk
import onnxruntime as ort
import numpy as np
import bettercam
import keyboard
import pygame
import win32api
import win32con
import win32gui
import shutil
import random
import serial
import torch
import json
import time
import math
import cv2
import sys
import os
import re

script_directory = os.path.dirname(os.path.abspath(__file__ if "__file__" in locals() else __file__))
if not os.path.exists(f"{script_directory}/yolov5"):
    os.system(f"git clone https://github.com/ultralytics/yolov5 {script_directory}/yolov5")
    os.system(f"pip install -r {script_directory}/yolov5/requirements.txt")
sys.path.append(f"{script_directory}/yolov5")

from utils.general import non_max_suppression

if torch.cuda.is_available():
    import cupy as cp

with open(f"{script_directory}/configuration/key_mapping.json", 'r') as json_file:
    key_mapping = json.load(json_file)

with open(f"{script_directory}/configuration/config.json", 'r') as json_file:
    settings = json.load(json_file)

model, screen, overlay, canvas, random_x, random_y, arduino = None, None, None, None, 0, 0, None
models_path = os.path.join(script_directory, "models")
launcher_models = [os.path.splitext(file)[0] for file in [file for file in os.listdir(models_path) if file.endswith(".pt")]]

targets = []
distances = []
coordinates = []

# Initialize Pygame
pygame.init()

# Initialize the joystick module
pygame.joystick.init()

# Attempt to setup the Xbox controller
try:
    controller = pygame.joystick.Joystick(0)  # Assumes the controller is the first joystick
    controller.init()
except pygame.error:
    print("Unable to initialize the Xbox Controller")
    
# Define a function to get the left trigger value
def get_left_trigger():
    return (controller.get_axis(4) + 1) / 2  # Normalize to 0 (unpressed) to 1 (fully pressed)

    
from Main.shared import (
    label_sensitivity,
    label_headshot,
    label_trigger_bot,
    label_confidence,
    label_recoil_strength,
    label_aim_shake_strength,
    label_max_move,
    label_mask_width,
    label_mask_height,
    settings,
    label_activation_bind,
    label_quit_bind,
    label_activation_key,
    label_quit_key,
    image_label_preview
)
from Main.events import (
    toggle_overlay,
    checkbox_auto_aim_event,
    checkbox_trigger_bot_event,
    checkbox_toggle_event,
    checkbox_recoil_event,
    checkbox_aim_shake_event,
    checkbox_overlay_event,
    checkbox_preview_event,
    checkbox_mask_left_event,
    checkbox_mask_right_event,
    slider_sensitivity_event,
    slider_headshot_event,
    slider_trigger_bot_event,
    slider_confidence_event,
    slider_recoil_strength_event,
    slider_aim_shake_strength_event,
    slider_max_move_event,
    slider_mask_width_event,
    slider_mask_height_event
)


def combobox_fps_callback(choice):
    settings['max_fps'] = int(choice)
    with open(f"{script_directory}/configuration/config.json", 'w') as json_file:
        json.dump(settings, json_file, indent=4)
    button_reload_event()


def combobox_yolo_version_callback(choice):
    models = [yolo_model for yolo_model in launcher_models if yolo_model.startswith(combobox_yolo_version.get())]
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
    settings['mouse_input'] = choice


def combobox_arduino_callback(choice):
    global arduino
    settings['arduino'] = choice
    arduino = serial.Serial(choice, 9600, timeout=1)


def update_aim_shake():
    global random_x, random_y
    if settings['aim_shake'] == "on":
        aim_shake_strength = int(settings['aim_shake_strength'])
        random_x = random.randint(-aim_shake_strength, aim_shake_strength)
        random_y = random.randint(-aim_shake_strength, aim_shake_strength)
    else:
        random_x = 0
        random_y = 0


def get_keycode(key):
    new_key = key_mapping.get(key)
    if new_key is None:
        new_key = win32api.VkKeyScan(key)
    return new_key


def combobox_mouse_activation_bind_callback(choice):
    label_activation_bind.configure(text=f"Activation key: {choice}")
    settings['activation_key_string'] = choice
    activation_key = get_keycode(settings['activation_key_string'])
    settings['activation_key'] = activation_key
    label_activation_key.configure(text=f"Activation key: {choice}")


def combobox_mouse_quit_bind_callback(choice):
    label_quit_bind.configure(text=f"Quit key: {choice}")
    settings['quit_key_string'] = choice
    quit_key = get_keycode(settings['quit_key_string'])
    settings['quit_key'] = quit_key
    label_quit_key.configure(text=f"Quit key: {choice}")


def button_activation_bind_event():
    activation_key = keyboard.read_event(suppress=True).name
    label_activation_bind.configure(text=f"Activation key: {activation_key}")
    settings['activation_key_string'] = activation_key
    activation_key = get_keycode(activation_key)
    settings['activation_key'] = activation_key
    label_activation_key.configure(text=f"Activation key: {settings['activation_key_string']}")


def button_quit_bind_event():
    quit_key = keyboard.read_event(suppress=True).name
    label_quit_bind.configure(text=f"Quit key: {quit_key}")
    settings['quit_key_string'] = quit_key
    quit_key = get_keycode(quit_key)
    settings['quit_key'] = quit_key
    label_quit_key.configure(text=f"Quit key: {settings['quit_key_string']}")


def toggle_overlay():
    global overlay, canvas
    if overlay is None:
        left = int(win32api.GetSystemMetrics(0) / 2 - settings['width'] / 2)
        top = int(win32api.GetSystemMetrics(1) / 2 - settings['height'] / 2)
        overlay = ctk.CTkToplevel()
        overlay.geometry(f"{settings['width']}x{settings['height']}+{left}+{top}")
        overlay.overrideredirect(True)
        overlay.config(bg='#000000')
        overlay.attributes("-alpha", 0.75)
        overlay.wm_attributes("-topmost", 1)
        overlay.attributes('-transparentcolor', '#000000', '-topmost', 1)
        overlay.resizable(False, False)
        set_clickthrough(overlay.winfo_id())
        canvas = ctk.CTkCanvas(overlay, width=640, height=640, bg='black', highlightbackground='white')
        canvas.pack()
    else:
        overlay.destroy()
        canvas.destroy()
        overlay = None
        canvas = None


def update_overlay():
    overlay.update()
    if len(distances) > 0 and canvas is not None:
        canvas.delete("all")
        for coord in coordinates:
            x_min, y_min, x_max, y_max = map(int, coord)
            canvas.create_rectangle(x_min, y_min, x_max, y_max, outline="white", width=2)
    elif canvas is not None:
        canvas.delete("all")


def update_preview(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if len(distances) > 0:
        min_distance_index = np.argmin(distances)
        half_width = int(settings['width'] / 2)
        half_height = int(settings['height'] / 2)
        for coord in coordinates:
            x_min, y_min, x_max, y_max = map(int, coord)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        for i, target in enumerate(targets):
            cv2.line(frame, (half_width, half_height), (target[0] + half_width, target[1] + half_height), (255, 255, 0) if i == min_distance_index else (0, 0, 255), 4)
            cv2.circle(frame, (target[0] + half_width, target[1] + half_height), 8, (0, 255, 0), -1)

    frame = Image.fromarray(cv2.resize(frame, (240, 240), interpolation=cv2.INTER_NEAREST))
    ctkframe = ctk.CTkImage(size=(240, 240), dark_image=frame, light_image=frame)
    image_label_preview.configure(image=ctkframe)


def set_clickthrough(hwnd):
    styles = win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, styles)
    win32gui.SetLayeredWindowAttributes(hwnd, 0, 255, 0x00000001)


def button_keybindings_event():
    if keybindings.state() == 'withdrawn':
        keybindings.deiconify()
        keybindings.geometry(f"400x160+{root.winfo_x() + 40}+{root.winfo_y() + 100}")
        keybindings.focus()
        keybindings.attributes('-topmost', True)
    else:
        keybindings.withdraw()


def on_close():
    keybindings.withdraw()


def on_minimize(event):
    keybindings.withdraw()


def button_reload_event():
    global screen, overlay
    settings['yolo_version'] = combobox_yolo_version.get()
    settings['yolo_model'] = combobox_yolo_model.get()
    settings['yolo_mode'] = combobox_yolo_mode.get()
    settings['yolo_device'] = combobox_yolo_device.get()
    settings['height'], settings['width'] = map(int, combobox_yolo_model_size.get().split('x'))
    slider_mask_width.configure(to=settings['width'])
    slider_mask_height.configure(to=settings['height'])
    combobox_fps.set(str(settings['max_fps']))
    slider_mask_width.set(0)
    slider_mask_height.set(0)
    if settings['mask_width'] <= settings['width']:
        slider_mask_width.set(settings['mask_width'])
    else:
        slider_mask_width.set(settings['width'])
        slider_mask_width_event(settings['width'])
    if settings['mask_height'] <= settings['height']:
        slider_mask_height.set(settings['mask_height'])
    else:
        slider_mask_height.set(settings['height'])
        slider_mask_height_event(settings['height'])
    left = int(win32api.GetSystemMetrics(0) / 2 - settings['width'] / 2)
    top = int(win32api.GetSystemMetrics(1) / 2 - settings['height'] / 2)
    right = int(left + settings['width'])
    bottom = int(top + settings['height'])
    if screen is None:
        screen = bettercam.create(output_color="BGRA", max_buffer_len=512)
    else:
        screen.stop()
    screen.start(region=(left, top, right, bottom), target_fps=int(settings['max_fps']), video_mode=True)
    load_model()
    if overlay is not None:
        toggle_overlay()
        toggle_overlay()


def get_mode():
    if settings['yolo_mode'] == "pytorch":
        return "engine"
    elif settings['yolo_mode'] == "onnx":
        return "engine"
    elif settings['yolo_mode'] == "tensorrt":
        return "engine"
    return None


def get_model_name():
    if settings['yolo_mode'] == "pytorch":
        return f"{settings['yolo_model']}.pt"
    else:
        return f"{settings['yolo_model']}{settings['yolo_version']}{settings['height']}{settings['width']}Half.{get_mode()}"


def model_existence_check():
    if not settings['yolo_mode'] == "pytorch":
        if not os.path.exists(f"{models_path}/{get_model_name()}"):
            pr_cyan(f"Exporting {settings['yolo_model']}.pt to {get_model_name()} with yolo{settings['yolo_version']}.")
            export_model()


def load_model():
    global model
    model_existence_check()
    pr_blue(f"Loading {get_model_name()} with yolo{settings['yolo_version']} for {settings['yolo_mode']} inference.")

    if not settings['yolo_mode'] == "onnx":
        if settings['yolo_version'] == "v8":
            model = YOLO(f"{models_path}/{get_model_name()}", task="detect")
        elif settings['yolo_version'] == "v5":
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=f"{models_path}/{get_model_name()}", verbose=False, trust_repo=True, force_reload=True)
    else:
        onnx_provider = ""
        if settings['yolo_device'] == "cpu":
            onnx_provider = "CPUExecutionProvider"
        elif settings['yolo_device'] == "amd":
            onnx_provider = "DmlExecutionProvider"
        elif settings['yolo_device'] == "nvidia":
            onnx_provider = "CUDAExecutionProvider"

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        model = ort.InferenceSession(f"{models_path}/{get_model_name()}", sess_options=so, providers=[onnx_provider])

    pr_green("Model loaded.")


def export_model():
    shutil.copy(f"{models_path}/{settings['yolo_model']}.pt", f"{script_directory}/temp.pt")
    if settings['yolo_version'] == "v8":
        x_model = YOLO(f"{script_directory}/temp.pt")
        x_model.export(format=get_mode(), imgsz=[int(settings['height']), int(settings['width'])], half=True, device=0)
    elif settings['yolo_version'] == "v5":
        os.system(f"python {script_directory}/yolov5/export.py --weights {script_directory}/temp.pt --include {get_mode()} --imgsz {int(settings['height'])} {int(settings['width'])} --half --device 0")
    os.remove(f"{script_directory}/temp.pt")
    if settings['yolo_mode'] == "tensorrt":
        os.remove(f"{script_directory}/temp.onnx")
        shutil.move(f"{script_directory}/temp.engine", f"{models_path}/{get_model_name()}")
    else:
        shutil.move(f"{script_directory}/temp.onnx", f"{models_path}/{get_model_name()}")
    pr_green("Export complete.")


def mask_frame(frame):
    if settings['mask_left'] == "on":
        frame[settings['height'] - int(settings['mask_height']):settings['height'], 0:int(settings['mask_width']),:] = 0
    if settings['mask_right'] == "on":
        frame[settings['height'] - int(settings['mask_height']):settings['height'], settings['width'] - int(settings['mask_width']):settings['width'], :] = 0
    return frame


def extract_original_name(name):
    pattern = re.compile(r'^(.+?)v\d{7}Half$')
    match = pattern.match(name)
    if match:
        return match.group(1)
    else:
        return name


def mouse_click():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def mouse_move(move_x, move_y, click):
    if settings['mouse_input'] == "arduino":
        arduino.write("{}:{}:{}x".format(move_x, move_y, click).encode())
    else:
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, move_x, move_y, 0, 0)
    if not settings['mouse_input'] == "arduino" and click == 1:
        mouse_click()


def calculate_targets(x1, y1, x2, y2):
    width_half = settings['width'] / 2
    height_half = settings['height'] / 2
    headshot_percent = settings['headshot'] / 100
    x = int(((x1 + x2) / 2) - width_half)
    y = int(((y1 + y2) / 2) + headshot_percent * (y1 - ((y1 + y2) / 2)) - height_half)
    distance = math.sqrt(x ** 2 + y ** 2)
    return (x, y), distance


def send_targets():
    global random_x, random_y

    if not distances:
        return

    trigger_distance = round(settings['trigger_bot_distance'])
    sensitivity_factor = settings['sensitivity'] / 20
    click = 0

    target_x, target_y = targets[distances.index(min(distances))]
    target_x = min(max(target_x, -settings['max_move']), settings['max_move'])
    target_y = min(max(target_y, -settings['max_move']), settings['max_move'])

    trigger_pressed = get_left_trigger() > 0.5  # Assuming half-press as activation threshold

    if settings['auto_aim'] == "on" and (trigger_pressed and settings['toggle'] == "off" or settings['toggle'] == "on"):
        recoil = int(settings['recoil_strength']) if settings['recoil'] == "on" and trigger_pressed else 0
        mouse_move_x = int((target_x + random_x) * sensitivity_factor)
        mouse_move_y = int((target_y + random_y + recoil) * sensitivity_factor)

        if settings['trigger_bot'] == "on" and target_x <= trigger_distance and target_y <= trigger_distance and target_x >= -trigger_distance and target_y >= -trigger_distance:
            click = 1

        mouse_move(mouse_move_x, mouse_move_y, click)



root = ctk.CTk()
root.title("Spawn-Aim")
root.geometry("600x800+40+40")
root.resizable(width=False, height=False)

var_auto_aim = ctk.StringVar(value="off")
checkbox_auto_aim = ctk.CTkCheckBox(root, text="Auto aim", variable=var_auto_aim, onvalue="on", offvalue="off", command=lambda: checkbox_auto_aim_event(var_auto_aim, settings))
checkbox_auto_aim.place(x=10, y=10)

var_trigger_bot = ctk.StringVar(value="off")
checkbox_trigger_bot = ctk.CTkCheckBox(root, text="Trigger bot", variable=var_trigger_bot, onvalue="on", offvalue="off", command=lambda: checkbox_trigger_bot_event(var_trigger_bot, settings))
checkbox_trigger_bot.place(x=10, y=40)

var_toggle = ctk.StringVar(value="off")
checkbox_toggle = ctk.CTkCheckBox(root, text="Auto aim toggle", variable=var_toggle, onvalue="on", offvalue="off", command=lambda: checkbox_toggle_event(var_toggle, settings))
checkbox_toggle.place(x=110, y=10)

var_recoil = ctk.StringVar(value="off")
checkbox_recoil = ctk.CTkCheckBox(root, text="Recoil control", variable=var_recoil, onvalue="on", offvalue="off", command=lambda: checkbox_recoil_event(var_recoil, settings))
checkbox_recoil.place(x=110, y=40)

var_aim_shake = ctk.StringVar(value="off")
checkbox_aim_shake = ctk.CTkCheckBox(root, text="Aim shake", variable=var_aim_shake, onvalue="on", offvalue="off", command=lambda: checkbox_aim_shake_event(var_aim_shake, settings))
checkbox_aim_shake.place(x=10, y=70)

var_overlay = ctk.StringVar(value="off")
checkbox_overlay = ctk.CTkCheckBox(root, text="Overlay", variable=var_overlay, onvalue="on", offvalue="off", command=lambda: checkbox_overlay_event(var_overlay, settings, toggle_overlay))
checkbox_overlay.place(x=110, y=70)

label_sensitivity = ctk.CTkLabel(root, text="Sensitivity: 0%")
label_sensitivity.place(x=10, y=100)

slider_sensitivity = ctk.CTkSlider(root, from_=0, to=100, command=slider_sensitivity_event)
slider_sensitivity.place(x=10, y=125)

label_headshot = ctk.CTkLabel(root, text="Headshot offset: 0%")
label_headshot.place(x=10, y=150)

slider_headshot = ctk.CTkSlider(root, from_=0, to=100, command=slider_headshot_event)
slider_headshot.place(x=10, y=175)

label_trigger_bot = ctk.CTkLabel(root, text=f"Trigger bot distance: 0 px")
label_trigger_bot.place(x=10, y=200)

slider_trigger_bot = ctk.CTkSlider(root, from_=0, to=100, command=slider_trigger_bot_event)
slider_trigger_bot.place(x=10, y=225)

label_confidence = ctk.CTkLabel(root, text=f"Confidence: 0%")
label_confidence.place(x=10, y=250)

slider_confidence = ctk.CTkSlider(root, from_=0, to=100, command=slider_confidence_event)
slider_confidence.place(x=10, y=275)

label_recoil_strength = ctk.CTkLabel(root, text=f"Recoil control strength: 0%")
label_recoil_strength.place(x=10, y=300)

slider_recoil_strength = ctk.CTkSlider(root, from_=0, to=100, command=slider_recoil_strength_event)
slider_recoil_strength.place(x=10, y=325)

label_aim_shake_strength = ctk.CTkLabel(root, text=f"Aim shake strength: 0%")
label_aim_shake_strength.place(x=10, y=350)

slider_aim_shake_strength = ctk.CTkSlider(root, from_=0, to=100, command=slider_aim_shake_strength_event)
slider_aim_shake_strength.place(x=10, y=375)

label_max_move = ctk.CTkLabel(root, text=f"Max move speed: 0 px")
label_max_move.place(x=10, y=400)

slider_max_move = ctk.CTkSlider(root, from_=0, to=100, command=slider_max_move_event)
slider_max_move.place(x=10, y=425)

label_fps = ctk.CTkLabel(root, text="FPS:")
label_fps.place(x=10, y=670)

combobox_fps = ctk.CTkComboBox(root, values=["30", "60", "90", "120", "144", "165", "180"], command=combobox_fps_callback, state="readonly")
combobox_fps.place(x=10, y=700)
combobox_fps.set(str(settings['max_fps']))

label_yolo_version = ctk.CTkLabel(root, text="Yolo version:")
label_yolo_version.place(x=10, y=550)

combobox_yolo_version = ctk.CTkComboBox(root, values=["v8", "v5"], command=combobox_yolo_version_callback, state="readonly")
combobox_yolo_version.place(x=10, y=580)

label_yolo_model = ctk.CTkLabel(root, text="Yolo model:")
label_yolo_model.place(x=10, y=610)

combobox_yolo_model = ctk.CTkComboBox(root, values=["Default"], command=combobox_yolo_model_callback, state="readonly")
combobox_yolo_model.place(x=10, y=640)

label_yolo_mode = ctk.CTkLabel(root, text="Inference mode:")
label_yolo_mode.place(x=160, y=550)

combobox_yolo_mode = ctk.CTkComboBox(root, values=["pytorch", "onnx", "tensorrt"], command=combobox_yolo_mode_callback, state="readonly")
combobox_yolo_mode.place(x=160, y=580)

label_yolo_device = ctk.CTkLabel(root, text="Device:")
label_yolo_device.place(x=160, y=610)

combobox_yolo_device = ctk.CTkComboBox(root, values=["cpu", "amd", "nvidia"], command=combobox_yolo_device_callback, state="readonly")
combobox_yolo_device.place(x=160, y=640)

label_model_size = ctk.CTkLabel(root, text="Model size:")
label_model_size.place(x=310, y=550)

combobox_yolo_model_size = ctk.CTkComboBox(root, values=["160x160", "320x320", "480x480", "640x640"], command=combobox_yolo_model_size_callback, state="readonly")
combobox_yolo_model_size.place(x=310, y=580)

button_reload = ctk.CTkButton(root, text="Reload model", command=button_reload_event)
button_reload.place(x=310, y=640)

button_keybindings = ctk.CTkButton(root, text="Configure keybindings", command=button_keybindings_event)
button_keybindings.place(x=10, y=525)

label_activation_key = ctk.CTkLabel(root, text=f"Activation key: None")
label_activation_key.place(x=10, y=455)

label_quit_key = ctk.CTkLabel(root, text=f"Quit key: None")
label_quit_key.place(x=10, y=485)

var_preview = ctk.StringVar(value="off")
checkbox_preview = ctk.CTkCheckBox(root, text="Preview", variable=var_preview, onvalue="on", offvalue="off", command=lambda: checkbox_preview_event(var_preview, settings, image_label_preview, image_preview))
checkbox_preview.place(x=320, y=10)

label_fps = ctk.CTkLabel(root, text="Fps:")
label_fps.place(x=240, y=10)

image_preview = ctk.CTkImage(size=(240, 240), dark_image=Image.open(f"{script_directory}/preview.png"), light_image=Image.open(f"{script_directory}/preview.png"))
image_label_preview = ctk.CTkLabel(root, image=image_preview, text="")
image_label_preview.place(x=240, y=40)

var_mask_left = ctk.StringVar(value="off")
checkbox_mask_left = ctk.CTkCheckBox(root, text="Mask left", variable=var_mask_left, onvalue="on", offvalue="off", command=lambda: checkbox_mask_left_event(var_mask_left, settings))
checkbox_mask_left.place(x=240, y=290)

var_mask_right = ctk.StringVar(value="off")
checkbox_mask_right = ctk.CTkCheckBox(root, text="Mask right", variable=var_mask_right, onvalue="on", offvalue="off", command=lambda: checkbox_mask_right_event(var_mask_right, settings))
checkbox_mask_right.place(x=380, y=290)

label_mask_width = ctk.CTkLabel(root, text=f"Mask width: 0 px")
label_mask_width.place(x=240, y=320)

slider_mask_width = ctk.CTkSlider(root, from_=0, to=640, command=slider_mask_width_event)
slider_mask_width.place(x=240, y=345)

label_mask_height = ctk.CTkLabel(root, text=f"Mask height: 0 px")
label_mask_height.place(x=240, y=370)

slider_mask_height = ctk.CTkSlider(root, from_=0, to=640, command=slider_mask_height_event)
slider_mask_height.place(x=240, y=395)

label_mouse_input = ctk.CTkLabel(root, text=f"Mouse input method:")
label_mouse_input.place(x=310, y=420)

combobox_mouse_input = ctk.CTkComboBox(root, values=["default"], command=combobox_mouse_input_callback, state="readonly")
combobox_mouse_input.place(x=310, y=450)

label_mouse_input = ctk.CTkLabel(root, text=f"Arduino port:")
label_mouse_input.place(x=310, y=480)

combobox_arduino = ctk.CTkComboBox(root, values=["COM1"], command=combobox_arduino_callback, state="readonly")
combobox_arduino.place(x=310, y=510)


keybindings = ctk.CTkToplevel()
keybindings.title("Spawn-Aim Keybinder")
keybindings.geometry(f"400x160+{root.winfo_x() + 40}+{root.winfo_y() + 100}")
keybindings.resizable(width=False, height=False)
keybindings.withdraw()
keybindings.protocol("WM_DELETE_WINDOW", on_close)
keybindings.bind("<Unmap>", on_minimize)

button_activation_bind = ctk.CTkButton(keybindings, text="Configure activation key", command=button_activation_bind_event)
button_activation_bind.place(x=10, y=10)

button_quit_bind = ctk.CTkButton(keybindings, text="Configure quit key", command=button_quit_bind_event)
button_quit_bind.place(x=10, y=50)

label_activation_bind = ctk.CTkLabel(keybindings, text=f"Activation key: {settings['activation_key_string']}")
label_activation_bind.place(x=10, y=90)

label_quit_bind = ctk.CTkLabel(keybindings, text=f"Quit key: {settings['quit_key_string']}")
label_quit_bind.place(x=10, y=120)

combobox_mouse_activation_bind = ctk.CTkComboBox(keybindings, values=list(key_mapping.keys()), command=combobox_mouse_activation_bind_callback, state="readonly")
combobox_mouse_activation_bind.place(x=200, y=10)

combobox_mouse_quit_bind = ctk.CTkComboBox(keybindings, values=list(key_mapping.keys()), command=combobox_mouse_quit_bind_callback, state="readonly")
combobox_mouse_quit_bind.place(x=200, y=50)

combobox_mouse_activation_bind.set("Select special")
combobox_mouse_quit_bind.set("Select special")



def main(**argv):
    global model, screen, settings, overlay, canvas

    print_colored('''
  ██████  ██▓███   ▄▄▄      █     █░ ███▄    █      ▄▄▄       ██▓ ███▄ ▄███▓ ▄▄▄▄    ▒█████  ▄▄▄█████▓
▒██    ▒ ▓██░  ██ ▒████▄   ▓█░ █ ░█░ ██ ▀█   █     ▒████▄   ▒▓██▒▓██▒▀█▀ ██▒▓█████▄ ▒██▒  ██▒▓  ██▒ ▓▒
░ ▓██▄   ▓██░ ██▓▒▒██  ▀█▄ ▒█░ █ ░█ ▓██  ▀█ ██▒    ▒██  ▀█▄ ▒▒██▒▓██    ▓██░▒██▒ ▄██▒██░  ██▒▒ ▓██░ ▒░
  ▒   ██▒▒██▄█▓▒ ▒░██▄▄▄▄██░█░ █ ░█ ▓██▒  ▐▌██▒    ░██▄▄▄▄██░░██░▒██    ▒██ ▒██░█▀  ▒██   ██░░ ▓██▓ ░ 
▒██████▒▒▒██▒ ░  ░▒▓█   ▓██░░██▒██▓ ▒██░   ▓██░     ▓█   ▓██░░██░▒██▒   ░██▒░▓█  ▀█▓░ ████▓▒░  ▒██▒ ░ 
▒ ▒▓▒ ▒ ░▒▓▒░ ░  ░░▒▒   ▓▒█░ ▓░▒ ▒  ░ ▒░   ▒ ▒      ▒▒   ▓▒█ ░▓  ░ ▒░   ░  ░░▒▓███▀▒░ ▒░▒░▒░   ▒ ░░   
░ ░▒  ░  ░▒ ░     ░ ░   ▒▒   ▒ ░ ░  ░ ░░   ░ ▒░      ░   ▒▒ ░ ▒ ░░  ░      ░▒░▒   ░   ░ ▒ ▒░     ░    
░  ░  ░  ░░         ░   ▒    ░   ░     ░   ░ ░       ░   ▒  ░ ▒ ░░      ░    ░    ░ ░ ░ ░ ▒    ░      
      ░                 ░      ░             ░           ░    ░         ░    ░          ░ ░''', "purple")
    print("https://github.com/spawn9859/Spawn-Aim")
    print_colored("\nMake sure your game is in the center of your screen!", "yellow")

    with open(os.path.join(script_directory, "aimbotSettings", f"{argv['settingsProfile'].lower()}.json"), "r") as f:
        launcher_settings = json.load(f)

    mouse_inputs = ["default", "arduino"]

    ports = []
    default_port = "COM1"
    portslist = list(serial.tools.list_ports.comports())
    for port in portslist:
        if "Arduino" in port.description:
            default_port = port[0]
        ports.append(port[0])
        
    # Attempt to setup the Xbox controller
    try:
        controller = pygame.joystick.Joystick(0)  # Assumes the controller is the first joystick
        controller.init()
    except pygame.error:
        print("Unable to initialize the Xbox Controller")

    activation_key = key_mapping.get(launcher_settings['activationKey'])
    quit_key = key_mapping.get(launcher_settings['quitKey'])

    if activation_key is None:
        activation_key = win32api.VkKeyScan(launcher_settings['activationKey'])
    if quit_key is None:
        quit_key = win32api.VkKeyScan(launcher_settings['quitKey'])

    settings['auto_aim'] = "off"
    settings['trigger_bot'] = "on" if launcher_settings['autoFire'] else "off"
    settings['toggle'] = "on" if launcher_settings['toggleable'] else "off"
    settings['recoil'] = "off"
    settings['aim_shake'] = "on" if launcher_settings['aimShakey'] else "off"
    settings['overlay'] = "off"
    settings['preview'] = "on" if launcher_settings['visuals'] else "off"
    settings['mask_left'] = "on" if (launcher_settings['maskLeft'] and launcher_settings['useMask']) else "off"
    settings['mask_right'] = "on" if (not launcher_settings['maskLeft'] and launcher_settings['useMask']) else "off"
    settings['sensitivity'] = launcher_settings['movementAmp'] * 100
    settings['headshot'] = launcher_settings['headshotDistanceModifier'] * 100 if launcher_settings['headshotMode'] else 40
    settings['trigger_bot_distance'] = launcher_settings['autoFireActivationDistance']
    settings['confidence'] = launcher_settings['confidence'] * 100
    settings['recoil_strength'] = 0
    settings['aim_shake_strength'] = launcher_settings['aimShakeyStrength']
    settings['max_move'] = 100
    settings['height'] = launcher_settings['screenShotHeight']
    settings['width'] = launcher_settings['screenShotHeight']
    settings['mask_width'] = launcher_settings['maskWidth']
    settings['mask_height'] = launcher_settings['maskHeight']
    settings['yolo_version'] = f"v{argv['yoloVersion']}"
    settings['yolo_model'] = "v5_Fortnite_taipeiuser"
    settings['yolo_mode'] = "tensorrt"
    settings['yolo_device'] = {1: "cpu", 2: "amd", 3: "nvidia"}.get(launcher_settings['onnxChoice'])
    settings['activation_key'] = activation_key
    settings['quit_key'] = quit_key
    settings['activation_key_string'] = launcher_settings['activationKey']
    settings['quit_key_string'] = launcher_settings['quitKey']
    settings['mouse_input'] = "default"
    settings['arduino'] = default_port

    var_auto_aim.set(settings['auto_aim'])
    var_trigger_bot.set(settings['trigger_bot'])
    var_toggle.set(settings['toggle'])
    var_recoil.set(settings['recoil'])
    var_aim_shake.set(settings['aim_shake'])
    var_overlay.set(settings['overlay'])
    var_preview.set(settings['preview'])
    var_mask_left.set(settings['mask_left'])
    var_mask_right.set(settings['mask_right'])
    label_sensitivity.configure(text=f"Sensitivity: {round(settings['sensitivity'])}%")
    slider_sensitivity.set(settings['sensitivity'])
    label_headshot.configure(text=f"Headshot offset: {round(settings['headshot'])}%")
    slider_headshot.set(settings['headshot'])
    label_trigger_bot.configure(text=f"Trigger bot distance: {round(settings['trigger_bot_distance'])} px")
    slider_trigger_bot.set(settings['trigger_bot_distance'])
    label_confidence.configure(text=f"Confidence: {round(settings['confidence'])}%")
    slider_confidence.set(settings['confidence'])
    label_recoil_strength.configure(text=f"Recoil control strength: {round(settings['recoil_strength'])}%")
    slider_recoil_strength.set(settings['recoil_strength'])
    label_aim_shake_strength.configure(text=f"Aim shake strength: {round(settings['aim_shake_strength'])}%")
    slider_aim_shake_strength.set(settings['aim_shake_strength'])
    label_max_move.configure(text=f"Max move speed: {round(settings['max_move'])} px")
    slider_max_move.set(settings['max_move'])
    combobox_yolo_model_size.set(f"{settings['height']}x{settings['width']}")
    label_mask_width.configure(text=f"Mask width: {round(settings['mask_width'])} px")
    slider_mask_width.set(settings['mask_width'])
    label_mask_height.configure(text=f"Mask height: {round(settings['mask_height'])} px")
    slider_mask_height.set(settings['mask_height'])
    combobox_yolo_version.set(settings['yolo_version'])
    combobox_yolo_model.configure(values=[yolo_model for yolo_model in launcher_models if yolo_model.startswith(combobox_yolo_version.get())])
    combobox_yolo_model.set(settings['yolo_model'])
    combobox_yolo_mode.set(settings['yolo_mode'])
    combobox_yolo_device.set(settings['yolo_device'])
    label_activation_key.configure(text=f"Activation key: {settings['activation_key_string']}")
    label_quit_key.configure(text=f"Quit key: {settings['quit_key_string']}")
    combobox_fps.set(str(settings['max_fps']))
    combobox_mouse_input.configure(values=mouse_inputs)
    combobox_mouse_input.set(settings['mouse_input'])
    combobox_arduino.configure(values=ports)
    combobox_arduino.set(settings['arduino'])

    checkbox_trigger_bot.configure(state="normal")
    checkbox_toggle.configure(state="normal")
    checkbox_recoil.configure(state="normal")
    checkbox_aim_shake.configure(state="normal")
    checkbox_overlay.configure(state="normal")
    checkbox_preview.configure(state="normal")
    checkbox_mask_left.configure(state="normal")
    checkbox_mask_right.configure(state="normal")
    slider_sensitivity.configure(state="normal")
    slider_headshot.configure(state="normal")
    slider_trigger_bot.configure(state="normal")
    slider_confidence.configure(state="normal")
    slider_recoil_strength.configure(state="normal")
    slider_aim_shake_strength.configure(state="normal")
    slider_max_move.configure(state="normal")
    slider_mask_width.configure(state="normal")
    slider_mask_height.configure(state="normal")
    button_keybindings.configure(state="normal")
    button_reload.configure(state="normal")
    combobox_mouse_input.configure(state="readonly")
    combobox_arduino.configure(state="readonly")
    combobox_yolo_version.configure(state="readonly")
    combobox_yolo_model.configure(state="readonly")
    combobox_yolo_mode.configure(state="readonly")
    combobox_yolo_device.configure(state="readonly")
    combobox_yolo_model_size.configure(state="readonly")

    button_reload_event()

    if settings['overlay'] == "on":
        toggle_overlay()

    start_time = time.time()
    frame_count = 0
    pressing = False

    while True:
        root.update()
        frame_count += 1
        np_frame = np.array(screen.get_latest_frame())
        pygame.event.pump()


        if np_frame.shape[2] == 4:
            np_frame = np_frame[:, :, :3]

        with torch.no_grad():
            frame = mask_frame(np_frame)

            if settings['yolo_mode'] == "pytorch" or settings['yolo_mode'] == "tensorrt":
                if settings['yolo_version'] == "v5":
                    model.conf = settings['confidence'] / 100
                    model.iou = settings['confidence'] / 100
                    results = model(frame, size=[settings['height'], settings['width']])
                    if len(results.xyxy[0]) != 0:
                        for box in results.xyxy[0]:
                            box_result = calculate_targets(box[0], box[1], box[2], box[3])
                            coordinates.append((box[0], box[1], box[2], box[3]))
                            targets.append(box_result[0])
                            distances.append(box_result[1])

                elif settings['yolo_version'] == "v8":
                    results = model.predict(frame, verbose=False, conf=settings['confidence'] / 100, iou=settings['confidence'] / 100, half=False, imgsz=[settings['height'], settings['width']])
                    for result in results:
                        if len(result.boxes.xyxy) != 0:
                            for box in result.boxes.xyxy:
                                box_result = calculate_targets(box[0], box[1], box[2], box[3])
                                coordinates.append((box[0], box[1], box[2], box[3]))
                                targets.append(box_result[0])
                                distances.append(box_result[1])

            elif settings['yolo_mode'] == "onnx":
                if settings['yolo_device'] == "nvidia":
                    frame = torch.from_numpy(frame).to('cuda')

                    frame = torch.movedim(frame, 2, 0)
                    frame = frame.half()
                    frame /= 255
                    if len(frame.shape) == 3:
                        frame = frame[None]
                else:
                    frame = frame / 255
                    frame = frame.astype(np.half)
                    frame = np.moveaxis(frame, 3, 1)

                if settings['yolo_device'] == "nvidia":
                    outputs = model.run(None, {'images': cp.asnumpy(frame)})
                else:
                    outputs = model.run(None, {'images': np.array(frame)})

                frame = torch.from_numpy(outputs[0])

                if settings['yolo_version'] == "v5":
                    predictions = non_max_suppression(frame, settings['confidence'] / 100, settings['confidence'] / 100, 0, False, max_det=4)

                elif settings['yolo_version'] == "v8":
                    predictions = ops.non_max_suppression(frame, settings['confidence'] / 100, settings['confidence'] / 100, 0, False, max_det=4)

                for i, det in enumerate(predictions):
                    if len(det):
                        for *xyxy, conf, cls in reversed(det):
                            if int(cls) == 0:
                                box_result = calculate_targets(xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item())
                                coordinates.append((xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()))
                                targets.append(box_result[0])
                                distances.append(box_result[1])

            send_targets()

        if settings['preview'] == "on":
            update_preview(np_frame)

        if settings['overlay'] == "on":
            update_overlay()

        elapsed_time = time.time() - start_time
        if elapsed_time >= 0.2:
            label_fps.configure(text=f"Fps: {round(frame_count / elapsed_time)}")
            frame_count = 0
            start_time = time.time()
            update_aim_shake()

        # if settings['toggle'] == "on":
            # if win32api.GetKeyState(settings['activation_key']) in (-127, -128) and not pressing:
                # pressing = True
                # checkbox_auto_aim.toggle()
            # elif not win32api.GetKeyState(settings['activation_key']) in (-127, -128) and pressing:
                # pressing = False
            # Toggle auto aim based on the left trigger of the Xbox controller
        if settings['toggle'] == "on":
            if get_left_trigger() > 0.5 and not pressing:
                pressing = True
                checkbox_auto_aim.toggle()
            elif get_left_trigger() <= 0.5 and pressing:
                pressing = False

        if win32api.GetKeyState(settings['quit_key']) in (-127, -128):
            pr_green("Goodbye!")
            screen.stop()
            screen.release()
            quit()

        targets.clear()
        distances.clear()
        coordinates.clear()
if __name__ == "__main__":
    main(settingsProfile="default", yoloVersion=5, version=0)
