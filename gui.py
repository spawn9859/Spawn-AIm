import customtkinter as ctk
from PIL import Image, ImageTk
from main import checkbox_auto_aim_event, checkbox_trigger_bot_event, checkbox_toggle_event, checkbox_recoil_event, checkbox_aim_shake_event, checkbox_overlay_event, slider_sensitivity_event, slider_headshot_event, slider_trigger_bot_event, slider_confidence_event, slider_recoil_strength_event, slider_aim_shake_strength_event, slider_max_move_event
import os

script_directory = os.path.dirname(os.path.abspath(__file__ if "__file__" in locals() else __file__))

def create_gui(settings, key_mapping, launcher_models):
    root = ctk.CTk()
    root.title("Spawn-Aim")
    root.geometry("600x800+40+40")
    root.resizable(width=False, height=False)

    var_auto_aim = ctk.StringVar(value="off")
    checkbox_auto_aim = ctk.CTkCheckBox(root, text="Auto aim", variable=var_auto_aim, onvalue="on", offvalue="off", command=checkbox_auto_aim_event)
    checkbox_auto_aim.place(x=10, y=10)

    var_trigger_bot = ctk.StringVar(value="off")
    checkbox_trigger_bot = ctk.CTkCheckBox(root, text="Trigger bot", variable=var_trigger_bot, onvalue="on", offvalue="off", command=checkbox_trigger_bot_event)
    checkbox_trigger_bot.place(x=10, y=40)

    var_toggle = ctk.StringVar(value="off")
    checkbox_toggle = ctk.CTkCheckBox(root, text="Auto aim toggle", variable=var_toggle, onvalue="on", offvalue="off", command=checkbox_toggle_event)
    checkbox_toggle.place(x=110, y=10)

    var_recoil = ctk.StringVar(value="off")
    checkbox_recoil = ctk.CTkCheckBox(root, text="Recoil control", variable=var_recoil, onvalue="on", offvalue="off", command=checkbox_recoil_event)
    checkbox_recoil.place(x=110, y=40)

    var_aim_shake = ctk.StringVar(value="off")
    checkbox_aim_shake = ctk.CTkCheckBox(root, text="Aim shake", variable=var_aim_shake, onvalue="on", offvalue="off", command=checkbox_aim_shake_event)
    checkbox_aim_shake.place(x=10, y=70)

    var_overlay = ctk.StringVar(value="off")
    checkbox_overlay = ctk.CTkCheckBox(root, text="Overlay", variable=var_overlay, onvalue="on", offvalue="off", command=checkbox_overlay_event)
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

    return root, var_auto_aim, var_trigger_bot, var_toggle, var_recoil, var_aim_shake, var_overlay, label_sensitivity, slider_sensitivity, label_headshot, slider_headshot, label_trigger_bot, slider_trigger_bot, label_confidence, slider_confidence, label_recoil_strength, slider_recoil_strength, label_aim_shake_strength, slider_aim_shake_strength, label_max_move, slider_max_move
