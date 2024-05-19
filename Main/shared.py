from customtkinter import CTkLabel, CTkSlider

label_sensitivity = CTkLabel(master=None)
label_headshot = CTkLabel(master=None)
label_trigger_bot = CTkLabel(master=None)
label_confidence = CTkLabel(master=None)
label_recoil_strength = CTkLabel(master=None)
label_aim_shake_strength = CTkLabel(master=None)
label_max_move = CTkLabel(master=None)
label_mask_width = CTkLabel(master=None)
label_mask_height = CTkLabel(master=None)
label_activation_bind = CTkLabel(master=None)
label_quit_bind = CTkLabel(master=None)
label_activation_key = CTkLabel(master=None)
label_quit_key = CTkLabel(master=None)
from PIL import Image
image_preview = Image.open(f"{script_directory}/preview.png")
image_preview_ctk = ctk.CTkImage(size=(240, 240), dark_image=image_preview, light_image=image_preview)
image_label_preview = CTkLabel(master=None, image=image_preview_ctk, text="")

settings = {}
