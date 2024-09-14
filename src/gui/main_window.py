import customtkinter as ctk
import sv_ttk
import cv2
from PIL import Image, ImageTk, ImageDraw
from .components import create_checkboxes, create_sliders, create_comboboxes, create_buttons, create_labels
import numpy as np
import win32gui
import win32con
import win32api
import threading
import queue
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class MainWindow:
    def __init__(self, config_manager):
        self.root = ctk.CTk()
        self.root.title("Spawn-Aim")
        self.root.geometry("800x900")
        self.root.minsize(600, 700)  # Set minimum size
        self.root.resizable(width=True, height=True)  # Allow resizing
        
        self.config_manager = config_manager
        self.fov_overlay = None
        
        sv_ttk.set_theme("dark")
        
        self.preview_label = None
        self.setup_ui()
        
        self.running = True
        self.root.after(100, self.periodic_update)  # Start periodic updates

    def periodic_update(self):
        if self.running:
            self.update_fov_overlay()
            # Add any other update operations here
            self.root.after(100, self.periodic_update)  # Schedule next update

    def setup_ui(self):
        # Use grid geometry manager for more flexibility
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Create tabs
        self.tabview = ctk.CTkTabview(self.main_frame)
        self.tabview.grid(row=0, column=0, sticky="nsew")

        self.tab_main = self.tabview.add("Main")
        self.tab_settings = self.tabview.add("Settings")
        self.tab_advanced = self.tabview.add("Advanced")

        # Main tab
        self.setup_main_tab()

        # Settings tab
        self.setup_settings_tab()

        # Advanced tab
        self.setup_advanced_tab()

        # Theme toggle button
        self.theme_button = ctk.CTkButton(self.root, text="Toggle Theme", command=self.toggle_theme)
        self.theme_button.pack(pady=10)

    def setup_main_tab(self):
        main_frame = ctk.CTkFrame(self.tab_main)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Left column
        left_column = ctk.CTkFrame(main_frame)
        left_column.pack(side="left", fill="both", expand=True, padx=(0, 5))

        # Checkboxes
        checkbox_frame = ctk.CTkFrame(left_column)
        checkbox_frame.pack(fill="x", pady=10)
        self.checkboxes = create_checkboxes(checkbox_frame, self.config_manager, self.update_setting)

        # FPS Label
        self.fps_label = create_labels(left_column)['fps']
        self.fps_label.pack(pady=5)

        # FOV controls
        fov_frame = ctk.CTkFrame(left_column)
        fov_frame.pack(fill="x", pady=10)
        
        self.fov_visible = ctk.BooleanVar(value=self.config_manager.get_setting("show_fov"))
        self.fov_checkbox = ctk.CTkCheckBox(
            fov_frame,
            text="Show FOV",
            variable=self.fov_visible,
            onvalue=True,
            offvalue=False,
            command=self.toggle_fov_visibility,
        )
        self.fov_checkbox.pack(side="left", padx=5)

        self.enable_fov_var = ctk.BooleanVar(value=self.config_manager.get_setting("fov_enabled"))
        self.enable_fov = ctk.CTkCheckBox(
            fov_frame,
            text="Enable FOV",
            variable=self.enable_fov_var,
            command=self.toggle_fov_enabled,
        )
        self.enable_fov.pack(side="left", padx=5)

        # Main sliders
        main_slider_frame = ctk.CTkFrame(left_column)
        main_slider_frame.pack(fill="x", pady=10)
        
        main_slider_configs = [
            ("sensitivity", "Sensitivity", 0, 100),
            ("confidence", "Confidence threshold", 0, 100),
            ("headshot", "Headshot offset", 0, 100),
            ("trigger_bot_distance", "Trigger bot distance", 0, 100),
            ("fov_size", "FOV Size", 0, 200),
        ]
        
        self.main_sliders = create_sliders(main_slider_frame, self.config_manager, self.update_setting, main_slider_configs)

        # Key bindings
        key_frame = ctk.CTkFrame(left_column)
        key_frame.pack(fill="x", pady=10)
        
        activation_key_label = ctk.CTkLabel(key_frame, text=f"Activation key: {self.config_manager.get_setting('activation_key_string')}")
        activation_key_label.pack(side="left", padx=5)
        
        quit_key_label = ctk.CTkLabel(key_frame, text=f"Quit key: {self.config_manager.get_setting('quit_key_string')}")
        quit_key_label.pack(side="left", padx=5)

        # Right column
        right_column = ctk.CTkFrame(main_frame)
        right_column.pack(side="right", fill="both", expand=True, padx=(5, 0))

        # Preview image
        self.preview_image = Image.new("RGB", (320, 320), color="gray")
        self.preview_photo = ImageTk.PhotoImage(self.preview_image)
        self.preview_label = ctk.CTkLabel(right_column, image=self.preview_photo, text="")
        self.preview_label.pack(pady=10)
        
        if not self.config_manager.get_setting("preview"):
            self.preview_label.pack_forget()

    def setup_settings_tab(self):
        settings_frame = ctk.CTkFrame(self.tab_settings)
        settings_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Additional sliders
        additional_slider_frame = ctk.CTkFrame(settings_frame)
        additional_slider_frame.pack(fill="x", pady=10)
        
        additional_slider_configs = [
            ("smoothing_factor", "Smoothing factor", 0, 100),
            ("recoil_strength", "Recoil control strength", 0, 100),
            ("aim_shake_strength", "Aim shake strength", 0, 100),
            ("max_move", "Max move speed", 0, 100),
            ("mask_width", "Mask width", 0, 640),
            ("mask_height", "Mask height", 0, 640),
        ]
        
        self.additional_sliders = create_sliders(additional_slider_frame, self.config_manager, self.update_setting, additional_slider_configs)

        # Comboboxes
        combo_frame = ctk.CTkFrame(settings_frame)
        combo_frame.pack(fill="x", pady=10)
        self.comboboxes = create_comboboxes(combo_frame, self.config_manager, self.config_manager.update_setting)

    def setup_advanced_tab(self):
        advanced_frame = ctk.CTkFrame(self.tab_advanced)
        advanced_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Buttons
        self.buttons = create_buttons(advanced_frame, self)

    def toggle_theme(self):
        current_theme = sv_ttk.get_theme()
        if current_theme == "dark":
            sv_ttk.use_light_theme()
        else:
            sv_ttk.use_dark_theme()

    def update_setting(self, key, value):
        self.config_manager.update_setting(key, value)
        if key == "preview":
            self.toggle_preview(value)

    def toggle_preview(self, value):
        if value:
            self.preview_label.pack(pady=10)
        else:
            self.preview_label.pack_forget()

    def update_preview(self, frame, coordinates, targets, distances):
        if self.config_manager.get_setting("preview"):
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (240, 240))
                image = Image.fromarray(frame)
                draw = ImageDraw.Draw(image)
                
                # Draw bounding boxes
                for coord in coordinates:
                    x1, y1, x2, y2 = [int(c * 240 / self.config_manager.get_setting("width")) for c in coord]
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                
                # Draw aim lines
                center_x, center_y = 120, 120
                if len(targets) > 0:
                    min_distance_index = np.argmin(distances)
                    for i, (target_x, target_y) in enumerate(targets):
                        color = "yellow" if i == min_distance_index else "blue"
                        scaled_x = int(target_x * 240 / self.config_manager.get_setting("width")) + center_x
                        scaled_y = int(target_y * 240 / self.config_manager.get_setting("height")) + center_y
                        draw.line((center_x, center_y, scaled_x, scaled_y), fill=color, width=2)
                        draw.ellipse((scaled_x-3, scaled_y-3, scaled_x+3, scaled_y+3), fill=color)

                # Draw FOV circle if enabled
                if self.config_manager.get_setting("show_fov"):
                    fov_size = self.config_manager.get_setting("fov_size")
                    scaled_fov_size = int(fov_size * 240 / self.config_manager.get_setting("width"))
                    draw.ellipse(
                        [center_x - scaled_fov_size, center_y - scaled_fov_size,
                         center_x + scaled_fov_size, center_y + scaled_fov_size],
                        outline="red", width=2
                    )

                self.preview_photo = ImageTk.PhotoImage(image=image)
                self.preview_label.configure(image=self.preview_photo)
                self.preview_label.image = self.preview_photo
            except Exception as e:
                logging.error(f"Failed to update preview: {e}")

    def update_fps_label(self, fps):
        self.root.after(0, self._update_fps_label_gui, fps)

    def _update_fps_label_gui(self, fps):
        self.fps_label.configure(text=f"FPS: {round(fps)}")

    def toggle_fov_visibility(self):
        value = self.fov_visible.get()
        self.config_manager.update_setting("show_fov", value)
        if value:
            self.create_fov_overlay()
        else:
            self.destroy_fov_overlay()

    def toggle_fov_enabled(self):
        current_value = self.config_manager.get_setting("fov_enabled")
        new_value = not current_value
        self.config_manager.update_setting("fov_enabled", new_value)
        
        # Update the checkbox state
        self.enable_fov_var.set(new_value)
        
        # Update the FOV visibility based on the new state
        if new_value:
            self.create_fov_overlay()
        else:
            self.destroy_fov_overlay()
        
        logging.debug(f"FOV Enabled: {new_value}")  # Replace print with logging

    def create_fov_overlay(self):
        if self.fov_overlay is None:
            screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
            screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)

            self.fov_overlay = ctk.CTkToplevel(self.root)
            self.fov_overlay.geometry(f"{screen_width}x{screen_height}+0+0")
            self.fov_overlay.overrideredirect(True)
            self.fov_overlay.attributes("-topmost", True)
            self.fov_overlay.attributes("-transparentcolor", "black")
            self.fov_overlay.configure(bg="black")

            self.fov_canvas = ctk.CTkCanvas(self.fov_overlay, width=screen_width, height=screen_height, 
                                            highlightthickness=0, bg="black")
            self.fov_canvas.pack()

            hwnd = self.fov_overlay.winfo_id()
            styles = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            styles = styles | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, styles)
            win32gui.SetLayeredWindowAttributes(hwnd, 0, 255, win32con.LWA_COLORKEY)

        self.update_fov_overlay()

    def destroy_fov_overlay(self):
        if self.fov_overlay:
            self.fov_overlay.destroy()
            self.fov_overlay = None

    def update_fov_overlay(self):
        if self.fov_overlay and self.config_manager.get_setting("show_fov"):
            self.fov_canvas.delete("all")
            
            screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
            screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)

            center_x = screen_width // 2
            center_y = screen_height // 2

            fov_size = self.config_manager.get_setting("fov_size")

            stipple_pattern = "gray75"

            self.fov_canvas.create_oval(
                center_x - fov_size, center_y - fov_size,
                center_x + fov_size, center_y + fov_size,
                outline="red", width=2, stipple=stipple_pattern
            )

    def reload_model(self):
        # Implement model reloading logic here
        logging.info("Reloading model...")
        # You might want to call a method from your config_manager or another appropriate class to reload the model

    def show_keybindings(self):
        # Implement keybindings window logic here
        logging.info("Showing keybindings...")
        # You might want to create a new window to display keybindings

    def run(self):
        self.root.mainloop()