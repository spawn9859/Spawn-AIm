import customtkinter as ctk
import cv2
from PIL import Image, ImageTk, ImageDraw
from .components import create_checkboxes, create_sliders, create_comboboxes, create_buttons, create_labels
import numpy as np

GREEN_COLOR = "#15AC8b"
GREY_COLOR = "#808080"

class MainWindow:
    def __init__(self, config_manager):
        self.root = ctk.CTk()
        self.root.title("Spawn-Aim")
        self.root.geometry("600x850+40+40")
        self.root.resizable(width=False, height=False)
        
        self.config_manager = config_manager
        self.setup_ui()

    def setup_ui(self):
        self.checkboxes = create_checkboxes(self.root, self.config_manager, self.update_setting)
        self.sliders = create_sliders(self.root, self.config_manager, self.config_manager.update_setting)
        self.comboboxes = create_comboboxes(self.root, self.config_manager, self.config_manager.update_setting)
        self.buttons = create_buttons(self.root, self)
        self.labels = create_labels(self.root)

        # Create preview image
        self.preview_image = ctk.CTkImage(
            light_image=Image.open("preview.png"),
            dark_image=Image.open("preview.png"),
            size=(240, 240)
        )
        self.preview_label = ctk.CTkLabel(self.root, image=self.preview_image, text="")
        self.preview_label.place(x=240, y=40)

        # Add FOV toggle checkbox
        self.fov_visible = ctk.StringVar(value="off")
        self.fov_checkbox = ctk.CTkCheckBox(
            self.root,
            text="Show FOV",
            variable=self.fov_visible,
            onvalue="on",
            offvalue="off",
            command=self.toggle_fov_visibility,
            fg_color=GREEN_COLOR,
            hover_color="#0F0"
        )
        self.fov_checkbox.place(x=320, y=280)  # Adjust position as needed

    def toggle_fov_visibility(self):
        # This method will be called when the FOV checkbox is toggled
        pass  # The visibility is controlled by the checkbox state, so we don't need to do anything here

    def update_preview(self, frame, coordinates, targets, distances):
        if self.config_manager.get_setting("preview") == "on":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (240, 240))
            image = Image.fromarray(frame)
            draw = ImageDraw.Draw(image)
            
            # Draw bounding boxes
            for coord in coordinates:
                x1, y1, x2, y2 = [int(c * 240 / self.config_manager.get_setting("width")) for c in coord]
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            
            # Draw aim lines
            center_x, center_y = 120, 120  # Center of the preview image
            if len(targets) > 0:
                min_distance_index = np.argmin(distances)
                for i, (target_x, target_y) in enumerate(targets):
                    color = "yellow" if i == min_distance_index else "blue"
                    scaled_x = int(target_x * 240 / self.config_manager.get_setting("width")) + center_x
                    scaled_y = int(target_y * 240 / self.config_manager.get_setting("height")) + center_y
                    draw.line((center_x, center_y, scaled_x, scaled_y), fill=color, width=2)
                    draw.ellipse((scaled_x-3, scaled_y-3, scaled_x+3, scaled_y+3), fill=color)

            # Draw FOV circle if enabled
            if self.fov_visible.get() == "on" and self.config_manager.get_setting("fov_enabled") == "on":
                fov_size = self.config_manager.get_setting("fov_size")
                scaled_fov_size = int(fov_size * 240 / self.config_manager.get_setting("width"))
                draw.ellipse(
                    [center_x - scaled_fov_size, center_y - scaled_fov_size,
                     center_x + scaled_fov_size, center_y + scaled_fov_size],
                    outline="red", width=2
                )

            photo = ImageTk.PhotoImage(image=image)
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo

    def update_setting(self, key, value):
        self.config_manager.update_setting(key, value)
        if key == "preview":
            self.toggle_preview(value)

    def toggle_preview(self, value):
        if value == "on":
            self.preview_label.place(x=240, y=40)
        else:
            self.preview_label.place_forget()

    def update_fps_label(self, fps):
        self.labels['fps'].configure(text=f"Fps: {round(fps)}")

    

    def toggle_overlay(self):
        import win32gui, win32con
        if self.overlay is None:
            self.overlay = ctk.CTkToplevel(self.root)
            self.overlay.geometry(f"{self.settings['width']}x{self.settings['height']}+{int(self.root.winfo_screenwidth()/2 - self.settings['width']/2)}+{int(self.root.winfo_screenheight()/2 - self.settings['height']/2)}")
            self.overlay.overrideredirect(True)
            self.overlay.attributes("-topmost", True)
            self.overlay.configure(bg="#000000")
            self.overlay.attributes("-alpha", 0.6)
            
            self.canvas = ctk.CTkCanvas(self.overlay, width=self.settings['width'], height=self.settings['height'], highlightthickness=0)
            self.canvas.pack()

            # Make the window click-through
            hwnd = self.overlay.winfo_id()
            styles = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            styles = styles | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, styles)
            win32gui.SetLayeredWindowAttributes(hwnd, 0, 255, win32con.LWA_ALPHA)
        else:
            self.overlay.destroy()
            self.overlay = None
            self.canvas = None

    def update_overlay(self, coordinates):
        if self.overlay and self.canvas:
            self.canvas.delete("all")
            for coord in coordinates:
                x1, y1, x2, y2 = map(int, coord)
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)

    def toggle_auto_aim(self):
        self.checkboxes['auto_aim'].toggle()

    def reload_model(self):
        # This should call the load_model function from main.py
        # You might need to pass this function as a callback when creating the MainWindow
        pass

    def show_keybindings(self):
        # Implementation for showing keybindings window
        pass

    def run(self):
        self.root.mainloop()