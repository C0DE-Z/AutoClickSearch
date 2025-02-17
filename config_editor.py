import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import cv2
import numpy as np
import threading
import time 
from PIL import Image, ImageTk
import pyautogui
import pytesseract
from detection import (CameraManager, load_training_data, detect_screen, 
                       detect_text, process_screen, apply_filters)

CONFIG_FILE = 'config.json'

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    else:
        return {
            "window_title": "",
            "target_text": "",
            "text_similarity_threshold": 0.65,
            "template_match_threshold": 0.75,
            "use_gpu": False,
            "feedback_for_ai": False,
            "screen_cap_toggle": True,
            "save_detected_regions": True,
            "click_delay": 0.0,
            "record_entire_screen": True,
            "hotkeys": {
                "toggle_detection": "ctrl+enter",
                "toggle_print": "ctrl+shift+p",
                "exit": "esc"
            },
            "filters": {
                "enabled": False,
                "grayscale": False,
                "blur": False,
                "blur_kernel_size": 5,
                "invert_colors": False
            },
            "camera": {
                "enabled": False,
                "device_id": 0,
                "resolution": {
                    "width": 1280,
                    "height": 720
                },
                "fps": 30
            },
            "messages": {
                "window_capture": "Captured window at position",
                "window_not_found": "Target window not found",
                "detection_started": "Detection started",
                "detection_stopped": "Detection stopped"
            },
            "display": {
                "enabled": True,
                "window_name": "Detection View",
                "scale": 0.5,
                "show_matches": True,
                "refresh_rate": 30
            },
            "search_target_window": True
        }

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    messagebox.showinfo("Success", "Configuration saved successfully!")

class DetectionManager:
    def __init__(self, config):
        self.config = config
        self.camera_manager = CameraManager(config)
        self.running = False
        self.training_data = load_training_data(config['use_gpu'])

    def run_detection(self, mode):
        if mode == "screen":
            return detect_screen(self.config, self.training_data, self.config['use_gpu'])
        elif mode == "text":
            return detect_text(self.config)
        elif mode == "camera":
            frame = self.camera_manager.get_frame()
            if frame is not None:
                return process_screen(frame, self.training_data, self.config, self.config['use_gpu'])
        elif mode == "combined":
            detect_screen(self.config, self.training_data, self.config['use_gpu'])
            detect_text(self.config)
        return None, None

class MainApplication:
    def __init__(self, root):
        self.root = root
        self.config = load_config()
        self.running = False
        self.camera_active = False
        self.detection_manager = DetectionManager(self.config)
        
        # Create main layout
        self.create_notebook()
        self.create_control_panel()
        self.create_visualization()
        
    def create_notebook(self):
        config = self.config

        # Apply a nicer theme and set window size
        import ttkbootstrap as tb
        notebook = tb.Notebook(self.root, padding=10)
        notebook.pack(pady=10, expand=True, fill='both')

        # General settings
        general_frame = tb.Frame(notebook)
        notebook.add(general_frame, text='General')

        # Remove separator.pack(...) and place it in row=0, spanning two columns:
        separator = tb.Separator(general_frame, bootstyle="info")
        separator.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky='ew')

        # Shift Window Title & subsequent entries down one row
        ttk.Label(general_frame, text="Window Title:").grid(row=1, column=0, padx=10, pady=5, sticky='w')
        window_title = ttk.Entry(general_frame, width=30)
        window_title.grid(row=1, column=1, padx=10, pady=5)
        window_title.insert(0, config['window_title'])

        ttk.Label(general_frame, text="Target Text:").grid(row=2, column=0, padx=10, pady=5, sticky='w')
        target_text = ttk.Entry(general_frame, width=30)
        target_text.grid(row=2, column=1, padx=10, pady=5)
        target_text.insert(0, config['target_text'])

        ttk.Label(general_frame, text="Text Similarity Threshold:").grid(row=3, column=0, padx=10, pady=5, sticky='w')
        text_similarity_threshold = ttk.Entry(general_frame, width=30)
        text_similarity_threshold.grid(row=3, column=1, padx=10, pady=5)
        text_similarity_threshold.insert(0, config['text_similarity_threshold'])

        ttk.Label(general_frame, text="Template Match Threshold:").grid(row=4, column=0, padx=10, pady=5, sticky='w')
        template_match_threshold = ttk.Entry(general_frame, width=30)
        template_match_threshold.grid(row=4, column=1, padx=10, pady=5)
        template_match_threshold.insert(0, config['template_match_threshold'])

        use_gpu = tk.BooleanVar(value=config['use_gpu'])
        ttk.Checkbutton(general_frame, text="Use GPU", variable=use_gpu).grid(row=5, column=0, padx=10, pady=5, sticky='w')

        save_detected_regions = tk.BooleanVar(value=config['save_detected_regions'])
        ttk.Checkbutton(general_frame, text="Save Detected Regions", variable=save_detected_regions).grid(row=6, column=0, padx=10, pady=5, sticky='w')

        ttk.Label(general_frame, text="Click Delay:").grid(row=7, column=0, padx=10, pady=5, sticky='w')
        click_delay = ttk.Entry(general_frame, width=30)
        click_delay.grid(row=7, column=1, padx=10, pady=5)
        click_delay.insert(0, config['click_delay'])

        record_entire_screen = tk.BooleanVar(value=config['record_entire_screen'])
        ttk.Checkbutton(general_frame, text="Record Entire Screen", variable=record_entire_screen).grid(row=8, column=0, padx=10, pady=5, sticky='w')

        search_target_window = tk.BooleanVar(value=config['search_target_window'])
        ttk.Checkbutton(general_frame, text="Search Target Window", variable=search_target_window).grid(row=9, column=0, padx=10, pady=5, sticky='w')

        # Filters settings
        filters_frame = tb.Frame(notebook)
        notebook.add(filters_frame, text='Filters')

        filters_enabled = tk.BooleanVar(value=config['filters']['enabled'])
        ttk.Checkbutton(filters_frame, text="Enable Filters", variable=filters_enabled).grid(row=0, column=0, padx=10, pady=5, sticky='w')

        grayscale = tk.BooleanVar(value=config['filters']['grayscale'])
        ttk.Checkbutton(filters_frame, text="Grayscale", variable=grayscale).grid(row=1, column=0, padx=10, pady=5, sticky='w')

        blur = tk.BooleanVar(value=config['filters']['blur'])
        ttk.Checkbutton(filters_frame, text="Blur", variable=blur).grid(row=2, column=0, padx=10, pady=5, sticky='w')

        ttk.Label(filters_frame, text="Blur Kernel Size:").grid(row=3, column=0, padx=10, pady=5, sticky='w')
        blur_kernel_size = ttk.Entry(filters_frame, width=30)
        blur_kernel_size.grid(row=3, column=1, padx=10, pady=5)
        blur_kernel_size.insert(0, config['filters']['blur_kernel_size'])

        invert_colors = tk.BooleanVar(value=config['filters']['invert_colors'])
        ttk.Checkbutton(filters_frame, text="Invert Colors", variable=invert_colors).grid(row=4, column=0, padx=10, pady=5, sticky='w')

        # Camera settings
        camera_frame = tb.Frame(notebook)
        notebook.add(camera_frame, text='Camera')

        camera_enabled = tk.BooleanVar(value=config['camera']['enabled'])
        ttk.Checkbutton(camera_frame, text="Enable Camera", variable=camera_enabled).grid(row=0, column=0, padx=10, pady=5, sticky='w')

        ttk.Label(camera_frame, text="Device ID:").grid(row=1, column=0, padx=10, pady=5, sticky='w')
        device_id = ttk.Entry(camera_frame, width=30)
        device_id.grid(row=1, column=1, padx=10, pady=5)
        device_id.insert(0, config['camera']['device_id'])

        ttk.Label(camera_frame, text="Resolution Width:").grid(row=2, column=0, padx=10, pady=5, sticky='w')
        resolution_width = ttk.Entry(camera_frame, width=30)
        resolution_width.grid(row=2, column=1, padx=10, pady=5)
        resolution_width.insert(0, config['camera']['resolution']['width'])

        ttk.Label(camera_frame, text="Resolution Height:").grid(row=3, column=0, padx=10, pady=5, sticky='w')
        resolution_height = ttk.Entry(camera_frame, width=30)
        resolution_height.grid(row=3, column=1, padx=10, pady=5)
        resolution_height.insert(0, config['camera']['resolution']['height'])

        ttk.Label(camera_frame, text="FPS:").grid(row=4, column=0, padx=10, pady=5, sticky='w')
        fps = ttk.Entry(camera_frame, width=30)
        fps.grid(row=4, column=1, padx=10, pady=5)
        fps.insert(0, config['camera']['fps'])

        # Display settings
        display_frame = tb.Frame(notebook)
        notebook.add(display_frame, text='Display')

        display_enabled = tk.BooleanVar(value=config['display']['enabled'])
        ttk.Checkbutton(display_frame, text="Enable Display", variable=display_enabled).grid(row=0, column=0, padx=10, pady=5, sticky='w')

        ttk.Label(display_frame, text="Window Name:").grid(row=1, column=0, padx=10, pady=5, sticky='w')
        window_name = ttk.Entry(display_frame, width=30)
        window_name.grid(row=1, column=1, padx=10, pady=5)
        window_name.insert(0, config['display']['window_name'])

        ttk.Label(display_frame, text="Scale:").grid(row=2, column=0, padx=10, pady=5, sticky='w')
        scale = ttk.Entry(display_frame, width=30)
        scale.grid(row=2, column=1, padx=10, pady=5)
        scale.insert(0, config['display']['scale'])

        show_matches = tk.BooleanVar(value=config['display']['show_matches'])
        ttk.Checkbutton(display_frame, text="Show Matches", variable=show_matches).grid(row=3, column=0, padx=10, pady=5, sticky='w')

        ttk.Label(display_frame, text="Refresh Rate:").grid(row=4, column=0, padx=10, pady=5, sticky='w')
        refresh_rate = ttk.Entry(display_frame, width=30)
        refresh_rate.grid(row=4, column=1, padx=10, pady=5)
        refresh_rate.insert(0, config['display']['refresh_rate'])

        # Detection Modes settings
        detection_frame = tb.Frame(notebook)
        notebook.add(detection_frame, text='Detection Modes')

        detection_mode_var = tk.StringVar(value=config.get('detection_mode', 'screen'))
        ttk.Radiobutton(detection_frame, text="Screen", variable=detection_mode_var, value="screen").grid(row=0, column=0, padx=10, pady=5, sticky='w')
        ttk.Radiobutton(detection_frame, text="Text", variable=detection_mode_var, value="text").grid(row=1, column=0, padx=10, pady=5, sticky='w')
        ttk.Radiobutton(detection_frame, text="Camera", variable=detection_mode_var, value="camera").grid(row=2, column=0, padx=10, pady=5, sticky='w')
        ttk.Radiobutton(detection_frame, text="Combined", variable=detection_mode_var, value="combined").grid(row=3, column=0, padx=10, pady=5, sticky='w')

        # Save button
        def save():
            config['window_title'] = window_title.get()
            config['target_text'] = target_text.get()
            config['text_similarity_threshold'] = float(text_similarity_threshold.get())
            config['template_match_threshold'] = float(template_match_threshold.get())
            config['use_gpu'] = use_gpu.get()
            config['save_detected_regions'] = save_detected_regions.get()
            config['click_delay'] = float(click_delay.get())
            config['record_entire_screen'] = record_entire_screen.get()
            config['filters']['enabled'] = filters_enabled.get()
            config['filters']['grayscale'] = grayscale.get()
            config['filters']['blur'] = blur.get()
            config['filters']['blur_kernel_size'] = int(blur_kernel_size.get())
            config['filters']['invert_colors'] = invert_colors.get()
            config['camera']['enabled'] = camera_enabled.get()
            config['camera']['device_id'] = int(device_id.get())
            config['camera']['resolution']['width'] = int(resolution_width.get())
            config['camera']['resolution']['height'] = int(resolution_height.get())
            config['camera']['fps'] = int(fps.get())
            config['display']['enabled'] = display_enabled.get()
            config['display']['window_name'] = window_name.get()
            config['display']['scale'] = float(scale.get())
            config['display']['show_matches'] = show_matches.get()
            config['display']['refresh_rate'] = int(refresh_rate.get())
            config['detection_mode'] = detection_mode_var.get()
            config['search_target_window'] = search_target_window.get()
            save_config(config)

        save_button = tb.Button(self.root, text="Save Configuration", bootstyle="success", command=save)
        save_button.pack(pady=20)

    def create_control_panel(self):
        control_frame = ttk.LabelFrame(self.root, text="Control Panel")
        control_frame.pack(pady=10, padx=10, fill='x')
        
        # Detection mode selection
        mode_frame = ttk.LabelFrame(control_frame, text="Detection Mode")
        mode_frame.pack(pady=5, padx=5, fill='x')
        
        self.mode_var = tk.StringVar(value='screen')
        ttk.Radiobutton(mode_frame, text="Screen", variable=self.mode_var, value="screen").pack(side='left', padx=5)
        ttk.Radiobutton(mode_frame, text="Text", variable=self.mode_var, value="text").pack(side='left', padx=5)
        ttk.Radiobutton(mode_frame, text="Camera", variable=self.mode_var, value="camera").pack(side='left', padx=5)
        ttk.Radiobutton(mode_frame, text="Combined", variable=self.mode_var, value="combined").pack(side='left', padx=5)
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=5, fill='x')
        
        ttk.Button(button_frame, text="Start Detection", command=self.start_detection).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Stop Detection", command=self.stop_detection).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Toggle Camera", command=self.toggle_camera).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Toggle Display", command=self.toggle_display).pack(side='left', padx=5)
        
    def create_visualization(self):
        self.viz_frame = ttk.LabelFrame(self.root, text="Visualization")
        self.viz_frame.pack(pady=10, padx=10, fill='both', expand=True)
        
        self.canvas = tk.Canvas(self.viz_frame)
        self.canvas.pack(fill='both', expand=True)
        
    def start_detection(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self.detection_loop, daemon=True).start()
            
    def stop_detection(self):
        self.running = False
        
    def toggle_camera(self):
        self.camera_active = not self.camera_active
        if self.camera_active:
            threading.Thread(target=self.camera_loop, daemon=True).start()
            
    def toggle_display(self):
        self.config['display']['enabled'] = not self.config['display']['enabled']
        
    def detection_loop(self):
        while self.running:
            mode = self.mode_var.get()
            result = self.detection_manager.run_detection(mode)
            if isinstance(result, tuple):
                frame, _ = result
            else:
                frame = result
            if frame is not None:
                self.update_visualization(frame)
            time.sleep(0.1)
            
    def camera_loop(self):
        cap = cv2.VideoCapture(self.config['camera']['device_id'])
        while self.camera_active:
            ret, frame = cap.read()
            if ret:
                self.update_visualization(frame)
            time.sleep(1/30)
        cap.release()
        
    def update_visualization(self, frame):
        if not self.config['display']['enabled']:
            return

        # Ensure the frame is a valid image
        if frame is None or not isinstance(frame, np.ndarray):
            return

        # Ensure the image is in the correct format
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Convert CV2 frame to PhotoImage
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.config['display']['scale'] != 1.0:
            h, w = frame.shape[:2]
            new_h = int(h * self.config['display']['scale'])
            new_w = int(w * self.config['display']['scale'])
            frame = cv2.resize(frame, (new_w, new_h))

        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=image)

        self.canvas.config(width=photo.width(), height=photo.height())
        self.canvas.create_image(0, 0, image=photo, anchor='nw')
        self.canvas.image = photo

def create_ui():
    root = tk.Tk()
    root.title("AutoClickSearch Control Panel")
    root.geometry("800x600")
    app = MainApplication(root)
    root.mainloop()

if __name__ == "__main__":
    create_ui()
