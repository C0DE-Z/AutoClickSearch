import cv2
import numpy as np
import autoit as pyautoit
import pyautogui 
import pytesseract
import os
import time
import threading
import keyboard
import win32con
import win32api
import pygetwindow as gw
from rich.console import Console
from difflib import SequenceMatcher
import json
from detection import (CameraManager, load_training_data, detect_screen, 
                       detect_text, process_screen, apply_filters)

# Global variables
use_gpu = False  # Default value
running = False
print_enabled = True
training_data = []
console = Console()  # Move console to global scope

def load_config():
    default_config = {
        "camera": {
            "enabled": False,
            "device_id": 0,
            "resolution": {
                "width": 1280,
                "height": 720
            },
            "fps": 30
        }
    }
    
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            # Ensure camera config exists
            if 'camera' not in config:
                config['camera'] = default_config['camera']
            # Save updated config
            with open('config.json', 'w') as f:
                json.dump(config, f, indent=4)
            return config
    except FileNotFoundError:
        print("[ERROR] Config file not found. Using defaults.")
        return default_config
    except json.JSONDecodeError:
        print("[ERROR] Invalid config file. Using defaults.")
        return default_config

def main():
    config = load_config()
    global use_gpu, running, print_enabled

    TEMPLATE_MATCH_THRESHOLD = config['template_match_threshold']
    TEXT_SIMILARITY_THRESHOLD = config['text_similarity_threshold']
    lock = threading.Lock()

    # Set use_gpu from config
    use_gpu = config['use_gpu']
    try:
        cv2.cuda_GpuMat()
    except cv2.error as e:
        if "No CUDA support" in str(e):
            console.log("[bold red]OpenCV was compiled without CUDA support. Falling back to CPU.[/bold red]")
            use_gpu = False

    if use_gpu:
        console.log("[bold green]GPU acceleration is enabled![/bold green]")
    else:
        console.log("[bold red]Using CPU processing.[/bold red]")

    camera_manager = CameraManager(config)

    training_data = load_training_data(use_gpu)

    def run_detection_mode():
        mode = config.get('detection_mode', 'screen')
        if mode == 'screen':
            detect_screen(config, training_data, use_gpu)
        elif mode == 'text':
            detect_text(config)
        elif mode == 'camera':
            # Start camera if not already started
            if not camera_manager.camera:
                camera_manager.start()
            # Optionally show camera detection logic or feed
            detect_screen(config, training_data, use_gpu)  # Or any custom camera-based detection
        elif mode == 'combined':
            detect_screen(config, training_data, use_gpu)
            detect_text(config)

    # In the main loop or detection thread, call run_detection_mode():
    def start_detection():
        global running
        while running:
            run_detection_mode()
            time.sleep(0.1)

    def toggle_running():
        global running
        running = not running
        if running:
            print("[ACTION] Detection started.")
            threading.Thread(target=start_detection, daemon=True).start()
        else:
            print("[ACTION] Detection stopped.")

    def toggle_print():
        global print_enabled
        print_enabled = not print_enabled
        status = "enabled" if print_enabled else "disabled"
        print(f"[ACTION] Print statements have been {status}.")

    def adjust_threshold(level):
        global TEMPLATE_MATCH_THRESHOLD
        TEMPLATE_MATCH_THRESHOLD = level
        print(f"[INFO] Template match threshold adjusted to {level}.")

    def toggle_display():
        config['display']['enabled'] = not config['display']['enabled']
        if not config['display']['enabled']:
            cv2.destroyAllWindows()
        print(f"[INFO] Display window {'enabled' if config['display']['enabled'] else 'disabled'}")

    def toggle_filters():
        config['filters']['enabled'] = not config['filters']['enabled']
        print(f"[INFO] Filters {'enabled' if config['filters']['enabled'] else 'disabled'}")

    def show_camera_preview():
        print("[INFO] Showing camera preview. Press 'q' to continue.")
        while True:
            frame = camera_manager.get_frame()
            if frame is not None:
                display = frame.copy()
                
                # Add detection results based on mode
                if mode == "1" or mode == "3":
                    with lock:
                        detected, position = process_screen(frame, training_data, config, use_gpu)
                    if detected:
                        x, y = position
                        # Get template size for box
                        template_w = training_data[0][1].shape[1]
                        template_h = training_data[0][1].shape[0]
                        # Draw rectangle around detected area
                        cv2.rectangle(display, 
                                    (x - template_w//2, y - template_h//2),
                                    (x + template_w//2, y + template_h//2),
                                    (0, 255, 0), 2)
                        cv2.putText(display, f"Pattern Found", (x - template_w//2, y - template_h//2 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if mode == "2" or mode == "3":
                    # Get all text from image
                    text = pytesseract.image_to_string(frame)
                    words = text.strip().split('\n')
                    
                    # Add background for text overlay
                    h, w = display.shape[:2]
                    overlay = display.copy()
                    cv2.rectangle(overlay, (5, 5), (400, 30 * len(words) + 40), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
                    
                    # Display all detected text
                    y_pos = 30
                    for word in words:
                        if word.strip():
                            cv2.putText(display, f"Text: {word}", (10, y_pos),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                            y_pos += 30
                            if print_enabled:
                                print(f"[INFO] Found text: {word}")

                cv2.imshow("Camera Feed", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def toggle_camera():
        config['camera']['enabled'] = not config['camera']['enabled']
        if config['camera']['enabled']:
            if camera_manager.start():
                print("[INFO] Camera enabled")
                print("\nChoose detection mode for camera:")
                print("1. Visual Pattern Detection")
                print("2. Text Detection")
                print("3. Combined Detection")
                detection_mode = input("Select mode (1-3): ").strip()
                if detection_mode in ["1", "2", "3"]:
                    global mode
                    mode = detection_mode
                    config['display']['enabled'] = True
                    show_camera_preview()
                    toggle_running()
                else:
                    print("[ERROR] Invalid mode selected")
                    camera_manager.stop()
                    config['camera']['enabled'] = False
            else:
                config['camera']['enabled'] = False
                print("[ERROR] Failed to enable camera")
        else:
            camera_manager.stop()
            print("[INFO] Camera disabled")

    for i in range(1, 10):
        keyboard.add_hotkey(f'alt+{i}', lambda level=i: adjust_threshold(level * 0.1))
    keyboard.add_hotkey(config['hotkeys']['toggle_detection'], toggle_running)
    keyboard.add_hotkey(config['hotkeys']['toggle_print'], toggle_print)
    keyboard.add_hotkey('ctrl+shift+d', toggle_display)
    keyboard.add_hotkey('ctrl+shift+f', toggle_filters)
    keyboard.add_hotkey('ctrl+shift+c', toggle_camera)

    if config['camera']['enabled']:
        camera_manager.start()

    print(f"[INFO] Press {config['hotkeys']['toggle_detection']} to start/stop detection.")
    print(f"[INFO] Press {config['hotkeys']['toggle_print']} to toggle messages.")
    print(f"[INFO] Press Ctrl+Shift+C to toggle camera")
    print(f"[INFO] Press {config['hotkeys']['exit']} to exit.")

    while True:
        if keyboard.is_pressed(config['hotkeys']['exit']):
            print("[INFO] Exiting...")
            cleanup()
            break

if __name__ == "__main__":
    main()
