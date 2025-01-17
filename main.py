import cv2
from cv2 import cuda
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

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

config = load_config()
console = Console()

running = False
print_enabled = True
training_data = []
TEMPLATE_MATCH_THRESHOLD = config['template_match_threshold']
TEXT_SIMILARITY_THRESHOLD = config['text_similarity_threshold']
lock = threading.Lock()

use_gpu = config['use_gpu']
if use_gpu:
    console.log("[bold green]GPU acceleration is enabled![/bold green]")
else:
    console.log("[bold red]Using CPU processing.[/bold red]")

console.log("[bold cyan]Choose detection mode:[/bold cyan]")
console.log("1. Visual Pattern Detection")
console.log("2. Text Detection")
console.log("3. Combined Detection")
console.log("4. Manage Training Data")
mode = input("Select mode (1-4): ").strip()

def load_training_data():
    for file in os.listdir("training_data"):
        if file.startswith("shake_") and file.endswith(".png"):
            img = cv2.imread(os.path.join("training_data", file), cv2.IMREAD_COLOR)
            if use_gpu:
                img = cv2.cuda_GpuMat().upload(img)
            training_data.append((file, img))

load_training_data()

def is_text_similar(text, threshold=TEXT_SIMILARITY_THRESHOLD):
    return SequenceMatcher(None, text.lower(), config['target_text'].lower()).ratio() >= threshold

def get_target_window_screenshot():
    try:
        window = gw.getWindowsWithTitle(config['window_title'])[0]
        x, y, w, h = window.left, window.top, window.width, window.height
        screen = np.array(pyautogui.screenshot(region=(x, y, w, h)))
        if print_enabled:
            print(f"[INFO] {config['messages']['window_capture']} ({x}, {y}, {w}, {h})")
        return screen, x, y
    except IndexError:
        if print_enabled:
            print(f"[ERROR] {config['messages']['window_not_found']}")
        return None, None, None

def process_screen(screen, templates):
    gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    if use_gpu:
        gray_screen = cv2.cuda_GpuMat().upload(gray_screen)

    for filename, template in templates:
        template_gray = cv2.cvtColor(template.download() if use_gpu else template, cv2.COLOR_BGR2GRAY)
        result = cuda.matchTemplate(gray_screen, template_gray, cv2.TM_CCOEFF_NORMED) if use_gpu else cv2.matchTemplate(gray_screen, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= TEMPLATE_MATCH_THRESHOLD:
            x, y = max_loc
            w, h = template_gray.shape[::-1]
            locked_position = (x + w // 2, y + h // 2)
            if print_enabled:
                print(f"[INFO] Template '{filename}' matched with confidence {max_val:.2f} at position {locked_position}.")
            return True, locked_position

    if print_enabled:
        print("[INFO] No templates matched.")
    return False, None

def detect_screen():
    screen, x_offset, y_offset = get_target_window_screenshot()
    if screen is None:
        return False

    with lock:
        detected, position = process_screen(screen, training_data)

    if detected:
        if print_enabled:
            print(f"[ACTION] Target detected at {position}")
        
        win32api.SetCursorPos((position[0] + x_offset, position[1] + y_offset))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
        
        if config['save_detected_regions']:
            save_detection(screen, position)
        return True
    return False

def save_detection(screen, position):
    x, y = position
    w, h = training_data[0][1].shape[1], training_data[0][1].shape[0]
    half_w, half_h = w // 2, h // 2
    center_x, center_y = x + half_w, y + half_h
    detected_region = screen[max(0, center_y - half_h):center_y + half_h, 
                           max(0, center_x - half_w):center_x + half_w]
    timestamp = int(time.time())
    filename = f"{config['training_data_prefix']}{timestamp}.png"
    filepath = os.path.join("training_data", filename)
    cv2.imwrite(filepath, cv2.cvtColor(detected_region, cv2.COLOR_RGB2BGR))
    if print_enabled:
        print(f"[INFO] Saved detection as {filename}")

def detect_text():
    screen, _, _ = get_target_window_screenshot()
    if screen is None:
        return False

    text = pytesseract.image_to_string(screen)
    if print_enabled:
        print(f"[INFO] OCR detected text: {text.strip()}")
    if is_text_similar(text):
        if print_enabled:
            print("[ACTION] Text matches target! Triggering action.")
        return True
    return False

def cleanup_training_data():
    if not training_data:
        if print_enabled:
            print("[ERROR] No training data to evaluate.")
        return
    
    reference_image = cv2.imread("reference_simulator.png", cv2.IMREAD_COLOR)
    if reference_image is None:
        if print_enabled:
            print("[ERROR] Reference image not found.")
        return

    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    for filename, img in training_data:
        gray_image = cv2.cvtColor(img.download() if use_gpu else img, cv2.COLOR_BGR2GRAY)
        result = cuda.matchTemplate(reference_gray, gray_image, cv2.TM_CCOEFF_NORMED) if use_gpu else cv2.matchTemplate(reference_gray, gray_image, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        if max_val < TEMPLATE_MATCH_THRESHOLD:
            os.remove(os.path.join("training_data", filename))
            if print_enabled:
                print(f"[INFO] Removed {filename} due to low confidence.")

def start_detection():
    global running
    while running:
        if mode == "1":
            detect_screen()
        elif mode == "2":
            detect_text()
        elif mode == "3":
            detect_screen()
            detect_text()
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

if mode == "4":
    cleanup_training_data()
else:
    for i in range(1, 10):
        keyboard.add_hotkey(f'alt+{i}', lambda level=i: adjust_threshold(level * 0.1))
    keyboard.add_hotkey(config['hotkeys']['toggle_detection'], toggle_running)
    keyboard.add_hotkey(config['hotkeys']['toggle_print'], toggle_print)
    print(f"[INFO] Press {config['hotkeys']['toggle_detection']} to start/stop detection.")
    print(f"[INFO] Press {config['hotkeys']['toggle_print']} to toggle messages.")
    print(f"[INFO] Press {config['hotkeys']['exit']} to exit.")

    while True:
        if keyboard.is_pressed(config['hotkeys']['exit']):
            print("[INFO] Exiting...")
            break
