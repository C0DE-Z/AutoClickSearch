import cv2
import numpy as np
import pytesseract
from difflib import SequenceMatcher
import pyautogui
import win32api
import win32con
import time
import os

class CameraManager:
    def __init__(self, config):
        self.config = config
        self.camera = None
        
    def start(self):
        if self.config['camera']['enabled'] and not self.camera:
            self.camera = cv2.VideoCapture(self.config['camera']['device_id'])
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['resolution']['width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['resolution']['height'])
            self.camera.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
            if not self.camera.isOpened():
                print("[ERROR] Failed to open camera")
                self.camera = None
                return False
            return True
        return False
    
    def stop(self):
        if self.camera:
            self.camera.release()
            self.camera = None
    
    def get_frame(self):
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                return frame
        return None

def load_training_data(use_gpu):
    training_data = []
    for file in os.listdir("training_data"):
        if file.endswith(".png"):
            img = cv2.imread(os.path.join("training_data", file), cv2.IMREAD_COLOR)
            if use_gpu:
                try:
                    img = cv2.cuda_GpuMat().upload(img)
                except cv2.error as e:
                    if "No CUDA support" in str(e):
                        print("[bold red]OpenCV was compiled without CUDA support. Falling back to CPU.[/bold red]")
                        use_gpu = False
                    else:
                        raise e
            training_data.append((file, img))
    return training_data

def is_text_similar(text, target_text, threshold):
    return SequenceMatcher(None, text.lower(), target_text.lower()).ratio() >= threshold

def apply_filters(screen, config):
    if not config['filters']['enabled']:
        return screen

    if config['filters']['grayscale']:
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    
    if config['filters']['blur']:
        kernel_size = config['filters']['blur_kernel_size']
        screen = cv2.GaussianBlur(screen, (kernel_size, kernel_size), 0)
    
    if config['filters']['invert_colors']:
        screen = cv2.bitwise_not(screen)
    
    return screen

def process_screen(screen, templates, config, use_gpu):
    screen = apply_filters(screen, config)
    gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY) if not config['filters']['grayscale'] else screen
    if use_gpu:
        try:
            gray_screen = cv2.cuda_GpuMat().upload(gray_screen)
        except cv2.error as e:
            if "No CUDA support" in str(e):
                print("[bold red]OpenCV was compiled without CUDA support. Falling back to CPU.[/bold red]")
                use_gpu = False
            else:
                raise e

    matches = []
    for filename, template in templates:
        template_gray = cv2.cvtColor(template.download() if use_gpu else template, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray_screen, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= config['template_match_threshold']:
            x, y = max_loc
            w, h = template_gray.shape[::-1]
            locked_position = (x + w // 2, y + h // 2)
            matches.append(locked_position)
            print(f"[INFO] Template '{filename}' matched with confidence {max_val:.2f} at position {locked_position}.")
            return True, locked_position

    print("[INFO] No templates matched.")
    return False, None

def detect_screen(config, training_data, use_gpu):
    screen, x_offset, y_offset = get_screenshot(config)
    if screen is None:
        return False

    detected, position = process_screen(screen, training_data, config, use_gpu)

    if detected:
        print(f"[ACTION] Target detected at {position}")
        win32api.SetCursorPos((position[0] + x_offset, position[1] + y_offset))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
        
        if config['save_detected_regions']:
            save_detection(screen, position, training_data)
        return True
    return False

def detect_text(config):
    screen, _, _ = get_screenshot(config)
    if screen is None:
        return False

    screen = apply_filters(screen, config)
    text = pytesseract.image_to_string(screen)
    print(f"[INFO] OCR detected text: {text.strip()}")
    if is_text_similar(text, config['target_text'], config['text_similarity_threshold']):
        print("[ACTION] Text matches target! Triggering action.")
        return True
    return False

def get_screenshot(config):
    if config['camera']['enabled']:
        frame = camera_manager.get_frame()
        if frame is not None:
            print("[INFO] Captured camera frame")
            return frame, 0, 0
        return None, None, None
    elif config['record_entire_screen']:
        screen = np.array(pyautogui.screenshot())
        print("[INFO] Captured entire screen.")
        return screen, 0, 0
    else:
        return get_target_window_screenshot(config)

def get_target_window_screenshot(config):
    try:
        window = gw.getWindowsWithTitle(config['window_title'])[0]
        x, y, w, h = window.left, window.top, window.width, window.height
        screen = np.array(pyautogui.screenshot(region=(x, y, w, h)))
        print(f"[INFO] {config['messages']['window_capture']} ({x}, {y}, {w}, {h})")
        return screen, x, y
    except IndexError:
        print(f"[ERROR] {config['messages']['window_not_found']}")
        return None, None, None

def save_detection(screen, position, training_data):
    x, y = position
    w, h = training_data[0][1].shape[1], training_data[0][1].shape[0]
    half_w, half_h = w // 2, h // 2
    center_x, center_y = x + half_w, y + half_h
    detected_region = screen[max(0, center_y - half_h):center_y + half_h, 
                           max(0, center_x - half_w):center_x + half_w]
    timestamp = int(time.time())
    filename = f"detected_{timestamp}.png"
    filepath = os.path.join("training_data", filename)
    cv2.imwrite(filepath, cv2.cvtColor(detected_region, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Saved detection as {filename}")
