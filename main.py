import cv2
from cv2 import cuda
import numpy  as np
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
import config_editor

# Global variables
use_gpu = False  # Default value
running = False
print_enabled = True
training_data = []
console = Console()  # Move console to global scope

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

def load_training_data():
    global use_gpu
    for file in os.listdir("training_data"):
        if file.endswith(".png"):
            img = cv2.imread(os.path.join("training_data", file), cv2.IMREAD_COLOR)
            if use_gpu:
                try:
                    img = cv2.cuda_GpuMat().upload(img)
                except cv2.error as e:
                    if "No CUDA support" in str(e):
                        console.log("[bold red]OpenCV was compiled without CUDA support. Falling back to CPU.[/bold red]")
                        use_gpu = False
                    else:
                        raise e
            training_data.append((file, img))

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

    def apply_filters(screen):
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

    def get_screenshot():
        if config['camera']['enabled']:
            frame = camera_manager.get_frame()
            if frame is not None:
                if print_enabled:
                    print("[INFO] Captured camera frame")
                return frame, 0, 0
            return None, None, None
        elif config['record_entire_screen']:
            screen = np.array(pyautogui.screenshot())
            if print_enabled:
                print("[INFO] Captured entire screen.")
            return screen, 0, 0
        else:
            return get_target_window_screenshot()

    def show_detection_window(screen, matches=None):
        if not config['display']['enabled']:
            return
            
        display = screen.copy()
        if matches and config['display']['show_matches']:
            for pos in matches:
                x, y = pos
                # Get template size for box
                template_w = training_data[0][1].shape[1]
                template_h = training_data[0][1].shape[0]
                # Draw rectangle around detected area
                cv2.rectangle(display, 
                            (x - template_w//2, y - template_h//2),
                            (x + template_w//2, y + template_h//2),
                            (0, 255, 0), 2)
        
        if config['display']['scale'] != 1.0:
            h, w = display.shape[:2]
            new_h, new_w = int(h * config['display']['scale']), int(w * config['display']['scale'])
            display = cv2.resize(display, (new_w, new_h))
        
        cv2.imshow("Camera Feed", display)
        cv2.waitKey(1)

    def cleanup():
        if config['display']['enabled']:
            cv2.destroyAllWindows()
        camera_manager.stop()

    def process_screen(screen, templates):
        screen = apply_filters(screen)
        gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY) if not config['filters']['grayscale'] else screen
        if use_gpu:
            gray_screen = cv2.cuda_GpuMat().upload(gray_screen)

        matches = []
        for filename, template in templates:
            template_gray = cv2.cvtColor(template.download() if use_gpu else template, cv2.COLOR_BGR2GRAY)
            result = cuda.matchTemplate(gray_screen, template_gray, cv2.TM_CCOEFF_NORMED) if use_gpu else cv2.matchTemplate(gray_screen, template_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val >= TEMPLATE_MATCH_THRESHOLD:
                x, y = max_loc
                w, h = template_gray.shape[::-1]
                locked_position = (x + w // 2, y + h // 2)
                matches.append(locked_position)
                if print_enabled:
                    print(f"[INFO] Template '{filename}' matched with confidence {max_val:.2f} at position {locked_position}.")
                return True, locked_position

        if config['display']['enabled']:
            show_detection_window(screen, matches)

        if print_enabled:
            print("[INFO] No templates matched.")
        return False, None

    def detect_screen():
        screen, x_offset, y_offset = get_screenshot()
        if screen is None:
            return False

        if print_enabled:
            print("[DEBUG] Screenshot captured. Shape:", screen.shape)

        with lock:
            detected, position = process_screen(screen, training_data)

        if config['display']['enabled'] and not detected:
            show_detection_window(screen)

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
        filename = f"detected_{timestamp}.png"
        filepath = os.path.join("training_data", filename)
        cv2.imwrite(filepath, cv2.cvtColor(detected_region, cv2.COLOR_RGB2BGR))
        if print_enabled:
            print(f"[INFO] Saved detection as {filename}")

    def detect_text():
        screen, _, _ = get_screenshot()
        if screen is None:
            return False

        if print_enabled:
            print("[DEBUG] Screenshot captured for OCR. Shape:", screen.shape)

        screen = apply_filters(screen)
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

    def run_detection_mode():
        mode = config.get('detection_mode', 'screen')
        if mode == 'screen':
            detect_screen()
        elif mode == 'text':
            detect_text()
        elif mode == 'camera':
            # Start camera if not already started
            if not camera_manager.camera:
                camera_manager.start()
            # Optionally show camera detection logic or feed
            detect_screen()  # Or any custom camera-based detection
        elif mode == 'combined':
            detect_screen()
            detect_text()

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
                        detected, position = process_screen(frame, training_data)
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
