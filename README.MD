# AutoClickSearch

A configurable automated pattern and text detection tool with mouse control capabilities.

## Features

- Visual pattern detection using OpenCV
- OCR text detection using Pytesseract
- GPU acceleration support
- Configurable hotkeys and settings
- Training data management
- Real-time window tracking
- Automated mouse control
- Multiple detection modes

## Requirements

- Python 3.7+
- OpenCV
- PyAutoGUI
- Pytesseract
- AutoIT
- Rich Console
- CUDA (optional, for GPU acceleration)

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Install Tesseract OCR
   - Windows: Download from [GitHub Tesseract Release](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr`

## Setup

1. Configure settings in `config.json`:
   ```json
   {
     "window_title": "Your Window Name",
     "target_text": "Text to Find",
     "use_gpu": false
   }
   ```

2. Create training data:
   - Create folder named `training_data`
   - Take screenshots of what you want to click
   - Save them as PNG files in the training_data folder
   - Name format: `detection_*.png`

3. Choose Detection Mode:
   - Mode 1: Visual Pattern (Uses training images)
   - Mode 2: Text Detection (Uses OCR)
   - Mode 3: Combined (Both methods)
   - Mode 4: Clean training data

## Configuration Editor
With the built-in UI, you can easily modify settings such as detection mode, thresholds, filters, and camera options.
To launch the editor:
1. Install ttkbootstrap if not already installed:  
   pip install ttkbootstrap
2. Run:
   python config_editor.py
3. Adjust your settings, then click “Save Configuration.”  
4. Restart or reload the main script to apply your changes.

## Usage

1. Run the program:
   ```bash
   python main.py
   ```

2. Controls:
   - `Ctrl+Enter`: Start/Stop detection
   - `Ctrl+Shift+P`: Toggle debug messages
   - `Alt+1-9`: Adjust detection sensitivity
   - `Esc`: Exit program

3. Performance Tips:
   - Enable GPU in config for faster detection
   - Use more training images for better accuracy
   - Adjust threshold if detection is unreliable
   - Keep window visible and unobstructed

## Troubleshooting

- Detection not working:
  - Verify window title matches exactly
  - Check training images quality
  - Increase sensitivity with Alt+1-9
  - Enable debug messages for more info

- Program crashes:
  - Verify Tesseract installation
  - Check config.json syntax
  - Ensure training_data folder exists
  - Run as administrator if needed

## Notes

- Only one window/process can be monitored at a time
- CPU mode is significantly slower than GPU
- Success rate depends on training data quality
- Keep training images clear and consistent
