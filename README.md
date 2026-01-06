# Accessibility Keyboard

Hands-free mouse, typing, and commands on Windows using only a webcam and microphone. Control the cursor with hand gestures or eye gaze, wake the voice assistant with “snowman” (or “snow”), and type on a semi-transparent on-screen keyboard with word predictions.

## What’s Inside
- Gesture mouse: hand tracking for move + left/right/double click, auto-light normalization, hardware-mouse priority pause.
- Gaze mouse: iris-based cursor with smoothing, deadzone, and blink-to-click.
- Voice assistant: wake word, speech commands (type, enter/delete, open/search, show/hide keyboard), pyttsx3 TTS, optional Ollama chat.
- On-screen keyboard: transparent, focus-safe overlay; larger keys for gesture tapping; centered trigram suggestions; only the full stop in punctuation row.

## Quick Start (Windows, Python 3.10)
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
py -3.10 main.py
```
Notes: MediaPipe wheels target Python 3.8–3.11; 3.10 is recommended. Ensure your webcam/mic are free and permitted.

## How to Use
- Gestures: index extended to move; index bent = left click; middle bent = right click; both bent = double click. Hardware mouse movement pauses gesture control for 1.5s.
- Gaze: look where you want the cursor; intentional blink to click; smoothing and deadzone reduce jitter.
- Voice: say “snowman” (or “snow”) to wake, then commands like “type hello world”, “press enter”, “open notepad”, “search accessibility tools”, “show keyboard”, or “hide keyboard”. Press `v` to toggle the assistant on/off.
- Keyboard: tap/click the translucent keys; predictions appear above the keys; focuses the last active app before sending keystrokes. Bottom row: Space, Back, Del, Enter. Punctuation row: only “.”

## Tips
- Lighting: face the light; avoid strong backlight for steadier hand/gaze tracking.
- Audio: reduce room noise; a closer mic helps the wake word; if still too strict, lower `MIN_ENERGY_THRESHOLD` in `main.py`.
- If Ollama chat is unused, ignore “Ollama is not running” messages.

## Common Issues
- “No matching distribution for mediapipe”: use Python 3.10/3.11, then `pip install -r requirements.txt`.
- Camera not showing: close other webcam apps or change `cv2.VideoCapture(0)` to `1` in `main.py`.
- Jittery gestures: improve lighting or tweak `MOVEMENT_MULTIPLIER` / `min_detection_confidence` in `main.py`.
- Gaze drift: increase `EYE_MIN_MOVE_PX` or lower `IRIS_SPEED_GAIN`.
- Voice misses wake word: ensure mic permissions; quiet the room; thresholds are in `MIN_ENERGY_THRESHOLD` and `ENERGY_MULTIPLIER`.

## Key Files
- `main.py`: gesture/gaze control loop, voice assistant, lighting adjustment, keyboard launcher.
- `keyboard.py`: transparent on-screen keyboard with trigram predictions and focus-safe typing.
- `requirements.txt`: dependencies (OpenCV, MediaPipe, pyautogui, SpeechRecognition, pyttsx3, etc.).
