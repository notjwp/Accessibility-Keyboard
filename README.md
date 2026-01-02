# Accessibility Keyboard

A lightweight multimodal console that keeps Windows usable with just a webcam and microphone. MediaPipe hand and eye tracking handle the mouse, a wake-word voice assistant executes commands like "snowman, type hello world", and a translucent Tkinter keyboard with punctuation/numpad delivers focus-safe typing.

## Highlights
- Gesture mouse with left/right/double clicks, gamma-adjusted video, and hardware-mouse priority pause
- Eye-gaze cursor with smoothing, blink-to-click, and deadzone filtering
- Voice assistant backed by SpeechRecognition + queued pyttsx3 TTS, plus optional Ollama chat
- Full-width keyboard overlay featuring trigram predictions, punctuation row, and Win32 focus handoff

## Quick Start
```powershell
git clone https://github.com/<you>/accessibility-keyboard.git
cd accessibility-keyboard
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
py -3.10 main.py
```
> MediaPipe ships wheels for Python 3.8–3.11 only. Use Python 3.10 for best compatibility.

## Controls Snapshot
| Mode | How to use |
|------|------------|
| Gestures | Index extended to move, index-bent for left click, middle-bent for right click, both bent for double click |
| Eye gaze | Look where you want the cursor; blink to click; physical mouse motion pauses gaze for 1.5 s |
| Voice | Press `v`, say "snowman" to wake, then commands like `type hello world`, `press enter`, `open notepad`, `brightness up`, `ask explain PID control` |
| Keyboard | Click or gesture-tap the translucent keys; predictions appear after two words; refocuses the last window before typing |

## Troubleshooting Cheatsheet
| Issue | Fix |
|-------|-----|
| `ERROR: No matching distribution found for mediapipe` | Install Python 3.10/3.11 and rerun `py -3.10 -m pip install -r requirements.txt` |
| Blank camera preview | Close other webcam apps or change `cv2.VideoCapture(0)` to `1`/`2` in `main.py` |
| Gestures jittery or missed | Improve lighting or adjust `min_detection_confidence` / `MOVEMENT_MULTIPLIER` |
| Gaze cursor noisy | Lower `EYE_SMOOTHING_ALPHA` or raise `EYE_MIN_MOVE_PX` |
| Voice not responding | Check mic permissions, rerun ambient calibration, or increase `MIN_ENERGY_THRESHOLD` |
| "Ollama is not running" | Start `ollama serve` or skip chat commands |

