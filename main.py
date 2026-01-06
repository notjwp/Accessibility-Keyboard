print('[DEBUG] main.py started')
import os
import warnings
# Suppress warnings from MediaPipe and TensorFlow (must be set before imports)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['GLOG_minloglevel'] = '2'  # Suppress MediaPipe C++ logging
warnings.filterwarnings('ignore', category=UserWarning)  # Suppress protobuf warnings
warnings.filterwarnings('ignore', message='.*SymbolDatabase.GetPrototype.*')  # Suppress protobuf deprecation

import cv2
import mediapipe as mp
import pyautogui
import util
from pynput.mouse import Button, Controller, Listener
import threading
import queue
import re
import time
import json
import webbrowser
import requests
from typing import Optional
import pyttsx3
import speech_recognition as sr
import numpy as np
import tkinter as tk
from keyboard import AccessibilityKeyboard

# Disable PyAutoGUI failsafe (since we're intentionally controlling mouse with gestures)
pyautogui.FAILSAFE = False

mouse = Controller()
LAST_HW_MOUSE_TS = 0.0
HW_PRIORITY_TIMEOUT = 1.5  # seconds to pause gesture control after hardware movement/click
hw_mouse_listener = None


def type_text_systemwide(text: str, interval: float = 0.04) -> bool:
    """Send keystrokes to whichever window currently has focus."""
    payload = (text or "").strip()
    if not payload:
        return False
    try:
        pyautogui.write(payload, interval=interval)
        return True
    except Exception as exc:
        print(f"[WARN] Failed to type text: {exc}")
        return False


def press_system_key(key: str, presses: int = 1) -> bool:
    """Press a single key via PyAutoGUI with basic error handling."""
    key_name = (key or "").strip().lower()
    if not key_name:
        return False
    count = max(1, int(presses or 1))
    try:
        for _ in range(count):
            pyautogui.press(key_name)
        return True
    except Exception as exc:
        print(f"[WARN] Failed to press key '{key_name}': {exc}")
        return False


def _hardware_mouse_event(*_args, **_kwargs):
    global LAST_HW_MOUSE_TS
    LAST_HW_MOUSE_TS = time.time()


def hardware_mouse_is_active() -> bool:
    return (time.time() - LAST_HW_MOUSE_TS) < HW_PRIORITY_TIMEOUT


def start_hw_mouse_listener():
    global hw_mouse_listener
    if hw_mouse_listener:
        return
    try:
        hw_mouse_listener = Listener(
            on_move=_hardware_mouse_event,
            on_click=lambda *a, **k: _hardware_mouse_event(),
            on_scroll=lambda *a, **k: _hardware_mouse_event(),
        )
        hw_mouse_listener.daemon = True
        hw_mouse_listener.start()
    except Exception as exc:
        print(f"[WARN] Could not start hardware mouse listener: {exc}")

# Voice Assistant Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")
LISTEN_LANG = "en-in"
TTS_RATE = 170

# Wake-word configuration (improved)
WAKE_WORDS = ("snowman", "snow man", "snow")
# regex covers common spacing/typo variants and accepts "snow" as a wake cue
WAKE_REGEX = re.compile(r"\b(sno+ ?man|snow ?man|snom?n|snowmin|snowmun|snow)\b", re.IGNORECASE)

# Fuzzy-match tolerance: allow up to 3 edits so short recognitions like "snow" still pass
WAKE_LEVENSHTEIN_THRESHOLD = 3

# Global flag to control voice assistant
voice_assistant_enabled = True  # Enabled by default
assistant_running = True
FOLLOWUP_WINDOW_SECONDS = 20
MIN_ENERGY_THRESHOLD = 1300
ENERGY_MULTIPLIER = 1.4

ENTER_COMMAND_PATTERN = re.compile(
    r"^(?:please |kindly |can you |could you |would you |will you )*"
    r"(?:press |hit |tap |push |trigger )?(?:the )?enter(?: key| button)?$"
)
DELETE_COMMAND_PATTERN = re.compile(
    r"^(?:please |kindly |can you |could you |would you |will you )*"
    r"(?:press |hit |tap |push |trigger )?(?:the )?(?:delete|backspace)(?: key| button)?$"
)

keyboard_root = None
keyboard_app = None
keyboard_thread = None
keyboard_running = False
keyboard_visible = False
keyboard_action_queue = queue.Queue()

screen_width, screen_height = pyautogui.size()

# Movement multiplier - cursor moves this many times the hand movement
MOVEMENT_MULTIPLIER = 4.5

# Track previous hand position for relative movement
prev_hand_x = None
prev_hand_y = None
cursor_x = screen_width   
cursor_y = screen_height 

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

mpFaceMesh = mp.solutions.face_mesh
face_mesh = mpFaceMesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

blink_cooldown_seconds = 1.0
last_blink_time = 0.0
IRIS_SPEED_GAIN = 12.0
EYE_SMOOTHING_ALPHA = 0.25  # fraction of new movement applied each frame
EYE_MIN_MOVE_PX = 5.0       # deadzone in pixels before issuing moveTo

smoothed_eye_x = None
smoothed_eye_y = None
last_eye_output_x = None
last_eye_output_y = None


def _reset_eye_tracking_state():
    global smoothed_eye_x, smoothed_eye_y, last_eye_output_x, last_eye_output_y
    smoothed_eye_x = None
    smoothed_eye_y = None
    last_eye_output_x = None
    last_eye_output_y = None

gesture_click_latch = {
    "left": False,
    "right": False,
    "double": False,
}


def configure_camera(cap: cv2.VideoCapture) -> None:
    """Attempt to enable auto exposure and auto white balance on the camera.
    Note: Support depends on camera driver/backend; failures are ignored gracefully.
    """
    try:
        # Many Windows drivers (DirectShow) use 0.25 for auto exposure, 0.75 for manual
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    except Exception:
        pass
    # Optional: set a reasonable white-balance temperature if supported
    try:
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)
    except Exception:
        pass


def _apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        return img
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)


def adjust_frame_lighting(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalize lighting per-frame using adaptive gamma and CLAHE.
    Returns (adjusted_frame, gamma_used).
    """
    # Compute mean brightness
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    mean_norm = max(1e-3, min(0.999, mean / 255.0))
    target = 0.5  # target mid-gray
    # Derive gamma to move mean towards target
    try:
        gamma = np.log(target) / np.log(mean_norm)
    except FloatingPointError:
        gamma = 1.0
    gamma = float(np.clip(gamma, 0.5, 2.0))

    # Apply gamma correction
    corrected = _apply_gamma(frame, gamma) if abs(gamma - 1.0) > 0.05 else frame

    # Apply CLAHE on L channel for local contrast
    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    result = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    return result, gamma


def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
        index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        return index_finger_tip
    return None


def move_mouse(index_finger_tip):
    global prev_hand_x, prev_hand_y, cursor_x, cursor_y
    if hardware_mouse_is_active():
        prev_hand_x = None
        prev_hand_y = None
        return
    if index_finger_tip is not None:
        # Get normalized hand position (0-1)
        hand_x = index_finger_tip.x
        hand_y = index_finger_tip.y   # Adjust for y-axis as in original code
        
        # Convert to pixel coordinates for hand position
        hand_x_pixels = hand_x * screen_width
        hand_y_pixels = hand_y * screen_height
        
        if prev_hand_x is not None and prev_hand_y is not None:
            # Calculate hand movement delta
            delta_x = hand_x_pixels - prev_hand_x
            delta_y = hand_y_pixels - prev_hand_y
            
            # Apply 2x multiplier to movement
            cursor_delta_x = delta_x * MOVEMENT_MULTIPLIER
            cursor_delta_y = delta_y * MOVEMENT_MULTIPLIER
            
            # Update cursor position
            cursor_x += cursor_delta_x
            cursor_y += cursor_delta_y
            
            # Clamp cursor to screen boundaries
            cursor_x = max(0, min(screen_width - 1, cursor_x))
            cursor_y = max(0, min(screen_height - 1, cursor_y))
            
            # Move cursor to new position
            pyautogui.moveTo(int(cursor_x), int(cursor_y))
        else:
            # First frame - initialize cursor position based on hand position
            cursor_x = hand_x_pixels
            cursor_y = hand_y_pixels
            pyautogui.moveTo(int(cursor_x), int(cursor_y))
        
        # Update previous hand position
        prev_hand_x = hand_x_pixels
        prev_hand_y = hand_y_pixels
    else:
        # No hand detected - reset tracking
        prev_hand_x = None
        prev_hand_y = None


def is_left_click(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and
            thumb_index_dist > 50
    )


def is_right_click(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90  and
            thumb_index_dist > 50
    )


def is_double_click(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            thumb_index_dist > 50
    )


# ==================== Voice Assistant Functions ====================
engine = None
recognizer = None
TTS_LOCK = threading.Lock()

def init_voice_assistant():
    global engine, recognizer
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', TTS_RATE)
        recognizer = sr.Recognizer()
        # Start with a modest threshold and allow automatic adjustment so mild noise doesn't block speech
        recognizer.energy_threshold = MIN_ENERGY_THRESHOLD
        recognizer.dynamic_energy_threshold = True
        recognizer.dynamic_energy_adjustment_damping = 0.12
        recognizer.dynamic_energy_ratio = 1.6
        # Keep listening through natural short pauses so full sentences are captured
        recognizer.pause_threshold = 1.0  # seconds of silence before we stop listening
        recognizer.phrase_threshold = 0.5  # minimum speech length to start a phrase
        recognizer.non_speaking_duration = 0.5  # trailing silence considered part of the phrase
        return True
    except Exception as e:
        print(f"Voice assistant initialization failed: {e}")
        return False
def _boost_energy_threshold(current: float) -> int:
    current = current or 0
    boosted = max(MIN_ENERGY_THRESHOLD, int(current * ENERGY_MULTIPLIER))
    return boosted

def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance between strings a and b (iterative DP)."""
    a = a or ""
    b = b or ""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la

    # use only two rows to save memory
    prev_row = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        cur_row = [i] + [0] * lb
        for j, cb in enumerate(b, start=1):
            insert_cost = prev_row[j] + 1
            delete_cost = cur_row[j - 1] + 1
            replace_cost = prev_row[j - 1] + (0 if ca == cb else 1)
            cur_row[j] = min(insert_cost, delete_cost, replace_cost)
        prev_row = cur_row
    return prev_row[lb]


def fuzzy_match_wake_word(text: str) -> bool:
    """
    Return True if any token or the whole normalized text
    is within WAKE_LEVENSHTEIN_THRESHOLD edits from 'snowman'.
    """
    if not text:
        return False
    text = text.lower().strip()
    # normalize: keep alphanum and spaces
    normalized = re.sub(r"[^a-z0-9\s]", "", text)
    # check whole phrase first
    if levenshtein(normalized.replace(" ", ""), "snowman") <= WAKE_LEVENSHTEIN_THRESHOLD:
        return True
    # check word-by-word for short utterances
    tokens = [t for t in normalized.split() if t]
    for tk in tokens:
        if levenshtein(tk, "snowman") <= WAKE_LEVENSHTEIN_THRESHOLD:
            return True
    return False

def _speak_blocking(text: str):
    """Serialize TTS to avoid pyttsx3 'run loop already started' errors."""
    if not engine or not text:
        return
    with TTS_LOCK:
        engine.say(text)
        engine.runAndWait()


def speak(text: str, block: bool = False):
    if not engine or not text:
        return
    print(f"Assistant: {text}")
    if block:
        _speak_blocking(text)
    else:
        threading.Thread(target=_speak_blocking, args=(text,), daemon=True).start()


def speak_long(text: str):
    """Speak long responses sentence-by-sentence, serialized to avoid run loop conflicts."""
    if not engine or not text:
        return
    import re as _re
    sentences = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    consolidated = " ".join(sentences)
    if consolidated:
        print(f"Assistant: {consolidated}")

    def _run():
        for sent in sentences:
            _speak_blocking(sent)

    threading.Thread(target=_run, daemon=True).start()

def recognize_once(timeout: Optional[int] = None, phrase_time_limit: Optional[int] = None) -> str:
    if not recognizer:
        return ""
    try:
        with sr.Microphone() as source:
            print("[Listening for command...]")
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        print("[Processing speech...]")
        text = recognizer.recognize_google(audio, language=LISTEN_LANG)
        cleaned = text.strip()
        print(f"You said: {cleaned}")
        return cleaned
    except sr.WaitTimeoutError:
        print("[No speech detected - timeout]")
        return ""
    except sr.UnknownValueError:
        print("[Could not understand audio]")
        return ""
    except sr.RequestError as e:
        print(f"[Speech recognition service error: {e}]")
        return ""
    except OSError as e:
        print(f"[Microphone error: {e}]")
        return ""

def shutdown_keyboard_window():
    """Safely destroy the accessibility keyboard window if it exists."""
    global keyboard_root, keyboard_app, keyboard_running, keyboard_visible, keyboard_action_queue
    keyboard_running = False
    keyboard_visible = False
    if keyboard_root:
        try:
            keyboard_root.after(0, keyboard_root.destroy)
        except Exception:
            pass
        keyboard_root = None
        keyboard_app = None
    keyboard_action_queue = queue.Queue()


def show_keyboard() -> bool:
    """Request the keyboard window to become visible; handled inside the Tk thread."""
    if not keyboard_root:
        return False
    try:
        keyboard_action_queue.put_nowait("show")
        return True
    except queue.Full:
        return False


def hide_keyboard() -> bool:
    """Request hiding the keyboard window without stopping the Tk loop."""
    if not keyboard_root:
        return False
    try:
        keyboard_action_queue.put_nowait("hide")
        return True
    except queue.Full:
        return False


def start_keyboard_background():
    """Launch the AccessibilityKeyboard in a background thread with manual update calls."""
    global keyboard_thread, keyboard_running, keyboard_root, keyboard_app, keyboard_action_queue

    if keyboard_thread and keyboard_thread.is_alive():
        return

    keyboard_running = True
    keyboard_action_queue = queue.Queue()

    def _tk_loop():
        global keyboard_root, keyboard_app, keyboard_running, keyboard_visible
        try:
            keyboard_root = tk.Tk()
            keyboard_root.withdraw()
            keyboard_app = AccessibilityKeyboard(keyboard_root)
            keyboard_root.update_idletasks()
            keyboard_root.deiconify()
            keyboard_root.lift()
            keyboard_root.attributes("-topmost", True)
            keyboard_visible = True
            print("[DEBUG] Keyboard background loop started.")
            while keyboard_running:
                keyboard_root.update()
                # Handle queued show/hide requests from other threads
                while True:
                    try:
                        action = keyboard_action_queue.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        if action == "show":
                            keyboard_root.deiconify()
                            keyboard_root.lift()
                            keyboard_root.attributes("-topmost", True)
                            keyboard_visible = True
                        elif action == "hide":
                            keyboard_root.withdraw()
                            keyboard_visible = False
                    except tk.TclError:
                        pass
                time.sleep(0.01)
        except tk.TclError as exc:
            print(f"[WARN] Keyboard loop stopped: {exc}")
        finally:
            keyboard_running = False
            keyboard_visible = False
            keyboard_root = None
            keyboard_app = None

    keyboard_thread = threading.Thread(target=_tk_loop, name="KeyboardLoop", daemon=True)
    keyboard_thread.start()

def wait_for_wake() -> bool:
    """Listen briefly and return True if wake word is detected (regex or fuzzy match)."""
    if not recognizer:
        return False
    try:
        with sr.Microphone() as source:
            # short listen so this loop isn't too heavy; phrase_time_limit tuned to short wake-word
            audio = recognizer.listen(source, timeout=None, phrase_time_limit=3)
        try:
            raw_text = recognizer.recognize_google(audio, language=LISTEN_LANG)
            text = raw_text.lower().strip()
            if not text:
                return False

            # normalized form for simple substring checks
            normalized = re.sub(r"[^a-z0-9\s]", "", text)

            # direct word list / substring match (fast)
            has_wake_word = any(hw in normalized for hw in WAKE_WORDS)

            # regex match (flexible)
            regex_match = bool(WAKE_REGEX.search(text))

            # fuzzy (levenshtein) match
            fuzzy_match = fuzzy_match_wake_word(normalized)

            if has_wake_word or regex_match or fuzzy_match:
                print(f"✓ WAKE WORD DETECTED: {raw_text}")
                return True
            else:
                print(f"(ignored): {raw_text}")
                return False

        except sr.UnknownValueError:
            return False
        except sr.RequestError as e:
            print(f"[Wake detection error: {e}]")
            return False
    except Exception as e:
        print(f"[Wake detection exception: {e}]")
        return False

def ask_ollama(prompt: str) -> str:
    """Call Ollama with a short timeout; return a clear fallback if unreachable."""
    try:
        with requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
            timeout=10,  
            stream=True,
        ) as resp:
            resp.raise_for_status()
            out = []
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        out.append(data["response"])
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
            return "".join(out).strip() or "No response."
    except requests.exceptions.ConnectionError:
        return "Ollama is not running or unreachable. Start Ollama and retry."
    except Exception as e:
        return f"LLM error: {e}"

def execute_voice_command(command: str):
    global voice_assistant_enabled, assistant_running

    raw_command = (command or "").strip()
    c = raw_command.lower()
    if not c:
        speak("I didn't catch that.")
        return

    # Gesture control toggle
    if "disable gesture" in c or "turn off gesture" in c:
        speak("Gesture control disabled.")
        return
    
    if "enable gesture" in c or "turn on gesture" in c:
        speak("Gesture control enabled.")
        return

    # Open apps/websites
    if c.startswith("open "):
        target = c.replace("open ", "", 1).strip()
        if "youtube" in target:
            speak("Opening YouTube")
            webbrowser.open("https://www.youtube.com")
        elif "google" in target and "." not in target:
            speak("Opening Google")
            webbrowser.open("https://www.google.com")
        elif "." in target or target.startswith("http"):
            speak(f"Opening {target}")
            webbrowser.open(target if target.startswith("http") else "https://" + target)
        else:
            try:
                os.startfile(target)
                speak(f"Opening {target}")
            except Exception:
                speak(f"Couldn't open {target}.")
        return

    # Search
    if c.startswith("search ") or c.startswith("google "):
        q = c.replace("search", "", 1).replace("google", "", 1).strip()
        if q:
            speak(f"Searching for {q}")
            webbrowser.open("https://www.google.com/search?q=" + q.replace(" ", "+"))
        return

    if ENTER_COMMAND_PATTERN.match(c):
        speak("Pressing enter")
        if not press_system_key("enter"):
            speak("I couldn't press enter.")
        return

    if DELETE_COMMAND_PATTERN.match(c):
        speak("Deleting")
        if not press_system_key("backspace"):
            speak("I couldn't delete that.")
        return

    # Accessibility keyboard visibility
    show_keywords = (
        "show keyboard",
        "open keyboard",
        "bring up keyboard",
        "display keyboard",
        "keyboard on",
    )
    hide_keywords = (
        "hide keyboard",
        "close keyboard",
        "dismiss keyboard",
        "keyboard off",
    )

    if any(kw in c for kw in show_keywords):
        if show_keyboard():
            speak("Keyboard visible.")
        else:
            speak("Keyboard is still starting up.")
        return

    if any(kw in c for kw in hide_keywords):
        if hide_keyboard():
            speak("Keyboard hidden.")
        else:
            speak("Keyboard is not running yet.")
        return

    # Type text (supports phrases like "please type" or "can you type")
    type_prefix = re.match(r"(?:please |kindly |can you |could you |will you |would you )*type\b", c)
    if type_prefix:
        text_to_type = raw_command[type_prefix.end():].strip()
        if not text_to_type:
            speak("What should I type?")
            follow_up = recognize_once(timeout=8, phrase_time_limit=20)
            text_to_type = (follow_up or "").strip()
        if text_to_type:
            speak("Typing")
            time.sleep(1)  # Give time to focus on target window
            if not type_text_systemwide(text_to_type):
                speak("I couldn't send the text. Please try again.")
        else:
            speak("I didn't hear any text to type.")
        return

    # Exit
    if any(k in c for k in ["exit", "quit", "stop", "shutdown"]):
        speak("Shutting down. Goodbye!")
        assistant_running = False
        return

    # Ollama chat
    if c.startswith(("ask ", "chat ", "explain ", "answer ")):
        prompt = re.sub(r"^(ask|chat|explain|answer)\s+", "", c, count=1).strip()
        if prompt:
            speak("Thinking...")
            resp = ask_ollama(prompt)
            # Use long speech for explanation-style responses
            speak_long(resp)
        return

    # Default: send to Ollama
    speak("Let me think about that.")
    resp = ask_ollama(f"You are a concise helpful assistant. User said: {c}")
    speak_long(resp)

def voice_assistant_thread():
    global assistant_running, voice_assistant_enabled
    
    if not init_voice_assistant():
        print("Voice assistant could not be initialized")
        return
    
    speak("Voice assistant ready. Say 'snowman' to activate. Chat requires Ollama running.", block=True)
    time.sleep(1)
    
    # Calibrate microphone for ambient noise
    try:
        with sr.Microphone() as source:
            print("[Calibrating microphone for ambient noise... Please stay quiet.]")
            recognizer.adjust_for_ambient_noise(source, duration=2.5)
            # Make it slightly more sensitive than the auto-calibrated value
            tuned = max(MIN_ENERGY_THRESHOLD, int(recognizer.energy_threshold * 0.7))
            recognizer.energy_threshold = tuned
            print(f"[Calibration complete. Energy threshold set to: {recognizer.energy_threshold}]")
    except Exception as e:
        print(f"[Calibration warning: {e}]")
        recognizer.energy_threshold = max(MIN_ENERGY_THRESHOLD, 800)
    
    last_wake_time = 0.0

    while assistant_running:
        try:
            if not voice_assistant_enabled:
                time.sleep(0.5)
                continue
            now = time.time()

            if last_wake_time and (now - last_wake_time) <= FOLLOWUP_WINDOW_SECONDS:
                cmd = recognize_once(timeout=8, phrase_time_limit=20)
                if cmd:
                    execute_voice_command(cmd)
                    last_wake_time = time.time()
                else:
                    time.sleep(0.2)
                continue

            if wait_for_wake():
                last_wake_time = time.time()
                speak("Yes?")
                cmd = recognize_once(timeout=10, phrase_time_limit=25)
                if cmd:
                    execute_voice_command(cmd)
                    last_wake_time = time.time()
                else:
                    speak("I didn't catch that.")
        except Exception as e:
            print(f"Voice assistant error: {e}")
            time.sleep(0.5)

def detect_gesture(frame, landmark_list, processed):
    global gesture_click_latch
    if len(landmark_list) >= 21:
        if hardware_mouse_is_active():
            gesture_click_latch["left"] = False
            gesture_click_latch["right"] = False
            gesture_click_latch["double"] = False
            return

        index_finger_tip = find_finger_tip(processed)
        thumb_index_dist = util.get_distance([landmark_list[4], landmark_list[5]])

        move_condition = util.get_distance([landmark_list[4], landmark_list[5]]) < 50 \
            and util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90

        left_condition = is_left_click(landmark_list, thumb_index_dist)
        right_condition = is_right_click(landmark_list, thumb_index_dist)
        double_condition = is_double_click(landmark_list, thumb_index_dist)

        if move_condition:
            move_mouse(index_finger_tip)
            gesture_click_latch["left"] = False
            gesture_click_latch["right"] = False
            gesture_click_latch["double"] = False
            return

        if left_condition and not gesture_click_latch["left"]:
            mouse.press(Button.left)
            mouse.release(Button.left)
            gesture_click_latch["left"] = True
            gesture_click_latch["right"] = False
            gesture_click_latch["double"] = False
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif not left_condition:
            gesture_click_latch["left"] = False

        if right_condition and not gesture_click_latch["right"]:
            mouse.press(Button.right)
            mouse.release(Button.right)
            gesture_click_latch["right"] = True
            gesture_click_latch["left"] = False
            gesture_click_latch["double"] = False
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif not right_condition:
            gesture_click_latch["right"] = False

        if double_condition and not gesture_click_latch["double"]:
            pyautogui.doubleClick()
            gesture_click_latch["double"] = True
            gesture_click_latch["left"] = False
            gesture_click_latch["right"] = False
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        elif not double_condition:
            gesture_click_latch["double"] = False


def process_eye_mouse(frame: np.ndarray, frame_rgb: np.ndarray):
    """Use MediaPipe Face Mesh to move mouse with eye gaze and blink to click."""
    global last_blink_time, smoothed_eye_x, smoothed_eye_y, last_eye_output_x, last_eye_output_y

    if not face_mesh:
        return

    if hardware_mouse_is_active():
        _reset_eye_tracking_state()
        return

    results = face_mesh.process(frame_rgb)
    if not results.multi_face_landmarks:
        _reset_eye_tracking_state()
        return

    landmarks = results.multi_face_landmarks[0].landmark
    frame_h, frame_w, _ = frame.shape

    # Use iris landmarks 474-477 for pointer tracking
    for idx, landmark in enumerate(landmarks[474:478]):
        x = int(landmark.x * frame_w)
        y = int(landmark.y * frame_h)
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        if idx == 1:
            cx = screen_width * 0.5
            cy = screen_height * 0.5
            screen_x = cx + (landmark.x - 0.5) * screen_width * IRIS_SPEED_GAIN
            screen_y = cy + (landmark.y - 0.5) * screen_height * IRIS_SPEED_GAIN
            screen_x = max(0, min(screen_width - 1, screen_x))
            screen_y = max(0, min(screen_height - 1, screen_y))

            if smoothed_eye_x is None or smoothed_eye_y is None:
                smoothed_eye_x = screen_x
                smoothed_eye_y = screen_y
            else:
                smoothed_eye_x += EYE_SMOOTHING_ALPHA * (screen_x - smoothed_eye_x)
                smoothed_eye_y += EYE_SMOOTHING_ALPHA * (screen_y - smoothed_eye_y)

            target_x = smoothed_eye_x
            target_y = smoothed_eye_y
            if (
                last_eye_output_x is None
                or last_eye_output_y is None
                or abs(target_x - last_eye_output_x) >= EYE_MIN_MOVE_PX
                or abs(target_y - last_eye_output_y) >= EYE_MIN_MOVE_PX
            ):
                pyautogui.moveTo(int(target_x), int(target_y), duration=0)
                last_eye_output_x = target_x
                last_eye_output_y = target_y

    left_eye = [landmarks[145], landmarks[159]]
    for landmark in left_eye:
        x = int(landmark.x * frame_w)
        y = int(landmark.y * frame_h)
        cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

    blink_gap = left_eye[0].y - left_eye[1].y
    now = time.time()
    if blink_gap < 0.004 and (now - last_blink_time) >= blink_cooldown_seconds:
        pyautogui.click()
        last_blink_time = now
        cv2.putText(frame, "Blink Click", (frame_w - 180, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def main():
    global assistant_running, voice_assistant_enabled
    
    # Suppress additional warnings during runtime
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    
    print("=" * 60)
    print("  Hand Gesture Mouse Controller with Voice Assistant")
    print("=" * 60)
    print("\nControls:")
    print("  - Say 'snowman' to wake the assistant")
    print("  - Give additional commands right away; no need to repeat the wake word")
    print("  - Press 'v' to toggle voice assistant ON/OFF")
    print("  - Press 'q' to quit")
    print("\nVoice assistant: ENABLED by default")
    print("Accessibility keyboard: ENABLED by default")
    print("Starting gesture recognition...")
    print("=" * 60)
    start_hw_mouse_listener()
    
    def gesture_loop():
        global voice_assistant_enabled, assistant_running
        print("[DEBUG] Starting gesture recognition thread...")
        draw = mp.solutions.drawing_utils
        cap = cv2.VideoCapture(0)
        configure_camera(cap)
        try:
            while cap.isOpened() and assistant_running:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                frame, gamma = adjust_frame_lighting(frame)
                frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed = hands.process(frameRGB)

                landmark_list = []
                hand_present = False
                if processed.multi_hand_landmarks:
                    hand_present = True
                    hand_landmarks = processed.multi_hand_landmarks[0]
                    draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                    for lm in hand_landmarks.landmark:
                        landmark_list.append((lm.x, lm.y))

                if hand_present:
                    detect_gesture(frame, landmark_list, processed)
                else:
                    process_eye_mouse(frame, frameRGB)

                status_text = "Voice: ON" if voice_assistant_enabled else "Voice: OFF (Press 'v')"
                status_color = (0, 255, 0) if voice_assistant_enabled else (0, 0, 255)
                cv2.putText(frame, status_text, (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                if abs(gamma - 1.0) > 0.15:
                    cv2.putText(frame, f"AutoLight gamma={gamma:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                mode_text = "Input: Hand" if hand_present else "Input: Eye"
                mode_color = (0, 200, 255) if hand_present else (255, 200, 0)
                cv2.putText(frame, mode_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

                cv2.imshow('Hand Gesture Controller', frame)
                try:
                    if cv2.getWindowProperty('Hand Gesture Controller', cv2.WND_PROP_VISIBLE) < 1:
                        assistant_running = False
                        shutdown_keyboard_window()
                        break
                except cv2.error:
                    break
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    assistant_running = False
                    shutdown_keyboard_window()
                    break
                elif key == ord('v'):
                    voice_assistant_enabled = not voice_assistant_enabled
                    status = "enabled" if voice_assistant_enabled else "disabled"
                    print(f"\nVoice assistant {status}")
        except Exception as e:
            print(f"[Gesture recognition error: {e}]")
        finally:
            assistant_running = False
            cap.release()
            cv2.destroyAllWindows()
            try:
                face_mesh.close()
            except Exception:
                pass
            shutdown_keyboard_window()
            print("\nShutting down gesture recognition...")
            time.sleep(0.5)

    print("[DEBUG] Starting gesture and voice threads...")
    gesture_thread = threading.Thread(target=gesture_loop, daemon=True)
    gesture_thread.start()
    voice_thread = threading.Thread(target=voice_assistant_thread, daemon=True)
    voice_thread.start()
    print("[DEBUG] Threads started. Launching accessibility keyboard window...")
    start_keyboard_background()

    try:
        while assistant_running and (gesture_thread.is_alive() or voice_thread.is_alive()):
            time.sleep(0.5)
    except KeyboardInterrupt:
        assistant_running = False
    finally:
        gesture_thread.join(timeout=1.0)
        voice_thread.join(timeout=1.0)
        shutdown_keyboard_window()
        print("[DEBUG] Main loop exited. Program will now exit.")

# Entry point
if __name__ == '__main__':
    print('[DEBUG] __name__ == __main__, calling main()')
    main()
