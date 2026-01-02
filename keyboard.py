import os
import tkinter as tk
from tkinter import font
from collections import defaultdict
import re
import ctypes

try:
    import pyautogui
except ImportError:  # Gracefully degrade if dependency is missing
    pyautogui = None

# ----------------------------
# CONFIG
# ----------------------------
BASE_BG = "#99df01"
PANEL_BG = "#D50404"
KEY_BG = "#2b522d"
KEY_ACTIVE = "#6b85c0"
TEXT_COLOR = "#f8fafc"
SUBTEXT_COLOR = "#94a3b8"
ACCENT_COLOR = "#38bdf8"
PRED_BG = "#253153"
TRANSPARENT_COLOR = "#010101"
WINDOW_ALPHA = 0.45  # More transparent overlay for clearer view of underlying apps

# ----------------------------
# Simple Trigram Language Model
# ----------------------------
class TrigramModel:
    def __init__(self):
        self.model = defaultdict(lambda: defaultdict(int))
        self.unigram = defaultdict(int)

    def train(self, text):
        if not text:
            return
        try:
            words = re.findall(r"\w+", text.lower())
        except Exception:
            return
        if not words:
            return
        # Track unigram frequency so we can fall back when no bigram is found
        for w in words:
            self.unigram[w] += 1
        for i in range(len(words) - 2):
            w1, w2, w3 = words[i], words[i + 1], words[i + 2]
            self.model[(w1, w2)][w3] += 1

    def predict(self, w1=None, w2=None, top_n=3):
        """Predict next words using a bigram context when available.
        Falls back to most common unigrams if the bigram is unseen."""
        ctx = None
        # Defensive defaults
        try:
            top_n = max(1, int(top_n or 3))
        except Exception:
            top_n = 3
        if w1 is not None and w2 is not None:
            ctx = self.model.get((w1.lower(), w2.lower()), {})
        elif w2 is not None:
            # If only one word is available, try a loose match on the second token
            ctx = {}
            for (a, b), nexts in self.model.items():
                if b == w2.lower():
                    for nxt, cnt in nexts.items():
                        ctx[nxt] = ctx.get(nxt, 0) + cnt

        # Primary: bigram-based suggestions
        if ctx:
            sorted_words = sorted(ctx.items(), key=lambda x: x[1], reverse=True)
            return [w for w, _ in sorted_words[:top_n]]

        # Fallback: most frequent unigrams
        if self.unigram:
            return [w for w, _ in sorted(self.unigram.items(), key=lambda x: x[1], reverse=True)[:top_n]]

        return ["---"] * top_n

# ----------------------------
# Accessibility Keyboard Class
# ----------------------------
class AccessibilityKeyboard:
    def __init__(self, root):
        self.root = root

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_h = 640
        win_w = screen_w
        pos_x = 0
        pos_y = max(0, screen_h - win_h - 20)
        self.root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        self.root.configure(bg=TRANSPARENT_COLOR) 
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(True)
        # Make the window background transparent instead of semi-opaque
        if os.name == "nt":
            try: 
                self.root.attributes("-transparentcolor", TRANSPARENT_COLOR)
            except tk.TclError:
                pass
            self.root.attributes("-alpha", WINDOW_ALPHA)
        else:
            self.root.attributes("-alpha", WINDOW_ALPHA)
        self.root.after(10, self._apply_no_activate_style)
        self.keyboard_hwnd = None
        self.last_target_hwnd = None

        self.text_buffer = tk.StringVar()
        self.text_buffer.set("")
        self.key_font = font.Font(size=14, weight="bold", family="Segoe UI")
        self.small_font = font.Font(size=12, family="Segoe UI")

        # Trigram language model
        self.language_model = TrigramModel()
        try:
            self.language_model.train("""
                hello how are you this is an assistive accessibility keyboard
                machine learning based system predicts next word using trigram model
                welcome to the project this helps disabled users to type faster using gaze or gesture
            """)
        except Exception:
            pass

        # Outer frame with subtle border
        self.outer_frame = tk.Frame(
            self.root,
            bg=TRANSPARENT_COLOR,
            highlightthickness=0
        )
        self.outer_frame.pack(padx=0, pady=(6, 6), fill="both", expand=True)

        self.build_ui()
        self.root.update_idletasks()
        if os.name == "nt":
            try:
                self.keyboard_hwnd = self.root.winfo_id()
            except Exception:
                self.keyboard_hwnd = None
            self.root.after(200, self._track_foreground_window)

    # -----------------------
    # Build whole UI
    # -----------------------
    def build_ui(self):
        header = tk.Frame(self.outer_frame, bg=TRANSPARENT_COLOR)
        header.configure(height=14)
        header.pack(fill="x", padx=20, pady=(5, 0))
        header.pack_propagate(False)

        # Allow dragging the frameless window
        header.bind("<ButtonPress-1>", self._start_drag)
        header.bind("<B1-Motion>", self._on_drag)

        # Prediction row (centered, minimal vertical push)
        self.create_prediction_buttons(parent=self.outer_frame)
        self.update_predictions()

        # Main keyboard area
        main_frame = tk.Frame(self.outer_frame, bg=TRANSPARENT_COLOR)
        main_frame.pack(pady=(6, 8), padx=10, fill="both", expand=True)

        kb_card = tk.Frame(main_frame, bg=TRANSPARENT_COLOR, bd=0, relief="flat")
        kb_card.pack(fill="both", expand=True, padx=24)

        self.create_qwerty_keyboard(kb_card)

    # -----------------------
    # Text Display
    # -----------------------
    # -----------------------
    # QWERTY Keyboard
    # -----------------------
    def create_qwerty_keyboard(self, parent):
        # Row 0: number row
        row0 = list("1234567890")
        # Traditional QWERTY rows
        row1 = list("QWERTYUIOP")
        row2 = list("ASDFGHJKL")
        row3 = list("ZXCVBNM.")

        rows = [row0, row1, row2, row3]

        for r_idx, row in enumerate(rows):
            frame = tk.Frame(parent, bg=TRANSPARENT_COLOR)
            frame.pack(pady=3)
            for key in row:
                self.create_key_button(frame, key)

        # Bottom row: SPACE, BACK, ENTER
        bottom_frame = tk.Frame(parent, bg=TRANSPARENT_COLOR)
        bottom_frame.pack(pady=6)

        self.create_key_button(bottom_frame, "SPACE", wide=True)
        self.create_key_button(bottom_frame, "BACK")
        self.create_key_button(bottom_frame, "DEL")
        self.create_key_button(bottom_frame, "ENTER")

    # -----------------------
    # Generic Key Button
    # -----------------------
    def create_key_button(self, parent, key, wide=False, small=False):
        if wide:
            width = 16
        elif small:
            width = 5
        else:
            width = 7

        btn_font_size = 16 if not small else 13

        btn = tk.Button(
            parent,
            text=key,
            width=width,
            height=2,
            pady=2,
            font=self.key_font if not small else self.small_font,
            fg=TEXT_COLOR,
            bg=KEY_BG,
            activebackground=KEY_ACTIVE,
            activeforeground=TEXT_COLOR,
            relief="flat",
            bd=0,
            cursor="hand2",
            takefocus=0,
            command=lambda k=key: self.select_key(k)
        )
        btn.pack(side=tk.LEFT, padx=4, pady=4)

    # -----------------------
    # Predictions
    # -----------------------
    def create_prediction_buttons(self, parent=None):
        parent = parent or self.outer_frame
        self.pred_frame = tk.Frame(parent, bg=TRANSPARENT_COLOR)
        self.pred_frame.pack(pady=(6, 10), anchor="center")

        inner = tk.Frame(self.pred_frame, bg=TRANSPARENT_COLOR)
        inner.pack(anchor="center")

        self.pred_btns = []
        for i in range(3):
            btn = tk.Button(
                inner,
                text="---",
                width=14,
                height=1,
                font=self.small_font,
                fg=TEXT_COLOR,
                bg=PRED_BG,
                activebackground=KEY_ACTIVE,
                activeforeground=TEXT_COLOR,
                bd=0,
                relief="flat",
                padx=10,
                pady=6,
                takefocus=0,
                command=lambda i=i: self.trigger_prediction(i)
            )
            btn.pack(side=tk.LEFT, padx=5)
            self.pred_btns.append(btn)

    # -----------------------
    # Core Logic
    # -----------------------
    def select_key(self, key):
        text = self.text_buffer.get()

        # Track which key event should be mirrored at the OS level
        outbound_key = None

        if key == "SPACE":
            text += " "
            outbound_key = "space"
        elif key in ("BACK", "DEL"):
            text = text[:-1]
            outbound_key = "backspace"
        elif key == "ENTER":
            text += "\n"
            outbound_key = "enter"
        else:
            text += key
            outbound_key = key

        self.text_buffer.set(text)
        # Continuously learn from the user's current buffer so predictions stay relevant
        if key in ("SPACE", "ENTER") or key.isalnum():
            try:
                self.language_model.train(text)
            except Exception:
                pass
        if outbound_key:
            self._emit_keypress(outbound_key)
        self.update_predictions()

    # -----------------------
    # Window Drag Support
    # -----------------------
    def _start_drag(self, event):
        self._drag_start = (event.x_root, event.y_root, self.root.winfo_x(), self.root.winfo_y())

    def _on_drag(self, event):
        if not hasattr(self, "_drag_start"):
            return
        x0, y0, win_x, win_y = self._drag_start
        dx = event.x_root - x0
        dy = event.y_root - y0
        self.root.geometry(f"+{win_x + dx}+{win_y + dy}")

    def update_predictions(self):
        try:
            text = self.text_buffer.get().strip().lower().split()
        except Exception:
            text = []
        preds = ["---", "---", "---"]
        if len(text) >= 2:
            w1, w2 = text[-2], text[-1]
            try:
                preds = self.language_model.predict(w1, w2)
            except Exception:
                preds = ["---", "---", "---"]
        elif len(text) == 1:
            try:
                preds = self.language_model.predict(None, text[-1])
            except Exception:
                preds = ["---", "---", "---"]
        else:
            # No context yet; fallback to most common unigrams
            try:
                preds = self.language_model.predict()
            except Exception:
                preds = ["---", "---", "---"]

        for i in range(3):
            self.pred_btns[i].config(text=preds[i] if i < len(preds) else "---")

    def trigger_prediction(self, index):
        pred = self.pred_btns[index].cget("text")
        if pred != "---":
            text = self.text_buffer.get()
            to_type = ""
            if not text.endswith(" "):
                text += " "
                to_type += " "
            text += pred + " "
            to_type += pred + " "
            self.text_buffer.set(text)
            # Learn from accepted suggestion so future predictions improve
            try:
                self.language_model.train(text)
            except Exception:
                pass
            if to_type:
                self._type_text(to_type)
            self.update_predictions()

    def _emit_keypress(self, key):
        if pyautogui is None:
            return
        try:
            self._refocus_last_target()
            special_keys = {"space", "backspace", "enter"}
            if key.lower() in special_keys:
                pyautogui.press(key.lower())
            else:
                pyautogui.write(key)
        except Exception:
            pass  # Avoid crashing the keyboard if the OS rejects the event

    def _type_text(self, text):
        if pyautogui is None:
            return
        try:
            self._refocus_last_target()
            pyautogui.write(text)
        except Exception:
            pass

    def hover_key(self, key):
        """Hook for gaze/gesture dwell highlighting (to be integrated with CV module)."""
        # You can implement key highlighting here if needed
        pass

    def _apply_no_activate_style(self):
        if os.name != "nt":
            return
        try:
            hwnd = self.root.winfo_id()
            GWL_EXSTYLE = -20
            WS_EX_NOACTIVATE = 0x08000000
            user32 = ctypes.windll.user32
            current = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            if not current & WS_EX_NOACTIVATE:
                user32.SetWindowLongW(hwnd, GWL_EXSTYLE, current | WS_EX_NOACTIVATE)
        except Exception as exc:
            print(f"[WARN] Unable to block focus on keyboard window: {exc}")

    def _track_foreground_window(self):
        if os.name != "nt":
            return
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            if hwnd and hwnd != self.keyboard_hwnd:
                self.last_target_hwnd = hwnd
        except Exception:
            pass
        finally:
            self.root.after(250, self._track_foreground_window)

    def _refocus_last_target(self):
        if os.name != "nt" or not self.last_target_hwnd:
            return
        try:
            user32 = ctypes.windll.user32
            hwnd = self.last_target_hwnd
            if user32.IsWindow(hwnd):
                user32.ShowWindow(hwnd, 5)
                user32.SetForegroundWindow(hwnd)
        except Exception:
            pass


# ----------------------------
# Main Entry Point
# ----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = AccessibilityKeyboard(root)
    root.mainloop()