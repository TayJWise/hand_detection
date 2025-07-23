import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.distance import euclidean
import glob
import tkinter as tk
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk, ImageDraw, ImageFont
import random

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mphands = mp.solutions.hands

QUOTES = [
    "I smell snow ❄️",
    "Oy with the poodles already!",
    "Coffee, coffee, coffee!",
    "You jump, I jump, Jack.",
    "I’m attracted to pie.",
    "This is a jumbo coffee morning."
]

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    origin = landmarks[0]
    landmarks -= origin
    max_val = np.max(np.abs(landmarks))
    if max_val > 0:
        landmarks /= max_val
    return landmarks.flatten()

def load_gesture_landmarks(gesture_name):
    files = sorted(glob.glob(f"landmarks/{gesture_name}_landmarks*.npy"))
    return [normalize_landmarks(np.load(f)) for f in files] if files else []

GESTURES = {
    "rock": load_gesture_landmarks("rock"),
    "paper": load_gesture_landmarks("paper"),
    "scissors": load_gesture_landmarks("scissors"),
    "heart": load_gesture_landmarks("heart"),
    "phone": load_gesture_landmarks("phone"),
}

def recognize_gesture(landmarks):
    landmarks = normalize_landmarks(landmarks)
    distances = {}
    for gesture, refs in GESTURES.items():
        if not refs:
            continue
        gesture_distances = [euclidean(landmarks, ref) for ref in refs]
        distances[gesture] = sorted(gesture_distances)[:3]
    averaged = {g: np.mean(d) for g, d in distances.items()}
    best = min(averaged, key=averaged.get)
    sorted_vals = sorted(averaged.values())
    if len(sorted_vals) > 1 and sorted_vals[0] / sorted_vals[1] > 0.85:
        return "Unknown", sorted_vals[0]
    return best, averaged[best]

class GestureApp:
    def __init__(self, window):
        self.window = window
        self.window.title("☕ Stars Hollow Gesture App ☕")
        self.window.configure(bg="#fefae0")
        self.window.state("zoomed")
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        self.main_frame = Frame(window, bg="#fefae0")
        self.main_frame.pack(fill="both", expand=True)

        self.camera_frame = Frame(self.main_frame, bg="#f5ede0",
            highlightbackground="#6b4c3b", highlightthickness=3, bd=0)
        self.camera_frame.pack(side="left", padx=40, pady=30)

        self.camera_title = Label(self.camera_frame,
            text="Your Cozy Cam 🎥", font=("Georgia", 16, "bold"),
            bg="#f5ede0", fg="#2a2a3b")
        self.camera_title.pack(pady=(5, 0))

        self.video_frame = Label(self.camera_frame, bg="#f5ede0",
            highlightbackground="#6b4c3b", highlightthickness=2, bd=0)
        self.video_frame.pack(padx=10, pady=10)

        self.confidence_label = Label(self.camera_frame,
            text="", font=("Georgia", 14), bg="#f5ede0", fg="#6b4c3b")
        self.confidence_label.pack(pady=(0, 5))

        self.quote_label = Label(self.camera_frame,
            text="", font=("Georgia", 14, "italic"), bg="#f5ede0", fg="#2a2a3b",
            wraplength=350, justify="center")
        self.quote_label.pack(pady=(0, 15))

        self.menu_frame = Frame(self.main_frame, bg="#e6ccb2", width=350)
        self.menu_frame.pack(side="right", fill="y", padx=(0, 40), pady=30)

        self.menu_inner = Frame(self.menu_frame, bg="#e6ccb2")
        self.menu_inner.pack(pady=(60, 10), padx=20, fill="both", expand=True)

        self.gesture_label = Label(self.menu_inner, text="Gesture: ...",
                                   font=("Georgia", 20, "bold"),
                                   bg="#e6ccb2", fg="#2a2a3b")
        self.gesture_label.pack(pady=(0, 10), anchor="w")

        self.result_label = Label(self.menu_inner, text="",
                                  font=("Georgia", 14),
                                  bg="#e6ccb2", fg="#6b4c3b",
                                  wraplength=300, justify="left")
        self.result_label.pack(pady=(0, 20), anchor="w")

        self.button_frame = Frame(self.menu_inner, bg="#e6ccb2")
        self.button_frame.pack(pady=10)

        self.start_button = Button(self.button_frame, text="🎮 Begin Match",
                                   command=self.start_countdown,
                                   bg="#6b4c3b", fg="white",
                                   font=("Georgia", 14, "bold"), relief="flat", bd=0,
                                   highlightthickness=0, width=22, padx=10, pady=5,
                                   cursor="hand2")
        self.start_button.pack(ipadx=10, ipady=10, pady=(0, 15))

        self.quit_button = Button(self.button_frame, text="🏡 Exit to Stars Hollow",
                                  command=self.close,
                                  bg="#a52a2a", fg="white",
                                  font=("Georgia", 14, "bold"), relief="flat", bd=0,
                                  highlightthickness=0, width=22, padx=10, pady=5,
                                  cursor="hand2")
        self.quit_button.pack(ipadx=10, ipady=10)

        self.toggle_button = Button(self.main_frame, text="Hide Menu", command=self.toggle_menu,
                                    font=("Georgia", 10), bg="#6b4c3b", fg="white", relief="flat")
        self.toggle_button.place(relx=0.98, rely=0.02, anchor="ne")

        self.menu_visible = True
        self.cap = cv2.VideoCapture(0)
        self.hands = mphands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.85,
            min_tracking_confidence=0.85
        )

        self.current_frame = None
        self.overlay_text = ""
        self.overlay_step = -1
        self.overlay_result = ""
        self.window.after(0, self.update)

 

    def toggle_menu(self):
        if self.menu_visible:
            self.menu_frame.pack_forget()
            self.toggle_button.config(text="Show Menu")
        else:
            self.menu_frame.pack(side="right", fill="y")
            self.toggle_button.config(text="Hide Menu")
        self.menu_visible = not self.menu_visible

    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        display_frame = frame.copy()

        gesture = "Unknown"
        confidence = 0.0
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(display_frame, hand_landmarks, mphands.HAND_CONNECTIONS)
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                gesture, confidence = recognize_gesture(landmarks)
                break

        self.gesture_label.config(text=f"Gesture: {gesture}")
        self.confidence_label.config(text=f"Confidence: {confidence:.2f}")

        img = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))

        if self.overlay_step >= 0 or self.overlay_result:
            img = img.convert("RGBA")
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 180))
            img = Image.alpha_composite(img, overlay)

            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("Georgia.ttf", 80)
            except:
                font = ImageFont.truetype("arial.ttf", 80)

            center = (img.width // 2, img.height // 2)
            text = self.overlay_text if self.overlay_step >= 0 else self.overlay_result
            draw.text(center, text, font=font, anchor="mm", fill=(255, 228, 196))
            draw.text((img.width - 70, img.height - 80), "☕", font=font, anchor="lt", fill="white")

        img = img.convert("RGB")
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)
        self.window.after(10, self.update)

    def start_countdown(self):
        self.result_label.config(text="")
        self.countdown_sequence = ["1", "2", "3", "THROW!"]
        self.overlay_result = ""
        self.overlay_step = 0
        self.show_countdown_step()

    def show_countdown_step(self):
        if self.overlay_step < len(self.countdown_sequence):
            self.overlay_text = self.countdown_sequence[self.overlay_step]
            self.overlay_step += 1
            self.window.after(1000, self.show_countdown_step)
        else:
            self.overlay_step = -1
            self.overlay_text = ""
            self.evaluate_throw()

    def evaluate_throw(self):
        ret, latest = self.cap.read()
        if ret:
            latest = cv2.flip(latest, 1)
            self.current_frame = Image.fromarray(cv2.cvtColor(latest, cv2.COLOR_BGR2RGB))
        else:
            self.result_label.config(text="⚠️ Camera error")
            return

        frame = cv2.cvtColor(np.array(self.current_frame), cv2.COLOR_RGB2BGR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        user_gesture = "Unknown"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                user_gesture, _ = recognize_gesture(landmarks)
                break

        valid_gestures = ["rock", "paper", "scissors"]
        computer_gesture = random.choice(valid_gestures)

        result = "🤝 DRAW!"
        if user_gesture != "Unknown":
            if (user_gesture == "rock" and computer_gesture == "scissors") or \
               (user_gesture == "scissors" and computer_gesture == "paper") or \
               (user_gesture == "paper" and computer_gesture == "rock"):
                result = "✅ YOU WIN!"
            elif user_gesture == computer_gesture:
                result = "🤝 DRAW!"
            else:
                result = "❌ YOU LOSE!"
        else:
            result = "🤷 COULDN'T READ HAND"

        quote = random.choice(QUOTES)
        self.result_label.config(
            text=f"You: {user_gesture} | Computer: {computer_gesture}\n\n\u201c{quote}”"
        )
        self.overlay_result = result
        self.window.after(2000, self.clear_overlay_result)

    def clear_overlay_result(self):
        self.overlay_result = ""

    def close(self):
        self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()
