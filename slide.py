import cv2
import mediapipe as mp
import math
import pyautogui
import time
import pygetwindow as gw
from collections import deque

# Window application names
apps = ["Edge", "Spotify", "Discord"]
window_names = {
    "Edge": "edge",
    "Spotify": "spotify",
    "Discord": "discord"
}
current_app_index = 0
media_playing = False

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

trail = deque(maxlen=5)
movement_threshold = 40
idle_threshold = 15
entry_ignore_frames = 10  # ignore this many frames after hand appears
entry_frame_counter = 0
hand_was_present = False
last_action_time = time.time()
cooldown = 1

cap = cv2.VideoCapture(0)

def switch_to(app_title):
    for win in gw.getAllWindows():
        print("Window title:", win.title)
        title = win.title.lower()
        if app_title.lower() in title:
            try:
                print(f"Switching to {app_title} window")
                # In case it's minimized
                win.restore() 
                win.activate()
                return True
            except:
                pass
    print(f"Window containing '{app_title}' not found.")
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        if not hand_was_present:
            entry_frame_counter = 0
            hand_was_present = True
        else:
            entry_frame_counter += 1

        if entry_frame_counter < entry_ignore_frames:
            cv2.putText(frame, "Ignoring hand (just appeared)", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
            cv2.imshow("Gesture App Switcher", frame)
            continue

        for handLms in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            landmarks = handLms.landmark

            fingers = []
            fingers.append(1 if landmarks[8].y < landmarks[6].y else 0)
            fingers.append(1 if landmarks[12].y < landmarks[10].y else 0)
            fingers.append(1 if landmarks[16].y < landmarks[14].y else 0)
            fingers.append(1 if landmarks[20].y < landmarks[18].y else 0)
            fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
            finger_count = sum(fingers)

            index_tip = landmarks[8]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            trail.append((cx, cy))

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            if len(trail) == trail.maxlen:
                x1, y1 = trail[0]
                x2, y2 = trail[-1]
                dx, dy = x2 - x1, y2 - y1
                distance = math.hypot(dx, dy)

                cv2.putText(frame, f"Movement: {int(distance)} px", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                angle = math.degrees(math.atan2(-dy, dx))
                angle = (angle + 360) % 360
                direction = None

                # Determine direction based on angle
                if 22.5 <= angle < 67.5:
                    direction = "Up-Right"
                elif 247.5 <= angle < 292.5:
                    direction = "Down"
                elif 337.5 <= angle or angle < 22.5:
                    direction = "Right"
                elif 67.5 <= angle < 112.5:
                    direction = "Up"
                elif 157.5 <= angle < 202.5:
                    direction = "Left"
                elif 202.5 <= angle < 247.5:
                    direction = "Down-Left"

                now = time.time()
                if now - last_action_time > cooldown:
                    # Check if all fingers are up
                    if finger_count == 5:
                        if distance > idle_threshold:
                            # Change application based on direction
                            if direction == "Right":
                                print("Switching to next app (5-finger swipe)")
                                current_app_index = (current_app_index + 1) % len(apps)
                                switch_to(window_names[apps[current_app_index]])
                            elif direction == "Left":
                                print("Switching to previous app (5-finger swipe)")
                                current_app_index = (current_app_index - 1) % len(apps)
                                switch_to(window_names[apps[current_app_index]])
                            last_action_time = now
                        else:
                            # Idle state with all fingers up
                            trail.clear()
                            cv2.putText(frame, "Idle: All fingers up", (10, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    elif finger_count < 5:
                        if distance > idle_threshold:
                            # Perform actions based direction
                            if direction == "Up" and fingers[0]:
                                for _ in range(5):
                                    pyautogui.press("volumeup")
                                last_action_time = now
                            elif direction == "Down" and fingers[0]:
                                for _ in range(5):
                                    pyautogui.press("volumedown")
                                last_action_time = now
                            elif direction == "Up-Right" and fingers[0]:
                                if not media_playing:
                                    pyautogui.press("playpause")
                                    media_playing = True
                                    print("Playing video")
                                else:
                                    print("Already playing")
                                last_action_time = now
                            elif direction == "Down-Left" and fingers[0]:
                                if media_playing:
                                    pyautogui.press("playpause")
                                    media_playing = False
                                    print("Pausing video")
                                else:
                                    print("Already paused")
                                last_action_time = now
                            elif direction == "Right" and fingers[0]:
                                print
                                pyautogui.hotkey("shift", "n")
                                pyautogui.press("nexttrack")
                        else:
                            trail.clear()
                            cv2.putText(frame, "Idle: Hand stationary", (10, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if direction:
                    cv2.putText(frame, f"Gesture: {direction}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        hand_was_present = False

    cv2.imshow("Gesture App Switcher", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
