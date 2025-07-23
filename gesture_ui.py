import cv2
import mediapipe as mp
import math
import pyautogui
import time
from collections import deque

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

trail = deque(maxlen=5)
movement_threshold = 40  
last_action_time = time.time()
cooldown = 1  # seconds to avoid spamming commands

media_playing = False  # assume paused at start

# Video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            landmarks = handLms.landmark

            # Finger detection
            fingers = []

            # Thumb conditional
            if landmarks[4].x < landmarks[3].x:
                fingers.append(1)
            else:
                fingers.append(0)

            # Other fingers conditional
            # tip.y < pip.y aka finger is up
            for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
                if landmarks[tip].y < landmarks[pip].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # If all 5 fingers are up, skip gesture detection
            if sum(fingers) == 5:
                trail.clear()
                cv2.putText(frame, "Idle: All fingers up", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                continue

            # Gesture real-time tracking
            index_tip = landmarks[8]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            trail.append((cx, cy))
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            if len(trail) == trail.maxlen:
                x1, y1 = trail[0]
                x2, y2 = trail[-1]
                dx, dy = x2 - x1, y2 - y1
                distance = math.hypot(dx, dy)

                if distance > movement_threshold:
                    angle = math.degrees(math.atan2(-dy, dx))
                    angle = (angle + 360) % 360
                    direction = None

                    if 337.5 <= angle or angle < 22.5:
                        direction = "Right"
                    elif 22.5 <= angle < 67.5:
                        direction = "Up-Right"
                    elif 67.5 <= angle < 112.5:
                        direction = "Up"
                    elif 112.5 <= angle < 157.5:
                        direction = "Up-Left"
                    elif 157.5 <= angle < 202.5:
                        direction = "Left"
                    elif 202.5 <= angle < 247.5:
                        direction = "Down-Left"
                    elif 247.5 <= angle < 292.5:
                        direction = "Down"
                    elif 292.5 <= angle < 337.5:
                        direction = "Down-Right"

                    if direction:
                        now = time.time()
                        if now - last_action_time > cooldown:
                            print(f"Detected gesture: {direction}")

                            # Conditionals for control flow
                            if direction == "Up":
                                for i in range(5):
                                    pyautogui.press("volumeup")
                            elif direction == "Right":
                                for i in range(5):
                                    pyautogui.press("volumedown")
                            elif direction == "Up-Right":
                                if not media_playing:
                                    # Play video/song
                                    pyautogui.press("playpause") 
                                    media_playing = True
                                    print("Playing video")
                                else:
                                    print("Already playing")
                            elif direction == "Down-Left":
                                if media_playing:
                                    # Pause video/song
                                    pyautogui.press("playpause") 
                                    media_playing = False
                                    print("⏸️ Pausing video")
                                else:
                                    print("Already paused")
                            last_action_time = now


                    cv2.putText(frame, f"Gesture: {direction}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Media Control", frame)
    # Exit when 'Esc' is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
