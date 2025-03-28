import cv2
import mediapipe as mp
import numpy as np
import os
import glob

# Initialize MediaPipe Hand module
mp_drawing = mp.solutions.drawing_utils
mphands = mp.solutions.hands

# Initialize webcam
cap = cv2.VideoCapture(0)
hands = mphands.Hands()

# Map keys to gestures
GESTURE_KEYS = {
    'r': "rock",
    'p': "paper",
    's': "scissors",
    'h': "heart",
    'o': "phone"
}

# Ensure the landmarks folder exists
if not os.path.exists("landmarks"):
    os.makedirs("landmarks")

def get_next_filename(gesture_name):
    """Find the next available filename by checking existing files."""
    existing_files = glob.glob(f"landmarks/{gesture_name}_landmarks*.npy")
    
    # Extract numbers at the end of the filename
    numbers = []
    for file in existing_files:
        try:
            num = int(file.split("_landmarks")[-1].split(".")[0])  # Extract number
            numbers.append(num)
        except ValueError:
            continue

    next_number = max(numbers) + 1 if numbers else 1
    return f"landmarks/{gesture_name}_landmarks{next_number}.npy"  # Save in landmarks folder

def save_landmarks(gesture_name, landmarks):
    """Save landmarks to a numbered .npy file."""
    landmarks = np.array(landmarks).flatten()
    filename = get_next_filename(gesture_name)
    np.save(filename, landmarks)
    print(f"Landmarks for '{gesture_name}' saved as '{filename}'!")

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert frame for processing
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Convert back to BGR for display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw landmarks and check for saving
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mphands.HAND_CONNECTIONS
            )

            # Extract landmarks (21 x 3)
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

            # Wait for user input to save landmarks
            key = cv2.waitKey(1) & 0xFF
            if chr(key) in GESTURE_KEYS:
                save_landmarks(GESTURE_KEYS[chr(key)], landmarks)

    # Display the image
    cv2.imshow('HandTracker', image)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
