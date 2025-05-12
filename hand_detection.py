import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.distance import euclidean
import glo

# Initialize MediaPipe Hand module
mp_drawing = mp.solutions.drawing_utils
mphands = mp.solutions.hands

# Function to load all landmark files for a given gesture
def load_gesture_landmarks(gesture_name):
    files = sorted(glob.glob(f"landmarks/{gesture_name}_landmarks*.npy"))  # Find all matching files
    return [np.load(f).flatten() for f in files] if files else []

# Predefined gestures with dynamically loaded landmarks
GESTURES = {
    "rock": load_gesture_landmarks("rock"),
    "paper": load_gesture_landmarks("paper"),
    "scissors": load_gesture_landmarks("scissors"),
    "heart": load_gesture_landmarks("heart"),
    "phone": load_gesture_landmarks("phone"),
}



def recognize_gesture(landmarks):
    """Match landmarks to predefined gestures using multiple reference sets."""
    landmarks = landmarks.flatten() if len(landmarks.shape) > 1 else landmarks
    distances = {}

    # Compare landmarks to each reference set for each gesture
    for gesture, refs in GESTURES.items():
        gesture_distances = [euclidean(landmarks, ref) for ref in refs]
        distances[gesture] = min(gesture_distances)  # Use the closest match from multiple sets
    
    # Return the gesture with the minimum distance and its confidence (distance)
    recognized_gesture = min(distances, key=distances.get)
    return recognized_gesture, distances[recognized_gesture]

# Start video capture
cap = cv2.VideoCapture(0)
hands = mphands.Hands()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert frame for processing
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Convert back to BGR for OpenCV display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the image
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mphands.HAND_CONNECTIONS
            )

            # Extract landmarks (21 x 3)
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

            # Recognize gesture by comparing landmarks
            gesture, confidence = recognize_gesture(landmarks)
            
            # Display the recognized gesture and confidence
            cv2.putText(
                image,
                f"Gesture: {gesture} (Confidence: {confidence:.2f})",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

    # Show the image with landmarks and gesture text
    cv2.imshow('HandGestureRecognition', image)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
