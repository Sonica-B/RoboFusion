import cv2
import mediapipe as mp
import numpy as np
from keras.src.saving import load_model

# Load trained model
model = load_model('gesture_model.h5')

# Initialize MediaPipe Hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

GESTURES = ['Neutral', 'Grip Open', 'Grip Close', 'Move Up', 'Move Down', 'Rotate Left', 'Rotate Right', 'Forward',
            'Backward']


def extract_features(results):
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
    return np.zeros(63)  # 21 landmarks * 3 coordinates


cap = cv2.VideoCapture(0)

print("Testing gesture prediction. Press 'q' to exit.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    features = extract_features(results)

    if results.multi_hand_landmarks:
        prediction = model.predict(features.reshape(1, -1))
        gesture = GESTURES[np.argmax(prediction)]
        confidence = np.max(prediction)
        cv2.putText(frame, f"{gesture} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Gesture Prediction', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
