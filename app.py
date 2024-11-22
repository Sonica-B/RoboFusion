import cv2
import mediapipe as mp
import time
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from DataCollection.processed_data.labels.gesture_labels import GESTURE_LABELS
from DataCollection.scripts.capture_gestures import initialize_csv, save_gesture_data
from DataCollection.scripts.utils.file_utils import get_xy_coords

# Load the trained gesture classification model
model_path = "D:/RoboFusion2/RoboFusion/RoboFusion/MLModel/models/gesture_classification_model.keras"
gesture_model = tf.keras.models.load_model(model_path)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize Hands model with a max number of 1 hand
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.7)

# Load scaler used during training
scaler_path = "D:/RoboFusion2/RoboFusion/RoboFusion/MLModel/models/scaler.pkl.npy"  # Replace with your scaler path
scaler = StandardScaler()
scaler.mean_, scaler.scale_ = np.load(scaler_path)

# Initialize CSV files
initialize_csv()

cap = cv2.VideoCapture(0)

data_collection_mode = False
current_label = None
last_save_time = 0
landmark_buffer = []  # Buffer for smoothing predictions
prediction_buffer = []  # Buffer for gesture prediction smoothing

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    height, width = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process image using Hands model
    results = hands.process(image_rgb)

    # Draw landmarks on image
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    row_data = []

    # Display mode
    cv2.putText(image, f"Mode: {'Data Collection' if data_collection_mode else 'Normal Operation'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(image, f"Label: {current_label if current_label is not None else 'None'}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            # Collect data for saving and prediction
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            flattened_landmarks = np.array(landmarks).flatten()
            row_data.extend(flattened_landmarks)

            # Preprocess landmarks
            scaled_landmarks = scaler.transform(flattened_landmarks.reshape(1, -1))

            # Smooth prediction
            prediction = gesture_model.predict(scaled_landmarks)
            prediction_buffer.append(prediction)
            if len(prediction_buffer) > 5:
                prediction_buffer.pop(0)
            averaged_prediction = np.mean(prediction_buffer, axis=0)

            predicted_label = np.argmax(averaged_prediction)
            confidence = np.max(averaged_prediction)

            # Display predicted gesture on the frame
            gesture_text = GESTURE_LABELS.get(predicted_label, "Unknown")
            wrist_coords = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            wrist_x, wrist_y = int(wrist_coords.x * width), int(wrist_coords.y * height)
            cv2.putText(image, f"{gesture_text} ({confidence:.2f})", (wrist_x, wrist_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Save data during data collection
    if row_data and data_collection_mode and current_label is not None:
        current_time = time.time()
        if current_time - last_save_time >= 0.1:
            save_gesture_data(row_data, current_label)
            print(f"Captured hand data for label {current_label}")
            last_save_time = current_time

    cv2.imshow("Hand Gesture Data Collection", image)

    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # ESC to exit
        break
    elif key == ord('m'):
        data_collection_mode = not data_collection_mode
        current_label = None if not data_collection_mode else current_label
    elif ord('0') <= key <= ord('9') and data_collection_mode:
        current_label = int(chr(key)) if int(chr(key)) in GESTURE_LABELS.keys() else current_label

cap.release()
cv2.destroyAllWindows()
