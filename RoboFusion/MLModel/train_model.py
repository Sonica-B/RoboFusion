import cv2
import mediapipe as mp
import numpy as np
from keras import Sequential
from keras.src.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten

from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Initialize MediaPipe Hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Gesture labels
GESTURES = ['Neutral', 'Grip Open', 'Grip Close', 'Move Up', 'Move Down', 'Rotate Left', 'Rotate Right', 'Forward',
            'Backward']
NUM_CLASSES = len(GESTURES)

# Data and labels
data = []
labels = []


def extract_features(results):
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
    return np.zeros(63)  # 21 landmarks * 3 coordinates


def collect_data(label_index):
    cap = cv2.VideoCapture(0)
    print(f"Collecting data for {GESTURES[label_index]}. Press 'q' to stop.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        features = extract_features(results)

        if results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            data.append(features)
            labels.append(label_index)

        cv2.imshow('Collecting Data', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Collect data for all gestures
for i in range(NUM_CLASSES):
    collect_data(i)

# Preprocess data
X = np.array(data)
y = to_categorical(labels, num_classes=NUM_CLASSES)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=1)

# Save the model
model.save('gesture_model.h5')
print("Model model complete and saved as 'gesture_model.h5'.")
