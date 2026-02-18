#!/usr/bin/env python3
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import os


class GestureRecognition:
    def __init__(self):
        # MediaPipe initialization
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Gesture labels
        self.gestures = {
            0: 'NEUTRAL',
            1: 'GRIP_OPEN',
            2: 'GRIP_CLOSE',
            3: 'MOVE_UP',
            4: 'MOVE_DOWN',
            5: 'ROTATE_LEFT',
            6: 'ROTATE_RIGHT',
            7: 'FORWARD',
            8: 'BACKWARD'
        }

        # Model parameters
        self.input_shape = 63  # 21 landmarks × 3 coordinates
        self.model = self._build_model()

        # Performance tracking
        self.prediction_history = []
        self.fps_history = []

    def _build_model(self):
        """Build and compile the gesture recognition model"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(self.input_shape,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(len(self.gestures), activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        return model

    def collect_training_data(self, num_samples_per_gesture=100):
        """Collect model data through webcam"""
        X = []
        y = []

        cap = cv2.VideoCapture(0)

        for gesture_id, gesture_name in self.gestures.items():
            print(f"\nCollecting data for gesture: {gesture_name}")
            print(f"Press 'c' to start collecting {num_samples_per_gesture} samples")

            while True:
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    break

                ret, frame = cap.read()
                cv2.putText(frame, f"Ready to collect: {gesture_name}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Data Collection', frame)

            samples_collected = 0
            while samples_collected < num_samples_per_gesture:
                ret, frame = cap.read()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    landmarks = self._extract_landmarks(results.multi_hand_landmarks[0])
                    X.append(landmarks)
                    y.append(gesture_id)
                    samples_collected += 1

                    self.mp_draw.draw_landmarks(
                        frame, results.multi_hand_landmarks[0],
                        self.mp_hands.HAND_CONNECTIONS
                    )

                cv2.putText(frame, f"Collecting {gesture_name}: {samples_collected}/{num_samples_per_gesture}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Data Collection', frame)
                cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()

        return np.array(X), np.array(y)

    def train_model(self, X, y, validation_split=0.2, epochs=50, batch_size=32):
        """Train the gesture recognition model"""
        # Convert labels to one-hot encoding
        y_onehot = tf.keras.utils.to_categorical(y)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_onehot, test_size=validation_split, random_state=42
        )

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint('best_gesture_model.keras', monitor='val_accuracy',
                            save_best_only=True, mode='max')
        ]

        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate model
        y_pred = self.model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val, axis=1)

        # Calculate metrics
        precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

        # Plot confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('confusion_matrix.png')

        # Save metrics
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'history': history.history
        }

        with open('training_metrics.pkl', 'wb') as f:
            pickle.dump(metrics, f)

        print("\nTraining Results:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return history

    def _extract_landmarks(self, hand_landmarks):
        """Extract landmarks from MediaPipe hand detection"""
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

    def predict_gesture(self, frame):
        """Predict gesture from a single frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            landmarks = self._extract_landmarks(results.multi_hand_landmarks[0])
            prediction = self.model.predict(landmarks.reshape(1, -1), verbose=0)
            gesture_id = np.argmax(prediction[0])
            confidence = prediction[0][gesture_id]

            # Draw landmarks
            self.mp_draw.draw_landmarks(
                frame, results.multi_hand_landmarks[0],
                self.mp_hands.HAND_CONNECTIONS
            )

            return self.gestures[gesture_id], confidence

        return None, 0.0

    def run_live_prediction(self):
        """Run live gesture prediction from webcam"""
        cap = cv2.VideoCapture(0)
        prev_time = datetime.now()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Predict gesture
            gesture, confidence = self.predict_gesture(frame)

            # Calculate FPS
            current_time = datetime.now()
            fps = 1.0 / (current_time - prev_time).total_seconds()
            prev_time = current_time

            # Display results
            if gesture:
                cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Gesture Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Initialize gesture recognition system
    gr = GestureRecognition()

    # Collect model data
    print("Starting data collection...")
    X, y = gr.collect_training_data(num_samples_per_gesture=100)

    # Train model
    print("\nStarting model model...")
    history = gr.train_model(X, y)

    # Run live prediction
    print("\nStarting live prediction...")
    gr.run_live_prediction()