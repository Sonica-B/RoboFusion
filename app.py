# Adding ROS 2 libraries
import rclpy

# Adding additional libraries
import cv2
import mediapipe as mp
import time
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

from DataCollection.processed_data.labels.gesture_labels import GESTURE_LABELS
from DataCollection.scripts.utils.capture_gestures import initialize_csv, save_gesture_data
from DataCollection.scripts.utils.configs import *
from DataCollection.scripts.utils.file_utils import map_wrist_to_base_angle, ArmAnglePublisher

# Load the trained gesture classification model
gesture_model = tf.keras.models.load_model(MODEL_PATH)

# Load scaler used during training
scaler_path = SCALER_PATH  # Replace with your scaler path
scaler = StandardScaler()
scaler.mean_, scaler.scale_ = np.load(scaler_path)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize Hands model with a max number of 1 hand
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=HAND_DETECTION_CONFIDENCE,
    min_tracking_confidence=HAND_TRACKING_CONFIDENCE
)

def main():
    # Initialize CSV files
    initialize_csv()

    cap = cv2.VideoCapture(0)

    landmark_buffer = []  # Buffer for smoothing predictions
    prediction_buffer = []  # Buffer for gesture prediction smoothing

    rclpy.init()
    angle_publisher = ArmAnglePublisher()

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
        cv2.putText(image, f"Label: {GESTURE_LABELS[current_label] if current_label is not None else 'None'}", (10, 60),
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

                if data_collection_mode == False:
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

                    # Assuming `hand_landmarks` is the detected hand landmarks from MediaPipe
                    index_tip_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    # Normalized y-coordinate (value between 0 and 1)
                    index_tip_y_normalized = index_tip_landmark.y
                    # Convert to pixel coordinate (y-coordinate in the frame)
                    index_tip_y = int(index_tip_y_normalized * height)  # frame_height is the height of the video frame

                    # Display Gripper Angle
                    start_time = time.time()
                    if start_time - end_time >= ANGLE_UPDATE_INTERVAL:
                        # Gripper Angle
                        if gesture_text == "Fist":
                            ARM_ANGLES["gripper_angle"] = max(ARM_ANGLES["gripper_angle"] - 0.1, DEFAULT_ANGLE_LIMIT)  # Close gripper
                        elif gesture_text == "Open Hand":
                            ARM_ANGLES["gripper_angle"] = min(ARM_ANGLES["gripper_angle"] + 0.1, MAX_ANGLE_LIMIT)  # Open gripper

                        # Base Motor Angle
                        base_motor_angle = map_wrist_to_base_angle(wrist_x, width)  # Map wrist x to base motor angle
                        ARM_ANGLES["base_motor_angle"] = max(min(base_motor_angle, MAX_ANGLE_LIMIT), MIN_ANGLE_LIMIT)  # Clamp to [-1.57, 1.57]

                        # Wrist Angle
                        if gesture_text == "Palm Flat Up":
                            ARM_ANGLES["wrist_angle"] = min(ARM_ANGLES["wrist_angle"] + 0.1, MAX_ANGLE_LIMIT)  # Rotate wrist upward
                        elif gesture_text == "Palm Flat Down":
                            ARM_ANGLES["wrist_angle"] = max(ARM_ANGLES["wrist_angle"] - 0.1, MIN_ANGLE_LIMIT)  # Rotate wrist downward
                        # elif index_tip_y < height * 0.4:  # Index finger tip moves up
                        #     ARM_ANGLES["wrist_angle"] = min(ARM_ANGLES["wrist_angle"] + 0.1, 1.57)
                        # elif index_tip_y > height * 0.6:  # Index finger tip moves down
                        #     ARM_ANGLES["wrist_angle"] = max(ARM_ANGLES["wrist_angle"] - 0.1, -1.57)

                        # Elbow Angle
                        if gesture_text == "Thumbs Down":
                            ARM_ANGLES["elbow_angle"] = min(ARM_ANGLES["elbow_angle"] + 0.1, MAX_ANGLE_LIMIT)  # Increment elbow angle
                        elif gesture_text == "Thumbs Up":
                            ARM_ANGLES["elbow_angle"] = max(ARM_ANGLES["elbow_angle"] - 0.1, MIN_ANGLE_LIMIT)  # Decrement elbow angle
                        # elif wrist_y < height * 0.4:  # Hand moves up
                        #     ARM_ANGLES["elbow_angle"] = min(ARM_ANGLES["elbow_angle"] + 0.1, 1.57)
                        # elif wrist_y > height * 0.6:  # Hand moves down
                        #     ARM_ANGLES["elbow_angle"] = max(ARM_ANGLES["elbow_angle"] - 0.1, -1.57)

                        # Shoulder Angle
                        # Vertical Shoulder Control
                        if wrist_y > height * 0.6:
                            ARM_ANGLES["shoulder_vertical_angle"] = min(ARM_ANGLES["shoulder_vertical_angle"] + 0.1, MAX_ANGLE_LIMIT)  # Move up
                        elif wrist_y < height * 0.4:
                            ARM_ANGLES["shoulder_vertical_angle"] = max(ARM_ANGLES["shoulder_vertical_angle"] - 0.1, MIN_ANGLE_LIMIT)  # Move down

                        # Send updated angles to the robot
                        joint_positions = [
                            ARM_ANGLES["base_motor_angle"],
                            ARM_ANGLES["shoulder_vertical_angle"],
                            ARM_ANGLES["elbow_angle"],
                            ARM_ANGLES["gripper_angle"],
                        ]
                        angle_publisher.send_to_service(joint_positions)

                        end_time = start_time
                    # Display motor angles
                    cv2.putText(image, f"Gripper: {ARM_ANGLES['gripper_angle']:.2f} rad", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 255, 0), 2)
                    cv2.putText(image, f"Base: {ARM_ANGLES['base_motor_angle']:.2f} rad", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 255, 0), 2)
                    cv2.putText(image, f"Elbow: {ARM_ANGLES['elbow_angle']:.2f} rad", (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,(0, 255, 0), 2)
                    cv2.putText(image, f"Wrist: {ARM_ANGLES['wrist_angle']:.2f} rad", (10, 140), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,(0, 255, 0), 2)
                    cv2.putText(image, f"Shoulder V: {ARM_ANGLES['shoulder_vertical_angle']:.2f} rad", (10, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 255, 0), 2)
                    cv2.putText(image, f"Shoulder H: {ARM_ANGLES['shoulder_horizontal_angle']:.2f} rad", (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 255, 0), 2)

        # Save data during data collection
        if row_data and data_collection_mode and current_label is not None:
            current_time = time.time()
            if current_time - last_save_time >= FRAME_UPDATE_INTERVAL:
                save_gesture_data(row_data, current_label)
                print(f"Captured hand data for label {current_label}")
                last_save_time = current_time

        cv2.imshow("Robotic Arm Control using Hand Gestures", image)

        # Handle keypress for mode switching and label updates
        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # ESC to exit
            break
        elif key == ord('m'):  # Toggle data collection mode
            data_collection_mode = not data_collection_mode
            print(f"Data collection mode: {'ON' if data_collection_mode else 'OFF'}")
            if not data_collection_mode:
                current_label = None  # Reset label when exiting data collection mode
        elif ord('0') <= key <= ord('9') and data_collection_mode:  # Update label during data collection mode
            new_label = int(chr(key))  # Convert pressed key to an integer
            if new_label in GESTURE_LABELS:
                current_label = new_label
                print(f"Switched to label {current_label}: {GESTURE_LABELS[current_label]}")
            else:
                print(f"Invalid label: {new_label}")

    cap.release()
    cv2.destroyAllWindows()
    angle_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()