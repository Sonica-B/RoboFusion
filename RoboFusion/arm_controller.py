import cv2
import mediapipe as mp
import numpy as np
from keras.src.saving import load_model
import rospy2
from ros_gesture_controller import RoboticArmController, GestureType

# Load trained gesture model
model = load_model('gesture_model.h5')

# Initialize MediaPipe Hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Gesture mapping to Enum
GESTURE_MAPPING = {
    'Neutral': GestureType.NEUTRAL,
    'Grip Open': GestureType.GRIP_OPEN,
    'Grip Close': GestureType.GRIP_CLOSE,
    'Move Up': GestureType.MOVE_UP,
    'Move Down': GestureType.MOVE_DOWN,
    'Rotate Left': GestureType.ROTATE_LEFT,
    'Rotate Right': GestureType.ROTATE_RIGHT,
    'Forward': GestureType.FORWARD,
    'Backward': GestureType.BACKWARD
}


class RoboticArmWithGesture(RoboticArmController):
    def run(self):
        cap = cv2.VideoCapture(0)
        print("Gesture-controlled Robotic Arm. Press 'q' to exit.")

        while not rospy2.is_shutdown() and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            features = extract_features(results)

            if results.multi_hand_landmarks:
                # Predict gesture
                prediction = model.predict(features.reshape(1, -1))
                gesture_name = GESTURES[np.argmax(prediction)]
                confidence = np.max(prediction)
                gesture = GESTURE_MAPPING[gesture_name]

                # Print details
                print(f"Gesture: {gesture_name} ({confidence:.2f})")
                print(f"Joint Angles: {self.current_angles}")

                # Process gesture
                self.process_gesture(gesture)

                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            cv2.imshow('Gesture Control', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        controller = RoboticArmWithGesture()
        controller.run()
    except rospy2.ROSInterruptException:
        pass