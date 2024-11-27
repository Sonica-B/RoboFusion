# Model Path
MODEL_PATH = "D:/RoboFusion2/RoboFusion/MLModel/models/gesture_classification_model.keras"

# Video Parameters
VIDEO_FEED_WIDTH = 640
VIDEO_FEED_HEIGHT = 480

# Hand Detection Parameters
HAND_DETECTION_CONFIDENCE = 0.6
HAND_TRACKING_CONFIDENCE = 0.7
MAX_HANDS = 1

FRAME_UPDATE_INTERVAL = 0.1  # seconds between updates for data collection
ANGLE_UPDATE_INTERVAL = 1    # seconds between updates for robotic arm joint angles

data_collection_mode = False
current_label = None
last_save_time = 0
end_time = 0

# Default Angles values for robotic arm in radians
ARM_ANGLES = {
    "gripper_angle": 0.0,
    "base_motor_angle": 0.0,
    "elbow_angle": 0.0,
    "wrist_angle": 0.0,
    "shoulder_vertical_angle": 0.0,
    "shoulder_horizontal_angle": 0.0,
}

# Base Motor Parameters
BASE_MOTOR_MIN_ANGLE = -1.57  # radians
BASE_MOTOR_MAX_ANGLE = 1.57  # radians
BASE_MOTOR_CENTER_TOLERANCE = 0.05  # tolerance for dead zone in normalized wrist position
SHOULDER_MAX_ANGLE = 1.57  # Max angle for shoulder
WRIST_MAX_ANGLE = 1.57  # Max angle for wrist
ELBOW_MAX_ANGLE = 1.57  # Max angle for elbow