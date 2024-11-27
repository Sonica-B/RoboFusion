from RoboFusion.DataCollection.scripts.utils.configs import *

# Function to map wrist position to base motor angle
def map_wrist_to_base_angle(wrist_x, frame_width):
    normalized_x = wrist_x / frame_width  # Normalize x-coordinate (0 to 1)
    return (normalized_x - 0.5) * 2 * BASE_MOTOR_MAX_ANGLE  # Map to -1.57 to +1.57 radians