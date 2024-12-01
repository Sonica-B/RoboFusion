# Adding ROS 2 libraries
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from open_manipulator_msgs.srv import SetJointPosition

from RoboFusion.DataCollection.scripts.utils.configs import *

# Function to map wrist position to base motor angle
def map_wrist_to_base_angle(wrist_x, frame_width):
    normalized_x = wrist_x / frame_width  # Normalize x-coordinate (0 to 1)
    return (normalized_x - 0.5) * 2 * BASE_MOTOR_MAX_ANGLE  # Map to -1.57 to +1.57 radians

# ROS Node: Arm Angle Publisher
class ArmAnglePublisher(Node):
    def __init__(self):
        super().__init__('arm_angle_publisher')

        # Create a client for the `/goal_joint_space_path` service
        self.client = self.create_client(SetJointPosition, '/goal_joint_space_path')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /goal_joint_space_path service...")

    def send_to_service(self, joint_positions):
        # Prepare the service request
        request = SetJointPosition.Request()
        request.joint_position.joint_name = ['joint1', 'joint2', 'joint3', 'joint4']
        request.joint_position.position = joint_positions[:4]  # Only include joints 1-4
        request.path_time = 2.0  # Path time for smooth movement

        # Call the service
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.done():
            try:
                response = future.result()
                self.get_logger().info(f"Service call successful: {response}")
            except Exception as e:
                self.get_logger().error(f"Service call failed: {e}")