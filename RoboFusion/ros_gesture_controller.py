#!/usr/bin/env python3
import rospy2
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
import numpy as np
from enum import Enum
import math

class GestureType(Enum):
    NEUTRAL = 0
    GRIP_OPEN = 1
    GRIP_CLOSE = 2
    MOVE_UP = 3
    MOVE_DOWN = 4
    ROTATE_LEFT = 5
    ROTATE_RIGHT = 6
    FORWARD = 7
    BACKWARD = 8

class RoboticArmController:
    def __init__(self):
        rospy2.init_node('gesture_arm_controller')
        
        # Initialize joint state publishers
        self.joint_publishers = {
            'base_rotation': rospy2.Publisher('/arm/joint1_position_controller/command', Float64, queue_size=10),
            'shoulder': rospy2.Publisher('/arm/joint2_position_controller/command', Float64, queue_size=10),
            'elbow': rospy2.Publisher('/arm/joint3_position_controller/command', Float64, queue_size=10),
            'gripper': rospy2.Publisher('/arm/gripper_position_controller/command', Float64, queue_size=10)
        }
        
        # Current joint states
        self.current_angles = {
            'base_rotation': 0.0,
            'shoulder': 0.0,
            'elbow': 0.0,
            'gripper': 0.0
        }
        
        # Movement increment values (in radians)
        self.angle_increments = {
            'base_rotation': math.pi/12,  # 15 degrees
            'shoulder': math.pi/16,       # 11.25 degrees
            'elbow': math.pi/16,          # 11.25 degrees
            'gripper': math.pi/6          # 30 degrees
        }
        
        # Joint limits (in radians)
        self.joint_limits = {
            'base_rotation': (-math.pi, math.pi),        # -180 to 180 degrees
            'shoulder': (-math.pi/2, math.pi/2),         # -90 to 90 degrees
            'elbow': (-2*math.pi/3, 2*math.pi/3),       # -120 to 120 degrees
            'gripper': (0, math.pi/2)                    # 0 to 90 degrees
        }
        
        # Subscribe to joint states
        rospy2.Subscriber('/arm/joint_states', JointState, self.joint_state_callback)
        
        self.rate = rospy2.Rate(10)  # 10Hz control rate
        
    def joint_state_callback(self, msg):
        """Update current joint states from robot feedback"""
        for i, name in enumerate(msg.name):
            if name in self.current_angles:
                self.current_angles[name] = msg.position[i]
    
    def check_limits(self, joint_name, new_angle):
        """Check if new angle is within joint limits"""
        min_angle, max_angle = self.joint_limits[joint_name]
        return max(min_angle, min(max_angle, new_angle))
    
    def process_gesture(self, gesture: GestureType):
        """Process gesture and calculate new joint angles"""
        if gesture == GestureType.NEUTRAL:
            # Return to default position
            for joint in self.current_angles:
                self.current_angles[joint] = 0.0
                self.publish_joint_command(joint, 0.0)
            
        elif gesture == GestureType.GRIP_OPEN:
            new_angle = self.current_angles['gripper'] + self.angle_increments['gripper']
            new_angle = self.check_limits('gripper', new_angle)
            self.publish_joint_command('gripper', new_angle)
            
        elif gesture == GestureType.GRIP_CLOSE:
            new_angle = self.current_angles['gripper'] - self.angle_increments['gripper']
            new_angle = self.check_limits('gripper', new_angle)
            self.publish_joint_command('gripper', new_angle)
            
        elif gesture == GestureType.MOVE_UP:
            new_angle = self.current_angles['shoulder'] + self.angle_increments['shoulder']
            new_angle = self.check_limits('shoulder', new_angle)
            self.publish_joint_command('shoulder', new_angle)
            
        elif gesture == GestureType.MOVE_DOWN:
            new_angle = self.current_angles['shoulder'] - self.angle_increments['shoulder']
            new_angle = self.check_limits('shoulder', new_angle)
            self.publish_joint_command('shoulder', new_angle)
            
        elif gesture == GestureType.ROTATE_LEFT:
            new_angle = self.current_angles['base_rotation'] + self.angle_increments['base_rotation']
            new_angle = self.check_limits('base_rotation', new_angle)
            self.publish_joint_command('base_rotation', new_angle)
            
        elif gesture == GestureType.ROTATE_RIGHT:
            new_angle = self.current_angles['base_rotation'] - self.angle_increments['base_rotation']
            new_angle = self.check_limits('base_rotation', new_angle)
            self.publish_joint_command('base_rotation', new_angle)
            
        elif gesture == GestureType.FORWARD:
            new_angle = self.current_angles['elbow'] + self.angle_increments['elbow']
            new_angle = self.check_limits('elbow', new_angle)
            self.publish_joint_command('elbow', new_angle)
            
        elif gesture == GestureType.BACKWARD:
            new_angle = self.current_angles['elbow'] - self.angle_increments['elbow']
            new_angle = self.check_limits('elbow', new_angle)
            self.publish_joint_command('elbow', new_angle)
    
    def publish_joint_command(self, joint_name, angle):
        """Publish command to specific joint"""
        msg = Float64()
        msg.data = angle
        self.joint_publishers[joint_name].publish(msg)
        self.current_angles[joint_name] = angle
        
    def run(self):
        """Main control loop"""
        while not rospy2.is_shutdown():
            # Here you would receive gesture recognition results
            # For testing, you can manually set gestures
            self.rate.sleep()

def main():
    try:
        controller = RoboticArmController()
        
        # Example usage:
        controller.process_gesture(GestureType.MOVE_UP)
        rospy2.sleep(1)
        controller.process_gesture(GestureType.ROTATE_LEFT)
        rospy2.sleep(1)
        controller.process_gesture(GestureType.GRIP_OPEN)
        
        controller.run()
        
    except rospy2.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
