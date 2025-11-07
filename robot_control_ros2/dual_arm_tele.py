#!/usr/bin/env python3
'''
ROS2 script for dual-arm teleoperation.
- Listens to '/teleop/left_tele_mode' to control the left arm.
- Listens to '/teleop/right_tele_mode' to control the right arm.
'''

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
import threading
import numpy as np
import time

class DualArmTeleop(Node):
    def __init__(self):
        super().__init__('dual_arm_teleop_node')

        # --- State Flags for each arm ---
        self.left_tele_enabled = False
        self.right_tele_enabled = False

        # --- Data storage for each controller ---
        self.left_controller_pose = None
        self.right_controller_pose = None
        self.lock = threading.Lock() # One lock for both controller data updates

        # --- Setup Publishers ---
        self.left_pub = self.create_publisher(PoseStamped, '/motion_targetsss/target_pose_arm_left', 10)
        self.right_pub = self.create_publisher(PoseStamped, '/motion_targetsss/target_pose_arm_right', 10)

        # --- Setup Subscribers for Mode Control ---
        self.left_mode_sub = self.create_subscription(
            Bool, '/teleop/left_tele_mode', self.left_mode_callback, 10)
        self.right_mode_sub = self.create_subscription(
            Bool, '/teleop/right_tele_mode', self.right_mode_callback, 10)

        # --- Setup Subscribers for Controller Poses ---
        self.left_controller_sub = self.create_subscription(
            PoseStamped, '/teleop/left_controller_pose', self.left_controller_callback, 10)
        self.right_controller_sub = self.create_subscription(
            PoseStamped, '/teleop/right_controller_pose', self.right_controller_callback, 10)

        self.get_logger().info("✅ Dual Arm Teleop Node is running.")
        self.get_logger().info("Waiting for mode topics to enable/disable arms...")

    # --- Callbacks for Mode Control ---
    def left_mode_callback(self, msg):
        """Controls the left arm's teleop state."""
        if msg.data != self.left_tele_enabled:
            self.left_tele_enabled = msg.data
            status = "ENABLED" if self.left_tele_enabled else "DISABLED"
            self.get_logger().info(f"Left Arm Teleop is now {status}")

    def right_mode_callback(self, msg):
        """Controls the right arm's teleop state."""
        if msg.data != self.right_tele_enabled:
            self.right_tele_enabled = msg.data
            status = "ENABLED" if self.right_tele_enabled else "DISABLED"
            self.get_logger().info(f"Right Arm Teleop is now {status}")

    # --- Callbacks for Controller Data ---
    def left_controller_callback(self, msg):
        """Stores the latest pose from the left controller."""
        with self.lock:
            self.left_controller_pose = msg

    def right_controller_callback(self, msg):
        """Stores the latest pose from the right controller."""
        with self.lock:
            self.right_controller_pose = msg

    # --- NEW: Safety Check Method ---
    def is_pose_safe(self, target_pose):
        """
        Checks if the target pose is outside the cylindrical safety shield.
        Returns:
            bool: True if safe, False if inside the shield.
        """
        x = target_pose.pose.position.x
        y = target_pose.pose.position.y
        z = target_pose.pose.position.z

        # Calculate horizontal distance from the center (z-axis)
        distance_from_center = np.sqrt(x**2 + y**2)

        # A pose is UNSAFE if it's within the cylinder's radius AND height
        if (distance_from_center < 0.14 and
            -0.1 <= z <= 0.5):
            return False  # Pose is inside the shield (unsafe)

        return True # Pose is outside the shield (safe)

    # --- Main Execution Method ---
    def run(self):
        """The main loop checks both arm flags and publishes commands."""
        while rclpy.ok():
            # --- Process Left Arm ---
            if self.left_tele_enabled:
                with self.lock:
                    left_pose_data = self.left_controller_pose

                if left_pose_data:
                    # Create and publish the pose for the left arm
                    base_pt_left = np.array(
                        [left_pose_data.pose.position.x,
                         left_pose_data.pose.position.y,
                         left_pose_data.pose.position.z]
                    )

                    base_pt_left[2] -= 0.2  # Example offset adjustment

                    q_base_ee_left = np.array([
                        left_pose_data.pose.orientation.x,
                        left_pose_data.pose.orientation.y,
                        left_pose_data.pose.orientation.z,
                        left_pose_data.pose.orientation.w
                    ])

                    left_target_pose = PoseStamped()
                    left_target_pose.header.stamp = self.get_clock().now().to_msg()
                    left_target_pose.header.frame_id = 'base_link'
                    left_target_pose.pose.position.x = base_pt_left[0]
                    left_target_pose.pose.position.y = base_pt_left[1]
                    left_target_pose.pose.position.z = base_pt_left[2]
                    left_target_pose.pose.orientation.x = q_base_ee_left[0]
                    left_target_pose.pose.orientation.y = q_base_ee_left[1]
                    left_target_pose.pose.orientation.z = q_base_ee_left[2]
                    left_target_pose.pose.orientation.w = q_base_ee_left[3]

                    # --- MODIFIED: Check safety before publishing ---
                    if self.is_pose_safe(left_target_pose):
                        self.left_pub.publish(left_target_pose)
                        self.get_logger().debug(f"Published left arm pose.")
                    else:
                        self.get_logger().warning("LEFT arm target inside safety shield! Command blocked.", throttle_duration_sec=1)

                else:
                    self.get_logger().warning("Left arm enabled, but no teleop data received.", throttle_duration_sec=2)

            # --- Process Right Arm ---
            if self.right_tele_enabled:
                with self.lock:
                    right_pose_data = self.right_controller_pose

                if right_pose_data:
                    # Create and publish the pose for the right arm
                    base_pt_right = np.array(
                        [right_pose_data.pose.position.x,
                         right_pose_data.pose.position.y,
                         right_pose_data.pose.position.z]
                    )
                    
                    base_pt_right[2] -= 0.2  # Example offset adjustment
                    
                    q_base_ee_right = np.array([
                        right_pose_data.pose.orientation.x,
                        right_pose_data.pose.orientation.y,
                        right_pose_data.pose.orientation.z,
                        right_pose_data.pose.orientation.w
                    ])

                    right_target_pose = PoseStamped()
                    right_target_pose.header.stamp = self.get_clock().now().to_msg()
                    right_target_pose.header.frame_id = 'base_link'
                    right_target_pose.pose.position.x = base_pt_right[0]
                    right_target_pose.pose.position.y = base_pt_right[1]
                    right_target_pose.pose.position.z = base_pt_right[2]
                    right_target_pose.pose.orientation.x = q_base_ee_right[0]
                    right_target_pose.pose.orientation.y = q_base_ee_right[1]
                    right_target_pose.pose.orientation.z = q_base_ee_right[2]
                    right_target_pose.pose.orientation.w = q_base_ee_right[3]

                    self.right_pub.publish(right_target_pose)
                    self.get_logger().debug(f"Published right arm pose: {right_target_pose}")
                else:
                    self.get_logger().warning("Right arm enabled, but no teleop data received.", throttle_duration_sec=2)

            # Process callbacks and sleep to maintain loop rate
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.01) # Loop at ~50Hz


def main():
    rclpy.init()
    node = DualArmTeleop()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()