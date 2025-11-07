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
from scipy.spatial.transform import Rotation as R

# Helper functions 
def pose_to_T(pose):
    """Convert a geometry_msgs/Pose to a 4x4 transformation matrix."""
    T = np.eye(4, dtype=float)
    p = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
    q = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w], dtype=float)
    R_mat = R.from_quat(q).as_matrix() # xyzw to rotation matrix
    T[0:3, 0:3] = R_mat
    T[0:3, 3] = p
    return T

def T_to_pose(T, frame_id, stamp, hand):
    """Convert a 4x4 transformation matrix to a geometry_msgs/PoseStamped."""
    R_mat = T[:3, :3]
    t = T[:3, 3]
    q = R.from_matrix(R_mat).as_quat() # xyzw
    if hand == 'left':   
        t[0] += 0.25
        t[1] += 0.1
        t[2] += 0.4
    elif hand == 'right':
        t[0] += 0.25
        t[1] -= 0.1
        t[2] += 0.4

    # keep sign consistent
    q = q if q[3] >= 0 else -q

    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.header.stamp = stamp
    pose.pose.position.x = t[0]
    pose.pose.position.y = t[1]
    pose.pose.position.z = t[2]
    pose.pose.orientation.x = q[0]
    pose.pose.orientation.y = q[1]
    pose.pose.orientation.z = q[2]
    pose.pose.orientation.w = q[3]

    return pose

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

        # --- Transformations placeholders ---
        theta = np.radians(180)  # 180 degrees rotation around Z axis
        Rz_180 = np.array([[np.cos(theta), -np.sin(theta), 0],
                           [np.sin(theta), np.cos(theta), 0],
                           [0, 0, 1]])
        self.T_leftctrlmirrored_to_leftee = np.eye(4)  # To be defined/calibrated
        self.T_leftctrlmirrored_to_leftee[:3, :3] = Rz_180
        self.T_rightctrlmirrored_to_rightee = np.eye(4)  # To be defined/calibrated
        self.T_rightctrlmirrored_to_rightee[:3, :3] = Rz_180
        self.M_H = np.array([  # Mirroring matrix across the XZ plane
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # --- Setup Publishers ---
        self.left_pub = self.create_publisher(PoseStamped, '/motion_target/target_pose_arm_left', 10)
        self.right_pub = self.create_publisher(PoseStamped, '/motion_target/target_pose_arm_right', 10)

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
                    # Mirror transformation logic here
                    # convert posestamp to T
                    T_base_to_leftctrl = pose_to_T(left_pose_data.pose)
                    T_base_to_leftctrl_mirrored = self.M_H @ T_base_to_leftctrl @ self.M_H
                    T_base_to_leftee = T_base_to_leftctrl_mirrored @ self.T_leftctrlmirrored_to_leftee

                    left_target_pose = T_to_pose(T_base_to_leftee, 'base_link', self.get_clock().now().to_msg(), 'left')

                    # --- Check safety before publishing ---
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
                    # Mirror transformation logic here
                    T_base_to_rightctrl = pose_to_T(right_pose_data.pose)
                    T_base_to_rightctrl_mirrored = self.M_H @ T_base_to_rightctrl @ self.M_H
                    T_base_to_rightee = T_base_to_rightctrl_mirrored @ self.T_rightctrlmirrored_to_rightee
                    right_target_pose = T_to_pose(T_base_to_rightee, 'base_link', self.get_clock().now().to_msg(), 'right')

                    # --- Check safety before publishing ---
                    if self.is_pose_safe(right_target_pose):
                        self.right_pub.publish(right_target_pose)
                        self.get_logger().debug(f"Published right arm pose: {right_target_pose}")
                    else:
                        self.get_logger().warning("RIGHT arm target inside safety shield! Command blocked.", throttle_duration_sec=1)

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