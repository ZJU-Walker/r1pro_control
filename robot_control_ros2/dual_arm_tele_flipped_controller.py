#!/usr/bin/env python3
'''
ROS2 script for dual-arm teleoperation using a responsive, executor-based design.
- Listens to '/teleop/left_tele_mode' to control the right arm.
- Listens to '/teleop/right_tele_mode' to control the left arm.
'''

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
import threading
import numpy as np

class DualArmTeleop(Node):
    def __init__(self):
        super().__init__('dual_arm_teleop_node')

        # --- State Flags and Lock for Thread Safety ---
        self.left_tele_enabled = False
        self.right_tele_enabled = False
        self.lock = threading.Lock() # Protects access to the enable flags

        # --- Define a QoS profile for real-time data ---
        realtime_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

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
            PoseStamped, '/teleop/left_ee_raw_pose', self.process_left_controller_pose, realtime_qos)
        self.right_controller_sub = self.create_subscription(
            PoseStamped, '/teleop/right_ee_raw_pose', self.process_right_controller_pose, realtime_qos)

        self.get_logger().info("✅ Dual Arm Teleop Node is running (Executor Mode).")
        self.get_logger().info("Publish True/False to '/teleop/...' topics to enable/disable arms.")

        # --- Add these lines ---
        self.last_published_left_pose = None
        self.last_published_right_pose = None
        
        # Safety thresholds
        self.MAX_POSE_JUMP_M = 0.2  # 20 centimeters
        self.MAX_POSE_JUMP_RAD = np.deg2rad(45) # 35 degrees
        # --- End of new lines ---


    def left_mode_callback(self, msg):
        """Controls the LEFT arm's teleop state."""
        with self.lock: # Safely update the shared flag
            if msg.data != self.left_tele_enabled:
                self.left_tele_enabled = msg.data
                status = "ENABLED" if self.left_tele_enabled else "DISABLED"
                self.get_logger().info(f"Left Arm Teleop is now {status}")

    def right_mode_callback(self, msg):
        """Controls the RIGHT arm's teleop state."""
        with self.lock: # Safely update the shared flag
            if msg.data != self.right_tele_enabled:
                self.right_tele_enabled = msg.data
                status = "ENABLED" if self.right_tele_enabled else "DISABLED"
                self.get_logger().info(f"Right Arm Teleop is now {status}")

    def process_right_controller_pose(self, msg: PoseStamped):
        """Processes RIGHT controller data to control the LEFT arm."""
        with self.lock: # Safely read the shared flag
            is_enabled = self.right_tele_enabled

        if not is_enabled:
            return

        target_pose = PoseStamped()
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.header.frame_id = 'base_link'
        target_pose.pose = msg.pose
        target_pose.pose.position.x += 0.15
        target_pose.pose.position.y -= 0.05
        target_pose.pose.position.z += 0.4

        if self.is_pose_safe(target_pose, self.last_published_right_pose, "RIGHT"):
            self.right_pub.publish(target_pose)  
            self.last_published_right_pose = target_pose 
        else:
            self.get_logger().warning("RIGHT arm target inside safety shield! Command blocked.", throttle_duration_sec=1)

    def process_left_controller_pose(self, msg: PoseStamped):
        """Processes LEFT controller data to control the RIGHT arm."""
        with self.lock: # Safely read the shared flag
            is_enabled = self.left_tele_enabled

        if not is_enabled:
            return

        target_pose = PoseStamped()
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.header.frame_id = 'base_link'
        target_pose.pose = msg.pose
        target_pose.pose.position.x += 0.15
        target_pose.pose.position.y += 0.05
        target_pose.pose.position.z += 0.4

        # Call the updated safety check for the LEFT arm
        if self.is_pose_safe(target_pose, self.last_published_left_pose, "LEFT"):
            self.left_pub.publish(target_pose)  
            self.last_published_left_pose = target_pose 
        else:
            self.get_logger().warning("LEFT arm target inside safety shield! Command blocked.", throttle_duration_sec=1)

    def is_pose_safe(self, target_pose: PoseStamped, current_pose: PoseStamped, arm_name: str) -> bool:
        """
        Checks if the target pose is safe by checking position and orientation jumps.
        """
        # If this is the first command for this arm, we can't compare. Assume it's safe.
        if current_pose is None:
            return True

        # --- 1. Position Jump Check ---
        current_pos = current_pose.pose.position
        target_pos = target_pose.pose.position
        distance = np.linalg.norm([
            target_pos.x - current_pos.x,
            target_pos.y - current_pos.y,
            target_pos.z - current_pos.z
        ])

        if distance > self.MAX_POSE_JUMP_M:
            self.get_logger().fatal(
                f"🚨 SAFETY VIOLATION on {arm_name} arm! "
                f"Position jump of {distance:.3f}m exceeds the limit of {self.MAX_POSE_JUMP_M}m."
            )
            self.get_logger().info("Shutting down node.")
            rclpy.shutdown()
            return False

        # --- 2. Orientation Jump Check ---
        q_current = current_pose.pose.orientation
        q_target = target_pose.pose.orientation
        
        q1 = np.array([q_current.x, q_current.y, q_current.z, q_current.w])
        q2 = np.array([q_target.x, q_target.y, q_target.z, q_target.w])
        
        dot_product = np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0)
        angle_rad = 2 * np.arccos(dot_product)
        
        if angle_rad > self.MAX_POSE_JUMP_RAD:
            self.get_logger().fatal(
                f"🚨 SAFETY VIOLATION on {arm_name} arm! "
                f"Orientation jump of {np.rad2deg(angle_rad):.1f}° exceeds the limit."
            )
            self.get_logger().info("Shutting down node.")
            rclpy.shutdown()
            return False

        return True

def main():
    rclpy.init()
    node = DualArmTeleop()
    
    # Create a MultiThreadedExecutor to handle callbacks in background threads
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()