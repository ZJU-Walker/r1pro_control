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
from std_msgs.msg import Bool, Float64
from sensor_msgs.msg import JointState
import threading
import numpy as np

class DualArmTeleop(Node):
    def __init__(self):
        super().__init__('dual_arm_teleop_node')

        # --- State Flags and Lock for Thread Safety ---
        self.left_tele_enabled = False
        self.right_tele_enabled = False
        self.torso_tele_enabled = False
        self.lock = threading.Lock() # Protects access to the enable flags

        # --- Torso State ---
        self.current_torso_joints = [0.0] * 4
        self.torso_feedback_received = False
        self.latest_torso_joint3_teleop = 0.0  # Latest teleop value for joint 3
        self.latest_torso_joint4_teleop = 0.0  # Latest teleop value for joint 4
        self.torso_joint3_teleop_offset = 0.0  # Teleop value when mode enabled
        self.torso_joint4_teleop_offset = 0.0  # Teleop value when mode enabled

        # --- Define a QoS profile for real-time data ---
        realtime_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- Setup Publishers ---
        self.left_pub = self.create_publisher(PoseStamped, '/motion_target/target_pose_arm_left', 10)
        self.right_pub = self.create_publisher(PoseStamped, '/motion_target/target_pose_arm_right', 10)
        self.torso_pub = self.create_publisher(JointState, '/motion_targetsss/target_joint_state_torso', 10)

        # --- Setup Subscribers for Mode Control ---
        self.left_mode_sub = self.create_subscription(
            Bool, '/teleop/left_tele_mode', self.left_mode_callback, 10)
        self.right_mode_sub = self.create_subscription(
            Bool, '/teleop/right_tele_mode', self.right_mode_callback, 10)
        self.torso_mode_sub = self.create_subscription(
            Bool, '/teleop/torso_tele_mode', self.torso_mode_callback, 10)

        # --- Setup Subscribers for Controller Poses ---
        self.left_controller_sub = self.create_subscription(
            PoseStamped, '/teleop/left_ee_raw_pose', self.process_left_controller_pose, realtime_qos)
        self.right_controller_sub = self.create_subscription(
            PoseStamped, '/teleop/right_ee_raw_pose', self.process_right_controller_pose, realtime_qos)

        # --- Setup Subscribers for Torso ---
        self.torso_feedback_sub = self.create_subscription(
            JointState, '/hdas/feedback_torso', self.torso_feedback_callback, 10)
        # Joint 3: only stores value
        self.torso_joint3_sub = self.create_subscription(
            Float64, '/teleop/torso_joint_3', self.on_torso_joint3_received, realtime_qos)
        # Joint 4: stores value AND triggers publish for both joints
        self.torso_joint4_sub = self.create_subscription(
            Float64, '/teleop/torso_joint_4', self.on_torso_joint4_received, realtime_qos)

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

    def torso_mode_callback(self, msg):
        """Controls the TORSO teleop state."""
        with self.lock:
            if msg.data != self.torso_tele_enabled:
                self.torso_tele_enabled = msg.data
                if self.torso_tele_enabled:
                    # Capture current teleop values as the zero reference
                    self.torso_joint3_teleop_offset = self.latest_torso_joint3_teleop
                    self.torso_joint4_teleop_offset = self.latest_torso_joint4_teleop
                    self.get_logger().info(f"Torso teleop offsets set: joint3={self.torso_joint3_teleop_offset:.4f}, joint4={self.torso_joint4_teleop_offset:.4f}")
                status = "ENABLED" if self.torso_tele_enabled else "DISABLED"
                self.get_logger().info(f"Torso Teleop is now {status}")

    def torso_feedback_callback(self, msg):
        """Updates current torso joint positions from feedback."""
        if len(msg.position) >= 4:
            with self.lock:
                self.current_torso_joints = list(msg.position)
                self.torso_feedback_received = True

    def on_torso_joint3_received(self, msg: Float64):
        """Stores the latest joint 3 teleop value (no publish)."""
        self.latest_torso_joint3_teleop = msg.data

    def on_torso_joint4_received(self, msg: Float64):
        """Stores joint 4 value and publishes both joints."""
        self.latest_torso_joint4_teleop = msg.data
        self.publish_torso()

    def publish_torso(self):
        """Publishes torso command with both joint 3 and 4."""
        with self.lock:
            is_enabled = self.torso_tele_enabled
            has_feedback = self.torso_feedback_received
            current_joints = self.current_torso_joints.copy()
            joint3_offset = self.torso_joint3_teleop_offset
            joint4_offset = self.torso_joint4_teleop_offset
            joint3_teleop = self.latest_torso_joint3_teleop
            joint4_teleop = self.latest_torso_joint4_teleop

        if not is_enabled or not has_feedback:
            return

        # Joint 3: with -0.32 as the zero position
        JOINT3_ZERO_POSITION = -0.32
        target_joint3 = (joint3_teleop - joint3_offset) + JOINT3_ZERO_POSITION
        target_joint3 *= 1.3
        target_joint3 = np.clip(target_joint3, -0.7, -0.25)

        # Joint 4: zero position is 0
        target_joint4 = joint4_teleop - joint4_offset
        target_joint4 = np.clip(target_joint4, -0.55, 0.55)

        torso_msg = JointState()
        torso_msg.header.stamp = self.get_clock().now().to_msg()
        torso_msg.position = current_joints[:2] + [target_joint3, target_joint4]
        torso_msg.velocity = [1.5] * 4
        self.torso_pub.publish(torso_msg)

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
        target_pose.pose.position.y -= 0.0
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
        target_pose.pose.position.x += 0.2
        target_pose.pose.position.y += 0.0
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