#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import pandas as pd
import numpy as np
import os
import time

# For keyboard input on Linux/macOS
import sys
import select
import termios
import tty

# NEW: Import Slerp and Rotation for accurate orientation interpolation
from scipy.spatial.transform import Slerp, Rotation

from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# Define the expected joint order for the right hand
EXPECTED_RIGHT_JOINT_ORDER = [
    "rj_dg_1_1", "rj_dg_1_2", "rj_dg_1_3", "rj_dg_1_4",
    "rj_dg_2_1", "rj_dg_2_2", "rj_dg_2_3", "rj_dg_2_4",
    "rj_dg_3_1", "rj_dg_3_2", "rj_dg_3_3", "rj_dg_3_4",
    "rj_dg_4_1", "rj_dg_4_2", "rj_dg_4_3", "rj_dg_4_4",
    "rj_dg_5_1", "rj_dg_5_2", "rj_dg_5_3", "rj_dg_5_4"
]

class KeyboardHandler:
    """A helper class to get non-blocking keyboard input."""
    def __init__(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    def get_key(self, timeout=0.01):
        if select.select([sys.stdin], [], [], timeout)[0]:
            return sys.stdin.read(1)
        return None

    def restore(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


class TrajectoryPublisherNode(Node):
    """
    A ROS2 node that reads a trajectory from a CSV file and publishes it interactively,
    moving by a constant distance and angle with each step.
    """
    def __init__(self):
        super().__init__('trajectory_publisher_node')
        
        # --- NEW PARAMETERS for constant step size ---
        self.declare_parameter('csv_path', '/home/irislab/r1pro_control/robot_control_ros2/robot_commands_human.csv')
        self.declare_parameter('rate_hz', 10.0) # Increased rate for smoother auto-mode
        self.declare_parameter('step_distance_m', 0.01)  # Target distance per step
        self.declare_parameter('step_angle_deg', 2.0)   # Target angle per step

        # Get parameters
        csv_path = self.get_parameter('csv_path').get_parameter_value().string_value
        rate_hz = self.get_parameter('rate_hz').get_parameter_value().double_value
        self.step_dist = self.get_parameter('step_distance_m').get_parameter_value().double_value
        self.step_angle = np.deg2rad(self.get_parameter('step_angle_deg').get_parameter_value().double_value) # Convert to radians

        # Publishers
        self.wrist_pose_pub = self.create_publisher(PoseStamped, '/motion_target/target_pose_arm_right', 1)
        self.hand_joints_pub = self.create_publisher(JointTrajectory, '/dg5f_right_controller/joint_trajectory', 10)
        
        try:
            self.trajectory_data = pd.read_csv(csv_path)
            self.get_logger().info(f"Successfully loaded {len(self.trajectory_data)} points from {csv_path}")
            self.get_logger().info(f"Step size: {self.step_dist:.3f} m, {self.get_parameter('step_angle_deg').get_parameter_value().double_value:.1f} deg")
        except FileNotFoundError:
            self.get_logger().error(f"Could not find CSV file at {csv_path}")
            rclpy.shutdown()
            return
            
        # State tracking
        self.current_index = 0
        self.interpolation_step = 0
        self.auto_mode = False

        # --- NEW state variables for dynamic calculation ---
        self.total_segment_steps = 0
        self.slerp = None
        self.start_pos = None
        self.end_pos = None
        self.start_joints = None
        self.end_joints = None
        
        # Timer for continuous mode
        self.timer_period = 1.0 / rate_hz
        self.timer = self.create_timer(self.timer_period, self.auto_mode_callback)

    def auto_mode_callback(self):
        if self.auto_mode: self.publish_step()

    def execute_single_step(self):
        if not self.auto_mode: self.publish_step()
            
    def toggle_auto_mode(self):
        self.auto_mode = not self.auto_mode
        self.get_logger().info(f"Continuous mode is now {'ON' if self.auto_mode else 'OFF'}")

    def _update_segment_parameters(self):
        """Calculate the total number of steps for the current segment."""
        if self.current_index == 0:
            start_step = self.trajectory_data.iloc[0]
            end_step = self.trajectory_data.iloc[0]
        else:
            start_step = self.trajectory_data.iloc[self.current_index - 1]
            end_step = self.trajectory_data.iloc[self.current_index]

        # --- Store start/end points for interpolation ---
        self.start_pos = np.array([start_step['right_wrist_x'], start_step['right_wrist_y'], start_step['right_wrist_z']])
        self.end_pos = np.array([end_step['right_wrist_x'], end_step['right_wrist_y'], end_step['right_wrist_z']])
        self.start_joints = np.array([start_step[f'right_hand_{i}'] for i in range(20)])
        self.end_joints = np.array([end_step[f'right_hand_{i}'] for i in range(20)])

        # --- Calculate total distance and angle for the segment ---
        total_dist = np.linalg.norm(self.end_pos - self.start_pos)
        
        # Use Scipy Rotation for robust angle calculation
        start_quat = np.array([start_step['right_wrist_qx'], start_step['right_wrist_qy'], start_step['right_wrist_qz'], start_step['right_wrist_qw']])
        end_quat = np.array([end_step['right_wrist_qx'], end_step['right_wrist_qy'], end_step['right_wrist_qz'], end_step['right_wrist_qw']])
        
        # Ensure shortest path for quaternion interpolation
        if np.dot(start_quat, end_quat) < 0.0:
            start_quat = -start_quat
            
        rotations = Rotation.from_quat([start_quat, end_quat])
        total_angle = (rotations[0].inv() * rotations[1]).magnitude()

        # --- Calculate number of steps needed for position and orientation ---
        pos_steps = np.ceil(total_dist / self.step_dist) if self.step_dist > 0 else 0
        rot_steps = np.ceil(total_angle / self.step_angle) if self.step_angle > 0 else 0
        
        # The total steps for the segment is the max of the two
        self.total_segment_steps = int(max(1, pos_steps, rot_steps))
        
        # Create the SLERP interpolator for this segment
        self.slerp = Slerp([0, 1], rotations)
        
        self.get_logger().info(f"New segment ({self.current_index-1}->{self.current_index}): Dist: {total_dist:.3f}m, Angle: {np.rad2deg(total_angle):.1f}deg. Requires {self.total_segment_steps} steps.")

    def publish_step(self):
        """The core logic to publish one interpolated step of the trajectory."""
        if self.current_index >= len(self.trajectory_data):
            self.get_logger().info("End of trajectory reached.")
            if self.auto_mode: self.auto_mode = False
            return
        
        # --- If at the start of a new segment, calculate its parameters ---
        if self.interpolation_step == 0:
            self._update_segment_parameters()

        # --- Interpolate based on the current step ---
        alpha = (self.interpolation_step + 1) / self.total_segment_steps
        alpha = np.clip(alpha, 0.0, 1.0)

        # Linear interpolation for position and joints
        interp_pos = self.start_pos + alpha * (self.end_pos - self.start_pos)
        interp_joints = self.start_joints + alpha * (self.end_joints - self.start_joints)

        # Spherical linear interpolation (SLERP) for orientation
        interp_rot = self.slerp([alpha])[0]
        interp_quat = interp_rot.as_quat() # Returns [x, y, z, w]

        # --- Publish Messages ---
        now = self.get_clock().now().to_msg()
        
        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = "base_link"
        pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = interp_pos
        pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = interp_quat
        self.wrist_pose_pub.publish(pose_msg)

        joint_traj_msg = JointTrajectory()
        joint_traj_msg.header.stamp = now
        joint_traj_msg.joint_names = EXPECTED_RIGHT_JOINT_ORDER
        point = JointTrajectoryPoint(positions=interp_joints.tolist())
        point.time_from_start = Duration(sec=0, nanosec=int(self.timer_period * 1e9))
        joint_traj_msg.points.append(point)
        self.hand_joints_pub.publish(joint_traj_msg)

        self.get_logger().info(f"Frame {self.current_index+1}/{len(self.trajectory_data)}, Sub-step {self.interpolation_step + 1}/{self.total_segment_steps}")

        # --- Update State for Next Step ---
        self.interpolation_step += 1
        if self.interpolation_step >= self.total_segment_steps:
            self.interpolation_step = 0
            if len(self.trajectory_data) > 1:
                self.current_index += 1

def main(args=None):
    rclpy.init(args=args)
    # Check for scipy
    try:
        from scipy.spatial.transform import Slerp
    except ImportError:
        print("\nERROR: 'scipy' is not installed. Please run 'pip install scipy' and try again.\n")
        return

    node = TrajectoryPublisherNode()
    kb = KeyboardHandler()

    print("\n" + "="*50)
    print("Interactive Trajectory Publisher (Constant Step)")
    print("="*50)
    print("  'n' -> Execute next micro-step")
    print("  'c' -> Toggle continuous playback mode")
    print("  'q' -> Quit")
    print("="*50 + "\n")

    try:
        while rclpy.ok():
            key = kb.get_key()
            if key == 'q': print("Quitting."); break
            elif key == 'n': node.execute_single_step()
            elif key == 'c': node.toggle_auto_mode()
            
            rclpy.spin_once(node, timeout_sec=0.01)

    except KeyboardInterrupt: pass
    finally:
        kb.restore()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()