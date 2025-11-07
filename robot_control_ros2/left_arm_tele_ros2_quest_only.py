#!/usr/bin/env python3
'''
ROS2 version of trajectory following script for LEFT ARM.
Reorganized with separate run methods for each mode.
Hand/serial control removed.

The input trajectory is wrist position and orientation in camera frame.
Transformation pipeline: Camera Frame → Base Frame → EE Frame
- First: Transform camera frame to robot base frame (position + orientation)
- Then: Transform hand frame to end-effector frame (orientation only, same origin)

Usage: press 's' to start the program, 'f' to stop it.
Modes:
- normal: Full trajectory playback from CSV
- move_step: Incremental movement toward trajectory points
- translation: Position-only movement with fixed orientation
- translation_smooth: Automatic smooth translation with fixed orientation
- ros_tele: Real-time teleop control via ROS topics
- ori: Test mode for setting fixed position/orientation without movement
- print: Print current pose without sending commands
'''

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import threading
import os
import csv
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import sys
import select
import termios
import tty

class TrajFollow(Node):
    def __init__(self, mode='normal'):
        super().__init__('traj_follow_left_node')
        self.mode = mode
        self.lock = threading.Lock()
        self.current_pose = None
        self.message_received = False
        
        # Program control flags
        self.program_running = False
        self.program_stopped = False
        self.keyboard_thread = None
        
        # Arm configuration - LEFT ARM
        self.arm = 'left'
        self.pose_topic = f'/motion_control/pose_ee_arm_{self.arm}'
        # Alternative: self.pose_topic = f'/relaxed_ik/motion_control/pose_ee_arm_{self.arm}'
        self.target_topic = f'/motion_target/target_pose_arm_{self.arm}'
        
        # Create publishers and subscribers
        self.create_core_publishers_subscribers()
        
        # Mode-specific initialization
        self.initialize_mode_specific()
        
        # Transformation matrices (same as original)
        self.setup_transformations()
        
        # Move step mode variables
        self.setup_move_step_variables()
        
        self.get_logger().info(f"TrajFollow node initialized in {mode.upper()} mode for LEFT arm")
        
        # Start keyboard listener thread
        self.start_keyboard_listener()
    
    def create_core_publishers_subscribers(self):
        """Create core publishers and subscribers needed for all modes"""
        # Core subscriber for current pose
        self.subscription = self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.pose_callback,
            10)
        
        # Core publishers
        self.pub = self.create_publisher(PoseStamped, self.target_topic, 10)
        self.left_joint_pub = self.create_publisher(
            JointState, 
            '/motion_target/target_joint_state_arm_left', 
            10)
        self.right_joint_pub = self.create_publisher(
            JointState,
            '/motion_target/target_joint_state_arm_right',
            10)
    
    def initialize_mode_specific(self):
        """Initialize mode-specific variables and subscribers"""
        if self.mode == 'ros_tele':
            self.teleop_wrist_data = None
            self.teleop_lock = threading.Lock()
            
            # Subscribe to LEFT wrist teleop topic
            self.teleop_sub = self.create_subscription(
                PoseStamped,
                '/teleop/left_wrist_pos',
                self.teleop_wrist_callback,
                10)
            self.get_logger().info("[ROS_TELE] Subscribed to /teleop/left_wrist_pos")
        
        elif self.mode == 'ori':
            self.fixed_position = None
        
        elif self.mode in ['translation', 'translation_smooth']:
            self.fixed_orientation = None
    
    def setup_transformations(self):
        """Setup transformation matrices for camera->base and hand->ee"""
        # Zed Camera (looking ~30 degree downwards) -> base transform
        self.T_cam_to_base = np.array([
            [ 0.05787445, -0.7673487,   0.63861296,  0.13727924],
            [-0.99832064, -0.04611135,  0.03506627,  0.01021076],
            [ 0.00253925, -0.63956994, -0.76872872,  0.50120488],
            [ 0., 0., 0., 1.]
        ])
        
        # Extract rotation matrix and convert to quaternion
        R_cam_to_base = self.T_cam_to_base[:3, :3]
        self.q_cam_to_base = R.from_matrix(R_cam_to_base).as_quat()
        
        # Hand to end-effector frame transformation
        # 1. Rotate 180° around hand Y-axis
        self.R_hand_y_180 = np.array([
            [-1,  0,  0],
            [ 0,  1,  0],
            [ 0,  0, -1]
        ])
        # 2. Rotate 90° around hand Z-axis
        self.R_hand_z_90 = np.array([
            [ 0, -1,  0],
            [ 1,  0,  0],
            [ 0,  0,  1]
        ])
        # Combined transformation
        self.R_hand_to_ee = np.dot(self.R_hand_y_180, self.R_hand_z_90)
        self.R_ee_last = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
        self.R_hand_to_ee = np.dot(self.R_hand_to_ee, self.R_ee_last)
        # Convert to quaternion
        self.q_hand_to_ee = R.from_matrix(self.R_hand_to_ee).as_quat()
    
    def start_keyboard_listener(self):
        """Start a thread to listen for keyboard input"""
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_thread.start()
    
    def keyboard_listener(self):
        """Listen for 's' to start and 'f' to stop the program"""
        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            # Set terminal to raw mode for immediate key detection
            tty.setraw(sys.stdin.fileno())
            
            while not self.program_stopped:
                # Check if input is available
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1).lower()
                    
                    if key == 's' and not self.program_running:
                        self.program_running = True
                        print(f"[CONTROL] Program STARTED - press 'f' to stop")
                    elif key == 'f':
                        self.program_running = False
                        self.program_stopped = True
                        print(f"[CONTROL] Program STOPPED")
                        break
                    elif key == '\x03':  # Ctrl+C
                        self.program_running = False
                        self.program_stopped = True
                        break
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    def wait_for_start(self):
        """Wait for user to press 's' to start the program"""
        self.get_logger().info("[CONTROL] Press 's' to START the program, 'f' to STOP at any time")
        
        while not self.program_running and not self.program_stopped:
            time.sleep(0.1)
            rclpy.spin_once(self, timeout_sec=0.01)
        
        return not self.program_stopped
    
    def setup_move_step_variables(self):
        """Setup variables for move_step and translation modes"""
        self.trajectory_points = []
        self.current_target_idx = 0
        self.step_size = 0.025  # Step size in meters
        self.position_threshold = 0.01  # Distance threshold to consider target reached
        self.min_movement_threshold = 0.025  # Minimum movement to send to robot
    
    # ============= Callbacks =============
    def pose_callback(self, msg):
        """Callback for current robot pose"""
        with self.lock:
            self.current_pose = msg
            self.message_received = True
    
    def teleop_wrist_callback(self, msg):
        """Callback for teleop wrist position data"""
        with self.teleop_lock:
            self.teleop_wrist_data = msg
    
    # ============= Utility Methods =============
    def wait_for_first_message(self, timeout=10.0):
        """Wait for first pose message with timeout"""
        self.get_logger().info(f"Waiting for EE pose of {self.arm} arm...")
        start_time = time.time()
        
        while not self.message_received and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.message_received:
                break
        
        if not self.message_received:
            self.get_logger().warning("Timeout waiting for first message")
            return False
        
        self.get_logger().info("EE pose received.")
        return True
    
    def wait_for_current_pose(self, timeout=10.0):
        """Wait for current pose and optionally set fixed position/orientation"""
        if self.mode == 'ori':
            self.get_logger().info("[TEST MODE: ori] Waiting for current robot pose...")
        elif self.mode in ['translation', 'translation_smooth']:
            self.get_logger().info(f"[TEST MODE: {self.mode}] Waiting for current robot pose...")
        
        start_time = time.time()
        
        while rclpy.ok():
            with self.lock:
                if self.current_pose is not None:
                    if self.mode == 'ori':
                        self.fixed_position = [
                            self.current_pose.pose.position.x,
                            self.current_pose.pose.position.y,
                            self.current_pose.pose.position.z
                        ]
                        self.get_logger().info(f"[ORI MODE] Fixed position set to: {self.fixed_position}")
                    elif self.mode in ['translation', 'translation_smooth']:
                        self.fixed_orientation = [
                            self.current_pose.pose.orientation.x,
                            self.current_pose.pose.orientation.y,
                            self.current_pose.pose.orientation.z,
                            self.current_pose.pose.orientation.w
                        ]
                        self.get_logger().info(f"[{self.mode.upper()}] Fixed orientation set")
                    return True
            
            if (time.time() - start_time) > timeout:
                self.get_logger().error("Timeout waiting for current pose")
                return False
            
            time.sleep(0.1)
        return False
    
    def send_vel_limit(self, vel_left, vel_right):
        """Send velocity limits to both arms"""
        left_msg = JointState()
        left_msg.velocity = vel_left
        right_msg = JointState()
        right_msg.velocity = vel_right
        
        self.left_joint_pub.publish(left_msg)
        self.right_joint_pub.publish(right_msg)
    
    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions [x, y, z, w]"""
        w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
        w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return [x, y, z, w]
    
    def load_trajectory_points(self, filepath):
        """Load trajectory points from CSV and transform to base frame"""
        trajectory_points = []
        
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Parse position (camera frame)
                cam_pt = np.array([
                    float(row['wrist_x']),
                    float(row['wrist_y']),
                    float(row['wrist_z']),
                    1.0
                ])
                # Transform to base frame
                base_pt = self.T_cam_to_base.dot(cam_pt)
                
                # Parse orientation (camera frame quaternion)
                q_cam = np.array([
                    float(row['qx']),
                    float(row['qy']),
                    float(row['qz']),
                    float(row['qw'])
                ])
                
                # Transform orientation from camera frame to base frame
                q_base_hand = self.quaternion_multiply(self.q_cam_to_base, q_cam)
                # Transform from hand frame to ee frame
                q_base_ee = self.quaternion_multiply(q_base_hand, self.q_hand_to_ee)
                
                # Store transformed point (no hand data)
                point = {
                    'position': [base_pt[0], base_pt[1], base_pt[2]],
                    'orientation': [q_base_ee[0], q_base_ee[1], q_base_ee[2], q_base_ee[3]]
                }
                trajectory_points.append(point)
        
        return trajectory_points
    
    def calculate_step_toward_target(self, current_pos, target_pos, current_ori=None, target_ori=None):
        """Calculate next step position and orientation toward target"""
        # Calculate direction vector for position
        direction = np.array([
            target_pos[0] - current_pos[0],
            target_pos[1] - current_pos[1], 
            target_pos[2] - current_pos[2]
        ])
        
        # Calculate distance to target
        distance = np.linalg.norm(direction)
        
        # Position is considered reached when very close
        position_reached = distance <= self.position_threshold
        
        # Calculate next position
        if distance > 0:
            direction_normalized = direction / distance
            # Always take minimum step size
            step_distance = self.min_movement_threshold
            
            next_pos = [
                current_pos[0] + direction_normalized[0] * step_distance,
                current_pos[1] + direction_normalized[1] * step_distance,
                current_pos[2] + direction_normalized[2] * step_distance
            ]
            
            print(f"[DEBUG] Distance to target: {distance:.6f}m")
            print(f"[DEBUG] Step distance: {step_distance:.6f}m")
        else:
            next_pos = current_pos
        
        # Calculate next orientation using SLERP
        next_ori = current_ori
        orientation_reached = True
        
        if current_ori is not None and target_ori is not None:
            # Convert to numpy arrays
            q_current = np.array(current_ori)
            q_target = np.array(target_ori)
            
            # Calculate quaternion distance
            dot_product = np.dot(q_current, q_target)
            
            # Handle quaternion double cover
            if dot_product < 0:
                q_target = -q_target
                dot_product = -dot_product
            
            # Clamp dot product
            dot_product = np.clip(dot_product, -1.0, 1.0)
            
            # Calculate angular distance
            theta = np.arccos(abs(dot_product))
            
            # Define orientation threshold
            orientation_threshold = 0.1  # ~5.7 degrees
            orientation_reached = theta <= orientation_threshold
            
            # Calculate interpolation
            angular_step_size = 0.05  # ~2.9 degrees per step
            
            if theta > orientation_threshold:
                t = min(angular_step_size / theta, 1.0)
                
                # Perform SLERP
                if theta > 1e-6:
                    sin_theta = np.sin(theta)
                    weight_current = np.sin((1.0 - t) * theta) / sin_theta
                    weight_target = np.sin(t * theta) / sin_theta
                    next_ori = weight_current * q_current + weight_target * q_target
                    # Normalize
                    next_ori = next_ori / np.linalg.norm(next_ori)
                else:
                    next_ori = q_target
            else:
                next_ori = q_target
        
        target_reached = position_reached and orientation_reached
        
        return next_pos, next_ori, target_reached
    
    def execute_move_step(self):
        """Execute one incremental step in move_step mode"""
        if self.current_target_idx >= len(self.trajectory_points):
            print("All trajectory points completed!")
            return False
        
        # Get current robot position and orientation
        with self.lock:
            if self.current_pose is None:
                print("No current pose available!")
                return True
            
            current_pos = [
                self.current_pose.pose.position.x,
                self.current_pose.pose.position.y,
                self.current_pose.pose.position.z
            ]
            
            current_ori = [
                self.current_pose.pose.orientation.x,
                self.current_pose.pose.orientation.y,
                self.current_pose.pose.orientation.z,
                self.current_pose.pose.orientation.w
            ]
        
        # Get current target point
        target_point = self.trajectory_points[self.current_target_idx]
        target_pos = target_point['position']
        target_ori = target_point['orientation']
        
        # Calculate next step
        next_pos, next_ori, target_reached = self.calculate_step_toward_target(
            current_pos, target_pos, current_ori, target_ori)
        
        # Calculate relative movement
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]
        dz = next_pos[2] - current_pos[2]
        
        print(f"[DEBUG] Relative movement: dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}")
        
        # Create and send pose message
        new_pose = PoseStamped()
        new_pose.header.stamp = self.get_clock().now().to_msg()
        new_pose.header.frame_id = "base_link"
        
        new_pose.pose.position.x = current_pos[0] + dx
        new_pose.pose.position.y = current_pos[1] + dy
        new_pose.pose.position.z = current_pos[2] + dz
        
        new_pose.pose.orientation.x = next_ori[0]
        new_pose.pose.orientation.y = next_ori[1]
        new_pose.pose.orientation.z = next_ori[2]
        new_pose.pose.orientation.w = next_ori[3]
        
        self.pub.publish(new_pose)
        time.sleep(0.1)
        
        if target_reached:
            print(f"✓ Point {self.current_target_idx + 1} reached!")
            self.current_target_idx += 1
        else:
            pos_dist = np.linalg.norm([target_pos[0] - current_pos[0],
                                       target_pos[1] - current_pos[1],
                                       target_pos[2] - current_pos[2]])
            print(f"Progress: pos_dist={pos_dist:.4f}m")
        
        return True
    
    def execute_translation_step(self):
        """Execute one translation step with fixed orientation"""
        if self.current_target_idx >= len(self.trajectory_points):
            print("All trajectory points completed!")
            return False
        
        # Get current robot position
        with self.lock:
            if self.current_pose is None:
                print("No current pose available!")
                return True
            
            current_pos = [
                self.current_pose.pose.position.x,
                self.current_pose.pose.position.y,
                self.current_pose.pose.position.z
            ]
        
        # Get target position
        target_point = self.trajectory_points[self.current_target_idx]
        target_pos = target_point['position']
        
        # Calculate next step (position only)
        next_pos, _, target_reached = self.calculate_step_toward_target(
            current_pos, target_pos, None, None)
        
        # Calculate relative movement
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]
        dz = next_pos[2] - current_pos[2]
        
        # Create pose message with fixed orientation
        new_pose = PoseStamped()
        new_pose.header.stamp = self.get_clock().now().to_msg()
        new_pose.header.frame_id = "base_link"
        
        new_pose.pose.position.x = current_pos[0] + dx
        new_pose.pose.position.y = current_pos[1] + dy
        new_pose.pose.position.z = current_pos[2] + dz
        
        # Use fixed orientation
        new_pose.pose.orientation.x = self.fixed_orientation[0]
        new_pose.pose.orientation.y = self.fixed_orientation[1]
        new_pose.pose.orientation.z = self.fixed_orientation[2]
        new_pose.pose.orientation.w = self.fixed_orientation[3]
        
        self.pub.publish(new_pose)
        time.sleep(0.1)
        
        if target_reached:
            print(f"✓ Point {self.current_target_idx + 1} reached (position only)!")
            self.current_target_idx += 1
        
        return True
    
    # ============= Mode Execution Methods =============
    def run_normal_mode(self):
        """Standard CSV trajectory playback"""
        # Wait for start signal
        if not self.wait_for_start():
            return
        
        self.get_logger().info(f"[{self.arm}] Starting NORMAL MODE - full trajectory playback...")
        # self.send_vel_limit([4,4,4,4,4,4], [4,4,4,4,4,4])
        
        # Use left arm trajectory file
        filepath = '/home/irislab/r1pro_control/robot_control_ros2/recorded_trajectories/left_wrist_20250815_144818.csv'
        if not os.path.exists(filepath):
            self.get_logger().error(f"Trajectory file not found: {filepath}")
            return
        
        trajectory_points = self.load_trajectory_points(filepath)
        self.get_logger().info(f"Loaded {len(trajectory_points)} trajectory points")
        
        for idx, point in enumerate(trajectory_points, start=1):
            # Check if program should stop
            if not self.program_running or self.program_stopped:
                self.get_logger().info("Program stopped by user")
                break
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'base_link'
            
            pose_msg.pose.position.x = point['position'][0]
            pose_msg.pose.position.y = point['position'][1]
            pose_msg.pose.position.z = point['position'][2]
            
            pose_msg.pose.orientation.x = point['orientation'][0]
            pose_msg.pose.orientation.y = point['orientation'][1]
            pose_msg.pose.orientation.z = point['orientation'][2]
            pose_msg.pose.orientation.w = point['orientation'][3]
            
            self.pub.publish(pose_msg)
            
            self.get_logger().info(f"[{idx}] Sent pose")
            time.sleep(0.2)  # 20Hz
            
            # Spin once to handle callbacks
            rclpy.spin_once(self, timeout_sec=0.01)
        
        self.get_logger().info(f"[{self.arm}] NORMAL MODE trajectory complete.")
    
    def run_move_step_mode(self):
        """Incremental movement mode"""
        # Wait for start signal
        if not self.wait_for_start():
            return
        
        self.get_logger().info(f"[{self.arm}] Starting MOVE_STEP MODE - incremental movement...")
        
        # Load trajectory for left arm
        filepath = '/home/irislab/r1pro_control/robot_control_ros2/recorded_trajectories/left_wrist_20250815_144818.csv'
        if not os.path.exists(filepath):
            self.get_logger().error(f"Trajectory file not found: {filepath}")
            return
        
        self.trajectory_points = self.load_trajectory_points(filepath)
        self.get_logger().info(f"Loaded {len(self.trajectory_points)} trajectory points")
        
        # Wait for current pose
        if not self.wait_for_first_message():
            return
        
        self.get_logger().info("Ready for incremental movement!")
        self.get_logger().info("Press '1' + Enter to take one step toward current target")
        self.get_logger().info("Press 'q' + Enter to quit")
        
        # self.send_vel_limit([4,4,4,4,4,4], [4,4,4,4,4,4])
        
        # Manual step execution loop
        while rclpy.ok() and self.program_running and not self.program_stopped:
            try:
                if self.current_target_idx >= len(self.trajectory_points):
                    print("All trajectory points completed!")
                    break
                
                current_target = self.current_target_idx + 1
                total_targets = len(self.trajectory_points)
                print(f"\n[TARGET {current_target}/{total_targets}] Press '1' to step, 'q' to quit: ", end='')
                
                user_input = input().strip()
                if user_input == '1':
                    if not self.execute_move_step():
                        break
                elif user_input.lower() == 'q':
                    print("Quitting move_step mode...")
                    break
                else:
                    print("Invalid input. Press '1' to step or 'q' to quit.")
                
                # Process callbacks
                rclpy.spin_once(self, timeout_sec=0.01)
                
            except KeyboardInterrupt:
                print("\nQuitting move_step mode...")
                break
        
        self.get_logger().info(f"[{self.arm}] MOVE_STEP MODE complete")
    
    def run_translation_mode(self):
        """Position only movement with fixed orientation"""
        # Wait for start signal
        if not self.wait_for_start():
            return
        
        self.get_logger().info(f"[{self.arm}] Starting TRANSLATION MODE - fixed orientation...")
        
        # Wait for current pose to set fixed orientation
        if not self.wait_for_current_pose():
            return
        
        # Load trajectory for left arm
        filepath = '/home/irislab/r1pro_control/robot_control_ros2/recorded_trajectories/left_wrist_20250815_144818.csv'
        if not os.path.exists(filepath):
            self.get_logger().error(f"Trajectory file not found: {filepath}")
            return
        
        self.trajectory_points = self.load_trajectory_points(filepath)
        self.get_logger().info(f"Loaded {len(self.trajectory_points)} trajectory points")
        
        self.get_logger().info("Ready for incremental position movement with fixed orientation!")
        self.get_logger().info("Press '1' + Enter to take one step")
        self.get_logger().info("Press 'q' + Enter to quit")
        
        self.send_vel_limit([4,4,4,4,4,4], [4,4,4,4,4,4])
        
        while rclpy.ok() and self.program_running and not self.program_stopped:
            try:
                if self.current_target_idx >= len(self.trajectory_points):
                    print("All trajectory points completed!")
                    break
                
                current_target = self.current_target_idx + 1
                total_targets = len(self.trajectory_points)
                print(f"\n[TARGET {current_target}/{total_targets}] Press '1' to step, 'q' to quit: ", end='')
                
                user_input = input().strip()
                if user_input == '1':
                    if not self.execute_translation_step():
                        break
                elif user_input.lower() == 'q':
                    print("Quitting translation mode...")
                    break
                else:
                    print("Invalid input.")
                
                rclpy.spin_once(self, timeout_sec=0.01)
                
            except KeyboardInterrupt:
                print("\nQuitting translation mode...")
                break
        
        self.get_logger().info(f"[{self.arm}] TRANSLATION MODE complete")
    
    def run_translation_smooth_mode(self):
        """Automatic smooth translation with fixed orientation"""
        # Wait for start signal
        if not self.wait_for_start():
            return
        
        self.get_logger().info(f"[{self.arm}] Starting TRANSLATION_SMOOTH MODE...")
        
        # Wait for current pose to set fixed orientation
        if not self.wait_for_current_pose():
            return
        
        # Load trajectory for left arm
        filepath = '/home/irislab/r1pro_control/robot_control_ros2/recorded_trajectories/left_wrist_20250815_144818.csv'
        if not os.path.exists(filepath):
            self.get_logger().error(f"Trajectory file not found: {filepath}")
            return
        
        self.trajectory_points = self.load_trajectory_points(filepath)
        self.get_logger().info(f"Loaded {len(self.trajectory_points)} trajectory points")
        
        self.get_logger().info("Starting automatic smooth translation movement!")
        self.send_vel_limit([4,4,4,4,4,4], [4,4,4,4,4,4])
        
        # Automatic smooth execution
        while rclpy.ok() and self.program_running and not self.program_stopped and self.current_target_idx < len(self.trajectory_points):
            if not self.execute_translation_step():
                break
            time.sleep(0.1)  # 10Hz update rate
            rclpy.spin_once(self, timeout_sec=0.01)
        
        if self.current_target_idx >= len(self.trajectory_points):
            self.get_logger().info("✓ All trajectory points completed successfully!")
        
        self.get_logger().info(f"[{self.arm}] TRANSLATION_SMOOTH MODE complete")
    
    def run_ros_tele_mode(self):
        """Real-time teleop control via ROS topics"""
        # Wait for start signal
        if not self.wait_for_start():
            return
        
        self.get_logger().info(f"[{self.arm}] Starting ROS_TELE MODE...")
        self.get_logger().info("Waiting for teleop data on /teleop/left_wrist_pos...")
        
        # self.send_vel_limit([4,4,4,4,4,4], [4,4,4,4,4,4])
        
        # Continuous teleop loop
        while rclpy.ok() and self.program_running and not self.program_stopped:
            with self.teleop_lock:
                wrist_data = self.teleop_wrist_data
            
            if wrist_data is not None:
                # Extract position and orientation from teleop data (camera frame)
                cam_pt = np.array([
                    wrist_data.pose.position.x,
                    wrist_data.pose.position.y,
                    wrist_data.pose.position.z,
                    1.0
                ])
                
                q_cam = np.array([
                    wrist_data.pose.orientation.x,
                    wrist_data.pose.orientation.y,
                    wrist_data.pose.orientation.z,
                    wrist_data.pose.orientation.w
                ])
                
                # Transform position from camera to base frame
                base_pt = self.T_cam_to_base.dot(cam_pt)
                
                # Transform orientation
                q_base_hand = self.quaternion_multiply(self.q_cam_to_base, q_cam)
                q_base_ee = self.quaternion_multiply(q_base_hand, self.q_hand_to_ee)
                
                # Build and send pose message
                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = 'base_link'
                
                pose_msg.pose.position.x = base_pt[0]
                pose_msg.pose.position.y = base_pt[1]
                pose_msg.pose.position.z = base_pt[2]
                
                pose_msg.pose.orientation.x = q_base_ee[0]
                pose_msg.pose.orientation.y = q_base_ee[1]
                pose_msg.pose.orientation.z = q_base_ee[2]
                pose_msg.pose.orientation.w = q_base_ee[3]
                
                self.pub.publish(pose_msg)
                
                self.get_logger().debug(f"[ROS_TELE] Sent pose")
            else:
                self.get_logger().warning("No wrist data received", throttle_duration_sec=2.0)
            
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.001)  # 20Hz
        
        self.get_logger().info(f"[{self.arm}] ROS_TELE MODE complete")
    
    def run_print_mode(self):
        """Debug mode - print transformations without sending commands"""
        # Wait for start signal
        if not self.wait_for_start():
            return
        
        self.get_logger().info("Starting PRINT MODE - will print transformations only...")
        
        # Use left arm trajectory file
        filepath = '/home/irislab/r1pro_control/robot_control_ros2/recorded_trajectories/left_wrist_20250815_144818.csv'
        if not os.path.exists(filepath):
            self.get_logger().error(f"Trajectory file not found: {filepath}")
            return
        
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            
            for idx, row in enumerate(reader, start=1):
                # Parse camera frame data
                cam_pt = np.array([
                    float(row['wrist_x']),
                    float(row['wrist_y']),
                    float(row['wrist_z']),
                    1.0
                ])
                
                q_cam = np.array([
                    float(row['qx']),
                    float(row['qy']),
                    float(row['qz']),
                    float(row['qw'])
                ])
                
                # Transform to base frame
                base_pt = self.T_cam_to_base.dot(cam_pt)
                q_base_hand = self.quaternion_multiply(self.q_cam_to_base, q_cam)
                q_base_ee = self.quaternion_multiply(q_base_hand, self.q_hand_to_ee)
                
                print(f"\n[TEST] Row {idx}:")
                print(f"  Camera frame: pos=[{cam_pt[0]:.6f}, {cam_pt[1]:.6f}, {cam_pt[2]:.6f}]")
                print(f"               quat=[{q_cam[0]:.6f}, {q_cam[1]:.6f}, {q_cam[2]:.6f}, {q_cam[3]:.6f}]")
                print(f"  Base frame:   pos=[{base_pt[0]:.6f}, {base_pt[1]:.6f}, {base_pt[2]:.6f}]")
                print(f"  EE frame:    quat=[{q_base_ee[0]:.6f}, {q_base_ee[1]:.6f}, {q_base_ee[2]:.6f}, {q_base_ee[3]:.6f}]")
                print(f"  Would publish to topic: {self.target_topic}")
        
        self.get_logger().info(f"PRINT MODE complete - printed {idx} transformation results.")
    
    def run_ori_mode(self):
        """Orientation only mode with fixed position"""
        # Wait for start signal
        if not self.wait_for_start():
            return
        
        self.get_logger().info("Starting ORI MODE - fixed position, orientation from CSV...")
        
        # Wait for current pose to set fixed position
        if not self.wait_for_current_pose():
            return
        
        # Use left arm trajectory file
        filepath = '/home/irislab/r1pro_control/robot_control_ros2/recorded_trajectories/left_wrist_20250815_144818.csv'
        if not os.path.exists(filepath):
            self.get_logger().error(f"Trajectory file not found: {filepath}")
            return
        
        trajectory_points = self.load_trajectory_points(filepath)
        self.get_logger().info(f"Loaded {len(trajectory_points)} trajectory points")
        
        self.send_vel_limit([4,4,4,4,4,4], [4,4,4,4,4,4])
        
        for idx, point in enumerate(trajectory_points, start=1):
            # Check if program should stop
            if not self.program_running or self.program_stopped:
                self.get_logger().info("Program stopped by user")
                break
            
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'base_link'
            
            # Use fixed position
            pose_msg.pose.position.x = self.fixed_position[0]
            pose_msg.pose.position.y = self.fixed_position[1]
            pose_msg.pose.position.z = self.fixed_position[2]
            
            # Use orientation from CSV
            pose_msg.pose.orientation.x = point['orientation'][0]
            pose_msg.pose.orientation.y = point['orientation'][1]
            pose_msg.pose.orientation.z = point['orientation'][2]
            pose_msg.pose.orientation.w = point['orientation'][3]
            
            self.pub.publish(pose_msg)
            
            self.get_logger().info(f"[{idx}] Sent orientation with fixed position")
            time.sleep(0.05)  # 20Hz
            
            rclpy.spin_once(self, timeout_sec=0.01)
        
        self.get_logger().info("ORI MODE complete")
    
    def run_test_keyboard_mode(self):
        """Test mode to verify keyboard control without ROS topics"""
        from datetime import datetime
        
        self.get_logger().info("[TEST MODE] Testing keyboard control...")
        self.get_logger().info("This mode will print timestamps to test 's' and 'f' keys")
        
        # Wait for start signal
        if not self.wait_for_start():
            self.get_logger().info("[TEST MODE] Stopped before starting")
            return
        
        self.get_logger().info("[TEST MODE] Started! Will print time every second...")
        self.get_logger().info("[TEST MODE] Press 'f' to stop")
        
        counter = 0
        while rclpy.ok() and self.program_running and not self.program_stopped:
            counter += 1
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            print(f"[{counter:04d}] Time: {current_time} - Program is running...")
            
            # Sleep for 1 second but check for stop signal more frequently
            for _ in range(10):
                if not self.program_running or self.program_stopped:
                    break
                time.sleep(0.1)
                rclpy.spin_once(self, timeout_sec=0.01)
        
        if self.program_stopped or not self.program_running:
            self.get_logger().info(f"[TEST MODE] Stopped by user after {counter} iterations")
        else:
            self.get_logger().info(f"[TEST MODE] Completed {counter} iterations")
    
    def run(self):
        """Main run method - dispatches to appropriate mode"""
        if self.mode == 'move_step':
            self.run_move_step_mode()
        elif self.mode == 'ros_tele':
            self.run_ros_tele_mode()
        elif self.mode == 'translation':
            self.run_translation_mode()
        elif self.mode == 'translation_smooth':
            self.run_translation_smooth_mode()
        elif self.mode == 'print':
            self.run_print_mode()
        elif self.mode == 'ori':
            self.run_ori_mode()
        elif self.mode == 'normal':
            self.run_normal_mode()
        else:
            # Test mode - print current time to test keyboard control
            self.run_test_keyboard_mode()

def main():
    """Main function - parse arguments and run node"""
    # Parse command line arguments
    mode = 'ros_tele'  # Default mode
    
    if len(sys.argv) > 1:
        arg_map = {
            '--normal': 'normal',
            '--move_step': 'move_step',
            '--ros_tele': 'ros_tele',
            '--translation': 'translation',
            '--translation_smooth': 'translation_smooth',
            '--print': 'print',
            '--ori': 'ori',
            '--time':  'time'
        }
        
        if sys.argv[1] in arg_map:
            mode = arg_map[sys.argv[1]]
            print(f"=== RUNNING IN {mode.upper().replace('_', ' ')} MODE ===")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Available modes: --normal, --move_step, --ros_tele, --translation, --translation_smooth, --print, --ori")
            return
    
    # Initialize ROS2
    rclpy.init()
    
    # Create and run node
    node = TrajFollow(mode=mode)
    
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        # Signal threads to stop
        node.program_stopped = True
        if node.keyboard_thread:
            node.keyboard_thread.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()