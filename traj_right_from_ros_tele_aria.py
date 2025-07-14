#!/usr/bin/env python3
'''
This script is used to follow a trajectory from a CSV file.
The CSV file has the following columns with header row:
- wrist_x
- wrist_y
- wrist_z
- qx
- qy
- qz
- qw
- hand_1
- hand_2
- hand_3
- hand_4
- hand_5
- hand_6

The input trajectory is wrist position and orientation in camera frame.
Transformation pipeline: Camera Frame → Base Frame → EE Frame
- First: Transform camera frame to robot base frame (position + orientation)
- Then: Transform hand frame to end-effector frame (orientation only, same origin)
  Hand-to-EE: (1) rotate 180° around hand Y-axis, (2) rotate 90° around new hand Z-axis, (3) rotate -180° around Z-axis
'''

import rospy
from geometry_msgs.msg import PoseStamped
import threading
import os
import math
import csv
import numpy as np
import tf
import serial
import time
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# ---------- Inspire Hand Register Definitions ----------
regdict = {
    'angleSet': 1486,
    'speedSet': 1522,
    'forceSet': 1498
}

def open_serial(port='/dev/ttyUSB1', baudrate=115200):
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baudrate
    ser.timeout = 1
    ser.open()
    return ser


def write_register(ser, id, add, num, val):
    packet = [0xEB, 0x90, id, num + 3, 0x12, add & 0xFF, (add >> 8) & 0xFF]
    for i in range(num):
        packet.append(val[i] & 0xFF)
    checksum = sum(packet[2:]) & 0xFF
    packet.append(checksum)
    ser.write(bytearray(packet))
    time.sleep(0.01)
    ser.read_all()


def write6(ser, id, param, val):
    if param in ['angleSet', 'forceSet', 'speedSet']:
        regs = []
        for v in val:
            regs.append(v & 0xFF)
            regs.append((v >> 8) & 0xFF)
        write_register(ser, id, regdict[param], 12, regs)
    else:
        rospy.logerr(f"[Hand] Invalid param {param}")


class TrajFollow:
    def __init__(self, test_mode=False):
        rospy.init_node('traj_follow_right_node', anonymous=True)
        self.lock = threading.Lock()
        self.current_pose = None
        self.test_mode = test_mode  # Can be False, 'print', 'ori', 'translation', 'move_step', or 'ros_tele'

        # Arm topics
        self.arm = 'right'
        # self.pose_topic = f'/motion_control/pose_ee_arm_{self.arm}'
        self.pose_topic = f'/relaxed_ik/motion_control/pose_ee_arm_{self.arm}'
        self.target_topic = f'/motion_target/target_pose_arm_{self.arm}'
        rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_callback)
        self.pub = rospy.Publisher(self.target_topic, PoseStamped, queue_size=1)
        self.left_joint_state_pub = rospy.Publisher('/motion_target/target_joint_state_arm_left', JointState, queue_size=10)
        self.right_joint_state_pub = rospy.Publisher('/motion_target/target_joint_state_arm_right', JointState, queue_size=10)

        # ROS Teleop subscribers for ros_tele mode
        self.teleop_wrist_data = None
        self.teleop_hand_data = None
        self.teleop_lock = threading.Lock()
        
        if self.test_mode == 'ros_tele':
            rospy.Subscriber('/teleop/right_wrist_pos', PoseStamped, self.teleop_wrist_callback)
            rospy.Subscriber('/teleop/right_hand_joint', Float64MultiArray, self.teleop_hand_callback)
            rospy.loginfo("[ROS_TELE] Subscribed to teleop topics: /teleop/right_wrist_pos and /teleop/right_hand_joint")
        elif self.test_mode == 'ros_hand_test':
            rospy.Subscriber('/teleop/right_hand_joint', Float64MultiArray, self.teleop_hand_callback)
            rospy.loginfo("[ROS_HAND_TEST] Subscribed to teleop hand topic: /teleop/right_hand_joint")

        # Zed Camera (looking ~30 degree downwards) -> base transform (provided transformation matrix)
        # self.T_cam_to_base = np.array([
        #     [ 0.05787445, -0.7673487,   0.63861296,  0.13727924],
        #     [-0.99832064, -0.04611135,  0.03506627,  0.01021076],
        #     [ 0.00253925, -0.63956994, -0.76872872,  0.50120488],
        #     [ 0., 0., 0., 1.]
        # ])
        
        
        # pure translation upwards from the camera base, slightliy tilting down
        self.T_cam_to_base = np.array([
            [-0.01359682, -0.25145117,  0.96777448,  0.11040003],
            [-0.99937955, -0.02803206, -0.02132427,  0.05098301],
            [ 0.03249072, -0.96746396, -0.25091401,  0.42589351],
            [ 0.        ,  0.        ,  0.        ,  1.        ]
        ])
                
        
        # Extract rotation matrix
        R_cam_to_base = self.T_cam_to_base[:3, :3]
        from scipy.spatial.transform import Rotation as R
        # Convert to quaternion [x, y, z, w]
        self.q_cam_to_base = R.from_matrix(R_cam_to_base).as_quat()
        # Quaternion for camera to base rotation: [x, y, z, w]
        # self.q_cam_to_base = np.array([0.56243872, -0.55598534, 0.44461075, -0.42055235])

        # Hand to end-effector frame transformation rotation matrices
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
        # 3. Rotate -180° around Z-axis
        # self.R_hand_z_neg180 = np.array([
        #     [-1,  0,  0],
        #     [ 0, -1,  0],
        #     [ 0,  0,  1]
        # ])
        # Combined hand to ee transformation: R_hand_to_ee = R1 * R2 * R3 (local frame order)
        # self.R_hand_to_ee = np.dot(np.dot(self.R_hand_y_180, self.R_hand_z_90), self.R_hand_z_neg180)
        self.R_hand_to_ee = np.dot(self.R_hand_y_180, self.R_hand_z_90)
        
        # Convert rotation matrix to quaternion
        # tf.transformations.quaternion_from_matrix() expects a 4x4 homogeneous matrix
        T_hand_to_ee = np.eye(4)
        T_hand_to_ee[:3, :3] = self.R_hand_to_ee
        self.q_hand_to_ee = tf.transformations.quaternion_from_matrix(T_hand_to_ee)

        # Fixed position for orientation testing (will be set from current pose)
        self.fixed_position = None
        # Fixed orientation for translation testing (will be set from current pose)
        self.fixed_orientation = None

        # Move step mode variables
        self.trajectory_points = []  # List of target poses for move_step mode
        self.current_target_idx = 0  # Current target point index
        self.step_size = 0.025  # Step size in meters for incremental movement
        self.position_threshold = 0.01  # Distance threshold to consider target reached (reduced from 0.01)
        self.min_movement_threshold = 0.025  # Minimum movement to send to robot (2cm - ALWAYS take 2cm steps)

        # Inspire Hand serial setup (right hand) - enable for translation mode too
        if not self.test_mode or self.test_mode in ['move_step', 'translation', 'ros_tele', 'ros_hand_test']:
            serial_port = '/dev/ttyUSB1'
            baudrate = 115200
            self.hand_id = 2
            try:
                self.ser = open_serial(serial_port, baudrate)
                write6(self.ser, self.hand_id, 'speedSet', [1000]*6)
                write6(self.ser, self.hand_id, 'forceSet', [800]*6)
                if self.test_mode in ['move_step', 'translation', 'ros_tele', 'ros_hand_test']:
                    rospy.loginfo(f"[TEST MODE: {self.test_mode}] Hand serial setup complete")
            except Exception as e:
                rospy.logwarn(f"Hand serial setup failed: {e}. Continuing without hand control.")
                self.ser = None
        else:
            rospy.loginfo(f"[TEST MODE: {self.test_mode}] Skipping hand serial setup")
            self.ser = None
            self.hand_id = 2

    def teleop_wrist_callback(self, msg):
        """Callback for teleop wrist position data"""
        with self.teleop_lock:
            self.teleop_wrist_data = msg
            
    def teleop_hand_callback(self, msg):
        """Callback for teleop hand joint data"""
        with self.teleop_lock:
            self.teleop_hand_data = msg.data

    def send_vel_limit(self, vel_limit_left, vel_limit_right):
        left_joint_state = JointState()
        left_joint_state.velocity = vel_limit_left
        right_joint_state = JointState()
        right_joint_state.velocity = vel_limit_right

        self.left_joint_state_pub.publish(left_joint_state)
        self.right_joint_state_pub.publish(right_joint_state)

    def pose_callback(self, msg):
        with self.lock:
            self.current_pose = msg

    def wait_for_current_pose(self, timeout=10.0):
        """Wait for current pose and set it as fixed position for orientation testing"""
        if self.test_mode == 'ori':
            rospy.loginfo("[TEST MODE: ori] Waiting for current robot pose...")
        elif self.test_mode == 'translation':
            rospy.loginfo("[TEST MODE: translation] Waiting for current robot pose...")
        
        start_time = rospy.Time.now()
        
        while not rospy.is_shutdown():
            with self.lock:
                if self.current_pose is not None:
                    self.fixed_position = [
                        self.current_pose.pose.position.x,
                        self.current_pose.pose.position.y,
                        self.current_pose.pose.position.z
                    ]
                    self.fixed_orientation = [
                        self.current_pose.pose.orientation.x,
                        self.current_pose.pose.orientation.y,
                        self.current_pose.pose.orientation.z,
                        self.current_pose.pose.orientation.w
                    ]
                    
                    if self.test_mode == 'ori':
                        rospy.loginfo(f"[TEST MODE: ori] Fixed position set to: [{self.fixed_position[0]:.3f}, {self.fixed_position[1]:.3f}, {self.fixed_position[2]:.3f}]")
                    elif self.test_mode == 'translation':
                        rospy.loginfo(f"[TEST MODE: translation] Fixed orientation set to: [{self.fixed_orientation[0]:.3f}, {self.fixed_orientation[1]:.3f}, {self.fixed_orientation[2]:.3f}, {self.fixed_orientation[3]:.3f}]")
                    
                    return True
                    
            if (rospy.Time.now() - start_time).to_sec() > timeout:
                if self.test_mode == 'ori':
                    rospy.logerr("[TEST MODE: ori] Timeout waiting for current pose")
                elif self.test_mode == 'translation':
                    rospy.logerr("[TEST MODE: translation] Timeout waiting for current pose")
                return False
                
            rospy.sleep(0.1)
        return False

    def load_trajectory_points(self, filepath):
        """Load all trajectory points from CSV for move_step mode"""
        trajectory_points = []
        
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            hand_cols = [col for col in reader.fieldnames if col.startswith('hand_')]
            
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
                q_base_hand = tf.transformations.quaternion_multiply(self.q_cam_to_base, q_cam)
                # Transform from hand frame to ee frame
                q_base_ee = tf.transformations.quaternion_multiply(q_base_hand, self.q_hand_to_ee)
                
                # Hand commands
                hand_vals = [int(float(row[col])) for col in hand_cols]
                
                # Store transformed point
                point = {
                    'position': [base_pt[0], base_pt[1], base_pt[2]],
                    'orientation': [q_base_ee[0], q_base_ee[1], q_base_ee[2], q_base_ee[3]],
                    'hand': hand_vals
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
        
        # Calculate next position - ALWAYS take minimum step size regardless of distance
        if distance > 0:
            direction_normalized = direction / distance
            
            # ALWAYS take minimum step size, even if it would overshoot
            step_distance = self.min_movement_threshold
            
            next_pos = [
                current_pos[0] + direction_normalized[0] * step_distance,
                current_pos[1] + direction_normalized[1] * step_distance,
                current_pos[2] + direction_normalized[2] * step_distance
            ]
            
            print(f"[DEBUG] Distance to target: {distance:.6f}m")
            print(f"[DEBUG] Step distance: {step_distance:.6f}m (ALWAYS minimum step size)")
            print(f"[DEBUG] Min movement threshold: {self.min_movement_threshold:.6f}m")
            
        else:
            next_pos = current_pos  # Already at exact target
            
        # Calculate next orientation using SLERP (Spherical Linear Interpolation)
        next_ori = current_ori  # Default to current orientation
        orientation_reached = True  # Default to true if no orientation interpolation
        
        if current_ori is not None and target_ori is not None:
            # Convert to numpy arrays for easier manipulation
            q_current = np.array(current_ori)
            q_target = np.array(target_ori)
            
            # Calculate quaternion difference/distance using dot product
            dot_product = np.dot(q_current, q_target)
            
            # Handle quaternion double cover (q and -q represent same rotation)
            if dot_product < 0:
                q_target = -q_target
                dot_product = -dot_product
            
            # Clamp dot product to avoid numerical issues
            dot_product = np.clip(dot_product, -1.0, 1.0)
            
            # Calculate angular distance
            theta = np.arccos(abs(dot_product))
            
            # Define orientation threshold (in radians, ~5.7 degrees)
            orientation_threshold = 0.1
            orientation_reached = theta <= orientation_threshold
            
            # Calculate interpolation factor based on step size
            # Use a fixed angular step size (in radians)
            angular_step_size = 0.05  # ~2.9 degrees per step
            
            if theta > orientation_threshold:
                # Calculate interpolation factor
                t = min(angular_step_size / theta, 1.0)
                
                # Perform SLERP
                if theta > 1e-6:  # Avoid division by zero
                    sin_theta = np.sin(theta)
                    weight_current = np.sin((1.0 - t) * theta) / sin_theta
                    weight_target = np.sin(t * theta) / sin_theta
                    next_ori = weight_current * q_current + weight_target * q_target
                    # Normalize the result
                    next_ori = next_ori / np.linalg.norm(next_ori)
                else:
                    next_ori = q_target  # Quaternions are too close, just use target
            else:
                next_ori = q_target  # Close enough, use target orientation
        
        # Target is reached when both position and orientation are close enough
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
        
        # Calculate next step for both position and orientation
        next_pos, next_ori, target_reached = self.calculate_step_toward_target(
            current_pos, target_pos, current_ori, target_ori)

        # DEBUG: Print Current pos and ori
        print(f"[DEBUG] Current pos: {current_pos}")
        print(f"[DEBUG] Current ori: {current_ori}")

        # DEBUG: Print Target pos and ori
        print(f"[DEBUG] Target pos: {target_pos}")
        print(f"[DEBUG] Target ori: {target_ori}")

        # DEBUG: Print next_pos and next_ori
        print(f"[DEBUG] Next pos: {next_pos}")
        print(f"[DEBUG] Next ori: {next_ori}")

        # Calculate relative movement (following key_torso.py pattern)
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1] 
        dz = next_pos[2] - current_pos[2]
        
        # Calculate relative orientation change
        dqx = next_ori[0] - current_ori[0]
        dqy = next_ori[1] - current_ori[1]
        dqz = next_ori[2] - current_ori[2]
        dqw = next_ori[3] - current_ori[3]
        
        print(f"[DEBUG] Relative movement: dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}")
        print(f"[DEBUG] Relative orientation: dqx={dqx:.6f}, dqy={dqy:.6f}, dqz={dqz:.6f}, dqw={dqw:.6f}")

        # Create pose message (following key_torso.py pattern exactly)
        with self.lock:
            if self.current_pose is None:
                print("No current pose available for relative movement!")
                return True
                
            pose = self.current_pose.pose
            
            new_pose = PoseStamped()
            new_pose.header.stamp = rospy.Time.now()
            new_pose.header.frame_id = "base_link"
            
            # Apply relative movement (same as keyboard control)
            new_pose.pose.position.x = pose.position.x + dx
            new_pose.pose.position.y = pose.position.y + dy
            new_pose.pose.position.z = pose.position.z + dz
            
            # Use interpolated orientation (not fixed orientation)
            new_pose.pose.orientation.x = next_ori[0]
            new_pose.pose.orientation.y = next_ori[1]
            new_pose.pose.orientation.z = next_ori[2]
            new_pose.pose.orientation.w = next_ori[3]
            
            # DEBUG: Print exact values being sent
            print(f"[DEBUG] SENDING POSE MESSAGE:")
            print(f"  Current robot pos: [{pose.position.x:.6f}, {pose.position.y:.6f}, {pose.position.z:.6f}]")
            print(f"  Current robot ori: [{pose.orientation.x:.6f}, {pose.orientation.y:.6f}, {pose.orientation.z:.6f}, {pose.orientation.w:.6f}]")
            print(f"  Command pos: [{new_pose.pose.position.x:.6f}, {new_pose.pose.position.y:.6f}, {new_pose.pose.position.z:.6f}]")
            print(f"  Command ori: [{new_pose.pose.orientation.x:.6f}, {new_pose.pose.orientation.y:.6f}, {new_pose.pose.orientation.z:.6f}, {new_pose.pose.orientation.w:.6f}]")
            print(f"  Command delta pos: [{new_pose.pose.position.x - pose.position.x:.6f}, {new_pose.pose.position.y - pose.position.y:.6f}, {new_pose.pose.position.z - pose.position.z:.6f}]")
            print(f"  Command delta ori: [{new_pose.pose.orientation.x - pose.orientation.x:.6f}, {new_pose.pose.orientation.y - pose.orientation.y:.6f}, {new_pose.pose.orientation.z - pose.orientation.z:.6f}, {new_pose.pose.orientation.w - pose.orientation.w:.6f}]")
            print(f"  Topic: {self.target_topic}")
            print(f"  Frame ID: {new_pose.header.frame_id}")
            
            # Send commands
            self.pub.publish(new_pose)
            
            # Add delay to ensure command is processed
            rospy.sleep(0.1)
            
            if self.ser:
                write6(self.ser, self.hand_id, 'angleSet', target_point['hand'])
                print(f"[DEBUG] Hand command sent: {target_point['hand']}")
            else:
                print(f"[DEBUG] No hand serial connection - hand command would be: {target_point['hand']}")

        if target_reached:
            print(f"✓ Point {self.current_target_idx + 1} reached (both position and orientation)!")
            self.current_target_idx += 1
            if self.current_target_idx < len(self.trajectory_points):
                print(f"Next target: Point {self.current_target_idx + 1}")
        else:
            # Calculate distances for progress feedback
            pos_dist = np.linalg.norm([target_pos[0] - current_pos[0], 
                                     target_pos[1] - current_pos[1], 
                                     target_pos[2] - current_pos[2]])
            
            # Calculate angular distance for orientation
            q_current = np.array(current_ori)
            q_target = np.array(target_ori)
            dot_product = np.abs(np.dot(q_current, q_target))
            dot_product = np.clip(dot_product, 0.0, 1.0)
            angular_dist = np.arccos(dot_product) * 180.0 / np.pi  # Convert to degrees
            
            print(f"Progress: pos_dist={pos_dist:.4f}m, ori_dist={angular_dist:.2f}°")
        
        return True

    def execute_translation_step(self):
        """Execute one incremental step in translation mode (position only, fixed orientation)"""
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
        
        # Get current target point (position only)
        target_point = self.trajectory_points[self.current_target_idx]
        target_pos = target_point['position']
        
        # Calculate next step for position only (no orientation interpolation)
        next_pos, _, target_reached = self.calculate_step_toward_target(
            current_pos, target_pos, None, None)

        # DEBUG: Print Current pos and target pos
        print(f"[DEBUG] Current pos: {current_pos}")
        print(f"[DEBUG] Target pos: {target_pos}")
        print(f"[DEBUG] Next pos: {next_pos}")

        # Calculate relative movement
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1] 
        dz = next_pos[2] - current_pos[2]
        
        print(f"[DEBUG] Relative movement: dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}")

        # Create pose message with fixed orientation
        with self.lock:
            if self.current_pose is None:
                print("No current pose available for relative movement!")
                return True
                
            pose = self.current_pose.pose
            
            new_pose = PoseStamped()
            new_pose.header.stamp = rospy.Time.now()
            new_pose.header.frame_id = "base_link"
            
            # Apply relative position movement
            new_pose.pose.position.x = pose.position.x + dx
            new_pose.pose.position.y = pose.position.y + dy
            new_pose.pose.position.z = pose.position.z + dz
            
            # Use fixed orientation (set at startup)
            new_pose.pose.orientation.x = self.fixed_orientation[0]
            new_pose.pose.orientation.y = self.fixed_orientation[1]
            new_pose.pose.orientation.z = self.fixed_orientation[2]
            new_pose.pose.orientation.w = self.fixed_orientation[3]
            
            # DEBUG: Print exact values being sent
            print(f"[DEBUG] SENDING POSE MESSAGE (TRANSLATION MODE):")
            print(f"  Current robot pos: [{pose.position.x:.6f}, {pose.position.y:.6f}, {pose.position.z:.6f}]")
            print(f"  Command pos: [{new_pose.pose.position.x:.6f}, {new_pose.pose.position.y:.6f}, {new_pose.pose.position.z:.6f}]")
            print(f"  Command delta pos: [{new_pose.pose.position.x - pose.position.x:.6f}, {new_pose.pose.position.y - pose.position.y:.6f}, {new_pose.pose.position.z - pose.position.z:.6f}]")
            print(f"  Fixed orientation: [{new_pose.pose.orientation.x:.6f}, {new_pose.pose.orientation.y:.6f}, {new_pose.pose.orientation.z:.6f}, {new_pose.pose.orientation.w:.6f}]")
            print(f"  Topic: {self.target_topic}")
            print(f"  Frame ID: {new_pose.header.frame_id}")
            
            # Send commands
            self.pub.publish(new_pose)
            
            # Add delay to ensure command is processed
            rospy.sleep(0.1)
            
            if self.ser:
                write6(self.ser, self.hand_id, 'angleSet', target_point['hand'])
                print(f"[DEBUG] Hand command sent: {target_point['hand']}")
            else:
                print(f"[DEBUG] No hand serial connection - hand command would be: {target_point['hand']}")

        if target_reached:
            print(f"✓ Point {self.current_target_idx + 1} reached (position only)!")
            self.current_target_idx += 1
            if self.current_target_idx < len(self.trajectory_points):
                print(f"Next target: Point {self.current_target_idx + 1}")
        else:
            # Calculate distance for progress feedback
            pos_dist = np.linalg.norm([target_pos[0] - current_pos[0], 
                                     target_pos[1] - current_pos[1], 
                                     target_pos[2] - current_pos[2]])
            
            print(f"Progress: pos_dist={pos_dist:.4f}m (orientation fixed)")
        
        return True

    def run(self):
        rate = rospy.Rate(30)
        if self.test_mode == 'print':
            rospy.loginfo(f"[{self.arm}] Starting TEST MODE: print - will print commands instead of sending...")
        elif self.test_mode == 'ori':
            rospy.loginfo(f"[{self.arm}] Starting TEST MODE: ori - fixed position, orientation from camera frame...")
            # Wait for current pose to set fixed position
            if not self.wait_for_current_pose():
                return
        elif self.test_mode == 'ros_tele':
            rospy.loginfo(f"[{self.arm}] Starting ROS_TELE MODE: subscribing to teleop topics for real-time control...")
            rospy.loginfo("[ROS_TELE] Waiting for teleop data...")
            rospy.loginfo("[ROS_TELE] Topics: /teleop/right_wrist_pos (PoseStamped) and /teleop/right_hand_joint (Float64MultiArray)")
            rospy.loginfo("[ROS_TELE] Robot will respond continuously to teleop commands")
            
            self.send_vel_limit([4,4,4,4,4,4],[4,4,4,4,4,4])
            
            # ROS_TELE mode: continuous loop waiting for teleop data
            while not rospy.is_shutdown():
                with self.teleop_lock:
                    wrist_data = self.teleop_wrist_data
                    hand_data = self.teleop_hand_data
                
                if wrist_data is not None and hand_data is not None:
                    # Extract wrist position and orientation from teleop data (assume camera frame)
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
                    
                    print(f"[DEBUG] cam_pt: {cam_pt}")
                    print(f"[DEBUG] q_cam: {q_cam}")
                    
                    # Apply same transformations as normal mode
                    # Step 1: Transform position from camera frame to base frame
                    base_pt = self.T_cam_to_base.dot(cam_pt)
                    
                    # Step 2: Transform orientation from camera frame to base frame
                    q_base_hand = tf.transformations.quaternion_multiply(self.q_cam_to_base, q_cam)
                    
                    # Step 3: Transform from hand frame to ee frame (orientation only)
                    q_base_ee = tf.transformations.quaternion_multiply(q_base_hand, self.q_hand_to_ee)
                    
                    # Build pose message
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = rospy.Time.now()
                    pose_msg.header.frame_id = 'base_link'
                    
                    # Set transformed position and orientation
                    pose_msg.pose.position.x = base_pt[0]
                    pose_msg.pose.position.y = base_pt[1]
                    pose_msg.pose.position.z = base_pt[2]
                    
                    pose_msg.pose.orientation.x = q_base_ee[0]
                    pose_msg.pose.orientation.y = q_base_ee[1]
                    pose_msg.pose.orientation.z = q_base_ee[2]
                    pose_msg.pose.orientation.w = q_base_ee[3]
                    
                    # Convert hand data to integers (assuming Float64MultiArray contains 6 values)
                    hand_vals = [int(val) for val in hand_data[:6]]  # Take first 6 values
                    
                    # Send commands to robot
                    self.pub.publish(pose_msg)
                    
                    if self.ser:
                        write6(self.ser, self.hand_id, 'angleSet', hand_vals)
                    
                    rospy.logdebug(f"[ROS_TELE] pos=({pose_msg.pose.position.x:.3f}, "
                                   f"{pose_msg.pose.position.y:.3f}, {pose_msg.pose.position.z:.3f}) | "
                                   f"hand={hand_vals}")
                else:
                    if wrist_data is None:
                        rospy.logwarn_throttle(2.0, "[ROS_TELE] No wrist data received on /teleop/right_wrist_pos")
                    if hand_data is None:
                        rospy.logwarn_throttle(2.0, "[ROS_TELE] No hand data received on /teleop/right_hand_joint")
                
                rate.sleep()
            
            rospy.loginfo(f"[{self.arm}] ROS_TELE MODE complete")
            return
        elif self.test_mode == 'ros_hand_test':
            rospy.loginfo(f"[{self.arm}] Starting ROS_HAND_TEST MODE: subscribing to teleop hand topic for hand-only control...")
            rospy.loginfo("[ROS_HAND_TEST] Waiting for hand data...")
            rospy.loginfo("[ROS_HAND_TEST] Topic: /teleop/right_hand_joint (Float64MultiArray)")
            rospy.loginfo("[ROS_HAND_TEST] Only hand commands will be sent - NO wrist position commands")
            
            # ROS_HAND_TEST mode: continuous loop waiting for hand teleop data only
            while not rospy.is_shutdown():
                with self.teleop_lock:
                    hand_data = self.teleop_hand_data
                
                if hand_data is not None:
                    # Convert hand data to integers (assuming Float64MultiArray contains 6 values)
                    hand_vals = [int(val) for val in hand_data[:6]]  # Take first 6 values
                    
                    # Send ONLY hand commands to robot (no wrist position)
                    if self.ser:
                        write6(self.ser, self.hand_id, 'angleSet', hand_vals)
                        rospy.logdebug(f"[ROS_HAND_TEST] hand={hand_vals}")
                    else:
                        rospy.logwarn_throttle(5.0, "[ROS_HAND_TEST] No hand serial connection available")
                else:
                    rospy.logwarn_throttle(2.0, "[ROS_HAND_TEST] No hand data received on /teleop/right_hand_joint")
                
                rate.sleep()
            
            rospy.loginfo(f"[{self.arm}] ROS_HAND_TEST MODE complete")
            return
        elif self.test_mode == 'translation':
            rospy.loginfo(f"[{self.arm}] Starting TEST MODE: translation - fixed orientation, updating position from CSV...")
            # Wait for current pose to set fixed position and orientation
            if not self.wait_for_current_pose():
                return
            
            # Load trajectory points for stepping
            filepath = '/home/nvidia/ke/r1_pro_sdk/install/share/mobiman/script/right/robot_commands_right_hand_translate.csv'
            # filepath = '/home/nvidia/ke/r1_pro_sdk/install/share/mobiman/script/right/robot_commands_right_hand.csv'
            if not os.path.exists(filepath):
                rospy.logerr(f"Trajectory file not found: {filepath}")
                return
            
            self.trajectory_points = self.load_trajectory_points(filepath)
            rospy.loginfo(f"Loaded {len(self.trajectory_points)} trajectory points")
            
            rospy.loginfo("Ready for incremental position movement with fixed orientation!")
            rospy.loginfo("Press '1' + Enter to take one step toward current target position")
            rospy.loginfo("Press 'q' + Enter to quit")
            rospy.loginfo("NOTE: Only position will move incrementally, orientation stays fixed")
            
            self.send_vel_limit([4,4,4,4,4,4],[4,4,4,4,4,4])
            
            # Manual step execution loop for translation mode
            while not rospy.is_shutdown():
                try:
                    if self.current_target_idx >= len(self.trajectory_points):
                        print("All trajectory points completed!")
                        break
                        
                    current_target = self.current_target_idx + 1
                    total_targets = len(self.trajectory_points)
                    print(f"\n[TARGET {current_target}/{total_targets}] Press '1' to step toward target position (fixed ori), 'q' to quit: ", end='')
                    
                    user_input = input().strip()
                    if user_input == '1':
                        if not self.execute_translation_step():
                            break
                    elif user_input.lower() == 'q':
                        print("Quitting translation mode...")
                        break
                    else:
                        print("Invalid input. Press '1' to step or 'q' to quit.")
                        
                except KeyboardInterrupt:
                    print("\nQuitting translation mode...")
                    break
            
            rospy.loginfo(f"[{self.arm}] TEST MODE: translation complete")
            return
        elif self.test_mode == 'translation_smooth':
            rospy.loginfo(f"[{self.arm}] Starting TEST MODE: translation_smooth - automatic smooth translation with fixed orientation...")
            # Wait for current pose to set fixed orientation
            if not self.wait_for_current_pose():
                return
            
            # Load trajectory points for automatic smooth execution
            filepath = '/home/nvidia/ke/r1_pro_sdk/install/share/mobiman/script/right/robot_commands_right_hand_translate.csv'
            if not os.path.exists(filepath):
                rospy.logerr(f"Trajectory file not found: {filepath}")
                return
            
            self.trajectory_points = self.load_trajectory_points(filepath)
            rospy.loginfo(f"Loaded {len(self.trajectory_points)} trajectory points")
            
            rospy.loginfo("Starting automatic smooth translation movement with fixed orientation!")
            rospy.loginfo("NOTE: Robot will automatically move through all waypoints smoothly")
            
            self.send_vel_limit([4,4,4,4,4,4],[4,4,4,4,4,4])
            
            # Automatic smooth execution loop for translation_smooth mode
            smooth_rate = rospy.Rate(10)  # Higher rate for smoother motion (10 Hz)
            
            while not rospy.is_shutdown() and self.current_target_idx < len(self.trajectory_points):
                if not self.execute_translation_step():
                    break
                smooth_rate.sleep()
            
            if self.current_target_idx >= len(self.trajectory_points):
                rospy.loginfo("✓ All trajectory points completed successfully!")
            
            rospy.loginfo(f"[{self.arm}] TEST MODE: translation_smooth complete")
            return
        elif self.test_mode == 'move_step':
            rospy.loginfo(f"[{self.arm}] Starting TEST MODE: move_step - incremental position AND orientation movement...")
            
            # Load trajectory points
            filepath = '/home/nvidia/ke/r1_pro_sdk_118/install/share/mobiman/scripts/right/right.csv'
            if not os.path.exists(filepath):
                rospy.logerr(f"Trajectory file not found: {filepath}")
                return
            
            self.trajectory_points = self.load_trajectory_points(filepath)
            rospy.loginfo(f"Loaded {len(self.trajectory_points)} trajectory points")
            
            # Wait for current pose (no need to capture fixed orientation for move_step)
            rospy.loginfo("Waiting for current robot pose...")
            start_time = rospy.Time.now()
            while not rospy.is_shutdown():
                with self.lock:
                    if self.current_pose is not None:
                        rospy.loginfo("Current robot pose received - ready for move_step mode!")
                        break
                if (rospy.Time.now() - start_time).to_sec() > 10.0:
                    rospy.logerr("Timeout waiting for current pose")
                    return
                rospy.sleep(0.1)
            
            rospy.loginfo("Ready for incremental movement!")
            rospy.loginfo("Press '1' + Enter to take one step toward current target")
            rospy.loginfo("Press 'q' + Enter to quit")
            rospy.loginfo("NOTE: Both position AND orientation will move incrementally toward CSV targets")
            
            self.send_vel_limit([4,4,4,4,4,4],[4,4,4,4,4,4])
            
            # Manual step execution loop
            while not rospy.is_shutdown():
                try:
                    if self.current_target_idx >= len(self.trajectory_points):
                        print("All trajectory points completed!")
                        break
                        
                    current_target = self.current_target_idx + 1
                    total_targets = len(self.trajectory_points)
                    print(f"\n[TARGET {current_target}/{total_targets}] Press '1' to step toward target (pos+ori), 'q' to quit: ", end='')
                    
                    user_input = input().strip()
                    if user_input == '1':
                        if not self.execute_move_step():
                            break
                    elif user_input.lower() == 'q':
                        print("Quitting move_step mode...")
                        break
                    else:
                        print("Invalid input. Press '1' to step or 'q' to quit.")
                        
                except KeyboardInterrupt:
                    print("\nQuitting move_step mode...")
                    break
            
            rospy.loginfo(f"[{self.arm}] TEST MODE: move_step complete")
            return
        else:
            rospy.loginfo(f"[{self.arm}] Starting trajectory with hand control at {rate} Hz...")
            self.send_vel_limit([4,4,4,4,4,4],[4,4,4,4,4,4])

        filepath = '/home/nvidia/ke/r1_pro_sdk_118/install/share/mobiman/scripts/right/right.csv'
        if not os.path.exists(filepath):
            rospy.logerr(f"Trajectory file not found: {filepath}")
            return

        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            # assume last 6 columns are hand values
            hand_cols = [col for col in reader.fieldnames if col.startswith('hand_')]

            for idx, row in enumerate(reader, start=1):
                # Parse position (camera frame)
                cam_pt = np.array([
                    float(row['wrist_x']),  # camera_pos_x
                    float(row['wrist_y']),  # camera_pos_y
                    float(row['wrist_z']),  # camera_pos_z
                    1.0                     # homogeneous coordinate
                ])
                # Step 1: Transform position from camera frame to base frame
                base_pt = self.T_cam_to_base.dot(cam_pt)

                # Parse orientation (camera frame quaternion)
                q_cam = np.array([
                    float(row['qx']),   # camera_quat_x
                    float(row['qy']),   # camera_quat_y
                    float(row['qz']),   # camera_quat_z
                    float(row['qw'])    # camera_quat_w
                ])
                
                # Step 2: Transform orientation from camera frame to base frame
                q_base_hand = tf.transformations.quaternion_multiply(self.q_cam_to_base, q_cam)
                
                # Step 3: Transform from hand frame to ee frame (orientation only)
                q_base_ee = tf.transformations.quaternion_multiply(q_base_hand, self.q_hand_to_ee)

                # build pose message
                pose_msg = PoseStamped()
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.header.frame_id = 'base_link'
                
                # Set position based on mode
                if self.test_mode == 'ori':
                    # Use fixed position
                    pose_msg.pose.position.x = self.fixed_position[0]
                    pose_msg.pose.position.y = self.fixed_position[1]
                    pose_msg.pose.position.z = self.fixed_position[2]
                else:
                    # Use transformed position from camera frame to base frame
                    pose_msg.pose.position.x = base_pt[0]
                    pose_msg.pose.position.y = base_pt[1]
                    pose_msg.pose.position.z = base_pt[2]
                
                # Set orientation based on mode
                # Use transformed orientation (normal and ori modes)
                pose_msg.pose.orientation.x = q_base_ee[0]
                pose_msg.pose.orientation.y = q_base_ee[1]
                pose_msg.pose.orientation.z = q_base_ee[2]
                pose_msg.pose.orientation.w = q_base_ee[3]

                # read hand commands
                hand_vals = [int(float(row[col])) for col in hand_cols]

                if self.test_mode == 'print':
                    # TEST MODE: print - Print transformation results
                    print(f"\n[TEST] Row {idx}:")
                    print(f"  Camera frame input:      pos=[{cam_pt[0]:.6f}, {cam_pt[1]:.6f}, {cam_pt[2]:.6f}]")
                    print(f"                           quat=[{q_cam[0]:.6f}, {q_cam[1]:.6f}, {q_cam[2]:.6f}, {q_cam[3]:.6f}]")
                    print(f"  Base frame (cam->base):  pos=[{base_pt[0]:.6f}, {base_pt[1]:.6f}, {base_pt[2]:.6f}]")
                    print(f"                           quat=[{q_base_hand[0]:.6f}, {q_base_hand[1]:.6f}, {q_base_hand[2]:.6f}, {q_base_hand[3]:.6f}]")
                    print(f"  EE frame (hand->ee):     quat=[{q_base_ee[0]:.6f}, {q_base_ee[1]:.6f}, {q_base_ee[2]:.6f}, {q_base_ee[3]:.6f}]")
                    print(f"  Hand commands: {hand_vals}")
                    print(f"  Final pose message:      pos=[{pose_msg.pose.position.x:.6f}, {pose_msg.pose.position.y:.6f}, {pose_msg.pose.position.z:.6f}]")
                    print(f"                           quat=[{pose_msg.pose.orientation.x:.6f}, {pose_msg.pose.orientation.y:.6f}, {pose_msg.pose.orientation.z:.6f}, {pose_msg.pose.orientation.w:.6f}]")
                    print(f"  Would publish to topic: {self.target_topic}")
                    print(f"  --- Mode comparisons ---")
                    print(f"  NORMAL mode: uses pos=[{base_pt[0]:.6f}, {base_pt[1]:.6f}, {base_pt[2]:.6f}] + transformed quat")
                    print(f"  ORI mode: uses FIXED pos + quat=[{q_base_ee[0]:.6f}, {q_base_ee[1]:.6f}, {q_base_ee[2]:.6f}, {q_base_ee[3]:.6f}]")
                    print(f"  TRANSLATION mode: incremental position movement toward pos=[{base_pt[0]:.6f}, {base_pt[1]:.6f}, {base_pt[2]:.6f}] + FIXED quat")
                    print(f"  MOVE_STEP mode: incremental position AND orientation movement toward pos=[{base_pt[0]:.6f}, {base_pt[1]:.6f}, {base_pt[2]:.6f}] + quat=[{q_base_ee[0]:.6f}, {q_base_ee[1]:.6f}, {q_base_ee[2]:.6f}, {q_base_ee[3]:.6f}]")
                    print(f"  ROS_TELE mode: real-time teleop control with same transformations as NORMAL mode")
                elif self.test_mode == 'ori':
                    # TEST MODE: ori - Send orientation commands to robot with fixed position
                    self.pub.publish(pose_msg)
                    rospy.loginfo(f"[{self.arm}] [{idx}] FIXED_POS=({pose_msg.pose.position.x:.3f},"
                                  f" {pose_msg.pose.position.y:.3f}, {pose_msg.pose.position.z:.3f}) |"
                                  f" CAM_ORI_TRANSFORMED=({q_base_ee[0]:.3f}, {q_base_ee[1]:.3f}, {q_base_ee[2]:.3f}, {q_base_ee[3]:.3f})")
                else:
                    # NORMAL MODE: Send commands to robot
                    self.pub.publish(pose_msg)
                    if self.ser:
                        write6(self.ser, self.hand_id, 'angleSet', hand_vals)
                        print(f"[DEBUG] Hand command sent: {hand_vals}")
                    else:
                        print(f"[DEBUG] No hand serial connection - hand command would be: {hand_vals}")

                    rospy.loginfo(f"[{self.arm}] [{idx}] pos=({pose_msg.pose.position.x:.3f},"
                                  f" {pose_msg.pose.position.y:.3f}, {pose_msg.pose.position.z:.3f}) |"
                                  f" hand={hand_vals}")

                rate.sleep()

        if self.test_mode == 'print':
            rospy.loginfo(f"[{self.arm}] TEST MODE: print complete - printed {idx} transformation results.")
        elif self.test_mode == 'ori':
            rospy.loginfo(f"[{self.arm}] TEST MODE: ori complete - sent {idx} orientation commands with fixed position.")
        else:
            rospy.loginfo(f"[{self.arm}] Trajectory and hand sequence complete.")

if __name__ == '__main__':
    try:
        # Test mode options:
        # False - normal operation
        # 'print' - print transformations without sending commands  
        # 'ori' - test only orientation with fixed position
        # 'translation' - test only position with fixed orientation
        # 'move_step' - incremental position movement toward CSV targets
        # 'translation_smooth' - automatic smooth translation with fixed orientation
        # 'ros_tele' - real-time teleop control via ROS topics
        # 'ros_hand_test' - hand-only control via ROS topics
        test_mode = 'ros_tele'  # Change this to False, 'print', 'ori', 'translation', 'move_step', 'translation_smooth', 'ros_tele', or 'ros_hand_test'

        # Check for command line argument
        import sys
        if len(sys.argv) > 1:
            if sys.argv[1] == '--print':
                test_mode = 'print'
            elif sys.argv[1] == '--ori':
                test_mode = 'ori'
            elif sys.argv[1] == '--translation':
                test_mode = 'translation'
            elif sys.argv[1] == '--move_step':
                test_mode = 'move_step'
            elif sys.argv[1] == '--translation_smooth':
                test_mode = 'translation_smooth'
            elif sys.argv[1] == '--ros_tele':
                test_mode = 'ros_tele'
            elif sys.argv[1] == '--ros_hand_test':
                test_mode = 'ros_hand_test'
            elif sys.argv[1] == '--normal':
                test_mode = False
            
        if test_mode == 'print':
            print("=== RUNNING IN TEST MODE: print ===")
            print("Will read CSV and print transformed coordinates without sending commands to robot")
            print("Camera frame -> Base frame -> EE frame transformations will be shown")
            print("To run normally, use --normal or change test_mode=False")
            print("=" * 50)
        elif test_mode == 'ori':
            print("=== RUNNING IN TEST MODE: ori ===")
            print("Will keep position fixed and only update orientation from camera frame")
            print("Commands will be sent to robot topic")
            print("To run normally, use --normal or change test_mode=False")
            print("=" * 50)
        elif test_mode == 'translation':
            print("=== RUNNING IN TEST MODE: translation ===")
            print("Will keep orientation fixed and move incrementally toward CSV position targets")
            print("Press '1' to take small steps toward each position target")
            print("Commands will be sent to robot topic")
            print("To run normally, use --normal or change test_mode=False")
            print("=" * 50)
        elif test_mode == 'translation_smooth':
            print("=== RUNNING IN TEST MODE: translation_smooth ===")
            print("Will execute automatic smooth translation movement with fixed orientation")
            print("Robot will automatically and smoothly move through all trajectory points")
            print("No user input required - runs completely automatically")
            print("Commands will be sent to robot topic")
            print("To run normally, use --normal or change test_mode=False")
            print("=" * 50)
        elif test_mode == 'move_step':
            print("=== RUNNING IN TEST MODE: move_step ===")
            print("Will execute incremental position AND orientation movement toward CSV targets")
            print("Both position and orientation will smoothly interpolate toward each target point")
            print("Press '1' to take small steps toward each target point")
            print("To run normally, use --normal or change test_mode=False")
            print("=" * 50)
        elif test_mode == 'ros_tele':
            print("=== RUNNING IN ROS_TELE MODE ===")
            print("Will subscribe to teleop topics for real-time robot control:")
            print("  - /teleop/right_wrist_pos (PoseStamped) for wrist position/orientation")
            print("  - /teleop/right_hand_joint (Float64MultiArray) for hand joint values")
            print("Same camera->base->EE transformations applied as normal mode")
            print("Robot will respond continuously to teleop commands")
            print("To run normally, use --normal or change test_mode=False")
            print("=" * 50)
        elif test_mode == 'ros_hand_test':
            print("=== RUNNING IN ROS_HAND_TEST MODE ===")
            print("Will subscribe to teleop hand topic for hand-only control:")
            print("  - /teleop/right_hand_joint (Float64MultiArray) for hand joint values")
            print("No wrist position commands will be sent")
            print("To run normally, use --normal or change test_mode=False")
            print("=" * 50)
        
        TrajFollow(test_mode=test_mode).run()
    except rospy.ROSInterruptException:
        pass
