#!/usr/bin/env python3
'''
This script is used to follow a trajectory from a CSV file or real-time commands.
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
from std_msgs.msg import Float32MultiArray
import threading
import os
import math
import csv
import numpy as np
import tf
import serial
import time
from sensor_msgs.msg import JointState
import scipy.spatial.transform

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
    def __init__(self, mode='normal'):
        rospy.init_node('traj_follow_right_node', anonymous=True)
        self.lock = threading.Lock()
        self.current_pose = None
        self.mode = mode  # 'normal', 'move_step', or 'real_time'

        # Arm topics
        self.arm = 'right'
        self.pose_topic = f'/relaxed_ik/motion_control/pose_ee_arm_{self.arm}'
        self.target_topic = f'/motion_target/target_pose_arm_{self.arm}'
        rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_callback)
        self.pub = rospy.Publisher(self.target_topic, PoseStamped, queue_size=1)
        self.left_joint_state_pub = rospy.Publisher('/motion_target/target_joint_state_arm_left', JointState, queue_size=10)
        self.right_joint_state_pub = rospy.Publisher('/motion_target/target_joint_state_arm_right', JointState, queue_size=10)

        # Real-time mode subscriber
        if self.mode == 'real_time':
            self.inference_sub = rospy.Subscriber('/inference/right/command', Float32MultiArray, self.inference_callback)
            self.new_command_available = False
            self.latest_command = None
            rospy.loginfo("[REAL_TIME MODE] Subscribed to /inference/right/command")

        # Move step mode variables
        self.trajectory_points = []  # List of target poses for move_step mode
        self.current_target_idx = 0  # Current target point index
        self.step_size = 0.025  # Step size in meters for incremental movement
        self.position_threshold = 0.01  # Distance threshold to consider target reached
        self.min_movement_threshold = 0.025  # Minimum movement to send to robot (2cm - ALWAYS take 2cm steps)

        # Inspire Hand serial setup (right hand)
        serial_port = '/dev/ttyUSB1'
        baudrate = 115200
        self.hand_id = 2
        try:
            self.ser = open_serial(serial_port, baudrate)
            write6(self.ser, self.hand_id, 'speedSet', [1000]*6)
            write6(self.ser, self.hand_id, 'forceSet', [800]*6)
            rospy.loginfo(f"[{self.mode.upper()} MODE] Hand serial setup complete")
        except Exception as e:
            rospy.logwarn(f"Hand serial setup failed: {e}. Continuing without hand control.")
            self.ser = None

    def inference_callback(self, msg):
        """Callback for real-time inference commands"""
        with self.lock:
            # Parse the Float32MultiArray
            # Expected format: [wrist_x, wrist_y, wrist_z, qx, qy, qz, qw, hand_1, ..., hand_6]
            if len(msg.data) >= 13:  # 3 pos + 4 quat + 6 hand
                self.latest_command = {
                    'position': [msg.data[0], msg.data[1], msg.data[2]],
                    'orientation': [msg.data[3], msg.data[4], msg.data[5], msg.data[6]],  # qx, qy, qz, qw
                    'hand': [int(val) for val in msg.data[7:13]]
                }
                self.new_command_available = True
                rospy.loginfo(f"[REAL_TIME] Received new command: pos=({msg.data[0]:.3f}, {msg.data[1]:.3f}, {msg.data[2]:.3f})")
            else:
                rospy.logwarn(f"[REAL_TIME] Invalid command length: {len(msg.data)}, expected at least 13")

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
        """Wait for current pose"""
        rospy.loginfo("Waiting for current robot pose...")
        
        start_time = rospy.Time.now()
        
        while not rospy.is_shutdown():
            with self.lock:
                if self.current_pose is not None:
                    rospy.loginfo("Current robot pose received!")
                    return True
                    
            if (rospy.Time.now() - start_time).to_sec() > timeout:
                rospy.logerr("Timeout waiting for current pose")
                return False
                
            rospy.sleep(0.1)
        return False

    def transform_and_send_command(self, base_pos, base_quat, hand_vals):
        """Transform camera frame command to base frame and send to robot"""

        base_pt = np.array(base_pos)
        base_quat = np.array(base_quat)  # [qx, qy, qz, qw]
        
        # Build pose message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = 'base_link'
        
        # Set transformed position
        pose_msg.pose.position.x = base_pt[0]
        pose_msg.pose.position.y = base_pt[1]
        pose_msg.pose.position.z = base_pt[2]
        
        # Set transformed orientation
        pose_msg.pose.orientation.x = base_quat[0]
        pose_msg.pose.orientation.y = base_quat[1]
        pose_msg.pose.orientation.z = base_quat[2]
        pose_msg.pose.orientation.w = base_quat[3]
        
        # Send commands to robot
        self.pub.publish(pose_msg)
        if self.ser:
            write6(self.ser, self.hand_id, 'angleSet', hand_vals)
            
        rospy.loginfo(f"[{self.arm}] Sent: pos=({pose_msg.pose.position.x:.3f},"
                      f" {pose_msg.pose.position.y:.3f}, {pose_msg.pose.position.z:.3f}) |"
                      f" hand={hand_vals}")

    def load_trajectory_points(self, filepath):
        """Load trajectory points directly in base frame (EE poses from CSV)"""
        trajectory_points = []

        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            hand_cols = [col for col in reader.fieldnames if col.startswith('hand_')]

            for row in reader:
                # Position is already in base frame
                base_pos = [
                    float(row['wrist_x']),
                    float(row['wrist_y']),
                    float(row['wrist_z'])
                ]

                # Orientation is already in base frame
                base_ori = [
                    float(row['qx']),
                    float(row['qy']),
                    float(row['qz']),
                    float(row['qw'])
                ]

                # Hand values
                hand_vals = [int(float(row[col])) for col in hand_cols]

                point = {
                    'position': base_pos,
                    'orientation': base_ori,
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
            
        else:
            next_pos = current_pos  # Already at exact target
            
        # Calculate next orientation using SLERP (Spherical Linear Interpolation)
        next_ori = current_ori  # Default to current orientation
        orientation_reached = False  # Default to true if no orientation interpolation
        
        # if current_ori is not None and target_ori is not None:
        #     # Convert to numpy arrays for easier manipulation
        #     q_current = np.array(current_ori)
        #     q_target = np.array(target_ori)
            
        #     # Calculate quaternion difference/distance using dot product
        #     dot_product = np.dot(q_current, q_target)
            
        #     # Handle quaternion double cover (q and -q represent same rotation)
        #     if dot_product < 0:
        #         q_target = -q_target
        #         dot_product = -dot_product
            
        #     # Clamp dot product to avoid numerical issues
        #     dot_product = np.clip(dot_product, -1.0, 1.0)
            
        #     # Calculate angular distance
        #     theta = np.arccos(abs(dot_product))
        #     print("[DEBUG] Angular distance (theta):", theta)
            
        #     # Define orientation threshold (in radians, ~5.7 degrees)
        #     orientation_threshold = 0.1
        #     orientation_reached = theta <= orientation_threshold
            
        #     # Calculate interpolation factor based on step size
        #     # Use a fixed angular step size (in radians)
        #     angular_step_size = 0.15  # ~8.6 degrees per step

        #     if theta > orientation_threshold:
        #         # Calculate interpolation factor
        #         t = min(angular_step_size / theta, 1.0)
                
        #         # Perform SLERP
        #         if theta > 1e-6:  # Avoid division by zero
        #             sin_theta = np.sin(theta)
        #             weight_current = np.sin((1.0 - t) * theta) / sin_theta
        #             weight_target = np.sin(t * theta) / sin_theta
        #             next_ori = weight_current * q_current + weight_target * q_target
        #             # Normalize the result
        #             next_ori = next_ori / np.linalg.norm(next_ori)
        #             print("performing SLERP with t =", t)
        #         else:
        #             next_ori = q_target  # Quaternions are too close, just use target
        #             print("next_ori is close to target, using target orientation directly 1")
        #     else:
        #         next_ori = q_target  # Close enough, use target orientation
        #         print("Orientation already close enough, using target orientation directly 2")
        
        if current_ori is not None and target_ori is not None:
            # Convert quaternions to scipy Rotation objects
            R_current = scipy.spatial.transform.Rotation.from_quat(current_ori)
            R_target = scipy.spatial.transform.Rotation.from_quat(target_ori)

            # Compute relative rotation from current to target
            R_relative = R_target * R_current.inv()

            # Convert to axis-angle
            # Get rotation vector (axis * angle)
            rotvec = R_relative.as_rotvec()
            angle = np.linalg.norm(rotvec)

            print(f"[ANGLE] Total rotation angle: {angle:.4f} rad")

            if angle < np.deg2rad(2.0):  # 2 degrees threshold
                print("[ANGLE] angle is very small, using target orientation directly")
                next_ori = target_ori
                orientation_reached = True
            else:
                print("[ANGLE] angle is significant, calculating next orientation step")
                step_angle = np.deg2rad(8.0)  # ~0.1396 rad
                # orientation_reached = angle <= step_angle
                orientation_threshold = np.deg2rad(2.0)  # 2° threshold
                # orientation_reached = angle <= orientation_threshold
                step_rotvec = rotvec * min(step_angle / angle, 1.0)
                step_rotation = scipy.spatial.transform.Rotation.from_rotvec(step_rotvec)
                R_next = step_rotation * R_current
                next_ori = R_next.as_quat()
        
        # Target is reached when both position and orientation are close enough
        target_reached = position_reached and orientation_reached
        print(f"[DEBUG] Position reached: {position_reached}, Orientation reached: {orientation_reached}")
        print(f"[DEBUG] target_reached: {target_reached}")
        
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
        
        print("IN Ex: current_ori:", current_ori)
        print("IN Ex: target_ori:", target_ori)
        
        
        # Calculate next step for both position and orientation
        next_pos, next_ori, target_reached = self.calculate_step_toward_target(
            current_pos, target_pos, current_ori, target_ori)
        print("target_reached:", target_reached)
        # print("current_ori:", current_ori)
        # print("next_ori:", next_ori)
        # print("target_ori:", target_ori)

        # Calculate relative movement
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1] 
        dz = next_pos[2] - current_pos[2]
        
        # print(f"[DEBUG] Relative movement: dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}")

        # Create pose message
        with self.lock:
            if self.current_pose is None:
                print("No current pose available for relative movement!")
                return True
                
            pose = self.current_pose.pose
            
            new_pose = PoseStamped()
            new_pose.header.stamp = rospy.Time.now()
            new_pose.header.frame_id = "base_link"
            
            # Apply relative movement
            new_pose.pose.position.x = pose.position.x + dx
            new_pose.pose.position.y = pose.position.y + dy
            new_pose.pose.position.z = pose.position.z + dz
            
            # Use interpolated orientation
            new_pose.pose.orientation.x = next_ori[0]
            new_pose.pose.orientation.y = next_ori[1]
            new_pose.pose.orientation.z = next_ori[2]
            new_pose.pose.orientation.w = next_ori[3]
            
            # Send commands
            self.pub.publish(new_pose)
            
            # Add delay to ensure command is processed
            rospy.sleep(0.1)
            
            if self.ser:
                write6(self.ser, self.hand_id, 'angleSet', target_point['hand'])
                print(f"[DEBUG] Hand command sent: {target_point['hand']}")

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
    
    def execute_rotate_step(self):
        """Execute one rotation step only"""
        if self.current_target_idx >= len(self.trajectory_points):
            print("All trajectory points completed!")
            return False

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

        target_point = self.trajectory_points[self.current_target_idx]
        target_ori = target_point['orientation']

        # Keep position fixed, only rotate
        next_pos = current_pos
        _, next_ori, target_reached = self.calculate_step_toward_target(
            current_pos, current_pos, current_ori, target_ori)


        # Create pose message
        new_pose = PoseStamped()
        new_pose.header.stamp = rospy.Time.now()
        new_pose.header.frame_id = "base_link"
        new_pose.pose.position.x = current_pos[0]
        new_pose.pose.position.y = current_pos[1]
        new_pose.pose.position.z = current_pos[2]
        new_pose.pose.orientation.x = next_ori[0]
        new_pose.pose.orientation.y = next_ori[1]
        new_pose.pose.orientation.z = next_ori[2]
        new_pose.pose.orientation.w = next_ori[3]

        self.pub.publish(new_pose)
        rospy.sleep(0.1)

        if self.ser:
            write6(self.ser, self.hand_id, 'angleSet', target_point['hand'])

        if target_reached:
            print(f"✓ Point {self.current_target_idx + 1} rotation complete!")
            self.current_target_idx += 1
        else:
            print("[INFO] Continuing rotation...")

        return True


    def run_normal_mode(self):
        """Run normal trajectory following mode - just follow EE base frame CSV"""
        rate = rospy.Rate(20)
        rospy.loginfo(f"[{self.arm}] Starting NORMAL MODE - full trajectory playback...")
        self.send_vel_limit([4,4,4,4,4,4], [4,4,4,4,4,4])

        filepath = '/home/nvidia/ke/r1_pro_sdk_118/install/share/mobiman/scripts/right/right.csv'
        if not os.path.exists(filepath):
            rospy.logerr(f"Trajectory file not found: {filepath}")
            return

        # Load trajectory as in move_step
        trajectory_points = self.load_trajectory_points(filepath)
        rospy.loginfo(f"[{self.arm}] Loaded {len(trajectory_points)} trajectory points")

        for idx, point in enumerate(trajectory_points, start=1):
            pos = point['position']
            ori = point['orientation']
            hand = point['hand']

            self.transform_and_send_command(pos, ori, hand)

            rospy.loginfo(f"[{self.arm}] [{idx}] Sent pose=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) | hand={hand}")
            rate.sleep()

        rospy.loginfo(f"[{self.arm}] NORMAL MODE trajectory complete.")


    def run_move_step_mode(self):
        """Run move step mode with incremental movement"""
        rospy.loginfo(f"[{self.arm}] Starting MOVE_STEP MODE - incremental position AND orientation movement...")
        
        # Load trajectory points
        filepath = '/home/nvidia/ke/r1_pro_sdk_118/install/share/mobiman/scripts/right/right.csv'
        if not os.path.exists(filepath):
            rospy.logerr(f"Trajectory file not found: {filepath}")
            return
        
        self.trajectory_points = self.load_trajectory_points(filepath)
        rospy.loginfo(f"Loaded {len(self.trajectory_points)} trajectory points")
        
        # Wait for current pose
        if not self.wait_for_current_pose():
            return
        
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
        
        rospy.loginfo(f"[{self.arm}] MOVE_STEP MODE complete")
        
    def run_real_time_mode(self):
        """Run real-time mode - follow commands from inference topic"""
        rospy.loginfo(f"[{self.arm}] Starting REAL_TIME MODE - following /inference/right/command...")
        
        # Wait for current pose
        if not self.wait_for_current_pose():
            return
            
        # Set velocity limits
        self.send_vel_limit([4,4,4,4,4,4],[4,4,4,4,4,4])
        
        rospy.loginfo("Ready to receive real-time commands!")
        rospy.loginfo("Waiting for commands on /inference/right/command...")
        
        rate = rospy.Rate(20)  # 20 Hz control loop
        
        while not rospy.is_shutdown():
            with self.lock:
                if self.new_command_available and self.latest_command is not None:
                    # Get the command
                    command = self.latest_command
                    self.new_command_available = False
                    fixed_quat = [0.29, 0.73, -0.34, -0.498]
                    
                    # Transform and send the command
                    self.transform_and_send_command(
                        command['position'],
                        command['orientation'],
                        # fixed_quat,
                        command['hand']
                    )
            
            rate.sleep()
        
        rospy.loginfo(f"[{self.arm}] REAL_TIME MODE stopped")
            
    def run_translation_mode(self):
        """Run translation-only trajectory playback, using current orientation"""
        rospy.loginfo(f"[{self.arm}] Starting TRANSLATION MODE - follow positions with fixed current orientation...")

        # Load trajectory
        filepath = '/home/nvidia/ke/r1_pro_sdk_118/install/share/mobiman/scripts/right/right.csv'
        if not os.path.exists(filepath):
            rospy.logerr(f"Trajectory file not found: {filepath}")
            return

        self.trajectory_points = self.load_trajectory_points(filepath)
        rospy.loginfo(f"[{self.arm}] Loaded {len(self.trajectory_points)} trajectory points")

        # Wait for current pose to get the orientation
        if not self.wait_for_current_pose():
            return

        # Use current EE orientation for all points
        with self.lock:
            if self.current_pose is None:
                rospy.logerr("No current pose available!")
                return

            fixed_ori = [
                self.current_pose.pose.orientation.x,
                self.current_pose.pose.orientation.y,
                self.current_pose.pose.orientation.z,
                self.current_pose.pose.orientation.w
            ]

        # Set joint velocity limits
        self.send_vel_limit([4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4])

        rate = rospy.Rate(20)
        for idx, point in enumerate(self.trajectory_points, start=1):
            pos = point['position']
            hand = point['hand']

            self.transform_and_send_command(pos, fixed_ori, hand)

            rospy.loginfo(f"[{self.arm}] [{idx}] Sent TRANSLATION pose=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) | hand={hand}")
            rate.sleep()

        rospy.loginfo(f"[{self.arm}] TRANSLATION MODE complete.")

    def run_rotate_test_mode(self):
        """Run rotate-only mode with incremental orientation changes"""
        rospy.loginfo(f"[{self.arm}] Starting ROTATE_TEST MODE - orientation only...")

        # Load trajectory points
        filepath = '/home/nvidia/ke/r1_pro_sdk_118/install/share/mobiman/scripts/right/right.csv'
        if not os.path.exists(filepath):
            rospy.logerr(f"Trajectory file not found: {filepath}")
            return

        self.trajectory_points = self.load_trajectory_points(filepath)
        rospy.loginfo(f"Loaded {len(self.trajectory_points)} trajectory points")

        # Wait for current pose
        if not self.wait_for_current_pose():
            return

        rospy.loginfo("Ready for incremental rotation only!")
        rospy.loginfo("Press '1' + Enter to rotate toward current target")
        rospy.loginfo("Press 'q' + Enter to quit")
        rospy.loginfo("NOTE: Only orientation will change. Position remains fixed.")

        self.send_vel_limit([4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4])

        while not rospy.is_shutdown():
            try:
                if self.current_target_idx >= len(self.trajectory_points):
                    print("All trajectory points completed!")
                    break

                current_target = self.current_target_idx + 1
                total_targets = len(self.trajectory_points)
                print(f"\n[TARGET {current_target}/{total_targets}] Press '1' to rotate toward target, 'q' to quit: ", end='')

                user_input = input().strip()
                if user_input == '1':
                    if not self.execute_rotate_step():
                        break
                elif user_input.lower() == 'q':
                    print("Quitting rotate_test mode...")
                    break
                else:
                    print("Invalid input. Press '1' to rotate or 'q' to quit.")

            except KeyboardInterrupt:
                print("\nQuitting rotate_test mode...")
                break

        rospy.loginfo(f"[{self.arm}] ROTATE_TEST MODE complete")


    def run(self):
        if self.mode == 'move_step':
            self.run_move_step_mode()
        elif self.mode == 'real_time':
            self.run_real_time_mode()
        elif self.mode == 'translation':
            self.run_translation_mode()
        elif self.mode == 'rotate_test':
            self.run_rotate_test_mode()
        else:
            self.run_normal_mode()



if __name__ == '__main__':
    try:
        # Check for command line argument
        import sys
        mode = 'real_time'  # Default mode
        
        if len(sys.argv) > 1:
            if sys.argv[1] == '--move_step':
                mode = 'move_step'
                print("=== RUNNING IN MOVE_STEP MODE ===")
            elif sys.argv[1] == '--real_time':
                mode = 'real_time'
                print("=== RUNNING IN REAL_TIME MODE ===")
            elif sys.argv[1] == '--translation':
                mode = 'translation'
                print("=== RUNNING IN TRANSLATION MODE ===")
            elif sys.argv[1] == '--rotate_test':
                mode = 'rotate_test'
                print("=== RUNNING IN ROTATE_TEST MODE ===")
                print("Will perform orientation-only trajectory following.")
                print("=" * 50)
            elif sys.argv[1] == '--normal':
                mode = 'normal'


        
        TrajFollow(mode=mode).run()
    except rospy.ROSInterruptException:
        pass