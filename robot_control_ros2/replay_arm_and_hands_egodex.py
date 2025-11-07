#!/usr/bin/env python3
'''
ROS2 node to control a robot arm and hand based on a CSV trajectory file.
This version uses the 'pynput' library for keyboard input and adds a
Low-Pass Filter to smooth out hand joint commands.

- 'n': Moves arm and hand one incremental step forward.
- 'c': Plays the entire arm and hand trajectory by publishing all points directly.
- 'r': Resets the trajectory progress to the beginning.
- 'q': Quits the program.
'''

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import threading
import os
import csv
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import time
from pynput import keyboard

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

EXPECTED_RIGHT_JOINT_ORDER = ["rj_dg_1_1", "rj_dg_1_2", "rj_dg_1_3", "rj_dg_1_4",
                             "rj_dg_2_1", "rj_dg_2_2", "rj_dg_2_3", "rj_dg_2_4",
                             "rj_dg_3_1", "rj_dg_3_2", "rj_dg_3_3", "rj_dg_3_4",
                             "rj_dg_4_1", "rj_dg_4_2", "rj_dg_4_3", "rj_dg_4_4",
                             "rj_dg_5_1", "rj_dg_5_2", "rj_dg_5_3", "rj_dg_5_4"]

EXPECTED_LEFT_JOINT_ORDER = ["lj_dg_1_1", "lj_dg_1_2", "lj_dg_1_3", "lj_dg_1_4",
                            "lj_dg_2_1", "lj_dg_2_2", "lj_dg_2_3", "lj_dg_2_4",
                            "lj_dg_3_1", "lj_dg_3_2", "lj_dg_3_3", "lj_dg_3_4",
                            "lj_dg_4_1", "lj_dg_4_2", "lj_dg_4_3", "lj_dg_4_4",
                            "lj_dg_5_1", "lj_dg_5_2", "lj_dg_5_3", "lj_dg_5_4"]

class CsvStepController(Node):
    def __init__(self, csv_filepath):
        super().__init__('csv_step_controller_node')
        
        # --- Configuration ---
        self.arm = 'right'
        self.csv_filepath = csv_filepath
        self.target_topic = f'/motion_target/target_pose_arm_{self.arm}'
        self.pose_topic = f'/relaxed_ik/motion_control/pose_ee_arm_{self.arm}'

        # Select correct joint order based on arm
        self.hand_joint_order = EXPECTED_LEFT_JOINT_ORDER if self.arm == 'left' else EXPECTED_RIGHT_JOINT_ORDER
        
        # Movement and playback settings
        self.POSITION_STEP_SIZE = 0.01
        self.ORIENTATION_STEP_SIZE = 0.03
        self.DIRECT_PLAYBACK_RATE = 20

        # --- NEW: Low-Pass Filter setting for hand joints ---
        # A smaller alpha means MORE smoothing. A larger alpha means LESS smoothing.
        # Good values are typically between 0.1 (very smooth) and 0.5 (less smooth).
        self.HAND_JOINT_LPF_ALPHA = 1

        # Thresholds for 'n' stepping
        self.POSITION_REACHED_THRESHOLD = 0.015
        self.ORIENTATION_REACHED_THRESHOLD = 0.1

        # --- State Variables ---
        self.lock = threading.Lock()
        self.current_pose = None
        self.message_received = False
        self.trajectory_points = []
        self.current_target_idx = 0
        self.keyboard_listener = None
        self.filtered_hand_joints = None # <-- NEW: Stores the smoothed joint state
        
        # --- Control Flags ---
        self.step_requested = False
        self.shutdown_requested = False
        self.direct_playback_requested = False

        # --- ROS2 Setup ---
        self.arm_pub = self.create_publisher(PoseStamped, self.target_topic, 10)
        self.subscription = self.create_subscription(
            PoseStamped, self.pose_topic, self.pose_callback, 10)

        # Hand controller topic based on arm selection
        hand_controller_topic = f'/dg5f_{self.arm}_controller/joint_trajectory'
        self.hand_pub = self.create_publisher(JointTrajectory, hand_controller_topic, 10)

        self.get_logger().info("CSV Step Controller for Arm and Hand initialized.")
    
    def load_trajectory_from_csv(self):
        """Loads trajectory points, including hand joints, from the CSV file based on the selected arm."""
        if not os.path.exists(self.csv_filepath):
            self.get_logger().error(f"Trajectory file not found: {self.csv_filepath}")
            return False

        # Use column names based on the selected arm
        POS_X_KEY = f'{self.arm}_wrist_x'
        POS_Y_KEY = f'{self.arm}_wrist_y'
        POS_Z_KEY = f'{self.arm}_wrist_z'
        ORI_X_KEY = f'{self.arm}_wrist_qx'
        ORI_Y_KEY = f'{self.arm}_wrist_qy'
        ORI_Z_KEY = f'{self.arm}_wrist_qz'
        ORI_W_KEY = f'{self.arm}_wrist_qw'

        with open(self.csv_filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Read hand joints based on the selected arm
                    hand_joints = [float(row[f'{self.arm}_hand_joint_{i+1}']) for i in range(20)]
                    point = {
                        'position': [float(row[POS_X_KEY]), float(row[POS_Y_KEY]), float(row[POS_Z_KEY])],
                        'orientation': [float(row[ORI_X_KEY]), float(row[ORI_Y_KEY]), float(row[ORI_Z_KEY]), float(row[ORI_W_KEY])],
                        'hand_joints': hand_joints
                    }
                    self.trajectory_points.append(point)
                except (KeyError, ValueError) as e:
                    self.get_logger().error(f"Error reading row from CSV: {e}.")
                    return False
        self.get_logger().info(f"Successfully loaded {len(self.trajectory_points)} points for {self.arm} arm.")
        return True

    def pose_callback(self, msg):
        # This function remains unchanged
        with self.lock:
            self.current_pose = msg
            if not self.message_received:
                self.message_received = True
                self.get_logger().info("First pose message received from robot.")

    def on_key_press(self, key):
        # This function remains unchanged
        try:
            if key.char == 'n':
                self.step_requested = True
                print("\r--> 'n': Step requested.              ")
            elif key.char == 'c':
                self.direct_playback_requested = True
                print("\r--> 'c': Direct playback requested.   ")
            elif key.char == 'r':
                self.current_target_idx = 0
                self.filtered_hand_joints = None # Reset filter on trajectory reset
                print("\r--> 'r': Trajectory reset to beginning.")
            elif key.char == 'q':
                print("\r--> 'q': Shutting down.               ")
                self.shutdown_requested = True
                return False
        except AttributeError:
            pass

    # --- NEW: Low-Pass Filter implementation ---
    def _apply_hand_joint_lpf(self, raw_joints):
        """Applies a simple low-pass filter to the hand joint commands."""
        raw_joints_np = np.array(raw_joints)
        
        if self.filtered_hand_joints is None:
            # If this is the first command, initialize the filter state
            self.filtered_hand_joints = raw_joints_np
        else:
            # Apply the exponential moving average formula
            self.filtered_hand_joints = (self.HAND_JOINT_LPF_ALPHA * raw_joints_np + 
                                        (1 - self.HAND_JOINT_LPF_ALPHA) * self.filtered_hand_joints)
        
        return self.filtered_hand_joints.tolist()

    def execute_step(self):
        """Calculates and publishes one INCREMENTAL step for arm and hand."""
        with self.lock:
            if self.current_pose is None: return
            position = self.current_pose.pose.position
            orientation = self.current_pose.pose.orientation
            current_pos = np.array([position.x, position.y, position.z])
            current_ori_quat = np.array([orientation.x, orientation.y, orientation.z, orientation.w])

        target_point = self.trajectory_points[self.current_target_idx]
        target_pos = np.array(target_point['position'])
        
        # --- Arm Logic (unchanged) ---
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)
        next_pos = current_pos + (direction / distance * self.POSITION_STEP_SIZE) if distance > self.POSITION_STEP_SIZE else target_pos
        current_rot, target_rot = R.from_quat(current_ori_quat), R.from_quat(np.array(target_point['orientation']))
        angular_dist = (current_rot.inv() * target_rot).magnitude()
        if angular_dist > self.ORIENTATION_STEP_SIZE:
            slerp = Slerp([0, 1], R.from_quat([current_ori_quat, np.array(target_point['orientation'])]))
            next_rot = slerp(self.ORIENTATION_STEP_SIZE / angular_dist)
        else:
            next_rot = target_rot
        next_ori_quat = next_rot.as_quat()

        now = self.get_clock().now().to_msg()
        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = 'base_link'
        pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = next_pos
        pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = next_ori_quat
        self.arm_pub.publish(pose_msg)
        
        # --- MODIFIED: Apply LPF to hand joints before publishing ---
        raw_hand_joints = target_point['hand_joints']
        smoothed_hand_joints = self._apply_hand_joint_lpf(raw_hand_joints)
        
        hand_traj = JointTrajectory()
        hand_traj.header.stamp = now
        hand_traj.joint_names = self.hand_joint_order
        traj_point = JointTrajectoryPoint(
            positions=smoothed_hand_joints,
            time_from_start=Duration(sec=0, nanosec=50000000)
        )
        hand_traj.points.append(traj_point)
        self.hand_pub.publish(hand_traj)

        self.get_logger().info(f"Step published for target #{self.current_target_idx + 1}. Dist: {distance:.3f}m")
        if distance < self.POSITION_REACHED_THRESHOLD and angular_dist < self.ORIENTATION_REACHED_THRESHOLD:
            self.get_logger().info(f"✅ Target {self.current_target_idx + 1} reached!")
            self.current_target_idx += 1

    def run_direct_playback(self):
        """Publishes all points for arm and hand directly, with LPF on hand."""
        self.get_logger().info(f"Starting direct playback of {len(self.trajectory_points)} points...")
        for idx, point in enumerate(self.trajectory_points):
            if self.shutdown_requested:
                self.get_logger().warn("Playback interrupted.")
                break
            
            now = self.get_clock().now().to_msg()
            pose_msg = PoseStamped()
            pose_msg.header.stamp = now
            pose_msg.header.frame_id = 'base_link'
            pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = point['position']
            pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = point['orientation']
            self.arm_pub.publish(pose_msg)
            
            # --- MODIFIED: Apply LPF to hand joints before publishing ---
            raw_hand_joints = point['hand_joints']
            smoothed_hand_joints = self._apply_hand_joint_lpf(raw_hand_joints)

            hand_traj = JointTrajectory()
            hand_traj.header.stamp = now
            hand_traj.joint_names = self.hand_joint_order
            traj_point = JointTrajectoryPoint(
                positions=smoothed_hand_joints,
                time_from_start=Duration(sec=0, nanosec=50000000)
            )
            hand_traj.points.append(traj_point)
            self.hand_pub.publish(hand_traj)

            time.sleep(1.0 / self.DIRECT_PLAYBACK_RATE)
        
        if not self.shutdown_requested:
            self.get_logger().info("✅ Direct playback finished.")
            self.current_target_idx = len(self.trajectory_points)

    def run(self):
        # This function remains unchanged
        if not self.load_trajectory_from_csv(): return
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
        self.keyboard_listener.start()
        self.get_logger().info("Waiting for first pose message...")
        while rclpy.ok() and not self.message_received and not self.shutdown_requested:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.shutdown_requested: return
        print("\n" + "="*50)
        print("🚀 Robot is ready. Controls:")
        print("   'n': Move one incremental step.")
        print("   'c': Play entire trajectory directly.")
        print("   'r': Reset trajectory to start.")
        print("   'q': Quit the program.")
        print(f"Loaded from: {os.path.basename(self.csv_filepath)}")
        print("="*50 + "\n", end="")
        while rclpy.ok() and not self.shutdown_requested:
            if self.step_requested:
                self.step_requested = False
                if self.current_target_idx < len(self.trajectory_points):
                    self.execute_step()
                else:
                    self.get_logger().info("End of trajectory. Press 'r' to reset.")
            if self.direct_playback_requested:
                self.direct_playback_requested = False
                self.run_direct_playback()
            rclpy.spin_once(self, timeout_sec=0.05)
        self.get_logger().info("Shutdown requested. Exiting...")
        if self.keyboard_listener:
            self.keyboard_listener.stop()

def main(args=None):
    rclpy.init(args=args)
    csv_file = '/home/irislab/r1pro_control/robot_control_ros2/egodex_robotcmd.csv'
    node = CsvStepController(csv_filepath=csv_file)
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()