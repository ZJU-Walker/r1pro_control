#!/usr/bin/env python3
"""
Replay arms and hands from CSV file with keyboard control.
- 'n': Move to next frame
- 'c': Play continuously
- 'r': Reset to beginning
- 'q': Quit
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from pynput import keyboard
import csv
import time
import numpy as np

from scipy.spatial.transform import Rotation as R

# Global state
trajectory = []
current_idx = 0
step_requested = False
play_requested = False
reset_requested = False
shutdown_requested = False

PLAYBACK_RATE = 10  # Hz

# Topic names
TOPIC_WRIST_LEFT_TARGET = '/motion_targetsss/target_pose_arm_left'
TOPIC_WRIST_RIGHT_TARGET = '/motion_target/target_pose_arm_right'
TOPIC_HAND_LEFT_CMD = '/teleopssss/inspire_left_command'
TOPIC_HAND_RIGHT_CMD = '/teleop/inspire_right_command'


def on_key_press(key):
    global step_requested, play_requested, reset_requested, shutdown_requested
    try:
        if key.char == 'n':
            step_requested = True
        elif key.char == 'c':
            play_requested = True
            print("\r--> 'c': Playing trajectory continuously...")
        elif key.char == 'r':
            reset_requested = True
        elif key.char == 'q':
            print("\r--> 'q': Quitting...")
            shutdown_requested = True
            return False
    except AttributeError:
        pass


def load_csv(filepath):
    """Load trajectory from CSV file."""
    global trajectory
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = {
                'frame_id': row['frame_id'],
                # Left arm pose
                'left_pos': np.array([
                    float(row['left_wrist_x']),
                    float(row['left_wrist_y']),
                    float(row['left_wrist_z'])
                ]),
                'left_ori': np.array([
                    float(row['left_wrist_qx']),
                    float(row['left_wrist_qy']),
                    float(row['left_wrist_qz']),
                    float(row['left_wrist_qw'])
                ]),
                # Right arm pose
                'right_pos': np.array([
                    float(row['right_wrist_x']),
                    float(row['right_wrist_y']),
                    float(row['right_wrist_z'])
                ]),
                'right_ori': np.array([
                    float(row['right_wrist_qx']),
                    float(row['right_wrist_qy']),
                    float(row['right_wrist_qz']),
                    float(row['right_wrist_qw'])
                ]),
                # Left hand joints (6 values)
                'left_hand': np.array([
                    float(row['left_hand_0']),
                    float(row['left_hand_1']),
                    float(row['left_hand_2']),
                    float(row['left_hand_3']),
                    float(row['left_hand_4']),
                    float(row['left_hand_5'])
                ]),
                # Right hand joints (6 values)
                'right_hand': np.array([
                    float(row['right_hand_0']),
                    float(row['right_hand_1']),
                    float(row['right_hand_2']),
                    float(row['right_hand_3']),
                    float(row['right_hand_4']),
                    float(row['right_hand_5'])
                ])
            }
            trajectory.append(frame)
    print(f"Loaded {len(trajectory)} frames from CSV")


def publish_frame(node, idx, wrist_pub_L, wrist_pub_R, hand_pub_L, hand_pub_R):
    """Publish a single frame to all controllers."""
    if idx >= len(trajectory):
        return

    frame = trajectory[idx]
    stamp = node.get_clock().now().to_msg()

    # Publish left arm pose
    pose_L = PoseStamped()
    pose_L.header.stamp = stamp
    pose_L.header.frame_id = "base_link"
    pose_L.pose.position.x = float(frame['left_pos'][0])
    pose_L.pose.position.y = float(frame['left_pos'][1])
    pose_L.pose.position.z = float(frame['left_pos'][2])

    pose_L.pose.orientation.x = float(frame['left_ori'][0])
    pose_L.pose.orientation.y = float(frame['left_ori'][1])
    pose_L.pose.orientation.z = float(frame['left_ori'][2])
    pose_L.pose.orientation.w = float(frame['left_ori'][3])
    wrist_pub_L.publish(pose_L)

    # Publish right arm pose
    pose_R = PoseStamped()
    pose_R.header.stamp = stamp
    pose_R.header.frame_id = "base_link"
    pose_R.pose.position.x = float(frame['right_pos'][0])
    pose_R.pose.position.y = float(frame['right_pos'][1])
    pose_R.pose.position.z = float(frame['right_pos'][2])

    pose_R.pose.position.x = pose_R.pose.position.x - 0.05  # Adjust right arm x-position for better visualization
    pose_R.pose.position.y = pose_R.pose.position.y - 0.08  # Adjust left arm y-position for better visualization
    pose_R.pose.position.z = pose_R.pose.position.z + 0.0   # Adjust left arm z-position for better visualization

    # pose_R.pose.orientation.x = float(frame['right_ori'][0])
    # pose_R.pose.orientation.y = float(frame['right_ori'][1])
    # pose_R.pose.orientation.z = float(frame['right_ori'][2])
    # pose_R.pose.orientation.w = float(frame['right_ori'][3])

    right_quat = np.array([
        float(frame['right_ori'][0]),
        float(frame['right_ori'][1]),
        float(frame['right_ori'][2]),
        float(frame['right_ori'][3])
    ], dtype=np.float64)

    R_base_ee = R.from_quat(right_quat).as_matrix()

    theta = np.deg2rad(30.0)
    c = np.cos(theta)
    s = np.sin(theta)

    R_ee_eenew = np.array([
        [1.0, 0.0, 0.0],
        [0.0,   c,  -s],
        [0.0,   s,   c]
    ], dtype=np.float64)

    # local rotation
    R_base_eenew = R_base_ee @ R_ee_eenew

    quat_new = R.from_matrix(R_base_eenew).as_quat()

    pose_R.pose.orientation.x = float(quat_new[0])
    pose_R.pose.orientation.y = float(quat_new[1])
    pose_R.pose.orientation.z = float(quat_new[2])
    pose_R.pose.orientation.w = float(quat_new[3])

    wrist_pub_R.publish(pose_R)

    # Publish left hand joints
    hand_msg_L = Float32MultiArray()
    hand_msg_L.data = [float(v) for v in frame['left_hand'].tolist()]
    hand_pub_L.publish(hand_msg_L)

    # Publish right hand joints
    hand_msg_R = Float32MultiArray()
    hand_msg_R.data = [float(v) for v in frame['right_hand'].tolist()]
    hand_pub_R.publish(hand_msg_R)


def main():
    global current_idx, step_requested, play_requested, reset_requested, shutdown_requested

    csv_file = '/home/irislab/r1pro_control/robot_control_ros2/replay_files/teleop_data.csv'
    load_csv(csv_file)

    rclpy.init()
    node = rclpy.create_node('replay_arms_and_hands_node')

    # Publishers
    wrist_pub_L = node.create_publisher(PoseStamped, TOPIC_WRIST_LEFT_TARGET, 10)
    wrist_pub_R = node.create_publisher(PoseStamped, TOPIC_WRIST_RIGHT_TARGET, 10)
    hand_pub_L = node.create_publisher(Float32MultiArray, TOPIC_HAND_LEFT_CMD, 10)
    hand_pub_R = node.create_publisher(Float32MultiArray, TOPIC_HAND_RIGHT_CMD, 10)

    listener = keyboard.Listener(on_press=on_key_press)
    listener.start()

    print("\n" + "=" * 50)
    print("Arms + Hands Replay Controller")
    print("  'n': Next frame")
    print("  'c': Play continuously")
    print("  'r': Reset to beginning")
    print("  'q': Quit")
    print("=" * 50 + "\n")

    try:
        while rclpy.ok() and not shutdown_requested:
            if reset_requested:
                reset_requested = False
                current_idx = 0
                print("Reset to frame 0")

            if step_requested:
                step_requested = False
                if current_idx < len(trajectory):
                    publish_frame(node, current_idx, wrist_pub_L, wrist_pub_R, hand_pub_L, hand_pub_R)
                    current_idx += 1
                    print(f"[STEP] Frame {current_idx}/{len(trajectory)}")
                else:
                    print("End of trajectory. Press 'r' to reset.")

            if play_requested:
                play_requested = False
                print(f"Playing from frame {current_idx}...")
                while current_idx < len(trajectory) and not shutdown_requested:
                    publish_frame(node, current_idx, wrist_pub_L, wrist_pub_R, hand_pub_L, hand_pub_R)
                    print(f"\rPlaying frame {current_idx + 1}/{len(trajectory)}", end="")
                    current_idx += 1
                    rclpy.spin_once(node, timeout_sec=0.01)
                    time.sleep(1.0 / PLAYBACK_RATE)
                print("\nPlayback done.")

            rclpy.spin_once(node, timeout_sec=0.05)

    finally:
        listener.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
