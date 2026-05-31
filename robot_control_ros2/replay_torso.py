#!/usr/bin/env python3
"""
Replay torso_joint_3 from CSV file with keyboard control.
- 'n': Move one step toward target (0.02 increment)
- 'c': Play entire trajectory
- 'r': Reset to beginning
- 'q': Quit
"""

import rclpy
from sensor_msgs.msg import JointState
from pynput import keyboard
import csv
import time

# Global state
current_joint_pos = [0.0] * 4
message_received = False
trajectory = []
current_idx = 0
step_requested = False
play_requested = False
reset_requested = False
shutdown_requested = False

STEP_SIZE = 0.02
REACHED_THRESHOLD = 0.01
PLAYBACK_RATE = 10


def feedback_callback(msg):
    global current_joint_pos, message_received
    if len(msg.position) >= 4:
        current_joint_pos = list(msg.position)
        message_received = True


def on_key_press(key):
    global step_requested, play_requested, reset_requested, shutdown_requested
    try:
        if key.char == 'n':
            step_requested = True
        elif key.char == 'c':
            play_requested = True
            print("\r--> 'c': Playing trajectory...")
        elif key.char == 'r':
            reset_requested = True
        elif key.char == 'q':
            print("\r--> 'q': Quitting...")
            shutdown_requested = True
            return False
    except AttributeError:
        pass


def load_csv(filepath):
    global trajectory
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            trajectory.append(float(row['torso_joint_3']))
    print(f"Loaded {len(trajectory)} points from CSV")


def publish_torso(pub, node, joint3_value):
    global current_joint_pos
    msg = JointState()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.position = current_joint_pos[:2] + [joint3_value] + [current_joint_pos[3]]
    msg.velocity = [1.5] * 4
    pub.publish(msg)


def step_toward_target(pub, node):
    """Move one step toward target, return True if target reached."""
    global current_idx, current_joint_pos

    if current_idx >= len(trajectory):
        return True

    current_val = current_joint_pos[2]
    target_val = trajectory[current_idx]
    diff = target_val - current_val

    if abs(diff) <= REACHED_THRESHOLD:
        # Target reached, move to next
        current_idx += 1
        print(f"Target {current_idx}/{len(trajectory)} reached")
        return True

    # Move by step size toward target
    if abs(diff) > STEP_SIZE:
        next_val = current_val + STEP_SIZE * (1 if diff > 0 else -1)
    else:
        next_val = target_val

    publish_torso(pub, node, next_val)
    print(f"Step: {current_val:.4f} -> {next_val:.4f} (target: {target_val:.4f}, idx: {current_idx + 1}/{len(trajectory)})")
    return False


def main():
    global current_idx, step_requested, play_requested, reset_requested, shutdown_requested, message_received

    csv_file = '/home/irislab/r1pro_control/robot_control_ros2/replay_files/torso_20260116_115053.csv'
    load_csv(csv_file)

    rclpy.init()
    node = rclpy.create_node('replay_torso_node')

    node.create_subscription(JointState, '/hdas/feedback_torso', feedback_callback, 10)
    pub = node.create_publisher(JointState, '/motion_target/target_joint_state_torso', 10)

    print("Waiting for torso feedback...")
    while not message_received and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
    print("Torso feedback received.")

    listener = keyboard.Listener(on_press=on_key_press)
    listener.start()

    print("\n" + "=" * 40)
    print("Torso Replay Controller")
    print("  'n': Step toward target (0.02)")
    print("  'c': Play all")
    print("  'r': Reset")
    print("  'q': Quit")
    print("=" * 40 + "\n")

    try:
        while rclpy.ok() and not shutdown_requested:
            if reset_requested:
                reset_requested = False
                current_idx = 0
                print("Reset to index 0")

            if step_requested:
                step_requested = False
                if current_idx < len(trajectory):
                    step_toward_target(pub, node)
                else:
                    print("End of trajectory. Press 'r' to reset.")

            if play_requested:
                play_requested = False
                print(f"Playing from index {current_idx}...")
                while current_idx < len(trajectory) and not shutdown_requested:
                    target_val = trajectory[current_idx]
                    publish_torso(pub, node, target_val)
                    print(f"\rPlaying {current_idx + 1}/{len(trajectory)}: {target_val:.4f}", end="")
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
