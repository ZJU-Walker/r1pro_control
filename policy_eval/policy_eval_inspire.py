#!/usr/bin/env python3
'''
Policy Evaluation Script for Robot Tasks

This script evaluates robot policies by:
1. Prompting for task name
2. Loading initial state from CSV files (initial_state/{task_name}.csv)
3. Resetting BOTH arms and hands to initial position when 'r' + Enter is pressed
4. Recording success/failure scores for multiple attempts
5. Logging statistics and results

Workflow:
1. Enter task name
2. Type 'r' + Enter to reset both arms to initial state
3. Perform the task manually
4. Enter score: 1 (success), 0 (fail), or 'r' to reset again
5. Repeat for all attempts
6. View statistics and logs

Controls:
- Type 'r' + Enter: Reset both arms and hands to initial state (works at any prompt)
- Type 'q' + Enter: Quit the program
'''

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
import threading
import os
import csv
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
from datetime import datetime
import json

# Global Configuration
TOTAL_ATTEMPTS = 15  # Number of evaluation attempts per task

# Inspire hand has 6 joints per hand
# Order: [pinky, ring, middle, index, thumb_bend, thumb_rotate]
# Values in range [0, 1000]
NUM_HAND_JOINTS = 6

# Torso has 4 joints, we use the last 2 (torso_joint_2, torso_joint_3)
NUM_TORSO_JOINTS = 4


class PolicyEvaluator(Node):
    def __init__(self):
        super().__init__('policy_evaluator_node')

        # --- State Variables ---
        self.lock = threading.Lock()
        self.current_pose_right = None
        self.current_pose_left = None
        self.message_received_right = False
        self.message_received_left = False

        # Task and evaluation state
        self.task_name = None
        self.initial_state_right = None
        self.initial_state_left = None
        self.initial_state_torso = None
        self.current_attempt = 0
        self.scores = []
        self.attempt_times = []
        self.task_start_time = None

        # Control Flags
        self.shutdown_requested = False

        # --- ROS2 Setup for RIGHT ARM ---
        self.arm_pub_right = self.create_publisher(
            PoseStamped, '/motion_target/target_pose_arm_right', 10)
        self.subscription_right = self.create_subscription(
            PoseStamped, '/relaxed_ik/motion_control/pose_ee_arm_right',
            self.pose_callback_right, 10)
        # Inspire hand uses Float32MultiArray
        self.hand_pub_right = self.create_publisher(
            Float32MultiArray, '/teleop/inspire_right_command', 10)

        # --- ROS2 Setup for LEFT ARM ---
        self.arm_pub_left = self.create_publisher(
            PoseStamped, '/motion_target/target_pose_arm_left', 10)
        self.subscription_left = self.create_subscription(
            PoseStamped, '/relaxed_ik/motion_control/pose_ee_arm_left',
            self.pose_callback_left, 10)
        # Inspire hand uses Float32MultiArray
        self.hand_pub_left = self.create_publisher(
            Float32MultiArray, '/teleop/inspire_left_command', 10)

        # --- ROS2 Setup for TORSO ---
        self.torso_pub = self.create_publisher(
            JointState, '/motion_targetssss/target_joint_state_torso', 10)

        self.get_logger().info("Policy Evaluator initialized for both arms and torso.")

    def pose_callback_right(self, msg):
        with self.lock:
            self.current_pose_right = msg
            if not self.message_received_right:
                self.message_received_right = True
                self.get_logger().info("First pose message received from right arm.")

    def pose_callback_left(self, msg):
        with self.lock:
            self.current_pose_left = msg
            if not self.message_received_left:
                self.message_received_left = True
                self.get_logger().info("First pose message received from left arm.")

    def load_initial_state(self, task_name):
        """Loads initial state for both arms and torso from CSV file."""
        csv_filepath = f'/home/irislab/r1pro_control/policy_eval/initial_state/{task_name}.csv'

        if not os.path.exists(csv_filepath):
            self.get_logger().error(f"Initial state file not found: {csv_filepath}")
            return False

        try:
            with open(csv_filepath, 'r') as f:
                reader = csv.DictReader(f)
                row = next(reader)  # Read first row only for initial state

                # Load RIGHT arm state (Inspire hand - 6 joints)
                right_hand_joints = [float(row[f'right_hand_{i}']) for i in range(NUM_HAND_JOINTS)]
                self.initial_state_right = {
                    'position': [float(row['right_pos_x']), float(row['right_pos_y']), float(row['right_pos_z'])],
                    'orientation': [float(row['right_ori_x']), float(row['right_ori_y']),
                                   float(row['right_ori_z']), float(row['right_ori_w'])],
                    'hand_joints': right_hand_joints
                }

                # Load LEFT arm state (Inspire hand - 6 joints)
                left_hand_joints = [float(row[f'left_hand_{i}']) for i in range(NUM_HAND_JOINTS)]
                self.initial_state_left = {
                    'position': [float(row['left_pos_x']), float(row['left_pos_y']), float(row['left_pos_z'])],
                    'orientation': [float(row['left_ori_x']), float(row['left_ori_y']),
                                   float(row['left_ori_z']), float(row['left_ori_w'])],
                    'hand_joints': left_hand_joints
                }

                # Load TORSO state (4 joints)
                torso_joints = [float(row[f'torso_joint_{i}']) for i in range(NUM_TORSO_JOINTS)]
                self.initial_state_torso = torso_joints

            self.get_logger().info(f"Successfully loaded initial state for both arms and torso - task: {task_name}")
            self.get_logger().info(f"  Left hand joints: {self.initial_state_left['hand_joints']}")
            self.get_logger().info(f"  Right hand joints: {self.initial_state_right['hand_joints']}")
            self.get_logger().info(f"  Torso joints: {self.initial_state_torso}")
            return True
        except Exception as e:
            self.get_logger().error(f"Error loading initial state: {e}")
            return False

    def reset_to_initial_state(self):
        """Resets both arms, hands, and torso to initial state."""
        if self.initial_state_right is None or self.initial_state_left is None:
            self.get_logger().warn("No initial state loaded. Cannot reset.")
            return

        self.get_logger().info("Resetting both arms, hands, and torso to initial state...")

        now = self.get_clock().now().to_msg()

        # --- Reset RIGHT arm ---
        pose_msg_right = PoseStamped()
        pose_msg_right.header.stamp = now
        pose_msg_right.header.frame_id = 'base_link'
        pose_msg_right.pose.position.x = self.initial_state_right['position'][0]
        pose_msg_right.pose.position.y = self.initial_state_right['position'][1]
        pose_msg_right.pose.position.z = self.initial_state_right['position'][2]
        pose_msg_right.pose.orientation.x = self.initial_state_right['orientation'][0]
        pose_msg_right.pose.orientation.y = self.initial_state_right['orientation'][1]
        pose_msg_right.pose.orientation.z = self.initial_state_right['orientation'][2]
        pose_msg_right.pose.orientation.w = self.initial_state_right['orientation'][3]
        self.arm_pub_right.publish(pose_msg_right)

        # Publish right hand joints (Inspire hand - Float32MultiArray)
        hand_msg_right = Float32MultiArray()
        hand_msg_right.data = [float(v) for v in self.initial_state_right['hand_joints']]
        self.hand_pub_right.publish(hand_msg_right)

        # --- Reset LEFT arm ---
        pose_msg_left = PoseStamped()
        pose_msg_left.header.stamp = now
        pose_msg_left.header.frame_id = 'base_link'
        pose_msg_left.pose.position.x = self.initial_state_left['position'][0]
        pose_msg_left.pose.position.y = self.initial_state_left['position'][1]
        pose_msg_left.pose.position.z = self.initial_state_left['position'][2]
        pose_msg_left.pose.orientation.x = self.initial_state_left['orientation'][0]
        pose_msg_left.pose.orientation.y = self.initial_state_left['orientation'][1]
        pose_msg_left.pose.orientation.z = self.initial_state_left['orientation'][2]
        pose_msg_left.pose.orientation.w = self.initial_state_left['orientation'][3]
        self.arm_pub_left.publish(pose_msg_left)

        # Publish left hand joints (Inspire hand - Float32MultiArray)
        hand_msg_left = Float32MultiArray()
        hand_msg_left.data = [float(v) for v in self.initial_state_left['hand_joints']]
        self.hand_pub_left.publish(hand_msg_left)

        # --- Reset TORSO ---
        if self.initial_state_torso is not None:
            torso_msg = JointState()
            torso_msg.header.stamp = now
            torso_msg.position = [float(v) for v in self.initial_state_torso]
            torso_msg.velocity = [1.5, 1.5, 1.5, 1.5]  # Same as teleop controller
            self.torso_pub.publish(torso_msg)

        time.sleep(2.0)  # Wait for robot to reach initial position
        self.get_logger().info("Reset complete for both arms, hands, and torso. Ready for evaluation.")

    def log_task_results(self):
        """Logs all evaluation results for the current task."""
        if not self.scores:
            return

        success_count = sum(self.scores)
        total_count = len(self.scores)
        success_rate = (success_count / total_count) * 100 if total_count > 0 else 0

        # Calculate statistics
        total_time = sum(self.attempt_times)
        avg_time = total_time / len(self.attempt_times) if self.attempt_times else 0

        # Create log entry
        log_entry = {
            'task_name': self.task_name,
            'timestamp': datetime.now().isoformat(),
            'total_attempts': total_count,
            'successes': success_count,
            'failures': total_count - success_count,
            'success_rate': f"{success_rate:.2f}%",
            'scores': self.scores,
            'attempt_times_seconds': self.attempt_times,
            'total_time_seconds': total_time,
            'average_time_per_attempt': avg_time,
            'arms_used': 'both'
        }

        # Print summary
        print("\n" + "="*60)
        print(f"EVALUATION RESULTS FOR: {self.task_name}")
        print("="*60)
        print(f"Total Attempts:    {total_count}")
        print(f"Successes:         {success_count}")
        print(f"Failures:          {total_count - success_count}")
        print(f"Success Rate:      {success_rate:.2f}%")
        print(f"Total Time:        {total_time:.2f}s")
        print(f"Average Time/Attempt: {avg_time:.2f}s")
        print(f"Scores:            {self.scores}")
        print("="*60 + "\n")

        # Save to log file
        log_dir = '/home/irislab/r1pro_control/policy_eval/logs'
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{log_dir}/{self.task_name}_{timestamp}.json"

        with open(log_filename, 'w') as f:
            json.dump(log_entry, f, indent=2)

        self.get_logger().info(f"Results saved to: {log_filename}")

        # Also append to master log
        master_log = f"{log_dir}/master_log.jsonl"
        with open(master_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def start_new_task(self):
        """Prompts for a new task and initializes evaluation."""
        print("\n" + "="*60)
        task_name = input("Enter task name (or 'q' to quit): ").strip()
        print("="*60 + "\n")

        if task_name.lower() == 'q':
            self.shutdown_requested = True
            return False

        if not task_name:
            print("Invalid task name. Please try again.")
            return False

        self.task_name = task_name
        if not self.load_initial_state(task_name):
            print(f"Failed to load initial state for task: {task_name}")
            return False

        # Reset evaluation state
        self.current_attempt = 0
        self.scores = []
        self.attempt_times = []
        self.task_start_time = time.time()

        print(f"\nTask '{task_name}' loaded successfully!")
        return True

    def get_score_input(self):
        """Gets score input from user. Returns score (0/1), 'r' for reset, or 'q' for quit."""
        while True:
            score_input = input(f"\nAttempt {self.current_attempt}/{TOTAL_ATTEMPTS} - Enter score (1=success, 0=fail) or 'r' to reset: ").strip().lower()

            if score_input == 'r':
                return 'r'
            elif score_input == 'q':
                return 'q'

            try:
                score = int(score_input)
                if score in [0, 1]:
                    return score
                else:
                    print("Invalid input. Please enter 1 for success, 0 for fail, or 'r' to reset.")
            except ValueError:
                print("Invalid input. Please enter 1 for success, 0 for fail, or 'r' to reset.")

    def run(self):
        """Main evaluation loop."""
        # Wait for first pose messages from both arms
        self.get_logger().info("Waiting for pose messages from both arms...")
        while rclpy.ok() and not self.shutdown_requested:
            if self.message_received_right and self.message_received_left:
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.shutdown_requested:
            return

        print("\n" + "="*60)
        print("POLICY EVALUATION SYSTEM")
        print("="*60)
        print("Controls:")
        print("  Type 'r' + Enter: Reset both arms to initial state")
        print("  Type 'q' + Enter: Quit program")
        print("="*60 + "\n")

        # Main loop
        while rclpy.ok() and not self.shutdown_requested:
            # Start new task if needed
            if self.task_name is None or self.current_attempt >= TOTAL_ATTEMPTS:
                if self.task_name is not None and self.current_attempt >= TOTAL_ATTEMPTS:
                    # Log results for completed task
                    self.log_task_results()

                # Start new task
                if not self.start_new_task():
                    break

            # Evaluation loop for current task
            while self.current_attempt < TOTAL_ATTEMPTS and rclpy.ok() and not self.shutdown_requested:
                # Prompt for reset
                print(f"\n--- Attempt {self.current_attempt + 1}/{TOTAL_ATTEMPTS} ---")
                reset_input = input("Type 'r' + Enter to reset both arms (or 'q' to quit): ").strip().lower()

                if reset_input == 'q':
                    self.shutdown_requested = True
                    break
                elif reset_input != 'r':
                    print("Invalid input. Please type 'r' to reset or 'q' to quit.")
                    continue

                # Start timing this attempt
                attempt_start_time = time.time()

                # Reset both arms to initial state
                print("\n[Resetting both arms to initial state...]")
                self.reset_to_initial_state()

                # Increment attempt counter BEFORE getting score
                self.current_attempt += 1

                print(f"\n[Attempt {self.current_attempt}/{TOTAL_ATTEMPTS}] Reset complete!")
                print("Perform the task, then enter the score below.")

                # Get score input (user can also type 'r' to reset again)
                while True:
                    score_or_command = self.get_score_input()

                    if score_or_command == 'q':
                        self.shutdown_requested = True
                        break
                    elif score_or_command == 'r':
                        # User wants to reset again - reset and continue this attempt
                        print("\n[Resetting both arms again...]")
                        self.reset_to_initial_state()
                        print("\n[Reset complete!] Perform the task, then enter the score.")
                        continue  # Ask for score again
                    else:
                        # Valid score received
                        score = score_or_command
                        self.scores.append(score)
                        break

                if self.shutdown_requested:
                    break

                # Record attempt duration
                attempt_duration = time.time() - attempt_start_time
                self.attempt_times.append(attempt_duration)

                result = "SUCCESS" if score == 1 else "FAIL"
                print(f"\n[Attempt {self.current_attempt}] Recorded: {result} (Duration: {attempt_duration:.2f}s)")

                if self.current_attempt < TOTAL_ATTEMPTS:
                    print(f"\nReady for next attempt ({self.current_attempt + 1}/{TOTAL_ATTEMPTS})")
                else:
                    print(f"\n{'='*60}")
                    print(f"ALL {TOTAL_ATTEMPTS} ATTEMPTS COMPLETED!")
                    print(f"{'='*60}")

                # Spin ROS to keep node alive
                rclpy.spin_once(self, timeout_sec=0.01)

            if self.shutdown_requested:
                break

        # Final cleanup
        if self.task_name is not None and self.scores:
            self.log_task_results()

        self.get_logger().info("Shutdown requested. Exiting...")


def main(args=None):
    rclpy.init(args=args)
    node = PolicyEvaluator()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
