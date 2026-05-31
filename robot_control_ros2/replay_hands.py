#!/usr/bin/env python3
'''
ROS2 node to control ONLY the hand based on a CSV trajectory file.
Keyboard:
- 'n': send next hand point (LPF-smoothed)
- 'c': play the whole hand trajectory
- 'r': reset to the beginning
- 'q': quit
'''

import rclpy
from rclpy.node import Node
import threading
import os
import csv
import numpy as np
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

class HandCsvReplayer(Node):
    def __init__(self, csv_filepath):
        super().__init__('hand_csv_replayer_node')

        # choose left or right
        self.arm = 'left'   # change to 'right' if needed
        self.csv_filepath = csv_filepath

        # correct joint order
        self.hand_joint_order = EXPECTED_LEFT_JOINT_ORDER if self.arm == 'left' else EXPECTED_RIGHT_JOINT_ORDER

        # playback settings
        self.DIRECT_PLAYBACK_RATE = 20  # Hz

        # LPF
        self.HAND_JOINT_LPF_ALPHA = 0.2
        self.filtered_hand_joints = None

        # state
        self.lock = threading.Lock()
        self.trajectory_points = []
        self.current_idx = 0
        self.step_requested = False
        self.shutdown_requested = False
        self.direct_playback_requested = False

        # publisher: ONLY hand
        hand_controller_topic = f'/dg5f_{self.arm}_controller/joint_trajectory'
        self.hand_pub = self.create_publisher(JointTrajectory, hand_controller_topic, 10)

        self.get_logger().info("Hand-only CSV replayer initialized.")

    def load_trajectory_from_csv(self):
        """Load CSV and extract ONLY hand joints for the selected arm."""
        if not os.path.exists(self.csv_filepath):
            self.get_logger().error(f"CSV not found: {self.csv_filepath}")
            return False

        with open(self.csv_filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    hand_joints = [float(row[f'{self.arm}_hand_{i}']) for i in range(20)]
                    # we still keep pos/ori in case CSV has them, but we ignore for publishing
                    self.trajectory_points.append({
                        'hand_joints': hand_joints
                    })
                except (KeyError, ValueError) as e:
                    self.get_logger().error(f"Error reading hand joints from CSV: {e}")
                    return False

        self.get_logger().info(f"Loaded {len(self.trajectory_points)} hand points for {self.arm}.")
        return True

    def on_key_press(self, key):
        try:
            if key.char == 'n':
                self.step_requested = True
                print("\r--> 'n': send next hand point")
            elif key.char == 'c':
                self.direct_playback_requested = True
                print("\r--> 'c': direct hand playback")
            elif key.char == 'r':
                self.current_idx = 0
                self.filtered_hand_joints = None
                print("\r--> 'r': reset to start")
            elif key.char == 'q':
                print("\r--> 'q': quitting")
                self.shutdown_requested = True
                return False
        except AttributeError:
            pass

    def _apply_hand_joint_lpf(self, raw_joints):
        raw_joints_np = np.array(raw_joints)
        if self.filtered_hand_joints is None:
            self.filtered_hand_joints = raw_joints_np
        else:
            self.filtered_hand_joints = (
                self.HAND_JOINT_LPF_ALPHA * raw_joints_np +
                (1 - self.HAND_JOINT_LPF_ALPHA) * self.filtered_hand_joints
            )
        return self.filtered_hand_joints.tolist()

    def publish_hand_point(self, hand_joints):
        now = self.get_clock().now().to_msg()
        smoothed = self._apply_hand_joint_lpf(hand_joints)

        jt = JointTrajectory()
        jt.header.stamp = now
        jt.joint_names = self.hand_joint_order

        pt = JointTrajectoryPoint(
            positions=smoothed,
            time_from_start=Duration(sec=0, nanosec=50000000)  # 50 ms
        )
        jt.points.append(pt)
        self.hand_pub.publish(jt)

    def execute_step(self):
        if self.current_idx >= len(self.trajectory_points):
            self.get_logger().info("End of hand trajectory. Press 'r' to restart.")
            return
        point = self.trajectory_points[self.current_idx]
        self.publish_hand_point(point['hand_joints'])
        self.get_logger().info(f"Hand point {self.current_idx+1}/{len(self.trajectory_points)} sent.")
        self.current_idx += 1

    def run_direct_playback(self):
        self.get_logger().info(f"Starting hand direct playback of {len(self.trajectory_points)} points...")
        for i, point in enumerate(self.trajectory_points):
            if self.shutdown_requested:
                self.get_logger().warn("Playback interrupted.")
                break
            self.publish_hand_point(point['hand_joints'])
            time.sleep(1.0 / self.DIRECT_PLAYBACK_RATE)
        if not self.shutdown_requested:
            self.get_logger().info("✅ Hand direct playback finished.")
            self.current_idx = len(self.trajectory_points)

    def run(self):
        if not self.load_trajectory_from_csv():
            return

        kb = keyboard.Listener(on_press=self.on_key_press)
        kb.start()

        print("\n" + "="*50)
        print("🖐 Hand-only replayer ready.")
        print("   'n': send next hand point")
        print("   'c': play all hand points")
        print("   'r': reset")
        print("   'q': quit")
        print(f"Loaded from: {os.path.basename(self.csv_filepath)}")
        print("="*50 + "\n", end="")

        while rclpy.ok() and not self.shutdown_requested:
            if self.step_requested:
                self.step_requested = False
                self.execute_step()
            if self.direct_playback_requested:
                self.direct_playback_requested = False
                self.run_direct_playback()
            rclpy.spin_once(self, timeout_sec=0.05)

        self.get_logger().info("Shutting down hand-only replayer...")
        kb.stop()


def main(args=None):
    rclpy.init(args=args)
    csv_file = '/home/irislab/r1pro_control/robot_control_ros2/replay_files/ee_hand_composition.csv'
    node = HandCsvReplayer(csv_filepath=csv_file)
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
