#!/usr/bin/env python3
"""
ROS2 node: publish only HAND joint trajectories from a CSV.
- No arm commands.
- No smoothing.
- Just: read CSV -> publish each row's 20 joint values -> sleep -> next row.

Intended for replaying recorded hand motion on the robot hand.
"""

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import csv
import os
import time


EXPECTED_RIGHT_JOINT_ORDER = [
    "rj_dg_1_1", "rj_dg_1_2", "rj_dg_1_3", "rj_dg_1_4",
    "rj_dg_2_1", "rj_dg_2_2", "rj_dg_2_3", "rj_dg_2_4",
    "rj_dg_3_1", "rj_dg_3_2", "rj_dg_3_3", "rj_dg_3_4",
    "rj_dg_4_1", "rj_dg_4_2", "rj_dg_4_3", "rj_dg_4_4",
    "rj_dg_5_1", "rj_dg_5_2", "rj_dg_5_3", "rj_dg_5_4"
]

EXPECTED_LEFT_JOINT_ORDER = [
    "lj_dg_1_1", "lj_dg_1_2", "lj_dg_1_3", "lj_dg_1_4",
    "lj_dg_2_1", "lj_dg_2_2", "lj_dg_2_3", "lj_dg_2_4",
    "lj_dg_3_1", "lj_dg_3_2", "lj_dg_3_3", "lj_dg_3_4",
    "lj_dg_4_1", "lj_dg_4_2", "lj_dg_4_3", "lj_dg_4_4",
    "lj_dg_5_1", "lj_dg_5_2", "lj_dg_5_3", "lj_dg_5_4"
]


class HandPlaybackNode(Node):
    def __init__(self, csv_filepath, arm='right'):
        """
        csv_filepath : path to the CSV with columns like
            right_hand_joint_1 ... right_hand_joint_20
            (or left_... if arm='left')
        arm          : 'right' or 'left'
        """
        super().__init__('hand_playback_node')

        assert arm in ['left', 'right'], "arm must be 'left' or 'right'"
        self.arm = arm
        self.csv_filepath = csv_filepath

        # choose correct joint order
        self.hand_joint_order = (
            EXPECTED_LEFT_JOINT_ORDER if self.arm == 'left'
            else EXPECTED_RIGHT_JOINT_ORDER
        )

        # publisher for that arm's hand controller
        hand_topic = f'/dg5f_{self.arm}_controller/joint_trajectory'
        self.hand_pub = self.create_publisher(JointTrajectory, hand_topic, 10)

        # playback rate [Hz]
        # e.g. 20 means 20 msgs/sec -> 0.05s between points
        self.playback_rate_hz = 20.0

        self.get_logger().info(f"HandPlaybackNode init for {self.arm} hand")
        self.get_logger().info(f"CSV: {self.csv_filepath}")
        self.get_logger().info(f"Publishing to: {hand_topic}")

        # load CSV rows into memory once
        self.hand_traj_points = self._load_hand_joints_from_csv()
        self.get_logger().info(f"Loaded {len(self.hand_traj_points)} hand points")

    def _load_hand_joints_from_csv(self):
        """
        Returns list of [20 floats] for each timestep.
        We do NOT reorder anything, we assume CSV
        column naming matches self.arm and is consistent.
        """
        if not os.path.exists(self.csv_filepath):
            self.get_logger().error(f"CSV not found: {self.csv_filepath}")
            return []

        joint_key_prefix = f'{self.arm}_hand_joint_'

        traj_points = []
        with open(self.csv_filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                try:
                    joints = [float(row[f'{joint_key_prefix}{i+1}']) for i in range(20)]
                except KeyError as e:
                    self.get_logger().error(
                        f"Missing column {e} in CSV at row {row_idx}. "
                        f"Expected columns like '{joint_key_prefix}1' ... '{joint_key_prefix}20'"
                    )
                    break
                except ValueError as e:
                    self.get_logger().error(
                        f"Bad numeric value in CSV at row {row_idx}: {e}"
                    )
                    break

                traj_points.append(joints)

        return traj_points

    def _publish_hand_position(self, joint_positions):
        """
        Publish one JointTrajectory message with a single point.
        No smoothing, raw values.
        """
        now = self.get_clock().now().to_msg()

        msg = JointTrajectory()
        msg.header.stamp = now
        msg.joint_names = self.hand_joint_order

        point = JointTrajectoryPoint()
        point.positions = joint_positions
        # small time_from_start so the controller treats it as near-immediate
        point.time_from_start = Duration(sec=0, nanosec=50_000_000)

        msg.points.append(point)

        self.hand_pub.publish(msg)

    def play_once_and_exit(self):
        """
        Iterate over all recorded hand poses and publish them
        at playback_rate_hz, then return.
        """
        if len(self.hand_traj_points) == 0:
            self.get_logger().warn("No points loaded. Nothing to play.")
            return

        period = 1.0 / self.playback_rate_hz

        self.get_logger().info("Starting hand-only playback...")
        for idx, joints in enumerate(self.hand_traj_points):
            self._publish_hand_position(joints)
            # optional debug print
            if idx % 10 == 0:
                self.get_logger().info(f"Published hand frame {idx+1}/{len(self.hand_traj_points)}")
            time.sleep(period)

        self.get_logger().info("✅ Hand playback finished.")


def main(args=None):
    rclpy.init(args=args)

    # >>> EDIT THESE <<<:
    csv_file = '/home/irislab/r1pro_control/robot_control_ros2/egodex_robotcmd.csv'
    arm = 'right'   # change to 'left' if you're driving the left hand

    node = HandPlaybackNode(csv_filepath=csv_file, arm=arm)

    try:
        node.play_once_and_exit()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
