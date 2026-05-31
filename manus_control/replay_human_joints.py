#!/usr/bin/env python3

import csv
import argparse
import time
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np


class HumanJointReplayNode(Node):
    def __init__(self, hand_side='right'):
        super().__init__('human_joint_replay_node')

        self.hand_side = hand_side.lower()

        # Set up publisher based on hand side
        if self.hand_side == 'left':
            topic_name = '/dg5f_left_controller/joint_trajectory'
            self.joint_names = [
                "lj_dg_1_1", "lj_dg_1_2", "lj_dg_1_3", "lj_dg_1_4",  # Thumb
                "lj_dg_2_1", "lj_dg_2_2", "lj_dg_2_3", "lj_dg_2_4",  # Index
                "lj_dg_3_1", "lj_dg_3_2", "lj_dg_3_3", "lj_dg_3_4",  # Middle
                "lj_dg_4_1", "lj_dg_4_2", "lj_dg_4_3", "lj_dg_4_4",  # Ring
                "lj_dg_5_1", "lj_dg_5_2", "lj_dg_5_3", "lj_dg_5_4"   # Pinky
            ]
        else:  # right
            topic_name = '/dg5f_right_controller/joint_trajectory'
            self.joint_names = [
                "rj_dg_1_1", "rj_dg_1_2", "rj_dg_1_3", "rj_dg_1_4",  # Thumb
                "rj_dg_2_1", "rj_dg_2_2", "rj_dg_2_3", "rj_dg_2_4",  # Index
                "rj_dg_3_1", "rj_dg_3_2", "rj_dg_3_3", "rj_dg_3_4",  # Middle
                "rj_dg_4_1", "rj_dg_4_2", "rj_dg_4_3", "rj_dg_4_4",  # Ring
                "rj_dg_5_1", "rj_dg_5_2", "rj_dg_5_3", "rj_dg_5_4"   # Pinky
            ]

        # Create publisher for joint trajectory
        self.trajectory_publisher = self.create_publisher(
            JointTrajectory,
            topic_name,
            1
        )

        self.get_logger().info(f'Human Joint Replay Node initialized for {hand_side.upper()} hand')
        self.get_logger().info(f'Publishing to: {topic_name}')

    def convert_landmarks_to_joint_angles(self, landmarks, wrist_quat):
        """
        Convert 21 hand landmarks (3D positions) to 20 joint angles for DG-5F hand.

        Args:
            landmarks: List of 21 3D points (X, Y, Z) for hand landmarks
            wrist_quat: Wrist orientation as quaternion (qx, qy, qz, qw)

        Returns:
            List of 20 joint angles in radians for the DG-5F hand

        Hand landmark indices (MediaPipe format):
            0: Wrist
            1-4: Thumb (CMC, MCP, IP, TIP)
            5-8: Index (MCP, PIP, DIP, TIP)
            9-12: Middle (MCP, PIP, DIP, TIP)
            13-16: Ring (MCP, PIP, DIP, TIP)
            17-20: Pinky (MCP, PIP, DIP, TIP)

        DG-5F motors:
            0-3: Thumb (spread, MCP, PIP, DIP)
            4-7: Index (spread, MCP, PIP, DIP)
            8-11: Middle (spread, MCP, PIP, DIP)
            12-15: Ring (spread, MCP, PIP, DIP)
            16-19: Pinky (spread, MCP, PIP, DIP)
        """
        # TODO: Implement proper landmark to joint angle conversion
        # For now, using a simplified approach - calculate angles from 3D positions

        joint_angles = [0.0] * 20

        # Helper function to calculate angle between three points
        def calculate_angle(p1, p2, p3):
            """Calculate angle at p2 formed by p1-p2-p3"""
            v1 = np.array(p1) - np.array(p2)
            v2 = np.array(p3) - np.array(p2)

            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)

            return angle

        # Calculate spread angles (simplified - using X displacement)
        wrist = landmarks[0]

        # Thumb (landmarks 1-4)
        thumb_base = landmarks[1]
        thumb_mcp = landmarks[2]
        thumb_ip = landmarks[3]
        thumb_tip = landmarks[4]

        joint_angles[0] = calculate_angle(wrist, thumb_base, thumb_mcp)  # Thumb spread/abduction
        joint_angles[1] = calculate_angle(thumb_base, thumb_mcp, thumb_ip)  # Thumb MCP
        joint_angles[2] = calculate_angle(thumb_mcp, thumb_ip, thumb_tip)  # Thumb IP
        joint_angles[3] = joint_angles[2] * 0.8  # Thumb DIP (coupled with IP)

        # Index (landmarks 5-8)
        index_mcp = landmarks[5]
        index_pip = landmarks[6]
        index_dip = landmarks[7]
        index_tip = landmarks[8]

        joint_angles[4] = (index_mcp[0] - wrist[0]) * 0.1  # Index spread (simplified)
        joint_angles[5] = calculate_angle(wrist, index_mcp, index_pip)  # Index MCP
        joint_angles[6] = calculate_angle(index_mcp, index_pip, index_dip)  # Index PIP
        joint_angles[7] = calculate_angle(index_pip, index_dip, index_tip)  # Index DIP

        # Middle (landmarks 9-12)
        middle_mcp = landmarks[9]
        middle_pip = landmarks[10]
        middle_dip = landmarks[11]
        middle_tip = landmarks[12]

        joint_angles[8] = (middle_mcp[0] - wrist[0]) * 0.1  # Middle spread (simplified)
        joint_angles[9] = calculate_angle(wrist, middle_mcp, middle_pip)  # Middle MCP
        joint_angles[10] = calculate_angle(middle_mcp, middle_pip, middle_dip)  # Middle PIP
        joint_angles[11] = calculate_angle(middle_pip, middle_dip, middle_tip)  # Middle DIP

        # Ring (landmarks 13-16)
        ring_mcp = landmarks[13]
        ring_pip = landmarks[14]
        ring_dip = landmarks[15]
        ring_tip = landmarks[16]

        joint_angles[12] = (ring_mcp[0] - wrist[0]) * 0.1  # Ring spread (simplified)
        joint_angles[13] = calculate_angle(wrist, ring_mcp, ring_pip)  # Ring MCP
        joint_angles[14] = calculate_angle(ring_mcp, ring_pip, ring_dip)  # Ring PIP
        joint_angles[15] = calculate_angle(ring_pip, ring_dip, ring_tip)  # Ring DIP

        # Pinky (landmarks 17-20)
        pinky_mcp = landmarks[17]
        pinky_pip = landmarks[18]
        pinky_dip = landmarks[19]
        pinky_tip = landmarks[20]

        joint_angles[16] = (pinky_mcp[0] - wrist[0]) * 0.1  # Pinky spread (simplified)
        joint_angles[17] = calculate_angle(wrist, pinky_mcp, pinky_pip)  # Pinky MCP
        joint_angles[18] = calculate_angle(pinky_mcp, pinky_pip, pinky_dip)  # Pinky PIP
        joint_angles[19] = calculate_angle(pinky_pip, pinky_dip, pinky_tip)  # Pinky DIP

        return joint_angles

    def publish_trajectory(self, positions):
        """Publish joint trajectory message."""
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_names

        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = [float(pos) for pos in positions]
        point.time_from_start = Duration(sec=0, nanosec=30000000)  # 0.03 seconds

        # Add point to trajectory
        trajectory_msg.points.append(point)

        # Publish the message
        self.trajectory_publisher.publish(trajectory_msg)

        self.get_logger().debug(f'Published trajectory with {len(positions)} joint positions')

    def replay_from_csv(self, csv_file, playback_rate=1.0):
        """
        Read and replay joint data from CSV file.

        Args:
            csv_file: Path to the CSV file
            playback_rate: Speed multiplier (1.0 = normal speed, 2.0 = double speed)
        """
        self.get_logger().info(f'Reading data from: {csv_file}')
        self.get_logger().info(f'Playback rate: {playback_rate}x')

        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

                self.get_logger().info(f'Loaded {len(rows)} frames from CSV')

                # Determine column prefix based on hand side
                prefix = self.hand_side

                for idx, row in enumerate(rows):
                    # Extract 21 joint landmarks (X, Y, Z)
                    landmarks = []
                    for joint_idx in range(1, 22):  # Joints 1-21
                        x = float(row[f'{prefix}_{joint_idx}_X'])
                        y = float(row[f'{prefix}_{joint_idx}_Y'])
                        z = float(row[f'{prefix}_{joint_idx}_Z'])
                        landmarks.append([x, y, z])

                    # Extract wrist quaternion
                    wrist_quat = [
                        float(row[f'{prefix}_wrist_qx']),
                        float(row[f'{prefix}_wrist_qy']),
                        float(row[f'{prefix}_wrist_qz']),
                        float(row[f'{prefix}_wrist_qw'])
                    ]

                    # Convert landmarks to joint angles
                    joint_angles = self.convert_landmarks_to_joint_angles(landmarks, wrist_quat)

                    # Publish joint angles
                    self.publish_trajectory(joint_angles)

                    # Log progress every 100 frames
                    if (idx + 1) % 100 == 0:
                        self.get_logger().info(f'Processed {idx + 1}/{len(rows)} frames')

                    # Sleep to control playback rate (assuming 30 Hz recording)
                    time.sleep((1.0 / 30.0) / playback_rate)

                self.get_logger().info('Playback completed!')

        except FileNotFoundError:
            self.get_logger().error(f'CSV file not found: {csv_file}')
        except KeyError as e:
            self.get_logger().error(f'Missing column in CSV: {e}')
        except Exception as e:
            self.get_logger().error(f'Error reading CSV: {e}')


def main(args=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Replay human joint data on DG-5F robotic hand')
    parser.add_argument('--hand', type=str, default='right', choices=['left', 'right'],
                        help='Which hand to control (default: right)')
    parser.add_argument('--csv', type=str, default='human_joint.csv',
                        help='Path to CSV file (default: human_joint.csv)')
    parser.add_argument('--rate', type=float, default=1.0,
                        help='Playback rate multiplier (default: 1.0)')

    # Parse known args (ROS2 args will be handled separately)
    parsed_args, ros_args = parser.parse_known_args()

    # Initialize ROS2
    rclpy.init(args=ros_args)

    try:
        # Create node
        node = HumanJointReplayNode(hand_side=parsed_args.hand)

        # Start replay in a separate thread or just run it directly
        node.replay_from_csv(parsed_args.csv, playback_rate=parsed_args.rate)

        # Keep node alive briefly after playback
        time.sleep(1.0)

    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
