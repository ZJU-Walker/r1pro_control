#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from datetime import datetime

class ContinuousPoseLogger(Node):
    """
    A ROS2 node that listens to two PoseStamped topics and logs every
    message received to a file.
    """
    def __init__(self):
        super().__init__('continuous_pose_logger')

        # --- Subscribers ---
        self.target_subscriber = self.create_subscription(
            PoseStamped,
            '/motion_targetsss/target_pose_arm_left',
            lambda msg: self.pose_callback(msg, '/motion_targetsss/target_pose_arm_left'),
            10)

        self.teleop_subscriber = self.create_subscription(
            PoseStamped,
            '/teleop/left_ee_raw_pose',
            lambda msg: self.pose_callback(msg, '/teleop/left_ee_raw_pose'),
            10)

        # --- Logging ---
        try:
            self.log_file = open('log_continuous.txt', 'a')
            self.get_logger().info("Successfully opened log_continuous.txt. Logging all incoming poses.")
        except IOError as e:
            self.get_logger().error(f"Failed to open log file: {e}")
            self.log_file = None

    def pose_callback(self, msg: PoseStamped, topic_name: str):
        """
        Generic callback to process and log every incoming PoseStamped message.
        """
        log_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Format the log message with the received pose data
        log_message = (
            f"[{log_time}] Topic: {topic_name}\n"
            f"  - Position (x,y,z):    {msg.pose.position.x:.4f}, {msg.pose.position.y:.4f}, {msg.pose.position.z:.4f}\n"
            f"  - Orientation (x,y,z,w): {msg.pose.orientation.x:.4f}, {msg.pose.orientation.y:.4f}, {msg.pose.orientation.z:.4f}, {msg.pose.orientation.w:.4f}\n"
            f"--------------------------------------------------\n"
        )

        # Print to console and write to file
        # We use print() for continuous streams to avoid cluttering the ROS console too much.
        print(log_message)
        
        if self.log_file:
            self.log_file.write(log_message)
            self.log_file.flush() # Ensure it's written immediately

    def destroy_node(self):
        """Custom cleanup."""
        self.get_logger().info("Shutting down and closing log file.")
        if self.log_file:
            self.log_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    pose_logger_node = ContinuousPoseLogger()
    try:
        rclpy.spin(pose_logger_node)
    except KeyboardInterrupt:
        pass
    finally:
        pose_logger_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()