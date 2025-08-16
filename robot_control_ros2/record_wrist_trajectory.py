#!/usr/bin/env python3
"""
ROS2 script to record wrist positions from teleop topics and save to CSV files.
Records both left and right wrist positions in camera frame.

Controls:
- 'r': Start recording
- 's': Stop recording  
- 'n': Save and exit

CSV format matches the requirements for right_arm_tele_ros2.py move_step mode.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import threading
import csv
import time
from datetime import datetime
import os
import sys
import select
import termios
import tty


class WristTrajectoryRecorder(Node):
    def __init__(self):
        super().__init__('wrist_trajectory_recorder')
        
        # Recording state
        self.is_recording = False
        self.left_trajectory = []
        self.right_trajectory = []
        self.lock = threading.Lock()
        
        # Latest wrist data
        self.left_wrist_data = None
        self.right_wrist_data = None
        
        # Create subscribers for both wrists
        self.left_sub = self.create_subscription(
            PoseStamped,
            '/teleop/left_wrist_pos',
            self.left_wrist_callback,
            10
        )
        
        self.right_sub = self.create_subscription(
            PoseStamped,
            '/teleop/right_wrist_pos',
            self.right_wrist_callback,
            10
        )
        
        # Timer for recording at fixed rate
        self.record_timer = self.create_timer(0.05, self.record_callback)  # 20Hz recording
        
        # Terminal settings for non-blocking keyboard input
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        
        self.get_logger().info("Wrist Trajectory Recorder initialized")
        self.get_logger().info("Controls:")
        self.get_logger().info("  'r' - Start recording")
        self.get_logger().info("  's' - Stop recording")
        self.get_logger().info("  'n' - Save and exit")
        self.get_logger().info("Waiting for wrist data...")
    
    def left_wrist_callback(self, msg):
        """Callback for left wrist position"""
        with self.lock:
            self.left_wrist_data = msg
    
    def right_wrist_callback(self, msg):
        """Callback for right wrist position"""
        with self.lock:
            self.right_wrist_data = msg
    
    def record_callback(self):
        """Timer callback to record data at fixed rate"""
        if not self.is_recording:
            return
        
        with self.lock:
            # Record left wrist if available
            if self.left_wrist_data is not None:
                left_point = {
                    'wrist_x': self.left_wrist_data.pose.position.x,
                    'wrist_y': self.left_wrist_data.pose.position.y,
                    'wrist_z': self.left_wrist_data.pose.position.z,
                    'qx': self.left_wrist_data.pose.orientation.x,
                    'qy': self.left_wrist_data.pose.orientation.y,
                    'qz': self.left_wrist_data.pose.orientation.z,
                    'qw': self.left_wrist_data.pose.orientation.w
                }
                self.left_trajectory.append(left_point)
            
            # Record right wrist if available
            if self.right_wrist_data is not None:
                right_point = {
                    'wrist_x': self.right_wrist_data.pose.position.x,
                    'wrist_y': self.right_wrist_data.pose.position.y,
                    'wrist_z': self.right_wrist_data.pose.position.z,
                    'qx': self.right_wrist_data.pose.orientation.x,
                    'qy': self.right_wrist_data.pose.orientation.y,
                    'qz': self.right_wrist_data.pose.orientation.z,
                    'qw': self.right_wrist_data.pose.orientation.w
                }
                self.right_trajectory.append(right_point)
    
    def start_recording(self):
        """Start recording trajectory"""
        if self.is_recording:
            self.get_logger().warn("Already recording!")
            return
        
        self.is_recording = True
        self.left_trajectory.clear()
        self.right_trajectory.clear()
        self.get_logger().info("🔴 RECORDING STARTED")
    
    def stop_recording(self):
        """Stop recording trajectory"""
        if not self.is_recording:
            self.get_logger().warn("Not currently recording!")
            return
        
        self.is_recording = False
        self.get_logger().info("⏹ RECORDING STOPPED")
        self.get_logger().info(f"  Left wrist: {len(self.left_trajectory)} points")
        self.get_logger().info(f"  Right wrist: {len(self.right_trajectory)} points")
    
    def save_trajectories(self):
        """Save recorded trajectories to CSV files"""
        if len(self.left_trajectory) == 0 and len(self.right_trajectory) == 0:
            self.get_logger().warn("No data to save!")
            return False
        
        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(__file__), 'recorded_trajectories')
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        
        # Save left trajectory if available
        if len(self.left_trajectory) > 0:
            left_filename = os.path.join(output_dir, f'left_wrist_{timestamp}.csv')
            with open(left_filename, 'w', newline='') as f:
                fieldnames = ['wrist_x', 'wrist_y', 'wrist_z', 'qx', 'qy', 'qz', 'qw']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.left_trajectory)
            self.get_logger().info(f"✅ Saved left wrist: {left_filename}")
            saved_files.append(left_filename)
        
        # Save right trajectory if available
        if len(self.right_trajectory) > 0:
            right_filename = os.path.join(output_dir, f'right_wrist_{timestamp}.csv')
            with open(right_filename, 'w', newline='') as f:
                fieldnames = ['wrist_x', 'wrist_y', 'wrist_z', 'qx', 'qy', 'qz', 'qw']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.right_trajectory)
            self.get_logger().info(f"✅ Saved right wrist: {right_filename}")
            saved_files.append(right_filename)
        
        return len(saved_files) > 0
    
    def check_keyboard(self):
        """Check for keyboard input (non-blocking)"""
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            return key
        return None
    
    def run(self):
        """Main run loop"""
        try:
            while rclpy.ok():
                # Check for keyboard input
                key = self.check_keyboard()
                
                if key:
                    if key.lower() == 'r':
                        self.start_recording()
                    elif key.lower() == 's':
                        self.stop_recording()
                    elif key.lower() == 'n':
                        # Stop recording if active
                        if self.is_recording:
                            self.stop_recording()
                        
                        # Save and exit
                        if self.save_trajectories():
                            self.get_logger().info("Trajectories saved successfully. Exiting...")
                        else:
                            self.get_logger().info("No trajectories to save. Exiting...")
                        break
                    elif key == 'q':
                        self.get_logger().info("Exiting without saving...")
                        break
                
                # Process ROS callbacks
                rclpy.spin_once(self, timeout_sec=0.01)
                
                # Display status periodically
                if hasattr(self, '_last_status_time'):
                    if time.time() - self._last_status_time > 2.0:
                        self.display_status()
                        self._last_status_time = time.time()
                else:
                    self._last_status_time = time.time()
                
        except KeyboardInterrupt:
            self.get_logger().info("\nKeyboard interrupt received")
            if self.is_recording:
                self.stop_recording()
            
            # Ask if user wants to save
            self.get_logger().info("Save recorded data? (y/n): ")
            # Restore terminal for input
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            response = input().strip().lower()
            
            if response == 'y':
                self.save_trajectories()
        
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def display_status(self):
        """Display current recording status"""
        with self.lock:
            left_receiving = self.left_wrist_data is not None
            right_receiving = self.right_wrist_data is not None
        
        if self.is_recording:
            status = "🔴 RECORDING"
            left_count = len(self.left_trajectory)
            right_count = len(self.right_trajectory)
            self.get_logger().info(
                f"{status} | Left: {left_count} pts {'✓' if left_receiving else '✗'} | "
                f"Right: {right_count} pts {'✓' if right_receiving else '✗'}"
            )
        else:
            if left_receiving or right_receiving:
                self.get_logger().info(
                    f"Ready | Left: {'✓' if left_receiving else '✗'} | "
                    f"Right: {'✓' if right_receiving else '✗'} | Press 'r' to start"
                )
    
    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'old_settings'):
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


def main():
    """Main function"""
    rclpy.init()
    
    recorder = WristTrajectoryRecorder()
    
    try:
        recorder.run()
    finally:
        recorder.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()