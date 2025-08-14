#!/usr/bin/env python3
"""
Script for setting up automatic calibration trajectory points.
Combines keyboard control with ArUco detection and trajectory recording.
Records EE positions for future automatic calibration routines.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage
import threading
import time
import readchar
import os
import json
import numpy as np
import cv2
import cv2.aruco as aruco
from datetime import datetime
import transforms3d.euler as tfe
import transforms3d.quaternions as tfq


class AutoCalibrationTrajSetup(Node):
    def __init__(self, arm_choice):
        super().__init__('auto_calibration_traj_setup_node')
        
        self.lock = threading.Lock()
        self.current_pose = None
        self.trajectory_points = []  # List to store trajectory points
        self.message_received = False
        
        # ArUco detection variables
        self.aruco_img = None
        self.latest_tvec = None
        self.latest_rvec = None
        self.marker_detected = False
        
        # Arm selection
        if arm_choice == '1':
            self.arm = 'left'
        elif arm_choice == '2':
            self.arm = 'right'
        else:
            self.get_logger().error("Invalid input. Use '1' for left or '2' for right arm")
            exit()
        
        # Topics configuration
        self.pose_topic = f'/relaxed_ik/motion_control/pose_ee_arm_{self.arm}'
        # Alternative: self.pose_topic = f'/motion_control/pose_ee_arm_{self.arm}'
        self.target_topic = f'/motion_target/target_pose_arm_{self.arm}'
        
        # Create subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.pose_callback,
            10)
        
        self.image_sub = self.create_subscription(
            CompressedImage,
            "/zedm/zed_node/left/image_rect_color/compressed",
            self.image_callback,
            10)
        
        # Create publisher
        self.pub = self.create_publisher(PoseStamped, self.target_topic, 10)
        
        # Camera calibration parameters (ZED Mini)
        self.camera_matrix = np.array([[730.2571411132812, 0.0, 637.2598876953125],
                                       [0.0, 730.2571411132812, 346.41082763671875],
                                       [0.0, 0.0, 1.0]])
        
        self.dist_coeffs = np.zeros((5, 1))
        self.marker_length = 0.03813  # ArUco marker size in meters
        self.target_id = 23  # Target ArUco ID for calibration
        
        # ArUco detection setup
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
        self.aruco_params = aruco.DetectorParameters()
        
        # Transformation matrix from base to camera
        self.T_base_cam = np.array([
            [ 0.07868202, -0.82816949,  0.55492742,  0.14031877],
            [-0.99689748, -0.06417249,  0.04557752,  0.02810416],
            [-0.00213484, -0.55679188, -0.83064929,  0.46576442],
            [ 0.        ,  0.        ,  0.        ,  1.        ]
        ])
        
        # Create directory for saving trajectories
        self.traj_dir = "calibration_trajectories"
        os.makedirs(self.traj_dir, exist_ok=True)
        
        # Movement parameters
        self.step_size = 0.01  # 1cm for position
        self.angle_step = 0.1  # radians for rotation
        
        self.get_logger().info(f"Auto Calibration Trajectory Setup initialized for {self.arm} arm")
        self.get_logger().info(f"Waiting for EE pose on {self.pose_topic}...")
        
        # Wait for first message
        self.wait_for_first_message()
        time.sleep(0.5)
        self.get_logger().info("EE pose received. Ready for trajectory setup.")
    
    def wait_for_first_message(self, timeout=10.0):
        """Wait for first pose message with timeout"""
        start_time = time.time()
        while not self.message_received and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.message_received:
                break
        if not self.message_received:
            self.get_logger().warning("Timeout waiting for first message")
    
    def pose_callback(self, msg):
        """Callback for robot pose updates"""
        with self.lock:
            self.current_pose = msg
            self.message_received = True
    
    def image_callback(self, msg):
        """Process camera images and detect ArUco markers"""
        try:
            # Decode compressed image
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                return
            
            # ArUco detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            
            # Create display image
            img_display = img.copy()
            
            # Draw detected markers
            if ids is not None:
                img_display = aruco.drawDetectedMarkers(img_display, corners, ids)
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
                
                for i in range(len(ids)):
                    # Draw coordinate axes for each marker
                    cv2.drawFrameAxes(img_display, self.camera_matrix, self.dist_coeffs, 
                                     rvecs[i], tvecs[i], 0.03)
                    
                    # Check if target marker is detected
                    if ids[i][0] == self.target_id:
                        self.latest_tvec = tvecs[i].reshape(3)
                        self.latest_rvec = rvecs[i].reshape(3)
                        self.marker_detected = True
                        
                        # Highlight target marker
                        cv2.putText(img_display, f"Target ID: {self.target_id}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (0, 255, 0), 2)
            else:
                self.marker_detected = False
            
            # Project EE position into image
            with self.lock:
                if self.current_pose is not None:
                    pos = self.current_pose.pose.position
                    ee_base = np.array([pos.x, pos.y, pos.z, 1.0])
                    
                    # Transform EE to camera frame
                    ee_cam = np.dot(np.linalg.inv(self.T_base_cam), ee_base)
                    
                    if ee_cam[2] > 0:  # In front of camera
                        # Project to image plane
                        point_2d = self.camera_matrix @ ee_cam[:3]
                        u = int(point_2d[0] / ee_cam[2])
                        v = int(point_2d[1] / ee_cam[2])
                        
                        # Draw EE position
                        cv2.circle(img_display, (u, v), 10, (0, 0, 255), -1)
                        cv2.putText(img_display, "EE", (u + 15, v - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add trajectory info
            cv2.putText(img_display, f"Trajectory Points: {len(self.trajectory_points)}", 
                       (10, img_display.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(img_display, f"Arm: {self.arm.upper()}", 
                       (10, img_display.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Display image
            cv2.imshow("Auto Calibration Setup - ArUco Detection", img_display)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Image processing failed: {e}")
    
    def quaternion_from_euler(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion"""
        return tfe.euler2quat(roll, pitch, yaw)
    
    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions [x, y, z, w]"""
        # Convert to [w, x, y, z] for transforms3d
        q1_wxyz = [q1[3], q1[0], q1[1], q1[2]]
        q2_wxyz = [q2[3], q2[0], q2[1], q2[2]]
        
        # Multiply
        result_wxyz = tfq.qmult(q1_wxyz, q2_wxyz)
        
        # Convert back to [x, y, z, w]
        return [result_wxyz[1], result_wxyz[2], result_wxyz[3], result_wxyz[0]]
    
    def update_pose_by_key(self, key):
        """Update robot pose based on keyboard input"""
        with self.lock:
            if self.current_pose is None:
                self.get_logger().warning("No pose received yet.")
                return
            
            pose = self.current_pose.pose
            dx = dy = dz = 0.0
            drot = [0.0, 0.0, 0.0]
            
            # Position control
            if key == 'w': dx = self.step_size
            elif key == 's': dx = -self.step_size
            elif key == 'a': dy = self.step_size
            elif key == 'd': dy = -self.step_size
            elif key == 'q': dz = self.step_size
            elif key == 'e': dz = -self.step_size
            
            # Rotation control
            elif key == 'i': drot[0] = self.angle_step
            elif key == 'k': drot[0] = -self.angle_step
            elif key == 'j': drot[1] = self.angle_step
            elif key == 'l': drot[1] = -self.angle_step
            elif key == 'u': drot[2] = self.angle_step
            elif key == 'o': drot[2] = -self.angle_step
            else:
                return
            
            # Current orientation
            quat = [pose.orientation.x, pose.orientation.y, 
                   pose.orientation.z, pose.orientation.w]
            
            # Convert Euler to quaternion for delta rotation
            delta_q_wxyz = tfe.euler2quat(drot[0], drot[1], drot[2])
            delta_q = [delta_q_wxyz[1], delta_q_wxyz[2], delta_q_wxyz[3], delta_q_wxyz[0]]
            
            # Apply rotation
            new_quat = self.quaternion_multiply(quat, delta_q)
            
            # Create and publish new pose
            new_pose = PoseStamped()
            new_pose.header.stamp = self.get_clock().now().to_msg()
            new_pose.header.frame_id = "base_link"
            new_pose.pose.position.x = pose.position.x + dx
            new_pose.pose.position.y = pose.position.y + dy
            new_pose.pose.position.z = pose.position.z + dz
            new_pose.pose.orientation.x = new_quat[0]
            new_pose.pose.orientation.y = new_quat[1]
            new_pose.pose.orientation.z = new_quat[2]
            new_pose.pose.orientation.w = new_quat[3]
            
            self.pub.publish(new_pose)
    
    def record_trajectory_point(self):
        """Record current EE pose as a trajectory point"""
        with self.lock:
            if self.current_pose is None:
                self.get_logger().warning("No pose available to record")
                return
            
            # Extract pose information
            p = self.current_pose.pose.position
            o = self.current_pose.pose.orientation
            
            # Create trajectory point
            point = {
                'index': len(self.trajectory_points),
                'position': {
                    'x': p.x,
                    'y': p.y,
                    'z': p.z
                },
                'orientation': {
                    'x': o.x,
                    'y': o.y,
                    'z': o.z,
                    'w': o.w
                },
                'marker_detected': self.marker_detected,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            }
            
            # Add ArUco info if detected
            if self.marker_detected and self.latest_tvec is not None:
                point['aruco_tvec'] = self.latest_tvec.tolist()
                point['aruco_rvec'] = self.latest_rvec.tolist()
            
            self.trajectory_points.append(point)
            
            # Log recording
            self.get_logger().info(
                f"[Point {len(self.trajectory_points)}] Recorded at "
                f"pos({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) "
                f"ori({o.x:.3f}, {o.y:.3f}, {o.z:.3f}, {o.w:.3f}) "
                f"Marker: {self.marker_detected}"
            )
            
            print(f"\n✓ Point {len(self.trajectory_points)} recorded successfully!")
            print(f"  Position: [{p.x:.3f}, {p.y:.3f}, {p.z:.3f}]")
            print(f"  Marker detected: {self.marker_detected}")
    
    def save_trajectory(self):
        """Save recorded trajectory points to file"""
        if len(self.trajectory_points) == 0:
            self.get_logger().warning("No trajectory points to save")
            return
        
        # Generate filename with timestamp and arm
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"calibration_traj_{self.arm}_{timestamp}.json"
        filepath = os.path.join(self.traj_dir, filename)
        
        # Prepare trajectory data
        trajectory_data = {
            'arm': self.arm,
            'timestamp': timestamp,
            'total_points': len(self.trajectory_points),
            'camera_matrix': self.camera_matrix.tolist(),
            'marker_length': self.marker_length,
            'target_aruco_id': self.target_id,
            'points': self.trajectory_points
        }
        
        # Save to JSON file
        with open(filepath, 'w') as f:
            json.dump(trajectory_data, f, indent=2)
        
        self.get_logger().info(f"Trajectory saved to {filepath}")
        print(f"\n✓ Trajectory saved successfully!")
        print(f"  File: {filepath}")
        print(f"  Total points: {len(self.trajectory_points)}")
        
        # Also save a simplified CSV version for easy inspection
        csv_filename = f"calibration_traj_{self.arm}_{timestamp}.csv"
        csv_filepath = os.path.join(self.traj_dir, csv_filename)
        
        with open(csv_filepath, 'w') as f:
            # Write header
            f.write("index,x,y,z,qx,qy,qz,qw,marker_detected\n")
            
            # Write points
            for point in self.trajectory_points:
                f.write(f"{point['index']},"
                       f"{point['position']['x']:.6f},"
                       f"{point['position']['y']:.6f},"
                       f"{point['position']['z']:.6f},"
                       f"{point['orientation']['x']:.6f},"
                       f"{point['orientation']['y']:.6f},"
                       f"{point['orientation']['z']:.6f},"
                       f"{point['orientation']['w']:.6f},"
                       f"{point['marker_detected']}\n")
        
        print(f"  CSV file: {csv_filepath}")
    
    def preview_trajectory(self):
        """Preview recorded trajectory points"""
        if len(self.trajectory_points) == 0:
            print("\nNo trajectory points recorded yet.")
            return
        
        print(f"\n=== Trajectory Preview ({self.arm} arm) ===")
        print(f"Total points: {len(self.trajectory_points)}\n")
        
        for point in self.trajectory_points:
            print(f"Point {point['index'] + 1}:")
            print(f"  Position: [{point['position']['x']:.3f}, "
                  f"{point['position']['y']:.3f}, "
                  f"{point['position']['z']:.3f}]")
            print(f"  Marker detected: {point['marker_detected']}")
        print("=" * 40)
    
    def clear_trajectory(self):
        """Clear all recorded trajectory points"""
        self.trajectory_points = []
        self.get_logger().info("Trajectory cleared")
        print("\n✓ Trajectory cleared!")
    
    def run(self):
        """Main control loop"""
        time.sleep(1.0)
        
        print("\n" + "=" * 50)
        print(f"AUTO CALIBRATION TRAJECTORY SETUP - {self.arm.upper()} ARM")
        print("=" * 50)
        print("\n⌨️  Keyboard Controls:")
        print("  Movement:")
        print("    w/s: x+/x-  |  a/d: y+/y-  |  q/e: z+/z-")
        print("    i/k: roll   |  j/l: pitch  |  u/o: yaw")
        print("\n  Recording:")
        print("    r: Record current position as trajectory point")
        print("    p: Preview recorded trajectory")
        print("    c: Clear trajectory")
        print("    n: Save trajectory and quit")
        print("\n  Target ArUco ID: {}".format(self.target_id))
        print("=" * 50)
        print("\n💡 Tip: Position the arm where ArUco marker ID {} is visible".format(self.target_id))
        print("        for better calibration results.\n")
        
        while rclpy.ok():
            try:
                key = readchar.readkey()
                
                if key == 'n':
                    # Save and quit
                    if len(self.trajectory_points) > 0:
                        self.save_trajectory()
                    else:
                        print("\nNo points to save. Exiting...")
                    break
                    
                elif key == 'r':
                    # Record trajectory point
                    self.record_trajectory_point()
                    
                elif key == 'p':
                    # Preview trajectory
                    self.preview_trajectory()
                    
                elif key == 'c':
                    # Clear trajectory
                    confirm = input("\nClear all trajectory points? (y/n): ").strip().lower()
                    if confirm == 'y':
                        self.clear_trajectory()
                    
                else:
                    # Movement control
                    self.update_pose_by_key(key)
                
                # Process callbacks
                rclpy.spin_once(self, timeout_sec=0.01)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                if len(self.trajectory_points) > 0:
                    save = input("Save trajectory before exiting? (y/n): ").strip().lower()
                    if save == 'y':
                        self.save_trajectory()
                break


def main():
    try:
        print("\n" + "=" * 50)
        print("AUTO CALIBRATION TRAJECTORY SETUP")
        print("=" * 50)
        print("\nChoose arm to control:")
        print("  1 - Left Arm")
        print("  2 - Right Arm")
        choice = input("\nEnter 1 or 2: ").strip()
        
        # Initialize ROS2
        rclpy.init()
        
        # Create and run node
        node = AutoCalibrationTrajSetup(choice)
        
        try:
            node.run()
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
            
    except Exception as e:
        print(f"Error: {e}")
        if rclpy.ok():
            rclpy.shutdown()
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()