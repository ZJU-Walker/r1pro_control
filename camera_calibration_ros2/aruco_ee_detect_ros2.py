#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
# Removed sensor_msgs imports as we're using ZED SDK directly
import threading
import time
import select
import sys
import termios
import tty
import os
import csv
import numpy as np
import cv2
import cv2.aruco as aruco
from datetime import datetime
import transforms3d.euler as tfe
import transforms3d.quaternions as tfq
import pyzed.sl as sl


### ---------- Controller ----------
class ArmKeyboardControl(Node):
    def __init__(self, arm_choice):
        super().__init__('arm_keyboard_control_node')
        
        self.lock = threading.Lock()
        self.current_pose = None
        self.recorded_data = []
        self.aruco_img = None
        self.detecting = False
        self.latest_tvec = None
        self.latest_rvec = None
        self.message_received = False
        
        # Latest image for display
        self.latest_image = None
        self.image_lock = threading.Lock()
        
        # Frame rate tracking
        self.last_frame_time = time.time()
        self.fps = 0.0
        self.frame_count = 0
        self.fps_update_time = time.time()
        self.image_resolution = (0, 0)
        
        # Initialize ZED camera
        self.init_zed_camera()

        if arm_choice == '1':
            self.arm = 'left'
        elif arm_choice == '2':
            self.arm = 'right'
        else:
            self.get_logger().error("Invalid input. Use '1' or '2'")
            exit()

        self.pose_topic = f'/relaxed_ik/motion_control/pose_ee_arm_{self.arm}'
        self.target_topic = f'/motion_target/target_pose_arm_{self.arm}'

        # Create subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.pose_callback,
            10)

        # Create publisher
        self.pub = self.create_publisher(PoseStamped, self.target_topic, 10)

        self.dist_coeffs = np.zeros((5, 1))
        self.marker_length = 0.048
        self.target_id = 23
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
        self.aruco_params = aruco.DetectorParameters()

        # Create directory for camera calibration data
        self.camera_cali_dir = "camera_cali"
        os.makedirs(self.camera_cali_dir, exist_ok=True)

        self.get_logger().info(f"Waiting for EE pose of {self.arm} arm...")
        self.wait_for_first_message()
        time.sleep(0.5)
        self.get_logger().info("EE pose received.")
        cv2.namedWindow("Aruco Detection + EE", cv2.WINDOW_NORMAL)
        
        # Start image processing thread
        self.image_thread = threading.Thread(target=self.image_processing_loop)
        self.image_thread.daemon = True
        self.image_thread.start()

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
        with self.lock:
            self.current_pose = msg
            self.message_received = True

    def init_zed_camera(self):
        init_params = sl.InitParameters()
        init_params.camera_fps = 30
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = sl.DEPTH_MODE.NONE

        self.zed_cam = sl.Camera()
        err = self.zed_cam.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f"Failed to open ZED camera: {err}")
            raise RuntimeError(f"ZED open failed: {err}")

        # >>> IMPORTANT: use rectified intrinsics for VIEW.LEFT
        cam_info = self.zed_cam.get_camera_information()
        calib = cam_info.camera_configuration.calibration_parameters  # rectified model
        left = calib.left_cam  # has fx, fy, cx, cy for this resolution

        self.camera_matrix = np.array([
            [left.fx, 0.0,   left.cx],
            [0.0,     left.fy, left.cy],
            [0.0,     0.0,   1.0]
        ], dtype=np.float64)

        # Rectified images => zero distortion
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        self.runtime_params = sl.RuntimeParameters()
        self.zed_image = sl.Mat()

        # Optional: log and sanity-check against incoming image size
        self.get_logger().info(
            f"Rectified K: fx={left.fx:.2f}, fy={left.fy:.2f}, cx={left.cx:.2f}, cy={left.cy:.2f}"
        )

    
    def image_processing_loop(self):
        """Continuously grab and process images from ZED camera"""
        while rclpy.ok():
            try:
                if self.zed_cam.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                    # Retrieve left image
                    self.zed_cam.retrieve_image(self.zed_image, sl.VIEW.LEFT)
                    
                    # Convert to OpenCV format (BGRA to BGR)
                    img_data = self.zed_image.get_data()
                    if img_data is not None:
                        # ZED returns BGRA, convert to BGR for OpenCV
                        img = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
                        self.process_image(img)
                else:
                    time.sleep(0.01)  # Small delay if grab fails
            except Exception as e:
                self.get_logger().error(f"Error in image processing loop: {e}")
                time.sleep(0.1)
    
    def process_image(self, img):
        """Process the image for ArUco detection and display"""
        try:
            
            # Update resolution and frame rate
            self.image_resolution = (img.shape[1], img.shape[0])  # (width, height)
            self.frame_count += 1
            current_time = time.time()
            
            # Calculate FPS every second
            if current_time - self.fps_update_time > 1.0:
                self.fps = self.frame_count / (current_time - self.fps_update_time)
                self.frame_count = 0
                self.fps_update_time = current_time

            # ArUco detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            img_with_detections = img.copy()

            if ids is not None:
                img_with_detections = aruco.drawDetectedMarkers(img_with_detections, corners, ids)
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
                for i in range(len(ids)):
                    cv2.drawFrameAxes(img_with_detections, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], 0.03)
                    if ids[i][0] == self.target_id and self.detecting:
                        self.latest_tvec = tvecs[i].reshape(3)
                        self.latest_rvec = rvecs[i].reshape(3)
                        print(f"[Aruco] ID {self.target_id} detected")
                        self.print_and_record()
                        self.save_aruco_result()
                        print("[INFO] current aruco detect over.")
                        self.detecting = False

            # === Project EE into image ===
            with self.lock:
                if self.current_pose is not None:
                    pos = self.current_pose.pose.position
                    ee_base = np.array([pos.x, pos.y, pos.z, 1.0])  # homogeneous

                    # T_base_cam from Open3D (cam_T_base = np.linalg.inv(T_base_cam))
                    T_base_cam = np.array([
                        [ 0.0047011 , -0.86116956,  0.50829607,  0.13993535],
                        [-0.99997691, -0.00654272, -0.00183634,  0.04082897],
                        [ 0.00490704, -0.5082757 , -0.86118043,  0.46200302],
                        [ 0.        ,  0.        ,  0.        ,  1.        ]
                    ])

                    # Compute EE in camera frame
                    # ee_cam = np.dot(np.linalg.inv(T_base_cam), ee_base)
                    T_cam_base = np.linalg.inv(T_base_cam)
                    ee_cam = T_cam_base @ ee_base


                    if ee_cam[2] > 0:  # in front of the camera
                        point_2d = self.camera_matrix @ ee_cam[:3]
                        u = int(point_2d[0] / ee_cam[2])
                        v = int(point_2d[1] / ee_cam[2])

                        cv2.circle(img_with_detections, (u, v), 8, (0, 0, 255), -1)
                        cv2.putText(img_with_detections, "EE", (u + 10, v - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Display FPS and resolution
            info_text = f"FPS: {self.fps:.1f} | Resolution: {self.image_resolution[0]}x{self.image_resolution[1]}"
            cv2.putText(img_with_detections, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display ArUco detection status
            if ids is not None:
                detection_text = f"ArUco detected: {len(ids)} marker(s)"
                cv2.putText(img_with_detections, detection_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Store the latest processed image
            with self.image_lock:
                # self.latest_image = img_with_detections.copy()
                self.latest_image = img_with_detections


        except Exception as e:
            self.get_logger().error(f"Aruco detection or projection failed: {e}")

    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions [x, y, z, w]"""
        # Convert to [w, x, y, z] for transforms3d
        q1_wxyz = [q1[3], q1[0], q1[1], q1[2]]
        q2_wxyz = [q2[3], q2[0], q2[1], q2[2]]
        
        # Multiply
        result_wxyz = tfq.qmult(q1_wxyz, q2_wxyz)
        
        # Convert back to [x, y, z, w]
        return [result_wxyz[1], result_wxyz[2], result_wxyz[3], result_wxyz[0]]

    def run(self):
        time.sleep(1.0)
        print(f"[{self.arm}] Controls:")
        print("  w/s/a/d/q/e - move xyz")
        print("  i/k/j/l/u/o - rotate rpy")
        print("  r - record pose")
        print("  f - detect ArUco & record")
        print("  n - save and quit")

        # Setup non-blocking keyboard input
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        try:
            while rclpy.ok():
                # Check for keyboard input (non-blocking)
                if select.select([sys.stdin], [], [], 0.001)[0]:
                    key = sys.stdin.read(1)
                    if key == 'n':
                        self.save_recorded_data()
                        break
                    elif key == 'r':
                        self.print_and_record()
                    elif key == 'f':
                        self.detecting = True
                        print("[INFO] Start detecting ArUco ID 23...")
                    else:
                        self.update_pose_by_key(key)
            
                
                # Display the latest image
                with self.image_lock:
                    if self.latest_image is not None:
                        cv2.imshow("Aruco Detection + EE", self.latest_image)
                        cv2.waitKey(1)
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            # Close ZED camera
            self.zed_cam.close()
            self.get_logger().info("ZED camera closed")

    def update_pose_by_key(self, key):
        with self.lock:
            if self.current_pose is None:
                self.get_logger().warning("No pose received yet.")
                return

            pose = self.current_pose.pose
            dx = dy = dz = 0.0
            drot = [0.0, 0.0, 0.0]

            if key == 'w': dx = 0.01
            elif key == 's': dx = -0.01
            elif key == 'a': dy = 0.01
            elif key == 'd': dy = -0.01
            elif key == 'q': dz = 0.01
            elif key == 'e': dz = -0.01
            elif key == 'i': drot[0] = 0.1
            elif key == 'k': drot[0] = -0.1
            elif key == 'j': drot[1] = 0.1
            elif key == 'l': drot[1] = -0.1
            elif key == 'u': drot[2] = 0.1
            elif key == 'o': drot[2] = -0.1
            else: return

            quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            
            # Convert Euler to quaternion for delta rotation
            delta_q_wxyz = tfe.euler2quat(drot[0], drot[1], drot[2])
            # Convert to [x, y, z, w]
            delta_q = [delta_q_wxyz[1], delta_q_wxyz[2], delta_q_wxyz[3], delta_q_wxyz[0]]
            
            # Apply rotation
            new_quat = self.quaternion_multiply(delta_q, quat)

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

    def print_and_record(self):
        with self.lock:
            if self.current_pose is None:
                return
            p = self.current_pose.pose.position
            o = self.current_pose.pose.orientation
            self.get_logger().info(f"[{self.arm}] Pose: pos({p.x:.3f}, {p.y:.3f}, {p.z:.3f})  ori({o.x:.3f}, {o.y:.3f}, {o.z:.3f}, {o.w:.3f})")
            self.recorded_data.append((p, o))

    def save_recorded_data(self):
        folder = os.path.expanduser(f'{self.arm}')
        os.makedirs(folder, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(folder, f'{timestamp}.txt')
        with open(path, 'w') as f:
            for p, o in self.recorded_data:
                f.write(f"{p.x} {p.y} {p.z} {o.x} {o.y} {o.z} {o.w}\n")

        self.get_logger().info(f"[{self.arm}] Recorded {len(self.recorded_data)} poses to {path}")

    def save_aruco_result(self):
        with self.lock:
            if self.latest_tvec is None or self.latest_rvec is None or self.current_pose is None:
                print("[WARN] Missing ArUco or EE pose, skip saving.")
                return

            try:
                rvec = self.latest_rvec
            except Exception as e:
                self.get_logger().error(f"Failed to correct ArUco rotation: {e}")
                return

            p = self.current_pose.pose.position
            o = self.current_pose.pose.orientation
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

            row = [
                timestamp,
                *self.latest_tvec,
                *self.latest_rvec,
                p.x, p.y, p.z,
                o.x, o.y, o.z, o.w
            ]

            today_str = datetime.now().strftime("%Y%m%d")
            file_path = os.path.join(self.camera_cali_dir, f"camera_{today_str}.csv")
            write_header = not os.path.exists(file_path)

            with open(file_path, "a", newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "timestamp",
                        "tvec_x", "tvec_y", "tvec_z",
                        "rvec_x", "rvec_y", "rvec_z",
                        "ee_x", "ee_y", "ee_z",
                        "ee_qx", "ee_qy", "ee_qz", "ee_qw"
                    ])

                writer.writerow(row)

            print(f"[INFO] Saved synchronized ArUco + EE pose to {file_path}")


def main():
    try:
        print("Choose arm to control:\n  1 - Left Arm\n  2 - Right Arm")
        choice = input("Enter 1 or 2: ").strip()
        
        rclpy.init()
        node = ArmKeyboardControl(choice)
        
        # --- in main(), after creating node ---
        from rclpy.executors import MultiThreadedExecutor
        exec = MultiThreadedExecutor()
        exec.add_node(node)
        spin_thread = threading.Thread(target=exec.spin, daemon=True)
        spin_thread.start()

        try:
            node.run()  # your keyboard/UI loop
        except KeyboardInterrupt:
            pass
        finally:
            exec.shutdown()          # stop executor
            cv2.destroyAllWindows()
            node.destroy_node()
            rclpy.shutdown()
                    
    except Exception as e:
        print(f"Error: {e}")
        if rclpy.ok():
            rclpy.shutdown()
    finally:
        pass


if __name__ == '__main__':
    main()