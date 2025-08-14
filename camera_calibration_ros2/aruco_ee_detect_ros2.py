#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage
import threading
import time
import readchar
import os
import csv
import numpy as np
import cv2
import cv2.aruco as aruco
from datetime import datetime
import transforms3d.euler as tfe
import transforms3d.quaternions as tfq


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
        
        self.image_sub = self.create_subscription(
            CompressedImage,
            "/zedm/zed_node/left/image_rect_color/compressed",
            self.image_callback,
            10)
        # Alternative: "/zedm/zed_node/left_raw/image_raw_color/compressed"

        # Create publisher
        self.pub = self.create_publisher(PoseStamped, self.target_topic, 10)

        # Camera calibration parameters (zedmini)
        self.camera_matrix = np.array([[730.2571411132812, 0.0, 637.2598876953125],
                                       [0.0, 730.2571411132812, 346.41082763671875],
                                       [0.0, 0.0, 1.0]])
        # Alternative calibration:
        # self.camera_matrix = np.array([[528.4229125976562, 0.0, 635.5908203125],
        #                         [0.0, 528.4229125976562, 359.7709045410156],
        #                         [0.0, 0.0, 1.0]])

        self.dist_coeffs = np.zeros((5, 1))
        self.marker_length = 0.03813
        self.target_id = 23
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
        self.aruco_params = aruco.DetectorParameters()

        # Create directory for camera calibration data
        self.camera_cali_dir = "camera_cali"
        os.makedirs(self.camera_cali_dir, exist_ok=True)

        self.get_logger().info(f"Waiting for EE pose of {self.arm} arm...")
        self.wait_for_first_message()
        time.sleep(0.5)
        self.get_logger().info("EE pose received.")

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

    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                return

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
                        [ 0.07868202, -0.82816949,  0.55492742,  0.14031877],
                        [-0.99689748, -0.06417249,  0.04557752,  0.02810416],
                        [-0.00213484, -0.55679188, -0.83064929,  0.46576442],
                        [ 0.        ,  0.        ,  0.        ,  1.        ]
                    ])
                    # Alternative transformations:
                    # T_base_cam = np.array([
                    #     [ 0.0754638632, -0.801161710,  0.593670886,  0.154402045],
                    #     [-0.996995113, -0.0501787037, 0.0590156055, -0.000844939755],
                    #     [-0.0174914079, -0.596340517, -0.802540988,  0.473978148],
                    #     [0.0, 0.0, 0.0, 1.0]
                    # ])

                    # Compute EE in camera frame
                    ee_cam = np.dot(np.linalg.inv(T_base_cam), ee_base)

                    if ee_cam[2] > 0:  # in front of the camera
                        point_2d = self.camera_matrix @ ee_cam[:3]
                        u = int(point_2d[0] / ee_cam[2])
                        v = int(point_2d[1] / ee_cam[2])

                        cv2.circle(img_with_detections, (u, v), 8, (0, 0, 255), -1)
                        cv2.putText(img_with_detections, "EE", (u + 10, v - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Aruco Detection + EE", img_with_detections)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Aruco detection or projection failed: {e}")

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

    def run(self):
        time.sleep(1.0)
        print(f"[{self.arm}] Controls:")
        print("  w/s/a/d/q/e - move xyz")
        print("  i/k/j/l/u/o - rotate rpy")
        print("  r - record pose")
        print("  f - detect ArUco & record")
        print("  n - save and quit")

        while rclpy.ok():
            key = readchar.readkey()
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
            
            # Process callbacks
            rclpy.spin_once(self, timeout_sec=0.01)

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
                R_detected, _ = cv2.Rodrigues(self.latest_rvec)
                # Rotation correction (if needed)
                # R_y = cv2.Rodrigues(np.array([0, -np.pi/2, 0]))[0]
                # R_z = cv2.Rodrigues(np.array([0, 0, -np.pi/2]))[0]
                # R_correction = R_y @ R_z
                # R_corrected = R_correction @ R_detected
                R_corrected = R_detected
                rvec_corrected = cv2.Rodrigues(R_corrected)[0]
            except Exception as e:
                self.get_logger().error(f"Failed to correct ArUco rotation: {e}")
                return

            p = self.current_pose.pose.position
            o = self.current_pose.pose.orientation
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

            row = [
                timestamp,
                *self.latest_tvec,
                *rvec_corrected.flatten(),
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