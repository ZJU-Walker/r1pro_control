#!/usr/bin/env python3
import time
import cv2
import cv2.aruco as aruco
import pyzed.sl as sl
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
import csv
from rclpy.node import Node
import threading
from typing import Dict, Tuple, Optional

# --- Configuration ---
MARKER_SIZE = 0.048  # 4.8 cm

# Note: This transform is for the ZED's left camera sensor relative to the robot base.
# T_BASE_CAM_LEFT = np.array([
#     [ 0.00692993, -0.87310148,  0.48748926,  0.14062141],
#     [-0.99995006, -0.00956093, -0.00290894,  0.03612369],
#     [ 0.00720065, -0.48744476, -0.87312414,  0.46063114],
#     [ 0.        ,  0.        ,  0.        ,  1.        ]
# ])
T_BASE_CAM_LEFT =  np.array([
    [0.02169645, -0.70143451,  0.71240361,  0.14534308],
    [-0.99949077,  0.00145864,  0.03187594,  0.03543434],
    [-0.02339802, -0.71273242, -0.70104567,  0.47400349],
    [0.0,          0.0,          0.0,          1.0]
])

# T_BASE_CAM_RIGHT = np.array([
#     [-0.00334115, -0.8768872 ,  0.48068458,  0.14700305],
#     [-0.99996141,  0.0068351 ,  0.00551836, -0.02680847],
#     [-0.00812451, -0.4806476 , -0.87687621,  0.46483729],
#     [ 0.        ,  0.        ,  0.        ,  1.        ]
# ])

T_BASE_CAM_RIGHT =  np.array([
    [ 2.37879823e-02, -6.94501664e-01,  7.19097748e-01,  1.39044362e-01],
    [-9.99432223e-01,  6.47597050e-04,  3.36869826e-02, -3.04603840e-02],
    [-2.38613510e-02, -7.19490806e-01, -6.94091937e-01,  4.70721741e-01],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
])

class StereoCalibrator:
    """
    A class to handle the stereo calibration data collection process.
    """
    def __init__(self, node: Node, arm_name: str, marker_id: int):
        self.node = node
        self.arm_name = arm_name
        self.target_marker_id = marker_id
        
        self.latest_ee_pose: Optional[PoseStamped.Pose] = None
        self.zed = sl.Camera()
        self.aruco_detector: Optional[aruco.ArucoDetector] = None
        
        self.cam_params: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.csv_files: Dict[str, object] = {}
        self.csv_writers: Dict[str, object] = {}
        
        self.T_CAM_BASE_LEFT = np.linalg.inv(T_BASE_CAM_LEFT)
        self.T_CAM_BASE_RIGHT = np.linalg.inv(T_BASE_CAM_RIGHT)

        self._init_ros()
        self._init_camera_and_aruco()
        self._setup_csv_files()

    def _init_ros(self):
        """Initializes the ROS subscriber."""
        pose_topic = f'/relaxed_ik/motion_control/pose_ee_arm_{self.arm_name}'
        self.node.create_subscription(PoseStamped, pose_topic, self._pose_callback, 10)
        self.node.get_logger().info(f"Subscribing to '{pose_topic}'")

    def _pose_callback(self, msg: PoseStamped):
        """Callback to store the latest end-effector pose."""
        self.latest_ee_pose = msg.pose

    def _init_camera_and_aruco(self):
        """Initializes the ZED camera and ArUco detector."""
        init_params = sl.InitParameters(
            camera_resolution=sl.RESOLUTION.HD720,
            camera_fps=30,
            depth_mode=sl.DEPTH_MODE.NONE,
            coordinate_units=sl.UNIT.METER
        )
        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise ConnectionError(f"ZED camera open failed: {status}")

        cam_info = self.zed.get_camera_information()
        calib = cam_info.camera_configuration.calibration_parameters
        
        self.cam_params['left'] = (
            np.array([[calib.left_cam.fx, 0, calib.left_cam.cx], [0, calib.left_cam.fy, calib.left_cam.cy], [0, 0, 1]]),
            calib.left_cam.disto.flatten()
        )
        self.cam_params['right'] = (
            np.array([[calib.right_cam.fx, 0, calib.right_cam.cx], [0, calib.right_cam.fy, calib.right_cam.cy], [0, 0, 1]]),
            calib.right_cam.disto.flatten()
        )
        
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
        self.aruco_detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())
        self.node.get_logger().info("ZED camera (Stereo) and ArUco detector initialized.")

    def _setup_csv_files(self):
        """Creates CSV files for left and right camera data."""
        for name in ['left', 'right']:
            filename = f'pose_data_cam_{name}_{self.arm_name}_arm.csv'
            csv_file = open(filename, 'w', newline='')
            writer = csv.writer(csv_file)
            header = ['timestamp', 'tvec_x', 'tvec_y', 'tvec_z', 'rvec_x', 'rvec_y', 'rvec_z',
                      'ee_x', 'ee_y', 'ee_z', 'ee_qx', 'ee_qy', 'ee_qz', 'ee_qw']
            writer.writerow(header)
            self.csv_files[name] = csv_file
            self.csv_writers[name] = writer
            self.node.get_logger().info(f"Saving data for '{name}' camera to '{filename}'")

    def _process_frame(self, frame: np.ndarray, cam_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detects markers, draws overlays, and returns pose of the target marker."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        
        target_tvec, target_rvec = None, None
        
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            K, D = self.cam_params[cam_name]
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, K, D)
            
            for i, marker_id in enumerate(ids.flatten()):
                cv2.drawFrameAxes(frame, K, D, rvecs[i], tvecs[i], 0.03)
                if marker_id == self.target_marker_id:
                    target_tvec, target_rvec = tvecs[i].flatten(), rvecs[i].flatten()

        # Project EE pose and add status text
        if cam_name == 'left':
            T_CAM_BASE = self.T_CAM_BASE_LEFT
        else:
            T_CAM_BASE = self.T_CAM_BASE_RIGHT
        self._draw_overlays(frame, cam_name, is_marker_found=(target_tvec is not None), T_CAM_BASE=T_CAM_BASE)
        return target_tvec, target_rvec

    def _draw_overlays(self, frame: np.ndarray, cam_name: str, is_marker_found: bool, T_CAM_BASE=None):
        """Draws EE pose projection and status text on the frame."""
        # Draw EE pose
        if self.latest_ee_pose:
            K, _ = self.cam_params[cam_name]
            p = self.latest_ee_pose.position
            P_base = np.array([p.x, p.y, p.z, 1.0])
            P_cam_h = T_CAM_BASE @ P_base
            P_cam = P_cam_h[:3] / P_cam_h[3]

            if P_cam[2] > 0:
                p_img_h = K @ P_cam
                u, v = int(p_img_h[0] / p_img_h[2]), int(p_img_h[1] / p_img_h[2])
                if 0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]:
                    cv2.circle(frame, (u, v), 7, (255, 0, 255), -1)

        # Status Text
        pose_status = "OK" if self.latest_ee_pose else "NO POSE"
        marker_status = "DETECTED" if is_marker_found else "NOT FOUND"
        cv2.putText(frame, f"EE Pose: {pose_status}", (10, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Marker {self.target_marker_id}: {marker_status}", (10, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if is_marker_found else (0, 0, 255), 2)

    def _save_data_point(self, tvec_l, rvec_l, tvec_r, rvec_r):
        """Writes a synchronized row of data to both CSV files."""
        if self.latest_ee_pose is None:
            self.node.get_logger().warn("Cannot save: No robot pose received yet.")
            return

        p = self.latest_ee_pose.position
        q = self.latest_ee_pose.orientation
        timestamp = self.node.get_clock().now().nanoseconds
        ee_pose_row = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]

        row_left = [timestamp, *tvec_l, *rvec_l] + ee_pose_row
        row_right = [timestamp, *tvec_r, *rvec_r] + ee_pose_row
        
        self.csv_writers['left'].writerow(row_left)
        self.csv_files['left'].flush()
        self.csv_writers['right'].writerow(row_right)
        self.csv_files['right'].flush()
        self.node.get_logger().info(f"Saved synchronized stereo data point for marker {self.target_marker_id}.")

    def run(self):
        """Main loop for capturing and processing frames."""
        image_left_zed = sl.Mat()
        image_right_zed = sl.Mat()
        window_name = "Stereo Calibration View"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while rclpy.ok():
            if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(image_left_zed, sl.VIEW.LEFT)
                self.zed.retrieve_image(image_right_zed, sl.VIEW.RIGHT)
                frame_left = cv2.cvtColor(image_left_zed.get_data(), cv2.COLOR_BGRA2BGR)
                frame_right = cv2.cvtColor(image_right_zed.get_data(), cv2.COLOR_BGRA2BGR)

                tvec_l, rvec_l = self._process_frame(frame_left, 'left')
                tvec_r, rvec_r = self._process_frame(frame_right, 'right')

                combined_frame = np.hstack((frame_left, frame_right))
                cv2.putText(combined_frame, "LEFT VIEW", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(combined_frame, "RIGHT VIEW", (frame_left.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(window_name, combined_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('s'):
                    if tvec_l is not None and tvec_r is not None:
                        self._save_data_point(tvec_l, rvec_l, tvec_r, rvec_r)
                    else:
                        self.node.get_logger().warn("Cannot save: Target marker must be visible in BOTH views.")
    
    def shutdown(self):
        """Cleans up resources."""
        self.node.get_logger().info("Closing resources...")
        self.zed.close()
        for f in self.csv_files.values():
            f.close()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    
    # --- User Input ---
    while True:
        choice = input("Choose which arm the ArUco tag is on: (1 for left, 2 for right): ")
        if choice in ['1', '2']:
            arm_name = 'left' if choice == '1' else 'right'
            break
        print("Invalid input. Please enter 1 or 2.")
        
    while True:
        try:
            marker_id_str = input("Enter the target ArUco marker ID (e.g., 42): ")
            marker_id = int(marker_id_str)
            break
        except ValueError:
            print("Invalid input. Please enter a number.")

    node = rclpy.create_node('stereo_aruco_collector')
    calibrator = StereoCalibrator(node=node, arm_name=arm_name, marker_id=marker_id)
    
    # Spin ROS in a separate thread
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    try:
        calibrator.run()
    finally:
        calibrator.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()