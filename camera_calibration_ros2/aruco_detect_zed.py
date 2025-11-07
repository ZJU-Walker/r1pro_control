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

# --- Configuration ---
MARKER_SIZE = 0.048  # 4.8 centimeters

# --- Provided Transformation Matrix (from Camera to Robot Base) ---
T_BASE_CAM = np.array([
    [ 0.00692993, -0.87310148,  0.48748926,  0.14062141],
    [-0.99995006, -0.00956093, -0.00290894,  0.03612369],
    [ 0.00720065, -0.48744476, -0.87312414,  0.46063114],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])
T_CAM_BASE = np.linalg.inv(T_BASE_CAM)

# --- Global variable to share state between threads ---
LATEST_EE_POSE = None

def choose_arm():
    """Prompts the user to select an arm."""
    while True:
        choice = input("Choose an arm: (1 for left, 2 for right): ")
        if choice == '1':
            return 'left'
        elif choice == '2':
            return 'right'
        else:
            print("Invalid input. Please enter 1 or 2.")

def pose_callback(msg):
    """ROS subscriber callback to update the global pose variable."""
    global LATEST_EE_POSE
    LATEST_EE_POSE = msg.pose

def init_camera_and_aruco(node: Node):
    """Initializes the ZED camera and ArUco detector."""
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.coordinate_units = sl.UNIT.METER

    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        node.get_logger().error(f"ZED camera open failed: {status}")
        return None, None, None

    cam_info = zed.get_camera_information()
    cam_params = cam_info.camera_configuration.calibration_parameters.left_cam
    K = np.array([[cam_params.fx, 0, cam_params.cx], [0, cam_params.fy, cam_params.cy], [0, 0, 1]])
    
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
    aruco_detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())
    
    node.get_logger().info("ZED camera and ArUco detector initialized.")
    return zed, K, aruco_detector

def setup_csv_file(arm_name: str, node: Node):
    """Creates a CSV file and writes the specified header."""
    filename = f'pose_data_arm_{arm_name}.csv'
    csv_file = open(filename, 'w', newline='')
    writer = csv.writer(csv_file)
    header = [
        'timestamp',
        'tvec_x', 'tvec_y', 'tvec_z',
        'rvec_x', 'rvec_y', 'rvec_z',
        'ee_x', 'ee_y', 'ee_z',
        'ee_qx', 'ee_qy', 'ee_qz', 'ee_qw'
    ]
    writer.writerow(header)
    node.get_logger().info(f"Saving data to '{filename}'")
    return csv_file, writer

def save_data_point(csv_writer, csv_file, node: Node, tvec, rvec):
    """Writes a single row of pose data to the CSV file."""
    if LATEST_EE_POSE is None:
        node.get_logger().warn("Cannot save: No robot pose received yet.")
        return

    p = LATEST_EE_POSE.position
    q = LATEST_EE_POSE.orientation
    row = [
        node.get_clock().now().nanoseconds,
        tvec[0][0], tvec[0][1], tvec[0][2], # ArUco translation
        rvec[0][0], rvec[0][1], rvec[0][2], # ArUco rotation
        p.x, p.y, p.z,                     # EE position
        q.x, q.y, q.z, q.w                  # EE orientation
    ]
    csv_writer.writerow(row)
    csv_file.flush()
    node.get_logger().info(f"Saved data point.")

def project_ee_pose_to_image(frame, ee_pose, T_cam_base, camera_matrix):
    """Transforms and projects the 3D EE pose onto the 2D image."""
    if ee_pose is None:
        return

    p = ee_pose.position
    P_base = np.array([p.x, p.y, p.z, 1.0])
    P_cam_homogeneous = T_cam_base @ P_base
    P_cam = P_cam_homogeneous[:3] / P_cam_homogeneous[3]

    if P_cam[2] > 0:
        p_image_homogeneous = camera_matrix @ P_cam
        u = int(p_image_homogeneous[0] / p_image_homogeneous[2])
        v = int(p_image_homogeneous[1] / p_image_homogeneous[2])

        height, width, _ = frame.shape
        if 0 <= u < width and 0 <= v < height:
            cv2.circle(frame, (u, v), 5, (255, 0, 255), -1)
            cv2.putText(frame, "EE", (u + 15, v + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


def main(args=None):
    # --- Initialization ---
    rclpy.init(args=args)
    arm_name = choose_arm()
    node = rclpy.create_node('aruco_pose_collector')

    pose_topic = f'/relaxed_ik/motion_control/pose_ee_arm_{arm_name}'
    node.create_subscription(PoseStamped, pose_topic, pose_callback, 10)
    
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()
    node.get_logger().info(f"ROS subscriber for '{pose_topic}' is running in the background.")

    zed, camera_matrix, aruco_detector = init_camera_and_aruco(node)
    if zed is None:
        rclpy.shutdown()
        return
    
    csv_file, csv_writer = setup_csv_file(arm_name, node)
    dist_coeffs = np.zeros((1, 5))
    image_zed = sl.Mat()
    prev_frame_time = 0

    # --- Main Loop ---
    try:
        while True:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                frame = cv2.cvtColor(image_zed.get_data(), cv2.COLOR_BGRA2BGR)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = aruco_detector.detectMarkers(gray)
                
                if ids is not None:
                    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, camera_matrix, dist_coeffs)
                    aruco.drawDetectedMarkers(frame, corners, ids)
                    for i in range(len(ids)):
                        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)

                project_ee_pose_to_image(frame, LATEST_EE_POSE, T_CAM_BASE, camera_matrix)

                new_frame_time = time.time()
                if prev_frame_time > 0:
                    fps = 1 / (new_frame_time - prev_frame_time)
                    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                prev_frame_time = new_frame_time

                cv2.imshow("ArUco and EE Pose", frame)
                key = cv2.waitKey(1)

                if key == ord('f'):
                    if ids is not None and len(ids) == 1:
                        save_data_point(csv_writer, csv_file, node, tvecs[0], rvecs[0])
                    elif ids is not None and len(ids) > 1:
                        node.get_logger().warn("Multiple markers detected. Isolate one to save data.")
                    else:
                        node.get_logger().warn("No markers detected. Cannot save data.")
    finally:
        # --- Cleanup ---
        node.get_logger().info("Closing resources.")
        zed.close()
        csv_file.close()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()