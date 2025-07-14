#!/usr/bin/env python3
import rospy
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
import tf.transformations as tf
import serial

### ---------- Inspire Hand Setup ----------
regdict = {
    'angleSet': 1486,
    'speedSet': 1522,
    'forceSet': 1498
}

def open_serial(port='/dev/ttyUSB0', baudrate=115200):
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baudrate
    ser.timeout = 1
    ser.open()
    return ser

def write_register(ser, id, add, num, val):
    bytes_out = [0xEB, 0x90, id, num + 3, 0x12, add & 0xFF, (add >> 8) & 0xFF]
    for i in range(num):
        bytes_out.append(val[i])
    checksum = sum(bytes_out[2:]) & 0xFF
    bytes_out.append(checksum)
    ser.write(bytearray(bytes_out))
    time.sleep(0.01)
    ser.read_all()

def write6(ser, id, param, val):
    if param in ['angleSet', 'forceSet', 'speedSet']:
        val_reg = []
        for i in range(6):
            val_reg.append(val[i] & 0xFF)
            val_reg.append((val[i] >> 8) & 0xFF)
        write_register(ser, id, regdict[param], 12, val_reg)
    else:
        print("[Hand] Invalid param for write6.")

### ---------- Controller ----------
class ArmKeyboardControl:
    def __init__(self, arm_choice):
        rospy.init_node('arm_keyboard_control_node', anonymous=True)

        self.lock = threading.Lock()
        self.current_pose = None
        self.recorded_data = []
        self.current_hand = [1000] * 6
        self.aruco_img = None
        self.detecting = False
        self.latest_tvec = None
        self.latest_rvec = None

        if arm_choice == '1':
            self.arm = 'left'
            serial_port = '/dev/ttyUSB0'
            self.hand_id = 1
            baudrate_in = 57600
        elif arm_choice == '2':
            self.arm = 'right'
            serial_port = '/dev/ttyUSB0'
            self.hand_id = 2
            baudrate_in = 115200
        else:
            rospy.logerr("Invalid input. Use '1' or '2'")
            exit()

        self.ser = open_serial(serial_port, baudrate_in)
        write6(self.ser, self.hand_id, 'speedSet', [800] * 6)
        write6(self.ser, self.hand_id, 'forceSet', [500] * 6)

        self.pose_topic = f'/relaxed_ik/motion_control/pose_ee_arm_{self.arm}'
        self.target_topic = f'/motion_target/target_pose_arm_{self.arm}'

        rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_callback)
        rospy.Subscriber("/zedm/zed_node/left/image_rect_color/compressed", CompressedImage, self.image_callback)

        self.pub = rospy.Publisher(self.target_topic, PoseStamped, queue_size=1)
        # zed
        # self.camera_matrix = np.array([[528.4206, 0.0, 635.5908],
        #                                [0.0, 528.4206, 359.7711],
        #                                [0.0, 0.0, 1.0]])

        # zedmini
        self.camera_matrix = np.array([[730.2571411132812, 0.0, 637.2598876953125],
                                       [0.0, 730.2571411132812, 346.41082763671875],
                                       [0.0, 0.0, 1.0]])
        # self.camera_matrix = np.array([[365.1285705566406, 0.0, 318.62994384765625],
        #                        [0.0, 365.1285705566406, 173.20541381835938],
        #                        [0.0, 0.0, 1.0]])



        self.dist_coeffs = np.zeros((5, 1))
        self.marker_length = 0.05
        self.target_id = 23
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters_create()
        # self.aruco_params = aruco.DetectorParameters()



        self.camera_cali_dir = "camera_cali"
        os.makedirs(self.camera_cali_dir, exist_ok=True)

        rospy.loginfo(f"Waiting for EE pose of {self.arm} arm...")
        rospy.wait_for_message(self.pose_topic, PoseStamped)
        rospy.sleep(0.5)
        rospy.loginfo("EE pose received.")

    def pose_callback(self, msg):
        with self.lock:
            self.current_pose = msg

    def image_callback(self, msg):
        if not self.detecting:
            return
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                return
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            if ids is not None:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
                for i in range(len(ids)):
                    if ids[i][0] == self.target_id:
                        self.latest_tvec = tvecs[i].reshape(3)
                        self.latest_rvec = rvecs[i].reshape(3)
                        print(f"[Aruco] ID {self.target_id} detected")
                        self.print_and_record()
                        self.save_aruco_result()
                        print("[INFO] current aruco detect over.")
                        # ---------------------------------------------------
                        # Visualize Aruco result: Added by 25/06 delete me
                        img_with_detections = aruco.drawDetectedMarkers(img.copy(), corners, ids)
                        cv2.drawFrameAxes(img_with_detections, self.camera_matrix, self.dist_coeffs, self.latest_rvec, self.latest_tvec, 0.03)
                        # Save image with ArUco overlay
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        img_path = os.path.join(self.camera_cali_dir, f"aruco_detect_{timestamp}.png")
                        cv2.imwrite(img_path, img_with_detections)
                        print(f"[INFO] Saved ArUco detection image to {img_path}")
                        # --------------------------------------------------
                        self.detecting = False
                        return
        except Exception as e:
            rospy.logerr(f"Aruco detection failed: {e}")

    def send_hand(self, values):
        write6(self.ser, self.hand_id, 'angleSet', values)

    def run(self):
        rospy.sleep(1.0)
        print(f"[{self.arm}] Controls:")
        print("  w/s/a/d/q/e - move xyz")
        print("  i/k/j/l/u/o - rotate rpy")
        print("  1/2/3/4 - hand presets")
        print("  r - record pose")
        print("  f - detect ArUco & record")
        print("  n - save and quit")

        while not rospy.is_shutdown():
            key = readchar.readkey()
            if key == 'n':
                self.save_recorded_data()
                break
            elif key == 'r':
                self.print_and_record()
            elif key == 'f':
                self.detecting = True
                print("[INFO] Start detecting ArUco ID 23...")
            elif key == '1':
                self.current_hand = [1000] * 6
                self.send_hand(self.current_hand)
            elif key == '2':
                self.current_hand = [800] * 6
                self.send_hand(self.current_hand)
            elif key == '3':
                self.current_hand = [0, 0, 0, 0, 500, 1000]
                self.send_hand(self.current_hand)
            elif key == '4':
                self.current_hand = [800, 800, 800, 800, 800, 0]
                self.send_hand(self.current_hand)
            else:
                self.update_pose_by_key(key)

    def update_pose_by_key(self, key):
        with self.lock:
            if self.current_pose is None:
                rospy.logwarn("No pose received yet.")
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
            delta_q = tf.quaternion_from_euler(*drot)
            new_quat = tf.quaternion_multiply(delta_q, quat)

            new_pose = PoseStamped()
            new_pose.header.stamp = rospy.Time.now()
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
            rospy.loginfo(f"[{self.arm}] Pose: pos({p.x:.3f}, {p.y:.3f}, {p.z:.3f})  ori({o.x:.3f}, {o.y:.3f}, {o.z:.3f}, {o.w:.3f}) hand: {self.current_hand}")
            self.recorded_data.append((p, o, list(self.current_hand)))

    def save_recorded_data(self):
        folder = os.path.expanduser(f'{self.arm}')
        os.makedirs(folder, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(folder, f'{timestamp}.txt')
        with open(path, 'w') as f:
            for p, o, hand in self.recorded_data:
                f.write(f"{p.x} {p.y} {p.z} {o.x} {o.y} {o.z} {o.w} {' '.join(str(h) for h in hand)}\n")
        rospy.loginfo(f"[{self.arm}] Recorded {len(self.recorded_data)} poses to {path}")

    def save_aruco_result(self):
        with self.lock:
            if self.latest_tvec is None or self.latest_rvec is None or self.current_pose is None:
                print("[WARN] Missing ArUco or EE pose, skip saving.")
                return

            try:
                R_detected, _ = cv2.Rodrigues(self.latest_rvec)
                R_y = cv2.Rodrigues(np.array([0, -np.pi/2, 0]))[0]
                R_z = cv2.Rodrigues(np.array([0, 0, -np.pi/2]))[0]
                R_correction = R_y @ R_z
                R_corrected = R_correction @ R_detected
                rvec_corrected = cv2.Rodrigues(R_corrected)[0]
            except Exception as e:
                rospy.logerr(f"Failed to correct ArUco rotation: {e}")
                return

            p = self.current_pose.pose.position
            o = self.current_pose.pose.orientation
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

            row = [
                timestamp,
                *self.latest_tvec,
                *rvec_corrected.flatten(),
                p.x, p.y, p.z,
                o.x, o.y, o.z, o.w,
                *self.current_hand
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
                        "ee_qx", "ee_qy", "ee_qz", "ee_qw",
                        "hand_0", "hand_1", "hand_2", "hand_3", "hand_4", "hand_5"
                    ])
                writer.writerow(row)

            print(f"[INFO] Saved synchronized ArUco + EE pose to {file_path}")


if __name__ == '__main__':
    try:
        print("Choose arm to control:\n  1 - Left Arm\n  2 - Right Arm")
        choice = input("Enter 1 or 2: ").strip()
        ArmKeyboardControl(choice).run()
    except rospy.ROSInterruptException:
        pass
