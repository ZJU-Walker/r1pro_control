#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import threading
import time
import readchar
import os
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

### ---------- Arm Keyboard Control ----------
class ArmKeyboardControl:
    def __init__(self, arm_choice):
        rospy.init_node('arm_keyboard_control_node', anonymous=True)

        self.lock = threading.Lock()
        self.current_pose = None
        self.recorded_data = []
        self.current_hand = [1000] * 6  # Default: open

        # Arm selection and serial port mapping
        if arm_choice == '1':
            self.arm = 'left'
            serial_port = '/dev/ttyUSB0'
            self.hand_id = 1
            baudrate_in = 57600
        elif arm_choice == '2':
            self.arm = 'right'
            serial_port = '/dev/ttyUSB1'
            self.hand_id = 2
            baudrate_in = 115200
        else:
            rospy.logerr("Invalid input. Use '1' for left arm or '2' for right arm.")
            exit()

        # Initialize serial port for Inspire Hand
        self.ser = open_serial(serial_port, baudrate_in)
        write6(self.ser, self.hand_id, 'speedSet', [800] * 6)
        write6(self.ser, self.hand_id, 'forceSet', [500] * 6)

        self.pose_topic = f'/relaxed_ik/motion_control/pose_ee_arm_{self.arm}'
        # self.pose_topic = f'/motion_control/pose_ee_arm_{self.arm}'
        self.target_topic = f'/motion_target/target_pose_arm_{self.arm}'

        rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_callback)
        self.pub = rospy.Publisher(self.target_topic, PoseStamped, queue_size=1)

        rospy.loginfo(f"Waiting for EE pose of {self.arm} arm...")
        rospy.wait_for_message(self.pose_topic, PoseStamped)
        rospy.sleep(0.5)
        rospy.loginfo("EE pose received.")

    def pose_callback(self, msg):
        with self.lock:
            self.current_pose = msg

    def send_hand(self, values):
        write6(self.ser, self.hand_id, 'angleSet', values)

    def run(self):
        rospy.sleep(1.0)
        print(f"[{self.arm}] Use keyboard to control:")
        print("  w/s: x+/x- | a/d: y+/y- | q/e: z+/z-")
        print("  i/k: roll | j/l: pitch | u/o: yaw")
        print("  1: hand open | 2: hand half-close | 3: hand custom | 4: fully closed")
        print("  r: record pose | n: save and quit")

        while not rospy.is_shutdown():
            key = readchar.readkey()
            if key == 'n':
                self.save_recorded_data()
                break
            elif key == 'r':
                self.print_and_record()
                continue
            elif key == '1':
                self.current_hand = [1000, 1000, 1000, 1000, 0, 0]
                self.send_hand(self.current_hand)
                print("[Hand] Fully open")
                continue
            elif key == '2':
                self.current_hand = [1000] * 6
                self.send_hand(self.current_hand)
                print("[Hand] Partially closed")
                continue
            elif key == '3':
                self.current_hand = [0, 0, 0, 0, 500, 1000]
                self.send_hand(self.current_hand)
                print("[Hand] Custom pose")
                continue
            elif key == '4':
                self.current_hand = [800, 800, 800, 800, 800, 0]
                self.send_hand(self.current_hand)
                print("[Hand] Fully closed")
                continue
            elif key == '5':
                self.current_hand = [408, 599, 630, 586, 1000, 761]
                self.send_hand(self.current_hand)
                print("[Hand] start position")
                continue
            elif key == '6':
                self.current_hand = [1000-536, 1000-329, 1000-268, 1000-277, 1000-0, 946]
                self.send_hand(self.current_hand)
                print("[Hand] start position")
                continue
            elif key == '7':
                self.current_hand = [1000-741, 1000-505, 1000-425, 1000-414, 1000-499, 1000-163]
                self.send_hand(self.current_hand)
                print("[Hand] median position")
                continue

            with self.lock:
                if self.current_pose is None:
                    rospy.logwarn("No pose received yet.")
                    continue

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
                else:
                    print("Unknown key")
                    continue

                current_quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
                delta_quat = tf.quaternion_from_euler(*drot)
                # new_quat = tf.quaternion_multiply(delta_quat, current_quat)
                new_quat = tf.quaternion_multiply(current_quat, delta_quat)

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

### ---------- Torso Keyboard Control ----------
class TorsoKeyboardControl:
    def __init__(self):
        rospy.init_node('torso_keyboard_control_node', anonymous=True)

        self.current_joint_pos = [0.0] * 4
        self.target_joint_pos = [0.0] * 4
        self.selected_joint = 0  # default to joint 1

        rospy.Subscriber('/hdas/feedback_torso', JointState, self.feedback_callback)
        self.pub = rospy.Publisher('/motion_target/target_joint_state_torso', JointState, queue_size=1)

        rospy.loginfo("Waiting for torso joint feedback...")
        rospy.wait_for_message('/hdas/feedback_torso', JointState)
        rospy.sleep(0.5)
        rospy.loginfo("Torso joint feedback received.")

    def feedback_callback(self, msg):
        if len(msg.position) >= 4:
            self.current_joint_pos = list(msg.position)
            self.target_joint_pos = list(msg.position)
            
    def run(self):
        print("[Torso] Control mode")
        print("  Press 1-4 to select joint 1~4")
        print("  u: increase pos | j: decrease pos")
        print("  n: exit")

        # 显示当前躯干关节状态
        print(f"[Torso] Current joint positions: {['%.3f' % p for p in self.current_joint_pos]}")

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            key = readchar.readkey()
            if key == 'n':
                print("Exiting torso control.")
                break
            elif key in ['1', '2', '3', '4']:
                self.selected_joint = int(key) - 1
                print(f"[Torso] Selected joint {self.selected_joint + 1}")
                continue
            elif key == 'u':
                self.target_joint_pos[self.selected_joint] += 0.05
            elif key == 'j':
                self.target_joint_pos[self.selected_joint] -= 0.05
            else:
                print("Unknown key")
                continue

            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.position = self.target_joint_pos
            msg.velocity = [1.5] * 4
            self.pub.publish(msg)

            print(f"[Torso] Joint positions: {['%.3f' % p for p in self.target_joint_pos]}")
            rate.sleep()


### ---------- Main ----------
if __name__ == '__main__':
    try:
        print("Choose part to control:")
        print("  1 - Left Arm")
        print("  2 - Right Arm")
        print("  3 - Torso")
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice == '3':
            TorsoKeyboardControl().run()
        else:
            ArmKeyboardControl(choice).run()
    except rospy.ROSInterruptException:
        pass
