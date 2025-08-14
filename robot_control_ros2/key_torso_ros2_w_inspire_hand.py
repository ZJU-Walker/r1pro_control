#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import threading
import time
import readchar
import os
from datetime import datetime
import transforms3d.quaternions as tfq
import transforms3d.euler as tfe
import numpy as np
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
class ArmKeyboardControl(Node):
    def __init__(self, arm_choice):
        super().__init__('arm_keyboard_control_node')
        
        self.lock = threading.Lock()
        self.current_pose = None
        self.recorded_data = []
        self.current_hand = [1000] * 6  # Default: open
        self.message_received = False

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
            self.get_logger().error("Invalid input. Use '1' for left arm or '2' for right arm.")
            exit()

        # Initialize serial port for Inspire Hand
        self.ser = open_serial(serial_port, baudrate_in)
        write6(self.ser, self.hand_id, 'speedSet', [800] * 6)
        write6(self.ser, self.hand_id, 'forceSet', [500] * 6)

        self.pose_topic = f'/relaxed_ik/motion_control/pose_ee_arm_{self.arm}'
        self.target_topic = f'/motion_target/target_pose_arm_{self.arm}'

        self.subscription = self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.pose_callback,
            10)
        
        self.pub = self.create_publisher(PoseStamped, self.target_topic, 10)

        self.get_logger().info(f"Waiting for EE pose of {self.arm} arm...")
        
        # Wait for first message
        self.wait_for_first_message()
        time.sleep(0.5)
        self.get_logger().info("EE pose received.")

    def wait_for_first_message(self, timeout=10.0):
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

    def send_hand(self, values):
        write6(self.ser, self.hand_id, 'angleSet', values)

    def quaternion_from_euler(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion."""
        return tfq.axangle2quat([1, 0, 0], roll) @ \
               tfq.axangle2quat([0, 1, 0], pitch) @ \
               tfq.axangle2quat([0, 0, 1], yaw)

    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions."""
        # q1 and q2 are [x, y, z, w]
        w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
        w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return [x, y, z, w]

    def run(self):
        time.sleep(1.0)
        print(f"[{self.arm}] Use keyboard to control:")
        print("  w/s: x+/x- | a/d: y+/y- | q/e: z+/z-")
        print("  i/k: roll | j/l: pitch | u/o: yaw")
        print("  1: hand open | 2: hand half-close | 3: hand custom | 4: fully closed")
        print("  r: record pose | n: save and quit")

        while rclpy.ok():
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
                    self.get_logger().warning("No pose received yet.")
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
                
                # Convert Euler angles to quaternion for the delta rotation
                delta_quat = tfe.euler2quat(drot[0], drot[1], drot[2])
                # delta_quat is [w, x, y, z] from transforms3d, need to convert to [x, y, z, w]
                delta_quat = [delta_quat[1], delta_quat[2], delta_quat[3], delta_quat[0]]
                
                # Apply rotation
                new_quat = self.quaternion_multiply(current_quat, delta_quat)

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
                
            # Spin once to process callbacks
            rclpy.spin_once(self, timeout_sec=0.01)

    def print_and_record(self):
        with self.lock:
            if self.current_pose is None:
                return
            p = self.current_pose.pose.position
            o = self.current_pose.pose.orientation
            self.get_logger().info(f"[{self.arm}] Pose: pos({p.x:.3f}, {p.y:.3f}, {p.z:.3f})  ori({o.x:.3f}, {o.y:.3f}, {o.z:.3f}, {o.w:.3f}) hand: {self.current_hand}")
            self.recorded_data.append((p, o, list(self.current_hand)))

    def save_recorded_data(self):
        folder = os.path.expanduser(f'{self.arm}')
        os.makedirs(folder, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(folder, f'{timestamp}.txt')
        with open(path, 'w') as f:
            for p, o, hand in self.recorded_data:
                f.write(f"{p.x} {p.y} {p.z} {o.x} {o.y} {o.z} {o.w} {' '.join(str(h) for h in hand)}\n")
        self.get_logger().info(f"[{self.arm}] Recorded {len(self.recorded_data)} poses to {path}")

### ---------- Torso Keyboard Control ----------
class TorsoKeyboardControl(Node):
    def __init__(self):
        super().__init__('torso_keyboard_control_node')

        self.current_joint_pos = [0.0] * 4
        self.target_joint_pos = [0.0] * 4
        self.selected_joint = 0  # default to joint 1
        self.message_received = False

        self.subscription = self.create_subscription(
            JointState,
            '/hdas/feedback_torso',
            self.feedback_callback,
            10)
        
        self.pub = self.create_publisher(JointState, '/motion_target/target_joint_state_torso', 10)

        self.get_logger().info("Waiting for torso joint feedback...")
        self.wait_for_first_message()
        time.sleep(0.5)
        self.get_logger().info("Torso joint feedback received.")

    def wait_for_first_message(self, timeout=10.0):
        start_time = time.time()
        while not self.message_received and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.message_received:
                break
        if not self.message_received:
            self.get_logger().warning("Timeout waiting for first message")

    def feedback_callback(self, msg):
        if len(msg.position) >= 4:
            self.current_joint_pos = list(msg.position)
            self.target_joint_pos = list(msg.position)
            self.message_received = True
            
    def run(self):
        print("[Torso] Control mode")
        print("  Press 1-4 to select joint 1~4")
        print("  u: increase pos | j: decrease pos")
        print("  n: exit")

        # Display current torso joint status
        print(f"[Torso] Current joint positions: {['%.3f' % p for p in self.current_joint_pos]}")

        while rclpy.ok():
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
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.position = self.target_joint_pos
            msg.velocity = [1.5] * 4
            self.pub.publish(msg)

            print(f"[Torso] Joint positions: {['%.3f' % p for p in self.target_joint_pos]}")
            
            # Spin once to process callbacks
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.1)  # Rate limiting


### ---------- Main ----------
def main():
    try:
        print("Choose part to control:")
        print("  1 - Left Arm")
        print("  2 - Right Arm")
        print("  3 - Torso")
        choice = input("Enter 1, 2, or 3: ").strip()
        
        rclpy.init()
        
        if choice == '3':
            node = TorsoKeyboardControl()
        else:
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

if __name__ == '__main__':
    main()