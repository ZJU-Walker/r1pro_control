#!/usr/bin/env python3
"""
ROS2 Node for Inspire Right Hand Control
Subscribes to /teleop/inspire_right_command and sends commands to the hand via RS485

Topic type: std_msgs/msg/Float32MultiArray
  - data[0-5]: 6 DOF angles (0-1000), use -1 to skip a finger
  - Order: [thumb_bend, thumb_rotate, index, middle, ring, pinky] (check manual for exact mapping)

To check ttyUSB devices:
  ls /dev/ttyUSB*

To find which USB device is the hand:
  dmesg | grep ttyUSB
  or
  udevadm info -a -n /dev/ttyUSB0 | grep -E 'ATTRS{idVendor}|ATTRS{idProduct}'
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import serial
import time


class InspireRightHandController(Node):
    # Register addresses from manual
    REGDICT = {
        'ID': 1000,
        'baudrate': 1001,
        'clearErr': 1004,
        'forceClb': 1009,
        'angleSet': 1486,
        'forceSet': 1498,
        'speedSet': 1522,
        'angleAct': 1546,
        'forceAct': 1582,
        'errCode': 1606,
        'statusCode': 1612,
        'temp': 1618,
        'actionSeq': 2320,
        'actionRun': 2322
    }

    def __init__(self):
        super().__init__('inspire_right_hand_controller')

        # Parameters
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('hand_id', 2)  # Right hand ID is 2
        self.declare_parameter('default_speed', 1000)
        self.declare_parameter('default_force', 500)

        self.serial_port = self.get_parameter('serial_port').value
        self.baudrate = self.get_parameter('baudrate').value
        self.hand_id = self.get_parameter('hand_id').value
        self.default_speed = self.get_parameter('default_speed').value
        self.default_force = self.get_parameter('default_force').value

        # Initialize serial connection
        self.ser = None
        self.connect_serial()

        # Initialize hand with default speed and force
        if self.ser and self.ser.is_open:
            self.init_hand()

        # Create subscriber
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/teleop/inspire_right_command',
            self.command_callback,
            10
        )

        self.get_logger().info(f'Inspire Right Hand Controller started')
        self.get_logger().info(f'  Hand ID: {self.hand_id}')
        self.get_logger().info(f'  Serial port: {self.serial_port}')
        self.get_logger().info(f'  Subscribing to: /teleop/inspire_right_command')

    def connect_serial(self):
        """Open serial connection to the hand"""
        try:
            self.ser = serial.Serial()
            self.ser.port = self.serial_port
            self.ser.baudrate = self.baudrate
            self.ser.timeout = 0.1
            self.ser.open()
            self.get_logger().info(f'Serial port {self.serial_port} opened successfully')
        except serial.SerialException as e:
            self.get_logger().error(f'Failed to open serial port {self.serial_port}: {e}')
            self.get_logger().error('Check available ports with: ls /dev/ttyUSB*')
            self.ser = None

    def init_hand(self):
        """Initialize hand with default speed and force settings"""
        self.get_logger().info('Initializing hand with default speed and force...')
        speed_values = [self.default_speed] * 6
        force_values = [self.default_force] * 6
        self.write6('speedSet', speed_values)
        time.sleep(0.1)
        self.write6('forceSet', force_values)
        time.sleep(0.1)
        self.get_logger().info('Hand initialization complete')

    def write_register(self, add, num, val):
        """Write to hand register"""
        if not self.ser or not self.ser.is_open:
            self.get_logger().warn('Serial port not open, cannot write')
            return

        bytes_data = [0xEB, 0x90]  # Frame header
        bytes_data.append(self.hand_id)  # ID
        bytes_data.append(num + 3)  # Length
        bytes_data.append(0x12)  # Write command
        bytes_data.append(add & 0xFF)  # Address low byte
        bytes_data.append((add >> 8) & 0xFF)  # Address high byte

        for i in range(num):
            bytes_data.append(val[i])

        # Calculate checksum
        checksum = 0x00
        for i in range(2, len(bytes_data)):
            checksum += bytes_data[i]
        checksum &= 0xFF
        bytes_data.append(checksum)

        self.ser.write(bytes(bytes_data))
        time.sleep(0.01)
        self.ser.read_all()  # Clear response

    def write6(self, reg_name, val):
        """Write 6 DOF values to hand"""
        if reg_name not in ['angleSet', 'forceSet', 'speedSet']:
            self.get_logger().error(f'Invalid register name: {reg_name}')
            return

        val_reg = []
        for i in range(6):
            v = int(val[i])
            val_reg.append(v & 0xFF)
            val_reg.append((v >> 8) & 0xFF)

        self.write_register(self.REGDICT[reg_name], 12, val_reg)

    def command_callback(self, msg):
        """Handle incoming hand commands"""
        if len(msg.data) < 6:
            self.get_logger().warn(f'Expected 6 values, got {len(msg.data)}')
            return

        # Clamp values to valid range [0, 1000] or -1 for skip
        angles = []
        for i in range(6):
            val = msg.data[i]
            if val < 0:
                angles.append(-1)  # Skip this finger
            else:
                angles.append(int(min(max(val, 0), 1000)))

        self.get_logger().debug(f'Setting angles: {angles}')
        self.write6('angleSet', angles)

    def destroy_node(self):
        """Clean up on shutdown"""
        if self.ser and self.ser.is_open:
            self.get_logger().info('Closing serial port')
            self.ser.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = InspireRightHandController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
