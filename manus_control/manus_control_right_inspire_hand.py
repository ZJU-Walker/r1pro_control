#!/usr/bin/env python3
"""
ROS2 Node for Manus Glove to Inspire Right Hand Control

Subscribes to Manus glove data and publishes commands to Inspire hand.
Topic: /teleop/inspire_right_command (Float32MultiArray)

Output order: [pinky, ring, middle, index, thumb_bend, thumb_rotate]
Each value in range [0, 1000]

Mapping from qd array:
    qd[0]  -> thumb_rotate
    qd[2]  -> thumb_bend
    qd[5]  -> index
    qd[9]  -> middle
    qd[13] -> ring
    qd[17] -> pinky
"""

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from manus_ros2_msgs.msg import ManusGlove

USER_NAME = 'brain'


def set_target_inspire(q, motor_count=20):
    """
    Calculates the intermediate qd values from Manus glove input.

    Args:
        q (list[float]): A list of 20 input angles in degrees from the Manus glove.
        motor_count (int): The total number of motors. Defaults to 20.

    Returns:
        list[float]: The qd values in radians.
    """
    qd = [0.0] * motor_count

    # Convert input degrees to radians with initial angle offsets
    # Thumb
    qd[0] = (38.5 - q[1]) * (math.pi / 180)  # Thumb rotate/spread
    qd[1] = (q[0] + 36) * (math.pi / 180)
    qd[2] = (q[2] + 10) * (math.pi / 180)    # Thumb bend (PIP stretch)
    qd[3] = (q[3] + 5) * (math.pi / 180)

    # Index
    qd[4] = q[4] * (math.pi / 180)
    qd[5] = q[5] * (math.pi / 180)           # Index MCP stretch
    qd[6] = q[6] * (math.pi / 180)
    qd[7] = q[7] * (math.pi / 180)

    # Middle
    qd[8] = q[8] * (math.pi / 180)
    qd[9] = q[9] * (math.pi / 180)           # Middle MCP stretch
    qd[10] = q[10] * (math.pi / 180)
    qd[11] = q[11] * (math.pi / 180)

    # Ring
    qd[12] = q[12] * (math.pi / 180)
    qd[13] = q[13] * (math.pi / 180)         # Ring MCP stretch
    qd[14] = q[14] * (math.pi / 180)
    qd[15] = q[15] * (math.pi / 180)

    # Pinky
    if q[17] > 55 and q[18] > 25:
        qd[16] = abs(q[16]) * 2 * (math.pi / 180)
    else:
        qd[16] = abs(q[16]) / 1.5 * (math.pi / 180)

    qd[17] = q[16] * (math.pi / 180)         # Pinky (from q[16])
    qd[18] = q[17] * (math.pi / 180)
    qd[19] = q[19] * (math.pi / 180)

    return q


def qd_to_inspire_command(qd):
    """
    Convert qd values (radians) to Inspire hand command values (0-1000).

    Mapping:
        qd[0]  -> thumb_rotate
        qd[2]  -> thumb_bend
        qd[5]  -> index
        qd[9]  -> middle
        qd[13] -> ring
        qd[17] -> pinky

    Output order: [pinky, ring, middle, index, thumb_bend, thumb_rotate]

    Args:
        qd (list[float]): The qd values in radians.

    Returns:
        list[float]: 6 command values in range [0, 1000].
    """
    # Conversion factor: radians to 0-1000 range
    # Assuming max angle ~90 degrees (pi/2 radians) maps to 1000
    # scale = 1000.0 / (math.pi / 2)  # ~636.6
    scale = 1000.0 / 90.0 

    # Extract and convert relevant qd values
    # thumb_rotate = abs(qd[1]) / 30.0 * 1000 
    thumb_rotate = 1000 - abs(qd[0]) / 15 * 1000
    thumb_bend = (1000 - abs(qd[3]) / 50.0 * 1000) 
    index_val = 1000 - abs(qd[6]) * scale
    middle_val = 1000 - abs(qd[10]) * scale
    ring_val = 1000 - abs(qd[14]) * scale
    pinky_val = 1000 - abs(qd[18]) * scale

    # Clamp to [0, 1000]
    def clamp(val):
        return max(0.0, min(1000.0, val))

    # Output order: pinky, ring, middle, index, thumb_bend, thumb_rotate
    return [
        clamp(pinky_val),
        clamp(ring_val),
        clamp(middle_val),
        clamp(index_val),
        clamp(thumb_bend),
        clamp(thumb_rotate)
    ]


class ManusInspireControlNode(Node):
    def __init__(self):
        super().__init__('manus_inspire_control_node')

        # Create subscriber for Manus glove data
        self.glove_subscriber = self.create_subscription(
            ManusGlove,
            '/manus_glove_1',
            self.glove_callback,
            10
        )

        # Create publisher for Inspire right hand command
        self.command_publisher = self.create_publisher(
            Float32MultiArray,
            '/teleop/inspire_right_command',
            1
        )

        # Mapping from ergonomics type names to q array indices
        self.ergonomics_mapping = {
            'ThumbMCPSpread': 0,
            'ThumbMCPStretch': 1,
            'ThumbPIPStretch': 2,
            'ThumbDIPStretch': 3,
            'IndexSpread': 4,
            'IndexMCPStretch': 5,
            'IndexPIPStretch': 6,
            'IndexDIPStretch': 7,
            'MiddleSpread': 8,
            'MiddleMCPStretch': 9,
            'MiddlePIPStretch': 10,
            'MiddleDIPStretch': 11,
            'RingSpread': 12,
            'RingMCPStretch': 13,
            'RingPIPStretch': 14,
            'RingDIPStretch': 15,
            'PinkySpread': 16,
            'PinkyMCPStretch': 17,
            'PinkyPIPStretch': 18,
            'PinkyDIPStretch': 19,
        }

        self.get_logger().info('Manus Inspire Control Node initialized')
        self.get_logger().info(f'User: {USER_NAME}')
        self.get_logger().info('Subscribing to: /manus_glove_0')
        self.get_logger().info('Publishing to: /teleop/inspire_right_command')
        self.get_logger().info('Output order: [pinky, ring, middle, index, thumb_bend, thumb_rotate]')

    def glove_callback(self, msg: ManusGlove):
        """Callback function for processing Manus glove data."""
        try:
            # Extract ergonomics data into q array
            q = self.extract_ergonomics_to_q(msg.ergonomics)

            # Convert to qd values (radians)
            qd = set_target_inspire(q)

            # Convert qd to Inspire hand command values (0-1000)
            inspire_cmd = qd_to_inspire_command(qd)

            # Print for debugging
            print(f'Inspire cmd: pinky={inspire_cmd[0]:.0f}, ring={inspire_cmd[1]:.0f}, '
                  f'middle={inspire_cmd[2]:.0f}, index={inspire_cmd[3]:.0f}, '
                  f'thumb_bend={inspire_cmd[4]:.0f}, thumb_rotate={inspire_cmd[5]:.0f}')

            # Publish the command
            self.publish_command(inspire_cmd)

        except Exception as e:
            self.get_logger().error(f'Error processing glove data: {e}')

    def extract_ergonomics_to_q(self, ergonomics):
        """Extract ergonomics data and map to q array format."""
        q = [0.0] * 20

        for ergo in ergonomics:
            ergo_type = ergo.type.replace('type: ', '')
            if ergo_type in self.ergonomics_mapping:
                index = self.ergonomics_mapping[ergo_type]
                q[index] = ergo.value

        return q

    def publish_command(self, cmd):
        """Publish Inspire hand command as Float32MultiArray."""
        msg = Float32MultiArray()
        msg.data = [float(v) for v in cmd]
        self.command_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    try:
        node = ManusInspireControlNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
