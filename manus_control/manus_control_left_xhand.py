#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from manus_ros2_msgs.msg import ManusGlove
from sensor_msgs.msg import JointState

import math

def glove_to_xhand_control(glove_joints):
    """
    Convert 20 Manus glove joint values to 12 XHand control values in radians.

    Args:
        glove_joints (list[float]): 20 joint values from the glove (in degrees).

    Returns:
        list[float]: 12 control values for the XHand (in radians).
    """
    if len(glove_joints) != 20:
        raise ValueError(f"Expected 20 glove joint values, got {len(glove_joints)}")

    q_xhand = [0.0] * 12

    # degree command
    # q_xhand[0] = (35 - glove_joints[1]) * 2  # thumb rotate
    q_xhand[0] = glove_joints[0] * 1.55 + (35 - glove_joints[1]) * 1.5  # thumb rotate
    q_xhand[1] = glove_joints[2] * 2.5         # thumb bend1
    q_xhand[2] = glove_joints[3] * 1.1        # thumb bend2

    q_xhand[3] = (glove_joints[4] + 8)         # index bend1
    q_xhand[4] = glove_joints[6] * 1.5          # index bend2
    q_xhand[5] = glove_joints[7] * 1.1         

    q_xhand[6] = glove_joints[10] * 1.5          # middle bend2
    q_xhand[7] = glove_joints[11] * 1.1       

    q_xhand[8] = glove_joints[14] * 1.5
    q_xhand[9] = glove_joints[15] * 1.1       

    q_xhand[10] = glove_joints[18] * 1.5
    q_xhand[11] = glove_joints[19] * 1.1      

    # Convert all values to radians
    q_xhand = [math.radians(v) for v in q_xhand]

    return q_xhand



class ManusToXHandLeft(Node):
    def __init__(self):
        super().__init__('manus_to_xhand_left')

        # Sub: Manus glove data
        self.glove_sub = self.create_subscription(
            ManusGlove,
            '/manus_glove_1',
            self.glove_cb,
            10
        )

        # Pub: XHand left command as JointState (expects 12 positions)
        self.left_pub = self.create_publisher(
            JointState,
            '/xhand/left/control',
            10
        )

        # Map ergonomics to q indices
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

        # Pre-create 12 joint names (placeholder)
        self.joint_names_12 = [f'joint_{i}' for i in range(12)]

        self.get_logger().info('✅ Manus→XHand Left bridge initialized')
        self.get_logger().info('Subscribing: /manus_glove_1')
        self.get_logger().info('Publishing:  /xhand/left/control (JointState, 12 positions)')

    def glove_cb(self, msg: ManusGlove):
        """Callback for incoming Manus glove data."""
        try:
            q = self._extract_q(msg.ergonomics)            # 20 floats from glove
            control_12 = glove_to_xhand_control(q)         # currently all zeros
            print(f'Glove q0: {q[0]}')
            # print(f'XHand control: {control_12}')

            js = JointState()
            js.header.stamp = self.get_clock().now().to_msg()
            js.header.frame_id = 'left_hand'
            js.name = self.joint_names_12
            js.position = [float(x) for x in control_12]

            self.left_pub.publish(js)

        except Exception as e:
            self.get_logger().error(f'Error in glove_cb: {e}')

    def _extract_q(self, ergonomics):
        """Extract 20 joint values from the glove ergonomics message."""
        q = [0.0] * 20
        for ergo in ergonomics:
            etype = ergo.type.replace('type: ', '')
            idx = self.ergonomics_mapping.get(etype)
            if idx is not None and 0 <= idx < 20:
                q[idx] = ergo.value
        return q


def main(args=None):
    rclpy.init(args=args)
    node = ManusToXHandLeft()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
