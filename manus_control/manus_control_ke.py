#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from manus_ros2_msgs.msg import ManusGlove


def set_target_right(q, motor_count=20):
    """
    Calculates the target joint data for a 20-motor robotic hand (DG-5F).

    This function translates input angles from a control device (like Quantum metagloves)
    into target radian values for the robotic hand's motors, applying calibration,
    direction correction, and joint limits.

    Args:
        q (list[float]): A list of 20 input angles in degrees from the control device.
        motor_count (int): The total number of motors. Defaults to 20.

    Returns:
        list[float]: The calculated target joint data (mQd) in radians for each motor.
    """
    # Direction multipliers for each motor. 1 for forward, -1 for reverse.
    dir_vals = [1, -1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1]
    # mGripperCalibrationData = [1.12, 1.45, 1.5, 1.1, 
    #                            1.3, 1.1, 1.1, 1.1, 
    #                            1.3, 1.1, 1.1, 1.1, 
    #                            1.3, 1.3, 1.15, 1.15, 
    #                            2.0, 1.1, 1.1, 1.2]

    # mGripperCalibrationData = [1.0, 1.7, 1.5, 1.1, 
    #                            1, 1.3, 1.0, 0.9, 
    #                            1, 1, 1.05, 1.0, 
    #                            1, 1, 1.05, 1.0, 
    #                            1, 1, 1.05, 1.0 ]

    # mGripperCalibrationData = [1.15, 1.8, 1.95, 1.35, 
    #                            1, 2.0, 0.8, 0.95, 
    #                            1, 2.0, 1.0, 0.7, 
    #                            1, 1, 1.05, 1.0, 
    #                            1, 1, 1.05, 1.0 ]
    mGripperCalibrationData = [1.15, 1.55, 1.6, 1.0, 
                               1, 1.45, 1.05, 0.8, 
                               1, 1.2, 0.7, 0.7, 
                               1, 1, 1.0, 1.0, 
                               1, 1, 1.0, 1.0 ]
    
    
    # qd will store the intermediate radian values.
    qd = [0.0] * motor_count

    # --- Convert input degrees to radians with initial angle offsets ---
    # Note: 58.5 and 20 are initial angles of the Quantum metagloves.
    # qd[0] = (58.5 - q[1]) * (math.pi / 180)
    # qd[1] = (q[0] + 20) * (math.pi / 180)
    qd[0] = (40.5 - q[1]) * (math.pi / 180)
    qd[1] = (q[0] + 40) * (math.pi / 180)
    qd[2] = (q[2]) * (math.pi / 180)
    qd[3] = q[3] * (math.pi / 180)

    qd[4] = q[4] * (math.pi / 180)
    qd[5] = (q[5] + 15) * (math.pi / 180)
    qd[6] = q[6] * (math.pi / 180)
    qd[7] = q[7] * (math.pi / 180)

    qd[8] = q[8] * (math.pi / 180)
    qd[9] = (q[9]+50) * (math.pi / 180)
    qd[10] = q[10] * (math.pi / 180)
    qd[11] = q[11] * (math.pi / 180)

    qd[12] = q[12] * (math.pi / 180)
    qd[13] = q[13] * (math.pi / 180)
    qd[14] = q[14] * (math.pi / 180)
    qd[15] = q[15] * (math.pi / 180)

    # Special handling for the pinky finger based on its bent/straight position.
    # The condition `q[18] > 20` is redundant if `q[18] > 25`, so it's omitted.
    if q[17] > 55 and q[18] > 25:
        qd[16] = abs(q[16]) * 2 * (math.pi / 180)
    else:
        qd[16] = abs(q[16]) / 1.5 * (math.pi / 180)
    
    qd[17] = q[16] * (math.pi / 180)
    qd[18] = q[17] * (math.pi / 180)
    qd[19] = q[19] * (math.pi / 180)

    # --- Final calculation and applying joint limits ---
    mQd = [0.0] * motor_count
    for i in range(motor_count):
        # Apply calibration data and direction multiplier
        mQd[i] = qd[i] * mGripperCalibrationData[i] * dir_vals[i]

        # Apply joint limits to prevent backward bending, translated from the C++ switch statement.
        if i == 1:
            if mQd[i] >= 0:
                mQd[i] = 0
        elif i in [4, 8, 12, 16, 17]:
            # These motors have no special limits in the original code.
            pass
        else: # Default case for all other motors
            if mQd[i] <= 0:
                mQd[i] = 0
                
    return mQd


class ManusControlNode(Node):
    def __init__(self):
        super().__init__('manus_control_node')
        
        # Create subscriber for Manus glove data
        self.glove_subscriber = self.create_subscription(
            ManusGlove,
            '/manus_glove_0',
            self.glove_callback,
            10
        )
        
        # Create publisher for joint trajectory (for right hand DG5F controller)
        self.trajectory_publisher = self.create_publisher(
            JointTrajectory,
            '/dg5f_right_controller/joint_trajectory',
            1
        )
        
        # Joint names for the robotic hand (DG-5F)
        self.joint_names = [
            "rj_dg_1_1", "rj_dg_1_2", "rj_dg_1_3", "rj_dg_1_4",  # Thumb
            "rj_dg_2_1", "rj_dg_2_2", "rj_dg_2_3", "rj_dg_2_4",  # Index
            "rj_dg_3_1", "rj_dg_3_2", "rj_dg_3_3", "rj_dg_3_4",  # Middle
            "rj_dg_4_1", "rj_dg_4_2", "rj_dg_4_3", "rj_dg_4_4",  # Ring
            "rj_dg_5_1", "rj_dg_5_2", "rj_dg_5_3", "rj_dg_5_4"   # Pinky
        ]
        
        # Mapping from ergonomics type names to q array indices
        # Note: The data shows ergonomics_count: 20 but only 19 values are present
        # The missing value appears to be IndexSpread
        self.ergonomics_mapping = {
            'ThumbMCPSpread': 0,
            'ThumbMCPStretch': 1,
            'ThumbPIPStretch': 2,
            'ThumbDIPStretch': 3,
            'IndexSpread': 4,  # This might be missing from the actual data
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
        
        self.get_logger().info('Manus Control Node initialized')
        self.get_logger().info(f'Subscribing to: /manus_glove_1')
        self.get_logger().info(f'Publishing to: /dg5f_right_controller/joint_trajectory')
    
    def glove_callback(self, msg: ManusGlove):
        """Callback function for processing Manus glove data."""
        try:
            # Extract ergonomics data into q array
            q = self.extract_ergonomics_to_q(msg.ergonomics)
            
            # Convert to target joint positions using set_target_right function
            target_positions = set_target_right(q)
            # print(f'Target Positions: {target_positions}')
            
            # Publish the joint trajectory
            self.publish_trajectory(target_positions)
            
        except Exception as e:
            self.get_logger().error(f'Error processing glove data: {e}')
    
    def extract_ergonomics_to_q(self, ergonomics):
        """Extract ergonomics data and map to q array format."""
        q = [0.0] * 20  # Initialize with zeros
        print("pinky control: ", ergonomics[16].value, ergonomics[17].value, ergonomics[18].value, ergonomics[19].value)
        
        # Count actual ergonomics values received
        # print(f'Received {len(ergonomics)} ergonomics values')
        
        # Process each ergonomics entry
        for ergo in ergonomics:
            # Remove any 'type: ' prefix if present
            ergo_type = ergo.type.replace('type: ', '')
            
            if ergo_type in self.ergonomics_mapping:
                index = self.ergonomics_mapping[ergo_type]
                q[index] = ergo.value
            else:
                self.get_logger().debug(f'Unknown ergonomics type: {ergo_type}')
        
        # IndexSpread (index 4) might be missing from the data
        # If it's not provided, it will remain 0.0
        
        # Log the extracted values for debugging
        self.get_logger().debug(f'Extracted q values: {q}')
        
        return q
    
    def publish_trajectory(self, positions):
        """Publish joint trajectory message."""
        # Create trajectory message
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_names
        
        # Create trajectory point
        point = JointTrajectoryPoint()
        # Ensure all positions are floats (not integers)
        point.positions = [float(pos) for pos in positions]
        point.time_from_start = Duration(sec=0, nanosec=30000000)  # 0.03 seconds

        # Add point to trajectory
        trajectory_msg.points.append(point)
        
        # Publish the message
        self.trajectory_publisher.publish(trajectory_msg)
        
        # Log for debugging (comment out in production for performance)
        # self.get_logger().debug(f'Published trajectory with positions: {positions}')


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ManusControlNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()