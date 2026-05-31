#!/usr/bin/env python3
'''
ROS2 script to visualize teleoperation coordinate frames in real-time using Open3D.
- Subscribes to the raw controller pose and the final target arm pose.
- Renders a 3D window showing only the frames from these two topics.
'''

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import threading
import copy

# Helper function to convert a ROS Pose message to a 4x4 transformation matrix
def pose_to_T(pose):
    """Convert a geometry_msgs/Pose to a 4x4 transformation matrix."""
    p = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
    q = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w], dtype=float)
    
    T = np.eye(4, dtype=float)
    T[0:3, 0:3] = R.from_quat(q).as_matrix()
    T[0:3, 3] = p
    return T

class Open3DVisualizer(Node):
    def __init__(self):
        super().__init__('open3d_visualizer')
        
        # --- ROS Subscriptions ---
        self.create_subscription(
            PoseStamped,
            '/teleop/left_ee_raw_pose',
            self.controller_pose_callback,
            10)
        
        self.create_subscription(
            PoseStamped,
            '/motion_targetsss/target_pose_arm_left',
            self.target_pose_callback,
            10)

        # --- Data Storage & Threading ---
        self.lock = threading.Lock()
        self.T_controller = np.eye(4)
        self.T_final_target = np.eye(4)
        self.controller_pose_updated = False
        self.final_target_pose_updated = False

        # --- Open3D Initialization ---
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='Teleop Frame Visualization', width=1280, height=720)

        # Create "pristine" versions of the coordinate frame geometries
        self.base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        self.controller_frame_pristine = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        self.final_target_frame_pristine = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12)

        # Add the static base frame to the scene
        self.vis.add_geometry(self.base_frame)
        
        # Set the default camera view
        view_control = self.vis.get_view_control()
        view_control.set_lookat([0.5, 0.0, 0.0])
        view_control.set_up([0.0, 0.0, 1.0])
        view_control.set_front([1.0, 0.0, 0.0])
        view_control.set_zoom(0.8)
        
        self.get_logger().info("✅ Open3D Visualizer node is running.")

    def controller_pose_callback(self, msg):
        """Callback for the raw controller pose."""
        with self.lock:
            self.T_controller = pose_to_T(msg.pose)
            self.controller_pose_updated = True

    def target_pose_callback(self, msg):
        """Callback for the final target arm pose."""
        with self.lock:
            self.T_final_target = pose_to_T(msg.pose)
            self.final_target_pose_updated = True

    def run_visualization(self):
        """Main loop to handle ROS callbacks and render the Open3D scene."""
        self.get_logger().info("Starting Open3D visualization loop... Close the window to exit.")
        
        # Create display copies for each frame
        display_controller = copy.deepcopy(self.controller_frame_pristine)
        display_final_target = copy.deepcopy(self.final_target_frame_pristine)

        # Add the geometries to the visualizer
        self.vis.add_geometry(display_controller)
        self.vis.add_geometry(display_final_target)
        
        while True:
            rclpy.spin_once(self, timeout_sec=0.001)

            with self.lock:
                # Update controller frame if new data arrived
                if self.controller_pose_updated:
                    display_controller.vertices = self.controller_frame_pristine.vertices
                    display_controller.triangles = self.controller_frame_pristine.triangles
                    display_controller.vertex_colors = self.controller_frame_pristine.vertex_colors
                    display_controller.compute_vertex_normals()

                    display_controller.transform(self.T_controller)
                    self.vis.update_geometry(display_controller)
                    self.controller_pose_updated = False

                # Update the final target frame if new data arrived
                if self.final_target_pose_updated:
                    display_final_target.vertices = self.final_target_frame_pristine.vertices
                    display_final_target.triangles = self.final_target_frame_pristine.triangles
                    display_final_target.vertex_colors = self.final_target_frame_pristine.vertex_colors
                    display_final_target.compute_vertex_normals()
                    
                    display_final_target.transform(self.T_final_target)
                    self.vis.update_geometry(display_final_target)
                    self.final_target_pose_updated = False

            if not self.vis.poll_events():
                break
            self.vis.update_renderer()

        self.vis.destroy_window()
        self.get_logger().info("Visualization window closed.")

def main(args=None):
    rclpy.init(args=args)
    node = Open3DVisualizer()
    try:
        node.run_visualization()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()