import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from math import degrees
import open3d as o3d

# Load data
df = pd.read_csv("/home/irislab/r1pro_control/camera_calibration_ros2/pose_data_cam_left_right_arm.csv")

# Extract camera and base frame points
cam_points = df[['tvec_x', 'tvec_y', 'tvec_z']].values  # aruco
base_points = df[['ee_x', 'ee_y', 'ee_z']].values  # robot

def compute_rigid_transform_open3d(cam_points, base_points):
    """
    Estimate rigid transform from camera frame to base frame using Open3D.
    cam_points: Nx3 numpy array (in camera frame)
    base_points: Nx3 numpy array (in base frame)
    Returns:
        T_base_cam: 4x4 transformation matrix
    """
    assert cam_points.shape == base_points.shape
    N = cam_points.shape[0]

    # Convert to Open3D point clouds
    pcd_cam = o3d.geometry.PointCloud()
    pcd_base = o3d.geometry.PointCloud()
    pcd_cam.points = o3d.utility.Vector3dVector(cam_points)
    pcd_base.points = o3d.utility.Vector3dVector(base_points)

    # Generate correspondences (i to i)
    corres = np.array([[i, i] for i in range(N)])
    corres = o3d.utility.Vector2iVector(corres)

    # Estimate transformation
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    T_base_cam = estimation.compute_transformation(pcd_cam, pcd_base, corres)

    return T_base_cam

def compute_rigid_transform_ransac_open3d(cam_points, base_points, threshold=0.02, max_iterations=2000):
    """
    Estimate rigid transform from camera frame to base frame using Open3D's RANSAC.
    
    Parameters:
    - cam_points: Nx3 numpy array (in camera frame)
    - base_points: Nx3 numpy array (in base frame)
    - threshold: RANSAC threshold distance (meters)
    - max_iterations: maximum RANSAC iterations
    
    Returns:
    - transformation: 4x4 transformation matrix
    - correspondence_set: correspondence indices of inliers
    """
    assert cam_points.shape == base_points.shape
    N = cam_points.shape[0]
    
    # Convert to Open3D point clouds
    pcd_cam = o3d.geometry.PointCloud()
    pcd_base = o3d.geometry.PointCloud()
    pcd_cam.points = o3d.utility.Vector3dVector(cam_points)
    pcd_base.points = o3d.utility.Vector3dVector(base_points)
    
    # Create correspondence pairs (assuming points are already paired)
    correspondences = o3d.utility.Vector2iVector(
        np.array([[i, i] for i in range(N)])
    )
    
    # Use RANSAC for robust estimation
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source=pcd_cam,
        target=pcd_base,
        corres=correspondences,
        max_correspondence_distance=threshold,
        # max_correspondence_distance=np.inf,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=4,  # minimum number of points for estimation
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=max_iterations,
            confidence=0.999
        )
    )
    
    return result.transformation, result.correspondence_set

def apply_transform(points, T):
    """Apply 4x4 transformation matrix to points."""
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed = (T @ points_h.T).T
    return transformed[:, :3]

def analyze_ransac_results(cam_points, base_points, transformation, correspondence_set):
    """Analyze RANSAC results using Open3D output."""
    n_points = len(cam_points)
    n_inliers = len(correspondence_set)
    
    # Create inlier mask - handle the read-only array issue
    inlier_mask = np.zeros(n_points, dtype=bool)
    
    # Convert correspondence_set to numpy array to avoid read-only issues
    if hasattr(correspondence_set, '__len__'):
        corr_array = np.asarray(correspondence_set).copy()
        for i in range(len(corr_array)):
            inlier_mask[corr_array[i][0]] = True
    
    # Make a copy of the transformation matrix to avoid read-only issues
    transformation = np.array(transformation, copy=True)
    
    # Compute errors
    T_cam_base = np.linalg.inv(transformation)
    estimated_cam_points = apply_transform(base_points, T_cam_base)
    errors = np.linalg.norm(estimated_cam_points - cam_points, axis=1)
    
    print("\n=== Open3D RANSAC Results ===")
    print(f"Total points: {n_points}")
    print(f"Inliers: {n_inliers} ({100*n_inliers/n_points:.1f}%)")
    print(f"Outliers: {n_points - n_inliers} ({100*(n_points - n_inliers)/n_points:.1f}%)")
    
    print(f"\nReprojection Error Statistics (all points):")
    print(f"  Mean error: {np.mean(errors):.4f} m ({np.mean(errors)*1000:.2f} mm)")
    print(f"  Std deviation: {np.std(errors):.4f} m ({np.std(errors)*1000:.2f} mm)")
    print(f"  Min error: {np.min(errors):.4f} m ({np.min(errors)*1000:.2f} mm)")
    print(f"  Max error: {np.max(errors):.4f} m ({np.max(errors)*1000:.2f} mm)")
    
    if n_inliers > 0:
        print(f"\nReprojection Error Statistics (inliers only):")
        inlier_errors = errors[inlier_mask]
        if len(inlier_errors) > 0:
            print(f"  Mean error: {np.mean(inlier_errors):.4f} m ({np.mean(inlier_errors)*1000:.2f} mm)")
            print(f"  Std deviation: {np.std(inlier_errors):.4f} m ({np.std(inlier_errors)*1000:.2f} mm)")
            print(f"  Min error: {np.min(inlier_errors):.4f} m ({np.min(inlier_errors)*1000:.2f} mm)")
            print(f"  Max error: {np.max(inlier_errors):.4f} m ({np.max(inlier_errors)*1000:.2f} mm)")
    
    # Extract rotation and translation - make copies to avoid read-only issues
    rotation_matrix = np.array(transformation[:3, :3], copy=True)
    translation = np.array(transformation[:3, 3], copy=True)
    
    # Convert rotation to Euler angles
    r = R.from_matrix(rotation_matrix)
    euler_angles = r.as_euler('xyz', degrees=True)
    
    print(f"\nTransformation from camera to robot base (T_base_cam):")
    print(f"  Translation: [{translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}] m")
    print(f"  Rotation (Euler XYZ): [{euler_angles[0]:.2f}°, {euler_angles[1]:.2f}°, {euler_angles[2]:.2f}°]")
    
    return inlier_mask, errors

def visualize_calibration_results(cam_points, base_points, transformation, inlier_mask):
    """Visualize the calibration results using Open3D."""
    # Create point clouds
    pcd_cam = o3d.geometry.PointCloud()
    pcd_base = o3d.geometry.PointCloud()
    
    # Set points
    pcd_cam.points = o3d.utility.Vector3dVector(cam_points)
    pcd_base.points = o3d.utility.Vector3dVector(base_points)
    
    # Color inliers green and outliers red for camera points
    colors_cam = np.zeros((len(cam_points), 3))
    colors_cam[inlier_mask] = [0, 1, 0]  # Green for inliers
    colors_cam[~inlier_mask] = [1, 0, 0]  # Red for outliers
    pcd_cam.colors = o3d.utility.Vector3dVector(colors_cam)
    
    # Color base points blue
    colors_base = np.zeros((len(base_points), 3))
    colors_base[:] = [0, 0, 1]  # Blue
    pcd_base.colors = o3d.utility.Vector3dVector(colors_base)
    
    # Transform camera points to base frame for visualization
    pcd_cam_transformed = o3d.geometry.PointCloud()
    pcd_cam_transformed.points = pcd_cam.points
    pcd_cam_transformed.colors = pcd_cam.colors
    pcd_cam_transformed.transform(transformation)
    
    # Create coordinate frames
    coord_frame_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    coord_frame_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    coord_frame_base.transform(transformation)
    
    print("\nVisualization Legend:")
    print("- Green points: Inliers (camera frame)")
    print("- Red points: Outliers (camera frame)")
    print("- Blue points: Robot base frame")
    print("- Coordinate frames show the transformation")
    
    # Visualize
    o3d.visualization.draw_geometries(
        [pcd_cam_transformed, pcd_base, coord_frame_cam, coord_frame_base],
        window_name="Calibration Results"
    )

# Main execution
if __name__ == "__main__":
    # Using Open3D's built-in RANSAC
    print("=== Using Open3D's Built-in RANSAC ===")

    ransac_threshold = 0.01  # 10mm threshold
    max_iterations = 800000000
    
    try:
        # Method 1: Using correspondence-based RANSAC
        T_ransac, correspondence_set = compute_rigid_transform_ransac_open3d(
            cam_points, 
            base_points,
            threshold=ransac_threshold,
            max_iterations=max_iterations
        )
        
        # Make a copy of the transformation matrix to avoid read-only issues
        T_ransac = np.array(T_ransac, copy=True)
        
        print("Open3D RANSAC Transformation matrix (T_base_cam):")
        print(T_ransac)
        
        # Analyze results
        inlier_mask, errors = analyze_ransac_results(cam_points, base_points, T_ransac, correspondence_set)
        
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()