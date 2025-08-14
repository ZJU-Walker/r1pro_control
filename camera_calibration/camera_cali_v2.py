import numpy as np

# Rotation matrix you provided
R_ee_to_tag = np.array([
    [8.66025404e-01, 1.11022302e-16, -5.00000000e-01],
    [5.00000000e-01, 1.92296269e-16, 8.66025404e-01],
    [0.00000000e+00, -1.00000000e+00, 2.22044605e-16]
])

# Translation is zero
t_ee_to_tag = np.array([0.0, 0.0, 0.0])

# Create the 4x4 homogeneous transformation matrix
T_ee_to_tag = np.eye(4)
T_ee_to_tag[:3, :3] = R_ee_to_tag
T_ee_to_tag[:3, 3] = t_ee_to_tag

print("Transformation from End-Effector to Tag (Y):")
print(T_ee_to_tag)


# read csv
import pandas as pd
import cv2
from scipy.spatial.transform import Rotation as ScipyRotation

# Load your data
df = pd.read_csv('/home/iris/humanoid_robot/camera_calibration/camera_cali/camera_20250718_right.csv')

base_to_ee_list = []
cam_to_tag_list = []

for index, row in df.iterrows():
    # 1. Create the base-to-ee transformation matrix
    t_base_to_ee = np.array([row['ee_x'], row['ee_y'], row['ee_z']])
    # Note: SciPy quaternion format is (x, y, z, w)
    q_base_to_ee = np.array([row['ee_qx'], row['ee_qy'], row['ee_qz'], row['ee_qw']])
    R_base_to_ee = ScipyRotation.from_quat(q_base_to_ee).as_matrix()
    
    T_base_to_ee = np.eye(4)
    T_base_to_ee[:3, :3] = R_base_to_ee
    T_base_to_ee[:3, 3] = t_base_to_ee
    base_to_ee_list.append(T_base_to_ee)

    # 2. Create the camera-to-tag transformation matrix
    t_cam_to_tag = np.array([row['tvec_x'], row['tvec_y'], row['tvec_z']])
    rvec_cam_to_tag = np.array([row['rvec_x'], row['rvec_y'], row['rvec_z']])
    # Use OpenCV to convert Rodrigues vector to rotation matrix
    R_cam_to_tag, _ = cv2.Rodrigues(rvec_cam_to_tag)

    T_cam_to_tag = np.eye(4)
    T_cam_to_tag[:3, :3] = R_cam_to_tag
    T_cam_to_tag[:3, 3] = t_cam_to_tag
    cam_to_tag_list.append(T_cam_to_tag)
    
# : Calculate the Tag's Pose in the Base Frame
base_to_tag_list = []
for T_base_to_ee in base_to_ee_list:
    T_base_to_tag = T_base_to_ee @ T_ee_to_tag # Matrix multiplication
    base_to_tag_list.append(T_base_to_tag)
    
# Prepare the lists of rotations and translations for the solver
R_base_to_tag_list = [T[:3, :3] for T in base_to_tag_list]
t_base_to_tag_list = [T[:3, 3] for T in base_to_tag_list]

R_cam_to_tag_list = [T[:3, :3] for T in cam_to_tag_list]
t_cam_to_tag_list = [T[:3, 3] for T in cam_to_tag_list]

# Solve for the transformation
# The OpenCV function finds the transformation from the gripper to the base
# and the target to the camera. We use our tag poses as the "gripper".
# The function returns the transformation from camera to base.
R_cam_to_base, t_cam_to_base = cv2.calibrateHandEye(
    R_gripper2base=R_base_to_tag_list,
    t_gripper2base=t_base_to_tag_list,
    R_target2cam=R_cam_to_tag_list,
    t_target2cam=t_cam_to_tag_list,
    method=cv2.CALIB_HAND_EYE_TSAI # A common and robust method
)

# The result is R_cam_to_base and t_cam_to_base. We want the inverse: base_to_camera.
T_cam_to_base = np.eye(4)
T_cam_to_base[:3, :3] = R_cam_to_base
T_cam_to_base[:3, 3] = t_cam_to_base.flatten()

# Invert the matrix to get the final transformation you want
T_base_to_camera = np.linalg.inv(T_cam_to_base)

print("\n## Final Transformation: Base to Camera (X) ##")
print(T_base_to_camera)

T_cam_to_base = np.linalg.inv(T_base_to_camera)
print("\n## Final Transformation: Camera to Base (X) ##")
print(T_cam_to_base)