import pyzed.sl as sl

# Create a Camera object
zed = sl.Camera()

# Open the camera
init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.VGA  # Change as needed
status = zed.open(init)
if status != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED:", status)
    exit(1)

# Get camera calibration parameters
calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
left_cam_params = calibration_params.left_cam

# Intrinsic matrix
fx = left_cam_params.fx
fy = left_cam_params.fy
cx = left_cam_params.cx
cy = left_cam_params.cy

intrinsic_matrix = [
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
]

print("Intrinsic Matrix:")
for row in intrinsic_matrix:
    print(row)

zed.close()
