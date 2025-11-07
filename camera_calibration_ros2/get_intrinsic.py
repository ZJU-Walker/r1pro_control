import pyzed.sl as sl

zed = sl.Camera()
init_params = sl.InitParameters()
# (Optional) lock to a specific mode so intrinsics match it:
init_params.camera_resolution = sl.RESOLUTION.HD720  # HD720 for good latency/quality tradeoff
init_params.camera_fps = 30

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    raise SystemExit("Failed to open ZED")

cam_info = zed.get_camera_information()  # no 'resolution=' kwarg here
cfg = cam_info.camera_configuration
calib = cfg.calibration_parameters       # or .calibration_parameters_raw for raw (unrectified)

left  = calib.left_cam
right = calib.right_cam

print("Resolution:", cfg.resolution.width, "x", cfg.resolution.height)
print("LEFT  fx, fy, cx, cy:", left.fx, left.fy, left.cx, left.cy)
print("LEFT  disto (k1,k2,p1,p2,k3):", left.disto)
print("RIGHT fx, fy, cx, cy:", right.fx, right.fy, right.cx, right.cy)
print("RIGHT disto:", right.disto)

zed.close()
