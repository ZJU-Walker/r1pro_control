# R1 Pro Robot Control

A comprehensive ROS2-based control system for the R1 Pro humanoid robot, including teleoperation, camera calibration, hand control, and policy deployment.

## Quick Start

### Step 1: Create conda environment
```bash
conda create -n robot_control python=3.10 -y
conda activate robot_control
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

## Project Structure

### Robot Control (`robot_control_ros2/`)
Main teleoperation and control scripts:

#### Teleoperation
- **`left_arm_tele_ros2.py`** - Left arm teleoperation with ROS2
- **`right_arm_tele_ros2.py`** - Right arm teleoperation with ROS2
- **`dual_arm_tele.py`** - Dual arm teleoperation
- **`dual_arm_tele_mirror.py`** - Dual arm mirror teleoperation
- **`dual_arm_tele_flipped_controller.py`** - Dual arm with flipped controller
- **`left_arm_tele_ros2_quest_only.py`** - Left arm Quest-only control
- **`right_arm_tele_ros2_quest_only.py`** - Right arm Quest-only control

#### Hand Control
- **`hand_control.py`** - Direct hand control interface
- **`control_hand_single_joint.py`** - Single joint hand control
- **`replay_hands.py`** - Hand trajectory replay
- **`replay_arm_and_hands.py`** - Combined arm and hand replay
- **`replay_arm_and_hands_egodex.py`** - Replay with EgoDex data

#### Recording & Replay
- **`record_wrist_trajectory.py`** - Record wrist trajectories
- **`record_mirror_command_position.py`** - Record mirror command positions
- **`right_arm_replay_human.py`** - Replay human demonstrations

#### Utilities
- **`key_torso_ros2_no_hand.py`** - Keyboard control for torso (no hand)
- **`key_torso_ros2_w_inspire_hand.py`** - Keyboard control with Inspire hand
- **`log_control.py`** - Control logging utilities
- **`vis_wrist_ori.py`** - Visualize wrist orientation

### Camera Calibration (`camera_calibration_ros2/`)
Camera calibration and ArUco marker detection:
- **`aruco_ee_detect_ros2.py`** - ArUco end-effector detection with ROS2
- **`aruco_detect_zed.py`** - ArUco detection with ZED camera
- **`aruco_detect_zed_2cams.py`** - ArUco detection with dual ZED cameras
- **`camera_cali_v2.py`** - Camera calibration v2
- **`camera_cali_ransac.py`** - RANSAC-based camera calibration
- **`get_intrinsic.py`** - Extract camera intrinsic parameters
- **`check.py`** - Calibration verification

### Policy Deployment (`policy_deploy/`)
ACT-based policy deployment modules for various dates/versions

### Additional Modules
- **`manus_control/`** - Manus VR glove control integration
- **`egodex_load/`** - EgoDex dataset loading utilities
- **`tesollo_ik/`** - Inverse kinematics for Tesollo robot
- **`tip_to_joints/`** - Fingertip to joint mapping utilities
- **`urdf/`** - Robot URDF models

## Usage Examples

### Run teleoperation
```bash
# Single arm teleoperation
python robot_control_ros2/left_arm_tele_ros2.py
python robot_control_ros2/right_arm_tele_ros2.py

# Dual arm teleoperation
python robot_control_ros2/dual_arm_tele.py
```

### Camera calibration
```bash
# Run ArUco detection with ZED camera
python camera_calibration_ros2/aruco_detect_zed.py

# Perform camera calibration
python camera_calibration_ros2/camera_cali_v2.py
```

### Hand control
```bash
# Control hands directly
python robot_control_ros2/hand_control.py

# Replay hand trajectories
python robot_control_ros2/replay_hands.py
```

## Requirements
- Python 3.10
- ROS2 (Humble recommended)
- OpenCV (< version 5)
- NumPy
- transforms3d
- readchar

## Notes
- Recorded trajectories are stored in `robot_control_ros2/recorded_trajectories/`
- Calibration data is stored in `camera_calibration_ros2/camera_cali/`
- Make sure ROS2 workspace is sourced before running scripts