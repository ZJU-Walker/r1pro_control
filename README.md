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

# Robot teleop guide

# 1. On R1-Pro Robot Computer

ssh galaxea 

password: nvidia

## 1.1 Start Robot Controller

new terminal

```bash
sudo ip link set can0 type can bitrate 1000000 sample-point 0.875 dbitrate 5000000 fd on dsample-point 0.8
sudo ip link set up can0  

cd ~/
bash start_robot.sh
```

---

## 1.2 Start Inspire Hand Controllers

new terminal

```bash
cd ~/inspire_hand

python inspire_left_control.py   # Left hand
python inspire_right_control.py  # Right hand
```

---

# 2. On iris-robot-ws-3

## 2.0 open folders

- `r1pro_teleop`
- `r1pro_control`

## 2.1 Start ZED Node (Camera Streaming)

new terminal

```bash
cd ~/r1pro_teleop/teleop
conda activate teleop
python zed_to_ros_no_wrist.py

# optionally, run following to launch with wrist images
python zed_to_ros.py
```

---

## 2.2 visualize zed images

### Required Components

Before recording, ensure the following are running:

- Robot (only for ROS2 setup, no controller needed so just start the robot don't need to run any scripts or commands)
- ZED node
- Manus node
- Quest teleop node

---

### Start Recording Script

new terminal

```bash
cd ~/r1pro_teleop/demo_record
conda activate teleop
source ~/manus_ws/install/setup.bash

python demo_record_wrist_new_usingpedal_tesollo_inspirehand_everything_single_arm.py 
```

Controls:

- `*` → Start recording
- `*` → Stop recording

## 2.3 Keyboard Control (Manual Arm + Torso Control)

new terminal

```bash
cd ~/r1pro_control/robot_control_ros2
conda activate robot_control
python key_torso_ros2_no_hand.py
```

- Follow terminal instructions for keyboard control
- Used for manual control

## 2.4 Start Manus Glove Node (Hand Teleoperation / Data Collection)

new terminal:

```
cd ~/manus_ws
source install/setup.bash
ros2 run manus_ros2 manus_data_publisher
```

new terminal

```bash
conda activate manus
source ~/manus_ws/install/setup.bash

cd ~/r1pro_control/manus_control

python manus_control_left_inspire_hand.py   # Left hand
python manus_control_right_inspire_hand.py  # Right hand
```

### ⚠️ Troubleshooting (Wrong Hand Mapping)

If glove controls the wrong hand, modify:

```python
'/manus_glove_0'  ↔ '/manus_glove_1'
```

Inside:

```python
self.glove_subscriber = self.create_subscription(
    ManusGlove,
    '/manus_glove_0',
    self.glove_callback,
    10
)
```

---

## 2.5 Start Quest Teleoperation Node (Controller Pose → Robot Frame)

### Step 1: Start ngrok

new terminal

```bash
ngrok http --domain=aria-stream.ngrok.app 8012
```

- Open Quest browser
- Refresh page
- Press enter to start tracking

---

### Step 2: Start Teleop Node

new terminal

```bash
cd ~/r1pro_teleop/teleop
conda activate teleop

python teleop_flipped_controller_with_torso.py
```

- End-effector (EE) pose will start publishing

## 2.6 Pedal control

new terminal

```
conda activate teleop
cd ~/r1pro_teleop/teleop

python pedal_control_ros
```



## 2.6 Start Robot Control Node: Note this will start publishing commands to robot!

new terminal

```
cd ~/r1pro_control/robot_control_ros2
conda activate robot_control

python dual_arm_tele_flipped_controller.py
```

