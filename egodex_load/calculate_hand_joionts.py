#!/usr/bin/env python3
import os
import csv
import numpy as np
from scipy.spatial.transform import Rotation as R

# --------- edit these two lines ----------
INPUT_CSV  = "5_egodex_example.csv"
OUTPUT_CSV = "egodex_robotcmd.csv"
# Camera extrinsic (camera pose in base/world) = T_base_cam0
T_base_cam0 = np.array([
    [ 0.01988061, -0.43758429,  0.89895759,  0.14056752],
    [-0.99969330,  0.00457983,  0.02433772,  0.02539622],
    [-0.01476688, -0.89916573, -0.43735903,  0.43713101],
    [ 0.0,         0.0,         0.0,         1.0       ]
], dtype=np.float32)
# -----------------------------------------

LEFT_HAND_25 = [
    "leftThumbKnuckle","leftThumbIntermediateBase","leftThumbIntermediateTip","leftThumbTip",
    "leftIndexFingerMetacarpal","leftIndexFingerKnuckle","leftIndexFingerIntermediateBase","leftIndexFingerIntermediateTip","leftIndexFingerTip",
    "leftMiddleFingerMetacarpal","leftMiddleFingerKnuckle","leftMiddleFingerIntermediateBase","leftMiddleFingerIntermediateTip","leftMiddleFingerTip",
    "leftRingFingerMetacarpal","leftRingFingerKnuckle","leftRingFingerIntermediateBase","leftRingFingerIntermediateTip","leftRingFingerTip",
    "leftLittleFingerMetacarpal","leftLittleFingerKnuckle","leftLittleFingerIntermediateBase","leftLittleFingerIntermediateTip","leftLittleFingerTip",
]
RIGHT_HAND_25 = [
    "rightThumbKnuckle","rightThumbIntermediateBase","rightThumbIntermediateTip","rightThumbTip",
    "rightIndexFingerMetacarpal","rightIndexFingerKnuckle","rightIndexFingerIntermediateBase","rightIndexFingerIntermediateTip","rightIndexFingerTip",
    "rightMiddleFingerMetacarpal","rightMiddleFingerKnuckle","rightMiddleFingerIntermediateBase","rightMiddleFingerIntermediateTip","rightMiddleFingerTip",
    "rightRingFingerMetacarpal","rightRingFingerKnuckle","rightRingFingerIntermediateBase","rightRingFingerIntermediateTip","rightRingFingerTip",
    "rightLittleFingerMetacarpal","rightLittleFingerKnuckle","rightLittleFingerIntermediateBase","rightLittleFingerIntermediateTip","rightLittleFingerTip",
]

EXPECTED_LEFT_JOINT_ORDER = [
    "lj_dg_1_1","lj_dg_1_2","lj_dg_1_3","lj_dg_1_4",
    "lj_dg_2_1","lj_dg_2_2","lj_dg_2_3","lj_dg_2_4",
    "lj_dg_3_1","lj_dg_3_2","lj_dg_3_3","lj_dg_3_4",
    "lj_dg_4_1","lj_dg_4_2","lj_dg_4_3","lj_dg_4_4",
    "lj_dg_5_1","lj_dg_5_2","lj_dg_5_3","lj_dg_5_4"
]
EXPECTED_RIGHT_JOINT_ORDER = [
    "rj_dg_1_1","rj_dg_1_2","rj_dg_1_3","rj_dg_1_4",
    "rj_dg_2_1","rj_dg_2_2","rj_dg_2_3","rj_dg_2_4",
    "rj_dg_3_1","rj_dg_3_2","rj_dg_3_3","rj_dg_3_4",
    "rj_dg_4_1","rj_dg_4_2","rj_dg_4_3","rj_dg_4_4",
    "rj_dg_5_1","rj_dg_5_2","rj_dg_5_3","rj_dg_5_4"
]

JOINT_LIMITS = {
    "lj_dg_1_1": (-0.890, 0.384), "lj_dg_1_2": (0.0, 3.142), "lj_dg_1_3": (-1.571, 1.571), "lj_dg_1_4": (-1.571, 1.571),
    "lj_dg_2_1": (-0.611, 0.419), "lj_dg_2_2": (0.0, 2.007), "lj_dg_2_3": (-1.571, 1.571), "lj_dg_2_4": (-1.571, 1.571),
    "lj_dg_3_1": (-0.611, 0.611), "lj_dg_3_2": (0.0, 1.955), "lj_dg_3_3": (-1.571, 1.571), "lj_dg_3_4": (-1.571, 1.571),
    "lj_dg_4_1": (-0.419, 0.611), "lj_dg_4_2": (0.0, 1.902), "lj_dg_4_3": (-1.571, 1.571), "lj_dg_4_4": (-1.571, 1.571),
    "lj_dg_5_1": (-1.047, 0.017), "lj_dg_5_2": (-0.611, 0.419), "lj_dg_5_3": (-1.571, 1.571), "lj_dg_5_4": (-1.571, 1.571),
    "rj_dg_1_1": (-0.384, 0.890), "rj_dg_1_2": (-3.142, 0.0), "rj_dg_1_3": (-1.571, 1.571), "rj_dg_1_4": (-1.571, 1.571),
    "rj_dg_2_1": (-0.419, 0.611), "rj_dg_2_2": (0.0, 2.007), "rj_dg_2_3": (-1.571, 1.571), "rj_dg_2_4": (-1.571, 1.571),
    "rj_dg_3_1": (-0.611, 0.611), "rj_dg_3_2": (0.0, 1.955), "rj_dg_3_3": (-1.571, 1.571), "rj_dg_3_4": (-1.571, 1.571),
    "rj_dg_4_1": (-0.611, 0.419), "rj_dg_4_2": (0.0, 1.902), "rj_dg_4_3": (-1.571, 1.571), "rj_dg_4_4": (-1.571, 1.571),
    "rj_dg_5_1": (-0.017, 1.047), "rj_dg_5_2": (-0.419, 0.611), "rj_dg_5_3": (-1.571, 1.571), "rj_dg_5_4": (-1.571, 1.571),
}

# ---------- small helpers ----------
def se3_from_pos_quat(pos_xyz, quat_xyzw):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.from_quat(quat_xyzw).as_matrix().astype(np.float32)
    T[:3, 3]  = np.asarray(pos_xyz, dtype=np.float32)
    return T

def pos_quat_from_se3(T):
    pos = T[:3, 3].astype(np.float32)
    quat = R.from_matrix(T[:3, :3]).as_quat().astype(np.float32)  # (x,y,z,w)
    return pos, quat

def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c,-s],
                     [0, s, c]], dtype=np.float32)

def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0,  s],
                     [ 0, 1,  0],
                     [-s, 0,  c]], dtype=np.float32)

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c,-s, 0],
                     [ s, c, 0],
                     [ 0, 0, 1]], dtype=np.float32)

def angle_between(v1, v2):
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    v1 /= n1; v2 /= n2
    return float(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))

def compute_finger_angles(finger_mat):
    p0, p1, p2, p3, p4 = finger_mat
    v0, v1, v2, v3 = p1-p0, p2-p1, p3-p2, p4-p3
    palm_normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # +X
    v0_zy = v0.copy(); v0_zy[0] = 0.0
    v1_zy = v1.copy(); v1_zy[0] = 0.0
    if np.linalg.norm(v0_zy) == 0 or np.linalg.norm(v1_zy) == 0:
        theta_mcp_abd = 0.0
        print("Warning: zero-length vector in compute_finger_angles")
    else:
        v0_zy /= np.linalg.norm(v0_zy)
        v1_zy /= np.linalg.norm(v1_zy)
        ang = np.arccos(np.clip(np.dot(v0_zy, v1_zy), -1.0, 1.0))
        cross = np.cross(v0_zy, v1_zy)
        theta_mcp_abd = float(np.sign(np.dot(cross, palm_normal)) * ang)
    return {
        'mcp_abd_rad': theta_mcp_abd,
        'mcp_flex_rad': angle_between(v0, v1),
        'pip_rad': angle_between(v1, v2),
        'dip_rad': angle_between(v2, v3),
    }

def compute_thumb_angles(finger_mat, hand):
    p0, p1, p2, p3, p4 = finger_mat
    v0, v1, v2, v3 = p1-p0, p2-p1, p3-p2, p4-p3
    v1_zy = v1.copy(); v1_zy[0] = 0.0
    if np.linalg.norm(v1_zy) == 0:
        theta_mcp_0 = 0.0
    else:
        v1_zy /= np.linalg.norm(v1_zy)
        if hand == 'left':
            theta_mcp_0 = np.radians(30.0) - angle_between(v1_zy, [0.0,-1.0,0.0])
        else:
            theta_mcp_0 = -np.radians(30.0) + angle_between(v1_zy, [0.0, 1.0,0.0])
    v0_xy = v0.copy(); v0_xy[2] = 0.0
    if np.linalg.norm(v0_xy) == 0:
        theta_mcp_1 = np.radians(90.0)
    else:
        v0_xy /= np.linalg.norm(v0_xy)
        if hand == 'left':
            theta_mcp_1 = angle_between(v0_xy, [0.0,-1.0,0.0])
        else:
            theta_mcp_1 = -angle_between(v0_xy, [0.0, 1.0,0.0])
    theta_pip = angle_between(v1, v2)
    theta_dip = angle_between(v2, v3)
    v1_zy2 = np.array([v1[1], v1[2]], dtype=np.float32)
    v2_zy2 = np.array([v2[1], v2[2]], dtype=np.float32)
    v3_zy2 = np.array([v3[1], v3[2]], dtype=np.float32)
    if np.cross(v1_zy2, v2_zy2) < 0: theta_pip *= -1.0
    if np.cross(v2_zy2, v3_zy2) < 0: theta_dip *= -1.0
    return {'mcp_0': theta_mcp_0, 'mcp_1': theta_mcp_1, 'pip_rad': theta_pip, 'dip_rad': theta_dip}

def clamp_joint_positions(qpos, joint_names):
    q = np.array(qpos, dtype=np.float32)
    q[~np.isfinite(q)] = 0.0
    margin = np.radians(1.0)
    for i, name in enumerate(joint_names):
        lo, hi = JOINT_LIMITS[name]
        q[i] = np.clip(q[i], lo + margin, hi - margin)
    return q

def load_csv_rows(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))

def read_point(row, name):
    return np.array([
        float(row.get(f"{name}_x", "nan")),
        float(row.get(f"{name}_y", "nan")),
        float(row.get(f"{name}_z", "nan")),
    ], dtype=np.float32)

def build_headers():
    cols = ["t"]
    for arm in ["left", "right"]:
        cols += [f"{arm}_wrist_x", f"{arm}_wrist_y", f"{arm}_wrist_z",
                 f"{arm}_wrist_qx", f"{arm}_wrist_qy", f"{arm}_wrist_qz", f"{arm}_wrist_qw"]
    for name in LEFT_HAND_25:
        cols += [f"{name}_x", f"{name}_y", f"{name}_z"]
    for name in RIGHT_HAND_25:
        cols += [f"{name}_x", f"{name}_y", f"{name}_z"]
    for arm in ["left", "right"]:
        cols += [f"{arm}_hand_joint_{i}" for i in range(1, 21)]
    return cols

def main():
    assert os.path.exists(INPUT_CSV), f"Missing input CSV: {INPUT_CSV}"
    rows = load_csv_rows(INPUT_CSV)

    R_base_cam = T_base_cam0[:3, :3].astype(np.float32)
    t_base_cam = T_base_cam0[:3, 3].astype(np.float32)

    wrist_cols = {
        "left":  dict(px="leftHand_x",  py="leftHand_y",  pz="leftHand_z",
                      qx="leftHand_qx", qy="leftHand_qy", qz="leftHand_qz", qw="leftHand_qw"),
        "right": dict(px="rightHand_x", py="rightHand_y", pz="rightHand_z",
                      qx="rightHand_qx", qy="rightHand_qy", qz="rightHand_qz", qw="rightHand_qw"),
    }

    # These indices (0..24) assume a 25-length array where index 0 is the "CMC".
    # By prepending the wrist as CMC, the rest aligns naturally with your original mapping.
    L_idx = {"index":(5,6,7,8,9), "middle":(10,11,12,13,14), "ring":(15,16,17,18,19), "pinky":(20,21,22,23,24), "thumb":(0,1,2,3,4)}
    R_idx = L_idx

    with open(OUTPUT_CSV, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(build_headers())

        for row in rows:
            t = int(float(row["t"]))

            def wrist_T_from_row(arm):
                # TODO: change the input source for the wrist pose
                # here I read from the CSV input
                pos = np.array([float(row[wrist_cols[arm]["px"]]),
                                float(row[wrist_cols[arm]["py"]]),
                                float(row[wrist_cols[arm]["pz"]])], dtype=np.float32)
                quat = np.array([float(row[wrist_cols[arm]["qx"]]),
                                 float(row[wrist_cols[arm]["qy"]]),
                                 float(row[wrist_cols[arm]["qz"]]),
                                 float(row[wrist_cols[arm]["qw"]])], dtype=np.float32)
                T_cam = se3_from_pos_quat(pos, quat)
                return T_base_cam0 @ T_cam

            T_base_left  = wrist_T_from_row("left")
            T_base_right = wrist_T_from_row("right")

            l_pos_b = T_base_left[:3, 3].copy()
            r_pos_b = T_base_right[:3, 3].copy()

            # TODO add offset here
            l_pos_b[0] += 0.15
            r_pos_b[0] += 0.05

            # keypoints in base (24 points) — saved to CSV as-is
            # TODO: change these 24 hand keypoints input source, you can directly get from egodex output
            # here I read from the CSV input
            left25_cam  = np.stack([read_point(row, n) for n in LEFT_HAND_25], axis=0)
            right25_cam = np.stack([read_point(row, n) for n in RIGHT_HAND_25], axis=0)
            left24_base  = (R_base_cam @ left25_cam.T).T + t_base_cam
            right24_base = (R_base_cam @ right25_cam.T).T + t_base_cam

            # local for angles
            Rw_L, tw_L = T_base_left[:3, :3],  T_base_left[:3, 3]
            Rw_R, tw_R = T_base_right[:3, :3], T_base_right[:3, 3]
            left24_local  = (Rw_L.T @ (left24_base  - tw_L).T).T
            right24_local = (Rw_R.T @ (right24_base - tw_R).T).T

            # wrist-convention for ANGLES (intrinsic: Y then new Z)
            Ry_left, Ry_right = rot_y(+np.pi/2), rot_y(-np.pi/2) #TODO
            Rz_after = rot_z(+np.pi/2)
            R_y_final = rot_z(np.pi)
            Rc_left_angles  = Ry_left  @ Rz_after
            Rc_right_angles = Ry_right @ Rz_after @ R_y_final

            left24_local_conv  = (Rc_left_angles.T  @ left24_local.T ).T
            right24_local_conv = (Rc_right_angles.T @ right24_local.T).T

            # >>>>>>>>>>>>>>>>>>>> NEW: prepend wrist as thumb CMC (local origin) <<<<<<<<<<<<<<<<<<
            # Build 25-length arrays for angle computation: [wrist_as_CMC, original 24 pts]
            left25_local_conv_ang  = np.vstack([np.zeros((1,3), dtype=np.float32), left24_local_conv])
            right25_local_conv_ang = np.vstack([np.zeros((1,3), dtype=np.float32), right24_local_conv])
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            # per-finger from the 25-length arrays
            l_index  = left25_local_conv_ang [list(L_idx["index"]), :]
            l_middle = left25_local_conv_ang [list(L_idx["middle"]), :]
            l_ring   = left25_local_conv_ang [list(L_idx["ring"]), :]
            l_pinky  = left25_local_conv_ang [list(L_idx["pinky"]), :]
            l_thumb  = left25_local_conv_ang [list(L_idx["thumb"]), :]

            r_index  = right25_local_conv_ang[list(R_idx["index"]), :]
            r_middle = right25_local_conv_ang[list(R_idx["middle"]), :]
            r_ring   = right25_local_conv_ang[list(R_idx["ring"]), :]
            r_pinky  = right25_local_conv_ang[list(R_idx["pinky"]), :]
            r_thumb  = right25_local_conv_ang[list(R_idx["thumb"]), :]

            # angles
            la_index  = compute_finger_angles(l_index)
            la_middle = compute_finger_angles(l_middle)
            la_ring   = compute_finger_angles(l_ring)
            la_pinky  = compute_finger_angles(l_pinky)
            la_thumb  = compute_thumb_angles(l_thumb, 'left')

            ra_index  = compute_finger_angles(r_index)
            ra_middle = compute_finger_angles(r_middle)
            ra_ring   = compute_finger_angles(r_ring)
            ra_pinky  = compute_finger_angles(r_pinky)
            ra_thumb  = compute_thumb_angles(r_thumb, 'right')

            # fill qpos
            lq = np.zeros(20, dtype=np.float32)
            rq = np.zeros(20, dtype=np.float32)
            idx_l = {n:i for i,n in enumerate(EXPECTED_LEFT_JOINT_ORDER)}
            idx_r = {n:i for i,n in enumerate(EXPECTED_RIGHT_JOINT_ORDER)}

            # left
            lq[idx_l["lj_dg_2_1"]] = la_index['mcp_abd_rad'] + np.radians(7)
            lq[idx_l["lj_dg_2_2"]] = la_index['mcp_flex_rad']
            lq[idx_l["lj_dg_2_3"]] = la_index['pip_rad'] 
            lq[idx_l["lj_dg_2_4"]] = la_index['dip_rad']

            lq[idx_l["lj_dg_3_1"]] = la_middle['mcp_abd_rad']
            lq[idx_l["lj_dg_3_2"]] = la_middle['mcp_flex_rad']
            lq[idx_l["lj_dg_3_3"]] = la_middle['pip_rad'] 
            lq[idx_l["lj_dg_3_4"]] = la_middle['dip_rad']

            lq[idx_l["lj_dg_4_1"]] = la_ring['mcp_abd_rad'] 
            lq[idx_l["lj_dg_4_2"]] = la_ring['mcp_flex_rad']
            lq[idx_l["lj_dg_4_3"]] = la_ring['pip_rad'] 
            lq[idx_l["lj_dg_4_4"]] = la_ring['dip_rad']

            pinky_mcp_0 = la_pinky['mcp_abd_rad'] 
            if pinky_mcp_0 > 0:
                pinky_mcp_0 = -0.05
                lq[idx_l["lj_dg_5_1"]] = -la_pinky['mcp_abd_rad'] 
            lq[idx_l["lj_dg_5_2"]] = pinky_mcp_0 
            lq[idx_l["lj_dg_5_3"]] = la_pinky['pip_rad'] 
            lq[idx_l["lj_dg_5_4"]] = la_pinky['dip_rad'] 

            # lq[idx_l["lj_dg_1_1"]] = (la_thumb['mcp_0'] - np.radians(7)) * 1.1
            # lq[idx_l["lj_dg_1_2"]] =  la_thumb['mcp_1'] + np.radians(5)
            # lq[idx_l["lj_dg_1_3"]] = (la_thumb['pip_rad'] - np.radians(0)) * 1.1
            # lq[idx_l["lj_dg_1_4"]] =  la_thumb['dip_rad'] - np.radians(0)
            lq[idx_l["lj_dg_1_1"]] = la_thumb['mcp_0'] 
            lq[idx_l["lj_dg_1_2"]] =  la_thumb['mcp_1'] 
            lq[idx_l["lj_dg_1_3"]] = la_thumb['pip_rad'] 
            lq[idx_l["lj_dg_1_4"]] =  la_thumb['dip_rad'] 

            # right
            rq[idx_r["rj_dg_2_1"]] = ra_index['mcp_abd_rad'] 
            rq[idx_r["rj_dg_2_2"]] = ra_index['mcp_flex_rad'] 
            rq[idx_r["rj_dg_2_3"]] = ra_index['pip_rad'] 
            rq[idx_r["rj_dg_2_4"]] = ra_index['dip_rad']

            rq[idx_r["rj_dg_3_1"]] = ra_middle['mcp_abd_rad']
            rq[idx_r["rj_dg_3_2"]] = ra_middle['mcp_flex_rad']
            rq[idx_r["rj_dg_3_3"]] = ra_middle['pip_rad'] 
            rq[idx_r["rj_dg_3_4"]] = ra_middle['dip_rad']

            rq[idx_r["rj_dg_4_1"]] = ra_ring['mcp_abd_rad'] 
            rq[idx_r["rj_dg_4_2"]] = ra_ring['mcp_flex_rad']
            rq[idx_r["rj_dg_4_3"]] = ra_ring['pip_rad'] 
            rq[idx_r["rj_dg_4_4"]] = ra_ring['dip_rad']

            pinky_mcp_0_r = ra_pinky['mcp_abd_rad'] 
            if pinky_mcp_0_r < 0:
                pinky_mcp_0_r = 0.05
                rq[idx_r["rj_dg_5_1"]] = -ra_pinky['mcp_abd_rad'] 
            rq[idx_r["rj_dg_5_2"]] = pinky_mcp_0_r 
            rq[idx_r["rj_dg_5_3"]] = ra_pinky['pip_rad'] 
            rq[idx_r["rj_dg_5_4"]] = ra_pinky['dip_rad'] 

            # rq[idx_r["rj_dg_1_1"]] = (ra_thumb['mcp_0'] + np.radians(7)) * 1.1
            # rq[idx_r["rj_dg_1_2"]] =  ra_thumb['mcp_1'] - np.radians(7)
            # rq[idx_r["rj_dg_1_3"]] = (ra_thumb['pip_rad'] - np.radians(0)) * 1.1
            # rq[idx_r["rj_dg_1_4"]] =  ra_thumb['dip_rad'] - np.radians(5)
            rq[idx_r["rj_dg_1_1"]] = ra_thumb['mcp_0'] 
            rq[idx_r["rj_dg_1_2"]] =  ra_thumb['mcp_1'] 
            rq[idx_r["rj_dg_1_3"]] = (ra_thumb['pip_rad'] - np.radians(0)) 
            rq[idx_r["rj_dg_1_4"]] =  ra_thumb['dip_rad'] - np.radians(0)
            # clamp
            lq = clamp_joint_positions(lq, EXPECTED_LEFT_JOINT_ORDER)
            rq = clamp_joint_positions(rq, EXPECTED_RIGHT_JOINT_ORDER)

            # wrist orientation remap for CSV saving only
            Rw_L = T_base_left[:3, :3]
            Rw_R = T_base_right[:3, :3]
            Rc_left_save  = rot_x(np.pi) @ rot_y(-np.pi/2)  # X 180°, then new Y -90°
            Rc_right_save = rot_x(np.pi) @ rot_y(+np.pi/2)  # X 180°, then new Y +90°
            R_save_left   = Rw_L @ Rc_left_save
            R_save_right  = Rw_R @ Rc_right_save
            l_quat_b = R.from_matrix(R_save_left).as_quat().astype(np.float32)
            r_quat_b = R.from_matrix(R_save_right).as_quat().astype(np.float32)

            # write row (keypoints in base: still the original 24)
            out = [t]
            out += list(l_pos_b) + list(l_quat_b)
            out += list(r_pos_b) + list(r_quat_b)
            for p in left24_base:  out += [p[0], p[1], p[2]]
            for p in right24_base: out += [p[0], p[1], p[2]]
            out += list(lq) + list(rq)
            writer.writerow(out)

    print(f"Wrote: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()