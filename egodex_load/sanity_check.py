#!/usr/bin/env python3
import csv
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

"""
VIEW MODES
----------
- "base"        : world-frame points + wrist axes using the RAW wrist rotation (Rw_*).
- "base_saved"  : world-frame points + wrist axes using the REMAPPED rotation you save to CSV:
                  R_save_left  = Rw_L @ (Rx(pi) @ Ry(-pi/2))
                  R_save_right = Rw_R @ (Rx(pi) @ Ry(+pi/2))
- "local"       : hand keypoints in the local wrist frames (what you use for angle computation),
                  with your intrinsic (Y then new Z) convention applied.

Set VIEW_MODE below.
"""

# ================== EDIT THESE ==================
INPUT_CSV = "3754_egodex_example.csv"   # same as your processing input
FRAME_IDX = 0                        # which row/frame to visualize (0 = first after header)
VIEW_MODE = "local"             # "base", "base_saved", or "local"

# Camera extrinsic (camera pose in base/world) = T_base_cam0 (keep in sync with your main script)
T_base_cam0 = np.array([
    [ 0.01988061, -0.43758429,  0.89895759,  0.14056752],
    [-0.99969330,  0.00457983,  0.02433772,  0.02539622],
    [-0.01476688, -0.89916573, -0.43735903,  0.43713101],
    [ 0.0,         0.0,         0.0,         1.0       ]
], dtype=np.float32)
# =================================================

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

# Finger index mapping over a 25-length array where index 0 is "CMC".
# We'll prepend wrist-as-CMC so the rest aligns (0..4 thumb, 5..9 index, etc.)
L_idx = {"index":(5,6,7,8,9), "middle":(10,11,12,13,14), "ring":(15,16,17,18,19), "pinky":(20,21,22,23,24), "thumb":(0,1,2,3,4)}
R_idx = L_idx

def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float32)

def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)

def read_row_as_dict(path, frame_idx):
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        # advance to desired frame
        for i, row in enumerate(reader):
            if i == frame_idx:
                vals = [float(x) if x != '' else np.nan for x in row]
                return dict(zip(header, vals))
    raise IndexError(f"FRAME_IDX {frame_idx} out of range.")

def read_point(d, base):
    return np.array([d.get(f"{base}_x", np.nan),
                     d.get(f"{base}_y", np.nan),
                     d.get(f"{base}_z", np.nan)], dtype=np.float32)

def wrist_T_from_row(d, arm):
    if arm == "left":
        px,py,pz = d["leftHand_x"], d["leftHand_y"], d["leftHand_z"]
        qx,qy,qz,qw = d["leftHand_qx"], d["leftHand_qy"], d["leftHand_qz"], d["leftHand_qw"]
    else:
        px,py,pz = d["rightHand_x"], d["rightHand_y"], d["rightHand_z"]
        qx,qy,qz,qw = d["rightHand_qx"], d["rightHand_qy"], d["rightHand_qz"], d["rightHand_qw"]
    T_cam = np.eye(4, dtype=np.float32)
    T_cam[:3,:3] = R.from_quat([qx,qy,qz,qw]).as_matrix().astype(np.float32)
    T_cam[:3, 3] = np.array([px,py,pz], dtype=np.float32)
    return T_base_cam0 @ T_cam

def build_lines_for_25():
    # Build polyline connections per finger over the 25-length array
    lines = []
    colors = []
    finger_sets = [
        ("thumb",  [L_idx["thumb"]],  [1,0,0]),
        ("index",  [L_idx["index"]],  [0,1,0]),
        ("middle", [L_idx["middle"]], [0,0,1]),
        ("ring",   [L_idx["ring"]],   [1,1,0]),
        ("pinky",  [L_idx["pinky"]],  [1,0,1]),
    ]
    for _, idx_groups, col in finger_sets:
        for idxs in idx_groups:
            for a,b in zip(idxs[:-1], idxs[1:]):
                lines.append([a,b])
                colors.append(col)
    return np.array(lines, dtype=np.int32), np.array(colors, dtype=np.float32)

def to_o3d_pointcloud(points, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.paint_uniform_color(color)
    return pcd

def hand_lineset(points25):
    lines, lcols = build_lines_for_25()
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points25.astype(np.float64))
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(lcols.astype(np.float64))
    return ls

def main():
    d = read_row_as_dict(INPUT_CSV, FRAME_IDX)

    # Base <-> camera pieces
    R_base_cam = T_base_cam0[:3,:3].astype(np.float32)
    t_base_cam = T_base_cam0[:3, 3].astype(np.float32)

    # Wrist world poses
    T_base_left  = wrist_T_from_row(d, "left")
    T_base_right = wrist_T_from_row(d, "right")
    Rw_L, tw_L = T_base_left[:3,:3],  T_base_left[:3,3]
    Rw_R, tw_R = T_base_right[:3,:3], T_base_right[:3,3]

    # Saved-orientation remap (IDENTICAL to your writer)
    Rc_left_save  = rot_x(np.pi) @ rot_y(-np.pi/2)  # X 180°, then new Y -90°
    Rc_right_save = rot_x(np.pi) @ rot_y(+np.pi/2)  # X 180°, then new Y +90°
    R_save_left   = Rw_L @ Rc_left_save
    R_save_right  = Rw_R @ Rc_right_save

    # Collect 24 keypoints (no wrist) in camera, then to base
    left24_cam  = np.stack([read_point(d, n) for n in LEFT_HAND_25], axis=0)   # 25 names in CSV…
    right24_cam = np.stack([read_point(d, n) for n in RIGHT_HAND_25], axis=0)  # …but you recorded 24 (no wrist)

    # to base/world
    left24_base  = (R_base_cam @ left24_cam.T).T  + t_base_cam
    right24_base = (R_base_cam @ right24_cam.T).T + t_base_cam

    # ------ LOCAL (wrist) coords + convention used in your angle code ------
    # local = Rw^T * (p - t)
    left24_local  = (Rw_L.T @ (left24_base  - tw_L).T).T
    right24_local = (Rw_R.T @ (right24_base - tw_R).T).T

    # Apply the same “intrinsic Y then new Z” convention for angle computation
    Ry_left,  Ry_right = rot_y(+np.pi/2), rot_y(-np.pi/2)
    R_y_final = rot_z(np.pi)
    Rz_after = rot_z(+np.pi/2)
    Rz_after_right = rot_z(-np.pi/2)
    Rc_left_angles  = Ry_left  @ Rz_after
    # Rc_right_angles = Ry_right @ Rz_after @ R_y_final
    Rc_right_angles = Ry_right @ Rz_after_right
    left24_local_conv  = (Rc_left_angles.T  @ left24_local.T ).T
    right24_local_conv = (Rc_right_angles.T @ right24_local.T).T

    # Prepend wrist-as-CMC at origin for 25-length visualization in local
    left25_local_conv  = np.vstack([np.zeros((1,3), dtype=np.float32), left24_local_conv])
    right25_local_conv = np.vstack([np.zeros((1,3), dtype=np.float32), right24_local_conv])

    # For base-view visualization (optional): draw the original 24 and add the true wrist position
    left25_base  = np.vstack([tw_L[None,:], left24_base])
    right25_base = np.vstack([tw_R[None,:], right24_base])

    # ------- Geometries -------
    geoms = []
    axis_global = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geoms.append(axis_global)

    vm = VIEW_MODE.lower()
    if vm in ["base", "base_saved"]:
        # points + finger lines in world frame
        geoms += [to_o3d_pointcloud(left25_base,  [1,0.6,0.6]),
                  to_o3d_pointcloud(right25_base, [0.6,0.6,1])]
        geoms += [hand_lineset(left25_base),
                  hand_lineset(right25_base)]

        # wrist axes (raw vs saved)
        wrist_axis_L = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06)
        wrist_axis_R = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06)

        TL = np.eye(4); TR = np.eye(4)
        TL[:3,3], TR[:3,3] = tw_L, tw_R
        if vm == "base":
            TL[:3,:3], TR[:3,:3] = Rw_L, Rw_R
            msg = "BASE/World with RAW wrist orientation axes (left=red X, green Y, blue Z)."
        else:  # "base_saved"
            TL[:3,:3], TR[:3,:3] = R_save_left, R_save_right
            msg = "BASE/World with SAVED wrist orientation axes (exactly what you write to CSV)."

            # >>> print the exact quaternions we’re plotting, in BASE frame (xyzw)
            qL = R.from_matrix(R_save_left).as_quat().astype(np.float32)
            qR = R.from_matrix(R_save_right).as_quat().astype(np.float32)
            print(
                "left_saved_quat_xyzw (base): "
                f"[{qL[0]:+.6f}, {qL[1]:+.6f}, {qL[2]:+.6f}, {qL[3]:+.6f}]"
            )
            print(
                "right_saved_quat_xyzw (base): "
                f"[{qR[0]:+.6f}, {qR[1]:+.6f}, {qR[2]:+.6f}, {qR[3]:+.6f}]"
            )


        wrist_axis_L.transform(TL)
        wrist_axis_R.transform(TR)
        geoms += [wrist_axis_L, wrist_axis_R]
        print(msg)

    else:
        # LOCAL view: show left and right in their own local frames (wrist at origin)
        # (Points are what your angle computation used.)
        # To view each hand separately, offset the right-hand cloud a bit:
        offset = np.array([0.25, 0, 0], dtype=np.float32)
        left_pts_vis  = left25_local_conv
        right_pts_vis = right25_local_conv + offset

        geoms += [to_o3d_pointcloud(left_pts_vis,  [1,0.6,0.6]),
                  to_o3d_pointcloud(right_pts_vis, [0.6,0.6,1])]
        geoms += [hand_lineset(left_pts_vis), hand_lineset(right_pts_vis)]

        # Local wrist frames (identity at origin for left; identity+offset for right)
        axis_L_local = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06)
        axis_R_local = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06)
        TR = np.eye(4); TR[:3,3] = offset
        axis_R_local.transform(TR)
        geoms += [axis_L_local, axis_R_local]

        print("Showing LOCAL hand frames used for angle computation (left at origin, right offset on +X).")

    print(f"Left wrist (world): {tw_L}, Right wrist (world): {tw_R}")
    o3d.visualization.draw_geometries(geoms)

if __name__ == "__main__":
    main()
