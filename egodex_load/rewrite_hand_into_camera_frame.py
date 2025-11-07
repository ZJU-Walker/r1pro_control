#!/usr/bin/env python3
"""
Simple extractor:
- Load one EgoDex episode (HDF5 + MP4 path hard-coded)
- Use camera pose at t=0 as the reference frame (cam0)
- Express all left/right hand joints for all timesteps in cam0 frame
- Save everything into a 'result' dict variable for later processing
"""

import os, h5py, csv
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

from scipy.spatial.transform import Rotation as R

# --------- edit these two lines ----------
H5_PATH  = "5.hdf5"
MP4_PATH = "5.mp4"
# -----------------------------------------

LEFT_HAND_25 = [
    # Thumb (5)
    "leftThumbKnuckle",
    "leftThumbIntermediateBase",
    "leftThumbIntermediateTip",
    "leftThumbTip",

    # Index (5)
    "leftIndexFingerMetacarpal",
    "leftIndexFingerKnuckle",
    "leftIndexFingerIntermediateBase",
    "leftIndexFingerIntermediateTip",
    "leftIndexFingerTip",

    # Middle (5)
    "leftMiddleFingerMetacarpal",
    "leftMiddleFingerKnuckle",
    "leftMiddleFingerIntermediateBase",
    "leftMiddleFingerIntermediateTip",
    "leftMiddleFingerTip",

    # Ring (5)
    "leftRingFingerMetacarpal",
    "leftRingFingerKnuckle",
    "leftRingFingerIntermediateBase",
    "leftRingFingerIntermediateTip",
    "leftRingFingerTip",

    # Little (5)
    "leftLittleFingerMetacarpal",
    "leftLittleFingerKnuckle",
    "leftLittleFingerIntermediateBase",
    "leftLittleFingerIntermediateTip",
    "leftLittleFingerTip",
]

RIGHT_HAND_25 = [
    # Thumb (5)
    "rightThumbKnuckle",
    "rightThumbIntermediateBase",
    "rightThumbIntermediateTip",
    "rightThumbTip",

    # Index (5)
    "rightIndexFingerMetacarpal",
    "rightIndexFingerKnuckle",
    "rightIndexFingerIntermediateBase",
    "rightIndexFingerIntermediateTip",
    "rightIndexFingerTip",

    # Middle (5)
    "rightMiddleFingerMetacarpal",
    "rightMiddleFingerKnuckle",
    "rightMiddleFingerIntermediateBase",
    "rightMiddleFingerIntermediateTip",
    "rightMiddleFingerTip",

    # Ring (5)
    "rightRingFingerMetacarpal",
    "rightRingFingerKnuckle",
    "rightRingFingerIntermediateBase",
    "rightRingFingerIntermediateTip",
    "rightRingFingerTip",

    # Little (5)
    "rightLittleFingerMetacarpal",
    "rightLittleFingerKnuckle",
    "rightLittleFingerIntermediateBase",
    "rightLittleFingerIntermediateTip",
    "rightLittleFingerTip",
]

WRISTS = ["leftHand", "rightHand"]

# ---------- small SE(3) helpers ----------
def inv_SE3(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 homogeneous transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=T.dtype)
    Ti[:3, :3] = R.T
    Ti[:3, 3]  = -R.T @ t
    return Ti

def world_to_cam0(T_world_obj: np.ndarray, T_world_cam0: np.ndarray) -> np.ndarray:
    """Express obj in cam0 frame: T_cam0_obj = inv(T_world_cam0) @ T_world_obj."""
    return inv_SE3(T_world_cam0) @ T_world_obj

# ---------- rotation conversions ----------

def rotmats_to_quats(Rs: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrices (N,3,3) to quaternions (N,4)
    using scipy.spatial.transform.Rotation.
    The output order is (x, y, z, w).
    """
    r = R.from_matrix(Rs)
    q_xyzw = r.as_quat()  # SciPy returns (x, y, z, w)
    return q_xyzw.astype(np.float32)

# ---------- main routine ----------
def main() -> Dict:
    assert os.path.exists(H5_PATH), f"H5 not found: {H5_PATH}"
    # MP4 is not required for this conversion, but we keep it here for completeness
    if not os.path.exists(MP4_PATH):
        print(f"Warning: MP4 not found at {MP4_PATH}. Proceeding with HDF5 only.")

    with h5py.File(H5_PATH, "r") as f:
        # sanity
        assert "transforms" in f, "HDF5 missing 'transforms' group"
        T_cam_all = f["transforms"]["camera"]  # shape (N,4,4)
        N = int(T_cam_all.shape[0])
        print(f"Loaded episode with {N} frames")

        # reference camera pose at t=0
        T_world_cam0 = T_cam_all[0]

        # collect all joint names we care about, but only those that exist in file
        desired_names: List[str] = []
        for n in WRISTS + LEFT_HAND_25 + RIGHT_HAND_25:
            if n in f["transforms"]:
                desired_names.append(n)

        # allocate outputs
        # For each name: store per-timestep 4x4, pos3, rot6d
        T_cam0_of, pos_cam0_of, rot_cam0_of = {}, {}, {}

        for name in desired_names:
            Ts_world = f["transforms"][name]  # (N,4,4)

            # convert every frame to cam0 frame
            T_cam0_seq = np.empty_like(Ts_world)
            for t in range(N):
                T_cam0_seq[t] = world_to_cam0(Ts_world[t], T_world_cam0)

            # store full SE(3), position, and rotation *matrices*
            T_cam0_of[name]   = T_cam0_seq
            pos_cam0_of[name] = T_cam0_seq[:, :3, 3].astype(np.float32)      # (N,3)
            rot_cam0_of[name] = T_cam0_seq[:, :3, :3].astype(np.float32)      # (N,3,3)


        # Build a compact convenience slice for wrists and fingertips if you want
        left_tips  = ["leftThumbTip","leftIndexFingerTip","leftMiddleFingerTip","leftRingFingerTip","leftLittleFingerTip"]
        right_tips = ["rightThumbTip","rightIndexFingerTip","rightMiddleFingerTip","rightRingFingerTip","rightLittleFingerTip"]
        left_tips  = [n for n in left_tips  if n in pos_cam0_of]
        right_tips = [n for n in right_tips if n in pos_cam0_of]

        result = {
            "h5_path": H5_PATH,
            "mp4_path": MP4_PATH,
            "N": N,
            "T_world_cam0": T_world_cam0,     # (4,4) reference cam at t=0 (world frame)
            "T_cam0_of": T_cam0_of,           # dict[name] -> (N,4,4)
            "pos_cam0_of": pos_cam0_of,       # dict[name] -> (N,3)  in cam0
            "rot_cam0_of": rot_cam0_of,       # dict[name] -> (N,3,3) in cam0
            "wrists": [n for n in WRISTS if n in pos_cam0_of],
        }

        # # Example peek (safe to keep or remove)
        # for k in result["wrists"]:
        #     print(
        #         f"{k}: pos_cam0_of[{k}].shape = {result['pos_cam0_of'][k].shape}, "
        #         f"rot_cam0_of[{k}].shape = {result['rot_cam0_of'][k].shape}"
        #     )

        # ---- CSV dump (one row per timestep) ----
        # Output CSV path beside the H5, e.g. 0_cam0.csv
        base = os.path.splitext(os.path.basename(H5_PATH))[0]
        out_csv = os.path.join(os.path.dirname(H5_PATH), f"{base}_egodex_example.csv")

        # Column headers: t, left wrist pos+quat(xyzw), right wrist pos+quat(xyzw),
        # then left 25 keypoint positions (xyz), then right 25 (xyz)
        columns = ["t"]
        for wrist in ["leftHand", "rightHand"]:
            columns += [
                f"{wrist}_x", f"{wrist}_y", f"{wrist}_z",
                f"{wrist}_qx", f"{wrist}_qy", f"{wrist}_qz", f"{wrist}_qw",
            ]
        for name in LEFT_HAND_25:
            columns += [f"{name}_x", f"{name}_y", f"{name}_z"]
        for name in RIGHT_HAND_25:
            columns += [f"{name}_x", f"{name}_y", f"{name}_z"]

        # Helpers to fetch arrays with NaNs if a joint is missing
        def get_pos(name: str) -> np.ndarray:
            if name in pos_cam0_of:
                return pos_cam0_of[name]  # (N,3)
            return np.full((N, 3), np.nan, dtype=np.float32)

        def get_quat(name: str) -> np.ndarray:
            if name in rot_cam0_of:
                # SciPy returns (x,y,z,w); keep that
                return R.from_matrix(rot_cam0_of[name]).as_quat().astype(np.float32)  # (N,4)
            return np.full((N, 4), np.nan, dtype=np.float32)

        # Precompute wrists
        left_pos   = get_pos("leftHand")
        right_pos  = get_pos("rightHand")
        left_quat  = get_quat("leftHand")   # (qx, qy, qz, qw)
        right_quat = get_quat("rightHand")

        # Precompute all keypoints
        left_key_pos  = {n: get_pos(n) for n in LEFT_HAND_25}
        right_key_pos = {n: get_pos(n) for n in RIGHT_HAND_25}

        # Write CSV
        with open(out_csv, "w", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(columns)
            for t in range(N):
                row = [t]
                # left wrist: pos (x,y,z) + quat (qx,qy,qz,qw)
                row += list(left_pos[t]) + list(left_quat[t])
                # right wrist: pos (x,y,z) + quat (qx,qy,qz,qw)
                row += list(right_pos[t]) + list(right_quat[t])
                # left 25 keypoints (xyz)
                for name in LEFT_HAND_25:
                    row += list(left_key_pos[name][t])
                # right 25 keypoints (xyz)
                for name in RIGHT_HAND_25:
                    row += list(right_key_pos[name][t])
                writer.writerow(row)

        print(f"Wrote CSV: {out_csv}")


if __name__ == "__main__":
    result = main()
    # You can now use `result` in additional processing.
    # Example: access left wrist position at t=37 in cam0 frame:
    # lw_pos_t37 = result["pos_cam0_of"]["leftHand"][37]
