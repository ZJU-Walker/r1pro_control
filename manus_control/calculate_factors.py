#!/usr/bin/env python3
import numpy as np
import math
import sys

# ===== Headers (exactly as you used) =====
GLOVE_KEYS_8 = [
    "glove_Thumb1", "glove_Thumb2",
    "glove_Thumb3", "glove_Thumb4",
    "glove_Index1", "glove_Index2",
    "glove_Index3", "glove_Index4",
]

HAND_KEYS_8 = [
    "hand_rj_dg_1_1", "hand_rj_dg_1_2",
    "hand_rj_dg_1_3", "hand_rj_dg_1_4",
    "hand_rj_dg_2_1", "hand_rj_dg_2_2",
    "hand_rj_dg_2_3", "hand_rj_dg_2_4",
]

# Direction multipliers for the first 8 (thumb4 + index4)
DIR_VALS = np.array([1, -1, 1, 1,  -1, 1, 1, 1], dtype=float)

def deg2rad(x): 
    return x * (math.pi / 180.0)

def load_csv(path="tesollo_test.csv"):
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.shape == ():  # single-row structured -> wrap
        data = np.array([data])
    return data

def build_mats(data):
    try:
        glove_deg = np.column_stack([data[k] for k in GLOVE_KEYS_8]).astype(float)  # (N,8)
        hand_rad  = np.column_stack([data[k] for k in HAND_KEYS_8]).astype(float)   # (N,8)
    except Exception as e:
        print(f"[ERROR] Missing expected columns: {e}", file=sys.stderr)
        sys.exit(1)
    return glove_deg, hand_rad

def preprocess_qd(glove_deg):
    """
    Recreate your pre-scale/pre-dir qd[0:8] from glove degrees, per your code:
      qd[0] = (58.5 - q[1]) * rad
      qd[1] = (q[0] + 20) * rad
      qd[2]..qd[7] = q[2]..q[7] (deg->rad)
    """
    q = glove_deg  # (N,8)
    qd0 = deg2rad(58.5 - q[:, 1])
    qd1 = deg2rad(q[:, 0] + 20.0)
    qd2 = deg2rad(q[:, 2])
    qd3 = deg2rad(q[:, 3])
    qd4 = deg2rad(q[:, 4])
    qd5 = deg2rad(q[:, 5])
    qd6 = deg2rad(q[:, 6])
    qd7 = deg2rad(q[:, 7])
    return np.column_stack([qd0, qd1, qd2, qd3, qd4, qd5, qd6, qd7])  # (N,8)

def fit_scales_single(x_qd, y_hand):
    """Single-sample scale: s_i = y / (dir * x)."""
    x_eff = DIR_VALS * x_qd[0, :]  # (8,)
    scales = np.empty(8, dtype=float)
    for i in range(8):
        denom = x_eff[i]
        if abs(denom) < 1e-9:
            print(f"[WARN] Joint {i}: denom≈0 (qd={x_qd[0,i]:.6e}); using scale=1.0", file=sys.stderr)
            scales[i] = 1.0
        else:
            scales[i] = y_hand[0, i] / denom
    # rmse is zero by construction (1 row)
    rmse = np.zeros(8, dtype=float)
    return scales, rmse

def fit_scales_lsq(x_qd, y_hand):
    """
    Multi-row least-squares through origin per joint:
      y[:,i] ≈ dir[i] * s_i * x[:,i]
      => let z[:,i] = dir[i]*x[:,i]; s_i = (z·y) / (z·z)
    """
    z = x_qd * DIR_VALS[None, :]               # (N,8)
    num = (z * y_hand).sum(axis=0)
    den = (z * z).sum(axis=0)
    den = np.where(den < 1e-12, 1.0, den)      # guard
    scales = num / den                          # (8,)
    pred = z * scales[None, :]
    rmse = np.sqrt(((pred - y_hand)**2).mean(axis=0))
    return scales, rmse

def main():
    data = load_csv("tesollo_test.csv")
    N = len(data)
    if N < 1:
        print("[ERROR] No rows in CSV.", file=sys.stderr); sys.exit(1)
    if N == 1:
        print("[INFO] Detected 1 row → single-sample fit.", file=sys.stderr)
    else:
        print(f"[INFO] Detected {N} rows → least-squares fit over {N} samples.", file=sys.stderr)

    glove_deg, hand_rad = build_mats(data)
    qd = preprocess_qd(glove_deg)  # (N,8), radians pre-scale/pre-dir

    if N == 1:
        scales, rmse = fit_scales_single(qd, hand_rad)
    else:
        scales, rmse = fit_scales_lsq(qd, hand_rad)

    # Print scales to paste
    print("\n# === Paste into your mGripperCalibrationData (first 8 entries) ===")
    print("mGripperCalibrationData[:8] =", [round(float(s), 6) for s in scales])

    # Quality report
    print("\n# RMSE per joint (rad):", [round(float(r), 6) for r in rmse])

    # Suggested limits from observed data (+/- small margin)
    lim_min = hand_rad.min(axis=0) - 0.05
    lim_max = hand_rad.max(axis=0) + 0.05
    print("\n# Suggested limits for first 8 (radians) based on your samples:")
    print("LIM_MIN[:8] =", [round(float(v), 4) for v in lim_min])
    print("LIM_MAX[:8] =", [round(float(v), 4) for v in lim_max])

    # Quick check prediction against samples
    z = qd * DIR_VALS[None, :]
    pred = z * scales[None, :]
    err = pred - hand_rad
    # Print compact summary
    print("\n# Mean abs error per joint (rad):", [round(float(v), 6) for v in np.mean(np.abs(err), axis=0)])

if __name__ == "__main__":
    main()
