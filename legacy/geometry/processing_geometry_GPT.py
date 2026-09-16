#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Radar + IMU geometry correction GPT_2

Folder structure is preserved:
    source_root/person/hand_gesture/head_motion/*.csv
 -> target_root/person/hand_gesture/head_motion/*.csv

Main changes from GPT_1:
  1) q.j / q.k mapping bug fixed.
  2) relative_first reference mode added. This removes head motion relative to the
     first valid frame of each sequence and is usually safer than absolute world
     orientation when absolute yaw is not calibrated.
  3) raw radar columns are preserved; corrected columns are added.
  4) if source CSV already has *_raw columns, those can be used to avoid double correction.
"""

import os
import argparse
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_SOURCE_ROOT = (
    "/home/dokyeong/miniconda3/envs/hgr_sci/"
    "processing_data_set/MergedData_all_BLPF_10Hz_v2"
)

DEFAULT_TARGET_ROOT = (
    "/media/dokyeong/MINJI/dokyeong/"
    "processedData_radar_imu_geometry_GPT_2"
)

RADAR_COLS = [
    "range",
    "doppler",
    "horizontal_angle",
    "vertical_angle",
]

QUAT_COLS = [
    "q.w",
    "q.i",
    "q.j",
    "q.k",
]


###############################################################################
# Quaternion utilities
###############################################################################

def normalize_quaternion(q: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    q shape: (N, 4), columns: w, x, y, z
    """
    q = q.astype(np.float64, copy=False)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    valid = norm[:, 0] > eps

    q_norm = np.full_like(q, np.nan, dtype=np.float64)
    q_norm[valid] = q[valid] / norm[valid]

    return q_norm, valid


def conjugate_quaternion(q: np.ndarray) -> np.ndarray:
    q_conj = q.copy()
    q_conj[:, 1:] *= -1.0
    return q_conj


def multiply_quaternion(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamilton product q = q1 * q2.

    q1, q2 shape: (N, 4), columns: w, x, y, z
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    out = np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=1,
    )
    return out.astype(np.float64)


def rotate_vectors_by_quaternion(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Rotate vectors by unit quaternion q.

    v shape: (N, 3)
    q shape: (N, 4), columns: w, x, y, z

    Equivalent to q * [0, v] * q_conj.
    """
    w = q[:, 0:1]
    q_vec = q[:, 1:4]

    uv = np.cross(q_vec, v)
    uuv = np.cross(q_vec, uv)

    return v + 2.0 * (w * uv + uuv)


###############################################################################
# Rotation matrices and radar coordinate conversion
###############################################################################

def rotation_matrix(axis: str, deg: float) -> np.ndarray:
    """
    3D rotation matrix.
    Row-vector application is xyz @ R.T.
    """
    rad = np.deg2rad(deg)
    c = np.cos(rad)
    s = np.sin(rad)

    axis = axis.lower()

    if axis == "x":
        R = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, c, -s],
                [0.0, s, c],
            ],
            dtype=np.float64,
        )
    elif axis == "y":
        R = np.array(
            [
                [c, 0.0, s],
                [0.0, 1.0, 0.0],
                [-s, 0.0, c],
            ],
            dtype=np.float64,
        )
    elif axis == "z":
        R = np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"Unknown axis: {axis}. Use x, y, or z.")

    return R


def radar_polar_to_xyz(
    rng: np.ndarray,
    h_angle_deg: np.ndarray,
    v_angle_deg: np.ndarray,
    horizontal_sign: float = 1.0,
    vertical_sign: float = 1.0,
) -> np.ndarray:
    """
    Radar spherical coordinates -> xyz.

    Assumption:
      x: forward
      y: left/right
      z: up/down
      horizontal_angle: angle in x-y plane
      vertical_angle: elevation angle from x-y plane
    """
    h = np.deg2rad(horizontal_sign * h_angle_deg)
    v = np.deg2rad(vertical_sign * v_angle_deg)

    x = rng * np.cos(v) * np.cos(h)
    y = rng * np.cos(v) * np.sin(h)
    z = rng * np.sin(v)

    return np.stack([x, y, z], axis=1).astype(np.float64)


def xyz_to_radar_polar(xyz: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    rng = np.sqrt(x * x + y * y + z * z)
    h_angle = np.rad2deg(np.arctan2(y, x))
    xy_norm = np.sqrt(x * x + y * y)
    v_angle = np.rad2deg(np.arctan2(z, xy_norm + eps))

    return rng, h_angle, v_angle


###############################################################################
# Dataframe helpers
###############################################################################

def validate_columns(df: pd.DataFrame, file_path: str, needed_cols: List[str]) -> bool:
    missing = [col for col in needed_cols if col not in df.columns]
    if missing:
        print(f"[SKIP] Missing columns in {file_path}")
        print(f"       Missing: {missing}")
        return False
    return True


def to_numeric_safe(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def choose_radar_input_columns(df: pd.DataFrame, prefer_raw_backup: bool) -> Dict[str, str]:
    """
    If the input was already corrected once, *_raw columns may exist.
    Using *_raw prevents double-correction.
    """
    mapping = {}
    for col in RADAR_COLS:
        raw_col = f"{col}_raw"
        if prefer_raw_backup and raw_col in df.columns:
            mapping[col] = raw_col
        else:
            mapping[col] = col
    return mapping


def extract_quaternion(
    df: pd.DataFrame,
    valid_mask: np.ndarray,
    quat_yz_swap: bool,
) -> np.ndarray:
    """
    GPT_2 fixed behavior:
      default: q = (q.w, q.i, q.j, q.k) -> (w, x, y, z)
      --quat_yz_swap: q = (q.w, q.i, q.k, q.j)

    In GPT_1, the else branch comment said qy=q.j/qz=q.k, but the code still
    used qy=q.k/qz=q.j. This function makes the flag meaningful.
    """
    qw = df.loc[valid_mask, "q.w"].to_numpy(dtype=np.float64)
    qx = df.loc[valid_mask, "q.i"].to_numpy(dtype=np.float64)

    if quat_yz_swap:
        qy = df.loc[valid_mask, "q.k"].to_numpy(dtype=np.float64)
        qz = df.loc[valid_mask, "q.j"].to_numpy(dtype=np.float64)
    else:
        qy = df.loc[valid_mask, "q.j"].to_numpy(dtype=np.float64)
        qz = df.loc[valid_mask, "q.k"].to_numpy(dtype=np.float64)

    return np.stack([qw, qx, qy, qz], axis=1)


def make_sensor_to_world_quaternion(q: np.ndarray, quat_direction: str) -> np.ndarray:
    """
    Return quaternion that maps sensor/body vectors to world vectors.
    """
    if quat_direction == "sensor_to_world":
        return q
    if quat_direction == "world_to_sensor":
        return conjugate_quaternion(q)
    raise ValueError("quat_direction must be sensor_to_world or world_to_sensor")


def make_reference_quaternion(q_sensor_to_world: np.ndarray, reference_mode: str) -> np.ndarray:
    """
    Return q_used such that vector in current sensor frame is rotated into the
    selected reference frame.

    absolute:
      v_ref = R_sensor_to_world(t) * v_sensor(t)

    relative_first:
      v_ref = R_sensor_to_world(0)^T * R_sensor_to_world(t) * v_sensor(t)
      This removes head motion relative to the first valid frame of the sequence.
    """
    if reference_mode == "absolute":
        return q_sensor_to_world

    if reference_mode == "relative_first":
        q0 = q_sensor_to_world[0:1]
        q0_inv = conjugate_quaternion(q0)
        q0_inv_rep = np.repeat(q0_inv, repeats=len(q_sensor_to_world), axis=0)
        q_rel = multiply_quaternion(q0_inv_rep, q_sensor_to_world)
        q_rel, _ = normalize_quaternion(q_rel)
        return q_rel

    raise ValueError("reference_mode must be absolute or relative_first")


###############################################################################
# Main correction
###############################################################################

def correct_geometry_dataframe(
    df: pd.DataFrame,
    file_path: str,
    tilt_correction_deg: float = 45.0,
    tilt_axis: str = "y",
    quat_direction: str = "sensor_to_world",
    reference_mode: str = "relative_first",
    quat_yz_swap: bool = False,
    scale_factor: float = 1.0,
    horizontal_sign: float = 1.0,
    vertical_sign: float = 1.0,
    replace_original: bool = True,
    prefer_raw_backup: bool = True,
) -> Optional[pd.DataFrame]:
    radar_input_cols = choose_radar_input_columns(df, prefer_raw_backup=prefer_raw_backup)
    needed_cols = list(radar_input_cols.values()) + QUAT_COLS

    if not validate_columns(df, file_path, needed_cols):
        return None

    df = df.copy()
    df = to_numeric_safe(df, needed_cols)

    # Preserve the original radar columns once.
    for col in RADAR_COLS:
        raw_col = f"{col}_raw"
        if raw_col not in df.columns:
            df[raw_col] = df[col]

    valid_mask = np.ones(len(df), dtype=bool)
    for col in needed_cols:
        valid_mask &= np.isfinite(df[col].to_numpy(dtype=np.float64))

    if valid_mask.sum() == 0:
        print(f"[SKIP] No valid numeric rows in {file_path}")
        return None

    result_cols = [
        "radar_x_raw",
        "radar_y_raw",
        "radar_z_raw",
        "radar_x_tilt_corr",
        "radar_y_tilt_corr",
        "radar_z_tilt_corr",
        "radar_x_world",
        "radar_y_world",
        "radar_z_world",
        "radar_x_ref",
        "radar_y_ref",
        "radar_z_ref",
        "range_corr",
        "doppler_corr",
        "horizontal_angle_corr",
        "vertical_angle_corr",
    ]
    for col in result_cols:
        df[col] = np.nan

    rng = df.loc[valid_mask, radar_input_cols["range"]].to_numpy(dtype=np.float64)
    doppler = df.loc[valid_mask, radar_input_cols["doppler"]].to_numpy(dtype=np.float64)
    h_angle = df.loc[valid_mask, radar_input_cols["horizontal_angle"]].to_numpy(dtype=np.float64)
    v_angle = df.loc[valid_mask, radar_input_cols["vertical_angle"]].to_numpy(dtype=np.float64)

    q = extract_quaternion(df, valid_mask=valid_mask, quat_yz_swap=quat_yz_swap)
    q, quat_valid = normalize_quaternion(q)

    valid_indices = np.where(valid_mask)[0]
    final_indices = valid_indices[quat_valid]

    if len(final_indices) == 0:
        print(f"[SKIP] No valid quaternion rows in {file_path}")
        return None

    rng = rng[quat_valid]
    doppler = doppler[quat_valid]
    h_angle = h_angle[quat_valid]
    v_angle = v_angle[quat_valid]
    q = q[quat_valid]

    xyz_raw = radar_polar_to_xyz(
        rng=rng,
        h_angle_deg=h_angle,
        v_angle_deg=v_angle,
        horizontal_sign=horizontal_sign,
        vertical_sign=vertical_sign,
    )

    # Fixed mount correction from radar frame to IMU/head sensor frame.
    R_tilt = rotation_matrix(axis=tilt_axis, deg=tilt_correction_deg)
    xyz_tilt_corr = xyz_raw @ R_tilt.T

    # q_sensor_to_world is explicit regardless of sensor output convention.
    q_sensor_to_world = make_sensor_to_world_quaternion(q, quat_direction=quat_direction)

    # For debugging, save absolute world as well.
    xyz_world_abs = rotate_vectors_by_quaternion(xyz_tilt_corr, q_sensor_to_world)

    # Recommended corrected reference frame.
    q_ref = make_reference_quaternion(q_sensor_to_world, reference_mode=reference_mode)
    xyz_ref = rotate_vectors_by_quaternion(xyz_tilt_corr, q_ref)

    xyz_ref_scaled = xyz_ref * float(scale_factor)
    xyz_world_abs_scaled = xyz_world_abs * float(scale_factor)

    range_corr, horizontal_angle_corr, vertical_angle_corr = xyz_to_radar_polar(xyz_ref_scaled)
    doppler_corr = doppler.copy()

    df.loc[final_indices, "radar_x_raw"] = xyz_raw[:, 0]
    df.loc[final_indices, "radar_y_raw"] = xyz_raw[:, 1]
    df.loc[final_indices, "radar_z_raw"] = xyz_raw[:, 2]

    df.loc[final_indices, "radar_x_tilt_corr"] = xyz_tilt_corr[:, 0]
    df.loc[final_indices, "radar_y_tilt_corr"] = xyz_tilt_corr[:, 1]
    df.loc[final_indices, "radar_z_tilt_corr"] = xyz_tilt_corr[:, 2]

    # Keep the legacy names used by training code. In GPT_2 they contain the selected
    # reference-frame corrected vector, not necessarily global world coordinates.
    df.loc[final_indices, "radar_x_world"] = xyz_ref_scaled[:, 0]
    df.loc[final_indices, "radar_y_world"] = xyz_ref_scaled[:, 1]
    df.loc[final_indices, "radar_z_world"] = xyz_ref_scaled[:, 2]

    df.loc[final_indices, "radar_x_ref"] = xyz_ref_scaled[:, 0]
    df.loc[final_indices, "radar_y_ref"] = xyz_ref_scaled[:, 1]
    df.loc[final_indices, "radar_z_ref"] = xyz_ref_scaled[:, 2]

    df.loc[final_indices, "radar_x_abs_world"] = xyz_world_abs_scaled[:, 0]
    df.loc[final_indices, "radar_y_abs_world"] = xyz_world_abs_scaled[:, 1]
    df.loc[final_indices, "radar_z_abs_world"] = xyz_world_abs_scaled[:, 2]

    df.loc[final_indices, "range_corr"] = range_corr
    df.loc[final_indices, "doppler_corr"] = doppler_corr
    df.loc[final_indices, "horizontal_angle_corr"] = horizontal_angle_corr
    df.loc[final_indices, "vertical_angle_corr"] = vertical_angle_corr

    if replace_original:
        df.loc[final_indices, "range"] = range_corr
        df.loc[final_indices, "doppler"] = doppler_corr
        df.loc[final_indices, "horizontal_angle"] = horizontal_angle_corr
        df.loc[final_indices, "vertical_angle"] = vertical_angle_corr

    df["geometry_version"] = "GPT_2"
    df["geometry_tilt_axis"] = tilt_axis
    df["geometry_tilt_correction_deg"] = tilt_correction_deg
    df["geometry_quat_direction_input"] = quat_direction
    df["geometry_reference_mode"] = reference_mode
    df["geometry_quat_yz_swap"] = quat_yz_swap
    df["geometry_quat_mapping"] = "w,i,k,j" if quat_yz_swap else "w,i,j,k"
    df["geometry_scale_factor"] = scale_factor
    df["geometry_horizontal_sign"] = horizontal_sign
    df["geometry_vertical_sign"] = vertical_sign
    df["geometry_replace_original"] = replace_original
    df["geometry_prefer_raw_backup"] = prefer_raw_backup

    return df


###############################################################################
# Directory processing
###############################################################################

def process_directory(
    source_root: str,
    target_root: str,
    tilt_correction_deg: float,
    tilt_axis: str,
    quat_direction: str,
    reference_mode: str,
    quat_yz_swap: bool,
    scale_factor: float,
    horizontal_sign: float,
    vertical_sign: float,
    replace_original: bool,
    prefer_raw_backup: bool,
) -> None:
    if not os.path.isdir(source_root):
        raise FileNotFoundError(f"source_root does not exist: {source_root}")

    os.makedirs(target_root, exist_ok=True)

    total_files = 0
    saved_files = 0
    skipped_files = 0
    failed_files = 0

    print("========================================")
    print("Radar + IMU Geometry Correction GPT_2")
    print("========================================")
    print(f"Source root          : {source_root}")
    print(f"Target root          : {target_root}")
    print(f"Tilt axis            : {tilt_axis}")
    print(f"Tilt correction deg  : {tilt_correction_deg}")
    print(f"Quaternion direction : {quat_direction}")
    print(f"Reference mode       : {reference_mode}")
    print(f"Quaternion Y/Z swap  : {quat_yz_swap}")
    print(f"Scale factor         : {scale_factor}")
    print(f"Horizontal sign      : {horizontal_sign}")
    print(f"Vertical sign        : {vertical_sign}")
    print(f"Replace original     : {replace_original}")
    print(f"Prefer raw backup    : {prefer_raw_backup}")
    print("========================================")

    for dirpath, _, filenames in os.walk(source_root):
        csv_files = [name for name in filenames if name.lower().endswith(".csv")]

        for filename in csv_files:
            total_files += 1
            src_file = os.path.join(dirpath, filename)
            rel_dir = os.path.relpath(dirpath, source_root)
            dst_dir = os.path.join(target_root, rel_dir)
            os.makedirs(dst_dir, exist_ok=True)
            dst_file = os.path.join(dst_dir, filename)

            try:
                df = pd.read_csv(src_file)
                corrected_df = correct_geometry_dataframe(
                    df=df,
                    file_path=src_file,
                    tilt_correction_deg=tilt_correction_deg,
                    tilt_axis=tilt_axis,
                    quat_direction=quat_direction,
                    reference_mode=reference_mode,
                    quat_yz_swap=quat_yz_swap,
                    scale_factor=scale_factor,
                    horizontal_sign=horizontal_sign,
                    vertical_sign=vertical_sign,
                    replace_original=replace_original,
                    prefer_raw_backup=prefer_raw_backup,
                )

                if corrected_df is None:
                    skipped_files += 1
                    continue

                corrected_df.to_csv(dst_file, index=False, encoding="utf-8-sig")
                saved_files += 1
                print(f"[SAVE] {dst_file}")

            except Exception:
                failed_files += 1
                print(f"[FAIL] {src_file}")
                traceback.print_exc()

    print("\n========================================")
    print("Done")
    print("========================================")
    print(f"Total CSV files : {total_files}")
    print(f"Saved files     : {saved_files}")
    print(f"Skipped files   : {skipped_files}")
    print(f"Failed files    : {failed_files}")
    print(f"Output root     : {target_root}")


###############################################################################
# CLI
###############################################################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_root", type=str, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--target_root", type=str, default=DEFAULT_TARGET_ROOT)

    parser.add_argument("--tilt_correction_deg", type=float, default=45.0)
    parser.add_argument("--tilt_axis", type=str, default="y", choices=["x", "y", "z"])

    parser.add_argument(
        "--quat_direction",
        type=str,
        default="sensor_to_world",
        choices=["sensor_to_world", "world_to_sensor"],
    )
    parser.add_argument(
        "--reference_mode",
        type=str,
        default="relative_first",
        choices=["relative_first", "absolute"],
    )
    parser.add_argument(
        "--quat_yz_swap",
        action="store_true",
        help="Use q=(q.w,q.i,q.k,q.j). Default is q=(q.w,q.i,q.j,q.k).",
    )

    parser.add_argument("--scale_factor", type=float, default=1.0)
    parser.add_argument("--horizontal_sign", type=float, default=1.0, choices=[-1.0, 1.0])
    parser.add_argument("--vertical_sign", type=float, default=1.0, choices=[-1.0, 1.0])

    parser.add_argument("--no_replace_original", action="store_true")
    parser.add_argument(
        "--no_prefer_raw_backup",
        action="store_true",
        help="Do not use range_raw/doppler_raw/... even if they exist.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    process_directory(
        source_root=args.source_root,
        target_root=args.target_root,
        tilt_correction_deg=args.tilt_correction_deg,
        tilt_axis=args.tilt_axis,
        quat_direction=args.quat_direction,
        reference_mode=args.reference_mode,
        quat_yz_swap=args.quat_yz_swap,
        scale_factor=args.scale_factor,
        horizontal_sign=args.horizontal_sign,
        vertical_sign=args.vertical_sign,
        replace_original=not args.no_replace_original,
        prefer_raw_backup=not args.no_prefer_raw_backup,
    )


if __name__ == "__main__":
    main()
