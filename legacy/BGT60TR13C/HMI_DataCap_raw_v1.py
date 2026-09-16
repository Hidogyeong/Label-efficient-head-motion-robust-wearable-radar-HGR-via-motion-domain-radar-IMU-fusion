#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import os
import glob
import numpy as np
import cv2  # OpenCV로 이미지 저장 (혹은 PIL 등 사용 가능)

#########################################
# 0) 파일 검색 및 그룹핑
#########################################
def group_radar_files(radar_dir):
    """
    레이더 폴더(radar_dir) 내부를 재귀적으로 탐색하여,
    _doppler.csv, _range.csv, _horizontal.csv, _vertical.csv 파일을 찾고,
    동일 prefix(예: "xxxx_doppler.csv" -> "xxxx")로 묶어서 관리합니다.
    반환:
      {
        (rel_dir, prefix): {
           '_doppler.csv': <파일 경로>,
           '_range.csv': <파일 경로>,
           '_horizontal.csv': <파일 경로>,
           '_vertical.csv': <파일 경로>
        },
        ...
      }
    """
    suffixes = ['_doppler.csv', '_range.csv', '_horizontal.csv', '_vertical.csv']
    groups = {}
    all_csv = glob.glob(os.path.join(radar_dir, '**', '*.csv'), recursive=True)
    for filepath in all_csv:
        filename = os.path.basename(filepath)
        for suf in suffixes:
            if filename.endswith(suf):
                prefix = filename[:-len(suf)]
                rel_dir = os.path.dirname(os.path.relpath(filepath, radar_dir))
                key = (rel_dir, prefix)
                if key not in groups:
                    groups[key] = {}
                groups[key][suf] = filepath
                break
    return groups

#########################################
# 1) 레이더 & IMU 데이터 로드
#########################################
def load_radar_and_imu_data(radar_groups, radar_dir, imu_dir):
    results = {}
    for (rel_dir, prefix), files_dict in radar_groups.items():
        radar_data_dict = {}
        for suf, path in files_dict.items():
            try:
                arr = np.loadtxt(path, delimiter=",")
                radar_data_dict[suf] = arr
                print(f"[LOAD] Radar: {path}, shape={arr.shape}")
            except Exception as e:
                print(f"[ERROR] Radar file {path}: {e}")
                radar_data_dict[suf] = None

        imu_filename = "I" + prefix + ".csv"
        imu_filepath = os.path.join(imu_dir, rel_dir, imu_filename)
        imu_data = None
        if os.path.exists(imu_filepath):
            try:
                imu_data = np.loadtxt(imu_filepath, delimiter=",", skiprows=1)
                print(f"[LOAD] IMU: {imu_filepath}, shape={imu_data.shape}")
            except Exception as e:
                print(f"[ERROR] IMU file {imu_filepath}: {e}")
                imu_data = None
        else:
            print(f"[WARN] IMU file not found: {imu_filepath}")
        results[(rel_dir, prefix)] = {
            'radar': radar_data_dict,
            'imu': imu_data
        }
    return results

#########################################
# 2) Radar/IMU 데이터 맞추기
#########################################
def trim_imu_data(imu_data, target_frame_count):
    num_imu_frames = imu_data.shape[0]
    if num_imu_frames > target_frame_count:
        return imu_data[-target_frame_count:]
    elif num_imu_frames < target_frame_count:
        raise ValueError("IMU 데이터 프레임 수가 레이더 데이터 프레임 수보다 적습니다.")
    else:
        return imu_data

#########################################
# 3) Re_bin 함수 (모든 데이터에 대해 num_bins=64)
#########################################
def rebin_corrected_data(original_data, corrected_angles, num_bins):
    """
    original_data: 2D numpy 배열, shape = (num_frames, num_bins) (에너지 값)
    corrected_angles: 2D numpy 배열, shape = (num_frames, num_bins) (보정된 각도, 예: 40도)
    num_bins: 최종 원하는 bin 수 (여기서는 64)
    반환: new_data, shape = (num_frames, num_bins)
    """
    num_frames = original_data.shape[0]
    new_data = np.zeros_like(original_data)
    for f in range(num_frames):
        for b in range(original_data.shape[1]):
            new_idx = int(round(corrected_angles[f, b]))
            new_idx = max(0, min(new_idx, num_bins - 1))
            new_data[f, new_idx] += original_data[f, b]
    return new_data

#########################################
# 4) 쿼터니언/회전 관련 유틸 함수들
#########################################
def quat_to_rotation_matrix(q):
    w, x, y, z = q
    norm_q = np.sqrt(w*w + x*x + y*y + z*z)
    if norm_q < 1e-8:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w/norm_q, x/norm_q, y/norm_q, z/norm_q
    R = np.array([
        [1 - 2*y**2 - 2*z**2,   2*x*y - 2*w*z,       2*x*z + 2*w*y],
        [2*x*y + 2*w*z,         1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,         2*y*z + 2*w*x,       1 - 2*x**2 - 2*y**2]
    ], dtype=np.float64)
    return R

def rotate_vector_by_quaternion(v, q):
    R = quat_to_rotation_matrix(q)
    return R.dot(v)

def tilt_rotation_matrix(tilt_deg):
    tilt_rad = np.deg2rad(tilt_deg)
    R = np.array([
        [np.cos(tilt_rad), 0, np.sin(tilt_rad)],
        [0, 1, 0],
        [-np.sin(tilt_rad), 0, np.cos(tilt_rad)]
    ], dtype=np.float64)
    return R

#########################################
# 5) 보정에 필요한 파라미터 및 보정 함수들
#########################################
RADAR_TILT_DEG = 45.0
SCALE_FACTOR = 5.0
R_tilt_corr = tilt_rotation_matrix(RADAR_TILT_DEG)

def correct_range(rng, x2, y2, z2):
    new_range = np.sqrt(x2*x2 + y2*y2 + z2*z2 + 1e-8)
    return new_range

def correct_doppler(doppler):
    new_doppler = doppler
    return new_doppler

def correct_horizontal_angle(x2, y2):
    new_h_angle = np.rad2deg(np.arctan2(y2, x2))
    return new_h_angle

def correct_vertical_angle(z2, rng):
    if rng < 1e-8:
        return 0.0
    new_v_angle = np.rad2deg(np.arcsin(np.clip(z2 / rng, -1, 1)))
    return new_v_angle

def correct_single_bin(q, h_angle_deg, v_angle_deg, rng, doppler):
    # 쿼터니언 성분 순서: (w, x, z, y) (이전 코드와 동일)
    w, x, z, y = q
    norm_q = np.sqrt(w*w + x*x + y*y + z*z)
    if norm_q < 1e-8:
        return rng, doppler, h_angle_deg, v_angle_deg
    h_rad = np.deg2rad(h_angle_deg)
    v_rad = np.deg2rad(v_angle_deg)
    x_ = rng * np.cos(v_rad) * np.cos(h_rad)
    y_ = rng * np.cos(v_rad) * np.sin(h_rad)
    z_ = rng * np.sin(v_rad)
    xyz_tilt_corrected = R_tilt_corr.dot([x_, y_, z_])
    inv_q = (w/norm_q, -x/norm_q, -y/norm_q, -z/norm_q)
    xyz_world = rotate_vector_by_quaternion(xyz_tilt_corrected, inv_q)
    xyz_scaled = SCALE_FACTOR * xyz_world
    x2, y2, z2 = xyz_scaled
    new_range = correct_range(rng, x2, y2, z2)
    new_doppler = correct_doppler(doppler)
    new_h_angle = correct_horizontal_angle(x2, y2)
    new_v_angle = correct_vertical_angle(z2, new_range)
    return new_range, new_doppler, new_h_angle, new_v_angle

def apply_geometric_correction(range_data, doppler_data, horizontal_angles, vertical_angles, quaternions):
    """
    각 배열 shape = (num_frames, num_bins) (여기서 모든 데이터가 64 bin)
    quaternions: (num_frames, 4)
    반환: (corrected_range, corrected_doppler, corrected_h_angle, corrected_v_angle)
    """
    num_frames, num_bins = range_data.shape
    corrected_range = np.zeros_like(range_data)
    corrected_doppler = np.zeros_like(doppler_data)
    corrected_h_angle = np.zeros_like(horizontal_angles)
    corrected_v_angle = np.zeros_like(vertical_angles)
    for f in range(num_frames):
        q = quaternions[f]
        for b in range(num_bins):
            rng = range_data[f, b]
            dop = doppler_data[f, b]
            h_angle = horizontal_angles[f, b]
            v_angle = vertical_angles[f, b]
            new_r, new_d, new_h, new_v = correct_single_bin(q, h_angle, v_angle, rng, dop)
            corrected_range[f, b] = new_r
            corrected_doppler[f, b] = new_d
            corrected_h_angle[f, b] = new_h
            corrected_v_angle[f, b] = new_v
    return corrected_range, corrected_doppler, corrected_h_angle, corrected_v_angle

#########################################
# 6) 보정 결과를 이미지로 저장하기
#########################################
def save_2d_array_as_image(array_2d, save_path):
    arr_min = array_2d.min()
    arr_max = array_2d.max()
    if arr_max - arr_min < 1e-8:
        norm_img = np.zeros_like(array_2d, dtype=np.uint8)
    else:
        norm_img = ((array_2d - arr_min) / (arr_max - arr_min)) * 255.0
        norm_img = norm_img.astype(np.uint8)
    cv2.imwrite(save_path, norm_img)

#########################################
# 7) 메인 파트
#########################################
def main():
    radar_dir = "/media/dokyeong/MINJI/dokyeong/Radar_RADimages_fram_norm_csv/"
    imu_dir   = "/media/dokyeong/MINJI/dokyeong/IMUData/"
    output_dir = "/media/dokyeong/MINJI/dokyeong/processed_2D_frame_norm/"

    radar_groups = group_radar_files(radar_dir)
    all_data = load_radar_and_imu_data(radar_groups, radar_dir, imu_dir)

    for (rel_dir, prefix), data_dict in all_data.items():
        radar_dict = data_dict['radar']
        imu_data   = data_dict['imu']
        if imu_data is None:
            print(f"[WARN] No IMU for {rel_dir}/{prefix}, skip")
            continue

        # IMU: 쿼터니언 추출 (여기서는 imu_data의 7~10열 사용)
        quaternions = imu_data[:, 7:11]  # shape = (num_frames, 4)

        doppler_arr = radar_dict.get('_doppler.csv', None)
        range_arr   = radar_dict.get('_range.csv', None)
        h_arr       = radar_dict.get('_horizontal.csv', None)
        v_arr       = radar_dict.get('_vertical.csv', None)

        if (doppler_arr is None or range_arr is None or
            h_arr is None or v_arr is None):
            print(f"[WARN] Some radar array is missing for {rel_dir}/{prefix}")
            continue

        # CSV 모양: 여기서는 모두 bin×frame로 가정하므로, 전치하여 frame×bin 형태로 만듦.
        # 모든 데이터가 64개의 bin을 가짐
        num_bins, num_frames = range_arr.shape  # expected: (64, num_frames)
        range_arr = range_arr.T    # (num_frames, 64)
        doppler_arr = doppler_arr.T  # (num_frames, 64)
        h_arr = h_arr.T            # (num_frames, 64)
        v_arr = v_arr.T            # (num_frames, 64)

        num_frames = range_arr.shape[0]

        # IMU frame 수 맞추기
        quaternions = trim_imu_data(quaternions, num_frames)
        assert quaternions.shape[0] == num_frames, "IMU 프레임 수와 레이더 프레임 수 불일치!"

        # (A) 기하학적 보정 적용 (모든 데이터가 64 bin)
        c_range, c_dop, c_h, c_v = apply_geometric_correction(range_arr, doppler_arr, h_arr, v_arr, quaternions)

        # (B) Rebinning: 모든 데이터는 64 bin으로 재배열
        rebinned_r = rebin_corrected_data(range_arr, c_range, num_bins=64)
        rebinned_d = rebin_corrected_data(doppler_arr, c_dop, num_bins=64)
        rebinned_h = rebin_corrected_data(h_arr, c_h, num_bins=64)
        rebinned_v = rebin_corrected_data(v_arr, c_v, num_bins=64)

        # (C) 결과 이미지 저장 (프레임별)
        out_subdir = os.path.join(output_dir, rel_dir)
        os.makedirs(out_subdir, exist_ok=True)

        for f in range(num_frames):
            # Range 이미지
            range_row = rebinned_r[f, :]  # shape=(64,)
            range_2d = range_row[None, :]  # (1, 64)
            save_path = os.path.join(out_subdir, f"{prefix}_frame{f:04d}_range.png")
            save_2d_array_as_image(range_2d, save_path)

            # Doppler 이미지
            dop_row = rebinned_d[f, :]
            dop_2d = dop_row[None, :]
            save_path_dop = os.path.join(out_subdir, f"{prefix}_frame{f:04d}_doppler.png")
            save_2d_array_as_image(dop_2d, save_path_dop)

            # Horizontal 이미지
            h_row = rebinned_h[f, :]
            h_2d = h_row[None, :]
            save_path_h = os.path.join(out_subdir, f"{prefix}_frame{f:04d}_horizontal.png")
            save_2d_array_as_image(h_2d, save_path_h)

            # Vertical 이미지
            v_row = rebinned_v[f, :]
            v_2d = v_row[None, :]
            save_path_v = os.path.join(out_subdir, f"{prefix}_frame{f:04d}_vertical.png")
            save_2d_array_as_image(v_2d, save_path_v)

        print(f"[INFO] Saved images -> {out_subdir}/{prefix}_frame****_XXX.png")

    print("=== ALL DONE ===")

if __name__ == "__main__":
    main()
