#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import os
import glob
import numpy as np
import cv2  # OpenCV로 이미지 저장

#########################################
# Doppler 보정에 필요한 상수 및 함수들
#########################################
C = 3e8  # 빛의 속도 (m/s)
CENTER_FREQ_HZ = 60.75e9  # 예: 60.75GHz
LAMBDA = C / CENTER_FREQ_HZ  # 파장 (m)
LEVER_ARM = np.array([0.05, 0.0, 0.05])  # 예: 센서 기준, x축 기준 5cm 전방

def get_sensor_velocity_from_gyro(row):
    # IMU 1프레임: row의 인덱스 4~6 (각속도: rad/s)
    w_x, w_z, w_y = row  # 데이터 순서에 맞게 조정
    omega = np.array([w_x, w_y, w_z])
    v_sensor = np.cross(omega, LEVER_ARM)
    return v_sensor

def get_unit_vector_from_angles(h_deg, v_deg):
    h_rad = np.deg2rad(h_deg)
    v_rad = np.deg2rad(v_deg)
    x = np.cos(v_rad) * np.cos(h_rad)
    y = np.cos(v_rad) * np.sin(h_rad)
    z = np.sin(v_rad)
    r_hat = np.array([x, y, z])
    return r_hat / np.linalg.norm(r_hat)

def apply_doppler_correction(c_doppler, c_h_angle, c_v_angle, gyro_data, num_frame):
    num_bin = c_doppler.shape
    corrected_doppler = np.zeros_like(c_doppler)
    print(c_doppler)

    sensor_vel = get_sensor_velocity_from_gyro(gyro_data)
    r_hat = get_unit_vector_from_angles(c_h_angle, c_v_angle)
    delta_doppler = np.dot(sensor_vel, r_hat) / LAMBDA  # (Hz)
    # Δv = Δf * (λ/2); 각 bin 당 0.2 m/s 차이로 가정
    delta_velocity = delta_doppler * (LAMBDA / 2)
    corrected_doppler = c_doppler + delta_velocity

        # for b in range(num_bins):
        #     r_hat = get_unit_vector_from_angles(c_h_angle[f, b], c_v_angle[f, b])
        #     delta_doppler = np.dot(sensor_vel, r_hat) / LAMBDA  # (Hz)
        #     # Δv = Δf * (λ/2); 각 bin 당 0.2 m/s 차이로 가정
        #     delta_velocity = delta_doppler * (LAMBDA / 2)
        #     corrected_doppler[f, b] = c_doppler[f, b] + delta_velocity
    return corrected_doppler

#########################################
# 0) 파일 검색 및 그룹핑
#########################################
def group_radar_files(radar_dir):
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
                # print(f"[LOAD] Radar: {path}, shape={arr.shape}")
            except Exception as e:
                # print(f"[ERROR] Radar file {path}: {e}")
                radar_data_dict[suf] = None
        imu_filename = "I" + prefix + ".csv"
        imu_filepath = os.path.join(imu_dir, rel_dir, imu_filename)
        imu_data = None
        if os.path.exists(imu_filepath):
            try:
                imu_data = np.loadtxt(imu_filepath, delimiter=",", skiprows=1)
                # print(f"[LOAD] IMU: {imu_filepath}, shape={imu_data.shape}")
            except Exception as e:
                # print(f"[ERROR] IMU file {imu_filepath}: {e}")
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
# 3) Rebin 함수: 물리량을 bin 인덱스로 변환
#########################################
def rebin_with_conversion(original_data, corrected_values, conversion_func, num_bins):
    num_frames = original_data.shape[0]
    new_data = np.zeros((num_frames, num_bins))
    for f in range(num_frames):
        for b in range(original_data.shape[1]):
            new_idx = int(round(conversion_func(corrected_values[f, b])))
            new_idx = max(0, min(new_idx, num_bins - 1))
            new_data[f, new_idx] += original_data[f, b]
    return new_data

# 변환 함수들: 실제 물리량 -> bin 인덱스
def angle_to_bin(angle):
    # 보정 후 각도가 -40° ~ 40°이면, 1.25°/bin → bin = (angle + 40) / 1.25
    return (angle + 40) / 1.25

def range_to_bin(rng):
    return rng / 0.03

def doppler_to_bin(dop):
    # 1 bin 당 0.2 m/s, 중앙이 0 m/s일 때, bin index = dop / 0.2 + 32
    return dop / 0.2 + 32

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
SCALE_FACTOR = 1.0  # 상황에 맞게 조정
R_tilt_corr = tilt_rotation_matrix(RADAR_TILT_DEG)

def correct_range(rng, x2, y2, z2):
    return np.sqrt(x2*x2 + y2*y2 + z2*z2 + 1e-8)

def correct_doppler(doppler):
    return doppler

def correct_horizontal_angle(x2, y2):
    return np.rad2deg(np.arctan2(y2, x2))

def correct_vertical_angle(z2, rng):
    epsilon = 1e-8
    safe_rng = np.where(rng < epsilon, epsilon, rng)
    return np.rad2deg(np.arcsin(np.clip(z2 / safe_rng, -1, 1)))

def correct_single_bin(q, g, h_angle_deg, v_angle_deg, rng, doppler):
    # 쿼터니언 성분 순서: (w, x, z, y)
    w, x, z, y = q
    x, z, y = g
    norm_q = np.sqrt(w*w + x*x + y*y + z*z)
    # 1) concatenate로 묶기

    if norm_q < 1e-8:
        return rng, doppler, h_angle_deg, v_angle_deg
    
    h_rad = np.deg2rad(h_angle_deg)

    v_rad = np.deg2rad(v_angle_deg)

    x_ = rng * np.cos(v_rad) * np.cos(h_rad)
    y_ = rng * np.cos(v_rad) * np.sin(h_rad)
    z_ = rng * np.sin(v_rad)
    # print(x_)
    # print(y_)
    # print(z_)
    xyz_tilt_corrected = R_tilt_corr.dot([x_, y_, z_])
    # print(xyz_tilt_corrected)

    inv_q = (w/norm_q, -x/norm_q, -y/norm_q, -z/norm_q)

    xyz_world = rotate_vector_by_quaternion(xyz_tilt_corrected, inv_q)
    xyz_scaled = SCALE_FACTOR * xyz_world
    x2, y2, z2 = xyz_scaled

    new_range = correct_range(rng, x2, y2, z2)
    # new_doppler = correct_doppler(doppler)  # 도플러 보정은 별도 처리
    new_h_angle = correct_horizontal_angle(x2, y2)
    new_v_angle = correct_vertical_angle(z2, new_range)

    new_doppler = apply_doppler_correction(doppler, new_h_angle, new_v_angle, g)

    return new_range, new_doppler, new_h_angle, new_v_angle

def apply_geometric_correction(range_data, doppler_data, horizontal_angles, vertical_angles, quaternions, gyro):
    """
    모든 입력 배열의 shape = (num_frames, 64) (실제 물리량 단위)
    quaternions: (num_frames, 4)
    반환: (corrected_range, corrected_doppler, corrected_h_angle, corrected_v_angle)
    """
    num_frames, num_bins = range_data.shape
    # print(num_bins)
    # print(num_frames)
    c_range = np.zeros((num_bins, num_frames))
    c_dopp  = np.zeros((num_bins, num_frames))
    c_h     = np.zeros((num_bins, num_frames))
    c_v     = np.zeros((num_bins, num_frames))

    # for f in range(num_frames):
    for f in range(1):
        q = quaternions[f]
        g = gyro[f] * 0.07 #degree/s
        print(gyro[f])
        print(g)
        # Range
        rng = range_data[f] # 2 1 3 0 4 5 6
        sorted_indices_rng = np.argsort(rng)[::-1]
        top7_indices_rng = sorted_indices_rng[:7]
        real_rng = top7_indices_rng * 0.3


        # Doppler
        dop = doppler_data[f]
        sorted_indices_dop = np.argsort(dop)[::-1]
        top7_indices_dop = sorted_indices_dop[:7]
        real_dop = (top7_indices_dop - 32) * 0.2

        # H_Angle
        h_angle = horizontal_angles[f]
        sorted_indices_h_angle = np.argsort(h_angle)[::-1]
        top7_indices_h_angle = sorted_indices_h_angle[:7]
        real_h_angle = (top7_indices_h_angle - 31.5) * 1.25
        # print(real_h_angle)
        

        # V_Angle
        v_angle = vertical_angles[f]
        sorted_indices_v_angle = np.argsort(v_angle)[::-1]
        top7_indices_v_angle = sorted_indices_v_angle[:7]
        real_v_angle = (top7_indices_v_angle - 31.5) * 1.25

        nr, nd, nh, nv = correct_single_bin(q, g, real_h_angle, real_v_angle, real_rng, real_dop)

        c_range[f, :] = nr 
        c_dopp[f, :]  = nd
        c_h[f, :]     = nh
        c_v[f, :]     = nv

        # for b in range(num_bins):
        #     rng = range_data[f, b]
        #     print(rng.shape)
        #     dop = doppler_data[f, b]
        #     h_angle = horizontal_angles[f, b]
        #     v_angle = vertical_angles[f, b]
        #     nr, nd, nh, nv = correct_single_bin(q, h_angle, v_angle, rng, dop)
        #     c_range[f, b] = nr
        #     c_dopp[f, b]  = nd
        #     c_h[f, b]     = nh
        #     c_v[f, b]     = nv
    return c_range, c_dopp, c_h, c_v

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
# 7) 변환 함수: 실제 물리량 -> bin 인덱스
#########################################
def angle_to_bin(angle):
    return (angle + 40) / 1.25

def range_to_bin(rng):
    return rng / 0.03

def doppler_to_bin(dop):
    return dop / 0.2 + 32

#########################################
# 8) 새로운 Rebin 함수: 변환 함수를 인자로 받음.
#########################################
def rebin_with_conversion(original_data, corrected_values, conversion_func, num_bins):
    num_frames = original_data.shape[0]
    new_data = np.zeros((num_frames, num_bins))
    for f in range(num_frames):
        for b in range(original_data.shape[1]):
            new_idx = int(round(conversion_func(corrected_values[f, b])))
            new_idx = max(0, min(new_idx, num_bins - 1))
            new_data[f, new_idx] += original_data[f, b]
    return new_data

#########################################
# 9) CSV에 저장된 bin 인덱스를 실제 물리량으로 변환하는 함수들 (역변환)
#########################################
def bin_to_angle(bin_val):
    return bin_val * 1.25 - 40

def bin_to_range(bin_val):
    return bin_val * 0.03

def bin_to_doppler(bin_val):
    return (bin_val - 32) * 0.2

#########################################
# 10) 메인 파트
#########################################
def main():
    radar_dir = "/media/dokyeong/MINJI/dokyeong/Radar_RADimages_fram_norm_csv/junh/01RtoL/None/"
    imu_dir = "/media/dokyeong/MINJI/dokyeong/IMUData/junh/01RtoL/None/"
    output_dir = "/media/dokyeong/MINJI/dokyeong/processed_2D_frame_norm_1/junh/01RtoL/None/"

    PRF = 1000.0          # 예: 1 kHz
    CENTER_FREQ = 60e9    # 예: 60 GHz

    radar_groups = group_radar_files(radar_dir)
    all_data = load_radar_and_imu_data(radar_groups, radar_dir, imu_dir)

    for (rel_dir, prefix), data_dict in all_data.items():
        radar_dict = data_dict['radar']
        imu_data = data_dict['imu']
        if imu_data is None:
            print(f"[WARN] No IMU for {rel_dir}/{prefix}, skip")
            continue

        # IMU: 쿼터니언 추출 (7~10열 사용)
        quaternions = imu_data[:, 7:11]  # (num_frames, 4)
        gyro_data = imu_data[:,1:4]
        imu_full = imu_data  # 전체 IMU 데이터 사용

        doppler_arr = radar_dict.get('_doppler.csv', None)
        range_arr = radar_dict.get('_range.csv', None)
        h_arr = radar_dict.get('_horizontal.csv', None)
        v_arr = radar_dict.get('_vertical.csv', None)

        if (doppler_arr is None or range_arr is None or h_arr is None or v_arr is None):
            print(f"[WARN] Some radar array is missing for {rel_dir}/{prefix}")
            continue

        # CSV 모양: (bin x frame) → 전치하여 (frame x bin) 형태; 모든 데이터 64 bin으로 가정
        num_bins, num_frames = range_arr.shape  # 예상: (64, num_frames)
        range_arr = range_arr.T      # (frame, 64)
        doppler_arr = doppler_arr.T  # (frame, 64)
        h_arr = h_arr.T              # (frame, 64)
        v_arr = v_arr.T              # (frame, 64)
        num_frames = range_arr.shape[0]

        # print(num_frames)


        # IMU 프레임 수 맞추기
        quaternions = trim_imu_data(quaternions, num_frames)
        imu_full = trim_imu_data(imu_full, num_frames)
        assert quaternions.shape[0] == num_frames, "IMU 프레임 수와 레이더 프레임 수 불일치!"

        # (A) 기하학적 보정 적용 (실제 물리량 단위에서 보정)
        c_range, c_dop, c_h, c_v = apply_geometric_correction(range_arr, doppler_arr, h_arr, v_arr, quaternions, gyro_data)

        # (B) 도플러 보정 적용 (IMU 각속도 등 사용)
        # c_dop_corrected = apply_doppler_correction(c_dop, c_h, c_v, imu_full)

        # (C) Rebinning: 보정된 물리량을 다시 bin 인덱스로 변환 (모든 데이터 64 bin)
        # rebinned_r = rebin_with_conversion(range_arr, c_range, range_to_bin, num_bins=64)
        # rebinned_d = rebin_with_conversion(doppler_arr, c_dop, doppler_to_bin, num_bins=64)
        # rebinned_h = rebin_with_conversion(h_arr, c_h, angle_to_bin, num_bins=64)
        # rebinned_v = rebin_with_conversion(v_arr, c_v, angle_to_bin, num_bins=64)

    #     # (D) 결과 이미지 저장: 전체 파일(모든 프레임)을 하나의 이미지로 저장
    #     out_subdir = os.path.join(output_dir, rel_dir)
    #     os.makedirs(out_subdir, exist_ok=True)
        
    #     # 각 측정값에 대해 전체 프레임의 2D 배열을 이미지로 저장 (행: 프레임, 열: bin)
    #     save_path_range = os.path.join(out_subdir, f"{prefix}_all_frames_range.png")
    #     save_2d_array_as_image(rebinned_r, save_path_range)
        
    #     save_path_doppler = os.path.join(out_subdir, f"{prefix}_all_frames_doppler.png")
    #     save_2d_array_as_image(rebinned_d, save_path_doppler)
        
    #     save_path_horizontal = os.path.join(out_subdir, f"{prefix}_all_frames_horizontal.png")
    #     save_2d_array_as_image(rebinned_h, save_path_horizontal)
        
    #     save_path_vertical = os.path.join(out_subdir, f"{prefix}_all_frames_vertical.png")
    #     save_2d_array_as_image(rebinned_v, save_path_vertical)

    #     print(f"[INFO] Saved images -> {out_subdir}/{prefix}_all_frames_XXX.png")

    # print("=== ALL DONE ===")

if __name__ == "__main__":
    main()
