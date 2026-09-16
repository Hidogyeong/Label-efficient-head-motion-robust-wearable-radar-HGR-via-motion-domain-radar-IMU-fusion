#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np

###############################
# Quaternion Utility
###############################
def quat_to_rotation_matrix(q):
    """
    q: (w, x, y, z)
    3x3 회전 행렬 반환 (정규화 포함)
    """
    w, x, y, z = q
    norm_q = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm_q, x/norm_q, y/norm_q, z/norm_q
    R = np.array([
        [1 - 2*y**2 - 2*z**2,     2*x*y - 2*w*z,         2*x*z + 2*w*y],
        [2*x*y + 2*w*z,           1 - 2*x**2 - 2*z**2,   2*y*z - 2*w*x],
        [2*x*z - 2*w*y,           2*y*z + 2*w*x,         1 - 2*x**2 - 2*y**2]
    ], dtype=np.float64)
    return R

def rotate_vector_by_quaternion(v, q):
    """
    v: np.array([x, y, z])
    q: (w, x, y, z)
    쿼터니언으로 벡터 회전
    """
    R = quat_to_rotation_matrix(q)
    return R.dot(v)

def tilt_rotation_matrix(tilt_deg):
    """
    x축을 기준으로 tilt_deg 만큼 회전하는 행렬 생성.
    radar가 아래쪽으로 기울어져 있다면, 보정을 위해 -tilt_deg (즉, 위쪽으로 회전)시키면 됩니다.
    """
    tilt_rad = np.deg2rad(tilt_deg)
    R = np.array([
        [1, 0, 0],
        [0, np.cos(tilt_rad), -np.sin(tilt_rad)],
        [0, np.sin(tilt_rad),  np.cos(tilt_rad)]
    ], dtype=np.float64)
    return R

def geometric_correction(row):
    """
    각 행(row)에 있는 데이터를 이용해 기하학적 보정을 수행합니다.
    가정:
      - IMU 센서는 10cm 반경의 구 상에서 회전.
      - 손은 50cm 반경의 구 상에서 회전. (따라서 scale_factor=5)
      - 레이더가 실제 정면이 아니라 아래쪽으로 radar_tilt_deg 만큼 기울어져 있으므로 그 보정을 추가.
    """
    # 원본 데이터 추출
    # (여기서는 자이로/가속도 데이터는 사용하지 않고, 쿼터니언 및 레이더 데이터를 이용합니다)
    qw = row["q.w"]
    qx = row["q.i"]
    # 여기서는 쿼터니언 j, k의 순서를 사용자가 원한대로 변경 (예시 코드와 다르게)
    # (만약 따로 바꿔야 한다면 여기서 순서를 재정렬)
    qz = row["q.j"]   # 원래 q.j
    qy = row["q.k"]   # 원래 q.k
    
    h_angle = row["horizontal_angle"]
    v_angle = row["vertical_angle"]
    rng = row["range"]
    doppler = row["doppler"]
    
    # 각도를 라디안으로 변환
    h_rad = np.deg2rad(h_angle)
    v_rad = np.deg2rad(v_angle)
    
    # (1) 극좌표 -> 직교좌표 변환 (IMU 센서 기준: 10cm 반경)
    x = rng * np.cos(v_rad) * np.cos(h_rad)
    y = rng * np.cos(v_rad) * np.sin(h_rad)
    z = rng * np.sin(v_rad)
    
    # (1-1) 레이더가 아래쪽으로 기울어져 있다면 보정: 
    # 예를 들어, 레이더가 아래쪽으로 45° 기울어져 있다고 가정하면,
    # 보정을 위해 -45° (즉, 위로 회전)시키는 행렬을 적용
    radar_tilt_deg = 45.0  # 레이더가 아래쪽으로 45° 기울어져 있음
    R_tilt_corr = tilt_rotation_matrix(-radar_tilt_deg)
    xyz_tilt_corrected = R_tilt_corr.dot(np.array([x, y, z]))
    
    # (2) 쿼터니언 역회전 적용 (센서->월드 좌표 변환)
    norm_q = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    inv_q = (qw/norm_q, -qx/norm_q, -qy/norm_q, -qz/norm_q)
    xyz_world = rotate_vector_by_quaternion(xyz_tilt_corrected, inv_q)
    
    # (3) 센서와 손의 반경 차이 적용: 손 회전 반경은 50cm, 센서 회전 반경은 10cm
    scale_factor = 5.0  # 50 / 10
    xyz_scaled = scale_factor * xyz_world
    
    # (4) 스케일된 좌표를 다시 극좌표계로 변환
    x2, y2, z2 = xyz_scaled
    range_corr = np.sqrt(x2**2 + y2**2 + z2**2 + 1e-8)
    h_angle_corr = np.rad2deg(np.arctan2(y2, x2))
    v_angle_corr = np.rad2deg(np.arcsin(np.clip(z2 / (range_corr + 1e-8), -1, 1)))
    
    # Doppler는 그대로 사용 (추후 보정 추가 가능)
    doppler_corr = doppler

    return {
        "range_corr": range_corr,
        "doppler_corr": doppler_corr,
        "horizontal_angle_corr": h_angle_corr,
        "vertical_angle_corr": v_angle_corr
    }

# =====================================================================
# 메인 파트: 디렉토리를 순회하여 CSV를 처리하고, 동일한 폴더 구조로 출력
# =====================================================================
def main():
    source_root = "/home/dokyeong/miniconda3/envs/hgr_sci/processing_data_set/MergedData_all_BLPF_10Hz_v2"
    target_root = "/media/dokyeong/MINJI/dokyeong/processedData_geometry_with_head_v8"

    for dirpath, _, filenames in os.walk(source_root):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                src_file = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(dirpath, source_root)
                dst_dir = os.path.join(target_root, rel_path)
                os.makedirs(dst_dir, exist_ok=True)
                dst_file = os.path.join(dst_dir, filename)

                # CSV 읽기
                df = pd.read_csv(src_file)

                # 각 행에 대해 보정 수행
                correction_results = df.apply(geometric_correction, axis=1)
                corr_df = pd.DataFrame(list(correction_results))

                # 보정 결과 컬럼을 원본 DataFrame에 추가
                df["range_corr"] = corr_df["range_corr"]
                df["doppler_corr"] = corr_df["doppler_corr"]
                df["horizontal_angle_corr"] = corr_df["horizontal_angle_corr"]
                df["vertical_angle_corr"] = corr_df["vertical_angle_corr"]

                # CSV 저장
                df.to_csv(dst_file, index=False, encoding="utf-8-sig")
                print(f"[SAVE] {dst_file}")

if __name__ == "__main__":
    main()
