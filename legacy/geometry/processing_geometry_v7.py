#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np

# 상수 정의
RADAR_TILT_DEG = 45.0  # 레이더가 아래쪽으로 기울어진 각도 (보정 시 사용)
SCALE_FACTOR = 5.0     # 손(50cm)과 센서(10cm) 반경 비율

###############################
# Quaternion Utility
###############################
def quat_to_rotation_matrix(q):
    """
    q: (w, x, y, z)
    단위 쿼터니언으로 정규화한 후 3x3 회전 행렬 반환
    """
    if len(q) != 4:
        raise ValueError("입력 쿼터니언은 4개의 요소를 가져야 합니다.")
    w, x, y, z = q
    norm_q = np.sqrt(w*w + x*x + y*y + z*z)
    if norm_q < 1e-8:
        raise ValueError("정규화할 수 없는 0에 가까운 쿼터니언입니다.")
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
    쿼터니언으로 벡터 회전. q를 회전 행렬로 변환 후 v에 적용.
    """
    R = quat_to_rotation_matrix(q)
    return R.dot(v)

def tilt_rotation_matrix(tilt_deg):
    """
    Y축을 기준으로 tilt_deg 만큼 회전하는 행렬 생성.
    일반적인 레이더 좌표계(전방: X, 측면: Y, 상방: Z)에서,
    레이더가 아래로 기울어졌다면, 보정을 위해 양의 tilt_deg(예: 45°)만큼 위로 회전시킵니다.
    
    회전 행렬 (Y축 회전):
      [ cosθ    0    sinθ ]
      [   0     1      0  ]
      [ -sinθ   0    cosθ ]
    """
    tilt_rad = np.deg2rad(tilt_deg)
    R = np.array([
        [np.cos(tilt_rad), 0, np.sin(tilt_rad)],
        [0, 1, 0],
        [-np.sin(tilt_rad), 0, np.cos(tilt_rad)]
    ], dtype=np.float64)
    return R

def geometric_correction(row):
    """
    각 행(row)의 데이터를 사용하여 기하학적 보정을 수행합니다.
    가정:
      - IMU 센서는 10cm 반경 구 상에서 회전.
      - 손은 50cm 반경 구 상에서 회전 (따라서 SCALE_FACTOR = 5).
      - 레이더는 실제 정면이 아닌 아래쪽으로 RADAR_TILT_DEG만큼 기울어져 있으므로 보정 적용.
    """
    # 원본 데이터 추출 (쿼터니언 성분 순서 재정렬)
    qw = row["q.w"]
    qx = row["q.i"]
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
    
    # (1-1) 레이더 기울기 보정: 
    # 레이더가 아래로 RADAR_TILT_DEG 기울어졌다면, 보정을 위해 +RADAR_TILT_DEG 만큼 위로 회전
    R_tilt_corr = tilt_rotation_matrix(RADAR_TILT_DEG)
    xyz_tilt_corrected = R_tilt_corr.dot(np.array([x, y, z]))
    
    # (2) 쿼터니언 역회전 적용 (센서 -> 월드 좌표 변환)
    norm_q = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if norm_q < 1e-8:
        raise ValueError("정규화할 수 없는 0에 가까운 쿼터니언입니다.")
    inv_q = (qw/norm_q, -qx/norm_q, -qy/norm_q, -qz/norm_q)
    xyz_world = rotate_vector_by_quaternion(xyz_tilt_corrected, inv_q)
    
    # (3) 센서와 손의 반경 차이 적용: SCALE_FACTOR = 5 (50cm/10cm)
    xyz_scaled = SCALE_FACTOR * xyz_world
    
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
    target_root = "/media/dokyeong/MINJI/dokyeong/processedData_geometry_with_head_v9"

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
