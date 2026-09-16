#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np

# 쿼터니언을 회전 행렬로 변환
def quat_to_rotation_matrix(q):
    """
    q: (w, x, y, z)
    3x3 회전 행렬 반환
    """
    w, x, y, z = q
    R = np.array([
        [1 - 2*y**2 - 2*z**2,     2*x*y - 2*w*z,         2*x*z + 2*w*y],
        [2*x*y + 2*w*z,           1 - 2*x**2 - 2*z**2,   2*y*z - 2*w*x],
        [2*x*z - 2*w*y,           2*y*z + 2*w*x,         1 - 2*x**2 - 2*y**2]
    ], dtype=np.float64)
    return R

# 벡터를 쿼터니언으로 회전
def rotate_vector_by_quaternion(v, q):
    """
    v: np.array([x, y, z])
    q: (w, x, y, z)
    쿼터니언으로 벡터 회전
    """
    R = quat_to_rotation_matrix(q)
    return R.dot(v)

def geometric_correction(row):
    """
    각 행(row)에 있는 데이터를 이용해 기하학적 보정을 수행합니다.
    가정:
      - IMU 센서는 10cm 반경의 구 상에서 회전.
      - 손은 50cm 반경의 구 상에서 회전.
      
    따라서, 센서의 측정 결과에 대해 5배 스케일링을 적용하여 손의 회전 반경 기준 보정을 수행합니다.
    """
    # 원본 데이터 추출
    qw = row["q.w"]
    qx = row["q.i"]
    qz = row["q.j"]
    qy = row["q.k"]
    h_angle = row["horizontal_angle"]
    v_angle = row["vertical_angle"]
    rng = row["range"]
    doppler = row["doppler"]
    
    # 각도를 라디안으로 변환
    h_rad = np.deg2rad(h_angle)
    v_rad = np.deg2rad(v_angle)
    
    # (1) 극좌표 -> 직교좌표 변환 (IMU 센서 기준, 10cm 반경)
    x = rng * np.cos(v_rad) * np.cos(h_rad)
    y = rng * np.cos(v_rad) * np.sin(h_rad)
    z = rng * np.sin(v_rad)
    
    # (2) 쿼터니언 역회전 적용 (간단화를 위해 역쿼터니언은 (w, -x, -y, -z)로 계산)
    norm_q = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    inv_q = (qw/norm_q, -qx/norm_q, -qy/norm_q, -qz/norm_q)
    xyz_world = rotate_vector_by_quaternion(np.array([x, y, z]), inv_q)
    
    # (3) 센서와 손의 반경 차이 적용: 손 회전 반경은 50cm, 센서 회전 반경은 10cm
    scale_factor = 5.0  # 50 / 10
    xyz_scaled = scale_factor * xyz_world
    
    # (4) 스케일된 좌표를 다시 극좌표계로 변환
    x2, y2, z2 = xyz_scaled
    range_corr = np.sqrt(x2**2 + y2**2 + z2**2 + 1e-8)
    h_angle_corr = np.rad2deg(np.arctan2(y2, x2))
    v_angle_corr = np.rad2deg(np.arcsin(np.clip(z2 / (range_corr + 1e-8), -1, 1)))
    
    # Doppler는 그대로 사용 (추후 보정 필요시 추가 가능)
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
    target_root = "/media/dokyeong/MINJI/dokyeong/processedData_geometry_with_head_v2"

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
