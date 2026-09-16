#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np

# (선택) 필요하다면 quaternion 보정 함수를 구현 예시
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
    각 행(row)에 있는 Radar+IMU 정보로부터 
    '머리가 움직이지 않는 것처럼' 보정된 Radar 값을 산출하는 예시.
    여기서는 아주 단순히 horizontal_angle, vertical_angle만
    쿼터니언 역회전한다고 가정(실제 로직은 적절히 확장).
    """

    # 예시: row에서 쿼터니언과 기존 레이더 angle 추출
    qw = row["q.w"]
    qx = row["q.i"]
    qy = row["q.j"]
    qz = row["q.k"]
    h_angle = row["horizontal_angle"]
    v_angle = row["vertical_angle"]
    rng = row["range"]
    doppler = row["doppler"]

    # (1) 레이더 극좌표 -> 직교좌표
    h_rad = np.deg2rad(h_angle)
    v_rad = np.deg2rad(v_angle)
    x = rng * np.cos(v_rad) * np.cos(h_rad)
    y = rng * np.cos(v_rad) * np.sin(h_rad)
    z = rng * np.sin(v_rad)

    # (2) 쿼터니언을 이용해 역회전(머리→월드)
    # 여기서는 inverse_quaternion 대신 직접 (w, -x, -y, -z)를 적용
    # norm 보정을 위해서는 실제 역쿼터니언 계산이 필요
    # 예시를 간단화
    inv_q = (qw, -qx, -qy, -qz)
    xyz_world = rotate_vector_by_quaternion(np.array([x, y, z]), inv_q)

    # (3) 다시 spherical로
    x2, y2, z2 = xyz_world
    range_corr = np.sqrt(x2**2 + y2**2 + z2**2 + 1e-8)
    h_angle_corr = np.rad2deg(np.arctan2(y2, x2))
    v_angle_corr = np.rad2deg(np.arcsin(z2 / (range_corr + 1e-8)))

    # (4) Doppler도 센서 이동 속도 등을 빼주는 로직을 넣을 수 있음(예시 생략)
    doppler_corr = doppler  # 단순히 그대로 두거나

    # 보정된 값 반환 (DataFrame에 넣기 위해 dict로)
    return {
        "range_corr": range_corr,
        "doppler_corr": doppler_corr,
        "horizontal_angle_corr": h_angle_corr,
        "vertical_angle_corr": v_angle_corr
    }

# =====================================================================
# 메인 파트: 디렉토리를 순회하여 CSV를 처리하고, 동일 폴더 구조로 출력
# =====================================================================
def main():
    source_root = "/home/dokyeong/miniconda3/envs/hgr_sci/processing_data_set/MergedData_all_BLPF"
    target_root = "/media/dokyeong/MINJI/dokyeong/processedData_geometry_v1"

    # 확장자가 CSV인 파일만 처리한다고 가정(필요시 수정)
    for dirpath, dirnames, filenames in os.walk(source_root):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                # 소스 파일 전체 경로
                src_file = os.path.join(dirpath, filename)

                # 소스 경로에서 source_root 부분을 target_root로 대체
                # => 동일한 하위 디렉토리 구조를 만들기 위함
                rel_path = os.path.relpath(dirpath, source_root)
                dst_dir = os.path.join(target_root, rel_path)
                os.makedirs(dst_dir, exist_ok=True)

                dst_file = os.path.join(dst_dir, filename)

                # (1) CSV 읽기
                df = pd.read_csv(src_file)

                # (2) 각 행에 대해 보정 수행
                # 실제로는 df.apply(geometric_correction, axis=1) 등을 사용
                correction_results = df.apply(geometric_correction, axis=1)
                
                # correction_results는 Series of dict 형태가 됨
                # 이를 별도 dataframe으로 변환 후 원본 df와 합칠 수도 있음
                corr_df = pd.DataFrame(list(correction_results))

                # 원본 df에 보정 컬럼을 붙이거나 대체(필요에 맞게)
                df["range_corr"] = corr_df["range_corr"]
                df["doppler_corr"] = corr_df["doppler_corr"]
                df["horizontal_angle_corr"] = corr_df["horizontal_angle_corr"]
                df["vertical_angle_corr"] = corr_df["vertical_angle_corr"]

                # (3) 수정된 df를 CSV로 저장
                df.to_csv(dst_file, index=False, encoding="utf-8-sig")
                print(f"[SAVE] {dst_file}")

if __name__ == "__main__":
    main()
