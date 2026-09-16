#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np

###################################
# Quaternion Utilities
###################################
def quat_to_rotation_matrix(q):
    """
    q: (w, x, y, z)
    -> 3x3 rotation matrix
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2,   2*x*y - 2*w*z,       2*x*z + 2*w*y],
        [2*x*y + 2*w*z,         1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,         2*y*z + 2*w*x,       1 - 2*x**2 - 2*y**2]
    ], dtype=np.float64)

def rotate_vector_by_quaternion(v, q):
    """
    v: [x, y, z]  / q: (w, x, y, z)
    -> R*v
    """
    R = quat_to_rotation_matrix(q)
    return R @ v  # 행렬곱

def normalize_quaternion(q):
    return q / np.linalg.norm(q)

def quaternion_product(q1, q2):
    """
    q1, q2: (w, x, y, z)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return (w, x, y, z)

###################################
# 1) Quaternion derivative
###################################
def quaternion_derivative(q, gyr):
    """
    q: current quaternion (w, x, y, z)
    gyr: angular velocity (gyr_x, gyr_y, gyr_z) in rad/s
    -> dq/dt (size=4)
    dq/dt = 0.5 * q * (0, gyr_x, gyr_y, gyr_z)
    """
    w, x, y, z = q
    gx, gy, gz = gyr  # (rad/s)

    # 쿼터니언 곱을 직접 계산
    # dq/dt = 0.5 * ( w, x, y, z ) * (0, gx, gy, gz )
    dw = 0.5 * ( - x*gx - y*gy - z*gz )
    dx = 0.5 * (   w*gx + y*gz - z*gy )
    dy = 0.5 * (   w*gy - x*gz + z*gx )
    dz = 0.5 * (   w*gz + x*gy - y*gx )

    return (dw, dx, dy, dz)

def integrate_quaternion(q_prev, gyr, dt):
    """
    q_prev: previous quaternion
    gyr: (gx, gy, gz) in rad/s
    dt: time step
    -> returns updated quaternion q(t+dt)
    """
    dqdt = quaternion_derivative(q_prev, gyr)
    q_new = (
        q_prev[0] + dqdt[0]*dt,
        q_prev[1] + dqdt[1]*dt,
        q_prev[2] + dqdt[2]*dt,
        q_prev[3] + dqdt[3]*dt
    )
    # 정규화
    q_new = normalize_quaternion(np.array(q_new, dtype=np.float64))
    return tuple(q_new)

###################################
# 2) Geometric Correction
###################################
def geometric_correction(row, q_prev, dt):
    """
    row에는:
      - Gyr.X, Gyr.Y, Gyr.Z (deg/s or rad/s)
      - q.w, q.i, q.j, q.k
      - range, doppler, horizontal_angle, vertical_angle
    q_prev: 이전 쿼터니언 (누적 회전 상태)
    dt: time step(초단위)

    -> '머리가 움직이지 않는 것처럼' 보정
    """

    # 1) 자이로 (deg/s -> rad/s 변환) 필요 시
    gyr_x = np.deg2rad(row["Gyr.X"])
    gyr_y = np.deg2rad(row["Gyr.Y"])
    gyr_z = np.deg2rad(row["Gyr.Z"])
    gyr = (gyr_x, gyr_y, gyr_z)

    # 2) q_prev(이전 시점)에서 자이로 미분으로 q_now 업데이트
    q_now = integrate_quaternion(q_prev, gyr, dt)

    # 3) 레이더 극좌표 -> Cartesian
    rng = row["range"]
    doppler = row["doppler"]
    h = np.deg2rad(row["horizontal_angle"])
    v = np.deg2rad(row["vertical_angle"])
    x = rng*np.cos(v)*np.cos(h)
    y = rng*np.cos(v)*np.sin(h)
    z = rng*np.sin(v)

    # 4) 역쿼터니언 적용(머리->월드)
    #   inverse q = (w, -x, -y, -z)
    w, xi, yi, zi = q_now
    inv_q = (w, -xi, -yi, -zi)
    xyz_world = rotate_vector_by_quaternion([x, y, z], inv_q)

    # 5) 다시 spherical로
    x2, y2, z2 = xyz_world
    range_corr = np.sqrt(x2**2 + y2**2 + z2**2 + 1e-8)
    h_angle_corr = np.rad2deg(np.arctan2(y2, x2))
    v_angle_corr = np.rad2deg(np.arcsin(z2/(range_corr+1e-8)))

    doppler_corr = doppler  # 추가 보정 가능

    # (출력, q_now도 반환해서 다음 시점에 활용)
    return {
        "range_corr": range_corr,
        "doppler_corr": doppler_corr,
        "horizontal_angle_corr": h_angle_corr,
        "vertical_angle_corr": v_angle_corr
    }, q_now

###################################
# 3) Main flow
###################################
def main():
    source_root = "/home/dokyeong/miniconda3/envs/hgr_sci/processing_data_set/MergedData_all_BLPF_10Hz_v2"
    target_root = "/media/dokyeong/MINJI/dokyeong/processedData_geometry_v9"

    dt = 0.01  # 예시로 0.01s (100Hz)라 가정
    for dirpath, dirnames, filenames in os.walk(source_root):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                src_file = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(dirpath, source_root)
                dst_dir = os.path.join(target_root, rel_path)
                os.makedirs(dst_dir, exist_ok=True)

                dst_file = os.path.join(dst_dir, filename)
                df = pd.read_csv(src_file)

                # q_prev = 초기 쿼터니언(머리가 정면 보고 있다고 가정)
                q_prev = (1.0, 0.0, 0.0, 0.0)

                range_list = []
                doppler_list = []
                hangle_list = []
                vangle_list = []

                # (1) 행을 순회하면서, 매 시점마다 q_prev를 갱신하고
                # (2) 보정된 레이더 데이터를 구한다.
                for _, row in df.iterrows():
                    result, q_now = geometric_correction(row, q_prev, dt)
                    range_list.append(result["range_corr"])
                    doppler_list.append(result["doppler_corr"])
                    hangle_list.append(result["horizontal_angle_corr"])
                    vangle_list.append(result["vertical_angle_corr"])

                    # q_now를 다음 시점의 q_prev로 사용
                    q_prev = q_now

                df["range_corr"] = range_list
                df["doppler_corr"] = doppler_list
                df["horizontal_angle_corr"] = hangle_list
                df["vertical_angle_corr"] = vangle_list

                df.to_csv(dst_file, index=False, encoding="utf-8-sig")
                print(f"[SAVE] {dst_file}")

if __name__ == "__main__":
    main()
