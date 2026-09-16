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
    3x3 회전 행렬 반환 (자동 정규화 포함)
    공식:
      R = [[1 - 2(y^2+z^2),   2(xy - wz),       2(xz + wy)],
           [2(xy + wz),       1 - 2(x^2+z^2),   2(yz - wx)],
           [2(xz - wy),       2(yz + wx),       1 - 2(x^2+y^2)]]
    """
    w, x, y, z = q
    norm_q = np.sqrt(w*w + x*x + y*y + z*z)
    if norm_q < 1e-12:
        # 혹시 모를 0 나눗셈 방지
        # 단위 행렬 리턴 또는 경고 처리
        print("[WARN] Quaternion norm is near zero, returning identity matrix.")
        return np.eye(3, dtype=np.float64)

    w, x, y, z = w/norm_q, x/norm_q, y/norm_q, z/norm_q
    # 로드리게스 회전 공식에 맞게 계산
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y]
    ], dtype=np.float64)
    return R


def rotate_vector_by_quaternion(v, q):
    """
    v: np.array([x, y, z]) - 회전시킬 벡터
    q: (w, x, y, z)         - 회전 쿼터니언 (정규화 필요)
    
    벡터를 q로 회전:
      v_rot = R(q) * v
    여기서 R(q)는 q를 이용해 만든 3x3 회전 행렬
    """
    R = quat_to_rotation_matrix(q)
    return R.dot(v)


def get_inverse_quaternion(q):
    """
    주어진 쿼터니언 q의 역(=켤레) 쿼터니언 반환
    q가 단위 쿼터니언일 때:
      q^{-1} = ( w, -x, -y, -z )
    """
    w, x, y, z = q
    norm_q = np.sqrt(w*w + x*x + y*y + z*z)
    if norm_q < 1e-12:
        print("[WARN] Quaternion norm is near zero, returning same quaternion.")
        return q
    return (w/norm_q, -x/norm_q, -y/norm_q, -z/norm_q)


def tilt_rotation_matrix(tilt_deg):
    """
    radar가 아래로 tilt_deg (ex: 45)도 기울어졌을 때,
    위로 tilt_deg도 회전하는 x축 기준 회전행렬.
    여기서는 x축이 좌->우, y축이 전->후, z축이 상->하 라고 가정하는지 확인 필요.
    """
    tilt_rad = np.deg2rad(tilt_deg)
    R = np.array([
        [1,             0,              0],
        [0, np.cos(tilt_rad), -np.sin(tilt_rad)],
        [0, np.sin(tilt_rad),  np.cos(tilt_rad)]
    ], dtype=np.float64)
    return R


def doppler_correction(doppler, radar_velocity=None, target_vector=None):
    """
    필요하다면 Doppler를 보정하는 함수 예시.
    radar_velocity: 레이더의 선속도 벡터 (월드 좌표계)
    target_vector:  타겟(손) 방향 단위 벡터 (월드 좌표계)
    --> 둘 다 None이면 보정 없이 doppler 값 그대로 리턴
    
    사용 예시:
      doppler_corr = doppler_correction(doppler, v_radar, unit_dir)
    """
    if radar_velocity is None or target_vector is None:
        # 현재는 보정 로직 없이 반환
        return doppler
    
    # line-of-sight 성분 추출
    v_los = np.dot(radar_velocity, target_vector)
    # Doppler 측정값에서 빼줌 (부호 방향은 상황에 따라 맞춰야 함)
    doppler_corrected = doppler - v_los
    return doppler_corrected


def geometric_correction(row, radar_tilt_deg=45.0):
    """
    row: pandas Series
         - ["q.w"], ["q.x"], ["q.y"], ["q.z"]
         - ["horizontal_angle"], ["vertical_angle"], ["range"], ["doppler"]
    radar_tilt_deg: float (default=45.0)
                    레이더가 아래로 기울어진 각도 (x축 기준)
    """
    # -- 1) 쿼터니언 데이터 추출 -----------------------------------
    # !! 주의: 여기서 "q.j"와 "q.k" 등을 어떻게 매핑했는지 꼭 확인하세요.
    # 질문에 따르면 qz, qy 순서를 바꿔서 사용한다 했는데, 다시 한 번 점검 필요
    qw = row["q.w"]
    qx = row["q.i"]  # csv 상에서 'q.i' 가 x라 가정
    # 사용자가 "q.j" -> qz, "q.k" -> qy 로 바꾼다고 했음
    # 즉, 여기서 j->z, k->y로 사용 중 (주의!)
    qy = row["q.j"]
    qz = row["q.k"]
    
    # 만약 실제 IMU quaterion 순서가 w, x, y, z 인데,
    # CSV에서는 순서가 w, i, j, k 라고 표기만 다르다면:
    #   qw = row["q.w"], qx = row["q.i"], qy = row["q.j"], qz = row["q.k"]
    # 가 되어야 할 수도 있음. -> 현재 코드와 일치하는지 꼭 체크!!

    # -- 2) 레이더 데이터(수평/수직 각도, range, doppler) 추출 ---------
    h_angle = row["horizontal_angle"]
    v_angle = row["vertical_angle"]
    rng = row["range"]
    doppler = row["doppler"]

    # -- 3) 각도 -> 라디안 변환 -----------------------------------
    h_rad = np.deg2rad(h_angle)
    v_rad = np.deg2rad(v_angle)

    # -- 4) 극좌표 -> 직교좌표 변환 (레이다 기준)
    #     vertical_angle: (보통 0°가 수평, +각도가 위쪽)
    #     horizontal_angle: (0° 기준이 전방, +각도가 좌우 중 하나?)
    x_local = rng * np.cos(v_rad) * np.cos(h_rad)
    y_local = rng * np.cos(v_rad) * np.sin(h_rad)
    z_local = rng * np.sin(v_rad)

    # -- 5) 레이더 기울기 보정: 레이더가 x축을 기준으로 아래로 45°라면
    R_tilt_corr = tilt_rotation_matrix(-radar_tilt_deg)  # -45° -> 위로 회전
    xyz_tilt_corrected = R_tilt_corr.dot(np.array([x_local, y_local, z_local]))

    # -- 6) 쿼터니언 역회전(센서 -> 월드 좌표)
    q = (qw, qx, qy, qz)
    inv_q = get_inverse_quaternion(q)
    xyz_world = rotate_vector_by_quaternion(xyz_tilt_corrected, inv_q)

    # -- 7) 손-센서 반경차 보정(예시): sensor ~ 10cm, hand ~ 50cm
    #scale_factor = 1.0
    #xyz_scaled = scale_factor * xyz_world
    xyz_scaled = xyz_world
    # -- 8) 최종적으로 극좌표로 다시 변환
    x2, y2, z2 = xyz_scaled
    range_corr = np.sqrt(x2**2 + y2**2 + z2**2 + 1e-12)
    h_angle_corr = np.rad2deg(np.arctan2(y2, x2))
    # arcsin(z/r) : v_angle (수직각, z축 방향)
    v_angle_corr = np.rad2deg(
        np.arcsin(np.clip(z2 / range_corr, -1.0, 1.0))
    )

    # -- 9) Doppler 보정 (원하면 활용) ------------------------------
    # 현재는 보정하지 않고 그대로 사용
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
    # 경로는 사용자 환경에 맞게 변경
    source_root = "/home/dokyeong/miniconda3/envs/hgr_sci/processing_data_set/MergedData_all_BLPF_10Hz_v2"
    target_root = "/media/dokyeong/MINJI/dokyeong/processedData_geometry_with_head_v14"

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

                # 컬럼 검증(옵션)
                required_columns = ["q.w", "q.i", "q.j", "q.k",
                                    "horizontal_angle", "vertical_angle",
                                    "range", "doppler"]
                for c in required_columns:
                    if c not in df.columns:
                        print(f"[WARN] '{c}' column not found in {src_file}")
                
                # 각 행에 대해 보정 수행
                # apply에 전달 시, 추가 인자를 넣으려면 args=() 사용 가능
                correction_results = df.apply(
                    lambda row: geometric_correction(row, radar_tilt_deg=45.0),
                    axis=1
                )

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
