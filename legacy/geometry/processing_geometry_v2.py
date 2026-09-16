import os
import pandas as pd
import numpy as np

# 쿼터니언을 회전 행렬로 변환
def quat_to_rotation_matrix(q):
    w, x, y, z = q
    R = np.array([
        [1 - 2*y**2 - 2*z**2,     2*x*y - 2*w*z,         2*x*z + 2*w*y],
        [2*x*y + 2*w*z,           1 - 2*x**2 - 2*z**2,   2*y*z - 2*w*x],
        [2*x*z - 2*w*y,           2*y*z + 2*w*x,         1 - 2*x**2 - 2*y**2]
    ], dtype=np.float64)
    return R

# 벡터를 쿼터니언으로 회전
def rotate_vector_by_quaternion(v, q):
    R = quat_to_rotation_matrix(q)
    return R.dot(v)

# 보정 함수
def geometric_correction(row):
    qw, qx, qz, qy = row["q.w"], row["q.i"], row["q.j"], row["q.k"]
    h_angle, v_angle = row["horizontal_angle"], row["vertical_angle"]
    rng, doppler = row["range"], row["doppler"]

    # (1) 각도를 라디안으로 변환
    h_rad = np.deg2rad(h_angle)
    v_rad = np.deg2rad(v_angle)

    # (2) 극좌표 -> 직교좌표 변환
    x = rng * np.cos(v_rad) * np.cos(h_rad)
    y = rng * np.cos(v_rad) * np.sin(h_rad)
    z = rng * np.sin(v_rad)

    # (3) 쿼터니언 역회전 적용
    inv_q = (qw, -qx, -qy, -qz)
    xyz_world = rotate_vector_by_quaternion(np.array([x, y, z]), inv_q)

    # (4) 변환 후 다시 극좌표계로 변환
    x2, y2, z2 = xyz_world
    range_corr = np.sqrt(x2**2 + y2**2 + z2**2 + 1e-8)
    
    # ⚠ **단위 오류 방지**: `np.arctan2()`와 `np.arcsin()` 적용 방식 점검
    h_angle_corr = np.rad2deg(np.arctan2(y2, x2))  # 수평 각도
    v_angle_corr = np.rad2deg(np.arcsin(np.clip(z2 / (range_corr + 1e-8), -1, 1)))  # 수직 각도 (값 제한)

    # Doppler 속도 보정 추가 가능 (현재는 원본 유지)
    doppler_corr = doppler

    # **디버깅 로그**
    if np.abs(h_angle_corr) > 180 or np.abs(v_angle_corr) > 90:
        print(f"Warning: Unexpected angle values - h_angle_corr: {h_angle_corr}, v_angle_corr: {v_angle_corr}")

    return {
        "range_corr": range_corr,
        "doppler_corr": doppler_corr,
        "horizontal_angle_corr": h_angle_corr,
        "vertical_angle_corr": v_angle_corr
    }

# =====================================================================
# 파일 처리 메인 함수
# =====================================================================
def main():
    source_root = "/home/dokyeong/miniconda3/envs/hgr_sci/processing_data_set/MergedData_all_BLPF_10Hz_v2"
    target_root = "/media/dokyeong/MINJI/dokyeong/processedData_geometry_v10"

    for dirpath, _, filenames in os.walk(source_root):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                src_file = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(dirpath, source_root)
                dst_dir = os.path.join(target_root, rel_path)
                os.makedirs(dst_dir, exist_ok=True)
                dst_file = os.path.join(dst_dir, filename)

                # (1) CSV 읽기
                df = pd.read_csv(src_file)

                # (2) 보정 적용
                correction_results = df.apply(geometric_correction, axis=1)
                corr_df = pd.DataFrame(list(correction_results))

                # (3) 원본 df에 보정 데이터 추가
                df["range_corr"] = corr_df["range_corr"]
                df["doppler_corr"] = corr_df["doppler_corr"]
                df["horizontal_angle_corr"] = corr_df["horizontal_angle_corr"]
                df["vertical_angle_corr"] = corr_df["vertical_angle_corr"]

                # (4) CSV 저장
                df.to_csv(dst_file, index=False, encoding="utf-8-sig")
                print(f"[SAVE] {dst_file}")

if __name__ == "__main__":
    main()
