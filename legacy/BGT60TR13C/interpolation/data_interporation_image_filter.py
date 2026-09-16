#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# 상수 및 기본 변수
C = 3e8  # 빛의 속도 (m/s)
CENTER_FREQ_HZ = 60.75e9  # 예: 60.75GHz
LAMBDA = C / CENTER_FREQ_HZ  # 파장 (m)
LEVER_ARM = np.array([0.1, 0.0, 0.1])  # 예: 센서 기준, x축 5cm 전방 (z축도 5cm)

#########################################
# Doppler 보정 관련 함수들
#########################################
def get_sensor_velocity_from_gyro(row):
    # IMU 1프레임: row의 3개 요소 (각속도: rad/s)
    # row는 이미 [w_x, w_z, w_y] 형태로 제공된다고 가정
    row = np.deg2rad(row)
    # print(row)
    w_x, w_y, w_z = row
    omega = np.array([w_x, w_y, w_z])
    v_sensor = np.cross(omega, LEVER_ARM)
    # print(omega)
    # print(v_sensor)
    return v_sensor

def get_unit_vector_from_angles(h_deg, v_deg):
    h_rad = np.deg2rad(h_deg)
    v_rad = np.deg2rad(v_deg)
    x = np.cos(v_rad) * np.cos(h_rad)
    y = np.cos(v_rad) * np.sin(h_rad)
    z = np.sin(v_rad)
    r_hat = np.array([x, y, z])
    

    return r_hat / np.linalg.norm(r_hat)

def apply_doppler_correction(c_doppler, c_h_angle, c_v_angle, gyro_data):
    # 이 함수는 단일 프레임에 대해 보정을 진행함.
    # print(c_doppler)
    # print(c_h_angle)
    # print(c_v_angle)
    # print(gyro_data)

    corrected_doppler = np.zeros_like(c_doppler)
    sensor_vel = get_sensor_velocity_from_gyro(gyro_data)
    
    
    r_hat = get_unit_vector_from_angles(c_h_angle, c_v_angle)
    # delta_doppler = np.dot(sensor_vel, r_hat) / LAMBDA  # (Hz)
    delta_doppler = np.dot(sensor_vel, r_hat) # m/s
    # Δv = Δf * (λ/2)
    # delta_velocity = delta_doppler * (LAMBDA / 2)
    corrected_doppler = c_doppler - delta_doppler
    # print(corrected_doppler)
    # print(c_doppler)

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
            except Exception as e:
                radar_data_dict[suf] = None
        imu_filename = "I" + prefix + ".csv"
        imu_filepath = os.path.join(imu_dir, rel_dir, imu_filename)
        imu_data = None
        if os.path.exists(imu_filepath):
            try:
                imu_data = np.loadtxt(imu_filepath, delimiter=",", skiprows=1)
            except Exception as e:
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
        # 앞쪽에서부터 target_frame_count 개의 프레임을 선택 (뒤쪽 데이터는 버림)
        return imu_data[2:target_frame_count+2]
    elif num_imu_frames < target_frame_count:
        raise ValueError("IMU 데이터 프레임 수가 레이더 데이터 프레임 수보다 적습니다.")
    else:
        return imu_data


#########################################
# 3) 쿼터니언/회전 관련 유틸 함수들
#########################################
def quat_to_rotation_matrix(q):
    w, x, y, z = q
    # print(q)
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
    # print(v)
    # print(v.shape)
    R = quat_to_rotation_matrix(q)
    # print(R)
    # print(R.shape)

    # print(R.dot(v))
    # print(R.dot(v).shape) 
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
SCALE_FACTOR = 1.0
R_tilt_corr = tilt_rotation_matrix(RADAR_TILT_DEG)

def wrap_angle(angle_deg):
    
    wrapped = (angle_deg+180)%360 - 180
    wrapped = np.clip(wrapped,-90,90)

    return(wrapped)

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
    # 주의: 여기서 q와 g의 값을 덮어쓰지 않도록 주의해야 합니다.
    w, x, y, z = q
    # 만약 자이로 데이터와 쿼터니언 데이터를 구분해서 사용해야 한다면, 변수명을 분리하세요.
    # 예: gx, gz, gy = g
    norm_q = np.sqrt(w*w + x*x + y*y + z*z)
    if norm_q < 1e-8:
        return rng, doppler, h_angle_deg, v_angle_deg
    
    h_rad = np.deg2rad(h_angle_deg)
    # print(h_angle_deg)
    # print(h_rad)
    v_rad = np.deg2rad(v_angle_deg)
    # print(v_angle_deg)
    # print(v_rad)    

    x_ = rng * np.cos(v_rad) * np.cos(h_rad)
    y_ = rng * np.cos(v_rad) * np.sin(h_rad)
    z_ = rng * np.sin(v_rad)
    # print(x_)
    # print(x_.shape)
    # print(y_)
    # print(y_.shape)
    # print(z_)
    # print(z_.shape)
    # 머리 → 어깨 간 고정된 위치 차이 (m)
    
    xyz = np.array([x_, y_, z_])
    # print(xyz.shape)
    # print(xyz)
    xyz_tilt_corrected = R_tilt_corr.dot([x_, y_, z_])
    # print(xyz_tilt_corrected.shape)
    # print(xyz_tilt_corrected)
    
    xyz_tilt_corrected[2,:] += 0.3 # 레이더가 몸통에 있다고 생각
    # xyz_tilt_corrected[2,:] += 0.3 # 레이더가 몸통에 있다고 생각 
    # print(xyz_tilt_corrected)
    # print(xyz_tilt_corrected.shape)
    # print(xyz_tilt_corrected)
    inv_q = (w/norm_q, -x/norm_q, -y/norm_q, -z/norm_q)

    xyz_world = rotate_vector_by_quaternion(xyz_tilt_corrected, inv_q)
    

    xyz_scaled = SCALE_FACTOR * xyz_world
    x2, y2, z2 = xyz_scaled


    new_range = correct_range(rng, x2, y2, z2)
    # print(rng)
    # print(new_range)

    new_h_angle = correct_horizontal_angle(x2, y2)
    # print(h_angle_deg)
    # print(new_h_angle)
    new_h_angle = wrap_angle(new_h_angle)

    new_v_angle = correct_vertical_angle(z2, new_range)
    # print(v_angle_deg)
    # print(new_v_angle)
    new_v_angle = wrap_angle(new_v_angle)

    new_doppler = apply_doppler_correction(doppler, new_h_angle, new_v_angle, g)

    # print(new_doppler)
    # print(doppler)

    return new_range, new_doppler, new_h_angle, new_v_angle

def apply_geometric_correction(range_data, doppler_data, horizontal_angles, vertical_angles, quaternions, gyro):
    """
    모든 입력 배열의 shape = (num_frames, 64) (실제 물리량 단위)
    quaternions: (num_frames, 4)
    반환: (corrected_range, corrected_doppler, corrected_h_angle, corrected_v_angle)
    """
    num_frames, num_bins = range_data.shape
    # 출력 배열을 (num_frames, num_bins)로 초기화


    c_range = np.zeros((num_frames, num_bins))
    c_dopp  = np.zeros((num_frames, num_bins))
    c_h     = np.zeros((num_frames, num_bins)) 
    c_v     = np.zeros((num_frames, num_bins))
    
    # c_range_b_all = np.zeros((num_frames,7))
    # c_dopp_b_all = np.zeros((num_frames,7))
    # c_h_b_all = np.zeros((num_frames,7))
    # c_v_b_all = np.zeros((num_frames,7))

    # range_b_all = np.zeros((num_frames,7))
    # dopp_b_all = np.zeros((num_frames,7))
    # h_b_all = np.zeros((num_frames,7))
    # v_b_all = np.zeros((num_frames,7))

    for f in range(num_frames):
    # for f in range(1):
        q = quaternions[f,:]
        # g = gyro[f,:] * 0.07 # degree/s
        g = gyro[f,:]
        # print(q)
        # print(q.shape)
        
        # print(g)
        # print(gyro[f,:])
        # print(g.shape)

        # Range
        rng = range_data[f, :]
        # print(rng.shape)

        sorted_indices_rng = np.argsort(rng)[::-1]
        top7_indices_rng = sorted_indices_rng[:7]
        real_rng = top7_indices_rng * 0.0273 #cm
        # print(rng)
        # print(sorted_indices_rng)
        # print(top7_indices_rng)
        # print(real_rng)

        # Doppler
        dop = doppler_data[f]
        sorted_indices_dop = np.argsort(dop)[::-1]
        top7_indices_dop = sorted_indices_dop[:7]
        real_dop = (top7_indices_dop - 31.5) * 0.2 # m/s
        # print(dop)
        # print(sorted_indices_dop)
        # print(top7_indices_dop)
        # print(real_dop)


        # H_Angle
        h_angle = horizontal_angles[f]
        sorted_indices_h_angle = np.argsort(h_angle)[::-1]
        top7_indices_h_angle = sorted_indices_h_angle[:7]
        # real_h_angle = (top7_indices_h_angle - 31.5) * 1.25
        real_h_angle = (31.5 - top7_indices_h_angle ) * 1.25

        # print(h_angle)
        # print(sorted_indices_h_angle)
        # print(top7_indices_h_angle)
        # print(real_h_angle)

        # V_Angle
        v_angle = vertical_angles[f]
        sorted_indices_v_angle = np.argsort(v_angle)[::-1]
        top7_indices_v_angle = sorted_indices_v_angle[:7]
        # real_v_angle = (top7_indices_v_angle - 31.5) * 1.25
        real_v_angle = (31.5 - top7_indices_v_angle) * 1.25
        # print(v_angle)
        # print(sorted_indices_v_angle)
        # print(top7_indices_v_angle)
        # print(real_v_angle)


        nr, nd, nh, nv = correct_single_bin(q, g, real_h_angle, real_v_angle, real_rng, real_dop)

        # print(nr)
        # print(nr.shape)

        # print(nd)
        # print(nd.shape)
        
        # print(nh)
        # print(nh.shape)

        # print(nv)
        # print(nv.shape)

        # Real value to bin
        # c_range_b_index = np.clip(np.round(nr / 0.0273, 0).astype(int))
        # c_dopp_b_index  = np.clip(np.round(((nd / 0.2) + 31.5), 0).astype(int))
        # c_h_b_index     = np.clip(np.round(((nh / 2.81) + 31.5), 0).astype(int))
        # c_v_b_index     = np.clip(np.round(((nv / 2.81) + 31.5), 0).astype(int))

        # c_range_b = (c_range_b_index >= 0) & (c_range_b_index <= 63)
        # c_dopp_b = (c_dopp_b_index >= 0) & (c_dopp_b_index <= 63)
        # c_h_b = (c_h_b_index >= 0) & (c_h_b_index <= 63)
        # c_v_b = (c_v_b_index >= 0) & (c_v_b_index <= 63)


        c_range_b = np.clip(np.round(nr / 0.0273, 0).astype(int),-1,64)
        c_dopp_b  = np.clip(np.round(((nd / 0.2) + 31.5), 0).astype(int),-1,64)
        c_h_b     = np.clip(np.round(((nh / 2.81) + 31.5), 0).astype(int),-1,64)
        c_v_b     = np.clip(np.round(((nv / 2.81) + 31.5), 0).astype(int),-1,64)


        # c_range_b_all[f, :] = np.clip(np.round(nr / 0.0273, 0).astype(int),0,63)
        # c_dopp_b_all[f, :]  = np.clip(np.round(((nd / 0.2) + 31.5), 0).astype(int),0,63)
        # c_h_b_all[f, :]     = np.clip(np.round(((nh / 2.81) + 31.5), 0).astype(int),0,63)
        # c_v_b_all[f, :]     = np.clip(np.round(((nv / 2.81) + 31.5), 0).astype(int),0,63)

        # range_b_all[f, :] = top7_indices_rng
        # dopp_b_all[f, :]  = top7_indices_dop
        # h_b_all[f, :]     = top7_indices_h_angle
        # v_b_all[f, :]     = top7_indices_v_angle


        # print(c_range_b)
        # print(nr)
        # print(top7_indices_rng)
        # print(real_rng)

        # print(c_dopp_b)
        # print(nd)
        # print(top7_indices_dop)
        # print(real_dop)

        # print(c_h_b)
        # print(nh)
        # print(top7_indices_h_angle)
        # print(real_h_angle)

        # print(c_v_b)
        # print(nv)
        # print(top7_indices_v_angle)
        # print(real_v_angle)

        for i in range(7):
            # 거리
            new_range_bin = c_range_b[i]        # 보정 후 bin 인덱스
            old_range_bin = top7_indices_rng[i]  # 원본 bin 인덱스
            if 0<=new_range_bin<=63:
                c_range[f, new_range_bin] = c_range[f, new_range_bin] + range_data[f, old_range_bin]
            else:
                pass

            # 도플러
            new_dopp_bin = c_dopp_b[i]         # 보정 후 bin 인덱스
            old_dopp_bin = top7_indices_dop[i]  # 원본 bin 인덱스
            if 0<=new_dopp_bin<=63:
                c_dopp[f, new_dopp_bin] = c_dopp[f, new_dopp_bin] + doppler_data[f, old_dopp_bin]
            else:
                pass


            # 수평각
            new_h_bin = c_h_b[i]         # 보정 후 bin 인덱스
            old_h_bin = top7_indices_h_angle[i]  # 원본 bin 인덱스
            if 0<=new_h_bin<=63:
                c_h[f, new_h_bin] = c_h[f, new_h_bin] + horizontal_angles[f, old_h_bin]
            else:
                pass

            # 수직각
            new_v_bin = c_v_b[i]         # 보정 후 bin 인덱스
            old_v_bin = top7_indices_v_angle[i]  # 원본 bin 인덱스
            if 0<=new_v_bin<=63:
                c_v[f, new_v_bin] = c_v[f, new_v_bin] + vertical_angles[f, old_v_bin]
            else:
                pass

            
    # np.set_printoptions(threshold=np.inf)
    # print(c_range)
    # print(c_range.shape)
    # print(c_dopp)
    # print(c_dopp.shape)
    # print(c_h)
    # print(c_h.shape)
    # print(c_v)
    # print(c_v.shape)

    return c_range, c_dopp, c_h, c_v

#########################################
# 6) 보정 결과를 이미지로 저장하기
#########################################
def save_2d_array_as_image(array_2d, save_path):
    """
    2D 배열(array_2d)을 로그 스케일로 변환한 후,
    컬러맵('jet')을 사용해 이미지로 저장하는 함수.
    - array_2d: shape = (num_frames, num_angle_bins)
    - 저장 시, x축: 프레임 인덱스, y축: 각도 bin 인덱스 (데이터는 전치하여 사용)
    """
    # array_2d shape: (num_frames, num_angle_bins)
    num_frames, num_angle_bins = array_2d.shape
    
    # x축: 프레임 인덱스, y축: 각도 bin 인덱스
    t = np.arange(num_frames)
    f = np.arange(num_angle_bins)
    
    # 전치하여 (num_angle_bins, num_frames) 형태로 만듦
    data = array_2d.T
    
    # 0이 아닌 값들만 선택하여 최소, 최대값 계산 (log10 변환을 위해)
    nonzero_data = data[data > 0]
    if nonzero_data.size > 0:
        vmin_val = np.min(nonzero_data)
        vmax_val = np.max(nonzero_data)
    else:
        vmin_val, vmax_val = 1e-12, 1  # 기본값 설정
    
    # 그림 생성 및 pcolormesh로 플롯
    plt.figure()
    plt.pcolormesh(t, f, 10 * np.log10(data + 1e-12), cmap='jet',
                   vmin=10 * np.log10(vmin_val + 1e-12),
                   vmax=10 * np.log10(vmax_val + 1e-12))
    plt.axis('off')
    plt.subplots_adjust(0, 0, 1, 1)
    
    # 저장
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

#########################################
# 10) 메인 파트
#########################################
def main():
    radar_dir = "/media/dokyeong/MINJI/dokyeong/Radar_RADimages_image_norm_csv/"
    imu_dir = "/media/dokyeong/MINJI/dokyeong/IMUData/"
    output_dir = "/media/dokyeong/MINJI/dokyeong/processed_2D_image_norm_filter_test/"

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
        imu_full = imu_data  # 전체 IMU 데이터 사용

        doppler_arr = radar_dict.get('_doppler.csv', None)
        range_arr = radar_dict.get('_range.csv', None)
        h_arr = radar_dict.get('_horizontal.csv', None)
        v_arr = radar_dict.get('_vertical.csv', None)

        if (doppler_arr is None or range_arr is None or h_arr is None or v_arr is None):
            print(f"[WARN] Some radar array is missing for {rel_dir}/{prefix}")
            continue

        # CSV 모양: (bin x frame) → 전치하여 (frame x bin) 형태로 변환
        num_bins, num_frames = range_arr.shape  # 예상: (64, num_frames)
        range_arr = range_arr.T      # (frame, 64)
        range_arr = range_arr[2:,:]

        doppler_arr = doppler_arr.T  # (frame, 64)
        doppler_arr = doppler_arr[2:,:]

        h_arr = h_arr.T              # (frame, 64)
        h_arr = h_arr[2:,:]

        v_arr = v_arr.T              # (frame, 64)
        v_arr = v_arr[2:,:]

        num_frames = range_arr.shape[0]
        
        # print(range_arr.shape)
        # print(doppler_arr.shape)
        # print(h_arr.shape)
        # print(v_arr.shape)
        # print(num_frames)

        # IMU 프레임 수 맞추기
        quaternions = imu_data[:, 7:11]  # (num_frames, 4)
        gyro_data = imu_data[:, 1:4]
        # print(quaternions)
        # print(gyro_data.shape)
        quaternions = trim_imu_data(quaternions, num_frames)
        gyro_data = trim_imu_data(gyro_data, num_frames)
        # print(quaternions.shape)
        # print(gyro_data.shape)
        assert quaternions.shape[0] == num_frames, "IMU 프레임 수와 레이더 프레임 수 불일치!"

        

        # 기하학적 보정 적용
        c_range, c_dop, c_h, c_v = apply_geometric_correction(range_arr, doppler_arr, h_arr, v_arr, quaternions, gyro_data)



        # (D) 결과 이미지 저장: 전체 파일(모든 프레임)을 하나의 이미지로 저장
        out_subdir = os.path.join(output_dir, rel_dir)
        os.makedirs(out_subdir, exist_ok=True)
        
        # 각 측정값에 대해 전체 프레임의 2D 배열을 이미지로 저장 (행: 프레임, 열: bin)
        save_path_range = os.path.join(out_subdir, f"{prefix}_all_frames_range.png")
        save_2d_array_as_image(c_range, save_path_range)
        
        save_path_doppler = os.path.join(out_subdir, f"{prefix}_all_frames_doppler.png")
        save_2d_array_as_image(c_dop, save_path_doppler)
        
        save_path_horizontal = os.path.join(out_subdir, f"{prefix}_all_frames_horizontal.png")
        save_2d_array_as_image(c_h, save_path_horizontal)
        
        save_path_vertical = os.path.join(out_subdir, f"{prefix}_all_frames_vertical.png")
        save_2d_array_as_image(c_v, save_path_vertical)

        print(f"[INFO] Saved images -> {out_subdir}/{prefix}_all_frames_XXX.png")

    print("=== ALL DONE ===")

if __name__ == "__main__":
    main()
