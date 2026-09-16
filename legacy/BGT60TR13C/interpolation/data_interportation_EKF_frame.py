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

    ekf_states = np.zeros((num_frames,8))
    dt = 0.34
    ekf = EKFRangeDoppAngle8D(dt=dt,
                            q_pos=0.01,   # 위치 측
                            q_vel=1.0,    # 속도 측
                            r_meas=0.1)

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
        real_h_angle = (top7_indices_h_angle - 31.5) * 1.25


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


        ekf.predict()
        
        for i in range(7):
            z = [ nr[i], nd[i], nh[i], nv[i] ]
            ekf.update(z)

        x_est = ekf.get_state()  # shape=(8,)
        ekf_states[f, :] = np.array(x_est)


    return ekf_states


#########################################
# 7) 보정 결과를 이미지로 저장하기
#########################################
class EKFRangeDoppAngle8D:
    """
    (1) 상태벡터 x (8차원):
        x[0] = r      (m)       - 거리
        x[1] = d      (m/s)     - 도플러(목표의 상대속도)
        x[2] = v_a    (deg)     - 수직각
        x[3] = h_a    (deg)     - 수평각
        x[4] = vr     (m/s)     - r의 변화율(거리속도)           (등속도 모델이라 '가속=0')
        x[5] = vd     (m/s^2)   - d의 변화율(도플러속도)         (등속도)
        x[6] = vv_a   (deg/s)   - v_a의 변화율(각속도)
        x[7] = vh_a   (deg/s)   - h_a의 변화율(각속도)

    (2) 측정 z (4차원):
        z = [r_meas, d_meas, v_a_meas, h_a_meas]
        => 모두 '물리값': m, m/s, deg, deg

    (3) 등속도(CV) 모델 with dt (초):
       r_{k+1}   = r_k   + vr_k  * dt
       d_{k+1}   = d_k   + vd_k  * dt
       v_a_{k+1} = v_a_k + vv_a_k* dt
       h_a_{k+1} = h_a_k + vh_a_k* dt

       vr_{k+1}  = vr_k
       vd_{k+1}  = vd_k
       vv_a_{k+1}= vv_a_k
       vh_a_{k+1}= vh_a_k

    (4) 프로세스 잡음 Q, 측정잡음 R => 튜닝 필요
        - Q: (8x8), 각 요소별 잡음 (예: 위치 vs 속도)
        - R: (4x4), 측정의 (r, d, v_a, h_a) 노이즈
    """

    def __init__(self, dt=0.34,
                 q_pos=0.01,   # 위치 잡음 크기
                 q_vel=1.0,    # 속도(변화율) 잡음 크기
                 r_meas=0.1,   # 측정잡음(단순히 동일값으로)
                 init_x=None,
                 init_P=None):
        """
        dt:    프레임 간격(초)
        q_pos: (r, d, v_a, h_a)에 대한 프로세스 잡음 크기
        q_vel: (vr, vd, vv_a, vh_a)에 대한 프로세스 잡음 크기
        r_meas: 측정잡음(4개 항목 일괄) -> R=diag(r_meas,...)
        init_x: 초기 상태 (shape=(8,)) [r, d, v_a, h_a, vr, vd, vv_a, vh_a]
        init_P: 초기 공분산 (8x8)
        """
        self.dt = dt

        # 초기 상태, 공분산
        if init_x is not None:
            self.x = init_x.copy()
        else:
            self.x = np.zeros(8, dtype=float)  # [r=0, d=0, v_a=0, h_a=0, vr=0, ...]
        
        if init_P is not None:
            self.P = init_P.copy()
        else:
            self.P = np.eye(8)*10.0

        # 상태전이행렬 F(8x8): 등속도
        #  x[0]=>x[0]+x[4]*dt, x[1]=>x[1]+x[5]*dt, x[2]=>x[2]+x[6]*dt, ...
        self.F = np.array([
            [1,0,0,0, self.dt,0,     0,     0],
            [0,1,0,0, 0,     self.dt,0,     0],
            [0,0,1,0, 0,     0,     self.dt,0],
            [0,0,0,1, 0,     0,     0,     self.dt],
            [0,0,0,0, 1,     0,     0,     0],
            [0,0,0,0, 0,     1,     0,     0],
            [0,0,0,0, 0,     0,     1,     0],
            [0,0,0,0, 0,     0,     0,     1],
        ], dtype=float)

        # 프로세스 잡음 공분산 Q(8x8)
        #  앞4개=위치(r,d,v_a,h_a)에 q_pos, 뒤4개=속도에 q_vel
        self.Q = np.diag([
            q_pos, q_pos, q_pos, q_pos,
            q_vel, q_vel, q_vel, q_vel
        ])

        # 측정잡음 R(4x4) => (r, d, v_a, h_a)
        self.R = np.diag([r_meas, r_meas, r_meas, r_meas])

        # 측정행렬 H(4x8): z=[x0, x1, x2, x3] => (r, d, v_a, h_a)
        #  z=H*x => ( x[0], x[1], x[2], x[3] )
        self.H = np.array([
            [1,0,0,0, 0,0,0,0],
            [0,1,0,0, 0,0,0,0],
            [0,0,1,0, 0,0,0,0],
            [0,0,0,1, 0,0,0,0],
        ], dtype=float)

    def predict(self):
        """예측 단계"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """
        z: shape=(4,) => [r_meas(m), d_meas(m/s), v_a_meas(deg), h_a_meas(deg)]
        """
        z = np.array(z, dtype=float).flatten()
        # 잔차
        y = z - (self.H @ self.x)  # (4,)
        # 잔차 공분산
        S = self.H @ self.P @ self.H.T + self.R # (4x4)
        # 칼만 이득
        K = self.P @ self.H.T @ np.linalg.inv(S) # (8x4)
        
        self.x = self.x + K @ y
        I = np.eye(8)
        self.P = (I - K@self.H) @ self.P

    def get_state(self):
        """
        반환 (8,):
          [r(m), d(m/s), v_a(deg), h_a(deg),
           vr(m/s), vd(m/s^2), vv_a(deg/s), vh_a(deg/s)]
        """
        return self.x.copy()

    def get_cov(self):
        """추정 공분산 P(8x8) 반환(디버그용)"""
        return self.P.copy()


def save_ekf_states_as_csv(ekf_states, save_path):
    """
    ekf_states: shape=(num_frames, 8) 
      - 예) [r, d, v_a, h_a, vr, vd, vv_a, vh_a] (물리단위)
    output_dir: CSV 저장할 디렉토리
    corr: (str) 예: "corr"
    prefix: (str) 예: "mydata"

    최종 저장 파일명은 => f"{corr}_{prefix}_ekf_states.csv"
    """

    # 3) 헤더 문자열 (각 열에 대한 의미)
    header_str = "r(m),d(m/s),v_a(deg),h_a(deg),vr(m/s),vd(m/s^2),vv_a(deg/s),vh_a(deg/s)"

    # 4) CSV로 저장
    np.savetxt(save_path, ekf_states, delimiter=",", header=header_str,
               comments="", fmt="%.6f")

    # print(f"[INFO] EKF states saved -> {save_path}")
#########################################
# 10) 메인 파트
#########################################
def main():
    radar_dir = "/media/dokyeong/MINJI/dokyeong/Radar_RADimages_image_norm_CFAR_csv/"
    imu_dir = "/media/dokyeong/MINJI/dokyeong/IMUData/"
    output_dir = "/media/dokyeong/MINJI/dokyeong/processed_2D_image_norm_CCFAR/"

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
        ekf_states = apply_geometric_correction(range_arr, doppler_arr, h_arr, v_arr, quaternions, gyro_data)



        # (D) 결과 이미지 저장: 전체 파일(모든 프레임)을 하나의 이미지로 저장
        out_subdir = os.path.join(output_dir, rel_dir)
        os.makedirs(out_subdir, exist_ok=True)
        corr = "corr"
        csv_path = os.path.join(out_subdir, f"{corr}_{prefix}_ekf.csv")
        save_ekf_states_as_csv(ekf_states, csv_path)
        print(f"[INFO] Saved EKF CSV -> {csv_path}")

    print("=== ALL DONE ===")

if __name__ == "__main__":
    main()
