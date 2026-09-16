import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import glob


from helpers.DopplerAlgo import *
from helpers.DigitalBeamForming import *

from collections import deque

# -----------------------------
# DopplerAlgo, DigitalBeamForming
# (외부 helpers에서 임포트)
# from helpers.DopplerAlgo import DopplerAlgo
# from helpers.DigitalBeamForming import DigitalBeamForming
# -----------------------------

class EKFRangeDoppAngle8D:
    """
    상태: x = [ r, d, v_a, h_a, vr, vd, vv_a, vh_a ]
    측정: z = [ r_meas, d_meas, v_a_meas, h_a_meas ]
    (등속도(등속) 모델 가정)
    """
    def __init__(self, dt=0.340, process_q=1.0, meas_r=0.1):
        self.dt = dt
        
        # 초기 상태, 공분산
        self.x = np.zeros(8)
        self.P = np.eye(8)*1.0
        
        # 상태전이행렬 F
        #   r   d   va  ha  vr  vd  vva vha
        # r' = r + vr*dt
        # d' = d + vd*dt
        # va'= va + vva*dt
        # ha'= ha + vha*dt
        # vr'= vr
        # vd'= vd
        # vva'=vva
        # vha'=vha
        self.F = np.array([
            [1,0,0,0, self.dt,0,       0,      0],
            [0,1,0,0, 0,      self.dt, 0,      0],
            [0,0,1,0, 0,      0,       self.dt,0],
            [0,0,0,1, 0,      0,       0,      self.dt],
            [0,0,0,0, 1,      0,       0,      0],
            [0,0,0,0, 0,      1,       0,      0],
            [0,0,0,0, 0,      0,       1,      0],
            [0,0,0,0, 0,      0,       0,      1],
        ], dtype=float)
        
        # 프로세스 잡음 Q
        self.Q = np.diag([0.01,0.01,0.01,0.01, process_q,process_q,process_q,process_q])
        
        # 측정잡음 R (4x4) => (r, d, v_a, h_a)
        self.R = np.diag([meas_r, meas_r, meas_r, meas_r])
        
        # 측정행렬 H (z=[r, d, v_a, h_a])
        #  z = H x => [x0, x1, x2, x3]
        self.H = np.array([
            [1,0,0,0, 0,0,0,0], # r
            [0,1,0,0, 0,0,0,0], # d
            [0,0,1,0, 0,0,0,0], # v_a
            [0,0,0,1, 0,0,0,0], # h_a
        ], dtype=float)
    
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, z):
        """
        z.shape=(4,) => [r_meas, d_meas, v_a_meas, h_a_meas]
        """
        y = z - (self.H @ self.x)     # residual (4,)
        S = self.H @ self.P @ self.H.T + self.R  # (4x4)
        K = self.P @ self.H.T @ np.linalg.inv(S) # (8x4)
        self.x = self.x + K @ y
        I = np.eye(8)
        self.P = (I - K@self.H) @ self.P
    
    def get_state(self):
        """
        반환: [r, d, v_a, h_a, vr, vd, vv_a, vh_a]
        """
        return self.x


def cfar_2d_ca(matrix_2d, guard=(2,2), train=(4,4), threshold_factor=1.5):
    """
    간단한 2D CA-CFAR (Cell Averaging) 예시
    - matrix_2d: shape=(R, C), 파워 맵 (range, doppler 등)
    - guard, train: (gR, gC), (tR, tC) 각각 가드셀/트레이닝 셀 크기
    - threshold_factor: 잡음 평균 대비 임계값 스케일
    반환: detection_map (0/1) shape=(R, C)
    """
    R, C = matrix_2d.shape
    detection_map = np.zeros((R, C), dtype=int)
    gR, gC = guard
    tR, tC = train
    
    for r in range(tR+gR, R - (tR+gR)):
        for c in range(tC+gC, C - (tC+gC)):
            r_min = r - (tR + gR)
            r_max = r + (tR + gR)
            c_min = c - (tC + gC)
            c_max = c + (tC + gC)
            
            guard_r_min = r - gR
            guard_r_max = r + gR
            guard_c_min = c - gC
            guard_c_max = c + gC
            
            train_cells = []
            for rr in range(r_min, r_max+1):
                for cc in range(c_min, c_max+1):
                    if rr<0 or rr>=R or cc<0 or cc>=C:
                        continue
                    # Guard+test 제외
                    if (guard_r_min <= rr <= guard_r_max) and (guard_c_min <= cc <= guard_c_max):
                        continue
                    if rr==r and cc==c:
                        continue
                    train_cells.append(matrix_2d[rr, cc])
            
            if len(train_cells) > 0:
                noise_est = np.mean(train_cells)
            else:
                noise_est = 0.0
            
            threshold = noise_est * threshold_factor
            if matrix_2d[r, c] > threshold:
                detection_map[r, c] = 1
    
    return detection_map


def cluster_2d(detect_map):
    """
    2D 검출맵(0/1)에서 연결요소(8방향) 찾아 클러스터 리스트 반환
    각 클러스터는 [(r,c), (r2,c2), ...] 형태
    """
    R, C = detect_map.shape
    visited = np.zeros((R, C), dtype=bool)
    clusters = []
    directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    
    for r in range(R):
        for c in range(C):
            if detect_map[r,c]==1 and not visited[r,c]:
                q = deque()
                q.append((r,c))
                visited[r,c] = True
                cluster_points = []
                
                while q:
                    rr, cc = q.popleft()
                    cluster_points.append((rr, cc))
                    
                    for dr, dc in directions:
                        nr, nc = rr+dr, cc+dc
                        if 0<=nr<R and 0<=nc<C:
                            if detect_map[nr,nc]==1 and not visited[nr,nc]:
                                visited[nr,nc] = True
                                q.append((nr,nc))
                clusters.append(cluster_points)
    
    return clusters


def get_centroid_2d(cluster_points, matrix_2d):
    """
    클러스터 내 (r, c) 각각의 파워(matrix_2d[r,c])를 가중치로 무게중심 계산
    반환: (r_center, c_center, total_power)
    """
    if len(cluster_points)==0:
        return (None,None,0.0)
    wsum_r, wsum_c, total = 0.0, 0.0, 0.0
    for (rr, cc) in cluster_points:
        p = matrix_2d[rr, cc]
        wsum_r += rr * p
        wsum_c += cc * p
        total  += p
    if total<1e-12:
        return (None,None,0.0)
    return (wsum_r/total, wsum_c/total, total)


def load_csv_files(directory):
    """
    Load all CSV files from the specified directory.
    """
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    data = []
    for file in csv_files:
        loaded_data = np.loadtxt(file, delimiter=',')
        data.append(loaded_data)
    return np.array(data)


def process_radar_data_with_cfar_clustering_ekf(data, 
                                                num_chirps=32, 
                                                num_rx_antennas=3, 
                                                num_samples=64, 
                                                num_beams=80, 
                                                max_angle_degrees=40):
    """
    CFAR + Clustering + EKF 예시 (Range–Doppler, Range–Beam 둘 다 CFAR)
    """
    # data shape => (num_frames, num_rx_antennas, num_chirps, num_samples)
    data = data.reshape(-1, num_rx_antennas, num_chirps, num_samples)
    
    doppler = DopplerAlgo(num_samples, num_chirps, num_rx_antennas)
    dbf = DigitalBeamForming(num_rx_antennas, num_beams=num_beams, max_angle_degrees=max_angle_degrees)
    
    ekf = EKFRangeDoppAngle8D(dt=0.34, process_q=0.5, meas_r=0.1)
    
    range_data = []
    doppler_data = []
    horizontal_angles = []
    vertical_angles = []
    ekf_states = []
    cluster_info_rd = []  # range–doppler 클러스터 정보 (각 프레임)
    cluster_info_angle_h = []  # range–beam(H) 클러스터
    cluster_info_angle_v = []  # range–beam(V) 클러스터
    
    for frame_idx, frame in enumerate(data):
        rd_spectrum_h = np.zeros((num_samples, 2*num_chirps, num_rx_antennas), dtype=complex)
        rd_spectrum_v = np.zeros((num_samples, 2*num_chirps, num_rx_antennas), dtype=complex)
        data_all_antennas = []

        # 1) 도플러 맵 계산 (각 안테나)
        for i_ant in range(num_rx_antennas):
            mat = frame[i_ant, :, :]
            dfft_dbfs_complex = doppler.compute_doppler_map(mat, i_ant)
            dfft_dbfs_abs = np.abs(dfft_dbfs_complex)
            data_all_antennas.append(dfft_dbfs_abs)

            if i_ant == 0:
                rd_spectrum_v[:, :, 0] = dfft_dbfs_complex
            elif i_ant == 1:
                rd_spectrum_h[:, :, 0] = dfft_dbfs_complex
            elif i_ant == 2:
                rd_spectrum_v[:, :, 1] = dfft_dbfs_complex
                rd_spectrum_h[:, :, 1] = dfft_dbfs_complex

        # 1) 빔포밍 -> 수평/수직 angle (range, beam)
        rd_beam_formed_h = dbf.run(rd_spectrum_h)
        rd_beam_formed_v = dbf.run(rd_spectrum_v)

        # beam_map_h, beam_map_v = (range, beam)
        beam_map_h = np.abs(rd_beam_formed_h.mean(axis=-1)) if rd_beam_formed_h.ndim==3 else np.abs(rd_beam_formed_h)
        beam_map_v = np.abs(rd_beam_formed_v.mean(axis=-1)) if rd_beam_formed_v.ndim==3 else np.abs(rd_beam_formed_v)

        # CFAR => cluster => best cluster
        # [A] horizontal
        detect_beam_h = cfar_2d_ca(beam_map_h, guard=(2,2), train=(4,4), threshold_factor=1.5)
        clus_h = cluster_2d(detect_beam_h)
        best_pw_h = 0
        best_h    = (None,None)  # (r_c, beam_c)
        all_h = []
        for c in clus_h:
            r_c, b_c, pw = get_centroid_2d(c, beam_map_h)
            all_h.append((r_c, b_c, pw))
            if pw>best_pw_h:
                best_pw_h = pw
                best_h    = (r_c, b_c)
        cluster_info_angle_h.append(all_h)
        
        # range축 합산 => horizontal_angles
        sum_beam_h = np.sum(beam_map_h, axis=0)  # shape=(beam,)
        # top7
        top7_h = np.argsort(sum_beam_h)[-7:]
        filtered_h = np.zeros_like(sum_beam_h)
        filtered_h[top7_h] = sum_beam_h[top7_h]
        horizontal_angles.append(filtered_h)
        
        # [B] vertical
        detect_beam_v = cfar_2d_ca(beam_map_v, guard=(2,2), train=(4,4), threshold_factor=1.5)
        clus_v = cluster_2d(detect_beam_v)
        best_pw_v= 0
        best_v   = (None,None)
        all_v = []
        for c in clus_v:
            r_c, b_c, pw = get_centroid_2d(c, beam_map_v)
            all_v.append((r_c, b_c, pw))
            if pw>best_pw_v:
                best_pw_v= pw
                best_v   = (r_c, b_c)
        cluster_info_angle_v.append(all_v)
        
        sum_beam_v = np.sum(beam_map_v, axis=0)
        top7_v = np.argsort(sum_beam_v)[-7:]
        filtered_v = np.zeros_like(sum_beam_v)
        filtered_v[top7_v] = sum_beam_v[top7_v]
        vertical_angles.append(filtered_v)

        # 2) Range–Doppler
        # data_all_antennas => shape=(num_rx_antennas, range, doppler) => average => (range,doppler)
        if len(data_all_antennas)>0:
            averaged_data_2d = np.mean(data_all_antennas, axis=0)
            detect_map = cfar_2d_ca(averaged_data_2d, guard=(2,2), train=(4,4), threshold_factor=1.5)
            clus_rd = cluster_2d(detect_map)
            
            best_pw_rd= 0
            best_rd   = (None,None)
            all_rd    = []
            for c in clus_rd:
                r_c, d_c, pw = get_centroid_2d(c, averaged_data_2d)
                all_rd.append((r_c, d_c, pw))
                if pw>best_pw_rd:
                    best_pw_rd= pw
                    best_rd   = (r_c, d_c)
            cluster_info_rd.append(all_rd)
            
            # -> best_rd => (r,d)
            # -> best_h => (rh,bh)
            # -> best_v => (rv,bv)
            # 여기서 "r_meas, d_meas, va_meas, ha_meas"를 결정
            # 단순히 best_rd[0], best_rd[1], best_v[0], best_h[1]를 결합해도 되지만,
            #  'r'값이 어긋날 수 있음. 실제론 'r'가 유사한 클러스터끼리 매칭해야 함.
            
            # 임시 간단 로직: r_meas = best_rd[0], d_meas = best_rd[1], v_a_meas= best_v[1], h_a_meas= best_h[1]
            r_meas = best_rd[0] if best_rd[0] else 0
            d_meas = best_rd[1] if best_rd[1] else 0
            v_a_meas = best_v[1] if best_v[1] else 0
            h_a_meas = best_h[1] if best_h[1] else 0
            z_meas = np.array([r_meas, d_meas, v_a_meas, h_a_meas], dtype=float)
            
            # 8D EKF update
            ekf.predict()
            ekf.update(z_meas)
            x_est = ekf.get_state()   # [r, d, v_a, h_a, vr, vd, vv_a, vh_a]

            # range/doppler 1D 스펙트럼
            sum_dopp = np.sum(averaged_data_2d, axis=0)
            top7_v_idx = np.argsort(sum_dopp)[-7:]
            filtered_summed_velocity_all = np.zeros_like(sum_dopp)
            filtered_summed_velocity_all[top7_v_idx] = sum_dopp[top7_v_idx]

            sum_range = np.sum(averaged_data_2d, axis=1)
            top7_r_idx = np.argsort(sum_range)[-7:]
            filtered_summed_distance_all = np.zeros_like(sum_range)
            filtered_summed_distance_all[top7_r_idx] = sum_range[top7_r_idx]
        else:
            ekf.predict()
            x_est = ekf.get_state()
            filtered_summed_velocity_all = np.zeros(2*num_chirps, dtype=float)
            filtered_summed_distance_all = np.zeros(num_samples, dtype=float)
            cluster_info_rd.append([])
        
        doppler_data.append(filtered_summed_velocity_all)
        range_data.append(filtered_summed_distance_all)
        ekf_states.append(x_est)
    
    # np.array 변환
    range_data = np.array(range_data, dtype=float)
    doppler_data = np.array(doppler_data, dtype=float)
    horizontal_angles = np.array(horizontal_angles, dtype=float)
    vertical_angles = np.array(vertical_angles, dtype=float)
    ekf_states = np.array(ekf_states, dtype=float)
    
    return (
        range_data,
        doppler_data,
        horizontal_angles,
        vertical_angles,
        ekf_states,
        cluster_info_rd,
        cluster_info_angle_h,
        cluster_info_angle_v
    )


def save_processed_data(range_data, doppler_data, 
                        horizontal_angles, vertical_angles, 
                        base_output_path,
                        ekf_states=None,
                        cluster_info=None):
    """
    - 기존 스펙트로그램 이미지 저장 로직 유지
    - 추가로 EKF 상태, 클러스터 정보 등을 CSV/텍스트로 저장
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # ----------------------
    # 0) 폴더 준비
    # ----------------------
    output_dir = os.path.dirname(base_output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    base_name = os.path.basename(base_output_path)

    # ----------------------
    # 1) 수평각 / 수직각 / 거리 / 도플러 => 스펙트로그램 이미지
    #    (기존 코드: 1D (frame, bin) 형식 가정)
    # ----------------------
    horizontal_data = horizontal_angles[2:] if horizontal_angles.shape[0]>2 else horizontal_angles
    vertical_data   = vertical_angles[2:]   if vertical_angles.shape[0]>2 else vertical_angles
    range_data_mod  = range_data[2:]        if range_data.shape[0]>2 else range_data
    doppler_data_mod= doppler_data[2:]      if doppler_data.shape[0]>2 else doppler_data

    # CSV 파일로 저장
    horizontal_csv_file = os.path.join(output_dir, base_name + '_horizontal.csv')
    np.savetxt(horizontal_csv_file, horizontal_data.T, delimiter=",", fmt="%.6f")
    
    vertical_csv_file = os.path.join(output_dir, base_name + '_vertical.csv')
    np.savetxt(vertical_csv_file, vertical_data.T, delimiter=",", fmt="%.6f")
    
    range_csv_file = os.path.join(output_dir, base_name + '_range.csv')
    np.savetxt(range_csv_file, range_data_mod.T, delimiter=",", fmt="%.6f")
    
    doppler_csv_file = os.path.join(output_dir, base_name + '_doppler.csv')
    np.savetxt(doppler_csv_file, doppler_data_mod.T, delimiter=",", fmt="%.6f")
    
    print(f"Saved CSV files:\n  {horizontal_csv_file}\n  {vertical_csv_file}\n  {range_csv_file}\n  {doppler_csv_file}")


    # ----------------------
    # 2) EKF 상태 ekf_states 저장 (CSV)
    # ----------------------
    if ekf_states is not None:
        # ekf_states.shape = (num_frames, state_dim)
        ekf_csv_file = os.path.join(output_dir, base_name + '_ekf_states.csv')
        header_str = ','.join([f'x{i}' for i in range(ekf_states.shape[1])])
        np.savetxt(ekf_csv_file, ekf_states, delimiter=',', header=header_str, comments='')
        print(f"[INFO] EKF states saved -> {ekf_csv_file}")

    # ----------------------
    # 3) 클러스터 정보 cluster_info 저장 (CSV)
    # ----------------------
    if cluster_info is not None:
        # cluster_info: list of length num_frames
        # each element: list of (r_center, d_center, power)
        cluster_csv_file = os.path.join(output_dir, base_name + '_clusters.csv')
        with open(cluster_csv_file, 'w') as f:
            f.write("frame,cluster_idx,range_center,doppler_center,power\n")
            for frame_idx, clusters in enumerate(cluster_info):
                for c_idx, (r_c, d_c, pw) in enumerate(clusters):
                    f.write(f"{frame_idx},{c_idx},{r_c},{d_c},{pw}\n")
        print(f"[INFO] Cluster info saved -> {cluster_csv_file}")

    print(f"[INFO] Saved images at base -> {base_output_path}")


# ============================ 
# Main Function
# ============================ 
if __name__ == "__main__":
    radar_data_directory = "/media/dokyeong/MINJI/dokyeong/Radar_rawdata/"
    image_output_folder = "/media/dokyeong/MINJI/dokyeong/Radar_RADimages_image_norm_CFAR_csv/"

    radar_data_files = glob.glob(os.path.join(radar_data_directory, '**', '*.csv'), recursive=True)
    
    for file in radar_data_files:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            radar_data = np.genfromtxt(file, delimiter=',', invalid_raise=False)

        # (A) 레이더 데이터 처리 -> CFAR+클러스터+EKF 결과까지
        (range_data, doppler_data,
        horizontal_angles, vertical_angles,
        ekf_states, cluster_info,
        cluster_info_angle_h, cluster_info_angle_v
        ) = process_radar_data_with_cfar_clustering_ekf(radar_data)

        # (B) 저장 경로
        relative_path = os.path.relpath(file, radar_data_directory)
        base_output_path = os.path.join(image_output_folder, os.path.splitext(relative_path)[0])
        output_dir = os.path.dirname(base_output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # (C) 확장된 save 함수
        save_processed_data(
            range_data, doppler_data, 
            horizontal_angles, vertical_angles, 
            base_output_path,
            ekf_states=ekf_states,
            cluster_info=cluster_info
        )