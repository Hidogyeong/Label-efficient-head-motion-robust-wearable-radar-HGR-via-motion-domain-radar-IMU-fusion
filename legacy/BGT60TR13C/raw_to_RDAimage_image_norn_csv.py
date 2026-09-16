import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import glob


from helpers.DopplerAlgo import *
from helpers.DigitalBeamForming import *

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

def process_radar_data(data, num_chirps=32, num_rx_antennas=3, num_samples=64, num_beams=64, max_angle_degrees=40):
    """
    Process radar data to generate range, doppler, and angle information.
    """
    # Reshape data to (num_frames, 3, 32, 64)
    data = data.reshape(-1, num_rx_antennas, num_chirps, num_samples)
    
    # Initialize Doppler and BeamForming algorithms
    doppler = DopplerAlgo(num_samples, num_chirps, num_rx_antennas)
    dbf = DigitalBeamForming(num_rx_antennas, num_beams=num_beams, max_angle_degrees=max_angle_degrees)

    range_data = []
    doppler_data = []
    horizontal_angles = []
    vertical_angles = []

    # Iterate through each frame
    for frame in data:
        rd_spectrum_h = np.zeros((num_samples, 2 * num_chirps, num_rx_antennas), dtype=complex)
        rd_spectrum_v = np.zeros((num_samples, 2 * num_chirps, num_rx_antennas), dtype=complex)
        data_all_antennas = []

        # Process each antenna's data in the current frame
        for i_ant in range(num_rx_antennas):
            mat = frame[i_ant, :, :]
            dfft_dbfs_complex = doppler.compute_doppler_map(mat, i_ant)
            dfft_dbfs = np.abs(dfft_dbfs_complex)
            data_all_antennas.append(dfft_dbfs)

            if i_ant == 0:
                rd_spectrum_v[:, :, 0] = dfft_dbfs_complex
            elif i_ant == 1:
                rd_spectrum_h[:, :, 0] = dfft_dbfs_complex
            elif i_ant == 2:
                rd_spectrum_v[:, :, 1] = dfft_dbfs_complex
                rd_spectrum_h[:, :, 1] = dfft_dbfs_complex

        # Angle calculations
        rd_beam_formed_h = dbf.run(rd_spectrum_h)
        rd_beam_formed_v = dbf.run(rd_spectrum_v)

        beam_range_energy_2h = np.linalg.norm(rd_beam_formed_h, axis=1) / np.sqrt(num_beams)
        beam_range_energy_2v = np.linalg.norm(rd_beam_formed_v, axis=1) / np.sqrt(num_beams)

        beam_range_energy_h = np.linalg.norm(beam_range_energy_2h, axis=0) / np.sqrt(num_beams)
        beam_range_energy_v = np.linalg.norm(beam_range_energy_2v, axis=0) / np.sqrt(num_beams)
        
        # Filtering: 상위 7개 에너지 값만 남기고 나머지는 0으로 처리 (각도 각각)
        top7_indices_h = np.argsort(beam_range_energy_h)[-7:]
        filtered_beam_range_energy_h = np.zeros_like(beam_range_energy_h)
        filtered_beam_range_energy_h[top7_indices_h] = beam_range_energy_h[top7_indices_h]

        top7_indices_v = np.argsort(beam_range_energy_v)[-7:]
        filtered_beam_range_energy_v = np.zeros_like(beam_range_energy_v)
        filtered_beam_range_energy_v[top7_indices_v] = beam_range_energy_v[top7_indices_v]

        # 저장할 리스트에 각 프레임마다 계산된 결과 추가
        horizontal_angles.append(filtered_beam_range_energy_h)
        vertical_angles.append(filtered_beam_range_energy_v)

        # Range-Doppler 계산
        if data_all_antennas:
            averaged_data = np.mean(data_all_antennas, axis=0)

            # Doppler (velocity) 처리
            summed_velocity_all = np.sum(averaged_data, axis=0)
            top7_indices_velocity = np.argsort(summed_velocity_all)[-7:]
            filtered_summed_velocity_all = np.zeros_like(summed_velocity_all)
            filtered_summed_velocity_all[top7_indices_velocity] = summed_velocity_all[top7_indices_velocity]

            # Range (distance) 처리
            summed_distance_all = np.sum(averaged_data, axis=1)
            top7_indices_distance = np.argsort(summed_distance_all)[-7:]
            filtered_summed_distance_all = np.zeros_like(summed_distance_all)
            filtered_summed_distance_all[top7_indices_distance] = summed_distance_all[top7_indices_distance]
        else:
            filtered_summed_velocity_all = None
            filtered_summed_distance_all = None

        # 저장: 도플러는 velocity, 레인지는 distance 사용
        doppler_data.append(filtered_summed_velocity_all)
        range_data.append(filtered_summed_distance_all)

    return (np.array(range_data), 
            np.array(doppler_data), 
            np.array(horizontal_angles), 
            np.array(vertical_angles))

def save_processed_data_as_csv(range_data, doppler_data, horizontal_angles, vertical_angles, base_output_path):
    """
    base_output_path: 원본 CSV 파일 경로를 기반으로 한 저장 경로 (확장자 제거됨)
    각 파일명에 _horizontal, _vertical, _range, _doppler 접미사를 붙여 CSV로 저장합니다.
    """
    import numpy as np
    import os

    # 저장할 폴더 생성 (없으면)
    output_dir = os.path.dirname(base_output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_name = os.path.basename(base_output_path)
    
    # 예를 들어, 첫 2프레임을 제외하고 저장하는 경우 (필요시 수정 가능)
    horizontal_data = horizontal_angles[2:]  # shape = (num_frames-2, num_angle_bins)
    vertical_data   = vertical_angles[2:]
    range_data_mod  = range_data[2:]
    doppler_data_mod = doppler_data[2:]
    
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


# ============================ 
# Main Function
# ============================ 
if __name__ == "__main__":
    # 입력 데이터 폴더 및 저장할 이미지 출력 폴더 지정
    radar_data_directory = "/media/dokyeong/MINJI/dokyeong/Radar_rawdata/"
    image_output_folder = "/media/dokyeong/MINJI/dokyeong/Radar_RADimages_image_norm_csv/"

    # 폴더 내 모든 CSV 파일을 재귀적으로 검색
    radar_data_files = glob.glob(os.path.join(radar_data_directory, '**', '*.csv'), recursive=True)
    
    for file in radar_data_files:
        # CSV 파일에서 데이터 로드
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            radar_data = np.genfromtxt(file, delimiter=',', invalid_raise=False)

        # radar_data = np.loadtxt(file, delimiter=',')
        # 레이더 데이터 처리 (range, doppler, 수평/수직 각도)
        range_data, doppler_data, horizontal_angles, vertical_angles = process_radar_data(radar_data)
        
        # 원본 파일의 폴더 구조를 유지하기 위해 상대 경로 계산
        relative_path = os.path.relpath(file, radar_data_directory)
        # 확장자 제거 후, image_output_folder 내에 저장할 파일 경로 생성
        base_output_path = os.path.join(image_output_folder, os.path.splitext(relative_path)[0])
        
        # 폴더가 없으면 생성
        output_dir = os.path.dirname(base_output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 처리된 스펙트로그램 이미지를 저장 (수평, 수직각, 거리, 도플러)
        save_processed_data_as_csv(range_data, doppler_data, horizontal_angles, vertical_angles, base_output_path)
        
        # (필요 시) plt.show()를 이용해 이미지 출력할 수 있으나, 여기서는 저장만 합니다.
