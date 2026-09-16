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

import numpy as np
from helpers.DopplerAlgo import DopplerAlgo
from helpers.DigitalBeamForming import DigitalBeamForming

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

    # ---------------------------------------------------------
    # Helper function: 상위 7개 값을 10..4로 매핑하는 로직
    # ---------------------------------------------------------
    def map_top7_to_10_4(array_in):
        """
        array_in에서 상위 7개 인덱스를 찾아,
        가장 큰 값 = 10, 그 다음 = 9, ... 7번째 = 4로 매핑한 결과 배열을 반환.
        나머지는 0으로 둔다.
        """
        mapped_array = np.zeros_like(array_in)
        # 값이 모두 0이거나, 길이가 7 미만일 수도 있으므로 예외 처리
        if array_in.size < 7:
            return mapped_array
        
        # 1) 전체 중 상위 7개 인덱스 찾기
        top7_indices = np.argsort(array_in)[-7:]  # 오름차순 정렬 후 뒤에서 7개
        # 2) 그 7개 중에서도 값이 큰 순(내림차순)으로 정렬
        #    => array_in[top7_indices] 중에서 큰 순
        top7_values = array_in[top7_indices]
        desc_order = np.argsort(-top7_values)  # 음수 붙여서 내림차순
        
        # 3) 내림차순 순서대로 10, 9, 8, 7, 6, 5, 4 할당
        #    i=0 => 10, i=1 => 9, ..., i=6 => 4
        for i, rank in enumerate(desc_order):
            actual_idx = top7_indices[rank]
            mapped_array[actual_idx] = 10 - i  # 0 -> 10, 1 -> 9, ..., 6 -> 4
        
        return mapped_array

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
        
        # 상위 7개를 10..4로 매핑 (수평/수직 각도)
        mapped_beam_range_energy_h = map_top7_to_10_4(beam_range_energy_h)
        mapped_beam_range_energy_v = map_top7_to_10_4(beam_range_energy_v)

        # 저장할 리스트에 각 프레임마다 계산된 결과 추가
        horizontal_angles.append(mapped_beam_range_energy_h)
        vertical_angles.append(mapped_beam_range_energy_v)

        # Range-Doppler 계산
        if data_all_antennas:
            averaged_data = np.mean(data_all_antennas, axis=0)

            # Doppler (velocity) 처리
            summed_velocity_all = np.sum(averaged_data, axis=0)
            mapped_velocity = map_top7_to_10_4(summed_velocity_all)

            # Range (distance) 처리
            summed_distance_all = np.sum(averaged_data, axis=1)
            mapped_distance = map_top7_to_10_4(summed_distance_all)
        else:
            mapped_velocity = None
            mapped_distance = None

        # 저장: 도플러는 velocity, 레인지는 distance 사용
        doppler_data.append(mapped_velocity)
        range_data.append(mapped_distance)

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
    image_output_folder = "/media/dokyeong/MINJI/dokyeong/Radar_RADimages_fram_norm_csv/"

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
