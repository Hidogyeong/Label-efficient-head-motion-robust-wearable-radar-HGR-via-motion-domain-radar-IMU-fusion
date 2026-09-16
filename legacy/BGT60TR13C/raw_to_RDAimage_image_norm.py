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


def save_processed_data(range_data, doppler_data, horizontal_angles, vertical_angles, base_output_path):
    """
    base_output_path: 원본 CSV 파일 경로를 기반으로 한 저장 경로 (확장자 제거됨)
    각 파일명에 _horizontal, _vertical, _range, _doppler 접미사를 붙여 PNG로 저장합니다.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # 저장할 폴더 생성 (없으면)
    output_dir = os.path.dirname(base_output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_name = os.path.basename(base_output_path)
    
    # 맨 첫 프레임 제외: 슬라이싱을 통해 각 데이터에서 첫 프레임(인덱스 0) 제거
    horizontal_data = horizontal_angles[2:]  # shape = (num_frames-2, num_angle_bins)
    vertical_data   = vertical_angles[2:]
    range_data_mod  = range_data[2:]
    doppler_data_mod = doppler_data[2:]
    
    # --------------------------------------------------
    # 수평 각도 스펙트로그램 (horizontal_data: shape = (num_frames-2, 27))
    num_frames, num_angle_bins = horizontal_data.shape
    t = np.arange(num_frames)      # 프레임 인덱스 (맨 첫 프레임 제외)
    f = np.arange(num_angle_bins)   # 각도 bin 인덱스
    data = horizontal_data.T        # shape = (num_angle_bins, num_frames)
    # 0이 아닌 값들만 선택
    nonzero_data = data[data > 0]
    if nonzero_data.size > 0:
        vmin_val = np.min(nonzero_data)
        vmax_val = np.max(nonzero_data)
    else:
        vmin_val, vmax_val = 1e-12, 1  # 기본값 설정

    plt.figure()
    plt.pcolormesh(t, f, 10 * np.log10(data + 1e-12), cmap='jet', 
                   vmin=10 * np.log10(vmin_val + 1e-12),
                   vmax=10 * np.log10(vmax_val + 1e-12))
    plt.axis('off')
    plt.subplots_adjust(0, 0, 1, 1)
    horizontal_file = os.path.join(output_dir, base_name + '_horizontal.png')
    plt.savefig(horizontal_file, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # --------------------------------------------------
    # 수직 각도 스펙트로그램 (vertical_data: shape = (num_frames-1, 27))
    num_frames, num_angle_bins = vertical_data.shape
    t = np.arange(num_frames)
    f = np.arange(num_angle_bins)
    data = vertical_data.T  # shape = (num_angle_bins, num_frames)
    nonzero_data = data[data > 0]
    if nonzero_data.size > 0:
        vmin_val = np.min(nonzero_data)
        vmax_val = np.max(nonzero_data)
    else:
        vmin_val, vmax_val = 1e-12, 1

    plt.figure()
    plt.pcolormesh(t, f, 10 * np.log10(data + 1e-12), cmap='jet', 
                   vmin=10 * np.log10(vmin_val + 1e-12),
                   vmax=10 * np.log10(vmax_val + 1e-12))
    plt.axis('off')
    plt.subplots_adjust(0, 0, 1, 1)
    vertical_file = os.path.join(output_dir, base_name + '_vertical.png')
    plt.savefig(vertical_file, bbox_inches='tight', pad_inches=0)
    plt.close() 

    # --------------------------------------------------
    # Range 스펙트로그램 (range_data_mod: shape = (num_frames-1, num_range_bins))
    num_frames, num_range_bins = range_data_mod.shape
    t = np.arange(num_frames)
    f = np.arange(num_range_bins)
    data = range_data_mod.T  # shape = (num_range_bins, num_frames)
    nonzero_data = data[data > 0]
    if nonzero_data.size > 0:
        vmin_val = np.min(nonzero_data)
        vmax_val = np.max(nonzero_data)
    else:
        vmin_val, vmax_val = 1e-12, 1

    plt.figure()
    plt.pcolormesh(t, f, 10 * np.log10(data + 1e-12), cmap='jet',
                   vmin=10 * np.log10(vmin_val + 1e-12),
                   vmax=10 * np.log10(vmax_val + 1e-12))
    plt.axis('off')
    plt.subplots_adjust(0, 0, 1, 1)
    range_file = os.path.join(output_dir, base_name + '_range.png')
    plt.savefig(range_file, bbox_inches='tight', pad_inches=0)
    plt.close()

    # --------------------------------------------------
    # Doppler 스펙트로그램 (doppler_data_mod: shape = (num_frames-1, num_doppler_bins))
    num_frames, num_doppler_bins = doppler_data_mod.shape
    t = np.arange(num_frames)
    f = np.arange(num_doppler_bins)
    data = doppler_data_mod.T  # shape = (num_doppler_bins, num_frames)
    nonzero_data = data[data > 0]
    if nonzero_data.size > 0:
        vmin_val = np.min(nonzero_data)
        vmax_val = np.max(nonzero_data)
    else:
        vmin_val, vmax_val = 1e-12, 1

    plt.figure()
    plt.pcolormesh(t, f, 10 * np.log10(data + 1e-12), cmap='jet',
                   vmin=10 * np.log10(vmin_val + 1e-12),
                   vmax=10 * np.log10(vmax_val + 1e-12))
    plt.axis('off')
    plt.subplots_adjust(0, 0, 1, 1)
    doppler_file = os.path.join(output_dir, base_name + '_doppler.png')
    plt.savefig(doppler_file, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 최종적으로 저장된 파일 경로 출력
    print(f"Saved:\n  {horizontal_file}\n  {vertical_file}\n  {range_file}\n  {doppler_file}")



# ============================ 
# Main Function
# ============================ 
if __name__ == "__main__":
    # 입력 데이터 폴더 및 저장할 이미지 출력 폴더 지정
    radar_data_directory = "/media/dokyeong/MINJI/dokyeong/Radar_rawdata/"
    image_output_folder = "/media/dokyeong/MINJI/dokyeong/Radar_RADimages_image_norm/"

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
        save_processed_data(range_data, doppler_data, horizontal_angles, vertical_angles, base_output_path)
        
        # (필요 시) plt.show()를 이용해 이미지 출력할 수 있으나, 여기서는 저장만 합니다.
