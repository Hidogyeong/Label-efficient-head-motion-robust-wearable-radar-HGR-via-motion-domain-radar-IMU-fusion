#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import math
import numpy as np
from itertools import zip_longest
from scipy.signal import butter, filtfilt

# 미분(차분)을 적용할 컬럼 목록 (쿼터니언은 제외)
DERIV_COLS = ["Gyr.X", "Gyr.Y", "Gyr.Z", "Axl.X", "Axl.Y", "Axl.Z"]

#############################################
# Butterworth 저역통과 IIR 필터 함수 (Butterworth IIR Filtering)
#############################################
def apply_butterworth_filter(values, order=2, cutoff=5.0, fs=100):
    """
    Butterworth 저역통과 IIR 필터를 적용합니다.
    
    Parameters:
      - values: 필터링할 1차원 데이터 (numpy array 또는 리스트)
      - order: 필터 차수 (기본값: 2)
      - cutoff: 컷오프 주파수 (Hz) (기본값: 5Hz)
      - fs: 샘플링 주파수 (Hz) (기본값: 100)
    
    Returns:
      - 필터링된 데이터 (numpy array)
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, values)
    return filtered

#############################################
# CSV 파일 병합 및 미분, 필터링 함수
#############################################
def merge_files(imu_file_path: str, radar_file_path: str, output_file_path: str) -> None:
    """
    1) IMU 파일과 Radar 파일(둘 다 CSV 파일)을 행 단위로 병합 (행 길이 다르면 부족한 부분은 빈 리스트로 채움)
    2) IMU 파일에서 지정된 컬럼(DERIV_COLS)에 대해 미분(차분)을 계산 (첫 행은 0)
    3) 계산된 미분값에 Butterworth IIR 필터를 적용 (샘플링: 100Hz, 차수: 2, 컷오프: 10Hz)
    4) "range", "doppler", "horizontal_angle", "vertical_angle", "q.w", "q.i", "q.j", "q.k" 컬럼에는 미분 없이 원본 데이터를 Butterworth 필터 적용
    5) 결과 CSV 파일에 저장 (헤더는 그대로, 해당 컬럼에 필터링된 값 기록)
    """
    try:
        # 두 파일을 모두 읽어 리스트로 저장
        with open(imu_file_path, newline='', encoding='utf-8') as f_imu:
            reader_imu = csv.reader(f_imu)
            imu_rows = list(reader_imu)
        with open(radar_file_path, newline='', encoding='utf-8') as f_radar:
            reader_radar = csv.reader(f_radar)
            radar_rows = list(reader_radar)
        
        # 병합: 각 행을 좌우로 결합
        merged_rows = []
        for row_imu, row_radar in zip_longest(imu_rows, radar_rows, fillvalue=[]):
            merged_rows.append(row_imu + row_radar)
        
        if not merged_rows:
            print("병합된 데이터가 없습니다.")
            return

        # 첫 행은 헤더로 사용
        header = merged_rows[0]
        data_rows = merged_rows[1:]
        
        # 미분을 적용할 열의 인덱스 (DERIV_COLS)
        deriv_indices = []
        for col in DERIV_COLS:
            if col in header:
                deriv_indices.append(header.index(col))
        
        # 1) 미분 계산: 각 행에 대해 (현재 값 - 이전 값)
        prev_values = {}
        diff_results = {idx: [] for idx in deriv_indices}
        
        for i, row in enumerate(data_rows):
            for idx in deriv_indices:
                try:
                    current_val = float(row[idx])
                except ValueError:
                    current_val = 0.0
                if i == 0:
                    diff_val = 0.0
                else:
                    diff_val = current_val - prev_values.get(idx, current_val)
                prev_values[idx] = current_val
                diff_results[idx].append(diff_val)
        
        # 필터 파라미터
        filter_order = 2
        cutoff_freq = 10.0  # Hz
        sampling_rate = 100  # Hz
        
        # 2) DERIV_COLS에 대해 Butterworth IIR 필터 적용 (미분값에 적용)
        for idx in deriv_indices:
            values_array = np.array(diff_results[idx])
            filtered_vals = apply_butterworth_filter(values_array, order=filter_order, cutoff=cutoff_freq, fs=sampling_rate)
            diff_results[idx] = filtered_vals
        
        # 3) 원본 데이터에 대해, FILTER_ONLY_COLS (미분 없이 필터 적용)
        FILTER_ONLY_COLS = ["range", "doppler", "horizontal_angle", "vertical_angle", "q.w", "q.i", "q.j", "q.k"]
        filter_indices = []
        for col in FILTER_ONLY_COLS:
            if col in header:
                filter_indices.append(header.index(col))
        
        filter_results = {}
        for idx in filter_indices:
            raw_vals = []
            for row in data_rows:
                try:
                    raw_val = float(row[idx])
                except ValueError:
                    raw_val = 0.0
                raw_vals.append(raw_val)
            raw_vals_array = np.array(raw_vals)
            filtered_raw_vals = apply_butterworth_filter(raw_vals_array, order=filter_order, cutoff=cutoff_freq, fs=sampling_rate)
            filter_results[idx] = filtered_raw_vals
        
        # 4) 필터 결과를 data_rows에 반영 (소수점 4자리)
        # DERIV_COLS (미분 적용된 값)
        for i, row in enumerate(data_rows):
            for idx in deriv_indices:
                row[idx] = f"{diff_results[idx][i]:.4f}"
        # FILTER_ONLY_COLS (원본에 필터 적용)
        for i, row in enumerate(data_rows):
            for idx in filter_indices:
                row[idx] = f"{filter_results[idx][i]:.4f}"
        
        # 5) CSV 파일로 저장
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(header)
            writer.writerows(data_rows)
            
        print(f"Merged, diff, and Butterworth IIR filtering done:\n  IMU: {imu_file_path}\n  Radar: {radar_file_path}\n  -> {output_file_path}")
    except Exception as e:
        print(f"병합 오류: {output_file_path} / {e}")

#############################################
# 전체 폴더 순회
#############################################
def main():
    # 기본 경로 설정 (예시)
    base_imu = "/media/dokyeong/MINJI/dokyeong/IMUData/"
    base_radar = "/media/dokyeong/MINJI/dokyeong/Radar_RDAData/"
    base_output = "/home/dokyeong/miniconda3/envs/hgr_sci/processing_data_set/MergedData_all_BLPF_10Hz_v2"

    # IMUData 폴더 트리를 순회
    for root, dirs, files in os.walk(base_imu):
        for file in files:
            # IMU 파일은 "I"로 시작한다고 가정
            if not file.startswith("I"):
                continue
            # Radar 파일 이름: IMU 파일 이름에서 첫 글자 "I" 제거
            time_filename = file[1:]
            imu_file_path = os.path.join(root, file)
            rel_path = os.path.relpath(root, base_imu)
            radar_file_path = os.path.join(base_radar, rel_path, time_filename)
            if os.path.exists(radar_file_path):
                output_dir = os.path.join(base_output, rel_path)
                os.makedirs(output_dir, exist_ok=True)
                output_file_path = os.path.join(output_dir, "M_10Hz_" + time_filename)
                merge_files(imu_file_path, radar_file_path, output_file_path)
            else:
                print(f"Radar 파일을 찾을 수 없습니다: {radar_file_path}")

if __name__ == "__main__":
    main()
