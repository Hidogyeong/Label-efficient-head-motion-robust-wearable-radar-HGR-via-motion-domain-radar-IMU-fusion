import os
import glob
import pandas as pd
import numpy as np

# 모든 .txt 파일을 특정 디렉터리에서 불러오기
txt_file_path = "E:/dokyeong/IMU_rawdata/txt/"
csv_output_directory = "E:/dokyeong/IMU_rawdata/csv/"
final_output_directory = "E:/dokyeong/IMUData/"
radar_data_directory = "E:/dokyeong/Radar_RDAData/"

# Step 1: Convert .txt files to .csv files
txt_files = glob.glob(os.path.join(txt_file_path, '*.txt'))
for txt_file in txt_files:
    with open(txt_file, "r") as file:
        lines = file.readlines()

    # "DATA"라는 단어 이후부터 데이터 부분이 시작되므로, "DATA" 이후의 줄을 찾습니다.
    data_start_index = next(i for i, line in enumerate(lines) if line.strip() == "DATA") + 3
    data_lines = lines[data_start_index:]  # 데이터 부분
    
    # 데이터프레임 생성
    columns = ["Timestamp [ms][dev]", "Gyr.X", "Gyr.Y", "Gyr.Z", "Axl.X", "Axl.Y", "Axl.Z", "q.w", "q.i", "q.j", "q.k"]
    txt_data = []

    for line in data_lines:
        parts = line.strip().split("\t")
        if len(parts) == len(columns):
            txt_data.append(parts)

    df_txt = pd.DataFrame(txt_data, columns=columns)

    # Timestamp를 문자열로 변환
    df_txt["Timestamp [ms][dev]"] = df_txt["Timestamp [ms][dev]"].astype(str)

    # CSV 파일로 저장 (원본 파일 이름 유지)
    output_file_name = os.path.basename(txt_file).replace(".txt", ".csv")
    output_file_path = os.path.join(csv_output_directory, output_file_name)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    df_txt.to_csv(output_file_path, index=False, quotechar='"', header=True)
    print(f"CSV 파일 '{output_file_path}'이(가) 생성되었습니다.")

# Step 2: Read converted .csv files and interpolate data to match radar timestamps
csv_files = glob.glob(os.path.join(radar_data_directory, '**', '*.csv'), recursive=True)
txt_files = glob.glob(os.path.join(csv_output_directory, '*.csv'))

# 텍스트 파일에서 데이터 읽기 및 레이더 시간 범위에 맞춰 필터링
for txt_file in txt_files:
    # 데이터프레임 생성
    df_txt = pd.read_csv(txt_file)

    # Timestamp를 숫자로 변환
    df_txt["Timestamp [ms][dev]"] = pd.to_numeric(df_txt["Timestamp [ms][dev]"], errors='coerce').dropna().astype(float)

    # 각 .csv 파일의 제목에서 시간 정보를 추출하고, 해당 시간에 해당하는 데이터를 필터링 후 저장
    for csv_file in csv_files:
        # 파일 이름에서 시작 시간과 종료 시간 추출
        start_time, end_time = os.path.basename(csv_file).replace(".csv", "").split("_")
        start_time = float(start_time)
        end_time = float(end_time)

        # 해당 시간 범위에 있는 데이터 필터링
        filtered_data = df_txt[(df_txt["Timestamp [ms][dev]"] >= start_time) & (df_txt["Timestamp [ms][dev]"] <= end_time)]

        if not filtered_data.empty:
            # 레이더 데이터 시간에 맞게 IMU 데이터 보간
            radar_timestamps = np.linspace(start_time, end_time, 40)  # 레이더 데이터 수집 시간에 맞춰 40개의 시간 생성
            interpolated_data = {}

            for column in filtered_data.columns:
                if column != "Timestamp [ms][dev]":
                    interpolated_data[column] = np.interp(radar_timestamps, filtered_data["Timestamp [ms][dev]"], filtered_data[column])

            interpolated_data["Timestamp [ms][dev]"] = radar_timestamps
            df_interpolated = pd.DataFrame(interpolated_data)
            # Timestamp 열을 첫 번째로 이동
            columns = ['Timestamp [ms][dev]'] + [col for col in df_interpolated.columns if col != 'Timestamp [ms][dev]']
            df_interpolated = df_interpolated[columns]

            # 새로운 .csv 파일로 저장 (원본 폴더 구조 유지)
            relative_path = os.path.relpath(csv_file, radar_data_directory)
            output_file_name = 'I' + os.path.basename(relative_path)
            output_file_path = os.path.join(final_output_directory, os.path.dirname(relative_path), output_file_name)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

            df_interpolated.to_csv(output_file_path, index=False)
            print(f"파일 '{output_file_path}'이(가) 생성되었습니다.")
        else:
            print(f"파일 '{csv_file}'에 해당하는 시간 범위에 데이터가 없습니다.")
