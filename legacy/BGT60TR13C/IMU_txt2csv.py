import os
import glob
import pandas as pd

# 모든 .txt 파일을 특정 디렉터리에서 불러오기
txt_file_path = "E:/dokyeong/IMU_rawdata/txt/"
output_directory = "E:/dokyeong/IMU_rawdata/csv/"


txt_files = glob.glob(os.path.join(txt_file_path, '*.txt'))

# 텍스트 파일에서 데이터 읽기 및 CSV로 변환
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

    # Timestamp를 숫자로 변환
    df_txt["Timestamp [ms][dev]"] = df_txt["Timestamp [ms][dev]"].astype(str)

    # CSV 파일로 저장 (원본 파일 이름 유지)
    output_file_name = os.path.basename(txt_file).replace(".txt", ".csv")
    output_file_path = os.path.join(output_directory, output_file_name)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    df_txt.to_csv(output_file_path, index=False, quotechar='"', header=True)
    print(f"CSV 파일 '{output_file_path}'이(가) 생성되었습니다.")
