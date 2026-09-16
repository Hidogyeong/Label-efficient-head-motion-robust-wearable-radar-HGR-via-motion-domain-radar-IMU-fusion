import os
import glob
import pandas as pd

# 모든 .txt 파일을 특정 디렉터리에서 불러오기
txt_file_path = "E:/dokyeong/IMU_rawdata/txt/junh/01RtoL/"
csv_output_directory = "E:/dokyeong/IMU_rawdata/csv/junh/01RtoL/"

# Step 1: Convert .txt files to .csv files (Hdr 열 제거)
txt_files = glob.glob(os.path.join(txt_file_path, '**', '*.txt'), recursive=True)  # 하위 폴더 포함 검색
for txt_file in txt_files:
    with open(txt_file, "r") as file:
        lines = file.readlines()

    # "DATA"라는 단어 이후부터 데이터 부분이 시작되므로, "DATA" 이후의 줄 찾기 (예: 3줄 건너뛰기)
    try:
        data_start_index = next(i for i, line in enumerate(lines) if line.strip() == "DATA") + 3
    except StopIteration:
        print(f"'DATA'라는 단어를 찾지 못했습니다: {txt_file}")
        continue

    data_lines = lines[data_start_index:]  # 데이터 부분

    # 최종적으로 사용할 열 이름 (Hdr 열 제거)
    columns = ["Timestamp [ms][pc]", "Gyr.X", "Gyr.Y", "Gyr.Z", "Axl.X", "Axl.Y", "Axl.Z", "q.w", "q.i", "q.j", "q.k"]
    txt_data = []

    for line in data_lines:
        parts = line.strip().split("\t")
        # 원본 파일은 14개 항목이 있다고 가정 (0~13 인덱스)
        if len(parts) == 14:
            # 0~6번과 10~13번 인덱스 선택하여 Hdr 열 제거
            new_parts = parts[0:7] + parts[10:14]
            txt_data.append(new_parts)
        else:
            # 예상치 않은 열 개수인 경우 건너뜁니다.
            continue

    if not txt_data:
        print(f"데이터가 없는 파일: {txt_file}")
        continue

    df_txt = pd.DataFrame(txt_data, columns=columns)

    # Timestamp를 문자열로 변환 (필요 시)
    df_txt["Timestamp [ms][pc]"] = df_txt["Timestamp [ms][pc]"].astype(str)

    # 상대 경로를 기반으로 동일한 구조 유지하여 CSV 파일 저장
    relative_path = os.path.relpath(txt_file, txt_file_path)
    output_file_path = os.path.join(csv_output_directory, os.path.dirname(relative_path), os.path.basename(txt_file).replace(".txt", ".csv"))
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    df_txt.to_csv(output_file_path, index=False, quotechar='"', header=True)
    print(f"CSV 파일 '{output_file_path}'이(가) 생성되었습니다.")
