import os

# 입력 파일과 출력 파일 경로 설정
input_file = "E:/dokyeong/IMU_rawdata/txt/junh/01RtoL/None/25-02-2025 015053.txt"
output_file = "E:/dokyeong/IMU_rawdata/txt/junh/01RtoL/None/25-02-2025 015053.txt"

# 파일 전체 읽기
with open(input_file, "r") as f:
    lines = f.readlines()

# "DATA"라는 단어가 있는 줄을 찾은 뒤, 그 이후 3줄(헤더 포함)을 건너뛰어 실제 데이터 시작 인덱스 설정
try:
    data_start_index = next(i for i, line in enumerate(lines) if line.strip() == "DATA") + 3
except StopIteration:
    print("파일에서 'DATA' 라는 단어를 찾을 수 없습니다.")
    exit()

# 실제 데이터가 시작되는 부분
data_lines = lines[data_start_index:]

# 첫 번째 줄은 헤더라고 가정 (탭으로 구분)
header_line = data_lines[0].strip()
headers = header_line.split("\t")
# 원본 헤더 예시:
# Timestamp [ms][pc], Gyr.X, Gyr.Y, Gyr.Z, Axl.X, Axl.Y, Axl.Z, Hdr.X, Hdr.Y, Hdr.Z, q.w, q.i, q.j, q.k
# Hdr 관련 열은 인덱스 7, 8, 9이므로 이들을 제거하여 0~6번과 10~13번 항목만 사용
new_headers = headers[0:7] + headers[10:14]

# 데이터 행 처리 (첫 헤더 이후부터)
data_rows = []
for line in data_lines[1:]:
    parts = line.strip().split("\t")
    if len(parts) == len(headers):
        new_parts = parts[0:7] + parts[10:14]
        data_rows.append(new_parts)

# 결과를 새 텍스트 파일로 저장 (탭 구분)
with open(output_file, "w") as f:
    f.write("\t".join(new_headers) + "\n")
    for row in data_rows:
        f.write("\t".join(row) + "\n")

print(f"파일 '{output_file}'이(가) 생성되었습니다.")
