# 코드 기준 처리 흐름

이 문서는 파일 내용에서 확인한 흐름이다. 모든 단계가 최종 논문 데이터셋에서 그대로 사용됐다고 추정하지 않는다. `legacy`는 과거 실행 코드를 보존하는 구역이지 모든 파일을 순서대로 실행하라는 뜻이 아니다.

## A. 수집 단계

- Radar: 사용자 설명상 Infineon BGT60TR13C의 SDK example을 수정했다. `acquisition/infineon/notebook_example_sanitized.ipynb`는 기존 sequence 확인 및 한 프레임 읽기 예제다. 반복 수집·timestamp 파일명 저장을 수행하는 최종 스크립트는 이번 ZIP에서 식별되지 않았다.
- IMU: 사용자 설명상 221e Muse 제공 프로그램으로 수집했다. 프로그램은 repository에 포함되지 않았다. 실제 프로그램명/버전/설정은 나중에 기록한다.

## B. Radar descriptor 추출

`legacy/BGT60TR13C/raw_to_RDA.py` 및 같은 디렉터리의 `helpers/`를 함께 사용한다.

입력: `np.loadtxt(..., delimiter=',')`로 읽을 수 있는 numeric CSV. 기본 처리 shape은 `(-1, 3, 32, 64)`다. 출력: `frame, range, doppler, horizontal_angle, vertical_angle`.

`range`는 peak range-bin index, `doppler`는 선택된 Doppler index에서 32를 뺀 값이다. SI 단위 거리/속도 변환은 이 함수에 없다. Beamforming은 27 beams, ±40°이며 실제 코드는 상위 3개 range–beam 에너지 원소의 beam angle을 평균한다. 변수명이 `top_7`이어도 7개를 사용하는 것이 아니다.

기본 실행부는 `E:/...` 경로에 고정되어 있다. 작업 사본에서 input/output root를 수정해야 한다. Raw array의 저장 순서가 확인되지 않은 상태에서 단순 reshape를 일반적인 64→32 chirp 변환으로 간주하지 않는다.

## C. IMU 형식 변환과 trial 정렬 — 하나의 가지를 선택

| 파일 | 실제 역할 | 주의 |
|---|---|---|
| `IMU_txt2csv.py` | `DATA` marker 다음의 텍스트를 11개 column CSV로 변환 | 경로 고정, top-level 실행 |
| `IMU_data_cut.py` | CSV IMU를 radar filename의 start/end 구간으로 자르고 40점 선형 보간 | `start_end.csv` 형식 기대 |
| `IMU_processe.py` | TXT→CSV와 구간 선택/40점 보간을 한 파일에서 수행 | 개별 단계와 중복 실행하지 않음 |
| `IMU_processe_v1.py` | 재귀 폴더 변환형 대안 | 최종 사용 버전 확인 필요 |
| `IMU_txt2csv_junh.py`, `IMU_txt_junh.py` | 특정 기록 형식을 위한 변형 | 일반 처리기로 가정하지 않음 |

각 IMU channel을 `np.interp`로 보간한다. 이 코드 자체는 두 센서 clock offset/drift를 추정하지 않으며 quaternion SLERP도 사용하지 않는다. `Timestamp [ms][dev]`와 radar filename timestamp의 시간 기준이 같은지는 수집 단계에서 확인해야 한다.

IMU를 40점으로 만들지만 `raw_to_RDA.py`가 항상 40행을 내는 것은 아니다. Radar sequence의 40점 변환/선택 단계가 어디서 수행됐는지는 추가 provenance가 필요하다. `interpolation/`의 파일명만으로 그 연결을 단정하지 않는다.

## D. 병합 및 필터 — 이전 첨부본에서 복원

`recovered_prior_uploads/Make_Merged_data_v4.py`는 이번 ZIP에는 없었지만 이전 사용자 첨부가 남아 있어 복원했다. 코드상 IMU 파일의 앞 `I`를 제거하여 대응 radar 파일을 찾고, `zip_longest`로 row-wise 결합한다.

Gyro/accel은 first difference 후 필터, radar descriptor와 quaternion 성분은 직접 필터 처리한다. 사용되는 nominal 설정은 Butterworth order 2, cutoff 10 Hz, fs=100 Hz다. 저장명 `M_10Hz_...`는 downsampling이 있다는 근거가 아니다. 행 수가 다른 입력과 실제 40점 시간 간격을 반드시 점검한다.

최종 사용본 여부를 확인하기 전에는 복원 코드라는 출처 구분을 유지한다.

## E. Geometry — 최종 후보와 과거 변형 분리

현재 ZIP의 `legacy/geometry/processing_geometry_GPT.py`는 파일 내부에 GPT_2라고 명시되어 있고, 이전 `geometry_correct_radar_imu_GPT2 (2).py`와 newline-normalized text가 같다. 별도 새 코드로 합성하지 않았다.

주요 CLI는 `--source_root`, `--target_root`, `--tilt_correction_deg`, `--tilt_axis`, `--quat_direction`, `--reference_mode`, `--quat_yz_swap`, `--scale_factor`다. 실제 이름은 `--help`로 확인한다. 입력/출력을 분리하고 기존 처리 결과를 덮어쓰지 않는다.

```bash
# 도움말만 확인
python legacy/geometry/processing_geometry_GPT.py --help

# 새 입력/출력 경로를 설정했을 때만 수행하는 예
python legacy/geometry/processing_geometry_GPT.py \
  --source_root /absolute/path/to/merged_csv \
  --target_root /absolute/path/to/NEW_geometry_csv \
  --reference_mode relative_first \
  --tilt_correction_deg 45 --tilt_axis y --scale_factor 1
```

이는 코드의 설정 예이며 실제 최종 데이터 생성 명령이 확인되었다는 뜻은 아니다. `v1`~`v7`, `with_distance`, `xyz`는 과거 변형으로 보존한다. 여러 보정을 누적 적용하지 않는다.

## F. 학습과 분석

모델 입력: `data_root/person/gesture/head/*.csv`.

1. `read_csv_features()`가 radar 12차원, IMU 12차원, motion 11차원 시퀀스를 만든다.
2. `protocol_indices()`가 subject split 및 label budget을 설정한다.
3. training split만으로 standardization 값을 계산한다.
4. 선택된 fusion/backbone으로 학습한다.
5. validation macro-F1로 checkpoint를 선택하고 test를 평가한다.
6. summary, head/class metrics, checkpoint를 저장한다.

**radar-only 입력은 `*_corr` 우선**이며 raw backup 열을 우선 선택하는 구현이 아니다. 논문 baseline의 설명과 데이터 컬럼 출처가 일치하는지 확인이 필요하다.

본실험 engine과 revision engine은 별도 보존했다. 두 파일의 차이는 현재 소스 기준으로 summary에 hyperparameter를 추가 저장하는 부분이다. Diff는 `provenance/fusion_vs_revision_engine.diff`에 있다.

## G. 이미지 / STFT / EKF 가지

`raw_to_RDAimage*`, `raw_to_RDAimages*`, `raw_to_STFT*`, `interpolation/*`, `HMI_DataCap_raw_v1.py`는 별도 표현·필터·이미지 가지다. 이번 fusion manifest는 geometry CSV를 읽으며 이 이미지들을 직접 읽지 않는다. 이 파일들은 제거하지 않고 보존하되 최종 필수 단계로 표시하지 않았다.
