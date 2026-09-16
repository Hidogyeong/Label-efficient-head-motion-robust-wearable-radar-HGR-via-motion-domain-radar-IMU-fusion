# Label-efficient head-motion-robust wearable radar hand gesture recognition via motion-domain radar–IMU fusion

웨어러블 Radar–IMU 손동작 인식 연구 코드 보관소

[English](README.md) · [코드별 실행 흐름](docs/PIPELINE_KO.md) · [확인된 제약](docs/KNOWN_ISSUES.md) · [Git 업로드](docs/GIT_UPLOAD_KO.md)

## 1. 이 저장소의 목적과 범위

웨어러블 Radar–IMU 손동작 인식 연구의 **실제로 전달된 코드, 설정, 파일 간 연결**을 보관한다. 입력은 `processing_data.zip`이며, 예전 대화에서 제공된 병합 코드 1개를 출처를 명시하여 별도로 복원했다. 데이터는 사용자의 외장 SSD에 있으며 이번 배포에는 **연구 데이터·학습 가중치·최종 실험 결과값이 포함되지 않는다**.

하드웨어와 수집 방식은 사용자가 다음과 같이 설명했다.

- Radar: **Infineon BGT60TR13C**. 제조사 SDK example을 수정해 사용했다.
- IMU: **221e Muse**. 제품에서 제공하는 프로그램으로 수집한 뒤 데이터를 처리·병합했다. 프로그램의 정확한 이름과 버전은 아직 제공되지 않았다.

장비 공식 출처는 [하드웨어 및 출처 문서](docs/HARDWARE_AND_ORIGIN.md)에 있다. 제품이 지원하는 기능을 실제 실험 설정으로 임의 대체하지 않는다.

이 저장소는 논문 알고리듬을 새로 구현한 버전이 아니다. 연구 코드와 manifest를 보존하고, 실행 계획·로그·환경 기록을 위한 **별도 관리 도구**를 추가한 버전이다. 논문의 게재 상태나 성능 재현을 보증하지 않는다.

## 2. 정리 과정에서 바뀐 것

`processing_data.zip`에는 175개 파일이 들어 있었다. 두 실험 프로젝트 안에 다시 들어 있던 중첩 폴더의 50개 파일은 바깥 파일과 SHA-256이 같아 한 사본으로 정리했다. 16개 `.pyc`는 재생성 가능한 산출물로 제외했다. 원래 로그와 `paths.env`, 출력이 남아 있는 notebook은 공개 Git용 코드와 분리했다.

**연구 알고리듬과 학습 파라미터는 변경하지 않았다.** 다만 `make_sample_distribution_table.py`는 헤더가 UTF-8인데 실제 바이트가 CP949였으므로 실행 가능한 UTF-8로 인코딩만 바꿨다. 변환 전 원본은 별도 비공개 보관본에 있다. Notebook은 원래 source를 유지하고 출력·실행 횟수만 지운 공개용 사본을 만들었다.

모든 원본→정리본 경로, SHA-256, 중복 처리, 인코딩 변경은 [source_manifest.csv](provenance/source_manifest.csv)에 기록되어 있다. 기존 experiment 폴더의 README는 당시 기록으로 보존했다. 현재 실행 안내는 이 README와 `docs/`를 따른다.

## 3. 폴더 구조

```text
radar-imu-hgr/
├── README.md                         # 영어 안내
├── README_KO.md                      # 한국어 안내
├── acquisition/infineon/              # 출력 제거된 제조사 notebook 예제
├── legacy/
│   ├── BGT60TR13C/                    # radar/IMU 전처리, helpers, 이미지 실험
│   └── geometry/                     # geometry 보정 및 과거 변형들
├── recovered_prior_uploads/           # 이전 첨부에서 복원한 병합 코드
├── experiments/
│   ├── fusion_protocol/              # 본실험 원본 코드·manifest·shell
│   └── revision/                     # LR sensitivity·통계 분석 코드
├── configs/                          # 로컬 경로 예시, stage 목록, notebook 관측 설정
├── tools/                            # 새로 추가한 관리 도구
├── tests/                            # 합성 데이터 기반 인터페이스 검사
├── environment/                      # 의존성 목록 / 사용자 환경 export 안내
├── docs/                             # 처리 흐름, 실행 이력 연결, 한계, 공개 준비
├── provenance/                       # SHA-256, 중복/경로/import 목록
├── data/                             # 향후 데이터 연결 안내; 데이터 없음
└── results/                          # 향후 결과 보관 안내; 논문 결과 없음
```

`artifacts/`, `workspaces/`, `configs/paths.local.json`, `environment/exports/`, `private_archive/`는 로컬 전용이며 `.gitignore`에 포함된다.

## 4. 먼저 확인할 것 — 학습을 다시 돌릴 필요 없음

압축을 푼 폴더에서 기존 `hgr_sci` 환경을 활성화한다. Git 보관을 위해 설치된 환경 전체를 복사할 필요는 없다.

```bash
conda activate hgr_sci
cd /실제로/압축을/푼/경로/radar-imu-hgr

python tools/hgr.py verify
python tools/hgr.py list
python tools/scan_before_publish.py
```

위 명령은 실제 연구 데이터 학습을 하지 않는다. `verify`는 보존 파일의 hash와 Python/JSON 문법만 검사한다. 별도 비공개 보관본을 풀지 않은 경우 `private_archive` 파일이 없다는 NOTE는 정상이다.

실제 연구 환경 기록을 남기려면:

```bash
python tools/export_environment.py
```

출력 위치는 `environment/exports/날짜_고유번호/`다. 현재 활성 환경의 Conda export, pip 목록, Python/PyTorch/GPU 정보를 저장한다. **과거 환경이라고 가정하지 말고**, export 안의 패키지 URL·개인 경로·자격정보를 확인한 뒤 공개용 기록을 선별한다. 패키지 설치는 수행하지 않는다.

## 5. 데이터 연결

```bash
python tools/hgr.py init
```

생성된 `configs/paths.local.json`을 수정한다.

```json
{
  "data_root": "/media/dokyeong/MINJI/dokyeong/processedData_radar_imu_geometry_GPT_2",
  "result_root": "~/hgr_outputs/code_archive_replay_v1",
  "paper_results_root": "/media/dokyeong/MINJI/dokyeong/RadarIMU_HGR_FusionProtocol_Experiments"
}
```

- `data_root`: 모델 입력용 geometry 처리 CSV가 있는 루트.
- `result_root`: **새로운 재실행 결과용 폴더**. 기존 논문 결과 폴더와 분리하는 것이 기본값이다.
- `paper_results_root`: 기존 논문 실험 결과를 읽어 분석하려는 경우의 경로. 데이터 폴더가 아니라 `runs/`의 상위 폴더다.

모델 입력 구조:

```text
data_root/person/gesture/head_condition/*.csv
```

손동작 폴더는 `01RtoL`, `02LtoR`, `03UtoD`, `04DtoU`, `05None`이며, `head_condition/None`과 `gesture/05None`은 서로 다른 의미다. `None`은 머리 정지 조건이고, `05None`은 무동작 클래스다.

데이터를 연결한 뒤 읽기 전용 확인:

```bash
python tools/check_dataset.py \
  --root /media/dokyeong/MINJI/dokyeong/processedData_radar_imu_geometry_GPT_2 \
  --scan-rows
```

이는 파일 수/컬럼/행 수 검사다. 원래 학습 loader의 샘플 유효성 검사나 실제 데이터 검증을 대체하지 않는다.

## 6. 실험 실행 — 실제 필요할 때만

새 launcher는 원래 Python engine과 manifest를 호출한다. 원래 `.sh`의 작업 디렉터리 의존성과 `paths.env`를 우회할 뿐 모델·학습 설정은 바꾸지 않는다. 기본 실행은 계획 출력이며 `--execute`가 있어야 학습한다. 로그는 화면과 `artifacts/logs/날짜_단계.log`에 함께 저장된다.

```bash
# 실행 계획만 확인
python tools/hgr.py run table5

# 코드 동작 확인용 원래 2-epoch smoke test
python tools/hgr.py run smoke --execute

# 실제 본실험: 반드시 순차 실행
python tools/hgr.py run table5 --execute
python tools/hgr.py run backbone --execute
python tools/hgr.py run labels --execute
python tools/hgr.py run five_class --execute

# 리비전 LR 실험
python tools/hgr.py run lr_quick --execute   # 20-epoch quick: 논문 결과가 아님
python tools/hgr.py run lr --execute
```

| Stage | 실험 정의 | Run 수 |
|---|---|---:|
| smoke | 2가지 방식, fold 0, seed 42, 2 epochs | 2 |
| table5 | 5 fusion 방식 × 5 folds × 3 seeds | 75 |
| backbone | 3 fusion 방식 × 3 backbones × 5 folds × 3 seeds | 135 |
| labels | 2 fusion 방식 × 5 label fractions × 5 folds × 3 seeds | 150 |
| five_class | 5 fusion 방식 × 5 folds × 3 seeds | 75 |
| lr_quick | 3 learning rates × 2 folds × 1 seed, 20 epochs | 6 |
| lr | 5 learning rates × 5 folds × 3 seeds | 75 |

세부 조합은 [EXPERIMENT_MATRIX.csv](docs/EXPERIMENT_MATRIX.csv)를 참조한다. `table5`라는 명칭은 당시 실험 ID이며 최종 원고의 현재 표 번호를 보장하지 않는다. Backbone 실험은 **3×3**이며 모든 fusion 5개에 대해 수행하는 5×3 실험이 아니다.

`Ctrl+C`는 현재 실행을 중단한다. 재실행 시 완료된 `run_summary.json`을 가진 run은 원래 engine이 건너뛴다. **중단된 run의 epoch부터 이어지는 checkpoint resume은 아니다.** `status`는 summary와 model 존재를 구분한다.

```bash
python tools/hgr.py status
python tools/hgr.py status --source paper
```

원본 `.sh`를 직접 쓰려면 해당 experiment 폴더로 이동하고 `paths.env.example`을 `paths.env`로 복사해 수정해야 한다. 원본 `run_all_required.sh`는 Table 5/Backbone/Label 실험과 기존 분석만 실행하며 five-class와 LR을 포함하지 않는다.

## 7. 기존 결과 분석과 ChatGPT 전달

지금은 코드 정리 단계이므로 논문 실험을 다시 실행할 필요가 없다. 외장 SSD의 기존 결과가 연결되어 있다면:

```bash
python tools/hgr.py analyze --source paper
```

새로운 `result_root`에서 생성한 결과를 분석하려면:

```bash
python tools/hgr.py analyze --source new
```

매번 `artifacts/analysis/날짜_고유번호/`를 새로 만들어 이전 분석 파일을 덮어쓰지 않는다. 원래 revision analyzer를 `--analysis_dir`로 호출하며 기존 결과 root에 분석 파일을 덮어쓰지 않는다.

ChatGPT 전달 파일:

```text
artifacts/analysis/날짜_고유번호/REVISION_CHATGPT_UPLOAD_PACKAGE.zip
```

분석에 사용한 summary 파일 hash와 analyzer hash도 기록한다. 기존 통계 코드는 그대로 보존했다. 따라서 출력 p-value를 새로운 검증 없이 독립적인 참가자 수준의 확증 통계로 해석하면 안 된다. [KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)를 먼저 확인한다.

## 8. 전처리 실행 원칙

여러 `v1`, `v2`, 이미지/EKF 스크립트를 **전부 차례대로 실행하지 않는다**. 서로 다른 실험 가지다. 하드코딩된 Windows/Linux 경로가 있으므로 원본 보관 파일을 직접 편집하지 않고 작업 사본을 만든다.

```bash
python tools/prepare_preprocessing.py --include-recovered-merge
```

생성된 `workspaces/preprocessing_날짜/READ_FIRST.txt`, `PATHS_TO_EDIT.csv`와 [전처리 실행 흐름](docs/PIPELINE_KO.md)을 읽고 필요한 스크립트만 선택한다. 이 명령 자체는 전처리를 실행하지 않는다.

이번 ZIP의 `processing_geometry_GPT.py`는 이전에 업로드된 `geometry_correct_radar_imu_GPT2 (2).py`와 줄바꿈을 정규화한 텍스트가 같다. 반면 병합 코드 `Make_Merged_data_v4.py`는 이번 ZIP에 없어 이전 사용자 첨부본에서 복원하여 `recovered_prior_uploads/`에 따로 넣었다. 최종 실험 사용본인지 별도 확인이 필요하다.

## 9. 공개 전에 알아야 하는 핵심 제약

1. **완성된 원시 수집→논문 재현 전체가 확인된 것은 아니다.** Notebook은 SDK 기본 설정을 확인하고 한 프레임을 읽는 예제다. Timestamp 파일명으로 반복 저장하는 실제 수정 recorder는 이번 ZIP에서 식별되지 않았다. `HMI_DataCap_raw_v1.py`는 이름과 달리 저장된 radar/IMU 배열의 보정·이미지화 코드다.
2. `radar_only`는 IMU tensor를 모델에 직접 넣지 않는다는 뜻이다. 실제 loader는 `*_corr`를 우선하므로, upstream에서 IMU 보정한 열이면 **전처리까지 포함한 IMU-free baseline이라는 뜻은 아니다**.
3. Notebook의 64 chirps/frame 설정과 descriptor 코드의 32-chirp `reshape`는 자동으로 정합이 보장되지 않는다. Raw export의 배열 순서와 해당 설정의 데이터셋 적용 여부를 확인해야 한다.
4. Cache key는 데이터 파일 내용 hash가 아니다. 코드·설정·데이터가 바뀌면 새로운 `result_root`를 사용한다.
5. 환경 lockfile과 최종 결과가 이번 입력 ZIP에 없다. 설치 패키지 목록은 사용자의 실행 PC에서 export해야 한다.

추가 제약·원래 analyzer의 grouping 문제·quaternion 처리·nominal filter rate는 [KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)에 기록했다. 이번 정리에서 알고리듬 수정으로 숨기지 않았다.

## 10. Git 보관 및 향후 데이터 공개

먼저 비공개 저장소로 보관하고, 공동저자/기관 및 제3자 코드의 공개 범위를 확인한 뒤 공개 여부를 결정한다. Infineon 원본 copyright/조건은 그대로 보존했고 전체 repository에 임의로 MIT 등의 라이선스를 붙이지 않았다. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)를 참조한다.

```bash
python tools/scan_before_publish.py
git init -b main
git add .
git status --short
git diff --cached --stat
# 아래는 staged 파일 확인 후 실행
git commit -m "Archive radar-IMU HGR code, provenance, and bilingual documentation"
```

원격 생성과 push는 [GIT_UPLOAD_KO.md](docs/GIT_UPLOAD_KO.md)를 따른다. 관리 도구는 GitHub에 자동 업로드하지 않는다. 개인 경로와 노트북 출력 보관본은 공개 ZIP이나 Git commit에 넣지 않는다.

데이터 공개는 이번 작업과 별개다. 외장 SSD 자료를 확인한 뒤 참가자 비식별화, 공유 동의 범위, 데이터 라이선스, 실제 split, 데이터 버전/checksum, raw→processed 연결을 준비한다. [data/README.md](data/README.md)에 다음 단계 항목을 남겼다.

## 11. 검증 범위

[검증 기록](docs/VALIDATION.md)을 참조한다. 코드 parse, 보존 hash, manifest run 수, 모델 forward shape 등과 **실제 실험 결과 재현**은 다르다. 데이터·센서·원래 환경 없이 학습 성능을 재현했다고 주장하지 않는다.
