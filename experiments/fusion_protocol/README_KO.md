# Radar-IMU HGR Fusion Protocol Experiment Project

이 프로젝트는 교수님 통화 피드백에 맞춰 만든 **fusion / training protocol 비교 실험 패키지**입니다.

기존 Table 5의 문제는 `CNN/TCN vs GRU vs Transformer`처럼 단순 backbone 비교였다는 점입니다. 교수님 피드백의 핵심은 다음입니다.

> 새로운 학습/융합 프로토콜을 주장하려면, 기존 sensor fusion 학습 방법들과 동일 조건에서 비교해야 한다.

따라서 이 프로젝트는 아래 5개 방법을 같은 데이터셋, 같은 subject-disjoint split, 같은 backbone에서 비교합니다.

| Method | 의미 |
|---|---|
| `radar_only` | IMU 없이 radar time-series만 사용 |
| `early_fusion` | radar feature와 IMU quaternion feature를 time-step마다 concat |
| `late_fusion` | radar encoder와 IMU encoder를 따로 학습한 뒤 latent feature concat |
| `attention_fusion` | radar sequence와 IMU sequence를 cross-attention으로 결합 |
| `motion_domain_proposed` | geometry/motion-domain 변환 후 Doppler-aware motion-XYZ feature 사용 |

## 1. 설치 및 경로 설정

```bash
unzip RadarIMU_HGR_FusionProtocol_Project.zip
cd RadarIMU_HGR_FusionProtocol_Project
cp paths.env.example paths.env
```

`paths.env` 기본값:

```bash
export DATA_ROOT="/media/dokyeong/MINJI/dokyeong/processedData_radar_imu_geometry_GPT_2"
export RESULT_ROOT="/media/dokyeong/MINJI/dokyeong/RadarIMU_HGR_FusionProtocol_Experiments"
```

필요하면 경로만 수정하세요.

## 2. 가장 먼저 smoke test

```bash
bash run_00_smoke.sh
```

정상이라면 2 epoch짜리 짧은 실험이 끝나고 analysis package가 생성됩니다.

## 3. 교수님 피드백 대응용 핵심 실험

### 3.1 Table 5 재구성용 fusion strategy 비교

```bash
nohup bash run_01_table5_fusion.sh > table5_fusion.log 2>&1 &
tail -f table5_fusion.log
```

이 실험이 가장 중요합니다. CNN/TCN backbone을 고정하고 아래를 비교합니다.

```text
radar_only
early_fusion
late_fusion
attention_fusion
motion_domain_proposed
```

결과 해석:

- `motion_domain_proposed`가 가장 좋으면 현재 논문 방향 유지 가능.
- `early_fusion` 또는 `attention_fusion`이 더 좋으면 논문 방향을 benchmark/empirical study 쪽으로 바꿔야 함.

### 3.2 Backbone robustness 확인

```bash
nohup bash run_02_backbone_robustness.sh > backbone_robustness.log 2>&1 &
tail -f backbone_robustness.log
```

이 실험은 top fusion methods가 CNN/TCN, GRU, Transformer에서 일관되는지 확인합니다.

### 3.3 Label-efficiency 비교

```bash
nohup bash run_03_label_efficiency.sh > label_efficiency.log 2>&1 &
tail -f label_efficiency.log
```

`motion_domain_proposed`와 `early_fusion`에 대해 non-static head label을 0%, 10%, 25%, 50%, 100%로 늘렸을 때 성능을 비교합니다.

### 3.4 선택: 5-class false activation 확인

```bash
nohup bash run_04_five_class_optional.sh > five_class.log 2>&1 &
tail -f five_class.log
```

5-class에서는 `05None` false activation rate를 확인합니다.

## 4. 전체 필수 실험 한 번에 실행

```bash
nohup bash run_all_required.sh > all_required.log 2>&1 &
tail -f all_required.log
```

이미 완료된 run은 `--resume`으로 자동 skip됩니다.

## 5. 결과 분석 및 ChatGPT 전달

```bash
bash run_05_analyze.sh
```

아래 파일이 생성됩니다.

```text
$RESULT_ROOT/analysis_fusion_protocol/CHATGPT_UPLOAD_PACKAGE.zip
```

이 파일을 ChatGPT에 업로드하고, 함께 생성되는 `PROMPT_FOR_CHATGPT.txt` 내용을 입력하면 됩니다.

## 6. 논문에서 Table 5를 어떻게 바꿀지

기존 Table 5는 backbone 비교였지만, 새 실험 후에는 아래처럼 구성해야 합니다.

| Fusion / Training Strategy | Accuracy | Macro-F1 | Weighted-F1 | Moving Acc. | Moving F1 |
|---|---:|---:|---:|---:|---:|
| Radar-only |  |  |  |  |  |
| Early fusion |  |  |  |  |  |
| Late fusion |  |  |  |  |  |
| Attention fusion |  |  |  |  |  |
| Motion-domain proposed |  |  |  |  |  |

본문 설명도 아래 방향으로 바꿔야 합니다.

- `radar_only`가 낮으면: head motion에 대한 보조 정보가 없기 때문.
- `early_fusion`이 낮으면: raw IMU quaternion을 concat해도 radar coordinate shift를 물리적으로 해결하지 못하기 때문.
- `late_fusion`이 낮으면: sensor별 latent feature를 합치지만, coordinate mismatch가 먼저 해결되지 않기 때문.
- `attention_fusion`이 낮으면: attention이 head-motion contaminated timestep에 집중할 수 있기 때문.
- `motion_domain_proposed`가 높으면: geometry/motion-domain representation이 radar와 IMU를 더 물리적으로 적절한 방식으로 결합했기 때문.

## 7. 주의

- 이 프로젝트는 기존 데이터를 다시 읽어 feature cache를 만듭니다.
- `num_workers`를 쓰지 않아 open-file error가 나지 않도록 설계했습니다.
- 첫 실행은 feature cache 생성 때문에 조금 걸릴 수 있습니다.
- 이후 실험은 cache를 사용합니다.
