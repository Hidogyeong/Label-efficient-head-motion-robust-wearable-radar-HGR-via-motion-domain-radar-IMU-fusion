# Revision Experiments: Learning-rate Sensitivity and Statistical Significance

이 프로젝트는 ICT Express major revision 대응을 위한 추가 실험 패키지입니다.
기존 `RadarIMU_HGR_FusionProtocol_Project`를 기반으로 하며, reviewer가 요구한 두 가지를 추가로 생성합니다.

1. Learning-rate sensitivity analysis
2. Statistical significance test for performance comparisons

## 0. 준비

```bash
conda activate hgr_sci
cd ~/miniconda3/envs/hgr_sci/python_code/RadarIMU_HGR_Revision_Experiments_Project
cp paths.env.example paths.env
python -m pip install tabulate scipy
```

`paths.env`는 아래와 같이 설정되어 있으면 됩니다.

```bash
export DATA_ROOT="/media/dokyeong/MINJI/dokyeong/processedData_radar_imu_geometry_GPT_2"
export RESULT_ROOT="/media/dokyeong/MINJI/dokyeong/RadarIMU_HGR_FusionProtocol_Experiments"
```

기존 fusion 실험 결과가 같은 `RESULT_ROOT`에 있으면 통계 검정 스크립트가 같이 읽어옵니다.

## 1. Learning-rate sensitivity quick test

코드 동작만 확인하려면:

```bash
bash run_06_learning_rate_sensitivity_quick.sh 2>&1 | tee logs/06_lr_quick.log
bash run_07_revision_analysis.sh 2>&1 | tee logs/07_revision_analysis_after_quick.log
```

Quick 결과는 논문에 사용하지 마세요.

## 2. Full learning-rate sensitivity

Reviewer response와 논문용 결과는 full run으로 만드세요.

```bash
mkdir -p logs
bash run_06_learning_rate_sensitivity.sh 2>&1 | tee logs/06_learning_rate_sensitivity.log
```

비교 learning rates:

```text
1e-4, 3e-4, 5e-4, 1e-3, 2e-3
```

기본 조건:

```text
fusion_method = motion_domain_proposed
backbone = tcn
class_mode = 4
protocol = mixed_budget
folds = 0..4
seeds = 42, 43, 44
```

## 3. Revision analysis

기존 Table 5 / backbone / label-efficiency / five-class 결과와 새 learning-rate 결과를 한 번에 분석합니다.

```bash
bash run_07_revision_analysis.sh 2>&1 | tee logs/07_revision_analysis.log
```

분석 결과는 아래에 생성됩니다.

```text
$RESULT_ROOT/analysis_revision/
```

가장 중요한 파일:

```text
REVISION_CHATGPT_UPLOAD_PACKAGE.zip
REVISION_REPORT_FOR_CHATGPT.md
learning_rate_sensitivity_paper_table.csv
statistical_significance_table5_fusion_tcn.csv
table5_core_pvalues.csv
```

## 4. ChatGPT에 업로드할 파일

아래 zip 파일을 업로드하세요.

```text
/media/dokyeong/MINJI/dokyeong/RadarIMU_HGR_FusionProtocol_Experiments/analysis_revision/REVISION_CHATGPT_UPLOAD_PACKAGE.zip
```

함께 이 문장을 보내면 됩니다.

```text
업로드한 REVISION_CHATGPT_UPLOAD_PACKAGE.zip을 바탕으로 major revision 대응용 learning-rate sensitivity와 statistical significance 결과를 분석해줘. 특히 proposed motion-domain protocol이 radar-only 대비 유의한지, early/late/attention fusion 대비 유의한지, learning rate 5e-4가 안정적인지, reviewer response에 어떻게 써야 하는지 알려줘.
```

## 5. Reviewer response에서 쓸 핵심 논리

- Proposed vs radar-only의 gain이 작거나 유의하지 않으면, 이를 인정하고 overclaim을 줄입니다.
- Proposed vs early/late/attention fusion의 gain이 더 크면, raw IMU fusion보다 motion-domain representation이 안정적이라고 씁니다.
- LR sensitivity에서는 5e-4가 최선이거나 근처 learning rate들과 성능 차이가 크지 않으면 parameter stability를 강조합니다.
