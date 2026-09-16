# How to Send Results to ChatGPT

After running experiments:

```bash
bash run_05_analyze.sh
```

Upload this file:

```text
$RESULT_ROOT/analysis_fusion_protocol/CHATGPT_UPLOAD_PACKAGE.zip
```

Then paste this prompt:

```text
업로드한 CHATGPT_UPLOAD_PACKAGE.zip을 바탕으로 fusion/training protocol 실험 결과를 분석해줘. 특히 motion_domain_proposed가 radar_only, early_fusion, late_fusion, attention_fusion보다 좋은지, CNN/TCN/GRU/Transformer backbone에서 그 경향이 유지되는지, 논문 방향을 유지할 수 있는지 판단해줘.
```

Important files inside the zip:

- `REPORT_FOR_CHATGPT.md`: readable summary
- `fusion_strategy_aggregate.csv`: Table 5 후보
- `paired_comparisons.csv`: proposed vs existing fusion methods
- `label_efficiency_aggregate.csv`: partial moving-head labels
- `head_metrics_aggregate.csv`: head-wise robustness
- `class_metrics_aggregate.csv`: class-wise errors
