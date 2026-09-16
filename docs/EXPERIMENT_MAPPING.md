# Experiment-to-code mapping / 논문 결과 연결 메모

This file maps historical experiment IDs to code/configuration, not to verified final numerical tables. The final manuscript and final `run_summary.json` files are not part of `processing_data.zip`.

| Historical group | Engine | Manifest | Intended question |
|---|---|---|---|
| `T5_*` | fusion_protocol | `01_table5_fusion_tcn.json` | Five fusion strategies, TCN, four-class mixed-budget |
| `B_*` | fusion_protocol | `02_backbone_robustness.json` | Radar-only / early / motion × TCN / GRU / Transformer |
| `LE_*` | fusion_protocol | `03_label_efficiency_proposed_vs_best_existing.json` | Early vs motion at 0/10/25/50/100% non-static TRAINING labels |
| `F5_*` | fusion_protocol | `04_five_class_check.json` | Five-class mixed-budget including no gesture |
| `LR_*` | revision | `06_learning_rate_sensitivity.json` | Five default-learning-rate settings for motion+TCN |
| `SMOKE_*`, `LRQ_*` | respective project | smoke / quick manifests | Runtime checks; do not pool into paper results |

The partial-mixed implementation validates on all head conditions, even at 0% non-static training labels. Thus 0% does not mean that no non-static labels are used anywhere in model selection. The fraction applies to available non-static TRAINING trials, added to all static training trials.

Historical output layout:

```text
RESULT_ROOT/runs/exp_id/fold_00/seed_42/
  model.pt
  run_summary.json
  run_summary.csv
  training_history.csv
  test_metrics_by_head.csv
  test_metrics_by_class.csv
  test_predictions.csv       # only if save_predictions is true
```

Run-level mean scores and seed-averaged fold statistics are different outputs. Keep experiment families and exact code/data hashes separate. `run_all_required.sh` is not all manifests: it covers table5, backbone, labels, and analysis only.

When the SSD data/results are attached, record for each final table: manuscript table ID, data release/version, manifest SHA-256, engine SHA-256, input column provenance, source result root, fold/seed list, summary hashes, analysis version, and the exact exported table. Do not populate unknown fields from previous narrative descriptions.
