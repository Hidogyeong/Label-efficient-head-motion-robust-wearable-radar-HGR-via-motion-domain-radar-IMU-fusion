#!/usr/bin/env bash
set -euo pipefail
source ./paths.env
for M in manifests/01_table5_fusion_tcn.json manifests/02_backbone_robustness.json manifests/03_label_efficiency_proposed_vs_best_existing.json manifests/04_five_class_check.json; do
  echo "===== $M ====="
  python run_fusion_protocol_suite.py --data_root "$DATA_ROOT" --result_root "$RESULT_ROOT" --manifest "$M" --resume --show_pending
done
