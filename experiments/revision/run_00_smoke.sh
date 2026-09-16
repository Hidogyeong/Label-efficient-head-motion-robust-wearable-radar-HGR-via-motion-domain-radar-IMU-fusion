#!/usr/bin/env bash
set -euo pipefail
source ./paths.env
python run_fusion_protocol_suite.py \
  --data_root "$DATA_ROOT" \
  --result_root "$RESULT_ROOT" \
  --manifest manifests/00_smoke.json \
  --resume
python analyze_fusion_results.py --result_root "$RESULT_ROOT"
