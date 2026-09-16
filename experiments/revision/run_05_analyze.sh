#!/usr/bin/env bash
set -euo pipefail
source ./paths.env
python analyze_fusion_results.py --result_root "$RESULT_ROOT"
echo "Upload this file to ChatGPT:"
echo "$RESULT_ROOT/analysis_fusion_protocol/CHATGPT_UPLOAD_PACKAGE.zip"
