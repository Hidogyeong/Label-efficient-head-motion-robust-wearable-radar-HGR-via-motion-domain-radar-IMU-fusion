#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs
bash run_06_learning_rate_sensitivity.sh 2>&1 | tee logs/06_learning_rate_sensitivity.log
bash run_07_revision_analysis.sh 2>&1 | tee logs/07_revision_analysis.log
