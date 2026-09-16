#!/usr/bin/env bash
set -euo pipefail
source paths.env
python analyze_revision_lr_stats.py --result_root "$RESULT_ROOT"
