#!/usr/bin/env bash
set -euo pipefail
bash run_01_table5_fusion.sh
bash run_02_backbone_robustness.sh
bash run_03_label_efficiency.sh
bash run_05_analyze.sh
