# Results are not included / 논문 결과는 나중에 연결

`processing_data.zip` contains logs but no final run summaries, real datasets, or checkpoints. This repository therefore does not populate paper scores or assert that a particular run produced the final manuscript tables.

When historical results are mounted, set `paper_results_root` to the parent of `runs/` and run `python tools/hgr.py analyze --source paper`. New snapshots are stored outside tracked source, under `artifacts/analysis/`. Preserve final summary CSV/JSON, source hashes, manifest hashes, dataset version, and selected predictions separately from large model/cache files.

Do not delete original results to force reruns. Use a new result root for changed data/code/configurations.
