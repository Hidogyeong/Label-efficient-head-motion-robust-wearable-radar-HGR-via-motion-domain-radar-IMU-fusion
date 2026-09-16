# Label-efficient head-motion-robust wearable radar hand gesture recognition via motion-domain radar–IMU fusion

Research code archive

[한국어](README_KO.md) · [Pipeline](docs/PIPELINE_EN.md) · [Known issues](docs/KNOWN_ISSUES.md) · [Git publishing](docs/GIT_UPLOAD_EN.md)

## 1. Purpose and scope

This repository preserves the **supplied research code, experiment definitions, and provenance** for wearable radar–IMU hand gesture recognition. Its primary source is `processing_data.zip`. One missing merge script was recovered separately from an earlier user attachment and is explicitly labeled as such. The research data are stored on the user's external SSD. **No research dataset, trained checkpoints, or final numerical experiment results are released here.**

The user reported the following acquisition setup:

- **Infineon BGT60TR13C radar**, using modified manufacturer SDK examples.
- **221e Muse IMU**, recorded with the vendor-provided program and subsequently processed/merged. The program's exact name, version, and recording settings have not been supplied.

See [hardware and source provenance](docs/HARDWARE_AND_ORIGIN.md) for official product links. Supported device capabilities are not silently substituted for actual acquisition settings.

This is an archival organization, not a new implementation of the paper. Existing research engines and manifests are retained; new tools manage paths, execution plans, logs, and environment capture. This archive does not assert acceptance/publication status or guarantee reproduction of paper scores.

## 2. Changes made during organization

The upload contained 175 files. Fifty files in duplicated nested experiment folders had the same SHA-256 hashes as their outer counterparts and were mapped to one canonical copy. Sixteen generated `.pyc` files were excluded. Original logs, local `paths.env` files, and the notebook with saved outputs were separated from the Git-ready release.

**Research algorithms and training parameters were not changed.** One script, `make_sample_distribution_table.py`, declared UTF-8 but contained CP949 bytes. Only its encoding was converted to UTF-8; the pre-conversion bytes remain in the private companion archive. A shareable notebook copy retains the source cells but removes outputs and execution counts.

[provenance/source_manifest.csv](provenance/source_manifest.csv) records each original path, canonical path, hashes, duplication decision, and transformation. The old project READMEs are historical records; the present top-level README and `docs/` describe this organized archive.

## 3. Layout

```text
radar-imu-hgr/
├── README.md / README_KO.md
├── acquisition/infineon/              # Sanitized device-example notebook
├── legacy/
│   ├── BGT60TR13C/                    # Radar/IMU processing, helpers, image experiments
│   └── geometry/                     # Geometry script and historical alternatives
├── recovered_prior_uploads/           # Earlier merge-script attachment; provenance pending
├── experiments/
│   ├── fusion_protocol/              # Original main engines/manifests/shell scripts
│   └── revision/                     # Original LR and statistics implementation
├── configs/                          # Local path template, stages, observed config
├── tools/                            # New archival management tools
├── tests/                            # Synthetic interface tests
├── environment/                      # Dependency lists and environment export guide
├── docs/                             # Pipeline, experiment mapping, known limitations
├── provenance/                       # Hashes, imports, symbols, path inventory
├── data/                             # Future data-release guide, no research data
└── results/                          # Future results guide, no paper results
```

`artifacts/`, `workspaces/`, `configs/paths.local.json`, `environment/exports/`, and `private_archive/` are local-only and ignored by Git.

## 4. Inspect first — retraining is not required for archival

Activate the existing environment rather than installing a guessed replacement.

```bash
conda activate hgr_sci
cd /actual/path/to/radar-imu-hgr
python tools/hgr.py verify
python tools/hgr.py list
python tools/scan_before_publish.py
```

These commands do not train on research data. Verification checks source hashes and Python/JSON syntax. Notes about absent optional `private_archive` files are expected in a Git-ready checkout.

Record the actual environment on the original research PC:

```bash
python tools/export_environment.py
```

Outputs go to `environment/exports/<timestamp_id>/`. The exporter records the currently active environment using Conda exports, pip listings, Python/PyTorch information, and GPU driver metadata; it does not install packages. These are **current environment records, not automatically proven historical versions**. Review private package URLs, local paths, and credentials before publishing any exports.

## 5. Configure external paths

```bash
python tools/hgr.py init
```

Edit `configs/paths.local.json`:

```json
{
  "data_root": "/media/dokyeong/MINJI/dokyeong/processedData_radar_imu_geometry_GPT_2",
  "result_root": "~/hgr_outputs/code_archive_replay_v1",
  "paper_results_root": "/media/dokyeong/MINJI/dokyeong/RadarIMU_HGR_FusionProtocol_Experiments"
}
```

`data_root` points to prepared geometry CSV trials. `result_root` is a **new replay output location**. `paper_results_root` points to the parent of the historical `runs/` directory for read-only analysis input.

The experiment loader expects:

```text
data_root/person/gesture/head_condition/*.csv
```

Gesture folders are `01RtoL`, `02LtoR`, `03UtoD`, `04DtoU`, and `05None`. Head condition `None` means a static head; gesture `05None` means no gesture. Do not conflate these labels.

Read-only schema audit after mounting data:

```bash
python tools/check_dataset.py --root /absolute/path/to/geometry_data --scan-rows
```

The audit counts candidate files and checks headers/row counts. It is not a complete dataset validation or a replacement for the training loader.

## 6. Run only when needed

The new launcher calls the original Python engines and manifests, bypassing only shell working-directory/path configuration. It does not replace model definitions or training settings. `run` prints a plan unless `--execute` is specified. Executed stages stream logs to the terminal and unique files under `artifacts/logs/`.

```bash
python tools/hgr.py run table5                 # Plan only
python tools/hgr.py run smoke --execute       # Original 2-epoch smoke test

# Real experiments: run sequentially, never competing copies of the same stage.
python tools/hgr.py run table5 --execute
python tools/hgr.py run backbone --execute
python tools/hgr.py run labels --execute
python tools/hgr.py run five_class --execute
python tools/hgr.py run lr_quick --execute    # 20 epochs; not paper evidence
python tools/hgr.py run lr --execute
```

| Stage | Definition | Runs |
|---|---|---:|
| smoke | 2 methods × fold 0 × seed 42, 2 epochs | 2 |
| table5 | 5 fusion methods × 5 folds × 3 seeds | 75 |
| backbone | 3 fusion methods × 3 backbones × 5 folds × 3 seeds | 135 |
| labels | 2 methods × 5 label fractions × 5 folds × 3 seeds | 150 |
| five_class | 5 fusion methods × 5 folds × 3 seeds | 75 |
| lr_quick | 3 rates × 2 folds × 1 seed, 20 epochs | 6 |
| lr | 5 rates × 5 folds × 3 seeds | 75 |

[EXPERIMENT_MATRIX.csv](docs/EXPERIMENT_MATRIX.csv) contains every configuration. `table5` is a historical experiment ID, not a promise about the current manuscript's table numbering. Backbone robustness is a **3×3 subset**, not all five fusion methods across three backbones.

`Ctrl+C` stops the active execution. A subsequent invocation skips runs that already have `run_summary.json`; this legacy resume is **completed-run skipping, not restart from the interrupted epoch**. Status distinguishes summary-only folders from folders also containing `model.pt`:

```bash
python tools/hgr.py status
python tools/hgr.py status --source paper
```

To use an original shell script instead, enter its project directory and copy/edit `paths.env.example` as `paths.env`. The original `run_all_required.sh` does not include five-class or LR experiments.

## 7. Analyze saved results without overwriting reports

There is no need to retrain just to archive the project. With historical results mounted:

```bash
python tools/hgr.py analyze --source paper
```

For results under the new `result_root`:

```bash
python tools/hgr.py analyze --source new
```

Each invocation creates `artifacts/analysis/<timestamp_id>/` and calls the archived revision analyzer with a separate `--analysis_dir`. It does not overwrite analysis files under the input results folder. The tool records hashes of input summaries and the analyzer.

The ChatGPT handoff file is:

```text
artifacts/analysis/<timestamp_id>/REVISION_CHATGPT_UPLOAD_PACKAGE.zip
```

The original statistical implementation is preserved, not newly validated. In particular, its fold/seed paired tests should not automatically be treated as independent participant-level confirmatory inference. See [known issues](docs/KNOWN_ISSUES.md).

## 8. Preprocessing is branched, not one sequence of every script

Do not execute all versioned geometry, image, or EKF scripts consecutively. They represent alternative experiments. Many scripts have absolute Windows or Linux paths. Create a disposable working copy before adapting paths:

```bash
python tools/prepare_preprocessing.py --include-recovered-merge
```

Read the workspace's `READ_FIRST.txt`, `PATHS_TO_EDIT.csv`, and [PIPELINE_EN.md](docs/PIPELINE_EN.md). The command copies files only; it does not process data.

The supplied `processing_geometry_GPT.py` has the same newline-normalized text as the previously uploaded `geometry_correct_radar_imu_GPT2 (2).py`. `Make_Merged_data_v4.py` was absent from the current ZIP and recovered from an earlier attachment into a separate directory. Its final-experiment provenance must be confirmed before treating it as authoritative.

## 9. Important limitations before a public release

- **The complete raw-acquisition-to-paper chain is not established.** The notebook reads the device default sequence and one frame; a final repetitive timestamped raw-trial recorder was not identified. Despite its name, `HMI_DataCap_raw_v1.py` performs offline correction/rebinning/image generation.
- `radar_only` omits a direct IMU tensor, but the loader prefers `*_corr` radar columns. If those columns were produced using IMU correction, the end-to-end method is not necessarily IMU-free.
- A 64-chirp configuration in the notebook and a 32-chirp descriptor `reshape` do not establish a valid conversion on their own. Raw serialization order and acquisition provenance need checking.
- Cache identity does not include dataset-content hashes. Use a new result root after data or implementation changes.
- No original environment lockfile, dataset, or final run summaries were included in this upload.

These limitations are documented instead of silently changing research behavior. See [KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md).

## 10. Git and future data release

Start with private archival storage and resolve institutional/coauthor and third-party permissions before public release. Infineon copyright/license notices have been preserved. No blanket MIT/Apache license has been assigned; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

```bash
python tools/scan_before_publish.py
git init -b main
git add .
git status --short
git diff --cached --stat
# Commit only after reviewing staged files.
git commit -m "Archive radar-IMU HGR code, provenance, and bilingual documentation"
```

Follow [GIT_UPLOAD_EN.md](docs/GIT_UPLOAD_EN.md) for creating a remote and pushing. No tool in this archive uploads to GitHub automatically. Keep original notebook outputs/local logs outside a public release.

Data release is a separate future step: inspect SSD contents, de-identification, participant sharing permissions, licensing, splits, version hashes, and raw-to-processed mapping. [data/README.md](data/README.md) records the required metadata.

## 11. What was tested

See [VALIDATION.md](docs/VALIDATION.md). Source integrity, syntax, manifest task counts, synthetic feature dimensions, and model forward shapes are not the same as reproducing scientific results. No real-data score or sensor acquisition is claimed to have been rerun.
