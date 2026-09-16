# Known issues / 확인된 제약 — no silent research changes

The items below are observations from the supplied source, not diagnoses from rerunning the research dataset. They are retained for archival transparency. This organization does not repair or reinterpret paper results.

| ID | Source observation | Implication / 다음 확인 |
|---|---|---|
| K01 | `read_csv_features()` prefers `range_corr`, `doppler_corr`, corrected angle columns for all methods. `*_raw` is not preferentially used. | `radar_only` means no direct IMU tensor, not necessarily IMU-free upstream processing. Verify actual dataset columns before claiming modality ablation. |
| K02 | `raw_to_RDA.py` performs a flat `reshape(-1,3,32,64)`; the notebook's single saved sequence has 64 chirps/frame. | A correct split across chirps/antennas cannot be established without raw export shape/order. Do not describe it as verified 64→32 block conversion merely from the defaults. |
| K03 | The notebook only reads default sequence/one frame. `HMI_DataCap_raw_v1.py` is offline processing. | Final acquisition recorder and timestamp filename generation are not established from this ZIP. |
| K04 | IMU alignment uses filename start/end timestamps and component-wise `np.interp` to 40 points; merge then joins rows. | Clock-origin equivalence, dropped frames, IMU sign continuity, radar 40-point resampling, and actual row alignment need provenance. This is not demonstrated hardware synchronization. |
| K05 | Recovered merge code uses nominal fs=100 Hz, cutoff 10 Hz after trial normalization; it saves `M_10Hz_...`. | File naming does not establish 10 Hz data rate. The physical meaning of the filter depends on actual sample spacing. |
| K06 | Geometry uses unit-quaternion operations, a fixed tilt, default scale 1, and reference modes. Legacy variants may use other scales or axis swaps. | Preserve the exact generating command/data version. No full extrinsic calibration is inferred from the scripts. |
| K07 | Loader fills missing scalar channels with zeros, missing quaternion with identity, and prefers `radar_*_world` when available. | Successful loading does not prove that every sample contains the intended measurements. Use schema audit. |
| K08 | Motion displacement and the quantity named `speed` are differences/norms without division by physical time. | Describe as displacement per normalized sequence step unless physical scaling is separately established. |
| K09 | Cache key hashes the data-root path, class mode, steps and a fixed version token, not data contents. Resume tests summary-file existence, not code/config/data identity. | Do not reuse a result root after data/code changes. The new launcher records code/config fingerprints but does not prove dataset-content identity. |
| K10 | Summary is written before `model.pt`; a crash between them can leave summary-only runs. | `status` reports summary-only separately. No source claim that every SKIP is a fully recoverable checkpoint. |
| K11 | Original `analyze_fusion_results.py` groups without `exp_id`; its paired merge also omits label fraction in grouping. | Mixing SMOKE/T5/B/LR or label fractions may combine unlike experiments or create multiple pairings. Retained as legacy; prefer family-separated revision analyzer for inspection. |
| K12 | Revision analysis separates T5/B/LE/F5 and LR, but paired tests treat fold/seed results as pairs without a dependence/multiplicity correction. | Do not equate repeated seeds with new independent participants. P-values are historical outputs; inference needs separate statistical review. |
| K13 | `set_seed()` enables cuDNN benchmark and disables deterministic mode. | Same seed does not guarantee bitwise results across runs/hardware. |
| K14 | Current training code applies Gaussian jitter and feature scaling; no time-shift operation is present in `maybe_augment()`. Scheduler patience is 6 in provided manifests. | Old README/notes mentioning temporal shifts or different patience values are historical claims, not this implementation. |
| K15 | `helpers/DopplerAlgo.py` calls `scipy.signal.blackmanharris`. | Availability depends on the original SciPy version. Modern documented location is `scipy.signal.windows.blackmanharris`; do not silently patch archived source. Export the original environment or document a separate compatibility patch. |
| K16 | Only `make_sample_distribution_table.py` had CP949 bytes with a UTF-8 declaration. | Encoding-only conversion made in the organized copy; original bytes retained privately. Its counts still count CSV paths, not loader-accepted valid trials. |
| K17 | Original `.sh` scripts assume the experiment folder is the current directory and read local `paths.env`. | New launcher avoids that path issue. The original wrappers remain unchanged for provenance. |
| K18 | The archive contains mostly resume/analysis logs, not final `run_summary.json` files, model checkpoints, or datasets. | Do not derive completion, publication tables, or performance claims solely from logs. Data/results linkage will be a later step. |

## Source pointers

- `experiments/fusion_protocol/run_fusion_protocol_suite.py`: `read_csv_features`, `build_or_load_dataset`, `protocol_indices`, `prepare_inputs`, `train_one`, `run_experiment`.
- `experiments/revision/analyze_revision_lr_stats.py`: `fold_seed_aggregate`, `paired_stats`, `compute_comparison`.
- `legacy/BGT60TR13C/raw_to_RDA.py`, `IMU_data_cut.py`, `IMU_processe.py`.
- `recovered_prior_uploads/Make_Merged_data_v4.py` (earlier attachment, version unconfirmed).

Exact line numbers are indexed in `provenance/code_symbols.csv`. Existing paper-level claims are not automatically endorsed by this code archive.

External compatibility reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.blackmanharris.html
