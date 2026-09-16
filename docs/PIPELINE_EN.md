# Source-grounded processing pipeline

This describes the uploaded code, not a verified chain of every operation used for all paper data. Do not execute all versioned scripts in order.

## Acquisition

The user reports modified Infineon BGT60TR13C example code and the 221e Muse vendor recording application. The sanitized notebook reads the existing/default sequence and one frame only. The final repetitive timestamp-named CSV recorder was not identified. The Muse application name/version/settings are not supplied. `HMI_DataCap_raw_v1.py` performs offline correction/rebinning/image generation despite its name.

## Radar descriptors

Use `legacy/BGT60TR13C/raw_to_RDA.py` together with its `helpers/` directory. Numeric comma-delimited input is reshaped to `(-1,3,32,64)`. Outputs are `frame, range, doppler, horizontal_angle, vertical_angle`. Range is a peak bin index; Doppler is a selected index minus 32, not a calibrated velocity. DBF uses 27 beams over ±40° and averages angles associated with the top three flattened range–beam energy entries.

The main block has fixed Windows input/output paths. Edit a working copy before processing. The 64-chirp notebook output is not proof that flat reshaping into 32-chirp arrays preserves antenna/time organization; raw export ordering must be checked.

## IMU conversion/alignment: select one branch

- `IMU_txt2csv.py`: vendor TXT following a `DATA` marker → 11-column CSV.
- `IMU_data_cut.py`: select samples in the radar filename's start/end interval and linearly interpolate channels to 40 points.
- `IMU_processe.py`: combines those two operations.
- `IMU_processe_v1.py`: recursive-folder alternative.
- `IMU_txt2csv_junh.py`, `IMU_txt_junh.py`: special recording-format alternatives.

These scripts run top-level code and contain fixed paths; importing them is not a safe inspection method. Filename parsing expects `start_end.csv` in the alignment scripts. `np.interp` is component-wise interpolation, not quaternion SLERP. Clock-offset/drift estimation is not implemented. Matching numeric timestamps assumes compatible clock origins.

Forty-point IMU interpolation does not itself resample the radar output to 40 rows. The exact intermediate radar segmentation/resampling step needs acquisition/data provenance; filenames in `interpolation/` are insufficient evidence.

## Merge/filter: separately recovered attachment

`recovered_prior_uploads/Make_Merged_data_v4.py` was absent from this ZIP and recovered from an earlier attachment. It pairs IMU filenames by removing their leading `I`, joins rows with `zip_longest`, differences gyro/acceleration, and filters radar/quaternion channels directly. Nominal Butterworth settings are order 2, 10 Hz cutoff, fs=100 Hz. The `M_10Hz_...` prefix is not evidence of downsampling. Confirm row lengths, effective sample times, and whether this recovered version generated the final data.

## Geometry

`legacy/geometry/processing_geometry_GPT.py` describes itself as GPT_2 and matches the previously uploaded GPT2 script after newline normalization. `--help` lists the actual interface. An example using NEW output data is:

```bash
python legacy/geometry/processing_geometry_GPT.py --help
python legacy/geometry/processing_geometry_GPT.py \
  --source_root /absolute/path/to/merged_csv \
  --target_root /absolute/path/to/NEW_geometry_csv \
  --reference_mode relative_first \
  --tilt_correction_deg 45 --tilt_axis y --scale_factor 1
```

This is a supported command example, not a verified historical command. Other `v1`–`v7`, distance, and XYZ variants are retained as alternatives, not mandatory sequential steps.

## Training and analysis

The experiment loader reads `person/gesture/head/*.csv`, constructs radar (12), IMU (12), and motion (11) feature channels, performs subject-disjoint splitting, standardizes from training samples, selects the best validation-macro-F1 model, and saves test metrics/checkpoints. It **prefers corrected radar columns even for `radar_only`**; omission of a direct IMU tensor is not the same as IMU-free preprocessing.

The core and revision engines are separately preserved. Their current text difference is the extra hyperparameter fields written into revision summaries; see `provenance/fusion_vs_revision_engine.diff`.

`raw_to_RDAimage*`, STFT, interpolation/EKF, and `HMI_DataCap_raw_v1.py` remain separate exploratory representation branches. The supplied fusion manifests do not directly load these images.
