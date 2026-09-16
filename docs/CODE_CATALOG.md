# Code catalog / 파일별 역할

This catalog lists supplied/recovered Python source files. Original-to-current hashes are in `provenance/source_manifest.csv`. Symbols/line numbers are in `provenance/code_symbols.csv`.

| File | Role | Purpose |
|---|---|---|
| `experiments/fusion_protocol/analyze_fusion_results.py` | legacy_analysis | Original aggregate/report generation; mixed-family grouping caveats apply |
| `experiments/fusion_protocol/make_sample_distribution_table.py` | dataset_summary | Folder-based trial counts / LaTeX tables; encoding-only UTF-8 fix |
| `experiments/fusion_protocol/run_fusion_protocol_suite.py` | main_experiment | Fusion feature loader, subject split, model, training, evaluation engine |
| `experiments/revision/analyze_fusion_results.py` | legacy_analysis | Original aggregate/report generation; mixed-family grouping caveats apply |
| `experiments/revision/analyze_revision_lr_stats.py` | revision_analysis | Family-separated LR/significance analysis, historical statistical assumptions preserved |
| `experiments/revision/run_fusion_protocol_suite.py` | revision_engine | Fusion feature loader, subject split, model, training, evaluation engine |
| `legacy/BGT60TR13C/HMI_DataCap_raw_v1.py` | exploratory_image_branch | Offline array correction, rebinning, and image export; NOT acquisition |
| `legacy/BGT60TR13C/IMU_data_cut.py` | imu_preprocessing_variant | Vendor export conversion or trial alignment variant; choose based on recording format |
| `legacy/BGT60TR13C/IMU_processe.py` | imu_preprocessing_variant | Vendor export conversion or trial alignment variant; choose based on recording format |
| `legacy/BGT60TR13C/IMU_processe_v1.py` | imu_preprocessing_variant | Vendor export conversion or trial alignment variant; choose based on recording format |
| `legacy/BGT60TR13C/IMU_txt2csv.py` | imu_preprocessing_variant | Vendor export conversion or trial alignment variant; choose based on recording format |
| `legacy/BGT60TR13C/IMU_txt2csv_junh.py` | imu_preprocessing_variant | Vendor export conversion or trial alignment variant; choose based on recording format |
| `legacy/BGT60TR13C/IMU_txt_junh.py` | imu_preprocessing_variant | Vendor export conversion or trial alignment variant; choose based on recording format |
| `legacy/BGT60TR13C/helpers/DigitalBeamForming.py` | third_party_helper | Supplied Infineon example helper; retain header and import context |
| `legacy/BGT60TR13C/helpers/DistanceAlgo.py` | third_party_helper | Supplied Infineon example helper; retain header and import context |
| `legacy/BGT60TR13C/helpers/DopplerAlgo.py` | third_party_helper | Supplied Infineon example helper; retain header and import context |
| `legacy/BGT60TR13C/helpers/fft_spectrum.py` | third_party_helper | Supplied Infineon example helper; retain header and import context |
| `legacy/BGT60TR13C/interpolation/data_interporation_image_filter.py` | exploratory_variant | Archived interpolation/image/EKF experiment; not automatically selected by training manifests |
| `legacy/BGT60TR13C/interpolation/data_interportatiion_EKF_test.py` | exploratory_variant | Archived interpolation/image/EKF experiment; not automatically selected by training manifests |
| `legacy/BGT60TR13C/interpolation/data_interportation_EKF_frame.py` | exploratory_variant | Archived interpolation/image/EKF experiment; not automatically selected by training manifests |
| `legacy/BGT60TR13C/interpolation/data_interportation_EKF_test.py` | exploratory_variant | Archived interpolation/image/EKF experiment; not automatically selected by training manifests |
| `legacy/BGT60TR13C/interpolation/data_interportation_frame.py` | exploratory_variant | Archived interpolation/image/EKF experiment; not automatically selected by training manifests |
| `legacy/BGT60TR13C/interpolation/data_interportation_image.py` | exploratory_variant | Archived interpolation/image/EKF experiment; not automatically selected by training manifests |
| `legacy/BGT60TR13C/interpolation/data_interportation_test.py` | exploratory_variant | Archived interpolation/image/EKF experiment; not automatically selected by training manifests |
| `legacy/BGT60TR13C/raw_to_RDA.py` | descriptor_candidate | Numeric raw CSV to four scalar radar descriptors using helpers |
| `legacy/BGT60TR13C/raw_to_RDAimage_CFAR.py` | exploratory_representation | Archived radar image/STFT representation branch, distinct from scalar descriptor training |
| `legacy/BGT60TR13C/raw_to_RDAimage_frame_norn_csv.py` | exploratory_representation | Archived radar image/STFT representation branch, distinct from scalar descriptor training |
| `legacy/BGT60TR13C/raw_to_RDAimage_image_norm.py` | exploratory_representation | Archived radar image/STFT representation branch, distinct from scalar descriptor training |
| `legacy/BGT60TR13C/raw_to_RDAimage_image_norn_csv.py` | exploratory_representation | Archived radar image/STFT representation branch, distinct from scalar descriptor training |
| `legacy/BGT60TR13C/raw_to_RDAimages_frame_norm.py` | exploratory_representation | Archived radar image/STFT representation branch, distinct from scalar descriptor training |
| `legacy/BGT60TR13C/raw_to_STFT.py` | exploratory_representation | Archived radar image/STFT representation branch, distinct from scalar descriptor training |
| `legacy/BGT60TR13C/raw_to_STFT_NTT.py` | exploratory_representation | Archived radar image/STFT representation branch, distinct from scalar descriptor training |
| `legacy/geometry/processing_geometry_GPT.py` | geometry_candidate | GPT_2 geometric correction, relative_first default, raw backup and selected XYZ columns |
| `legacy/geometry/processing_geometry_v1.py` | geometry_variant | Archived geometry alternative; do not execute sequentially with other versions |
| `legacy/geometry/processing_geometry_v2.py` | geometry_variant | Archived geometry alternative; do not execute sequentially with other versions |
| `legacy/geometry/processing_geometry_v3.py` | geometry_variant | Archived geometry alternative; do not execute sequentially with other versions |
| `legacy/geometry/processing_geometry_v4.py` | geometry_variant | Archived geometry alternative; do not execute sequentially with other versions |
| `legacy/geometry/processing_geometry_v5.py` | geometry_variant | Archived geometry alternative; do not execute sequentially with other versions |
| `legacy/geometry/processing_geometry_v6.py` | geometry_variant | Archived geometry alternative; do not execute sequentially with other versions |
| `legacy/geometry/processing_geometry_v7.py` | geometry_variant | Archived geometry alternative; do not execute sequentially with other versions |
| `legacy/geometry/processing_geometry_with_distance.py` | geometry_variant | Archived geometry alternative; do not execute sequentially with other versions |
| `legacy/geometry/processing_geometry_with_distance_v2.py` | geometry_variant | Archived geometry alternative; do not execute sequentially with other versions |
| `legacy/geometry/processing_geometry_xyz.py` | geometry_variant | Archived geometry alternative; do not execute sequentially with other versions |
| `recovered_prior_uploads/Make_Merged_data_v4.py` | recovered_version_unconfirmed | Recovered earlier row-wise radar/IMU merge and nominal Butterworth filtering |
