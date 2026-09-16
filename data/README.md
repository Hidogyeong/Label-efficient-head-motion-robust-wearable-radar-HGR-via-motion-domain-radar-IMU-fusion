# Data are not included / 데이터는 추후 별도 연결

This release archives code only. Configure external data via `configs/paths.local.json`; do not copy the full SSD into Git now. Directories `data/raw`, `data/processed`, and `data/examples` are ignored.

The learning loader expects `person/gesture/head/*.csv`. Core radar column names are `range`, `doppler`, `horizontal_angle`, `vertical_angle`; corrected versions may be preferred. Quaternion fields are `q.w`, `q.i`, `q.j`, `q.k`; motion features may use `radar_x_world`, `radar_y_world`, `radar_z_world`. Full schema and precedence are in the archived code, not inferred from filenames.

A future release needs: participant consent/sharing scope; non-identifying subject IDs; raw/processed split; collection device/software versions; exact acquisition config; timestamp clock convention; serialization dimensions/order; segmentation/resampling steps; preprocessing command and commit; file hashes; label definitions; distribution counts after loader validation; train/validation/test membership; exclusion criteria; data license; repository/DOI/archive location. Do not announce public availability before these are actually supplied and approved.

한국어: 개인 식별 정보와 공유 허용 범위를 확인한 뒤 raw/processed 버전, 수집·전처리 설정, split, checksum을 먼저 문서화한다. 이번 단계는 코드 보관이며 데이터 공개는 완료되지 않았다.
