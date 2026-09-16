# Validation record / 검증 범위

## Completed on the packaging environment

- Archived file hashes checked against the source mapping: 110 stored paths including the optional private companion files.
- Exact nested duplicates checked: 50 files; all mapped copies were byte-identical.
- Supplied/recovered Python source catalog: 44 files. Encoding normalization of the sample-distribution script is documented separately.
- Python and JSON syntax parsing completed; original shell scripts passed `bash -n`.
- Six unit tests passed: manifest task counts; sanitized notebook outputs; presence of separated geometry/merge provenance; all 15 fusion/backbone forward interfaces; feature dimensions/priority; 51,460-parameter four-class motion-domain TCN.
- Original 2-epoch smoke manifest executed on **208 synthetic CSV trials** using CPU, 13 artificial subject IDs, four labels and two head conditions. Both runs reached their completion output. These synthetic scores are NOT research results and are not included in the Git-ready package.
- Re-running that smoke manifest skipped the two completed runs; resume behavior was observed, not assumed.
- The archived revision analyzer generated a ZIP into a new timestamped output directory using the synthetic run summaries.
- Read-only dataset audit counted all 208 synthetic trials and checked their headers/row counts.
- Best-effort secret/large-file scan completed. This does not certify absence of sensitive content or resolve third-party licensing.

## Not completed / not claimed

- No physical radar or IMU acquisition was run.
- No original external-SSD dataset, original checkpoints, or original environment export was available in `processing_data.zip`.
- No real-data recognition score, manuscript table, p-value, calibration, synchronization accuracy, or published claim was reproduced or certified.
- Raw-to-final preprocessing was not executed end-to-end because recorder/data provenance and original numerical environment are incomplete.
- No GitHub repository was created or pushed.

## Actual packaging environment (not the historical research environment)

```json
{
  "python": "3.13.5",
  "torch": "2.10.0+cpu",
  "numpy": "2.3.5",
  "pandas": "2.2.3",
  "scipy": "1.17.0",
  "platform": "Linux-6.18.44-x86_64-with-glibc2.41",
  "gpu_used": false,
  "scipy_signal_blackmanharris_available": false
}
```

The supplied radar helper calls `scipy.signal.blackmanharris`. Its availability in this packaging environment is **False**. That helper was left unchanged. Export the original environment or create a separately documented compatibility branch; do not assume the radar preprocessing works under any current SciPy version.

Run local checks with:

```bash
python tools/hgr.py verify
python -m unittest discover -s tests -v
```

한국어: 이 기록은 정리본의 파일 보존·문법·인터페이스 검증이다. 실제 논문 성능/물리 보정 검증이 아니며, 합성 smoke 결과를 논문 수치로 사용하면 안 된다.
