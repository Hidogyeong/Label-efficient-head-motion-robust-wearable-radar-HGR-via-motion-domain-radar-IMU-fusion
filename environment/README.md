# Environment records / 실행 환경

No exact Conda environment export was included in `processing_data.zip`. Historical logs contain Python 3.8 site-package paths, but that is not an environment lockfile. The bytecode also includes other Python versions and must not be used to infer the historical runtime.

Use the original research PC:

```bash
conda activate hgr_sci
python tools/export_environment.py
```

The current environment is exported under ignored `environment/exports/<timestamp>/`. Review sensitive URLs, local editable-package paths, user identifiers, and export failures. The script redacts some patterns only. A redacted `prefix` line must be removed before restoring YAML, not interpreted literally.

`requirements-core.in` and `requirements-visualization.in` are **unversioned import inventories**, not tested reconstruction environments. Do not upgrade the original environment in place by treating these files as lockfiles. Infineon `ifxradarsdk` must come from the appropriate vendor SDK distribution; native SDK components are not included. The Muse recording program is external.

Current Conda documentation for the export operation: https://docs.conda.io/projects/conda/en/latest/commands/env/export.html
Python dependency lists do not capture every firmware, device, native-library, driver, OS, or acquisition parameter.
