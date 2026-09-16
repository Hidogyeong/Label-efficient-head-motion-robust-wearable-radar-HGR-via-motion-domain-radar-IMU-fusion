# Git publishing workflow

Extract the Git-ready archive into a new folder; do not move/delete the original environment or research directories. Start with private archival storage and review permissions, vendor notices, credentials, and personal paths before public release.

```bash
conda activate hgr_sci
python tools/hgr.py verify
python tools/scan_before_publish.py
git init -b main
git add .
git status --short
git diff --cached --stat
```

Check that `private_archive/`, `artifacts/`, `workspaces/`, local config/environment paths, research data, and model/cache binaries are not staged. The supplied `.gitignore` prevents new tracking of these locations; it does not erase already tracked history in a pre-existing repository.

```bash
git commit -m "Archive radar-IMU HGR code and bilingual documentation"
```

Create an empty private repository on GitHub without separately initializing README/license/gitignore files. Replace the example SSH remote with your actual repository URL and configured authentication:

```bash
git remote add origin git@github.com:YOUR_ACCOUNT/radar-imu-hgr.git
git push -u origin main
```

No remote action is performed by this package. Never put credentials in scripts or READMEs. Do not upload the private companion archive.

Environment exports are ignored until reviewed. Copy only reviewed/sanitized records into a tracked directory such as `environment/records/<date>/` and then commit them. Exporting the current environment does not prove its historical identity.

Official references:
- https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
- https://git-scm.com/docs/gitignore
