# Git 업로드 방법

## 먼저 로컬 보관

Git-ready ZIP을 기존 코드 폴더와 다른 위치에 풀고 `radar-imu-hgr`로 이동한다. 원래 `hgr_sci` 디렉터리를 이동하거나 삭제할 필요가 없다.

```bash
conda activate hgr_sci
python tools/hgr.py verify
python tools/scan_before_publish.py
```

초기 저장소는 비공개로 만들고, 공개 전 권리·개인정보를 검토하는 것을 권한다. Source 안에 과거 사용자명/절대경로가 남아 있고 일부 scripts는 제3자 example에서 유래한다. 비공개 보관 ZIP 자체는 GitHub에 올리지 않는다.

## 로컬 Git 생성과 확인

```bash
git init -b main
git add .
git status --short
git diff --cached --stat
```

staged 목록에 `private_archive/`, `artifacts/`, `workspaces/`, `paths.local.json`, `paths.env`, 연구 데이터, `.pt/.pth/.npz`가 없는지 확인한다. `.gitignore`는 이미 추적 중인 파일을 자동으로 기록에서 제거하지 않으므로, 이 안내는 새로운 저장소 기준이다.

```bash
git commit -m "Archive radar-IMU HGR code and bilingual documentation"
```

GitHub에서 **빈 비공개 repository**를 만든다. 이미 로컬 README/.gitignore가 있으므로 원격 생성 시 README/license/.gitignore 초기화를 추가하지 않는 편이 간단하다. 아래 URL은 본인 저장소의 SSH 주소로 바꾼다.

```bash
git remote add origin git@github.com:YOUR_ACCOUNT/radar-imu-hgr.git
git push -u origin main
```

SSH를 사용하지 않으면 GitHub가 표시하는 인증된 원격 URL/설정을 사용한다. 비밀번호·토큰을 코드나 README에 넣지 않는다. 이 패키지에서는 원격 생성/업로드를 자동 실행하지 않았다.

## 환경 파일은 확인 후 추가

`python tools/export_environment.py` 출력은 `environment/exports/`에 있으며 기본적으로 무시된다. 내용을 확인·정리한 기록만 예를 들어 `environment/records/2026-.../`로 복사하고 commit한다. 환경 export에 남은 local package 경로와 인증정보를 확인한다. 버전을 임의로 추정해서 lockfile을 만들지 않는다.

## 공식 참고

- GitHub 로컬 코드 업로드: https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
- Git ignore 동작: https://git-scm.com/docs/gitignore
