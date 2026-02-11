# GitHub + Streamlit Cloud 실제 실행 가이드

요청하신 "실제로 돌려보기" 기준으로, **파일을 GitHub에 올리는 것부터** 최소 단계로 정리했습니다.

## 0) 준비물
- GitHub 계정
- Streamlit Cloud 계정(같은 GitHub 계정으로 로그인 권장)
- `stats.gjf.or.kr` 관계자 ID/PW

---

## 1) "파일이 저장 안 될 때" 먼저 확인

프로젝트 폴더에서 아래를 먼저 실행하세요.

```bash
git status
git branch --show-current
git remote -v
```

- `git status`에 변경 파일이 보이면: 아직 커밋 전 상태입니다.
- `git remote -v`가 비어 있으면: GitHub 원격 저장소 연결이 안 된 상태입니다.

---

## 2) GitHub 저장소에 코드 올리기 (CLI 방법)

### 2-1. 원격 저장소 연결

```bash
git remote add origin <YOUR_GITHUB_REPO_URL>
```

이미 origin이 있으면 URL만 교체:

```bash
git remote set-url origin <YOUR_GITHUB_REPO_URL>
```

### 2-2. 파일 저장(커밋)

```bash
git add .
git commit -m "chore: save project files"
```

### 2-3. GitHub로 업로드(push)

현재 브랜치 확인:

```bash
git branch --show-current
```

예: 브랜치가 `work`면

```bash
git push -u origin work
```

예: 브랜치가 `main`이면

```bash
git push -u origin main
```

> push 후 GitHub 웹에서 파일 목록이 보이면 정상입니다.

---

## 3) CLI가 어렵다면: GitHub 웹 업로드 방법

1. GitHub에서 새 저장소 생성
2. 저장소 페이지에서 **Add file → Upload files**
3. 프로젝트 파일 드래그&드롭
4. Commit message 입력 후 **Commit changes**

> 빠르게 확인할 때는 웹 업로드가 가장 단순합니다.

---

## 4) GitHub Secrets 설정 (중요)

GitHub 저장소 → **Settings → Secrets and variables → Actions → New repository secret**

아래 2개를 추가:
- `GJF_USER` = 관계자 ID
- `GJF_PASSWORD` = 관계자 비밀번호

---

## 5) Streamlit Cloud 배포

1. Streamlit Cloud 접속 → **New app**
2. 방금 push한 GitHub repo 선택
3. Branch: 배포할 브랜치(`work` 또는 `main`)
4. Main file path: `app.py`
5. Deploy

배포 후 앱이 열리면 `data/snapshots/*.csv` 기준으로 화면이 표시됩니다.

---

## 6) GitHub Actions 수집 실행(수동 테스트)

자동은 월말 1회(내부 게이트)지만, 지금 당장 테스트하려면 수동 실행:

1. GitHub 저장소 → **Actions**
2. `Monthly Collect` 워크플로 선택
3. **Run workflow** 클릭

워크플로는 Playwright 브라우저를 설치하고,
`COLLECTOR_SCRIPT=scripts/collect_playwright.py`로 수집을 시도합니다.

---

## 7) 실패했을 때 확인 순서

### A. Actions 로그 확인
- `Run collector on month-end`
- `Validate latest snapshot`

### B. 셀렉터 문제 가능성
사이트 DOM이 다르면 `config/selectors.example.json` 값을 실제 값으로 바꿔
`config/selectors.json`을 만들어야 합니다.

예시:
```bash
cp config/selectors.example.json config/selectors.json
# 이후 selectors.json 값을 실제 DOM으로 수정
```

### C. 로그인 성공 판정 요소
`login_success_anchor`가 실제 로그인 후 보이는 요소인지 확인하세요.

---

## 8) "진짜 수집되었는지" 확인 포인트

1. GitHub 저장소의 `data/snapshots/YYYY-MM.csv` 파일 생성/갱신 여부
2. Streamlit 앱에서 최신 월 데이터가 테이블로 노출되는지
3. Actions 로그에서 `Validate latest snapshot` 성공 여부

---

## 9) 가장 빠른 실전 점검 루트 (권장)

1. GitHub Secrets 2개 설정
2. `config/selectors.json` 실제 DOM에 맞게 수정 후 push
3. Actions에서 `Monthly Collect` 수동 실행
4. `data/snapshots` 갱신 확인
5. Streamlit 앱 새로고침

이 5단계를 통과하면, 월말 자동수집 + 보고서 조회 흐름이 완성됩니다.
