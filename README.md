# auto_report

경기도 고용보험 조기경보 지표를 월말 기준으로 수집하고, Streamlit Cloud에서 보고서를 확인하기 위한 프로젝트 템플릿입니다.

## 가능한가요?
가능합니다. 아래 구조로 구현하면 됩니다.

1. **수집(Playwright)**: `stats.gjf.or.kr` 로그인 후 데이터 수집
2. **저장(GitHub Repo)**: CSV/JSON 파일로 이력 저장
3. **리포트(Streamlit)**: 전국/17개 시도/경기도 31개 시군 현황 + 2개월 전 대비
4. **자동 실행(GitHub Actions)**: 월말 1회 자동 수집

## 이번 보강 내용 (중요)
초기 스캐폴드만으로는 부족하다는 피드백을 반영해 아래 문서를 추가했습니다.

- `docs/scraping_design.md`: 실제 접속/로그인/수집/파싱/오류처리/보고서 구조 설계서
- `docs/client_questions.md`: 구현 전 반드시 확인해야 하는 질문서

즉, 단순 "가능합니다" 수준이 아니라 **어떻게 접속해서, 어떤 규칙으로 데이터를 뽑고, 보고서 형태를 어떻게 확정할지**를 문서화했습니다.

## 프로젝트 구조

- `app.py`: Streamlit 보고서 화면
- `scripts/collect_stub.py`: 로컬 데모용 수집기 스텁
- `scripts/collect_playwright.py`: 관계자 로그인 기반 Playwright 수집기 골격
- `config/selectors.example.json`: 사이트 셀렉터 예시 파일
- `scripts/run_if_month_end.sh`: 월말일 때만 수집기 실행
- `.github/workflows/monthly_collect.yml`: 주기 실행 워크플로
- `docs/requirements_template.md`: 요구사항 정리 템플릿
- `docs/scraping_design.md`: 수집 상세 설계
- `docs/client_questions.md`: 사전 질의서

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```


## 실행 확인(로컬 데모)

아래 명령으로 "수집 → 검증 → 월말게이트 강제실행"까지 한 번에 확인할 수 있습니다.

```bash
bash scripts/demo_run.sh
```

실행 로그는 `artifacts/demo_run.log`에 저장됩니다.


## Playwright 실수집기 설정

1. `config/selectors.example.json`을 복사해 `config/selectors.json` 생성
2. 실제 화면 DOM에 맞게 셀렉터 수정
3. 환경변수 설정
   - `GJF_USER`
   - `GJF_PASSWORD`
4. 실행

```bash
COLLECTOR_SCRIPT=scripts/collect_playwright.py FORCE_MONTH_END=1 bash scripts/run_if_month_end.sh
```

위 명령의 뜻은 아래와 같습니다.

- `COLLECTOR_SCRIPT=scripts/collect_playwright.py`: 스텁이 아니라 Playwright 실수집기를 실행
- `FORCE_MONTH_END=1`: 오늘이 월말이 아니어도 강제로 실행
- `bash scripts/run_if_month_end.sh`: 월말 판단 + 수집 실행 래퍼 스크립트

즉, 로컬에서 "월말 기다리지 않고 지금 바로 수집 테스트"할 때 쓰는 명령입니다.

실수집기가 6개 지표 × 49개 지역을 모두 수집하지 못하면 실패하도록 되어 있습니다.
(강제 실행했는데 데이터가 비는 경우 대부분 셀렉터 불일치 이슈입니다.)

- `artifacts/collector_failure.png` 화면을 먼저 확인
- `config/selectors.json`의 지역/지표 관련 셀렉터를 실제 DOM에 맞게 수정
- 지역 이름은 `region_names.province`, `region_names.gyeonggi_city`에 명시하면 더 안정적


## GitHub에 올려서 실제 실행하기

"실제로 GitHub/Streamlit에서 돌려보는" 절차는 아래 문서로 바로 따라가면 됩니다.

- `docs/github_streamlit_runbook.md`

특히 **파일이 저장/업로드되지 않을 때 확인할 명령(`git status`, `git remote -v`)과 `git add/commit/push` 순서**를 문서 맨 앞에 추가해 두었습니다.
핵심은 ① GitHub에 commit/push 완료, ② GitHub Secrets(`GJF_USER`, `GJF_PASSWORD`) 등록, ③ `config/selectors.json` 실제 DOM 반영, ④ Actions 수동 실행 확인입니다.

## Streamlit Cloud 배포

1. 이 저장소를 GitHub에 push
2. Streamlit Cloud에서 repo 연결
3. Secrets 설정
   - `GJF_USER`: 관계자 계정 ID
   - `GJF_PASSWORD`: 관계자 계정 비밀번호
4. 메인 파일: `app.py`

## 월말 자동 수집

GitHub Actions가 매일 실행되며, `scripts/run_if_month_end.sh`가 월말 여부를 판단해 월말에만 실제 수집기를 실행합니다.

`run_if_month_end.sh`는 `COLLECTOR_SCRIPT` 환경변수로 실행 대상을 바꿀 수 있습니다. 기본값은 `scripts/collect_stub.py`이며, 운영에서는 `scripts/collect_playwright.py`를 사용합니다.

수동 실행 시에도 강제 테스트가 가능합니다.

1. GitHub → `Actions` → `Monthly Collect` → `Run workflow`
2. `force_month_end`를 `true`로 선택
3. 실행

이렇게 하면 월말이 아니어도 수집 단계를 강제로 실행해 동작 여부를 확인할 수 있습니다.

## 다음 구현 단계

1. `docs/client_questions.md` 답변 확정
2. `scripts/collect_stub.py`를 Playwright 실수집기로 교체
3. 지표 6개 x 지역단위(전국/17개 시도/경기도 31개 시군) 추출
4. 현재/2개월 전 수치 + 신호등 상태(정상/관심/주의) 정규화
5. `data/snapshots/YYYY-MM.csv` 저장
6. Streamlit에서 최신/전월 비교 보고서 시각화
