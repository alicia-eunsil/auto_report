# auto_report

조기경보서비스(고용보험 데이터) 월간 수집/검증/보고를 자동화하는 프로젝트입니다.

현재 Streamlit 보고서 타이틀:
- `조기경보서비스 월간 리포트(based on 고용보험 데이터)`

## 주요 기능

1. Playwright 수집기(`scripts/collect_playwright.py`)
- 관계자 로그인 후 조기경보 서비스 화면에서 6개 지표 수집
- 범위: 전국, 17개 시도, 경기도 31개 시군
- 결과: `data/snapshots/YYYY-MM.csv`

2. 품질 검증(`scripts/validate_snapshot.py`)
- 필수 컬럼/신호값/권역별 지역 수/지표 수 검증
- 엄격 모드(`REQUIRE_FULL_COVERAGE=1`) 지원

3. 마스터/리포트 생성(`scripts/postprocess_data.py`)
- 월별 스냅샷 누적(`data/master/employment_master.csv/.xlsx`)
- 월간 보고서 생성(`reports/YYYY-MM/report_YYYY-MM.xlsx`)

4. Streamlit 대시보드(`app.py`)
- 공통 상단: 월간 한눈에, 핵심요약, 지표별 현황(전국/경기도)
- 탭 구성:
  - `현황형 리포트`: 지표별 월간 악화건수 추이, 2개월 연속/반전 요약, 지역 월별 위험 히트맵
  - `지수형 리포트`: 우선관리 지역, 장기취약 지역

## 프로젝트 구조

- `app.py`: Streamlit 앱
- `config/selectors.json`: 실제 운영 셀렉터
- `config/selectors.example.json`: 셀렉터 예시
- `scripts/collect_playwright.py`: 실수집기
- `scripts/run_if_month_end.sh`: 월말 게이트 실행 스크립트
- `scripts/validate_snapshot.py`: 스냅샷 검증
- `scripts/postprocess_data.py`: 마스터/리포트 생성
- `.github/workflows/monthly_collect.yml`: GitHub Actions 워크플로
- `docs/github_streamlit_runbook.md`: 운영/배포/장애 대응 가이드

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 로컬 수집 테스트

```bash
COLLECTOR_SCRIPT=scripts/collect_playwright.py FORCE_MONTH_END=1 bash scripts/run_if_month_end.sh
python scripts/validate_snapshot.py
python scripts/postprocess_data.py
```

## GitHub Actions 운영 흐름

워크플로: `.github/workflows/monthly_collect.yml`

1. 매일 스케줄 실행
2. `run_if_month_end.sh`에서 월말 여부 판단
3. 월말(또는 `force_month_end=true`)이면 수집 실행
4. 검증 통과 후 마스터/리포트 생성
5. 변경 파일(`data/snapshots`, `data/master`, `reports`) 자동 커밋/푸시

## 운영 체크 문서

- `docs/github_streamlit_runbook.md`를 기준 운영 문서로 사용하세요.
- 권장 운영: 월말 본수집 + 익월초 재시도 정책 + 실패 알림 체크리스트 적용.
