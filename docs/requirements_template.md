# 요구사항 정리 템플릿 (auto_report)

아래 항목을 채우면 바로 개발 가능한 수준의 명세가 됩니다.

## 1) 대상 서비스/접근
- 서비스 URL: `https://stats.gjf.or.kr`
- 로그인 진입: 좌측 하단 `관계자` 버튼
- 인증 방식: ID/PW
- 세션 유지시간: 약 30분
- 자동화 차단 요소(CAPTCHA/2FA): (있음/없음)

## 2) 수집 범위
- 분석선택: `고용보험`
- 지표(6개):
  - 피보험자수
  - 취득자수
  - 상실자수
  - 사업장수
  - 사업장 성립
  - 사업장 소멸
- 지역 범위:
  - 전국
  - 17개 시도
  - 경기도 31개 시군
- 시점:
  - 현재
  - 2개월 전
- 상태값:
  - 정상(녹색)
  - 관심(노랑)
  - 위기(빨강)

## 3) 산출물(데이터)
- 포맷: CSV + JSON(선택)
- 파일 경로: `data/snapshots/YYYY-MM.csv`
- 최소 컬럼 제안:
  - `snapshot_month`
  - `region_level` (national/province/gyeonggi_city)
  - `region_name`
  - `indicator`
  - `current_value`
  - `current_signal`
  - `prev_2m_value`
  - `prev_2m_signal`
  - `collected_at`

## 4) 산출물(보고서)
- 플랫폼: Streamlit Cloud
- 화면 구성:
  1. 월 기준 요약(신호등 개수)
  2. 지역단위 필터(전국/시도/시군)
  3. 지표별 테이블(현재 vs 2개월 전)
  4. 위기/관심 지역 우선 정렬
- 다운로드:
  - CSV 다운로드 버튼

## 5) 자동 실행/운영
- 실행 빈도: 월말 1회
- 실행 위치: GitHub Actions
- 실패 정책: 최대 2회 재시도(워크플로 레벨)
- 알림: (선택) 실패 시 이메일/슬랙

## 6) 보안/권한
- 개인정보: 없음
- 비밀번호 저장: GitHub/Streamlit Secrets
- 저장소 공개 범위: 개인 사용(권장 private)

## 7) 완료 정의(Definition of Done)
- [ ] 월말 자동 수집이 정상 동작한다.
- [ ] 최소 최근 3개월 데이터가 누적 저장된다.
- [ ] Streamlit에서 최신월 보고서가 렌더링된다.
- [ ] 전국/17개 시도/경기 31개 시군 필터가 동작한다.
- [ ] 지표 6개의 현재/2개월 전 + 신호등 표시가 확인된다.
