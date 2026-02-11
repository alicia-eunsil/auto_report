"""월말 수집기 스텁.

실제 구현 시 Playwright로 아래 절차를 자동화하세요.
1) https://stats.gjf.or.kr 접속
2) 좌측 하단 '관계자' 버튼 클릭
3) ID/PW 로그인
4) 분석선택=고용보험, 지역 범위별 데이터 추출
5) 현재/2개월 전 수치 + 신호등 상태를 정규화하여 CSV 저장
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path("data/snapshots")


def build_sample_rows(snapshot_month: str) -> list[dict[str, str | int]]:
    # TODO: 실제 크롤링 데이터로 교체
    now = datetime.now().isoformat(timespec="seconds")
    return [
        {
            "snapshot_month": snapshot_month,
            "region_level": "national",
            "region_name": "전국",
            "indicator": "피보험자수",
            "current_value": 100,
            "current_signal": "정상",
            "prev_2m_value": 98,
            "prev_2m_signal": "정상",
            "collected_at": now,
        },
        {
            "snapshot_month": snapshot_month,
            "region_level": "gyeonggi_city",
            "region_name": "수원시",
            "indicator": "취득자수",
            "current_value": 55,
            "current_signal": "관심",
            "prev_2m_value": 60,
            "prev_2m_signal": "정상",
            "collected_at": now,
        },
    ]


def main() -> None:
    snapshot_month = datetime.now().strftime("%Y-%m")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = build_sample_rows(snapshot_month)
    fieldnames = list(rows[0].keys())
    output_path = OUTPUT_DIR / f"{snapshot_month}.csv"

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved snapshot: {output_path}")


if __name__ == "__main__":
    main()
