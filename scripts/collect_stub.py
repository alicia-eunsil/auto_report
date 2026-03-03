"""Fallback stub collector.

This script is only used when COLLECTOR_SCRIPT is not provided.
It writes a tiny snapshot CSV so local smoke tests can proceed.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path("data/snapshots")


def build_sample_rows(snapshot_month: str) -> list[dict[str, str]]:
    now = datetime.now().isoformat(timespec="seconds")
    return [
        {
            "snapshot_month": snapshot_month,
            "region_level": "national",
            "region_name": "전국",
            "indicator": "피보험자수",
            "current_value": "100",
            "current_signal": "정상",
            "prev_1m_signal": "정상",
            "prev_2m_value": "98",
            "prev_2m_signal": "정상",
            "collected_at": now,
        }
    ]


def main() -> None:
    snapshot_month = datetime.now().strftime("%Y-%m")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_sample_rows(snapshot_month)
    output_path = OUTPUT_DIR / f"{snapshot_month}.csv"

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved snapshot: {output_path}")


if __name__ == "__main__":
    main()
