"""Validate the latest monthly snapshot CSV schema and basic content."""

from __future__ import annotations

import csv
from pathlib import Path

REQUIRED_COLUMNS = {
    "snapshot_month",
    "region_level",
    "region_name",
    "indicator",
    "current_value",
    "current_signal",
    "prev_2m_value",
    "prev_2m_signal",
    "collected_at",
}

ALLOWED_SIGNALS = {"정상", "관심", "위기"}
DATA_DIR = Path("data/snapshots")


def main() -> None:
    if not DATA_DIR.exists():
        raise SystemExit("data/snapshots directory does not exist")

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise SystemExit("no snapshot CSV files found")

    latest = csv_files[-1]
    with latest.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise SystemExit(f"missing required columns: {sorted(missing)}")

        row_count = 0
        invalid_signal_rows: list[int] = []
        for idx, row in enumerate(reader, start=2):
            row_count += 1
            if row["current_signal"] not in ALLOWED_SIGNALS:
                invalid_signal_rows.append(idx)

        if row_count == 0:
            raise SystemExit("snapshot has no data rows")
        if invalid_signal_rows:
            raise SystemExit(f"invalid signal values at rows: {invalid_signal_rows}")

    print(f"OK: {latest} rows={row_count}")


if __name__ == "__main__":
    main()
