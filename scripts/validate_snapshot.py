"""Validate the latest monthly snapshot CSV schema and basic content."""

from __future__ import annotations

import csv
import os
from pathlib import Path

REQUIRED_COLUMNS = {
    "snapshot_month",
    "region_level",
    "region_name",
    "indicator",
    "current_value",
    "current_signal",
    "prev_1m_signal",
    "prev_2m_value",
    "prev_2m_signal",
    "collected_at",
}

ALLOWED_SIGNALS = {"정상", "관심", "주의"}
STRICT_FLAG_ENV = "REQUIRE_FULL_COVERAGE"

EXPECTED_LEVEL_COUNTS = {
    "national": 1,
    "province": 17,
    "gyeonggi_city": 31,
}
EXPECTED_INDICATOR_COUNT = 6
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
        by_level_region: dict[tuple[str, str], int] = {}

        for idx, row in enumerate(reader, start=2):
            row_count += 1
            if (
                row["current_signal"] not in ALLOWED_SIGNALS
                or row["prev_1m_signal"] not in ALLOWED_SIGNALS
                or row["prev_2m_signal"] not in ALLOWED_SIGNALS
            ):
                invalid_signal_rows.append(idx)

            level = row["region_level"]
            region = row["region_name"]
            by_level_region[(level, region)] = by_level_region.get((level, region), 0) + 1

        if row_count == 0:
            raise SystemExit("snapshot has no data rows")
        if invalid_signal_rows:
            raise SystemExit(f"invalid signal values at rows: {invalid_signal_rows}")

        require_full = os.getenv(STRICT_FLAG_ENV, "0") == "1"

        level_region_names: dict[str, set[str]] = {k: set() for k in EXPECTED_LEVEL_COUNTS}
        incomplete_regions: list[str] = []
        for (level, region), indicator_rows in sorted(by_level_region.items()):
            if level in level_region_names:
                level_region_names[level].add(region)
            if indicator_rows != EXPECTED_INDICATOR_COUNT:
                incomplete_regions.append(
                    f"{level}/{region}: indicators={indicator_rows}"
                )

        level_mismatches = []
        for level, expected in EXPECTED_LEVEL_COUNTS.items():
            actual = len(level_region_names[level])
            if actual != expected:
                level_mismatches.append(f"{level}: expected_regions={expected}, actual={actual}")

        if require_full and level_mismatches:
            raise SystemExit("region count mismatch: " + "; ".join(level_mismatches))
        if require_full and incomplete_regions:
            raise SystemExit("indicator count mismatch: " + "; ".join(incomplete_regions[:20]))

        if not require_full and (level_mismatches or incomplete_regions):
            print(
                "WARN: snapshot is not full coverage yet "
                f"(set {STRICT_FLAG_ENV}=1 to enforce)."
            )

    print(f"OK: {latest} rows={row_count}")


if __name__ == "__main__":
    main()
