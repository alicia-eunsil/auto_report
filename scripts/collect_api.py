"""API-based collector for stats.gjf.or.kr risk indicators.

This collector replaces brittle UI automation with direct API calls.
Output schema matches the existing snapshot CSV contract.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

API_BASE = os.getenv("GJF_API_BASE", "https://stats.gjf.or.kr/forecast/api").rstrip("/")
DATA_DIR = Path("data/snapshots")

EXPECTED_LEVEL_COUNTS = {
    "national": 1,
    "province": 17,
    "gyeonggi_city": 31,
}

TARGET_THR_CODES = ("7", "8", "9", "15", "16", "17")
TARGET_THR_SET = set(TARGET_THR_CODES)
EXPECTED_INDICATOR_COUNT = len(TARGET_THR_CODES)

THR_TO_INDICATOR = {
    "7": "피보험자수",
    "8": "취득자수",
    "9": "상실자수",
    "15": "사업장수",
    "16": "사업장 성립",
    "17": "사업장 소멸",
}

ALLOWED_SIGNALS = {"정상", "관심", "주의"}


@dataclass
class ExtractionContext:
    snapshot_month: str
    collected_at: str


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_snapshot_month(text: str) -> str:
    m = re.search(r"(20\d{2})\D{0,3}(\d{1,2})", text)
    if not m:
        return datetime.now().strftime("%Y-%m")
    year = int(m.group(1))
    month = int(m.group(2))
    return f"{year:04d}-{month:02d}"


def parse_number(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(int(value)) if isinstance(value, int) or float(value).is_integer() else str(value)
    cleaned = str(value).replace(",", "").strip()
    if not cleaned:
        return ""
    m = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    return m.group(0) if m else ""


def normalize_signal(value: Any) -> str:
    token = _clean_text(value).lower()
    if not token:
        return ""
    if "정상" in token or "normal" in token or "green" in token:
        return "정상"
    if "관심" in token or "caution" in token or "warning" in token or "yellow" in token or "orange" in token:
        return "관심"
    if "주의" in token or "danger" in token or "red" in token:
        return "주의"
    return ""


def normalize_thr_code(value: Any) -> str:
    token = _clean_text(value)
    if not token:
        return ""
    m = re.search(r"\d+", token)
    if not m:
        return ""
    try:
        return str(int(m.group(0)))
    except ValueError:
        return ""


def normalize_region_code(value: Any) -> str:
    token = _clean_text(value)
    m = re.search(r"\d+", token)
    return m.group(0) if m else ""


def classify_region_level(region_type: str, region_code: str) -> str:
    if region_type == "nat":
        if region_code == "1000000000":
            return "national"
        if region_code == "3611000000":
            # Sejong special self-governing city code pattern.
            return "province"
        if region_code.endswith("00000000"):
            return "province"
        return ""
    if region_type == "ggd":
        if region_code == "4100000000":
            return ""
        return "gyeonggi_city"
    return ""


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "X-Requested-By": "Frontend",
            "Accept": "application/json, text/plain, */*",
            "User-Agent": "auto-report-collector/1.0",
        }
    )
    return session


def _extract_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("list", "data", "result"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    return []


def api_get_json(session: requests.Session, path: str, params: dict[str, Any] | None = None) -> Any:
    url = f"{API_BASE}{path}"
    res = session.get(url, params=params or {}, timeout=30)
    res.raise_for_status()
    return res.json()


def fetch_last_risk_month(session: requests.Session) -> str:
    payload = api_get_json(session, "/common/risk/last-rc-dy")
    candidate = ""
    if isinstance(payload, dict):
        candidate = _clean_text(payload.get("data"))
    elif isinstance(payload, str):
        candidate = _clean_text(payload)

    m = re.search(r"(20\d{2})(\d{2})", candidate)
    if not m:
        now = datetime.now()
        return f"{now.year:04d}{now.month:02d}"
    return f"{m.group(1)}{m.group(2)}"


def query_indicator_detail(
    session: requests.Session,
    *,
    region_type: str,
    yyyymm: str,
    thr_cd: str | None = None,
    sig_cd: str | None = None,
) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    base = {"regionType": region_type, "date": yyyymm}
    if thr_cd:
        base["thrCd"] = thr_cd
    if sig_cd:
        base["sigCd"] = sig_cd

    variants.append(base)
    variants.append({**base, "standard": "base"})
    variants.append({**base, "ArmDataType": "EMPLOYMENT"})
    variants.append({**base, "ArmDataType": "EMPLOYMENT", "standard": "base"})

    seen: set[tuple[tuple[str, str], ...]] = set()
    last_error: Exception | None = None

    for params in variants:
        key = tuple(sorted((k, str(v)) for k, v in params.items()))
        if key in seen:
            continue
        seen.add(key)
        try:
            payload = api_get_json(session, "/indicator/status/detail", params)
            rows = _extract_list(payload)
            if rows:
                return rows
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    if last_error:
        print(
            "WARN: indicator detail query failed "
            f"regionType={region_type} thrCd={thr_cd or '-'} sigCd={sig_cd or '-'} "
            f"error={last_error}"
        )
    return []


def build_row(raw: dict[str, Any], region_type: str, ctx: ExtractionContext) -> dict[str, str] | None:
    thr_code = normalize_thr_code(raw.get("thrCd"))
    if thr_code not in TARGET_THR_SET:
        return None

    region_code = normalize_region_code(raw.get("regionCd"))
    if not region_code:
        return None

    level = classify_region_level(region_type, region_code)
    if not level:
        return None

    region_name = _clean_text(raw.get("regionNm"))
    if level == "national":
        region_name = "전국"
    if not region_name:
        return None

    indicator = THR_TO_INDICATOR.get(thr_code) or _clean_text(raw.get("thNm")) or thr_code

    row = {
        "snapshot_month": ctx.snapshot_month,
        "region_level": level,
        "region_name": region_name,
        "indicator": indicator,
        "current_value": parse_number(raw.get("daVl")),
        "current_signal": normalize_signal(raw.get("pre1Status")),
        "prev_1m_signal": normalize_signal(raw.get("pre2Status")),
        "prev_2m_value": parse_number(raw.get("pre3DaVl")),
        "prev_2m_signal": normalize_signal(raw.get("pre3Status")),
        "collected_at": ctx.collected_at,
        "_thr_cd": thr_code,
        "_region_cd": region_code,
        "_region_type": region_type,
    }
    return row


def dedupe_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    unique: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["region_level"], row["_region_cd"], row["_thr_cd"])
        unique[key] = row
    return list(unique.values())


def sort_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    level_order = {"national": 0, "province": 1, "gyeonggi_city": 2}
    indicator_order = {THR_TO_INDICATOR[code]: idx for idx, code in enumerate(TARGET_THR_CODES)}
    return sorted(
        rows,
        key=lambda r: (
            level_order.get(r["region_level"], 99),
            r["region_name"],
            indicator_order.get(r["indicator"], 99),
        ),
    )


def load_expected_regions(session: requests.Session) -> dict[str, dict[str, str]]:
    expected: dict[str, dict[str, str]] = {
        "national": {"1000000000": "전국"},
        "province": {},
        "gyeonggi_city": {},
    }

    for region_type in ("nat", "ggd"):
        try:
            payload = api_get_json(session, f"/region-{region_type}")
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: region list query failed regionType={region_type} error={exc}")
            continue

        items = _extract_list(payload)
        for item in items:
            code = normalize_region_code(item.get("aaCd") or item.get("regionCd"))
            name = _clean_text(item.get("aaNm") or item.get("regionNm") or item.get("commDtlCdNm"))
            if not code or not name:
                continue
            if region_type == "nat":
                if code == "1000000000":
                    continue
                if code.endswith("00000000"):
                    expected["province"][code] = name
            else:
                if code == "4100000000":
                    continue
                expected["gyeonggi_city"][code] = name

    return expected


def quality_report(
    rows: list[dict[str, str]],
    expected_regions: dict[str, dict[str, str]],
) -> tuple[list[str], list[tuple[str, str, str, str]]]:
    level_regions: dict[str, set[str]] = {k: set() for k in EXPECTED_LEVEL_COUNTS}
    by_region: dict[tuple[str, str], set[str]] = {}
    observed_name_by_code: dict[tuple[str, str], str] = {}

    for row in rows:
        level = row["region_level"]
        name = row["region_name"]
        code = row["_region_cd"]
        thr_code = row["_thr_cd"]
        if level in level_regions:
            level_regions[level].add(code)
        by_region.setdefault((level, code), set()).add(thr_code)
        observed_name_by_code[(level, code)] = name

    level_mismatches: list[str] = []
    for level, expected in EXPECTED_LEVEL_COUNTS.items():
        actual = len(level_regions[level])
        if actual != expected:
            level_mismatches.append(f"{level}: expected_regions={expected}, actual={actual}")

    missing_combos: list[tuple[str, str, str, str]] = []
    for level in ("national", "province", "gyeonggi_city"):
        expected_map = expected_regions.get(level, {})
        if expected_map:
            codes = sorted(expected_map.keys())
        else:
            codes = sorted({code for (lv, code) in by_region.keys() if lv == level})

        for code in codes:
            name = expected_map.get(code) or observed_name_by_code.get((level, code)) or code
            thr_codes = by_region.get((level, code), set())
            for expected_thr in TARGET_THR_CODES:
                if expected_thr not in thr_codes:
                    missing_combos.append((level, code, name, expected_thr))

    return level_mismatches, missing_combos


def attempt_backfill(
    session: requests.Session,
    yyyymm: str,
    ctx: ExtractionContext,
    rows: list[dict[str, str]],
    expected_regions: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    _, missing_combos = quality_report(rows, expected_regions)
    if not missing_combos:
        return rows

    cache: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    appended = 0

    for level, region_code, region_name, thr_code in missing_combos:
        region_type = "ggd" if level == "gyeonggi_city" else "nat"
        qkey = (region_type, thr_code, region_code)
        if qkey not in cache:
            cache[qkey] = query_indicator_detail(
                session,
                region_type=region_type,
                yyyymm=yyyymm,
                thr_cd=thr_code,
                sig_cd=region_code,
            )

        raw_rows = cache[qkey]
        picked: dict[str, str] | None = None
        for raw in raw_rows:
            row = build_row(raw, region_type, ctx)
            if not row:
                continue
            if (
                row["region_level"] == level
                and row["_region_cd"] == region_code
                and row["_thr_cd"] == thr_code
            ):
                picked = row
                break

        if picked:
            rows.append(picked)
            appended += 1

    if appended:
        print(f"INFO: backfilled rows={appended}")
    return rows


def ensure_quality(
    rows: list[dict[str, str]],
    expected_regions: dict[str, dict[str, str]],
    *,
    allow_partial: bool,
) -> None:
    if not rows:
        raise RuntimeError("No rows collected from API")

    invalid_signals = [
        f"{r['region_level']}/{r['region_name']}/{r['indicator']}"
        for r in rows
        if (
            r["current_signal"] not in ALLOWED_SIGNALS
            or r["prev_1m_signal"] not in ALLOWED_SIGNALS
            or r["prev_2m_signal"] not in ALLOWED_SIGNALS
        )
    ]

    level_mismatches, missing_combos = quality_report(rows, expected_regions)
    incomplete_regions: dict[tuple[str, str], int] = {}
    for level, _code, name, _thr in missing_combos:
        incomplete_regions[(level, name)] = incomplete_regions.get((level, name), 0) + 1

    errors: list[str] = []
    if invalid_signals:
        errors.append(
            "invalid signal rows: " + ", ".join(invalid_signals[:10])
            + ("" if len(invalid_signals) <= 10 else " ...")
        )
    if level_mismatches:
        errors.append("region count mismatch: " + "; ".join(level_mismatches))
    if incomplete_regions:
        preview = "; ".join(
            f"{level}/{name}: missing={missing}"
            for (level, name), missing in list(sorted(incomplete_regions.items()))[:20]
        )
        errors.append("indicator coverage mismatch: " + preview)

    if not errors:
        return

    message = " | ".join(errors)
    if allow_partial:
        print("WARN: " + message)
    else:
        raise RuntimeError(message)


def save_rows(rows: list[dict[str, str]]) -> Path:
    if not rows:
        raise RuntimeError("No rows to save")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_month = rows[0]["snapshot_month"]
    output = DATA_DIR / f"{snapshot_month}.csv"

    field_order = [
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
    ]

    normalized = [{k: row.get(k, "") for k in field_order} for row in rows]
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        writer.writeheader()
        writer.writerows(normalized)
    return output


def guard_snapshot_month(snapshot_month: str) -> bool:
    existing = sorted(DATA_DIR.glob("*.csv"))
    if not existing:
        return True

    latest_existing = existing[-1].stem
    if snapshot_month < latest_existing:
        print(
            "INFO: skip saving snapshot because source month is older than existing latest. "
            f"collected={snapshot_month}, existing_latest={latest_existing}"
        )
        return False
    return True


def collect_rows(session: requests.Session, yyyymm: str, ctx: ExtractionContext) -> list[dict[str, str]]:
    all_rows: list[dict[str, str]] = []
    for region_type in ("nat", "ggd"):
        raw_rows = query_indicator_detail(session, region_type=region_type, yyyymm=yyyymm)
        if not raw_rows:
            print(f"WARN: no rows returned for regionType={region_type}")
            continue
        for raw in raw_rows:
            row = build_row(raw, region_type, ctx)
            if row:
                all_rows.append(row)
    return all_rows


def run() -> None:
    allow_partial = os.getenv("ALLOW_PARTIAL_SNAPSHOT", "0") == "1"

    session = make_session()
    yyyymm = fetch_last_risk_month(session)
    expected_regions = load_expected_regions(session)
    snapshot_month = parse_snapshot_month(yyyymm)
    ctx = ExtractionContext(
        snapshot_month=snapshot_month,
        collected_at=datetime.now().isoformat(timespec="seconds"),
    )

    rows = collect_rows(session, yyyymm, ctx)
    rows = dedupe_rows(rows)
    rows = attempt_backfill(session, yyyymm, ctx, rows, expected_regions)
    rows = dedupe_rows(rows)
    rows = sort_rows(rows)

    ensure_quality(rows, expected_regions, allow_partial=allow_partial)
    if not guard_snapshot_month(ctx.snapshot_month):
        return
    output = save_rows(rows)
    print(f"Saved snapshot: {output} rows={len(rows)}")


if __name__ == "__main__":
    run()
