from __future__ import annotations

from pathlib import Path

import pandas as pd

SNAPSHOT_DIR = Path("data/snapshots")
MASTER_DIR = Path("data/master")
REPORT_DIR = Path("reports")

MASTER_CSV = MASTER_DIR / "employment_master.csv"
MASTER_XLSX = MASTER_DIR / "employment_master.xlsx"

KEY_COLS = ["snapshot_month", "region_level", "region_name", "indicator"]
SIGNAL_COLS = ["current_signal", "prev_1m_signal", "prev_2m_signal"]


def _latest_snapshot() -> Path:
    files = sorted(SNAPSHOT_DIR.glob("*.csv"))
    if not files:
        raise SystemExit("No snapshot CSV found in data/snapshots")
    return files[-1]


def _count_signals(group: pd.DataFrame, signal_col: str) -> pd.Series:
    counts = group[signal_col].value_counts()
    return pd.Series(
        {
            "정상": int(counts.get("정상", 0)),
            "관심": int(counts.get("관심", 0)),
            "주의": int(counts.get("주의", 0)),
        }
    )


def _summary_by_scope(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for scope_name, scoped in {
        "전국": df[df["region_level"] == "national"],
        "시도": df[df["region_level"] == "province"],
        "시군": df[df["region_level"] == "gyeonggi_city"],
    }.items():
        for indicator, g in scoped.groupby("indicator"):
            row = {
                "scope": scope_name,
                "indicator": indicator,
            }
            row.update(_count_signals(g, "current_signal").to_dict())
            records.append(row)
    out = pd.DataFrame(records)
    if out.empty:
        return out
    return out.sort_values(["scope", "indicator"]).reset_index(drop=True)


def _summary_with_mom_delta(master_df: pd.DataFrame, snapshot_month: str) -> pd.DataFrame:
    cur = master_df[master_df["snapshot_month"] == snapshot_month]
    months = sorted(master_df["snapshot_month"].dropna().unique().tolist())
    prev_month = months[-2] if len(months) >= 2 else None

    current = (
        cur[cur["region_level"].isin(["province", "gyeonggi_city"])]
        .groupby(["region_level", "indicator"])  # type: ignore[arg-type]
        .apply(lambda g: _count_signals(g, "current_signal"), include_groups=False)
        .reset_index()
        .rename(columns={"region_level": "scope_level"})
    )

    if prev_month is None:
        current["전월_정상_증감"] = 0
        current["전월_관심_증감"] = 0
        current["전월_주의_증감"] = 0
        return current

    prev = (
        master_df[(master_df["snapshot_month"] == prev_month) & (master_df["region_level"].isin(["province", "gyeonggi_city"]))]
        .groupby(["region_level", "indicator"])  # type: ignore[arg-type]
        .apply(lambda g: _count_signals(g, "current_signal"), include_groups=False)
        .reset_index()
        .rename(columns={"region_level": "scope_level"})
    )

    merged = current.merge(prev, on=["scope_level", "indicator"], how="left", suffixes=("", "_prev")).fillna(0)
    merged["전월_정상_증감"] = merged["정상"] - merged["정상_prev"]
    merged["전월_관심_증감"] = merged["관심"] - merged["관심_prev"]
    merged["전월_주의_증감"] = merged["주의"] - merged["주의_prev"]
    return merged[["scope_level", "indicator", "정상", "관심", "주의", "전월_정상_증감", "전월_관심_증감", "전월_주의_증감"]]


def _three_month_non_normal(df: pd.DataFrame, level: str) -> pd.DataFrame:
    scope = df[df["region_level"] == level].copy()
    if scope.empty:
        return pd.DataFrame(columns=["indicator", "region_name", "m0", "m1", "m2"])

    for c in SIGNAL_COLS:
        scope[c] = scope[c].fillna("")

    mask = scope[SIGNAL_COLS].apply(lambda s: s.isin(["관심", "주의"])).all(axis=1)
    out = scope.loc[mask, ["indicator", "region_name", "current_signal", "prev_1m_signal", "prev_2m_signal"]]
    return out.rename(
        columns={
            "current_signal": "m0",
            "prev_1m_signal": "m1",
            "prev_2m_signal": "m2",
        }
    ).sort_values(["indicator", "region_name"]).reset_index(drop=True)


def _top_regions(df: pd.DataFrame, level: str, top_n: int) -> pd.DataFrame:
    scope = df[df["region_level"] == level].copy()
    if scope.empty:
        return pd.DataFrame(columns=["indicator", "region_name", "주의개수", "관심개수", "비정상개수"])

    for c in SIGNAL_COLS:
        scope[c] = scope[c].fillna("")

    score = pd.DataFrame(
        {
            "주의개수": (scope[SIGNAL_COLS] == "주의").sum(axis=1),
            "관심개수": (scope[SIGNAL_COLS] == "관심").sum(axis=1),
        }
    )
    scope = pd.concat([scope.reset_index(drop=True), score], axis=1)
    scope["비정상개수"] = scope["주의개수"] + scope["관심개수"]

    rows: list[pd.DataFrame] = []
    for indicator, g in scope.groupby("indicator"):
        picked = g.sort_values(["주의개수", "관심개수", "region_name"], ascending=[False, False, True]).head(top_n)
        rows.append(picked[["indicator", "region_name", "주의개수", "관심개수", "비정상개수"]])

    if not rows:
        return pd.DataFrame(columns=["indicator", "region_name", "주의개수", "관심개수", "비정상개수"])
    return pd.concat(rows, ignore_index=True)


def update_master(latest_path: Path) -> pd.DataFrame:
    latest = pd.read_csv(latest_path)
    required = {
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
    missing = required - set(latest.columns)
    if missing:
        raise SystemExit(f"latest snapshot missing required columns: {sorted(missing)}")

    MASTER_DIR.mkdir(parents=True, exist_ok=True)
    if MASTER_CSV.exists():
        master = pd.read_csv(MASTER_CSV)
    else:
        master = pd.DataFrame(columns=latest.columns)

    month = str(latest["snapshot_month"].max())
    if not master.empty:
        drop_mask = master["snapshot_month"] == month
        master = master.loc[~drop_mask].copy()

    merged = pd.concat([master, latest], ignore_index=True)
    merged = merged.drop_duplicates(subset=KEY_COLS, keep="last").sort_values(["snapshot_month", "region_level", "region_name", "indicator"])

    merged.to_csv(MASTER_CSV, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(MASTER_XLSX, engine="openpyxl") as writer:
        for indicator, g in merged.groupby("indicator"):
            g.to_excel(writer, sheet_name=str(indicator)[:31], index=False)

    return merged


def build_monthly_report(master_df: pd.DataFrame, latest_month: str) -> Path:
    current = master_df[master_df["snapshot_month"] == latest_month].copy()

    report_month_dir = REPORT_DIR / latest_month
    report_month_dir.mkdir(parents=True, exist_ok=True)
    out = report_month_dir / f"report_{latest_month}.xlsx"

    summary_scope = _summary_by_scope(current)
    summary_delta = _summary_with_mom_delta(master_df, latest_month)
    nn_province = _three_month_non_normal(current, "province")
    nn_city = _three_month_non_normal(current, "gyeonggi_city")
    top_province = _top_regions(current, "province", top_n=5)
    top_city = _top_regions(current, "gyeonggi_city", top_n=10)

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        summary_scope.to_excel(writer, sheet_name="요약_권역별", index=False)
        summary_delta.to_excel(writer, sheet_name="요약_전월대비", index=False)
        nn_province.to_excel(writer, sheet_name="3개월연속_시도", index=False)
        nn_city.to_excel(writer, sheet_name="3개월연속_시군", index=False)
        top_province.to_excel(writer, sheet_name="주요지역_시도", index=False)
        top_city.to_excel(writer, sheet_name="주요지역_시군", index=False)

    return out


def main() -> None:
    latest = _latest_snapshot()
    master = update_master(latest)
    latest_month = str(pd.read_csv(latest)["snapshot_month"].max())
    report = build_monthly_report(master, latest_month)
    print(f"Updated master: {MASTER_CSV}")
    print(f"Updated master excel: {MASTER_XLSX}")
    print(f"Created monthly report: {report}")


if __name__ == "__main__":
    main()
