from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

MASTER_CSV = Path("data/master/employment_master.csv")
SNAPSHOT_DIR = Path("data/snapshots")
SIGNAL_COLS = ["current_signal", "prev_1m_signal", "prev_2m_signal"]
SIGNAL_SCORE = {"정상": 0, "관심": 1, "주의": 2}
SCOPE_MAP = {
    "national": "전국",
    "province": "시도",
    "gyeonggi_city": "경기 시군",
}


st.set_page_config(page_title="고용보험 조기경보 월간 리포트", layout="wide")
st.title("고용보험 조기경보 월간 리포트")
st.caption("현황 + 변화 + 지속성 중심 우선관리 리포트")
st.markdown(
    """
    <style>
    table.report-table { width: 100%; border-collapse: collapse; }
    table.report-table th, table.report-table td { text-align: center !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_data() -> pd.DataFrame:
    if MASTER_CSV.exists():
        return pd.read_csv(MASTER_CSV)

    csv_files = sorted(SNAPSHOT_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("데이터 파일이 없습니다. 수집기를 먼저 실행하세요.")
    return pd.read_csv(csv_files[-1])


def enrich_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    for c in SIGNAL_COLS:
        df[c] = df[c].fillna("").astype(str)
        df[f"{c}_score"] = df[c].map(SIGNAL_SCORE).fillna(0).astype(int)

    df["주의개수"] = (df[SIGNAL_COLS] == "주의").sum(axis=1)
    df["관심개수"] = (df[SIGNAL_COLS] == "관심").sum(axis=1)
    df["현재위험점수"] = df["주의개수"] * 2 + df["관심개수"]
    df["연속비정상지표"] = df[SIGNAL_COLS].isin(["관심", "주의"]).all(axis=1).astype(int)
    df["current_signal_score"] = df["current_signal"].map(SIGNAL_SCORE).fillna(0).astype(int)
    return df


def build_region_snapshot(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    if snapshot_df.empty:
        return pd.DataFrame(
            columns=[
                "region_level",
                "region_name",
                "현재위험점수",
                "주의개수",
                "관심개수",
                "연속비정상지표수",
                "지표수",
            ]
        )

    out = (
        snapshot_df.groupby(["region_level", "region_name"], as_index=False)
        .agg(
            현재위험점수=("현재위험점수", "sum"),
            주의개수=("주의개수", "sum"),
            관심개수=("관심개수", "sum"),
            연속비정상지표수=("연속비정상지표", "sum"),
            지표수=("indicator", "nunique"),
        )
        .sort_values(["region_level", "현재위험점수", "region_name"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    return out


def build_region_month_scores(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["snapshot_month", "region_level", "region_name"], as_index=False)["current_signal_score"]
        .sum()
        .rename(columns={"current_signal_score": "월위험점수"})
    )


def build_long_term_scores(region_month_df: pd.DataFrame, months: list[str], selected_month: str) -> pd.DataFrame:
    idx = months.index(selected_month)
    window_months = months[max(0, idx - 11) : idx + 1]
    if not window_months:
        return pd.DataFrame(columns=["region_level", "region_name", "장기취약점수", "장기취약정규점수", "추세변화"])

    window_df = region_month_df[region_month_df["snapshot_month"].isin(window_months)].copy()
    if window_df.empty:
        return pd.DataFrame(columns=["region_level", "region_name", "장기취약점수", "장기취약정규점수", "추세변화"])

    pivot = window_df.pivot_table(
        index=["region_level", "region_name"],
        columns="snapshot_month",
        values="월위험점수",
        aggfunc="sum",
        fill_value=0,
    )
    for m in window_months:
        if m not in pivot.columns:
            pivot[m] = 0
    pivot = pivot[window_months]

    weights: dict[str, float] = {}
    for i, m in enumerate(window_months):
        weights[m] = 1.5 if i >= len(window_months) - 3 else 1.0

    weighted = sum(pivot[m] * weights[m] for m in window_months)
    out = weighted.rename("장기취약점수").reset_index()

    recent3 = window_months[-3:]
    prev3 = window_months[-6:-3]
    recent_avg = pivot[recent3].mean(axis=1)
    if prev3:
        trend_delta = recent_avg - pivot[prev3].mean(axis=1)
    else:
        trend_delta = recent_avg
    out["추세변화"] = trend_delta.values

    min_v = float(out["장기취약점수"].min())
    max_v = float(out["장기취약점수"].max())
    if max_v > min_v:
        out["장기취약정규점수"] = ((out["장기취약점수"] - min_v) / (max_v - min_v) * 100).round(1)
    else:
        out["장기취약정규점수"] = 0.0
    return out


def build_priority_table(
    current_region: pd.DataFrame,
    prev_region: pd.DataFrame,
    long_term: pd.DataFrame,
) -> pd.DataFrame:
    prev = prev_region[["region_level", "region_name", "현재위험점수"]].rename(columns={"현재위험점수": "전월위험점수"})
    merged = current_region.merge(prev, on=["region_level", "region_name"], how="left").fillna({"전월위험점수": 0})
    merged["변화점수"] = merged["현재위험점수"] - merged["전월위험점수"]

    merged = merged.merge(
        long_term[["region_level", "region_name", "장기취약점수", "장기취약정규점수", "추세변화"]],
        on=["region_level", "region_name"],
        how="left",
    ).fillna({"장기취약점수": 0, "장기취약정규점수": 0, "추세변화": 0})

    merged["우선순위점수"] = (
        merged["현재위험점수"] * 0.5 + merged["변화점수"] * 0.3 + merged["장기취약정규점수"] * 0.2
    ).round(2)
    merged["권역"] = merged["region_level"].map(SCOPE_MAP).fillna(merged["region_level"])
    merged["추세"] = merged["추세변화"].apply(lambda x: "악화" if x >= 1 else ("개선" if x <= -1 else "유지"))
    return merged.sort_values(["우선순위점수", "현재위험점수"], ascending=[False, False]).reset_index(drop=True)


def build_indicator_flow(current_df: pd.DataFrame, prev_df: pd.DataFrame) -> pd.DataFrame:
    if current_df.empty or prev_df.empty:
        return pd.DataFrame()

    key_cols = ["region_level", "region_name", "indicator"]
    cur = current_df[key_cols + ["current_signal_score"]].rename(columns={"current_signal_score": "cur"})
    prev = prev_df[key_cols + ["current_signal_score"]].rename(columns={"current_signal_score": "prev"})
    merged = cur.merge(prev, on=key_cols, how="inner")
    merged["diff"] = merged["cur"] - merged["prev"]

    out = (
        merged.groupby("indicator", as_index=False)
        .agg(
            악화건수=("diff", lambda s: int((s > 0).sum())),
            개선건수=("diff", lambda s: int((s < 0).sum())),
            유지건수=("diff", lambda s: int((s == 0).sum())),
        )
        .sort_values("악화건수", ascending=False)
        .reset_index(drop=True)
    )
    return out


def build_export_excel(
    summary_df: pd.DataFrame,
    priority_df: pd.DataFrame,
    long_term_df: pd.DataFrame,
    current_raw_df: pd.DataFrame,
) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="월간요약", index=False)
        priority_df.to_excel(writer, sheet_name="우선관리지역", index=False)
        long_term_df.to_excel(writer, sheet_name="장기취약지역", index=False)
        current_raw_df.to_excel(writer, sheet_name="기준월원본", index=False)
    buffer.seek(0)
    return buffer.getvalue()


try:
    raw = load_data()
except FileNotFoundError as e:
    st.warning(str(e))
    st.stop()

required = {
    "snapshot_month",
    "region_level",
    "region_name",
    "indicator",
    "current_signal",
    "prev_1m_signal",
    "prev_2m_signal",
}
if not required.issubset(raw.columns):
    st.error("필수 컬럼이 누락되었습니다.")
    st.write("누락:", sorted(required - set(raw.columns)))
    st.stop()

df = enrich_features(raw)
months = sorted(df["snapshot_month"].dropna().astype(str).unique().tolist())
if not months:
    st.warning("표시할 기준월 데이터가 없습니다.")
    st.stop()

selected_month = st.selectbox("기준월", months, index=len(months) - 1)
selected_idx = months.index(selected_month)
prev_month = months[selected_idx - 1] if selected_idx > 0 else None

current = df[df["snapshot_month"] == selected_month].copy()
prev = df[df["snapshot_month"] == prev_month].copy() if prev_month else pd.DataFrame(columns=df.columns)

current_region = build_region_snapshot(current)
prev_region = build_region_snapshot(prev)
region_month = build_region_month_scores(df)
long_term = build_long_term_scores(region_month, months, selected_month)
priority = build_priority_table(current_region, prev_region, long_term)
indicator_flow = build_indicator_flow(current, prev)

cur_caution = set(
    zip(
        current.loc[current["current_signal"] == "주의", "region_level"],
        current.loc[current["current_signal"] == "주의", "region_name"],
    )
)
prev1m_caution = set(
    zip(
        current.loc[current["prev_1m_signal"] == "주의", "region_level"],
        current.loc[current["prev_1m_signal"] == "주의", "region_name"],
    )
)

kpi_caution_regions = len(cur_caution)

def is_worsened(prev_sig: str, cur_sig: str) -> bool:
    return (prev_sig == "정상" and cur_sig in {"관심", "주의"}) or (prev_sig == "관심" and cur_sig == "주의")


worsened_rows = current[current.apply(lambda r: is_worsened(str(r["prev_1m_signal"]), str(r["current_signal"])), axis=1)]
worsened_regions = set(zip(worsened_rows["region_level"], worsened_rows["region_name"]))
kpi_worsened_regions = len(worsened_regions)

kpi_persistent_regions = int((priority["연속비정상지표수"] > 0).sum()) if not priority.empty else 0
kpi_new_caution = len(cur_caution - prev1m_caution)

top_target = priority.head(1)
top_target_text = (
    f"{top_target.iloc[0]['권역']} {top_target.iloc[0]['region_name']}" if not top_target.empty else "해당 없음"
)
summary_lines = [
    f"- 이번 달 주의 지역은 **{kpi_caution_regions}개**이며, 신규 주의 지역(전달 주의 0개 → 이번달 주의 1개 이상)은 **{kpi_new_caution}개**입니다.",
    f"- 전월 대비 악화 지역은 **{kpi_worsened_regions}개**, 3개월 연속 비정상 지역은 **{kpi_persistent_regions}개**입니다.",
    f"- 우선 점검 대상 1순위는 **{top_target_text}** 입니다.",
]

st.markdown("## 1. 월간 한눈에")
c1, c2, c3, c4 = st.columns(4)
c1.metric("주의 지역 수", kpi_caution_regions)
c2.metric("전월 대비 악화 지역", kpi_worsened_regions)
c3.metric("3개월 연속 비정상", kpi_persistent_regions)
c4.metric("신규 주의 지역", kpi_new_caution)

st.markdown("### 핵심 요약")
st.markdown("\n".join(summary_lines))

def indicator_count_view(src: pd.DataFrame, levels: list[str]) -> pd.DataFrame:
    out = src[src["region_level"].isin(levels)].copy()
    if out.empty:
        return pd.DataFrame(columns=["지표", "정상", "관심", "주의"])

    out = (
        out.groupby(["indicator", "current_signal"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for col in ["정상", "관심", "주의"]:
        if col not in out.columns:
            out[col] = 0
    out = out.rename(columns={"indicator": "지표"})
    return out[["지표", "정상", "관심", "주의"]].sort_values("지표").reset_index(drop=True)


def render_centered_table(df: pd.DataFrame) -> None:
    st.markdown(df.to_html(index=False, classes="report-table"), unsafe_allow_html=True)


def indicator_count_view_gyeonggi(src: pd.DataFrame) -> pd.DataFrame:
    gg = pd.concat(
        [
            src[(src["region_level"] == "province") & (src["region_name"] == "경기도")],
            src[src["region_level"] == "gyeonggi_city"],
        ],
        ignore_index=True,
    )
    return indicator_count_view(gg, ["province", "gyeonggi_city"])


st.markdown("### 지표별 현황 (전국 / 경기도)")
v1, v2 = st.columns(2)
with v1:
    st.markdown("#### 전국(전국 + 17개 시도)")
    render_centered_table(indicator_count_view(current, ["national", "province"]))
with v2:
    st.markdown("#### 경기도(경기도 + 31개 시군)")
    render_centered_table(indicator_count_view_gyeonggi(current))

st.divider()

st.markdown("## 2. 우선관리 지역")
show_cols = [
    "region_name",
    "우선순위점수",
    "현재위험점수",
    "변화점수",
    "연속비정상지표수",
    "장기취약정규점수",
    "추세",
]
p1, p2 = st.columns(2)
with p1:
    st.markdown("### 시도 Top 10")
    render_centered_table(
        priority[priority["region_level"] == "province"][show_cols].head(10).rename(columns={"region_name": "지역명"})
    )
with p2:
    st.markdown("### 경기 시군 Top 15")
    render_centered_table(
        priority[priority["region_level"] == "gyeonggi_city"][show_cols].head(15).rename(columns={"region_name": "지역명"})
    )

st.divider()

st.markdown("## 3. 장기취약 지역 (최근 12개월)")
long_view = priority[
    [
        "권역",
        "region_name",
        "장기취약점수",
        "장기취약정규점수",
        "추세변화",
        "추세",
        "현재위험점수",
    ]
].rename(columns={"region_name": "지역명"}).sort_values("장기취약점수", ascending=False)
st.dataframe(long_view.head(30), use_container_width=True)

st.divider()

st.markdown("## 4. 확산/개선 흐름")
st.markdown("### 지표별 악화/개선")
if indicator_flow.empty:
    st.info("전월 데이터가 없어 변화 분석이 불가능합니다.")
else:
    st.dataframe(indicator_flow, use_container_width=True)

st.markdown("### 현재 신호 분포")
level_opt = st.selectbox("권역 선택", ["전국", "시도", "경기 시군"], index=1)
level_key = {"전국": "national", "시도": "province", "경기 시군": "gyeonggi_city"}[level_opt]
heat = (
    current[current["region_level"] == level_key]
    .groupby(["indicator", "current_signal"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)
st.dataframe(heat, use_container_width=True)

st.divider()

st.markdown("## 5. 지역 상세")
detail_levels = [k for k in ["province", "gyeonggi_city", "national"] if k in current["region_level"].unique().tolist()]
if not detail_levels:
    st.info("지역 상세를 표시할 데이터가 없습니다.")
else:
    d1, d2 = st.columns(2)
    with d1:
        level_sel = st.selectbox("상세 권역", detail_levels, format_func=lambda x: SCOPE_MAP.get(x, x))
    with d2:
        region_options = sorted(current[current["region_level"] == level_sel]["region_name"].dropna().unique().tolist())
        region_sel = st.selectbox("상세 지역", region_options)

    region_hist = df[(df["region_level"] == level_sel) & (df["region_name"] == region_sel)].copy()
    month_score = (
        region_hist.groupby("snapshot_month", as_index=False)["current_signal_score"]
        .sum()
        .sort_values("snapshot_month")
        .set_index("snapshot_month")
    )
    st.markdown("### 월별 위험점수")
    st.line_chart(month_score)

    signal_timeline = (
        region_hist.pivot_table(
            index="snapshot_month",
            columns="indicator",
            values="current_signal",
            aggfunc="first",
        )
        .sort_index()
        .reset_index()
    )
    st.markdown("### 지표 신호 타임라인")
    st.dataframe(signal_timeline, use_container_width=True)

st.divider()

st.markdown("## 6. 발간용 다운로드")
export_summary = pd.DataFrame(
    [
        {"항목": "기준월", "값": selected_month},
        {"항목": "주의 지역 수", "값": kpi_caution_regions},
        {"항목": "전월 대비 악화 지역", "값": kpi_worsened_regions},
        {"항목": "3개월 연속 비정상", "값": kpi_persistent_regions},
        {"항목": "신규 주의 지역", "값": kpi_new_caution},
    ]
)
priority_export = priority[
    [
        "권역",
        "region_name",
        "우선순위점수",
        "현재위험점수",
        "변화점수",
        "연속비정상지표수",
        "장기취약점수",
        "장기취약정규점수",
        "추세",
    ]
].rename(columns={"region_name": "지역명"})
long_export = priority[
    ["권역", "region_name", "장기취약점수", "장기취약정규점수", "추세변화", "추세", "현재위험점수"]
].rename(columns={"region_name": "지역명"}).sort_values("장기취약점수", ascending=False)

excel_bytes = build_export_excel(export_summary, priority_export, long_export, current)
st.download_button(
    "월간 발간용 Excel 다운로드",
    data=excel_bytes,
    file_name=f"monthly_brief_{selected_month}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
st.download_button(
    "우선관리 지역 CSV 다운로드",
    data=priority_export.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"priority_regions_{selected_month}.csv",
    mime="text/csv",
)
