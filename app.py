from __future__ import annotations

from pathlib import Path

import altair as alt
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


st.set_page_config(page_title="조기경보서비스 월간 리포트(based on 고용보험 데이터)", layout="wide")
st.title("조기경보서비스 월간 리포트(based on 고용보험 데이터)")
st.caption("현황 + 변화 + 지속성 중심 우선관리 리포트")
st.markdown(
    """
    <style>
    table.report-table { width: 100%; border-collapse: collapse; }
    table.report-table th, table.report-table td { text-align: center !important; }
    .score-method {
      font-size: 0.84rem;
      line-height: 1.45;
      margin-top: 0.1rem;
      margin-bottom: 0.8rem;
    }
    .score-method ul { margin: 0.25rem 0 0.55rem 1.0rem; padding-left: 0.7rem; }
    .score-method li { margin: 0.18rem 0; }
    .score-method-box {
      border: 1px solid #e6e8ee;
      border-radius: 8px;
      padding: 0.45rem 0.75rem;
      background: #fafbfc;
    }
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


def build_indicator_flow(current_df: pd.DataFrame) -> pd.DataFrame:
    if current_df.empty:
        return pd.DataFrame()

    work = current_df[
        ["region_level", "region_name", "indicator", "prev_2m_signal", "prev_1m_signal", "current_signal"]
    ].copy()
    score_map = {"정상": 0, "관심": 1, "주의": 2}
    work["prev2"] = work["prev_2m_signal"].map(score_map)
    work["prev1"] = work["prev_1m_signal"].map(score_map)
    work["cur"] = work["current_signal"].map(score_map)
    valid = work[work["prev2"].notna() & work["prev1"].notna() & work["cur"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    valid["d0"] = valid["prev1"] - valid["prev2"]  # 전전월 -> 전월 변화
    valid["d1"] = valid["cur"] - valid["prev1"]  # 전월 -> 당월 변화
    valid["is_consec_worse"] = ((valid["d0"] > 0) & (valid["d1"] > 0)).astype(int)
    valid["is_consec_improve"] = ((valid["d0"] < 0) & (valid["d1"] < 0)).astype(int)
    valid["is_reworse"] = ((valid["d0"] < 0) & (valid["d1"] > 0)).astype(int)
    valid["is_reimprove"] = ((valid["d0"] > 0) & (valid["d1"] < 0)).astype(int)

    out = (
        valid.groupby("indicator", as_index=False)
        .agg(
            전전월전월_악화=("d0", lambda s: int((s > 0).sum())),
            전전월전월_개선=("d0", lambda s: int((s < 0).sum())),
            전전월전월_유지=("d0", lambda s: int((s == 0).sum())),
            전월당월_악화=("d1", lambda s: int((s > 0).sum())),
            전월당월_개선=("d1", lambda s: int((s < 0).sum())),
            전월당월_유지=("d1", lambda s: int((s == 0).sum())),
            연속악화건수=("is_consec_worse", "sum"),
            연속개선건수=("is_consec_improve", "sum"),
            재악화건수=("is_reworse", "sum"),
            재개선건수=("is_reimprove", "sum"),
        )
        .rename(
            columns={
                "indicator": "지표",
                "전전월전월_악화": "전전월→전월 악화",
                "전전월전월_개선": "전전월→전월 개선",
                "전전월전월_유지": "전전월→전월 유지",
                "전월당월_악화": "전월→당월 악화",
                "전월당월_개선": "전월→당월 개선",
                "전월당월_유지": "전월→당월 유지",
                "연속악화건수": "2개월 연속 악화",
                "연속개선건수": "2개월 연속 개선",
                "재악화건수": "재악화",
                "재개선건수": "재개선",
            }
        )
        .sort_values(["전월→당월 악화", "2개월 연속 악화", "지표"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    return out


def build_indicator_worse_monthly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df[["snapshot_month", "indicator", "prev_1m_signal", "current_signal"]].copy()
    score_map = {"정상": 0, "관심": 1, "주의": 2}
    work["prev1"] = work["prev_1m_signal"].map(score_map)
    work["cur"] = work["current_signal"].map(score_map)
    work = work[work["prev1"].notna() & work["cur"].notna()].copy()
    if work.empty:
        return pd.DataFrame()

    work["is_worse"] = (work["cur"] > work["prev1"]).astype(int)
    monthly = (
        work.groupby(["snapshot_month", "indicator"], as_index=False)["is_worse"]
        .sum()
        .rename(columns={"is_worse": "악화건수"})
    )
    pivot = monthly.pivot(index="snapshot_month", columns="indicator", values="악화건수").fillna(0).astype(int)
    return pivot.sort_index()


def build_direction_summary(current_df: pd.DataFrame) -> dict[str, int | list[str]]:
    summary = {
        "consec_worse_regions": 0,
        "consec_improve_regions": 0,
        "reworse_regions": 0,
        "reimprove_regions": 0,
        "consec_worse_names": [],
        "consec_improve_names": [],
        "reworse_names": [],
        "reimprove_names": [],
    }
    if current_df.empty:
        return summary

    work = current_df[
        ["region_level", "region_name", "prev_2m_signal", "prev_1m_signal", "current_signal"]
    ].copy()
    score_map = {"정상": 0, "관심": 1, "주의": 2}
    work["prev2"] = work["prev_2m_signal"].map(score_map)
    work["prev1"] = work["prev_1m_signal"].map(score_map)
    work["cur"] = work["current_signal"].map(score_map)
    valid = work[work["prev2"].notna() & work["prev1"].notna() & work["cur"].notna()].copy()
    if valid.empty:
        return summary

    valid["d0"] = valid["prev1"] - valid["prev2"]
    valid["d1"] = valid["cur"] - valid["prev1"]
    region_cols = ["region_name"]

    def extract_region_names(mask: pd.Series) -> list[str]:
        names = (
            valid.loc[mask, region_cols]
            .drop_duplicates()["region_name"]
            .dropna()
            .astype(str)
            .tolist()
        )
        return sorted(names)

    consec_worse_names = extract_region_names((valid["d0"] > 0) & (valid["d1"] > 0))
    consec_improve_names = extract_region_names((valid["d0"] < 0) & (valid["d1"] < 0))
    reworse_names = extract_region_names((valid["d0"] < 0) & (valid["d1"] > 0))
    reimprove_names = extract_region_names((valid["d0"] > 0) & (valid["d1"] < 0))

    summary["consec_worse_regions"] = len(consec_worse_names)
    summary["consec_improve_regions"] = len(consec_improve_names)
    summary["reworse_regions"] = len(reworse_names)
    summary["reimprove_regions"] = len(reimprove_names)
    summary["consec_worse_names"] = consec_worse_names
    summary["consec_improve_names"] = consec_improve_names
    summary["reworse_names"] = reworse_names
    summary["reimprove_names"] = reimprove_names
    return summary


def format_region_names(names: list[str], limit: int = 6) -> str:
    if not names:
        return "-"
    shown = names[:limit]
    remain = len(names) - len(shown)
    text = ", ".join(shown)
    if remain > 0:
        text += f" 외 {remain}곳"
    return text


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
indicator_flow = build_indicator_flow(current)
indicator_worse_monthly = build_indicator_worse_monthly(df)
direction_summary = build_direction_summary(current)

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

def _indicator_signal_counts(src: pd.DataFrame, levels: list[str]) -> pd.DataFrame:
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


def _format_count_delta(cur: int, prev: int | None) -> str:
    if prev is None:
        return f"{cur} (-)"
    diff = cur - prev
    if diff > 0:
        return f"{cur} (↑{diff})"
    if diff < 0:
        return f"{cur} (↓{abs(diff)})"
    return f"{cur} (→0)"


def indicator_count_view(current_src: pd.DataFrame, prev_src: pd.DataFrame, levels: list[str]) -> pd.DataFrame:
    cur = _indicator_signal_counts(current_src, levels)
    if cur.empty:
        return pd.DataFrame(columns=["지표", "정상", "관심", "주의"])

    has_prev = not prev_src.empty
    prev = _indicator_signal_counts(prev_src, levels) if has_prev else pd.DataFrame(columns=cur.columns)

    merged = cur.rename(columns={c: f"{c}_cur" for c in ["정상", "관심", "주의"]}).merge(
        prev.rename(columns={c: f"{c}_prev" for c in ["정상", "관심", "주의"]}),
        on="지표",
        how="left",
    )
    for col in ["정상_prev", "관심_prev", "주의_prev"]:
        if col not in merged.columns:
            merged[col] = 0
    merged[["정상_prev", "관심_prev", "주의_prev"]] = merged[["정상_prev", "관심_prev", "주의_prev"]].fillna(0).astype(int)

    out = pd.DataFrame({"지표": merged["지표"]})
    for sig in ["정상", "관심", "주의"]:
        cur_col = f"{sig}_cur"
        prev_col = f"{sig}_prev"
        out[sig] = merged.apply(
            lambda r: _format_count_delta(int(r[cur_col]), int(r[prev_col]) if has_prev else None),
            axis=1,
        )
    return out[["지표", "정상", "관심", "주의"]].sort_values("지표").reset_index(drop=True)


def render_centered_table(df: pd.DataFrame) -> None:
    st.markdown(df.to_html(index=False, classes="report-table"), unsafe_allow_html=True)


def render_worse_monthly_line_chart(wide_df: pd.DataFrame) -> None:
    if wide_df.empty:
        st.info("월간 악화건수 그래프를 표시할 데이터가 없습니다.")
        return

    long_df = (
        wide_df.reset_index()
        .melt(id_vars="snapshot_month", var_name="지표", value_name="악화건수")
        .sort_values(["snapshot_month", "지표"])
    )
    if long_df.empty:
        st.info("월간 악화건수 그래프를 표시할 데이터가 없습니다.")
        return

    line = (
        alt.Chart(long_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("snapshot_month:O", title="월"),
            y=alt.Y("악화건수:Q", title="악화건수", scale=alt.Scale(domainMin=0)),
            color=alt.Color("지표:N", legend=alt.Legend(title="지표")),
            tooltip=["snapshot_month:N", "지표:N", "악화건수:Q"],
        )
    )
    # 점은 xOffset으로 살짝 분리해, 단일 월/동일 값에서도 지표별 점이 겹치지 않게 표시.
    point = (
        alt.Chart(long_df)
        .mark_point(size=95, filled=True)
        .encode(
            x=alt.X("snapshot_month:O", title="월"),
            xOffset=alt.XOffset("지표:N"),
            y=alt.Y("악화건수:Q", title="악화건수", scale=alt.Scale(domainMin=0)),
            color=alt.Color("지표:N", legend=alt.Legend(title="지표")),
            tooltip=["snapshot_month:N", "지표:N", "악화건수:Q"],
        )
    )
    st.altair_chart((line + point).properties(height=300), use_container_width=True)


def render_region_month_heatmap(
    src: pd.DataFrame, title: str, month_order: list[str], selected_month: str
) -> None:
    st.markdown(f"#### {title}")
    if src.empty:
        st.info("히트맵을 표시할 데이터가 없습니다.")
        return

    work = src.copy()
    work["snapshot_month"] = work["snapshot_month"].astype(str)
    work["region_name"] = work["region_name"].astype(str)

    latest = (
        work[work["snapshot_month"] == selected_month][["region_name", "월위험점수"]]
        .sort_values(["월위험점수", "region_name"], ascending=[False, True])
        .drop_duplicates("region_name")
    )
    region_order = latest["region_name"].tolist()
    remaining = sorted(set(work["region_name"].tolist()) - set(region_order))
    region_order = region_order + remaining

    chart = (
        alt.Chart(work)
        .mark_rect()
        .encode(
            x=alt.X("snapshot_month:O", sort=month_order, title="월"),
            y=alt.Y("region_name:N", sort=region_order, title="지역"),
            color=alt.Color("월위험점수:Q", title="월별 위험점수", scale=alt.Scale(scheme="yelloworangered", domainMin=0)),
            tooltip=["snapshot_month:N", "region_level:N", "region_name:N", "월위험점수:Q"],
        )
    )
    st.altair_chart(chart.properties(height=max(260, 22 * len(region_order))), use_container_width=True)


def indicator_count_view_gyeonggi(current_src: pd.DataFrame, prev_src: pd.DataFrame) -> pd.DataFrame:
    gg_cur = pd.concat(
        [
            current_src[(current_src["region_level"] == "province") & (current_src["region_name"] == "경기도")],
            current_src[current_src["region_level"] == "gyeonggi_city"],
        ],
        ignore_index=True,
    )
    gg_prev = pd.concat(
        [
            prev_src[(prev_src["region_level"] == "province") & (prev_src["region_name"] == "경기도")],
            prev_src[prev_src["region_level"] == "gyeonggi_city"],
        ],
        ignore_index=True,
    )
    return indicator_count_view(gg_cur, gg_prev, ["province", "gyeonggi_city"])


st.markdown("### 지표별 현황 (전국 / 경기도)")
st.caption("표기: 현재값 (전월 대비 증감, ↑증가 ↓감소 →변동없음)")
v1, v2 = st.columns(2)
with v1:
    st.markdown("#### 전국(전국 + 17개 시도)")
    render_centered_table(indicator_count_view(current, prev, ["national", "province"]))
with v2:
    st.markdown("#### 경기도(경기도 + 31개 시군)")
    render_centered_table(indicator_count_view_gyeonggi(current, prev))

st.divider()

status_tab, index_tab = st.tabs(["현황형 리포트", "지수형 리포트"])

with status_tab:
    st.markdown("## 2. 확산/개선 흐름")
    st.markdown("### 지표별 악화/개선")
    st.markdown("#### 지표별 월간 악화건수")
    render_worse_monthly_line_chart(indicator_worse_monthly)

    if indicator_flow.empty:
        st.info("신호 비교 데이터(prev_2m/prev_1m/current)가 없어 변화 분석이 불가능합니다.")
    else:
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            st.metric("2개월 연속 악화 지역", int(direction_summary["consec_worse_regions"]))
            st.caption(format_region_names(direction_summary["consec_worse_names"]))
        with f2:
            st.metric("2개월 연속 개선 지역", int(direction_summary["consec_improve_regions"]))
            st.caption(format_region_names(direction_summary["consec_improve_names"]))
        with f3:
            st.metric("재악화 지역", int(direction_summary["reworse_regions"]))
            st.caption(format_region_names(direction_summary["reworse_names"]))
        with f4:
            st.metric("재개선 지역", int(direction_summary["reimprove_regions"]))
            st.caption(format_region_names(direction_summary["reimprove_names"]))

    st.divider()
    st.markdown("## 3. 지역 월별 위험 히트맵")
    nationwide_heat = region_month[region_month["region_level"].isin(["national", "province"])].copy()
    gyeonggi_heat = pd.concat(
        [
            region_month[
                (region_month["region_level"] == "province") & (region_month["region_name"] == "경기도")
            ],
            region_month[region_month["region_level"] == "gyeonggi_city"],
        ],
        ignore_index=True,
    )
    render_region_month_heatmap(
        nationwide_heat,
        "전국 + 17개 시도",
        months,
        selected_month,
    )
    render_region_month_heatmap(
        gyeonggi_heat,
        "경기도 + 31개 시군",
        months,
        selected_month,
    )

with index_tab:
    st.markdown("## 2. 우선관리 지역")
    st.markdown(
        """
        <div class="score-method score-method-box">
          <div><strong>지수별 산출방법</strong></div>
          <ul>
            <li>현재위험점수: 주의×2 + 관심×1</li>
            <li>변화점수: 현재위험점수 - 전월위험점수</li>
            <li>장기취약정규점수: 최근 12개월 위험점수(최근 3개월 가중) 정규화</li>
            <li>연속비정상지표수: 3개월 연속(전전월·전월·당월) 관심/주의인 지표 개수</li>
            <li>우선순위점수: 현재위험점수×0.5 + 변화점수×0.3 + 장기취약정규점수×0.2</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    show_cols = [
        "region_name",
        "현재위험점수",
        "변화점수",
        "장기취약정규점수",
        "연속비정상지표수",
        "우선순위점수",
        "추세",
    ]
    st.markdown("### 시도 Top 10")
    render_centered_table(
        priority[priority["region_level"] == "province"][show_cols].head(10).rename(columns={"region_name": "지역명"})
    )

    st.markdown("### 경기 시군 Top 10")
    render_centered_table(
        priority[priority["region_level"] == "gyeonggi_city"][show_cols].head(10).rename(columns={"region_name": "지역명"})
    )

    st.divider()

    st.markdown("## 3. 장기취약 지역 (최근 12개월)")
    st.markdown(
        """
        <div class="score-method score-method-box">
          <div><strong>점수 산출방법</strong></div>
          <ul>
            <li>장기취약점수: 최근 최대 12개월 월위험점수 가중합(최근 3개월 1.5배, 그 외 1.0배)</li>
            <li>장기취약정규점수: 지역별 장기취약점수를 0~100으로 Min-Max 정규화</li>
            <li>추세변화: 최근 3개월 평균 월위험점수 - 이전 3개월 평균 월위험점수</li>
            <li>추세: 추세변화가 1 이상이면 악화, -1 이하이면 개선, 그 외는 유지</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    long_cols = [
        "region_name",
        "현재위험점수",
        "장기취약점수",
        "장기취약정규점수",
        "추세",
    ]
    l1, l2 = st.columns(2)
    with l1:
        st.markdown("### 시도 Top 10")
        long_province = (
            priority[priority["region_level"] == "province"][long_cols]
            .rename(columns={"region_name": "지역명"})
            .sort_values("장기취약점수", ascending=False)
            .head(10)
        )
        render_centered_table(long_province)
    with l2:
        st.markdown("### 경기 시군 Top 10")
        long_city = (
            priority[priority["region_level"] == "gyeonggi_city"][long_cols]
            .rename(columns={"region_name": "지역명"})
            .sort_values("장기취약점수", ascending=False)
            .head(10)
        )
        render_centered_table(long_city)
