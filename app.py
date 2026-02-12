from pathlib import Path

import pandas as pd
import streamlit as st

MASTER_CSV = Path("data/master/employment_master.csv")
SNAPSHOT_DIR = Path("data/snapshots")
SIGNAL_COLS = ["current_signal", "prev_1m_signal", "prev_2m_signal"]

st.set_page_config(page_title="고용보험 월간 리포트", layout="wide")
st.title("고용보험 조기경보 월간 보고서")
st.caption("전국/시도/경기도 시군 분리, 전월 대비, 3개월 연속 비정상, 주요지역")


def load_data() -> pd.DataFrame:
    if MASTER_CSV.exists():
        return pd.read_csv(MASTER_CSV)

    csv_files = sorted(SNAPSHOT_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("데이터 파일이 없습니다. 수집기를 먼저 실행하세요.")
    return pd.read_csv(csv_files[-1])


try:
    df = load_data()
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
if not required.issubset(df.columns):
    st.error("필수 컬럼이 누락되었습니다.")
    st.write("누락:", sorted(required - set(df.columns)))
    st.stop()

months = sorted(df["snapshot_month"].dropna().unique().tolist())
if not months:
    st.warning("표시할 기준월 데이터가 없습니다.")
    st.stop()

selected_month = st.selectbox("기준월", months, index=len(months) - 1)
current = df[df["snapshot_month"] == selected_month].copy()

st.markdown("## 1) 지표별 신호등 현황 (전국/시도/시군 분리)")

scope_map = {
    "national": "전국",
    "province": "시도",
    "gyeonggi_city": "시군",
}
summary_rows: list[dict[str, object]] = []
for level, scope_name in scope_map.items():
    level_df = current[current["region_level"] == level]
    for indicator, g in level_df.groupby("indicator"):
        counts = g["current_signal"].value_counts()
        summary_rows.append(
            {
                "구분": scope_name,
                "지표": indicator,
                "정상": int(counts.get("정상", 0)),
                "관심": int(counts.get("관심", 0)),
                "주의": int(counts.get("주의", 0)),
            }
        )

summary_df = pd.DataFrame(summary_rows).sort_values(["구분", "지표"]) if summary_rows else pd.DataFrame()

selected_idx = months.index(selected_month)
prev_month = months[selected_idx - 1] if selected_idx > 0 else None
if prev_month:
    prev = df[df["snapshot_month"] == prev_month]
    prev_rows: list[dict[str, object]] = []
    for level, scope_name in scope_map.items():
        level_df = prev[prev["region_level"] == level]
        for indicator, g in level_df.groupby("indicator"):
            counts = g["current_signal"].value_counts()
            prev_rows.append(
                {
                    "구분": scope_name,
                    "지표": indicator,
                    "정상_prev": int(counts.get("정상", 0)),
                    "관심_prev": int(counts.get("관심", 0)),
                    "주의_prev": int(counts.get("주의", 0)),
                }
            )
    prev_df = pd.DataFrame(prev_rows)
    if not prev_df.empty and not summary_df.empty:
        summary_df = summary_df.merge(prev_df, on=["구분", "지표"], how="left").fillna(0)
        summary_df["정상_전월대비"] = summary_df["정상"] - summary_df["정상_prev"]
        summary_df["관심_전월대비"] = summary_df["관심"] - summary_df["관심_prev"]
        summary_df["주의_전월대비"] = summary_df["주의"] - summary_df["주의_prev"]
        summary_df = summary_df.drop(columns=["정상_prev", "관심_prev", "주의_prev"])

st.dataframe(summary_df, use_container_width=True)

st.markdown("## 2) 3개월 연속 비정상 지역")
for c in SIGNAL_COLS:
    current[c] = current[c].fillna("")
mask = current[SIGNAL_COLS].apply(lambda s: s.isin(["관심", "주의"])).all(axis=1)
three_month = current.loc[mask, ["region_level", "indicator", "region_name", "current_signal", "prev_1m_signal", "prev_2m_signal"]]

c1, c2 = st.columns(2)
with c1:
    st.markdown("### 시도")
    province = three_month[three_month["region_level"] == "province"].drop(columns=["region_level"])
    st.dataframe(province, use_container_width=True)
with c2:
    st.markdown("### 시군")
    city = three_month[three_month["region_level"] == "gyeonggi_city"].drop(columns=["region_level"])
    st.dataframe(city, use_container_width=True)

st.markdown("## 3) 지표별 주요지역 (관심/주의 많은 순)")


def top_regions(level: str, top_n: int) -> pd.DataFrame:
    level_df = current[current["region_level"] == level].copy()
    if level_df.empty:
        return pd.DataFrame()
    level_df["주의개수"] = (level_df[SIGNAL_COLS] == "주의").sum(axis=1)
    level_df["관심개수"] = (level_df[SIGNAL_COLS] == "관심").sum(axis=1)
    level_df["비정상개수"] = level_df["주의개수"] + level_df["관심개수"]

    rows: list[pd.DataFrame] = []
    for indicator, g in level_df.groupby("indicator"):
        picked = g.sort_values(["주의개수", "관심개수", "region_name"], ascending=[False, False, True]).head(top_n)
        rows.append(picked[["indicator", "region_name", "주의개수", "관심개수", "비정상개수"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


col1, col2 = st.columns(2)
with col1:
    st.markdown("### 시도 Top 5")
    st.dataframe(top_regions("province", 5), use_container_width=True)
with col2:
    st.markdown("### 시군 Top 10")
    st.dataframe(top_regions("gyeonggi_city", 10), use_container_width=True)

st.markdown("## 4) 원본 데이터")
st.dataframe(current.sort_values(["region_level", "region_name", "indicator"]), use_container_width=True)

st.download_button(
    "기준월 데이터 CSV 다운로드",
    data=current.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"report_{selected_month}.csv",
    mime="text/csv",
)
