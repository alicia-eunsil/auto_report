from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path("data/snapshots")

st.set_page_config(page_title="고용보험 월간 리포트", layout="wide")
st.title("고용보험 조기경보 월간 보고서")

st.caption("데이터 원본: stats.gjf.or.kr (관계자 로그인 기반 월말 수집)")

if not DATA_DIR.exists():
    st.warning("data/snapshots 폴더가 없습니다. 수집기를 먼저 실행하세요.")
    st.stop()

csv_files = sorted(DATA_DIR.glob("*.csv"))
if not csv_files:
    st.warning("수집된 데이터가 없습니다. 월말 수집 워크플로를 확인하세요.")
    st.stop()

latest_file = csv_files[-1]
df = pd.read_csv(latest_file)

required_cols = {
    "snapshot_month",
    "region_level",
    "region_name",
    "indicator",
    "current_value",
    "current_signal",
    "prev_2m_value",
    "prev_2m_signal",
}
if not required_cols.issubset(df.columns):
    st.error("필수 컬럼이 누락되었습니다. 수집기 출력을 확인하세요.")
    st.write("필요 컬럼:", sorted(required_cols))
    st.write("현재 컬럼:", sorted(df.columns.tolist()))
    st.stop()

st.subheader(f"기준월: {df['snapshot_month'].max()}")

left, right = st.columns([1, 2])
with left:
    levels = ["all"] + sorted(df["region_level"].dropna().unique().tolist())
    selected_level = st.selectbox("지역 레벨", levels)
with right:
    indicators = ["all"] + sorted(df["indicator"].dropna().unique().tolist())
    selected_indicator = st.selectbox("지표", indicators)

filtered = df.copy()
if selected_level != "all":
    filtered = filtered[filtered["region_level"] == selected_level]
if selected_indicator != "all":
    filtered = filtered[filtered["indicator"] == selected_indicator]

st.markdown("### 신호등 현황")
signal_counts = filtered["current_signal"].value_counts().rename_axis("signal").reset_index(name="count")
st.dataframe(signal_counts, use_container_width=True)

st.markdown("### 상세 데이터")
st.dataframe(
    filtered.sort_values(["current_signal", "region_name", "indicator"]),
    use_container_width=True,
)

st.download_button(
    label="현재 데이터 CSV 다운로드",
    data=filtered.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"report_{df['snapshot_month'].max()}.csv",
    mime="text/csv",
)
