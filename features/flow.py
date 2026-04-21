"""E. 투자자별 수급 (외국인/기관 순매매)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from features._async import run_sync
from sources import naver


@st.cache_data(ttl=300, show_spinner="수급 데이터 불러오는 중...")
def _flow(code: str, days: int) -> list[dict]:
    return run_sync(naver.get_investor_flow(code, days))


def _fmt(v: int) -> str:
    if v == 0:
        return "0"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:,}"


def render(code: str, days: int = 20) -> None:
    if not code:
        st.info("종목코드를 입력하세요.")
        return
    try:
        rows = _flow(code, days)
    except Exception as e:
        st.error(f"수급 데이터 조회 실패: {e}")
        return

    if not rows:
        st.info("수급 데이터 없음.")
        return

    df = pd.DataFrame(rows)
    inst_sum = int(df["institutional"].sum())
    foreign_sum = int(df["foreign"].sum())

    c1, c2 = st.columns(2)
    c1.metric(f"기관 {days}일 누적", _fmt(inst_sum))
    c2.metric(f"외국인 {days}일 누적", _fmt(foreign_sum))

    df_chart = df[["date", "institutional", "foreign"]].iloc[::-1].copy()
    df_chart.columns = ["date", "기관", "외국인"]
    st.bar_chart(df_chart.set_index("date"), height=240)

    df_view = df[["date", "close", "institutional", "foreign", "volume"]].copy()
    df_view["institutional"] = df_view["institutional"].apply(_fmt)
    df_view["foreign"] = df_view["foreign"].apply(_fmt)
    df_view["volume"] = df_view["volume"].apply(lambda v: f"{v:,}")
    df_view.columns = ["날짜", "종가", "기관 순매매", "외국인 순매매", "거래량"]
    st.dataframe(df_view, use_container_width=True, hide_index=True, height=400)
