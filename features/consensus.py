"""C. 증권사 컨센서스 & 리포트 패널."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from features._async import run_sync
from sources import naver


@st.cache_data(ttl=600, show_spinner="컨센서스 불러오는 중...")
def _consensus(code: str) -> dict:
    return run_sync(naver.get_consensus(code))


@st.cache_data(ttl=600, show_spinner="리포트 불러오는 중...")
def _reports(code: str, count: int) -> list[dict]:
    return run_sync(naver.get_reports(code, count))


def render(code: str, current_price: float | None = None) -> None:
    if not code:
        st.info("종목코드를 입력하세요.")
        return

    try:
        c = _consensus(code)
    except Exception as e:
        st.error(f"컨센서스 조회 실패: {e}")
        c = {}

    target = c.get("target_price")
    opinion = c.get("opinion") or {}

    cols = st.columns(2)
    with cols[0]:
        if target:
            up_pct = (
                f" ({(target - current_price) / current_price * 100:+.1f}%)"
                if current_price else ""
            )
            st.metric("애널리스트 목표주가", f"{target:,}원{up_pct}", border=True)
        else:
            st.metric("애널리스트 목표주가", "-", border=True)
    with cols[1]:
        if opinion:
            order = ["매수", "강력매수", "중립", "매도", "강력매도"]
            sorted_op = sorted(
                opinion.items(),
                key=lambda x: order.index(x[0]) if x[0] in order else 99,
            )
            text = " · ".join(f"{k} {v}" for k, v in sorted_op if v)
            st.metric("투자의견 분포", text or "-", border=True)
        else:
            st.metric("투자의견 분포", "-", border=True)

    history = c.get("target_price_history") or []
    if history:
        df_h = pd.DataFrame(history)
        st.caption("목표주가 추이 (최근 6개)")
        st.line_chart(df_h.set_index("date")["price"], height=180)

    estimates = c.get("estimates") or {}
    if estimates and c.get("estimate_periods"):
        st.caption("증권사 실적 추정치")
        df_e = pd.DataFrame(estimates).T
        st.dataframe(df_e, use_container_width=True)

    st.markdown("**최근 증권사 리포트**")
    try:
        reports = _reports(code, 5)
    except Exception as e:
        st.error(f"리포트 조회 실패: {e}")
        reports = []

    if not reports:
        st.caption("리포트 없음.")
        return

    df_r = pd.DataFrame(reports)
    df_r = df_r[["date", "broker", "title"]]
    df_r.columns = ["날짜", "증권사", "제목"]
    st.dataframe(df_r, use_container_width=True, hide_index=True)
