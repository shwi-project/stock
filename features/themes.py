"""A. 테마/섹터 스크리닝 — 네이버 테마·업종 목록과 종목 조회."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from features._async import run_sync
from sources import naver


@st.cache_data(ttl=300, show_spinner=False)
def _themes_all() -> list[dict]:
    return run_sync(naver.list_themes_all(), timeout=60.0)


@st.cache_data(ttl=300, show_spinner=False)
def _theme_stocks(theme_id: str, count: int) -> list[dict]:
    return run_sync(naver.get_theme_stocks(theme_id, count))


@st.cache_data(ttl=300, show_spinner=False)
def _sectors() -> list[dict]:
    return run_sync(naver.list_sectors())


@st.cache_data(ttl=300, show_spinner=False)
def _sector_stocks(sector_id: str, count: int) -> list[dict]:
    return run_sync(naver.get_sector_stocks(sector_id, count))


def render() -> None:
    st.markdown("### 🏷️ 테마 / 섹터 스크리닝")
    st.caption("네이버 증권 기준 약 280개 테마 + 79개 업종. 종목 클릭 시 코드 복사.")

    tab_theme, tab_sector = st.tabs(["테마", "업종(섹터)"])

    with tab_theme:
        try:
            themes = _themes_all()
        except Exception as e:
            st.error(f"테마 목록 조회 실패: {e}")
            themes = []

        if not themes:
            st.info("테마 데이터 없음.")
            return

        df_theme = pd.DataFrame(themes)
        df_theme["leaders_str"] = df_theme["leaders"].apply(
            lambda lst: ", ".join(l["name"] for l in lst) if lst else ""
        )
        st.markdown(f"**총 {len(themes)}개 테마**")

        labels = [f"{t['name']} ({t['change_rate']})" for t in themes]
        selection = st.selectbox(
            "테마 선택",
            options=range(len(themes)),
            format_func=lambda i: labels[i],
            index=0,
            key="theme_select",
        )
        selected = themes[selection]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("등락률", selected["change_rate"])
        c2.metric("최근 3일", selected["recent_3d_rate"])
        c3.metric("상승/보합/하락", f"{selected['up_count']}/{selected['flat_count']}/{selected['down_count']}")
        c4.metric("주도주", ", ".join(l["name"] for l in selected.get("leaders", [])) or "-")

        try:
            stocks = _theme_stocks(selected["theme_id"], 50)
        except Exception as e:
            st.error(f"테마 종목 조회 실패: {e}")
            stocks = []

        if stocks:
            df_s = pd.DataFrame(stocks)
            df_s["거래량"] = df_s["volume"].apply(lambda v: f"{v:,}")
            df_s = df_s[["name", "code", "price", "change_rate", "거래량", "reason"]]
            df_s.columns = ["종목명", "코드", "현재가", "등락률", "거래량", "편입사유"]
            st.dataframe(df_s, use_container_width=True, hide_index=True, height=500)
        else:
            st.info("종목 데이터 없음.")

    with tab_sector:
        try:
            sectors = _sectors()
        except Exception as e:
            st.error(f"업종 목록 조회 실패: {e}")
            sectors = []

        if not sectors:
            st.info("업종 데이터 없음.")
            return

        st.markdown(f"**총 {len(sectors)}개 업종**")
        labels = [f"{s['name']} ({s['change_rate']})" for s in sectors]
        selection = st.selectbox(
            "업종 선택",
            options=range(len(sectors)),
            format_func=lambda i: labels[i],
            index=0,
            key="sector_select",
        )
        selected = sectors[selection]

        c1, c2, c3 = st.columns(3)
        c1.metric("등락률", selected["change_rate"])
        c2.metric("종목 수", selected["total_count"])
        c3.metric("상승/보합/하락", f"{selected['up_count']}/{selected['flat_count']}/{selected['down_count']}")

        try:
            stocks = _sector_stocks(selected["sector_id"], 50)
        except Exception as e:
            st.error(f"업종 종목 조회 실패: {e}")
            stocks = []

        if stocks:
            df_s = pd.DataFrame(stocks)
            df_s["거래량"] = df_s["volume"].apply(lambda v: f"{v:,}")
            df_s = df_s[["name", "code", "price", "change_rate", "거래량"]]
            df_s.columns = ["종목명", "코드", "현재가", "등락률", "거래량"]
            st.dataframe(df_s, use_container_width=True, hide_index=True, height=500)
        else:
            st.info("종목 데이터 없음.")
