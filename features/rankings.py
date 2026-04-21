"""B. 장중 랭킹 대시보드 — 거래량/등락률/시총 Top N."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from features._async import run_sync
from sources import naver


def _format_volume(v: int) -> str:
    if v >= 1_0000_0000:
        return f"{v/1_0000_0000:.1f}억"
    if v >= 1_0000:
        return f"{v/1_0000:.0f}만"
    return f"{v:,}"


def _format_money_won(v: int) -> str:
    if v >= 1_0000_0000_0000:
        return f"{v/1_0000_0000_0000:.2f}조"
    if v >= 1_0000_0000:
        return f"{v/1_0000_0000:.0f}억"
    return f"{v:,}"


@st.cache_data(ttl=120, show_spinner="거래대금 랭킹 불러오는 중...")
def _volume(market: str, count: int, sort_by: str) -> list[dict]:
    return run_sync(naver.get_volume_ranking(market=market, count=count, sort_by=sort_by))


@st.cache_data(ttl=120, show_spinner="등락률 랭킹 불러오는 중...")
def _change(direction: str, market: str, count: int) -> list[dict]:
    return run_sync(naver.get_change_ranking(direction=direction, market=market, count=count))


@st.cache_data(ttl=600, show_spinner="시가총액 랭킹 불러오는 중...")
def _market_cap(market: str, count: int) -> list[dict]:
    return run_sync(naver.get_market_cap_ranking(market=market, count=count))


def render() -> None:
    st.markdown("### 📈 장중 랭킹")
    st.caption("네이버 증권 기준. 캐시 TTL 2~10분.")

    col1, col2 = st.columns(2)
    with col1:
        market = st.selectbox(
            "시장",
            ["ALL", "KOSPI", "KOSDAQ"],
            index=0,
            key="rankings_market",
        )
    with col2:
        count = st.slider("표시 종목 수", 10, 100, 30, step=10, key="rankings_count")

    tab_vol, tab_up, tab_down, tab_cap = st.tabs(
        ["거래대금 Top", "상승률 Top", "하락률 Top", "시가총액 Top"]
    )

    with tab_vol:
        try:
            data = _volume(market, count, "trade_value")
        except Exception as e:
            st.error(f"데이터 조회 실패: {e}")
            data = []
        if not data:
            st.info("데이터 없음 (장중이 아니거나 네이버 응답 변경 가능).")
        else:
            df = pd.DataFrame(data)
            df["거래량"] = df["volume"].apply(_format_volume)
            df["거래대금"] = df["trade_value_krw"].apply(_format_money_won)
            df = df[["rank", "name", "code", "price", "change_rate", "거래량", "거래대금"]]
            df.columns = ["순위", "종목명", "코드", "현재가", "등락률", "거래량", "거래대금"]
            st.dataframe(df, use_container_width=True, hide_index=True, height=600)

    with tab_up:
        try:
            data = _change("up", market, count)
        except Exception as e:
            st.error(f"데이터 조회 실패: {e}")
            data = []
        if not data:
            st.info("데이터 없음.")
        else:
            df = pd.DataFrame(data)
            df["거래량"] = df["volume"].apply(_format_volume)
            df = df[["rank", "name", "code", "price", "change_rate", "거래량"]]
            df.columns = ["순위", "종목명", "코드", "현재가", "등락률", "거래량"]
            st.dataframe(df, use_container_width=True, hide_index=True, height=600)

    with tab_down:
        try:
            data = _change("down", market, count)
        except Exception as e:
            st.error(f"데이터 조회 실패: {e}")
            data = []
        if not data:
            st.info("데이터 없음.")
        else:
            df = pd.DataFrame(data)
            df["거래량"] = df["volume"].apply(_format_volume)
            df = df[["rank", "name", "code", "price", "change_rate", "거래량"]]
            df.columns = ["순위", "종목명", "코드", "현재가", "등락률", "거래량"]
            st.dataframe(df, use_container_width=True, hide_index=True, height=600)

    with tab_cap:
        cap_market = market if market != "ALL" else "KOSPI"
        if market == "ALL":
            st.caption("시가총액 랭킹은 시장별로만 조회 가능 → KOSPI 기준 표시")
        try:
            data = _market_cap(cap_market, count)
        except Exception as e:
            st.error(f"데이터 조회 실패: {e}")
            data = []
        if not data:
            st.info("데이터 없음.")
        else:
            df = pd.DataFrame(data)
            df["거래량"] = df["volume"].apply(_format_volume)
            df["시가총액"] = df["market_cap_billion"].apply(lambda b: f"{b:,}억원")
            df = df[["rank", "name", "code", "price", "change_rate", "시가총액", "거래량"]]
            df.columns = ["순위", "종목명", "코드", "현재가", "등락률", "시가총액", "거래량"]
            st.dataframe(df, use_container_width=True, hide_index=True, height=600)
