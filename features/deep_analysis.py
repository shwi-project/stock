"""🔬 종목 심층분석 탭 — C/D/E/F 통합 뷰.

자체 종목 선택기를 갖고, 선택된 코드에 대해:
- 컨센서스 / 리포트 (C)
- 고급 기술지표 (D)
- 투자자별 수급 (E)
- 공시 리스트 (F)
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
import streamlit as st

from features import consensus, disclosure, flow, indicators_ext
from features._async import run_sync
from features._style import inject as _inject_style
from sources import naver


@st.cache_data(ttl=300, show_spinner="OHLCV 불러오는 중...")
def _ohlcv(code: str, count: int) -> list[dict]:
    return run_sync(naver.get_ohlcv(code, "day", count))


def _indicator_section(code: str) -> None:
    st.markdown("**고급 기술지표**")
    rows = _ohlcv(code, 260)
    if not rows:
        st.info("OHLCV 데이터 없음.")
        return

    keys = [
        "ma_phase",
        "rsi",
        "macd",
        "bollinger",
        "stochastic",
        "obv",
        "volume",
        "position",
        "candle",
        "support_resistance",
        "volume_profile",
        "price_channel",
    ]
    result = indicators_ext.compute_indicators(rows, keys)

    # 요약 카드
    cards = st.columns(4)
    phase = result.get("ma_phase") or {}
    cards[0].metric("이평 Phase", phase.get("phase_label") or "-", border=True)
    rsi = result.get("rsi") or {}
    cards[1].metric("RSI(14)", rsi.get("value") or "-", delta=rsi.get("state"), border=True)
    macd = result.get("macd") or {}
    cards[2].metric(
        "MACD 히스토그램",
        macd.get("histogram") if macd.get("histogram") is not None else "-",
        delta=(macd.get("cross") or {}).get("type_label"),
        border=True,
    )
    bb = result.get("bollinger") or {}
    cards[3].metric(
        "볼린저 위치",
        bb.get("position") or "-",
        delta=f"%B={bb.get('percent_b')}" if bb.get("percent_b") is not None else None,
        border=True,
    )

    # 가격 위치
    pos = result.get("position") or {}
    if pos:
        st.markdown(
            f"**52주 고점** {pos.get('high_52w'):,}원 ({pos.get('pct_from_high_52w')}%) · "
            f"**저점** {pos.get('low_52w'):,}원 ({pos.get('pct_from_low_52w')}%)"
        )

    # Donchian 채널
    pc = result.get("price_channel") or {}
    if pc and "error" not in pc:
        c1, c2, c3 = st.columns(3)
        c1.metric("Donchian 상단", f"{pc.get('upper'):,}", border=True)
        c2.metric("위치", f"{pc.get('position_pct')}% ({pc.get('state')})", border=True)
        c3.metric("폭(변동성)", f"{pc.get('width_pct')}%", border=True)

    # Volume Profile
    vp = result.get("volume_profile") or {}
    if vp and "error" not in vp:
        st.markdown(
            f"**매물대 POC** {vp['poc']['price_range']} · **가치영역** "
            f"{vp['value_area']['low']:,}~{vp['value_area']['high']:,} "
            f"(현재가 {'안' if vp['current_in_value_area'] else '밖'})"
        )
        df_vp = pd.DataFrame([
            {"가격대": f"{p['price_range'][0]:,}~{p['price_range'][1]:,}",
             "거래량비중(%)": p["volume_pct"]}
            for p in vp["profile"]
        ])
        st.bar_chart(df_vp.set_index("가격대"), height=200)

    # Support / Resistance
    sr = result.get("support_resistance") or {}
    if sr and "error" not in sr:
        sup = sr.get("support_levels") or []
        res = sr.get("resistance_levels") or []
        col_s, col_r = st.columns(2)
        with col_s:
            st.caption("지지선 Top 3 (가까운 순)")
            for s in sup[:3]:
                st.write(
                    f"- {s['avg_price']:,}원 · {s['touches']}회 터치 · "
                    f"{s['strength']} · {s['pct_from_current']}%"
                )
        with col_r:
            st.caption("저항선 Top 3 (가까운 순)")
            for r in res[:3]:
                st.write(
                    f"- {r['avg_price']:,}원 · {r['touches']}회 터치 · "
                    f"{r['strength']} · {r['pct_from_current']}%"
                )

    # 캔들
    candle = result.get("candle") or {}
    if candle:
        st.markdown(
            f"**최근 캔들 ({candle.get('date')})**: {candle.get('color')} · "
            f"몸통 {candle.get('body_pct')}% · 윗꼬리 {candle.get('upper_wick_pct')}% · "
            f"아래꼬리 {candle.get('lower_wick_pct')}% · 갭 {candle.get('gap_pct')}%"
        )


def render(stock_options: Sequence[tuple[str, str]] | None = None) -> None:
    """심층분석 탭 본체.

    Args:
        stock_options: (code, label) 튜플 시퀀스. main.py가 load_all_stocks로
            만든 것을 그대로 재사용. None이면 직접 코드 입력 박스로 대체.
    """
    _inject_style()
    st.markdown("### 🔬 종목 심층분석")
    st.caption("네이버·wisereport 데이터 기반. 컨센서스·고급 지표·수급·공시 통합 조회.")

    code = ""
    name = ""

    if stock_options:
        labels = ["선택하세요"] + [label for _, label in stock_options]
        choice = st.selectbox(
            "종목 선택",
            options=range(len(labels)),
            format_func=lambda i: labels[i],
            index=0,
            key="deep_select",
        )
        if choice > 0:
            code, label = stock_options[choice - 1]
            name = label
    else:
        code = st.text_input("종목코드 (6자리)", value="", max_chars=6, key="deep_code")
        name = code

    if not code:
        st.info("종목을 선택하면 컨센서스/지표/수급/공시를 한 화면에 보여줍니다.")
        return

    st.markdown(f"#### {name} ({code})")

    tab_c, tab_d, tab_e, tab_f = st.tabs(
        ["💼 컨센서스·리포트", "📐 고급 지표", "💰 투자자 수급", "📰 최근 공시"]
    )
    with tab_c:
        consensus.render(code)
    with tab_d:
        _indicator_section(code)
    with tab_e:
        flow.render(code, days=20)
    with tab_f:
        disclosure.render(code)
