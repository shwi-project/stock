import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import GradientBoostingRegressor
import plotly.graph_objects as go
from streamlit_lightweight_charts import renderLightweightCharts
import requests
import json
import html as html_mod
import base64
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────

try:
    with open("logo.png", "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    logo_html = (
        f"<img src='data:image/png;base64,{logo_b64}' "
        "style='width:36px;height:36px;object-fit:contain;"
        "margin-right:10px;pointer-events:none;'>"
    )
except Exception:
    logo_html = "<span style='font-size:1.4rem;margin-right:5px'>🏰</span>"

st.set_page_config(
    page_title="Castle Stock AI | 믿지 못할 주식 예측",
    page_icon="🏰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# 커스텀 CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');

    .stApp { background: #0b0e17 !important; font-family: 'Noto Sans KR', sans-serif; }
    html, body { background: #0b0e17 !important; }
    [data-testid="stAppViewContainer"] { background: #0b0e17 !important; }
    [data-testid="stMain"] { background: #0b0e17 !important; }

    [data-testid="stHeader"]         { display: none !important; }
    [data-testid="stSidebar"]        { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }

    /* 모바일 좌우 스크롤 방지 */
    html, body, [data-testid="stAppViewContainer"], .main {
        overflow-x: hidden !important;
        max-width: 100vw !important;
    }

    .block-container {
        padding-top: 0.7rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 960px;
    }

    h3 { color: #e2e8f0 !important; font-size: 1.05rem !important; margin-bottom: 0.3rem !important; }
    hr { border-color: #1e2435 !important; margin: 0.3rem 0 0.5rem 0 !important; }

    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.4rem;
        margin-bottom: 0.5rem;
    }
    @media (max-width: 640px) {
        .metrics-grid { grid-template-columns: repeat(2, 1fr); }
        .metrics-grid .metric-card:last-child { grid-column: 1 / -1; }
    }

    .metric-card {
        background: #131929; border: 1px solid #1e2d45; border-radius: 8px;
        padding: 0.55rem 0.75rem;
    }
    .metric-label { font-size: 0.6rem; color: #4a5568; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 3px; }
    .metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; font-weight: 600; color: #e2e8f0; }
    .metric-sub   { font-size: 0.75rem; margin-top: 3px; }
    .unit         { font-size: 0.78rem; color: #4a5568; }
    .metric-up    { color: #fc5c5c; }
    .metric-down  { color: #4d9fff; }

    .section-header {
        font-size: 0.65rem; font-weight: 600; color: #4a5568;
        text-transform: uppercase; letter-spacing: 2px;
        padding: 0.35rem 0 0.2rem 0; border-bottom: 1px solid #1e2435; margin-bottom: 0.45rem;
    }
    
    /* 검색 돋보기 버튼 전용 스타일 (원본 복원) */
    [data-testid="stHorizontalBlock"] .stButton > button {
        height: 40px !important;
        min-height: 40px !important;
        max-height: 40px !important;
        width: 40px !important;
        min-width: 40px !important;
        padding: 0 !important;
        line-height: 40px !important;
        font-size: 1.0rem !important;
        margin-bottom: 5px !important;
        margin-left: 0px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: #2d3a55 !important;
        border: 1px solid #3d4f6e !important;
        color: #e2e8f0 !important;
    }
    [data-testid="stHorizontalBlock"] .stButton > button:hover {
        background: #3d4f6e !important;
        border-color: #4d9fff !important;
        color: #e2e8f0 !important;
    }
    [data-testid="stHorizontalBlock"] .stButton > button:active {
        background: #1e2d45 !important;
        border-color: #4d9fff !important;
    }
    [data-testid="stButton"] > div {
        display: flex !important;
        align-items: flex-end !important;
    }

    /* 스캐너 AI form — 카드 앞, height:0, top으로 헤더 줄에 고정 (카드 높이 무관) */
    [data-testid="stForm"] {
        border: none !important;
        background: transparent !important;
        height: 0 !important;
        overflow: visible !important;
        padding: 0 !important;
        margin: 0 !important;
        position: relative !important;
        z-index: 999 !important;
        pointer-events: none !important;
    }
    [data-testid="stForm"],
    [data-testid="stForm"] [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stForm"] [data-testid="stVerticalBlock"],
    [data-testid="stForm"] [data-testid="stElementContainer"],
    [data-testid="stForm"] [data-testid="stFormSubmitButton"],
    [data-testid="stForm"] [data-testid="stFormSubmitButton"] > div {
        direction: rtl !important;
        overflow: visible !important;
        pointer-events: none !important;
    }
    [data-testid="stFormSubmitButton"] > button {
        direction: ltr !important;
        background: #2d3a55 !important;
        border: 1px solid #4a5568 !important;
        color: #e2e8f0 !important;
        font-family: 'Noto Sans KR', sans-serif !important;
        font-size: 7px !important;
        transform: scale(0.85) !important;
        transform-origin: right center !important;
        font-weight: 500 !important;
        letter-spacing: 0.3px !important;
        padding: 0 5px !important;
        height: 26px !important;
        min-height: 26px !important;
        max-height: 26px !important;
        border-radius: 4px !important;
        transition: all 0.2s ease !important;
        width: auto !important;
        min-width: auto !important;
        max-width: 100px !important;
        white-space: nowrap !important;
        cursor: pointer !important;
        position: relative !important;
        top: 26px !important;
        margin-right: 16px !important;
        line-height: 13px !important;
        box-shadow: 0 1px 3px rgba(99,102,241,0.12) !important;
        pointer-events: all !important;
        z-index: 999 !important;
    }
    [data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(135deg, #1e2745, #243352) !important;
        color: #ddd6fe !important;
        border-color: rgba(99,102,241,0.55) !important;
        box-shadow: 0 2px 8px rgba(99,102,241,0.25) !important;
    }

    .ai-box {
        background: linear-gradient(135deg, #0f1a2e 0%, #111827 100%);
        border: 1px solid #1e3a5f; border-left: 3px solid #3b82f6; border-radius: 8px;
        padding: 0.75rem 1rem; color: #e2e8f0; font-size: 0.88rem; line-height: 1.75; margin-top: 0.3rem;
    }

    .news-card {
        background: #131929; border: 1px solid #1e2435; border-radius: 6px;
        padding: 0.5rem 0.75rem; margin-bottom: 0.3rem; font-size: 0.77rem; color: #a0aec0; line-height: 1.45;
    }
    .news-card:hover { border-color: #2d3a55; }

    .badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 0.6rem; font-weight: 600; }
    .badge-kospi  { background: #1a2a4a; color: #4d9fff; }
    .badge-kosdaq { background: #1a3a2a; color: #4dc98f; }

    .scanner-wrap { margin-top: 0.5rem; }
    .scanner-card {
        background: linear-gradient(135deg, #131929 0%, #0f1a2e 100%);
        border: 1px solid #1e2d45; border-radius: 10px;
        padding: 0.75rem 1rem; margin-bottom: 0.5rem;
        transition: border-color 0.2s;
    }
    .scanner-card:hover { border-color: #3b82f6; }
    .scanner-rank {
        display: inline-flex; align-items: center; justify-content: center;
        width: 24px; height: 24px; border-radius: 50%;
        font-size: 0.72rem; font-weight: 700; margin-right: 10px;
    }
    .rank-1 { background: linear-gradient(135deg, #f9a825, #ff6f00); color: #000; }
    .rank-2 { background: linear-gradient(135deg, #b0bec5, #78909c); color: #000; }
    .rank-3 { background: linear-gradient(135deg, #8d6e63, #6d4c41); color: #fff; }
    .rank-other { background: #2d3a55; color: #a0aec0; }
    .scanner-score {
        display: inline-block; padding: 2px 8px; border-radius: 4px;
        font-size: 0.68rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;
    }
    .score-high { background: #1a3a2a; color: #4dc98f; }
    .score-mid  { background: #3a3a1a; color: #f9a825; }
    .score-low  { background: #3a1a1a; color: #fc5c5c; }
    .signal-tag {
        display: inline-block; padding: 1px 6px; border-radius: 3px;
        font-size: 0.58rem; font-weight: 600; margin-right: 4px; margin-top: 2px;
        background: #1e2d45; color: #4d9fff;
    }
    /* Streamlit 탭 다크테마 */
    .stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid #1e2435; gap: 0; }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important; color: #4a5568 !important;
        font-size: 0.82rem !important; font-weight: 500; padding: 8px 20px !important;
        border: none !important; border-bottom: 2px solid transparent !important;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #a0aec0 !important; }
    .stTabs [aria-selected="true"] {
        color: #e2e8f0 !important; border-bottom: 2px solid #3b82f6 !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 0.5rem !important; }
    /* Streamlit expander 다크테마 */
    .streamlit-expanderHeader {
        background: #131929 !important; color: #a0aec0 !important;
        font-size: 0.75rem !important; border: 1px solid #1e2d45 !important;
        border-radius: 6px !important; padding: 4px 12px !important;
    }
    .streamlit-expanderContent { border: 1px solid #1e2d45 !important; border-top: none !important; }
    details[data-testid="stExpander"] { background: transparent !important; border: none !important; }
    details[data-testid="stExpander"] summary {
        background: #131929 !important; color: #a0aec0 !important;
        font-size: 0.75rem !important; border-radius: 6px !important;
    }
    details[data-testid="stExpander"] > div {
        background: #0d1117 !important; border: 1px solid #1e2d45 !important;
        border-top: none !important; border-radius: 0 0 6px 6px !important;
    }

    .scanner-ai {
        background: #0d1117; border-radius: 6px; padding: 8px 12px;
        margin-top: 6px; font-size: 0.78rem; color: #c8d6e8; line-height: 1.65;
        border-left: 2px solid #3b82f6;
    }

    iframe {
        border: none !important; outline: none !important;
        display: block !important; background: #0b0e17 !important;
    }
    div:has(> iframe) { background: #0b0e17 !important; border: none !important; padding: 0 !important; }
    [data-testid="stCustomComponentV1"],
    [data-testid="stCustomComponentV1"] > div { background: #0b0e17 !important; border: none !important; padding: 0 !important; }
    [data-testid="element-container"] { background: transparent !important; }

    label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span { color: #c8d6e8 !important; font-size: 0.8rem !important; }
    [data-baseweb="select"] > div, [data-baseweb="select"] > div > div {
        background: #131929 !important; border-color: #2d3a55 !important;
        min-height: 40px !important; max-height: 40px !important;
        padding-top: 0 !important; padding-bottom: 0 !important; line-height: 40px !important;
        font-size: 0.9rem !important;
    }
    [data-baseweb="select"] * { color: #e2e8f0 !important; }
    [data-baseweb="select"] input { height: 40px !important; }    
    .stSelectbox { margin-bottom: 0.3rem !important; }
    [data-baseweb="popover"], [data-baseweb="menu"] { background: #131929 !important; }
    [data-baseweb="option"]       { color: #e2e8f0 !important; background: #131929 !important; }
    [data-baseweb="option"]:hover { background: #1e2d45 !important; }

    [data-testid="stSpinner"] p, [data-testid="stSpinner"] span, .stSpinner p { font-size: 0.72rem !important; color: #4a5568 !important; }

    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    .stDeployButton { display: none; }
    [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; flex-direction: row !important; align-items: flex-start !important; gap: 0.3rem !important; }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width: 0 !important; }
    .bottom-wrap { display: grid; grid-template-columns: 3fr 2fr; gap: 0.8rem; margin-top: 0.5rem; }
    .bottom-left  { min-width: 0; }
    .bottom-right { min-width: 0; }
    @media (max-width: 768px) { .bottom-wrap { grid-template-columns: 1fr; } }
    @media (max-width: 640px) {
        .block-container { padding-left: 0.8rem !important; padding-right: 0.8rem !important; }
        .metric-value { font-size: 0.82rem !important; }
        .metric-label { font-size: 0.55rem !important; }
        .ai-box { font-size: 0.82rem !important; }
    }
    @media (max-width: 400px) {
        .block-container { padding-left: 0.2rem !important; padding-right: 0.2rem !important; }
        .metric-value { font-size: 0.75rem !important; }
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 한국 시간 헬퍼
# ─────────────────────────────────────────────
from datetime import timezone
KST = timezone(timedelta(hours=9))

def now_kst() -> datetime:
    return datetime.now(KST)


# ─────────────────────────────────────────────
# 전체 종목 목록 로딩 (캐시 1시간)
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_all_stocks() -> pd.DataFrame:
    def _make_df(rows):
        df = pd.DataFrame(rows, columns=["종목명","종목코드","시장"])
        df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
        df["ticker"]  = df.apply(
            lambda r: r["종목코드"] + (".KS" if r["시장"]=="KOSPI" else ".KQ"), axis=1)
        return df[["종목명","종목코드","시장","ticker"]].drop_duplicates("종목코드").reset_index(drop=True)

    try:
        import FinanceDataReader as fdr
        kospi  = fdr.StockListing('KOSPI')[["Name","Code"]].copy()
        kosdaq = fdr.StockListing('KOSDAQ')[["Name","Code"]].copy()
        kospi["시장"]  = "KOSPI"
        kosdaq["시장"] = "KOSDAQ"
        df_all = pd.concat([kospi, kosdaq], ignore_index=True)
        df_all.columns = ["종목명","종목코드","시장"]
        df_all = df_all.dropna(subset=["종목명","종목코드"])
        df_all["종목코드"] = df_all["종목코드"].astype(str).str.zfill(6)
        df_all["ticker"] = df_all.apply(
            lambda r: r["종목코드"] + (".KS" if r["시장"]=="KOSPI" else ".KQ"), axis=1)
        if len(df_all) > 100:
            return df_all[["종목명","종목코드","시장","ticker"]].drop_duplicates("종목코드").reset_index(drop=True)
    except Exception:
        pass

    try:
        from pykrx import stock as krx_stock
        today = now_kst().strftime("%Y%m%d")
        rows = []
        for mkt in ["KOSPI", "KOSDAQ"]:
            tickers = krx_stock.get_market_ticker_list(date=today, market=mkt)
            for t in tickers:
                name = krx_stock.get_market_ticker_name(t)
                if name:
                    rows.append([name, t.zfill(6), mkt])
        if len(rows) > 100:
            return _make_df(rows)
    except Exception:
        pass

    data = [
        ("삼성전자","005930","KOSPI"),("SK하이닉스","000660","KOSPI"),
        ("LG에너지솔루션","373220","KOSPI"),("삼성바이오로직스","207940","KOSPI"),
        ("현대차","005380","KOSPI"),("셀트리온","068270","KOSPI"),
        ("KB금융","105560","KOSPI"),("기아","000270","KOSPI"),
        ("신한지주","055550","KOSPI"),("POSCO홀딩스","005490","KOSPI"),
        ("NAVER","035420","KOSPI"),("LG화학","051910","KOSPI"),
        ("삼성SDI","006400","KOSPI"),("현대모비스","012330","KOSPI"),
        ("카카오","035720","KOSPI"),("하나금융지주","086790","KOSPI"),
        ("SK이노베이션","096770","KOSPI"),("LG전자","066570","KOSPI"),
        ("한화에어로스페이스","012450","KOSPI"),("현대로템","064350","KOSPI"),
        ("HD한국조선해양","009540","KOSPI"),("한화오션","042660","KOSPI"),
        ("하이브","352820","KOSPI"),("에스엠","041510","KOSPI"),
        ("에코프로","086520","KOSDAQ"),("에코프로비엠","247540","KOSDAQ"),
        ("HLB","028300","KOSDAQ"),("알테오젠","196170","KOSDAQ"),
        ("리노공업","058470","KOSDAQ"),("크래프톤","259960","KOSDAQ"),
        ("카카오뱅크","323410","KOSDAQ"),("실리콘투","257720","KOSDAQ"),
    ]
    return _make_df([[n,c,m] for n,c,m in data])


# ─────────────────────────────────────────────
# 전종목 벌크 스크리닝 (pykrx 벌크 API)
# ─────────────────────────────────────────────
_FALLBACK_UNIVERSE = [
    "005930","000660","373220","207940","005380","068270","105560","000270",
    "055550","005490","035420","051910","006400","012330","035720","086790",
    "096770","066570","012450","064350","009540","042660","352820","041510",
    "086520","247540","028300","196170","058470","259960","323410","257720",
]


@st.cache_data(ttl=600, show_spinner=False)
def _pre_screen_market(date_str: str, top_n: int = 150, _v: int = 5) -> tuple:
    """전종목 벌크 스크리닝 (FDR 기반, pykrx 의존 제거).
    Returns (candidate_codes: list, bulk_data: dict, debug: str)
    """
    bulk_data = {}
    all_scores = {}
    _debug = "start"

    try:
        import FinanceDataReader as fdr

        # ── 1. KOSPI + KOSDAQ 전종목 리스트 (FDR) ──
        frames = []
        for mkt in ["KOSPI", "KOSDAQ"]:
            try:
                df = fdr.StockListing(mkt)
                if df is not None and len(df) > 0:
                    frames.append(df)
            except Exception:
                pass
        if not frames:
            return _FALLBACK_UNIVERSE, {}, "listing_empty"
        listing = pd.concat(frames, ignore_index=True)
        _debug = f"listing_ok,rows={len(listing)}"

        # 컬럼 정규화 (FDR 버전에 따라 컬럼명이 다를 수 있음)
        col_map = {}
        for c in listing.columns:
            cl = c.lower().strip()
            if cl in ("code", "symbol", "종목코드"): col_map[c] = "Code"
            elif cl in ("name", "종목명"): col_map[c] = "Name"
            elif cl in ("close", "종가", "현재가"): col_map[c] = "Close"
            elif cl in ("volume", "거래량"): col_map[c] = "Volume"
            elif cl in ("amount", "거래대금"): col_map[c] = "Amount"
            elif cl in ("marcap", "시가총액"): col_map[c] = "Marcap"
            elif cl in ("stocks", "상장주식수"): col_map[c] = "Stocks"
            elif cl in ("chagesratio", "changesratio", "등락률"): col_map[c] = "ChagesRatio"
            elif cl in ("changes", "전일비"): col_map[c] = "Changes"
            elif cl in ("market",): col_map[c] = "Market"
        if col_map:
            listing = listing.rename(columns=col_map)

        needed = ["Code", "Close", "Volume", "Marcap"]
        for nc in needed:
            if nc not in listing.columns:
                return _FALLBACK_UNIVERSE, {}, f"missing_col={nc},cols={list(listing.columns)[:8]}"

        # ── 2. 필터링 + 스코어링 ──
        for _, row in listing.iterrows():
            try:
                code = str(row["Code"]).strip().zfill(6)
                close = float(row.get("Close", 0))
                volume = float(row.get("Volume", 0))
                marcap = float(row.get("Marcap", 0))

                if close < 1000 or volume < 1000:
                    continue
                if marcap < 100_000_000_000:  # 시총 1000억 미만 제외
                    continue

                change_pct = float(row.get("ChagesRatio", 0))
                amount = float(row.get("Amount", 0))
                stocks = float(row.get("Stocks", 0))

                # 거래대금 기반 점수 (1조 이상 → 5점)
                amount_score = min(amount / 200_000_000_000, 5.0)
                # 거래량 점수
                vol_score = min(volume / 1_000_000, 3.0)

                pre_score = amount_score + vol_score

                all_scores[code] = pre_score
                bulk_data[code] = {
                    "marcap": marcap,
                    "foreign_ratio": 0.0,  # FDR StockListing에는 외국인 비율 없음
                    "per": 0.0,
                    "pbr": 0.0,
                    "change_pct": change_pct,
                    "amount": amount,
                    "stocks": stocks,
                }
            except Exception:
                continue

    except Exception as e:
        return _FALLBACK_UNIVERSE, {}, f"fdr_err={str(e)[:80]}"

    if not all_scores:
        return _FALLBACK_UNIVERSE, {}, "no_scores"

    sorted_codes = sorted(all_scores, key=all_scores.get, reverse=True)[:top_n]
    _debug = f"fdr_ok,candidates={len(sorted_codes)}"
    return sorted_codes, bulk_data, _debug


# ─────────────────────────────────────────────
# 멀티팩터 퀀트 스캐너 (학술 기반 4-Pillar 모델)
# Momentum(25) + MeanReversion(15) + TrendQuality(20) + RiskAdjusted(15) + Supply(25) = 100
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_ohlcv(code: str, start_str: str, date_str: str):
    """FDR 우선, pykrx 폴백으로 OHLCV 가져오기."""
    try:
        import FinanceDataReader as fdr
        tmp = fdr.DataReader(code, start_str)
        if tmp is not None and len(tmp) > 60:
            tmp = tmp.reset_index()
            ohlcv = tmp[["Date","Open","High","Low","Close","Volume"]].copy()
            for c in ["Close","Volume","High","Low","Open"]:
                ohlcv[c] = ohlcv[c].astype(float)
            return ohlcv
    except Exception:
        pass
    try:
        from pykrx import stock as krx_stock
        start_d = (datetime.strptime(date_str, "%Y%m%d") - timedelta(days=180)).strftime("%Y%m%d")
        tmp = krx_stock.get_market_ohlcv_by_date(start_d, date_str, code)
        if tmp is not None and len(tmp) > 60:
            tmp = tmp.reset_index()
            return pd.DataFrame({
                "Date": tmp.iloc[:, 0],
                "Open": tmp["시가"].astype(float), "High": tmp["고가"].astype(float),
                "Low": tmp["저가"].astype(float), "Close": tmp["종가"].astype(float),
                "Volume": tmp["거래량"].astype(float),
            })
    except Exception:
        pass
    return None


def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-9)
    return (100 - 100 / (1 + rs)).fillna(50)


def _calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> dict:
    """ADX, +DI, -DI 계산 (Wilder's smoothing)."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    plus_dm = ((high - prev_high).clip(lower=0)).where(high - prev_high > prev_low - low, 0)
    minus_dm = ((prev_low - low).clip(lower=0)).where(prev_low - low > high - prev_high, 0)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return {"adx": float(adx.iloc[-1]), "plus_di": float(plus_di.iloc[-1]), "minus_di": float(minus_di.iloc[-1])}


def _calc_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> dict:
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, 1e-9)
    d = k.rolling(d_period).mean()
    return {"k": float(k.iloc[-1]), "d": float(d.iloc[-1]),
            "k_prev": float(k.iloc[-2]) if len(k) >= 2 else 50.0}


def _calc_ichimoku(high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
    """이치모쿠 클라우드 시그널 (9/26/52)."""
    def _mid(s, p): return (s.rolling(p).max() + s.rolling(p).min()) / 2
    tenkan = _mid(close, 9)
    kijun = _mid(close, 26)
    senkou_a = (tenkan + kijun) / 2
    senkou_b = _mid(close, 52)
    cur_price = float(close.iloc[-1])
    # 현재 구름 = 26봉 전에 계산된 값
    cloud_top = max(float(senkou_a.iloc[-26]), float(senkou_b.iloc[-26])) if len(senkou_a) >= 26 else cur_price
    cloud_bot = min(float(senkou_a.iloc[-26]), float(senkou_b.iloc[-26])) if len(senkou_a) >= 26 else cur_price
    above_cloud = cur_price > cloud_top
    tenkan_above_kijun = float(tenkan.iloc[-1]) > float(kijun.iloc[-1]) if len(tenkan.dropna()) > 0 and len(kijun.dropna()) > 0 else False
    future_bullish = float(senkou_a.iloc[-1]) > float(senkou_b.iloc[-1]) if len(senkou_a.dropna()) > 0 and len(senkou_b.dropna()) > 0 else False
    return {"above_cloud": above_cloud, "tenkan_kijun": tenkan_above_kijun, "future_bullish": future_bullish}


def _calc_bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0) -> dict:
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + std_mult * std
    lower = ma - std_mult * std
    bandwidth = ((upper - lower) / ma.replace(0, 1e-9))
    pct_b = (close - lower) / (upper - lower).replace(0, 1e-9)
    # Squeeze: 현재 bandwidth가 120일 내 최저에 가까운지
    bw_min_120 = bandwidth.rolling(min(120, len(bandwidth))).min()
    is_squeeze = float(bandwidth.iloc[-1]) <= float(bw_min_120.iloc[-1]) * 1.1 if len(bw_min_120.dropna()) > 0 else False
    above_mid = float(close.iloc[-1]) > float(ma.iloc[-1]) if len(ma.dropna()) > 0 else False
    return {"pct_b": float(pct_b.iloc[-1]), "bandwidth": float(bandwidth.iloc[-1]),
            "is_squeeze": is_squeeze, "above_mid": above_mid}


@st.cache_data(ttl=600, show_spinner=False)
def run_scanner(date_str: str, _v: int = 5) -> pd.DataFrame:
    """멀티팩터 퀀트 스캐너: 4-Pillar 모델로 종목 스코어링."""
    scanner_universe, bulk_data, _inv_debug_info = _pre_screen_market(date_str)
    start_str = (datetime.strptime(date_str, "%Y%m%d") - timedelta(days=250)).strftime("%Y-%m-%d")

    # ── KOSPI 벤치마크 데이터 ──
    kospi_return_60d = 0.0
    try:
        import FinanceDataReader as fdr
        _kospi = fdr.DataReader("KS11", start_str)
        if _kospi is not None and len(_kospi) >= 60:
            _kc = _kospi["Close"].astype(float)
            kospi_return_60d = float(_kc.iloc[-1] / _kc.iloc[-60] - 1)
    except Exception:
        pass

    # ── Pass 0: OHLCV 병렬 다운로드 ──
    _ohlcv_map = {}
    _fail_count = 0
    def _fetch_one(code):
        return code, _fetch_ohlcv(code, start_str, date_str)
    with ThreadPoolExecutor(max_workers=10) as executor:
        ohlcv_futures = {executor.submit(_fetch_one, c): c for c in scanner_universe}
        for fut in as_completed(ohlcv_futures):
            try:
                code, ohlcv = fut.result()
                if ohlcv is not None and len(ohlcv) >= 60:
                    _ohlcv_map[code] = ohlcv
                else:
                    _fail_count += 1
            except Exception:
                _fail_count += 1

    # ── Pass 1: 모든 종목 raw factor 수집 ──
    raw_results = []
    for code in scanner_universe:
        try:
            if code not in _ohlcv_map:
                continue
            ohlcv = _ohlcv_map[code]
            close = ohlcv["Close"]
            high = ohlcv["High"]
            low = ohlcv["Low"]
            volume = ohlcv["Volume"]
            if float(close.iloc[-1]) <= 0:
                continue

            returns = close.pct_change().dropna()
            n = len(close)

            # ─── PILLAR 1: MOMENTUM ───
            # 1a. 상대강도 vs KOSPI (60일)
            stock_return_60d = float(close.iloc[-1] / close.iloc[-60] - 1) if n >= 60 else 0.0
            rel_strength = stock_return_60d - kospi_return_60d

            # 1b. MACD 히스토그램 기울기 (5일)
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist = macd_line - signal_line
            macd_slope = float(macd_hist.iloc[-1] - macd_hist.iloc[-5]) / 5.0 if n >= 30 else 0.0

            # 1c. 가격 vs MA 캐스케이드 (5/20/60)
            ma5 = close.rolling(5).mean()
            ma20 = close.rolling(20).mean()
            ma60 = close.rolling(60).mean()
            cp = float(close.iloc[-1])
            ma_cascade = 0.0
            if len(ma5.dropna()) > 0 and cp > float(ma5.iloc[-1]):   ma_cascade += 0.33
            if len(ma20.dropna()) > 0 and cp > float(ma20.iloc[-1]): ma_cascade += 0.33
            if len(ma60.dropna()) > 0 and cp > float(ma60.iloc[-1]): ma_cascade += 0.34

            # 1d. ROC 20일
            roc_20 = float(close.iloc[-1] / close.iloc[-20] - 1) if n >= 20 else 0.0

            # 1e. OBV 추세 확인
            obv = (volume * ((close.diff() > 0).astype(int) * 2 - 1)).cumsum()
            obv_slope_20 = float(obv.iloc[-1] - obv.iloc[-20]) if n >= 20 else 0.0
            price_slope_20 = float(close.iloc[-1] - close.iloc[-20]) if n >= 20 else 0.0
            if price_slope_20 > 0 and obv_slope_20 > 0:
                obv_confirm = 1.0  # 가격↑ 거래량↑ 동조
            elif price_slope_20 < 0 and obv_slope_20 > 0:
                obv_confirm = 0.7  # 강세 다이버전스
            elif price_slope_20 > 0 and obv_slope_20 < 0:
                obv_confirm = 0.2  # 약세 다이버전스
            else:
                obv_confirm = 0.4

            # ─── RSI 1회 계산 후 재사용 ───
            rsi_series = _calc_rsi(close)

            # 1f. RSI Divergence Detection (20-bar lookback)
            _divergence_score = 0.0
            _divergence_tag = None
            if n >= 20:
                _lookback = 20
                _price_window = close.iloc[-_lookback:]
                _rsi_window = rsi_series.iloc[-_lookback:]
                # Find local lows/highs in price (simple: compare to neighbors)
                _p_vals = _price_window.values
                _r_vals = _rsi_window.values
                _lows_idx = [i for i in range(1, len(_p_vals)-1) if _p_vals[i] < _p_vals[i-1] and _p_vals[i] < _p_vals[i+1]]
                _highs_idx = [i for i in range(1, len(_p_vals)-1) if _p_vals[i] > _p_vals[i-1] and _p_vals[i] > _p_vals[i+1]]
                # Bullish divergence: price lower low, RSI higher low
                if len(_lows_idx) >= 2:
                    _l1, _l2 = _lows_idx[-2], _lows_idx[-1]
                    if _p_vals[_l2] < _p_vals[_l1] and _r_vals[_l2] > _r_vals[_l1]:
                        _divergence_score = 3.0
                        _divergence_tag = "강세다이버전스"
                # Bearish divergence: price higher high, RSI lower high
                if len(_highs_idx) >= 2 and _divergence_score == 0:
                    _h1, _h2 = _highs_idx[-2], _highs_idx[-1]
                    if _p_vals[_h2] > _p_vals[_h1] and _r_vals[_h2] < _r_vals[_h1]:
                        _divergence_score = -3.0
                        _divergence_tag = "약세다이버전스"

            # ─── PILLAR 2: MEAN REVERSION ───
            cur_rsi = float(rsi_series.iloc[-1])
            # RSI 존 점수 (30-40 최적 반등 구간)
            if 30 <= cur_rsi <= 40:
                rsi_zone = 1.0
            elif 20 <= cur_rsi < 30:
                rsi_zone = 0.7
            elif 40 < cur_rsi <= 50:
                rsi_zone = 0.6
            elif 50 < cur_rsi <= 60:
                rsi_zone = 0.3
            else:
                rsi_zone = 0.0  # 과매수(>70) 또는 극단적 과매도(<20)

            bb = _calc_bollinger(close)
            pct_b = bb["pct_b"]
            if pct_b <= 0.2:
                bb_score = 1.0
            elif pct_b <= 0.4:
                bb_score = 0.6
            elif pct_b <= 0.6:
                bb_score = 0.3
            else:
                bb_score = 0.0

            stoch = _calc_stochastic(high, low, close)
            stoch_score = 0.0
            if stoch["k"] < 30 and stoch["k"] > stoch["d"] and stoch["k_prev"] <= stoch["d"]:
                stoch_score = 1.0  # 과매도 구간 골든크로스
            elif stoch["k"] > stoch["d"]:
                stoch_score = 0.3

            # ─── PILLAR 3: TREND QUALITY ───
            adx_data = _calc_adx(high, low, close)
            if adx_data["adx"] > 25 and adx_data["plus_di"] > adx_data["minus_di"]:
                adx_score = 1.0  # 강한 상승추세
            elif adx_data["adx"] > 20 and adx_data["plus_di"] > adx_data["minus_di"]:
                adx_score = 0.6
            elif adx_data["adx"] < 20:
                adx_score = 0.2  # 추세 없음
            else:
                adx_score = 0.0  # 하락추세

            ichimoku = _calc_ichimoku(high, low, close) if n >= 52 else {"above_cloud": False, "tenkan_kijun": False, "future_bullish": False}
            ichi_score = (0.4 if ichimoku["above_cloud"] else 0) + \
                         (0.3 if ichimoku["tenkan_kijun"] else 0) + \
                         (0.3 if ichimoku["future_bullish"] else 0)

            vol_avg20 = float(volume.rolling(20).mean().iloc[-1]) if n >= 20 else 1
            vol_ratio = float(volume.iloc[-1]) / max(vol_avg20, 1)
            price_up = cp > float(close.iloc[-2]) if n >= 2 else False
            if vol_ratio > 2.0 and price_up:
                vol_confirm = 1.0
            elif vol_ratio > 1.5 and price_up:
                vol_confirm = 0.7
            elif vol_ratio > 2.0 and not price_up:
                vol_confirm = 0.1  # 분배 (매도)
            else:
                vol_confirm = 0.3

            squeeze_score = 0.0
            if bb["is_squeeze"] and bb["above_mid"]:
                squeeze_score = 1.0
            elif bb["is_squeeze"]:
                squeeze_score = 0.4

            # ─── PILLAR 4: RISK-ADJUSTED ───
            if len(returns) >= 60:
                r60 = returns.iloc[-60:]
                sharpe_60d = float(r60.mean() / r60.std() * np.sqrt(252)) if r60.std() > 0 else 0.0
                downside = r60[r60 < 0]
                downside_dev = float(downside.std()) if len(downside) > 2 else 0.0
            else:
                sharpe_60d = 0.0
                downside_dev = 0.01

            atr14 = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1).rolling(14).mean()
            atr_pct = float(atr14.iloc[-1]) / cp if cp > 0 else 0.0

            prev_close = float(close.iloc[-2]) if n >= 2 else cp
            change_pct = round((cp - prev_close) / max(prev_close, 1) * 100, 2)

            # ─── 외국인 보유비율 + PER/PBR (벌크 데이터) ───
            _bulk = bulk_data.get(code, {})
            foreign_ratio = _bulk.get("foreign_ratio", 0.0)
            per_val = _bulk.get("per", 0)
            pbr_val = _bulk.get("pbr", 0)

            # 외국인 보유비율 점수 (스코어링용): 높을수록 외국인 관심 종목
            _foreign_score = np.clip(foreign_ratio / 10, 0, 1)  # 10%+ → 1.0

            # ─── 매집 신호 (Accumulation) ───
            # 가격 횡보(±3%) + OBV 상승 = 세력 매집
            _accumulation = 0.0
            if n >= 20:
                _price_range_20 = (float(close.iloc[-1]) / float(close.iloc[-20]) - 1)
                if abs(_price_range_20) < 0.03 and obv_slope_20 > 0:
                    _accumulation = 1.0  # 강한 매집
                elif abs(_price_range_20) < 0.05 and obv_slope_20 > 0:
                    _accumulation = 0.6  # 약한 매집

            # ─── 시그널 태그 생성 ───
            signals = []
            macd_cur = float(macd_hist.iloc[-1])
            if rel_strength > 0.05: signals.append("상대강도↑")
            if adx_data["adx"] > 25 and adx_data["plus_di"] > adx_data["minus_di"]: signals.append(f"ADX{adx_data['adx']:.0f}")
            if vol_ratio > 1.5 and price_up: signals.append(f"거래량{vol_ratio:.1f}x")
            if bb["is_squeeze"]: signals.append("스퀴즈")
            if 20 <= cur_rsi <= 40: signals.append(f"RSI{cur_rsi:.0f}")
            if ichimoku["above_cloud"]: signals.append("구름↑")
            if ma_cascade >= 0.66: signals.append("MA정배열")
            if macd_cur > 0 and macd_slope > 0: signals.append("MACD▲")
            if stoch_score >= 0.8: signals.append("스토캐스틱↑")
            if obv_confirm >= 0.7: signals.append("OBV확인")
            if _divergence_tag == "강세다이버전스": signals.append("💡RSI반전")
            if _divergence_tag == "약세다이버전스": signals.append("⚠RSI괴리")
            # 외국인 보유비율/매집 시그널
            if foreign_ratio >= 30: signals.append(f"🏦외인{foreign_ratio:.0f}%")
            elif foreign_ratio >= 15: signals.append(f"🏦외인{foreign_ratio:.0f}%")
            if _accumulation >= 0.6: signals.append("🔒매집감지")
            if 0 < per_val < 10: signals.append(f"PER{per_val:.1f}")
            if 0 < pbr_val < 1.0: signals.append(f"PBR{pbr_val:.2f}")

            raw_results.append({
                "code": code, "price": int(cp), "change_pct": change_pct,
                "rsi": round(cur_rsi, 1), "vol_ratio": round(vol_ratio, 1),
                "adx": round(adx_data["adx"], 1), "sharpe": round(sharpe_60d, 2),
                "macd_hist": round(macd_cur, 2),
                "signals": signals,
                "foreign_ratio": foreign_ratio,
                "amount": bulk_data.get(code, {}).get("amount", 0),
                # Raw scores (Pillar 1 - 상대적 비교 필요한 것들)
                "_rel_strength": rel_strength, "_roc_20": roc_20,
                "_sharpe_60d": sharpe_60d, "_atr_pct": atr_pct, "_downside_dev": downside_dev,
                # Raw scores (Pillar별 절대점수)
                "_macd_slope": macd_slope, "_ma_cascade": ma_cascade, "_obv_confirm": obv_confirm,
                "_divergence_score": _divergence_score,
                "_rsi_zone": rsi_zone, "_bb_score": bb_score, "_stoch_score": stoch_score,
                "_adx_score": adx_score, "_ichi_score": ichi_score,
                "_vol_confirm": vol_confirm, "_squeeze_score": squeeze_score,
                "_foreign_score": _foreign_score,
                "_accumulation": _accumulation,
            })
        except Exception:
            continue

    # st.toast는 @st.cache_data 안에서 사용 불가 — 호출부에서 처리

    if not raw_results:
        return pd.DataFrame()

    # ── Pass 2: 교차 비교 (Percentile Ranking) ──
    df_raw = pd.DataFrame(raw_results)
    n_stocks = len(df_raw)

    # 높을수록 좋은 지표 → percentile rank (0~1)
    for col in ["_rel_strength", "_roc_20", "_sharpe_60d"]:
        df_raw[col + "_rank"] = df_raw[col].rank(pct=True)

    # 낮을수록 좋은 지표 → 역순 rank
    for col in ["_atr_pct", "_downside_dev"]:
        df_raw[col + "_rank"] = 1 - df_raw[col].rank(pct=True)

    # 외국인 보유비율 percentile rank
    df_raw["_foreign_score_rank"] = df_raw["_foreign_score"].rank(pct=True)

    # ── Pass 3: 최종 점수 계산 (100점 만점) ──
    # Momentum(25) + MeanReversion(15) + TrendQuality(20) + RiskAdjusted(15) + Supply(25) = 100
    scores = []
    for idx, row in df_raw.iterrows():
        # PILLAR 1: MOMENTUM (25점)
        p1_rel = row["_rel_strength_rank"] * 7                         # 상대강도 7점
        p1_macd = min(max(row["_macd_slope"] / 50 + 0.5, 0), 1) * 6  # MACD 기울기 6점
        p1_ma = row["_ma_cascade"] * 5                                 # MA 캐스케이드 5점
        p1_roc = row["_roc_20_rank"] * 4                               # ROC-20 4점
        p1_obv = row["_obv_confirm"] * 3                               # OBV 확인 3점
        p1_div = row["_divergence_score"]                              # 다이버전스 ±3점
        momentum = np.clip(p1_rel + p1_macd + p1_ma + p1_roc + p1_obv + p1_div, 0, 25)

        # PILLAR 2: MEAN REVERSION (15점)
        p2_rsi = row["_rsi_zone"] * 5                                  # RSI 존 5점
        p2_bb = row["_bb_score"] * 5                                   # 볼린저 %B 5점
        p2_stoch = row["_stoch_score"] * 5                             # 스토캐스틱 5점
        mean_rev = p2_rsi + p2_bb + p2_stoch

        # PILLAR 3: TREND QUALITY (20점)
        p3_adx = row["_adx_score"] * 6                                 # ADX 6점
        p3_ichi = row["_ichi_score"] * 5                               # 이치모쿠 5점
        p3_vol = row["_vol_confirm"] * 5                               # 거래량 확인 5점
        p3_squeeze = row["_squeeze_score"] * 4                         # 볼린저 스퀴즈 4점
        trend = p3_adx + p3_ichi + p3_vol + p3_squeeze

        # PILLAR 4: RISK-ADJUSTED (15점)
        p4_sharpe = row["_sharpe_60d_rank"] * 6                        # 샤프비율 6점
        p4_atr = row["_atr_pct_rank"] * 5                             # ATR 변동성 5점
        p4_dd = row["_downside_dev_rank"] * 4                          # 하방 편차 4점
        risk_adj = p4_sharpe + p4_atr + p4_dd

        # PILLAR 5: SUPPLY (수급) (25점)
        p5_foreign = row["_foreign_score_rank"] * 15                   # 외국인 보유비율 15점
        p5_accum = row["_accumulation"] * 10                           # 매집 신호 10점
        supply = p5_foreign + p5_accum

        total = momentum + mean_rev + trend + risk_adj + supply
        scores.append({
            "total": round(total, 1),
            "momentum": round(momentum, 1),
            "mean_rev": round(mean_rev, 1),
            "trend": round(trend, 1),
            "risk_adj": round(risk_adj, 1),
            "supply": round(supply, 1),
        })

    df_scores = pd.DataFrame(scores)
    df_raw["score"] = df_scores["total"]
    df_raw["momentum"] = df_scores["momentum"]
    df_raw["mean_rev"] = df_scores["mean_rev"]
    df_raw["trend"] = df_scores["trend"]
    df_raw["risk_adj"] = df_scores["risk_adj"]
    df_raw["supply"] = df_scores["supply"]

    # 내부 계산용 컬럼 제거
    drop_cols = [c for c in df_raw.columns if c.startswith("_")]
    df_raw = df_raw.drop(columns=drop_cols)

    # Top 5 정렬
    df_result = df_raw.sort_values("score", ascending=False).head(5).reset_index(drop=True)
    return df_result


# ─────────────────────────────────────────────
# 모닝 스캐너: Gemini 브리핑 (Top 5)
# ─────────────────────────────────────────────
def fetch_scanner_briefing(stock_code: str, stock_info: dict, today_str: str) -> str:
    """개별 종목에 대한 짧은 Gemini AI 브리핑을 반환. 종목별 캐시."""
    if "GEMINI_API_KEY" not in st.secrets:
        return ""

    # 종목별 캐시 확인
    cache_key = f"scanner_ai_{today_str}"
    if cache_key in st.session_state:
        cached = st.session_state[cache_key].get(stock_code)
        if cached:
            return cached

    gemini_key = str(st.secrets["GEMINI_API_KEY"]).strip()
    base_url = "https://generativelanguage.googleapis.com/v1beta/models"
    model = "gemini-2.5-flash-lite"
    today_display = f"{today_str[:4]}년 {today_str[4:6]}월 {today_str[6:]}일"

    stock = stock_info
    try:
        prompt = f"""당신은 한국 주식 전문 퀀트 애널리스트입니다. 오늘은 {today_display}입니다.

[종목 정보]
- {stock['name']} ({stock['code']}) | 현재가: {stock['price']:,}원 ({stock['change_pct']:+.2f}%)
- RSI: {stock['rsi']} | ADX: {stock.get('adx','N/A')} | Sharpe(60d): {stock.get('sharpe','N/A')} | 거래량배율: {stock['vol_ratio']}x | MACD: {stock['macd_hist']:+.2f}
- 퀀트 스코어: {stock.get('score','N/A')}/100 (모멘텀 {stock.get('momentum','?')}/35 · 진입타이밍 {stock.get('mean_rev','?')}/20 · 추세품질 {stock.get('trend','?')}/25 · 위험조정 {stock.get('risk_adj','?')}/20)
- 감지된 시그널: {', '.join(stock['signals'])}

[Google 검색 명령]
"{stock['name']} 최신 뉴스 {today_str}"

[출력 규칙]
- 반드시 4줄 이내 개조식으로 작성
- 1줄: 핵심 이슈 (최근 뉴스/공시 기반)
- 2줄: 퀀트 분석 요약 (어떤 팩터가 강하고 약한지)
- 3줄: 기술적 판단 (매수/관망/매도 + 근거)
- 4줄: 핵심 타점 (진입가, 손절가)
- 코드블록, disclaimer 등 불필요한 텍스트 절대 금지"""

        url = f"{base_url}/{model}:generateContent?key={gemini_key}"
        res = requests.post(url, json={
            "contents": [{"parts": [{"text": prompt}]}],
            "tools": [{"googleSearch": {}}]
        }, timeout=30)

        if res.status_code == 200:
            rj = res.json()
            if "candidates" in rj:
                import re as _re
                parts = rj["candidates"][0]["content"].get("parts", [])
                text = "".join(p.get("text", "") for p in parts if "text" in p).strip()
                text = _re.sub(r'```.*?```', '', text, flags=_re.DOTALL)
                text = _re.sub(r'print\s*\(.*?\)\s*', '', text, flags=_re.DOTALL)
                text = _re.sub(r'google_search\.\w+\(.*?\)', '', text, flags=_re.DOTALL)
                text = _re.sub(r'(?i)disclaimer.*', '', text, flags=_re.DOTALL)
                text = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
                text = text.strip()
                if text:
                    result = text.replace("\n", "<br>")
                    # 종목별 캐시 저장
                    if cache_key not in st.session_state:
                        st.session_state[cache_key] = {}
                    st.session_state[cache_key][stock_code] = result
                    return result
    except Exception:
        pass

    return ""


# ─────────────────────────────────────────────
# 예측 계산 (날짜 기반 캐시 → 일관성 보장)
# ─────────────────────────────────────────────
KR_HOLIDAYS = {
    datetime(2024,  1,  1), datetime(2024,  2,  9), datetime(2024,  2, 10),
    datetime(2024,  2, 12), datetime(2024,  3,  1), datetime(2024,  4, 10),
    datetime(2024,  5,  5), datetime(2024,  5,  6), datetime(2024,  5, 15),
    datetime(2024,  6,  6), datetime(2024,  8, 15), datetime(2024,  9, 16),
    datetime(2024,  9, 17), datetime(2024,  9, 18), datetime(2024, 10,  3),
    datetime(2024, 10,  9), datetime(2024, 12, 25),
    datetime(2025,  1,  1), datetime(2025,  1, 28), datetime(2025,  1, 29),
    datetime(2025,  1, 30), datetime(2025,  3,  1), datetime(2025,  3,  3),
    datetime(2025,  5,  5), datetime(2025,  5,  6), datetime(2025,  6,  6),
    datetime(2025,  8, 15), datetime(2025, 10,  3), datetime(2025, 10,  5),
    datetime(2025, 10,  6), datetime(2025, 10,  7), datetime(2025, 10,  9),
    datetime(2025, 12, 25),
    datetime(2026,  1,  1), datetime(2026,  2, 17), datetime(2026,  2, 18),
    datetime(2026,  2, 19), datetime(2026,  3,  1), datetime(2026,  3,  2),
    datetime(2026,  5,  5), datetime(2026,  5, 25), datetime(2026,  6,  6),
    datetime(2026,  8, 17), datetime(2026,  9, 24), datetime(2026,  9, 25),
    datetime(2026,  9, 28), datetime(2026, 10,  9), datetime(2026, 12, 25),
    # 2027
    datetime(2027,  1,  1), datetime(2027,  2,  6), datetime(2027,  2,  7),
    datetime(2027,  2,  8), datetime(2027,  3,  1), datetime(2027,  5,  5),
    datetime(2027,  5, 13), datetime(2027,  6,  6), datetime(2027,  8, 15),
    datetime(2027,  9, 25), datetime(2027,  9, 26), datetime(2027,  9, 27),
    datetime(2027, 10,  3), datetime(2027, 10,  9), datetime(2027, 12, 25),
    # 2028
    datetime(2028,  1,  1), datetime(2028,  1, 25), datetime(2028,  1, 26),
    datetime(2028,  1, 27), datetime(2028,  3,  1), datetime(2028,  5,  5),
    datetime(2028,  5,  2), datetime(2028,  6,  6), datetime(2028,  8, 15),
    datetime(2028, 10,  3), datetime(2028, 10,  9), datetime(2028, 10, 13),
    datetime(2028, 10, 14), datetime(2028, 10, 15), datetime(2028, 12, 25),
    # 2029
    datetime(2029,  1,  1), datetime(2029,  2, 12), datetime(2029,  2, 13),
    datetime(2029,  2, 14), datetime(2029,  3,  1), datetime(2029,  5,  5),
    datetime(2029,  5, 22), datetime(2029,  6,  6), datetime(2029,  8, 15),
    datetime(2029, 10,  1), datetime(2029, 10,  2), datetime(2029, 10,  3),
    datetime(2029, 10,  9), datetime(2029, 12, 25),
    # 2030
    datetime(2030,  1,  1), datetime(2030,  2,  2), datetime(2030,  2,  3),
    datetime(2030,  2,  4), datetime(2030,  3,  1), datetime(2030,  5,  5),
    datetime(2030,  5,  9), datetime(2030,  6,  6), datetime(2030,  8, 15),
    datetime(2030,  9, 21), datetime(2030,  9, 22), datetime(2030,  9, 23),
    datetime(2030, 10,  3), datetime(2030, 10,  9), datetime(2030, 12, 25),
}

def get_kr_trading_days(start_date, count):
    days = []
    cur = start_date + timedelta(days=1)
    while len(days) < count:
        if cur.weekday() < 5 and cur not in KR_HOLIDAYS:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def _ewma_volatility(returns, span=20):
    """EWMA (Exponentially Weighted Moving Average) volatility forecast.
    Returns annualized volatility as a decimal (e.g. 0.35 = 35%)."""
    if returns is None or len(returns) < span:
        return 0.20  # default 20% annualized vol
    ewma_var = returns.ewm(span=span, adjust=False).var()
    last_var = float(ewma_var.iloc[-1])
    if np.isnan(last_var) or last_var <= 0:
        return 0.20
    annualized_vol = np.sqrt(last_var * 252)
    return float(np.clip(annualized_vol, 0.05, 2.0))


def _detect_divergence(df, lookback=20):
    """Detect price-RSI divergence.
    Returns 'bearish_div', 'bullish_div', or None."""
    if len(df) < lookback:
        return None
    recent = df.tail(lookback)
    price = recent["Close"]
    rsi = recent["RSI"] if "RSI" in recent.columns else None
    if rsi is None:
        return None

    mid = lookback // 2
    # Check bearish: price higher high, RSI lower high
    price_h1 = price.iloc[:mid].max()
    price_h2 = price.iloc[mid:].max()
    rsi_h1 = rsi.iloc[:mid].max()
    rsi_h2 = rsi.iloc[mid:].max()

    if price_h2 > price_h1 and rsi_h2 < rsi_h1 - 2:
        return "bearish_div"

    # Check bullish: price lower low, RSI higher low
    price_l1 = price.iloc[:mid].min()
    price_l2 = price.iloc[mid:].min()
    rsi_l1 = rsi.iloc[:mid].min()
    rsi_l2 = rsi.iloc[mid:].min()

    if price_l2 < price_l1 and rsi_l2 > rsi_l1 + 2:
        return "bullish_div"

    return None


def detect_regime(df: pd.DataFrame) -> str:
    """Detect market regime: 'bull', 'bear', or 'sideways'.

    Uses ADX (trend strength), price vs MA20/MA60, and recent returns.
    - Bull:  ADX > 25 AND price > MA20 > MA60
    - Bear:  ADX > 25 AND price < MA20 < MA60
    - Sideways: otherwise
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    if len(close) < 60:
        return "sideways"

    # ADX calculation
    adx_data = _calc_adx(high, low, close)
    adx_val = adx_data["adx"]

    # Price vs MA20 and MA60
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma60 = float(close.rolling(60).mean().iloc[-1])
    cur_price = float(close.iloc[-1])

    if adx_val > 25 and cur_price > ma20 and ma20 > ma60:
        return "bull"
    elif adx_val > 25 and cur_price < ma20 and ma20 < ma60:
        return "bear"
    else:
        return "sideways"


@st.cache_data(ttl=86400, show_spinner=False)
def compute_prediction(stock_code: str, date_str: str, pred_days: int,
                       _last_date_str: str, news_raw_json: str) -> dict:
    """날짜+종목+마지막거래일 기반 캐시로 동일한 날 동일한 결과 보장."""
    news_raw = json.loads(news_raw_json)

    # 데이터를 내부에서 직접 가져옴 (캐시 키와 무관하게 일관된 데이터)
    df = fetch_stock_ohlcv(stock_code, days=730)

    np.random.seed(42)  # Prophet 재현성 보장

    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df.loc[df["Open"] <= 0, "Open"] = df["Close"]
    df.loc[df["High"] <= 0, "High"] = df["Close"]
    df.loc[df["Low"]  <= 0, "Low"]  = df["Close"]

    close = df["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    df["RSI"] = (100 - 100 / (1 + rs)).fillna(50)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = (macd_line - macd_signal_line).fillna(0)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_pct"] = ((close - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-9)).fillna(0.5).clip(0, 1)

    obv = (df["Volume"] * df["Close"].diff().apply(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    )).cumsum()
    obv_range = obv.max() - obv.min()
    df["OBV_norm"] = ((obv - obv.min()) / (obv_range + 1e-9)).fillna(0.5)

    df["vol_ratio"] = (df["Volume"] / df["Volume"].rolling(20).mean().replace(0, 1)).fillna(1.0)
    df["ma20_dist"] = ((close - close.rolling(20).mean()) / (close.rolling(20).mean() + 1e-9)).fillna(0)

    # Sentiment keywords with magnitude weights (higher = stronger signal)
    POS_WORDS = {
        "상승": 1.0, "급등": 2.0, "신고가": 2.0, "호실적": 1.5, "흑자": 1.5,
        "수주": 1.5, "계약": 1.0, "승인": 1.0, "성장": 1.0, "돌파": 1.5,
        "매수": 1.0, "강세": 1.0, "반등": 1.0, "기대": 0.5,
        "호재": 2.0, "흑자전환": 2.5, "배당": 1.5, "자사주": 1.5, "목표가상향": 2.0,
    }
    NEG_WORDS = {
        "하락": 1.0, "급락": 2.0, "적자": 1.5, "부진": 1.0, "리스크": 1.0,
        "매도": 1.0, "약세": 1.0, "손실": 1.5, "취소": 1.0, "조사": 1.0,
        "소송": 1.5, "경고": 1.0, "우려": 0.5, "악화": 1.5,
        "악재": 2.0, "신저가": 2.0, "하향": 1.5, "리콜": 2.0, "감자": 2.0,
    }

    sentiment_score = 0.0
    if news_raw:
        scores = []
        for i, n in enumerate(news_raw):
            title = n.get("title", "")
            pos_w = sum(w for k, w in POS_WORDS.items() if k in title)
            neg_w = sum(w for k, w in NEG_WORDS.items() if k in title)
            total_w = pos_w + neg_w
            raw_s = (pos_w - neg_w) / max(total_w, 1.0) if total_w > 0 else 0.0
            recency_w = 1.0 / (1 + i * 0.2)
            scores.append(raw_s * recency_w)
        sentiment_score = sum(scores) / sum(1.0 / (1 + i * 0.2) for i in range(len(scores))) if scores else 0.0

    df["sentiment"] = sentiment_score

    # ── KOSPI benchmark regressor (daily returns) ──
    df["kospi_ret"] = 0.0
    try:
        import FinanceDataReader as fdr
        _kospi_start = df["Date"].min().strftime("%Y-%m-%d")
        _kospi_df = fdr.DataReader("KS11", _kospi_start)
        if _kospi_df is not None and len(_kospi_df) > 5:
            _kospi_close = _kospi_df["Close"].astype(float)
            _kospi_ret_daily = _kospi_close.pct_change().fillna(0.0)
            # Align KOSPI data with stock dates
            _kospi_df_aligned = pd.DataFrame({
                "Date": pd.to_datetime(_kospi_df.index).tz_localize(None),
                "kospi_ret": _kospi_ret_daily.values,
            }).drop_duplicates(subset=["Date"], keep="last")
            df = df.merge(_kospi_df_aligned, on="Date", how="left", suffixes=("_old", ""))
            if "kospi_ret_old" in df.columns:
                df["kospi_ret"] = df["kospi_ret"].fillna(df["kospi_ret_old"])
                df.drop(columns=["kospi_ret_old"], inplace=True)
            df["kospi_ret"] = df["kospi_ret"].fillna(0.0)
    except Exception:
        df["kospi_ret"] = 0.0

    # ── Market Regime Detection ──
    regime = detect_regime(df)

    REGRESSOR_COLS = ["RSI", "MACD_hist", "BB_pct", "OBV_norm", "sentiment", "kospi_ret"]
    df_f = df[["Date", "Close"] + REGRESSOR_COLS].rename(
        columns={"Date": "ds", "Close": "y"}
    )
    df_f[REGRESSOR_COLS] = df_f[REGRESSOR_COLS].ffill().bfill()
    df_f = df_f.dropna(subset=["ds", "y"])  # keep only rows with valid date and price

    # 적응형 Prophet 하이퍼파라미터 (regime-aware)
    if regime in ("bull", "bear"):
        cps = 0.15  # Trending regime: higher changepoint sensitivity
    else:
        cps = 0.05  # Sideways regime: lower changepoint sensitivity

    np.random.seed(42)  # 재현성
    model = Prophet(
        daily_seasonality=False, weekly_seasonality=True,
        yearly_seasonality=True, changepoint_prior_scale=cps,
        n_changepoints=15,
    )
    for reg in REGRESSOR_COLS:
        model.add_regressor(reg, standardize=True)
    model.fit(df_f)

    trading_days = get_kr_trading_days(df["Date"].max(), pred_days)
    future_dates = pd.DataFrame({"ds": pd.to_datetime(trading_days)})

    # 평균회귀 기반 지표 외삽 (adaptive mean-reversion with half-life)
    last_row = df_f.iloc[-1]
    MEAN_TARGETS = {"RSI": 50.0, "BB_pct": 0.5, "MACD_hist": 0.0, "OBV_norm": 0.5, "sentiment": 0.0, "kospi_ret": 0.0}

    # Adaptive theta per regressor: calculate half-life from autocorrelation lag-1
    # half_life = -log(2) / log(autocorrelation_lag1)
    # theta = 1 - exp(-log(2) / half_life), clamped to [0.05, 0.5]
    _mr_regressors = ["RSI", "MACD_hist", "BB_pct", "OBV_norm", "sentiment"]
    REVERT_SPEED = {}
    for _mr_col in _mr_regressors:
        try:
            _series = df_f[_mr_col].dropna()
            if len(_series) > 30:
                _ac1 = float(_series.autocorr(lag=1))
                if 0 < _ac1 < 1:
                    _half_life = -np.log(2) / np.log(_ac1)
                    _theta = 1 - np.exp(-np.log(2) / max(_half_life, 1.0))
                    REVERT_SPEED[_mr_col] = float(np.clip(_theta, 0.05, 0.5))
                else:
                    REVERT_SPEED[_mr_col] = 0.15  # fallback for non-mean-reverting
            else:
                REVERT_SPEED[_mr_col] = 0.15
        except Exception:
            REVERT_SPEED[_mr_col] = 0.15

    # KOSPI daily returns: use mean of last 5 days for extrapolation
    _kospi_last5_mean = float(df_f["kospi_ret"].iloc[-5:].mean()) if len(df_f) >= 5 else 0.0

    for col in REGRESSOR_COLS:
        vals, curr_val = [], float(last_row[col])
        if col == "kospi_ret":
            # Flat extrapolation using mean of last 5 days
            for step in range(1, pred_days + 1):
                vals.append(_kospi_last5_mean)
        else:
            mu = MEAN_TARGETS[col]
            theta = REVERT_SPEED.get(col, 0.15)
            for step in range(1, pred_days + 1):
                curr_val = curr_val + theta * (mu - curr_val)
                vals.append(curr_val)
        future_dates[col] = vals

    future_dates["RSI"] = future_dates["RSI"].clip(0, 100)
    future_dates["BB_pct"] = future_dates["BB_pct"].clip(-0.2, 1.2)

    future_all = pd.concat(
        [df_f[["ds"] + REGRESSOR_COLS], future_dates],
        ignore_index=True
    )
    fc = model.predict(future_all)

    last_date_tmp = df["Date"].max()
    fc_future_tmp = fc[fc["ds"] > last_date_tmp].copy()
    prophet_pred = float(fc_future_tmp.iloc[-1]["yhat"])

    # GradientBoosting 앙상블
    GBR_FEATURES = ["RSI", "MACD_hist", "BB_pct", "OBV_norm", "vol_ratio", "ma20_dist"]
    df_gb = df.copy()
    df_gb[GBR_FEATURES] = df_gb[GBR_FEATURES].ffill().bfill()
    df_gb["target"] = df_gb["Close"].shift(-1)
    df_gb = df_gb.dropna(subset=["target"] + GBR_FEATURES + ["Close"])

    # GBR max_depth: trending regimes get deeper trees
    _gbr_max_depth = 5 if regime in ("bull", "bear") else 3

    gbr_pred = prophet_pred
    if len(df_gb) > 30:
        try:
            X_gb = df_gb[GBR_FEATURES].values
            y_gb = df_gb["target"].values
            gbr = GradientBoostingRegressor(
                n_estimators=100, max_depth=_gbr_max_depth, learning_rate=0.1,
                subsample=0.8, random_state=42
            )
            gbr.fit(X_gb, y_gb)
            last_features = df[GBR_FEATURES].iloc[-1:].values
            gbr_pred = float(gbr.predict(last_features)[0])
        except Exception:
            gbr_pred = prophet_pred

    # Regime-based ensemble weights
    if regime in ("bull", "bear"):
        _w_prophet, _w_gbr = 0.50, 0.50  # Trending: equal weight
    else:  # sideways
        _w_prophet, _w_gbr = 0.70, 0.30  # Sideways: Prophet-heavy
    ensemble_pred = prophet_pred * _w_prophet + gbr_pred * _w_gbr

    # 클램프 (±15% soft, ±30% hard)
    _current_close = float(df["Close"].iloc[-1])
    _max_daily_pct = 0.15
    _hard_limit_pct = 0.30

    _pred_pct_raw = (ensemble_pred - _current_close) / _current_close
    if abs(_pred_pct_raw) > _max_daily_pct:
        ensemble_pred = _current_close * (1 + np.sign(_pred_pct_raw) * _max_daily_pct)
    ensemble_pred = np.clip(ensemble_pred,
                            _current_close * (1 - _hard_limit_pct),
                            _current_close * (1 + _hard_limit_pct))

    # ── Price-RSI Divergence Adjustment ──
    divergence_signal = _detect_divergence(df)
    if divergence_signal == "bearish_div":
        ensemble_pred *= 0.995  # reduce by 0.5%
    elif divergence_signal == "bullish_div":
        ensemble_pred *= 1.005  # increase by 0.5%
    # Re-clamp after divergence adjustment
    ensemble_pred = np.clip(ensemble_pred,
                            _current_close * (1 - _hard_limit_pct),
                            _current_close * (1 + _hard_limit_pct))

    fc_future_tmp["yhat"] = ensemble_pred

    # ── EWMA Volatility-based Dynamic Confidence Intervals ──
    _daily_returns = df["Close"].pct_change().dropna()
    ewma_vol = _ewma_volatility(_daily_returns, span=20)
    _daily_vol = ewma_vol / np.sqrt(252)  # 연간→일간 변환
    _spread_pct = _daily_vol * np.sqrt(pred_days) * 1.2
    _spread_pct = min(_spread_pct, _hard_limit_pct)  # cap at hard limit

    fc_future_tmp["yhat_upper"] = min(ensemble_pred * (1 + _spread_pct),
                                      _current_close * (1 + _hard_limit_pct))
    fc_future_tmp["yhat_lower"] = max(ensemble_pred * (1 - _spread_pct),
                                      _current_close * (1 - _hard_limit_pct))

    # 백테스팅 — proper walk-forward cross-validation (no future data leakage)
    # Expanding window: train on [0:i], predict [i], compare with actual
    # Minimum training window: 60 data points
    backtest_mape = None
    _min_train_window = 60
    _n_test_days = 60
    if len(df_gb) > (_min_train_window + _n_test_days):
        try:
            X_bt = df_gb[GBR_FEATURES].values
            y_bt = df_gb["target"].values
            _total = len(X_bt)
            _test_start = max(_min_train_window, _total - _n_test_days)
            bt_pred_list, bt_actual_list = [], []
            for i in range(_test_start, _total):
                gbr_bt = GradientBoostingRegressor(
                    n_estimators=100, max_depth=_gbr_max_depth, learning_rate=0.1,
                    subsample=0.8, random_state=42
                )
                gbr_bt.fit(X_bt[:i], y_bt[:i])
                bt_pred_list.append(gbr_bt.predict(X_bt[i:i+1])[0])
                bt_actual_list.append(y_bt[i])
            if bt_pred_list:
                bt_pred_arr = np.array(bt_pred_list)
                bt_actual_arr = np.array(bt_actual_list)
                backtest_mape = float(np.mean(np.abs(
                    (bt_actual_arr - bt_pred_arr) / bt_actual_arr)) * 100)
        except Exception:
            backtest_mape = None

    return {
        "fc_future": fc_future_tmp.to_dict("records"),
        "sentiment_score": sentiment_score,
        "backtest_mape": backtest_mape,
        "regime": regime,
        "divergence": divergence_signal,
        "df_with_indicators": df.to_dict("records"),
    }


# ─────────────────────────────────────────────
# 주가 데이터 수집 (pykrx → FDR 폴백)
# ─────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_ohlcv(stock_code: str, days: int = 730) -> pd.DataFrame:
    """pykrx를 1차로, FinanceDataReader를 2차로 사용하여 OHLCV 데이터 수집"""
    end_date = now_kst().strftime("%Y%m%d")
    start_date = (now_kst() - timedelta(days=days)).strftime("%Y%m%d")

    # 1차: pykrx (KRX 공식 데이터)
    try:
        from pykrx import stock as krx_stock
        df = krx_stock.get_market_ohlcv_by_date(start_date, end_date, stock_code)
        if df is not None and len(df) > 30:
            df = df.reset_index()
            df.columns = ["Date", "Open", "High", "Low", "Close", "Volume",
                          *[c for c in df.columns[6:]]]
            df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
            df["Date"] = pd.to_datetime(df["Date"])
            df = df[df["Volume"] > 0].reset_index(drop=True)
            return df
    except Exception:
        pass

    # 2차: FinanceDataReader
    try:
        import FinanceDataReader as fdr
        start_str = (now_kst() - timedelta(days=days)).strftime("%Y-%m-%d")
        df = fdr.DataReader(stock_code, start_str)
        if df is not None and len(df) > 30:
            df = df.reset_index()
            df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
            df["Date"] = pd.to_datetime(df["Date"])
            df = df[df["Volume"] > 0].reset_index(drop=True)
            return df
    except Exception:
        pass

    return pd.DataFrame()


# ─────────────────────────────────────────────
# 실시간 현재가 (네이버 금융 폴링 API)
# ─────────────────────────────────────────────
def fetch_realtime_price(stock_code: str) -> dict | None:
    """네이버 금융 실시간 API로 현재가 조회. 장중에만 유효."""
    try:
        url = f"https://polling.finance.naver.com/api/realtime/domestic/stock/{stock_code}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            datas = data.get("datas", [])
            if datas:
                d = datas[0]
                return {
                    "price": int(d.get("closePrice", 0)),
                    "change": int(d.get("compareToPreviousClosePrice", 0)),
                    "change_pct": float(d.get("fluctuationsRatio", 0)),
                    "volume": int(d.get("accumulatedTradingVolume", 0)),
                    "high": int(d.get("highPrice", 0)),
                    "low": int(d.get("lowPrice", 0)),
                    "open": int(d.get("openPrice", 0)),
                    "time": d.get("localTradedAt", ""),
                }
    except Exception:
        pass
    return None


def fetch_nxt_price(stock_code: str) -> dict | None:
    """네이버 모바일 증권 API에서 NXT(대체거래소) 가격 조회."""
    try:
        url = f"https://m.stock.naver.com/api/stock/{stock_code}/basic"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            nxt = data.get("overMarketPriceInfo")
            if nxt and nxt.get("overPrice"):
                price_str = nxt["overPrice"].replace(",", "")
                change_str = nxt.get("compareToPreviousClosePrice", "0").replace(",", "")
                # API가 이미 부호를 포함하여 반환 (예: "-2,100", "-1.11")
                return {
                    "price": int(price_str),
                    "change": int(change_str),
                    "change_pct": float(nxt.get("fluctuationsRatio", 0)),
                    "status": nxt.get("overMarketStatus", ""),
                    "session": nxt.get("tradingSessionType", ""),
                    "time": nxt.get("localTradedAt", ""),
                }
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────────
def krx_tick(price: float) -> int:
    p = int(round(float(price)))
    if p < 1_000:       unit = 1
    elif p < 5_000:     unit = 5
    elif p < 10_000:    unit = 10
    elif p < 50_000:    unit = 50
    elif p < 100_000:   unit = 100
    elif p < 500_000:   unit = 500
    else:               unit = 1_000
    return int(round(p / unit) * unit)


def _show_fallback_briefing(name, cp, pred_days, pred_end, pred_pct, pred_lower, pred_upper):
    direction = "상승" if pred_pct >= 0 else "하락"
    mag = "강하게" if abs(pred_pct) > 3 else ("완만하게" if abs(pred_pct) > 1 else "소폭")
    summary = (
        f"📊 <b>{name}</b> — Prophet 모델 분석 결과<br><br>"
        f"현재가 <b>{int(cp):,}원</b> 기준으로 향후 {pred_days}영업일 후 "
        f"<b>{pred_end:,}원</b> ({pred_pct:+.2f}%)으로 {mag} {direction}할 것으로 예측됩니다.<br><br>"
        f"신뢰 구간은 <b>{pred_lower:,}원 ~ {pred_upper:,}원</b>이며, "
        f"예측 변동폭은 약 {pred_upper - pred_lower:,}원입니다.<br><br>"
        f"<span style='color:#4a5568;font-size:0.8rem'>"
        f"※ Gemini AI 브리핑을 활성화하려면 Streamlit Secrets에 <code>GEMINI_API_KEY</code>를 등록하세요.</span>"
    )
    st.markdown(f'<div class="ai-box">{summary}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 종목 목록 로딩
# ─────────────────────────────────────────────
all_stocks = load_all_stocks()
all_stocks["label"] = all_stocks["종목명"] + "  (" + all_stocks["종목코드"] + ")"
labels = all_stocks["label"].tolist()

# ─────────────────────────────────────────────
# 상단: 로고 + 시간
# ─────────────────────────────────────────────
_now = now_kst()
_wd = _now.weekday()
_hm = _now.hour * 100 + _now.minute
if _wd >= 5:
    _mkt_status = "🔴 휴장 (주말)"
elif _hm < 800:
    _mkt_status = "🟡 장 시작 전"
elif _hm < 900:
    _mkt_status = "🟢 NXT 프리마켓"
elif _hm <= 1530:
    _mkt_status = "🟢 정규장"
elif _hm <= 2000:
    _mkt_status = "🟢 NXT 애프터마켓"
else:
    _mkt_status = "🔴 장 마감"

st.markdown(f"""<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:2px'>
    <div style='display:flex;align-items:center'>
        {logo_html}
        <span style='font-size:1.05rem;font-weight:700;color:#e2e8f0'>Castle Stock AI</span> &nbsp;<span style='font-size:0.75rem;color:#8b95a5;font-weight:400'>믿지 못할 주식 예측</span>
    </div>
    <div style='font-size:0.68rem;color:#8b95a5;text-align:right'>
        {_mkt_status}<br>{_now.strftime("%H:%M")} KST
    </div>
</div>""", unsafe_allow_html=True)

# 예측기간 1영업일 고정
pred_days = 1

# 장중 여부 판별 (autorefresh는 scanner fragment 안에서만)
_is_market_open = (_wd < 5 and 800 <= _hm <= 2000)

# ─────────────────────────────────────────────
# 전체 화면 탭: 종목 검색 / 추천 종목
# ─────────────────────────────────────────────
_tab_analysis, _tab_scanner = st.tabs(["📊 종목 검색", "🔎 추천 종목"])

# ─── 탭 1: 종목 검색 (검색바 + 분석 결과) ───
with _tab_analysis:
    _col1, _col2 = st.columns([9, 1], vertical_alignment="bottom", gap="small")
    with _col1:
        selected_label = st.selectbox(
            "종목 검색",
            [None] + labels,
            index=0,
            label_visibility="collapsed",
            format_func=lambda x: "종목명 또는 코드를 입력하세요" if x is None else x,
        )
    with _col2:
        _query_btn = st.button("🔍", use_container_width=True)

# ─── 탭 2: 추천 종목 (모닝 스캐너) ── @st.fragment로 검색탭 영향 방지 ───
@st.fragment
def _render_scanner():
    # st_autorefresh 제거: 서드파티 컴포넌트라 fragment 스코프 무시하고 전체 페이지 리런 유발
    # 대신 st.cache_data(ttl=600)의 자연 만료에 의존 — 탭 전환 시 캐시 만료되면 자동 재조회
    _scanner_date = now_kst().strftime("%Y%m%d")
    _scanner_today_str = now_kst().strftime("%Y년 %m월 %d일")

    st.markdown(
        f'<div class="section-header">🔎 퀀트 AI 추천 — {_scanner_today_str} 멀티팩터 Top 5</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="font-size:0.68rem;color:#4a5568;margin:-8px 0 12px 0;line-height:1.6">'
        '모멘텀 · 평균회귀 · 추세품질 · 위험조정 <b>4-Pillar 퀀트 모델</b> 기반 스코어링'
        '</div>',
        unsafe_allow_html=True
    )

    _loading_placeholder = st.empty()
    # ── session_state 캐시: 검색탭 리런 시 블로킹 완전 방지 ──
    _ss_cache_key = f"scanner_df_v5_{_scanner_date}"
    _ss_time_key = f"scanner_time_{_scanner_date}"
    import time as _time_mod
    _cached_df_ss = st.session_state.get(_ss_cache_key)
    _cached_ts = st.session_state.get(_ss_time_key, 0)
    _now_ts = _time_mod.time()
    _elapsed_min = int((_now_ts - _cached_ts) / 60) if _cached_ts else 0

    if _cached_df_ss is not None:
        # 세션 캐시 존재 → 즉시 사용 (블로킹 zero)
        _scanner_df = _cached_df_ss
    else:
        # 최초 로드: 버튼으로 수동 조회
        _load_btn = st.button("🔎 추천 종목 불러오기", key="scanner_load", use_container_width=True)
        if not _load_btn:
            st.markdown(
                '<div style="text-align:center;padding:40px 20px;color:#4a5568;font-size:0.75rem">'
                '버튼을 눌러 오늘의 추천 종목을 조회하세요</div>',
                unsafe_allow_html=True
            )
            return
        # 버튼 클릭 → 조회 (st.cache_data 있으면 즉시, 없으면 계산)
        with st.spinner("🔄 추천 종목 조회 중..."):
            try:
                _scanner_df = run_scanner(_scanner_date)
                st.session_state[_ss_cache_key] = _scanner_df
                st.session_state[_ss_time_key] = _time_mod.time()
            except Exception as _scan_err:
                st.error(f"스캔 중 오류: {_scan_err}")
                return

    # 종목명 매핑
    if not _scanner_df.empty:
        try:
            _all_for_names = load_all_stocks()
            _name_map = dict(zip(_all_for_names["종목코드"], _all_for_names["종목명"]))
            _scanner_df["name"] = _scanner_df["code"].map(lambda c: _name_map.get(c, c))
        except Exception:
            pass

    if _scanner_df.empty:
        st.markdown(
            '<div class="scanner-card" style="text-align:center;color:#4a5568;padding:2rem">'
            '오늘은 뚜렷한 매수 시그널이 감지되지 않았습니다.</div>',
            unsafe_allow_html=True
        )
        return

    _scanner_ai_cache_key = f"scanner_ai_{_scanner_date}"
    if _scanner_ai_cache_key not in st.session_state:
        st.session_state[_scanner_ai_cache_key] = {}

    for _idx, _row in _scanner_df.iterrows():
        _rank = _idx + 1
        _score = _row["score"]
        _chg_arrow = "▲" if _row["change_pct"] >= 0 else "▼"
        _chg_color = "#fc5c5c" if _row["change_pct"] >= 0 else "#4d9fff"
        _signals_html = "".join(f'<span class="signal-tag">{s}</span>' for s in _row["signals"])
        _m = _row.get("momentum", 0)
        _mr = _row.get("mean_rev", 0)
        _t = _row.get("trend", 0)
        _ra = _row.get("risk_adj", 0)
        _su = _row.get("supply", 0)
        _code = _row["code"]
        _cached_ai = st.session_state[_scanner_ai_cache_key].get(_code)
        _score_cls = "score-high" if _score >= 60 else ("score-mid" if _score >= 40 else "score-low")
        # 외국인/기관 순매수 표시용
        _fr = _row.get("foreign_ratio", 0.0)
        _amt = _row.get("amount", 0)
        if _amt >= 1_000_000_000_000:
            _amt_str = f"{_amt/1_000_000_000_000:.1f}조"
        elif _amt >= 100_000_000:
            _amt_str = f"{_amt/100_000_000:.0f}억"
        else:
            _amt_str = f"{_amt/100_000_000:.1f}억"
        _fr_html = f'<span style="color:#4d9fff">외인보유 {_fr:.1f}%</span>' if _fr >= 1 else ""

        # ── 통합 카드: HTML 하나로 렌더 + AI 버튼만 별도 form ──
        _submitted = False

        # ── 카드 상단 HTML ──
        _mr_v = max(_mr, 0.5)
        _ra_v = max(_ra, 0.5)
        _su_v = max(_su, 0.5)
        _vr = f"{_row.get('vol_ratio',0):.1f}"

        _card_top = f'''<div class="scanner-card">
<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
<span class="scanner-rank rank-{_rank if _rank <= 3 else "other"}">{_rank}</span>
<span style="font-size:0.88rem;font-weight:600;color:#e2e8f0">{_row["name"]}</span>
<span style="font-size:0.68rem;color:#4a5568">{_code}</span>
<span class="scanner-score {_score_cls}">{_score:.0f}/100</span>
</div>
<div style="display:flex;align-items:baseline;gap:10px;margin-bottom:8px">
<span style="font-family:monospace;font-size:1rem;font-weight:600;color:#e2e8f0">{_row["price"]:,}원</span>
<span style="font-size:0.8rem;color:{_chg_color};font-weight:600">{_chg_arrow} {abs(_row["change_pct"]):.2f}%</span>
</div>
<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:6px">{_signals_html}</div>
<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;font-size:0.65rem;color:#8892a4">
<span>RSI {_row["rsi"]}</span><span>ADX {_row.get("adx",0)}</span>
<span>Sharpe {_row.get("sharpe",0)}</span>
<span>거래량 {_vr}x</span>
<span style="color:#f5a623">거래대금 {_amt_str}</span>
{_fr_html}</div></div>'''

        # ── 바 차트 HTML (별도 렌더) ──
        _bar_html = f'''<div style="display:flex;gap:3px;height:24px;font-size:0.7rem;font-family:sans-serif;font-weight:600;line-height:24px;margin:-12px 0 4px 0">
<div style="flex:{_m};background:#3a7bd5;color:#fff;text-align:center;border-radius:4px 0 0 4px;overflow:hidden">모멘텀 {_m:.0f}</div>
<div style="flex:{_mr_v};background:#e8961f;color:#fff;text-align:center;overflow:hidden">진입 {_mr:.0f}</div>
<div style="flex:{_t};background:#2d9f99;color:#fff;text-align:center;overflow:hidden">추세 {_t:.0f}</div>
<div style="flex:{_ra_v};background:#805ad5;color:#fff;text-align:center;overflow:hidden">리스크 {_ra:.0f}</div>
<div style="flex:{_su_v};background:#c53030;color:#fff;text-align:center;border-radius:0 4px 4px 0;overflow:hidden">수급 {_su:.0f}</div>
</div>
<div style="font-size:0.55rem;color:#4a5568;margin-bottom:8px">모멘텀 /25 · 진입 /15 · 추세 /20 · 리스크 /15 · 수급 /25</div>'''

        # ── 렌더 ──
        _submitted = False
        if not _cached_ai and "GEMINI_API_KEY" in st.secrets:
            with st.form(key=f"scanner_{_code}", clear_on_submit=False, border=False):
                _submitted = st.form_submit_button("✦ AI 예측")

        st.markdown(_card_top, unsafe_allow_html=True)
        st.markdown(_bar_html, unsafe_allow_html=True)

        # AI 결과 (카드 HTML 밖 — Gemini 응답이 카드를 깨뜨리지 않음)
        if _cached_ai:
            _safe_ai = html_mod.escape(str(_cached_ai))
            for _tag in ["br", "strong", "/strong", "b", "/b", "em", "/em"]:
                _safe_ai = _safe_ai.replace(f"&lt;{_tag}&gt;", f"<{_tag}>")
            _safe_ai = _safe_ai.replace("\n", "<br>")
            st.markdown(
                '<div style="margin-top:-6px;margin-bottom:8px;padding:12px 14px;'
                'background:linear-gradient(135deg,#0c1525 0%,#111d30 100%);'
                'border:1px solid #1e3a5f;border-left:3px solid #3b82f6;border-radius:0 0 6px 6px">'
                '<div style="font-size:0.58rem;font-weight:600;color:#3b82f6;'
                'letter-spacing:1.5px;margin-bottom:8px">✦ AI PREDICTION</div>'
                f'<div style="font-size:0.76rem;line-height:1.8;color:#cbd5e0">{_safe_ai}</div>'
                '</div>',
                unsafe_allow_html=True
            )

        # AI 호출 처리
        if _submitted:
            with st.spinner("🤖 AI 분석 중..."):
                _ai_text = fetch_scanner_briefing(_code, _row.to_dict(), _scanner_date)
            if _ai_text:
                st.session_state[_scanner_ai_cache_key][_code] = _ai_text
                st.rerun(scope="fragment")

    st.markdown(
        '<div style="text-align:center;color:#4a5568;font-size:0.6rem;margin-top:1rem">'
        '📊 4-Pillar 퀀트모델: 모멘텀(35) + 평균회귀(20) + 추세품질(25) + 위험조정(20) = 100점'
        ' · 스캔 10분 캐시 · AI 버튼 클릭 시 1회 호출 후 캐시</div>',
        unsafe_allow_html=True
    )

with _tab_scanner:
    _render_scanner()


with _tab_analysis:

    # ─── 종목 분석 로직 (탭 밖에서 계산, 결과는 _tab_analysis에 렌더) ───

    if selected_label is None and not st.session_state.get("cached_data_key"):
        st.stop()

    # 조회 버튼 클릭 시 캐시 초기화 (재조회 가능)
    if _query_btn:
        for _k in ["cached_data_key", "cached_df", "cached_news_raw",
                   "cached_news_status", "cached_ai_html", "cached_ai_src",
                   "cached_fc_future", "cached_sentiment", "cached_backtest"]:
            st.session_state[_k] = None

    if not _query_btn:
        if st.session_state.get("cached_data_key") is None:
            st.stop()
        else:
            ticker_cached = st.session_state["cached_data_key"].split("__")[0]
            if selected_label is not None:
                sel_row_tmp = all_stocks[all_stocks["label"] == selected_label].iloc[0]
                if ticker_cached != sel_row_tmp["ticker"]:
                    selected_label = all_stocks[all_stocks["ticker"] == ticker_cached]["label"].values[0]
            else:
                selected_label = all_stocks[all_stocks["ticker"] == ticker_cached]["label"].values[0]

    sel_row      = all_stocks[all_stocks["label"] == selected_label].iloc[0]
    ticker       = sel_row["ticker"]
    stock_code   = sel_row["종목코드"]
    market       = sel_row["시장"]
    display_name = sel_row["종목명"]

    # ─────────────────────────────────────────────
    # 시장 배지
    # ─────────────────────────────────────────────
    badge_class = "badge-kospi" if market == "KOSPI" else "badge-kosdaq"
    st.markdown(f'<span class="badge {badge_class}">{market}</span>', unsafe_allow_html=True)


    # ─────────────────────────────────────────────
    # session_state 캐시 키 초기화
    # ─────────────────────────────────────────────
    _data_key = f"{ticker}__{int(pred_days)}"
    for _k in ["cached_data_key", "cached_df", "cached_news_raw",
               "cached_news_status", "cached_ai_html", "cached_ai_src",
               "cached_fc_future", "cached_sentiment", "cached_backtest"]:
        if _k not in st.session_state:
            st.session_state[_k] = None


    # ─────────────────────────────────────────────
    # 데이터 로딩
    # ─────────────────────────────────────────────
    if ticker:
        try:
            _need_reload = (st.session_state["cached_data_key"] != _data_key)

            if _need_reload:
                with st.spinner("📡 데이터를 불러오는 중..."):
                    df = fetch_stock_ohlcv(stock_code, days=730)

                    def fetch_naver_news(stock_name: str, max_items: int = 5) -> tuple[list, str]:
                        import re
                        from urllib.parse import urlparse
                        if "NAVER_CLIENT_ID" not in st.secrets or "NAVER_CLIENT_SECRET" not in st.secrets:
                            return [], "no_key"
                        NAVER_HEADERS = {
                            "X-Naver-Client-Id":     st.secrets["NAVER_CLIENT_ID"],
                            "X-Naver-Client-Secret": st.secrets["NAVER_CLIENT_SECRET"],
                        }
                        def _call(query, display):
                            try:
                                resp = requests.get(
                                    "https://openapi.naver.com/v1/search/news.json",
                                    params={"query": query, "display": display, "sort": "date"},
                                    headers=NAVER_HEADERS, timeout=8,
                                )
                                return resp.json().get("items", []), resp.status_code
                            except Exception:
                                return [], -1
                        def _parse(raw_items):
                            parsed, seen = [], set()
                            for it in raw_items:
                                title = re.sub(r"<[^>]+>", "", it.get("title", "")).strip()
                                if not title or title in seen: continue
                                seen.add(title)
                                pub_raw = it.get("pubDate", "")
                                try: pub = datetime.strptime(pub_raw, "%a, %d %b %Y %H:%M:%S %z").strftime("%m/%d")
                                except Exception: pub = pub_raw[:10]
                                parsed.append({
                                    "title": title,
                                    "link":  it.get("originallink") or it.get("link", "#"),
                                    "pub":   pub,
                                    "desc":  re.sub(r"<[^>]+>", "", it.get("description", "")).strip(),
                                })
                            return parsed
                        FIN_DOMAINS = {
                            "hankyung.com","mk.co.kr","sedaily.com","edaily.co.kr",
                            "thebell.co.kr","infostock.co.kr","news.einfomax.co.kr",
                            "businesspost.co.kr","investchosun.com","newsis.com",
                            "yna.co.kr","etnews.com","finance.naver.com",
                        }
                        NOISE_WORDS = {
                            "패션","맛집","여행","스타일","뷰티","연애","요리",
                            "육아","인테리어","레시피","유튜브","BJ","아이돌",
                            "살 빠지는","모발","단발","젤리핑","서울시교육청",
                        }
                        def _is_noise(t): return any(w in t for w in NOISE_WORDS)
                        def _domain(url):
                            try: return __import__('urllib.parse', fromlist=['urlparse']).urlparse(url).netloc.replace("www.","")
                            except: return ""
                        try:
                            all_items, seen_titles = [], set()
                            status_codes = []
                            for query in [f"{stock_name} 증권 주가", f"{stock_name} 주식"]:
                                raw, sc = _call(query, 20)
                                status_codes.append(sc)
                                for it in _parse(raw):
                                    if it["title"] not in seen_titles and not _is_noise(it["title"]):
                                        seen_titles.add(it["title"])
                                        it["is_fin"] = any(fd in _domain(it["link"]) for fd in FIN_DOMAINS)
                                        all_items.append(it)
                            all_items.sort(key=lambda x: (0 if x["is_fin"] else 1))
                            if not all_items: return [], f"empty:{status_codes}"
                            return all_items[:max_items], "ok"
                        except requests.exceptions.Timeout:
                            return [], "api_error:요청 시간 초과"
                        except Exception as ex:
                            return [], f"api_error:{ex}"

                    news_raw, news_status = fetch_naver_news(display_name, max_items=5)
                    news_txt = " ".join([n["title"] for n in news_raw])

                st.session_state["cached_data_key"]    = _data_key
                st.session_state["cached_df"]          = df
                st.session_state["cached_news_raw"]    = news_raw
                st.session_state["cached_news_status"] = news_status
                st.session_state["cached_ai_html"]     = None
                st.session_state["cached_ai_src"]      = None
                st.session_state["cached_fc_future"]   = None
                st.session_state["cached_sentiment"]   = None
                st.session_state["cached_backtest"]    = None

            else:
                df          = st.session_state["cached_df"]
                news_raw    = st.session_state["cached_news_raw"]
                news_status = st.session_state["cached_news_status"]
                news_txt    = " ".join([n["title"] for n in news_raw])

            if df.empty:
                st.error("⚠️ 데이터를 불러올 수 없습니다. 티커를 확인해 주세요.")
                st.stop()

            df.loc[df["Open"] <= 0, "Open"] = df["Close"]
            df.loc[df["High"] <= 0, "High"] = df["Close"]
            df.loc[df["Low"]  <= 0, "Low"]  = df["Close"]

            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
            df["time"] = df["Date"].dt.strftime("%Y-%m-%d")
            df["MA3"]   = df["Close"].rolling(3).mean()
            df["MA5"]   = df["Close"].rolling(5).mean()
            df["MA10"]  = df["Close"].rolling(10).mean()
            df["MA20"]  = df["Close"].rolling(20).mean()
            df["MA60"]  = df["Close"].rolling(60).mean()
            df["MA120"] = df["Close"].rolling(120).mean()

            cp      = df.iloc[-1]["Close"]
            pp      = df.iloc[-2]["Close"]
            diff    = cp - pp
            pct     = (diff / pp) * 100
            high52  = df["High"].max()
            low52   = df["Low"].min()
            avg_vol = int(df["Volume"].tail(20).mean())

            # ─────────────────────────────────────────────
            # Prophet + GBR 앙상블 예측 (날짜 기반 캐시 → 일관성 보장)
            # ─────────────────────────────────────────────
            # 장중 당일 데이터 보완 (차트/메트릭용 df에만 적용)
            try:
                from pykrx import stock as _krx
                _today_str_krx = now_kst().strftime("%Y%m%d")
                _today_ohlcv = _krx.get_market_ohlcv_by_date(_today_str_krx, _today_str_krx, stock_code)
                if _today_ohlcv is not None and len(_today_ohlcv) > 0:
                    _tr = _today_ohlcv.iloc[-1]
                    _today_date = pd.Timestamp(_today_ohlcv.index[-1])
                    if _today_date.tz is not None:
                        _today_date = _today_date.tz_localize(None)
                    _last_date_check = pd.to_datetime(df["Date"]).dt.tz_localize(None).max()
                    if _today_date > _last_date_check and float(_tr["종가"]) > 0:
                        _new = pd.DataFrame([{
                            "Date": _today_date,
                            "Open": float(_tr["시가"]), "High": float(_tr["고가"]),
                            "Low": float(_tr["저가"]), "Close": float(_tr["종가"]),
                            "Volume": float(_tr["거래량"]),
                        }])
                        df = pd.concat([df, _new], ignore_index=True)
            except Exception:
                pass

            with st.spinner("🤖 기술지표 계산 및 AI 예측 모델 실행 중..."):
                _pred_date_key = now_kst().strftime("%Y%m%d")
                _last_date_str = df["Date"].max().strftime("%Y%m%d")
                _pred_result = compute_prediction(
                    stock_code, _pred_date_key, pred_days,
                    _last_date_str,
                    json.dumps(news_raw, ensure_ascii=False),
                )

            # 예측 결과 복원
            fc_future = pd.DataFrame(_pred_result["fc_future"])
            fc_future["ds"] = pd.to_datetime(fc_future["ds"])
            sentiment_score = _pred_result["sentiment_score"]
            backtest_mape = _pred_result["backtest_mape"]

            # 예측 함수에서 계산된 지표를 df에 반영 (차트용)
            _df_ind = pd.DataFrame(_pred_result["df_with_indicators"])
            _ind_cols = ["RSI", "MACD_hist", "BB_pct", "OBV_norm", "vol_ratio", "ma20_dist", "sentiment", "kospi_ret"]
            if len(df) == len(_df_ind):
                for _ind_col in _ind_cols:
                    if _ind_col in _df_ind.columns:
                        df[_ind_col] = _df_ind[_ind_col].values
            else:
                # df가 장중 보완으로 1행 더 길 경우, 마지막 행은 이전값으로 채움
                for _ind_col in _ind_cols:
                    if _ind_col in _df_ind.columns:
                        _vals = list(_df_ind[_ind_col].values)
                        while len(_vals) < len(df):
                            _vals.append(_vals[-1] if _vals else 0)
                        df[_ind_col] = _vals[:len(df)]

            pred_end    = krx_tick(fc_future.iloc[-1]["yhat"])
            pred_upper  = krx_tick(fc_future.iloc[-1]["yhat_upper"])
            pred_lower  = krx_tick(fc_future.iloc[-1]["yhat_lower"])
            pred_change = pred_end - int(cp)
            pred_pct    = (pred_change / cp) * 100

            cur_rsi     = round(float(df["RSI"].iloc[-1]), 1)
            cur_macd    = round(float(df["MACD_hist"].iloc[-1]), 2)
            cur_bb      = round(float(df["BB_pct"].iloc[-1]) * 100, 1)
            rsi_signal  = "과매수" if cur_rsi > 70 else ("과매도" if cur_rsi < 30 else "중립")
            macd_signal = "상승" if cur_macd > 0 else "하락"
            bb_signal   = "상단돌파" if cur_bb > 80 else ("하단이탈" if cur_bb < 20 else "중립")

            # ─────────────────────────────────────────────
            # 상단 메트릭
            # ─────────────────────────────────────────────
            color_cls  = "metric-up" if diff >= 0 else "metric-down"
            arrow      = "▲" if diff >= 0 else "▼"
            pred_color = "metric-up" if pred_change >= 0 else "metric-down"
            pred_arrow = "▲" if pred_change >= 0 else "▼"
            rsi_color  = "#fc5c5c" if cur_rsi > 70 else ("#4d9fff" if cur_rsi < 30 else "#a0aec0")

            st.markdown(f"""
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">현재가</div>
                    <div class="metric-value">{int(cp):,}<span class="unit">원</span></div>
                    <div class="{color_cls} metric-sub">{arrow} {abs(int(diff)):,}원 ({pct:+.2f}%)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">내일 예측가 (앙상블)</div>
                    <div class="metric-value {pred_color}">{pred_end:,}<span class="unit">원</span></div>
                    <div class="{pred_color} metric-sub">{pred_arrow} {abs(pred_change):,}원 ({pred_pct:+.2f}%)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">예측 범위</div>
                    <div class="metric-value" style="font-size:0.9rem"><span style="color:#00cec9">{pred_lower:,}</span> ~ <span style="color:#a55eea">{pred_upper:,}</span></div>
                    <div class="metric-sub" style="color:#4a5568">신뢰 구간 (하한~상한)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">예측 오차율</div>
                    <div class="metric-value" style="font-size:0.95rem;color:{'#4dc98f' if backtest_mape is not None and backtest_mape < 3 else '#f9a825' if backtest_mape is not None and backtest_mape < 5 else '#fc5c5c' if backtest_mape is not None else '#4a5568'}">{f'{backtest_mape:.1f}%' if backtest_mape is not None else '-'}</div>
                    <div class="metric-sub" style="color:#4a5568">{'MAPE 60일 백테스트' if backtest_mape is not None else '데이터 부족'}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">기술지표</div>
                    <div class="metric-value" style="font-size:0.9rem;line-height:1.5">
                        <span style="color:{rsi_color}">RSI {cur_rsi}</span>
                        <span style="color:#4a5568"> · </span>
                        <span style="color:{'#fc5c5c' if cur_macd>0 else '#4d9fff'}">MACD {'▲' if cur_macd>0 else '▼'}</span>
                        <span style="color:#4a5568"> · </span>
                        <span style="color:#a0aec0">BB {cur_bb}%</span>
                    </div>
                    <div class="metric-sub" style="color:#4a5568">{rsi_signal} · MACD {macd_signal} · {bb_signal}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ─────────────────────────────────────────────
            # 실시간 현재가 + NXT 배너
            # ─────────────────────────────────────────────
            if _is_market_open:
                # 정규장 실시간 (09:00~15:30)
                _rt = fetch_realtime_price(stock_code) if (900 <= _hm <= 1530) else None
                if _rt and _rt["price"] > 0:
                    _rt_color = "#fc5c5c" if _rt["change"] >= 0 else "#4d9fff"
                    _rt_arrow = "▲" if _rt["change"] >= 0 else "▼"
                    _rt_time = _rt["time"][-8:] if len(_rt["time"]) > 8 else _rt["time"]
                    st.markdown(
                        f'<div style="background:linear-gradient(90deg,#131929 0%,#0f1a2e 100%);'
                        f'border:1px solid #1e3a5f;border-radius:6px;padding:8px 14px;'
                        f'margin-bottom:0.5rem;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px">'
                        f'<div style="display:flex;align-items:center;gap:12px">'
                        f'<span style="font-size:0.6rem;color:#4dc98f;font-weight:600;letter-spacing:1px">LIVE</span>'
                        f'<span style="font-family:JetBrains Mono,monospace;font-size:1.15rem;font-weight:700;color:#e2e8f0">'
                        f'{_rt["price"]:,}<span style="font-size:0.75rem;color:#4a5568">원</span></span>'
                        f'<span style="font-size:0.85rem;color:{_rt_color}">'
                        f'{_rt_arrow} {abs(_rt["change"]):,}원 ({_rt["change_pct"]:+.2f}%)</span>'
                        f'</div>'
                        f'<div style="display:flex;gap:14px;font-size:0.7rem;color:#4a5568">'
                        f'<span>시가 {_rt["open"]:,}</span>'
                        f'<span>고가 <span style="color:#fc5c5c">{_rt["high"]:,}</span></span>'
                        f'<span>저가 <span style="color:#4d9fff">{_rt["low"]:,}</span></span>'
                        f'<span>거래량 {_rt["volume"]:,}</span>'
                        f'<span>{_rt_time}</span>'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )

                # NXT 가격 (프리마켓 / 애프터마켓)
                _nxt = fetch_nxt_price(stock_code)
                if _nxt and _nxt["price"] > 0:
                    _nxt_color = "#fc5c5c" if _nxt["change"] >= 0 else "#4d9fff"
                    _nxt_arrow = "▲" if _nxt["change"] >= 0 else "▼"
                    _nxt_session = "프리마켓" if _nxt["session"] == "PRE_MARKET" else "애프터마켓"
                    _nxt_status_dot = "🟢" if _nxt["status"] == "OPEN" else "⚫"
                    _nxt_time = _nxt["time"][-14:-6] if len(_nxt["time"]) > 14 else _nxt["time"]
                    st.markdown(
                        f'<div style="background:linear-gradient(90deg,#1a1a0e 0%,#1a1508 100%);'
                        f'border:1px solid #3d3a1e;border-radius:6px;padding:6px 14px;'
                        f'margin-bottom:0.5rem;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px">'
                        f'<div style="display:flex;align-items:center;gap:12px">'
                        f'<span style="font-size:0.6rem;color:#f9a825;font-weight:600;letter-spacing:1px">{_nxt_status_dot} NXT {_nxt_session}</span>'
                        f'<span style="font-family:JetBrains Mono,monospace;font-size:1.05rem;font-weight:700;color:#e2e8f0">'
                        f'{_nxt["price"]:,}<span style="font-size:0.75rem;color:#4a5568">원</span></span>'
                        f'<span style="font-size:0.8rem;color:{_nxt_color}">'
                        f'{_nxt_arrow} {abs(_nxt["change"]):,}원 ({_nxt["change_pct"]:+.2f}%)</span>'
                        f'</div>'
                        f'<div style="font-size:0.68rem;color:#4a5568">{_nxt_time}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            else:
                # 장 마감 후에도 NXT 종가 표시
                _nxt = fetch_nxt_price(stock_code)
                if _nxt and _nxt["price"] > 0:
                    _nxt_color = "#fc5c5c" if _nxt["change"] >= 0 else "#4d9fff"
                    _nxt_arrow = "▲" if _nxt["change"] >= 0 else "▼"
                    _nxt_session = "프리마켓" if _nxt["session"] == "PRE_MARKET" else "애프터마켓"
                    _nxt_time = _nxt["time"][-14:-6] if len(_nxt["time"]) > 14 else _nxt["time"]
                    st.markdown(
                        f'<div style="background:linear-gradient(90deg,#1a1a0e 0%,#1a1508 100%);'
                        f'border:1px solid #3d3a1e;border-radius:6px;padding:6px 14px;'
                        f'margin-bottom:0.5rem;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px">'
                        f'<div style="display:flex;align-items:center;gap:12px">'
                        f'<span style="font-size:0.6rem;color:#f9a825;font-weight:600;letter-spacing:1px">NXT {_nxt_session} (마감)</span>'
                        f'<span style="font-family:JetBrains Mono,monospace;font-size:1.05rem;font-weight:700;color:#e2e8f0">'
                        f'{_nxt["price"]:,}<span style="font-size:0.75rem;color:#4a5568">원</span></span>'
                        f'<span style="font-size:0.8rem;color:{_nxt_color}">'
                        f'{_nxt_arrow} {abs(_nxt["change"]):,}원 ({_nxt["change_pct"]:+.2f}%)</span>'
                        f'</div>'
                        f'<div style="font-size:0.68rem;color:#4a5568">{_nxt_time}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            # ─────────────────────────────────────────────
            # 차트
            # ─────────────────────────────────────────────
            st.markdown('<div class="section-header">📊 프로 차트 — TradingView Engine</div>', unsafe_allow_html=True)

            cdl, m3, m5, m10, m20, m60, m120, vol = [], [], [], [], [], [], [], []
            df_chart = df.dropna(subset=["Open","High","Low","Close","Volume"]).copy()

            for _, r in df_chart.iterrows():
                tm = r["time"]
                o, h, l, c = float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"])
                if any(v != v or abs(v) == float("inf") for v in [o, h, l, c]):
                    continue
                cdl.append({"time":tm,"open":krx_tick(o),"high":krx_tick(h),"low":krx_tick(l),"close":krx_tick(c)})
                if pd.notna(r["MA3"]):   m3.append({"time":tm,"value":int(krx_tick(float(r["MA3"])))})
                if pd.notna(r["MA5"]):   m5.append({"time":tm,"value":int(krx_tick(float(r["MA5"])))})
                if pd.notna(r["MA10"]):  m10.append({"time":tm,"value":int(krx_tick(float(r["MA10"])))})
                if pd.notna(r["MA20"]):  m20.append({"time":tm,"value":int(krx_tick(float(r["MA20"])))})
                if pd.notna(r["MA60"]):  m60.append({"time":tm,"value":int(krx_tick(float(r["MA60"])))})
                if pd.notna(r["MA120"]): m120.append({"time":tm,"value":int(krx_tick(float(r["MA120"])))})
                vol_val = float(r["Volume"])
                if vol_val == vol_val:
                    vc = "rgba(252,92,92,0.55)" if c >= o else "rgba(77,159,255,0.55)"
                    vol.append({"time":tm,"value":int(vol_val),"color":vc})

            pred_line = [{"time":df.iloc[-1]["time"],"value":krx_tick(float(cp))}]
            for _, r in fc_future.iterrows():
                pred_line.append({"time":r["ds"].strftime("%Y-%m-%d"),"value":krx_tick(r["yhat"])})

            pred_upper_line = [{"time":df.iloc[-1]["time"],"value":krx_tick(float(cp))}]
            pred_lower_line = [{"time":df.iloc[-1]["time"],"value":krx_tick(float(cp))}]
            for _, r in fc_future.iterrows():
                td = r["ds"].strftime("%Y-%m-%d")
                pred_upper_line.append({"time":td,"value":krx_tick(r["yhat_upper"])})
                pred_lower_line.append({"time":td,"value":krx_tick(r["yhat_lower"])})

            total_bars = len(cdl)
            zoom_from  = cdl[max(0, total_bars-60)]["time"] if cdl else None

            import streamlit.components.v1 as components
            import json as _json

            _chart_data = _json.dumps({
                "cdl":cdl,"vol":vol,
                "m3":m3,"m5":m5,"m10":m10,"m20":m20,"m60":m60,"m120":m120,
                "pred_line":pred_line,"pred_upper":pred_upper_line,"pred_lower":pred_lower_line,
                "pred_days":pred_days,"zoom_from":zoom_from,"cp":int(cp),
            })

            components.html(f"""
    <!DOCTYPE html>
    <html>
    <head>
    <script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
    <script>if(typeof LightweightCharts==='undefined')document.write('<scr'+'ipt src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"><\\/scr'+'ipt>')</script>
    <style>
      body {{ margin:0; background:#0b0e17; overflow:hidden; }}
      #chart-container {{ position:relative; width:100%; height:380px; }}
      #legend {{ display:flex; flex-wrap:wrap; gap:6px; padding:4px 2px; font-family:'JetBrains Mono',monospace; }}
      .leg-item {{ font-size:0.62rem; cursor:pointer; padding:2px 6px; border-radius:3px; border:1px solid #2d3a55; user-select:none; transition:opacity 0.2s; }}
      .leg-item.hidden {{ opacity:0.3; text-decoration:line-through; }}
      #hover-info {{ position:absolute; top:10px; left:10px; z-index:10; font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#a0aec0; pointer-events:none; text-shadow:1px 1px 2px #0b0e17; }}
    </style>
    </head>
    <body>
    <div id="legend"></div>
    <div id="chart-container">
      <div id="hover-info"></div>
      <div id="chart" style="width:100%;height:100%;"></div>
    </div>
    <script>
    const D = {_chart_data};
    function tryInit(attempt) {{
      if (typeof LightweightCharts !== 'undefined') {{ initChart(); return; }}
      if (attempt < 3) {{
        setTimeout(function(){{ tryInit(attempt+1); }}, 800);
      }} else {{
        document.getElementById('chart').innerHTML = '<div style="color:#fc5c5c;text-align:center;padding:80px 20px;font-size:0.85rem">⚠️ 차트 라이브러리 로딩 실패 — 페이지를 새로고침 해주세요</div>';
      }}
    }}
    tryInit(0);
    function initChart() {{
    const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
      layout:{{ background:{{type:'Solid',color:'#0b0e17'}}, textColor:'#a0aec0' }},
      grid:{{ vertLines:{{color:'#1a2030'}}, horzLines:{{color:'#1a2030'}} }},
      crosshair:{{ mode:1 }},
      rightPriceScale:{{ borderColor:'#1e2435' }},
      timeScale:{{ borderColor:'#1e2435', barSpacing:8, rightOffset:10 }},
      localization:{{ priceFormatter: p => Math.round(p).toLocaleString('ko-KR') }},
    }});
    const candle = chart.addCandlestickSeries({{
      upColor:'#fc5c5c', downColor:'#4d9fff', borderVisible:false,
      wickUpColor:'#fc5c5c', wickDownColor:'#4d9fff',
      priceFormat:{{type:'price',precision:0,minMove:1}},
      lastValueVisible:false, priceLineVisible:false,
    }});
    candle.setData(D.cdl);
    candle.createPriceLine({{ price:D.cp, color:'rgba(0,0,0,0)', lineWidth:0, lineStyle:2, axisLabelVisible:true, title:'현재가' }});
    const volSeries = chart.addHistogramSeries({{
      color:'rgba(120,120,120,0.5)', priceScaleId:'vol',
      lastValueVisible:false, priceLineVisible:false,
      priceFormat:{{type:'volume',precision:0,minMove:1}},
    }});
    chart.priceScale('vol').applyOptions({{ scaleMargins:{{top:0.85,bottom:0}}, borderVisible:false }});
    volSeries.setData(D.vol);
    const maList = [
      {{color:'#ff6b9d',width:1,  label:'3MA',  data:D.m3,  vis:false}},
      {{color:'#00e676',width:1,  label:'5MA',  data:D.m5,  vis:true}},
      {{color:'#00bcd4',width:1,  label:'10MA', data:D.m10, vis:true}},
      {{color:'#ffb300',width:1.5,label:'20MA', data:D.m20, vis:false}},
      {{color:'#e040fb',width:1,  label:'60MA', data:D.m60, vis:false}},
      {{color:'#ff6d00',width:1,  label:'120MA',data:D.m120,vis:false}},
    ];
    const seriesMap = {{}};
    maList.forEach(m => {{
      const s = chart.addLineSeries({{
        color:m.color, lineWidth:m.width, priceFormat:{{type:'price',precision:0,minMove:1}},
        lastValueVisible:m.vis, priceLineVisible:false, visible:m.vis, title:m.label,
      }});
      s.setData(m.data);
      seriesMap[m.label] = s;
    }});
    const predSeries = chart.addLineSeries({{
      color:'#f9a825', lineWidth:2, lineStyle:1,
      priceFormat:{{type:'price',precision:0,minMove:1}},
      lastValueVisible:true, priceLineVisible:false, title:`AI예측(${{D.pred_days}}일)`,
    }});
    predSeries.setData(D.pred_line);
    const upperSeries = chart.addLineSeries({{
      color:'#a55eea', lineWidth:1, lineStyle:2,
      priceFormat:{{type:'price',precision:0,minMove:1}},
      lastValueVisible:true, priceLineVisible:false, title:'예측상단',
    }});
    upperSeries.setData(D.pred_upper);
    const lowerSeries = chart.addLineSeries({{
      color:'#00cec9', lineWidth:1, lineStyle:2,
      priceFormat:{{type:'price',precision:0,minMove:1}},
      lastValueVisible:true, priceLineVisible:false, title:'예측하단',
    }});
    lowerSeries.setData(D.pred_lower);
    chart.subscribeCrosshairMove((param) => {{
      const info = document.getElementById('hover-info');
      if (param.time && param.seriesData) {{
        const d = param.seriesData.get(candle);
        const v = param.seriesData.get(volSeries);
        if (d) {{
          const vol = v && v.value ? v.value.toLocaleString('ko-KR') : '-';
          info.innerHTML = `시가:<span style="color:#e2e8f0">${{d.open.toLocaleString('ko-KR')}}</span> 고가:<span style="color:#e2e8f0">${{d.high.toLocaleString('ko-KR')}}</span> 저가:<span style="color:#e2e8f0">${{d.low.toLocaleString('ko-KR')}}</span> 종가:<span style="color:#e2e8f0">${{d.close.toLocaleString('ko-KR')}}</span> <span style="margin-left:8px">거래량:<span style="color:#e2e8f0">${{vol}}</span></span>`;
        }}
      }} else {{ info.innerHTML=''; }}
    }});
    const legend = document.getElementById('legend');
    const legItems = [
      ...maList.map(m=>({{label:m.label,color:m.color,vis:m.vis}})),
      {{label:'AI예측',color:'#f9a825',vis:true}},
      {{label:'예측상단',color:'#a55eea',vis:true}},
      {{label:'예측하단',color:'#00cec9',vis:true}},
    ];
    legItems.forEach(item => {{
      const el = document.createElement('span');
      el.className = 'leg-item' + (item.vis ? '' : ' hidden');
      el.style.color = item.color;
      el.style.borderColor = item.color + '66';
      el.textContent = '━ ' + item.label;
      el.onclick = () => {{
        const isHidden = el.classList.toggle('hidden');
        const show = !isHidden;
        if (item.label==='AI예측') predSeries.applyOptions({{visible:show,lastValueVisible:show,priceLineVisible:false}});
        else if (item.label==='예측상단') upperSeries.applyOptions({{visible:show,lastValueVisible:show,priceLineVisible:false}});
        else if (item.label==='예측하단') lowerSeries.applyOptions({{visible:show,lastValueVisible:show,priceLineVisible:false}});
        else if (seriesMap[item.label]) seriesMap[item.label].applyOptions({{visible:show,lastValueVisible:show,priceLineVisible:false}});
      }};
      legend.appendChild(el);
    }});
    new ResizeObserver(() => {{
      chart.applyOptions({{width:document.getElementById('chart-container').clientWidth}});
    }}).observe(document.getElementById('chart-container'));
    if (D.zoom_from) {{
      const last_dt = D.pred_line.length>0 ? D.pred_line[D.pred_line.length-1].time : D.cdl[D.cdl.length-1].time;
      chart.timeScale().setVisibleRange({{from:D.zoom_from, to:last_dt}});
    }}
    }}  // end initChart
    </script>
    </body>
    </html>
    """, height=430, scrolling=False)


            # ─────────────────────────────────────────────
            # 하단: AI 브리핑 + 뉴스
            # ─────────────────────────────────────────────
            st.markdown('<div class="bottom-wrap"><div class="bottom-left">', unsafe_allow_html=True)

            st.markdown('<div class="section-header">💡 AI Analyst Briefing — 시황·섹터·종목 종합</div>', unsafe_allow_html=True)

            _cached_ai  = st.session_state.get("cached_ai_html")
            _cached_src = st.session_state.get("cached_ai_src")

            if _cached_ai is not None:
                st.markdown(f'<div class="ai-box">{_cached_ai}{_cached_src}</div>', unsafe_allow_html=True)

            elif "GEMINI_API_KEY" in st.secrets:
                # ── Gemini 호출 ──
                _ai_error   = ""
                _res_json   = None
                _used_model = "gemini-2.5-flash-lite"
                _used_search = False

                try:
                    with st.spinner("📡 AI가 최신 뉴스를 검색하며 분석 중..."):
                        gemini_key = str(st.secrets["GEMINI_API_KEY"]).strip()
                        today_str  = now_kst().strftime("%Y년 %m월 %d일")

                        daily_pred = "\n".join(
                            f"  {r['ds'].strftime('%m/%d')} : 예측 {int(r['yhat']):,}원 "
                            f"(범위 {int(r['yhat_lower']):,}~{int(r['yhat_upper']):,})"
                            for _, r in fc_future.iterrows()
                        )
                        sent_label = (f"긍정적 ({sentiment_score:+.2f})" if sentiment_score > 0.2
                                      else f"부정적 ({sentiment_score:+.2f})" if sentiment_score < -0.2
                                      else f"중립 ({sentiment_score:+.2f})")
                        naver_section = ""
                        if news_raw:
                            titles = "\n".join(f"  - {n['title']}" for n in news_raw[:5])
                            naver_section = f"\n[네이버 수집 뉴스]\n{titles}"

                        # 프롬프트 정의 바로 윗줄에 이 코드를 추가하여 검색 시점을 동적으로 만듭니다.
                            current_ym = now_kst().strftime("%Y년 %m월") # 예: 시간이 지나면 '2026년 4월', '2026년 5월'로 자동 변경됨

                            prompt = f"""당신은 피도 눈물도 없는 여의도 탑티어 프랍 트레이더입니다. 
    당신의 분석 하나에 누군가의 전 재산과 목숨이 걸려있습니다. 
    오늘 날짜는 정확히 [{today_str}]입니다.

    [냉혹한 팩트 데이터]
    - 종목: {display_name} ({ticker}) | 현재가: {int(cp):,}원 ({pct:+.2f}%)
    - 52주 고/저: {int(high52):,} / {int(low52):,} 
    - 거래량 흐름: 20일 평균({avg_vol:,}) 대비 오늘 거래량
    - 기술적 상태: RSI {cur_rsi}({rsi_signal}) | MACD {macd_signal} | BB %B {cur_bb}%
    - Prophet 단기 추세({pred_days}일): {pred_end:,}원 (상단 {pred_upper:,} / 하단 {pred_lower:,})
    {naver_section}

    [Google 검색 명령 - ⚠️ 탑다운(Top-Down) & 바텀업(Bottom-Up) 입체 스캔]
    1. 매크로 & 정책: "{today_str} 미증시 시황", "최근 한국 {display_name} 관련 산업 정부 정책 규제 원자재 가격 동향" (예: 유가 상승과 정부 가격 상한제 충돌 등 정책 리스크 확인)
    2. 개별 종목: "{display_name} 최신 뉴스", "{today_str} {display_name} 공시 기술이전 수주 실적 계약"
    * 주의
    1. 구글 검색 시 '악재'나 '급락'이라는 단어에 얽매이지 마시오. 
    2. 호재성 제목("기술이전", "수주")의 기사라도 그것이 시장의 기대치에 못 미쳐서(재료 소멸) 주가 폭락의 원인이 될 수 있음을 명심하고, 오늘 발생한 모든 이슈를 철저히 스캔하시오.

    [작성 원칙 - ⚠️ 목숨 걸고 지킬 것]
    1. 표면적인 개별 뉴스만 보지 말고, 해당 기업의 이익을 훼손하거나 증대시키는 '정부 정책/규제' 및 '원자재/환율 흐름'과의 상관관계를 반드시 짚어낼 것.
    2. 개미를 꼬드기는 희망 고문, 모호한 표현(~우려, ~전망) 금지. 철저히 팩트 기반으로 작성.
    3. 개조식(~함, ~임, ~판단됨)으로 극도로 짧고 날카롭게 작성할 것.

    [출력 템플릿 - 토씨 하나 틀리지 말고 이 양식대로 출력]
    (주의: 응답의 가장 첫 글자는 반드시 '🌍' 기호로 시작할 것. 앞에 빈 줄이나 공백, 인사말을 넣으면 즉시 폐기함)
    🌍 0. 거시 환경 & 산업 규제 (Macro & Policy)
    - 전체 투심: (미 증시 및 국내 KOSPI/KOSDAQ 수급 1줄 요약)
    - 산업/정책 변수: (해당 기업에 직결되는 핵심 원자재 가격 흐름이나 정부 규제/정책 리스크가 주가를 어떻게 누르고/끌어올리고 있는지 날카롭게 1줄 요약)

    🔥 1. 팩트 폭격: 주가 변동의 진짜 이유
    - 핵심 트리거: (반드시 {today_str} 기준 최근 2주 이내의 날짜와 팩트만 기재. 없으면 '최근 이슈 없음' 기재)
    - 재료의 성격: (단발성 노이즈 vs 구조적 펀더멘털/정책 변화)

    🩸 2. 차트와 알고리즘의 이면 (세력의 의도)
    - 지표 진단: (RSI/MACD/거래량을 종합하여 현재 구간이 매집인지 설거지인지 확정적 어조로 판단)
    - Prophet 괴리: (예측가 {pred_end:,}원과 기술적 지표 간의 모순/일치점 1줄)

    🎯 3. 피도 눈물도 없는 실전 매매 타점
    - 투자의견: [풀매수 / 분할매수 / 관망 / 비중축소 / 전량매도] 중 택 1
    - 진입 및 목표가: (저항/지지 기반 구체적 '원' 단위 타점)
    - 목숨줄(손절선): (절대 깨지면 안 되는 마지노선 가격)
    - 트레이더 코멘트: (시황, 정책 변수, 개별 재료를 엮은 가장 현실적인 액션 플랜 1줄)
    """                       


                        base_url = "https://generativelanguage.googleapis.com/v1beta/models"
                        url = f"{base_url}/{_used_model}:generateContent?key={gemini_key}"

                        # 1차: Google Search Grounding 포함
                        res = requests.post(url, json={
                            "contents": [{"parts": [{"text": prompt}]}],
                            "tools": [{"googleSearch": {}}]
                        }, timeout=60)

                        if res.status_code == 200:
                            rj = res.json()
                            if "candidates" in rj:
                                _res_json    = rj
                                _used_search = True
                        elif res.status_code == 429:
                            _ai_error = "1분 호출 한도 초과. 잠시 후 다시 시도해주세요."
                        else:
                            # 2차: 검색 없이 재시도
                            res2 = requests.post(url, json={
                                "contents": [{"parts": [{"text": prompt}]}]
                            }, timeout=60)
                            if res2.status_code == 200:
                                rj2 = res2.json()
                                if "candidates" in rj2:
                                    _res_json = rj2
                            elif res2.status_code == 429:
                                _ai_error = "1분 호출 한도 초과. 잠시 후 다시 시도해주세요."
                            else:
                                _ai_error = f"HTTP {res2.status_code}: {res2.text[:100]}"

                except requests.exceptions.Timeout:
                    _ai_error = "API 응답 시간 초과"
                except Exception as e:
                    _ai_error = str(e)

                # ── spinner 밖에서 캐시 저장 & 렌더 ──
                if _res_json and "candidates" in _res_json:
                    import re as _re
                    parts   = _res_json["candidates"][0]["content"].get("parts", [])
                    ai_text = "".join(p.get("text", "") for p in parts if "text" in p).strip()
                    # Gemini이 삽입하는 불필요한 텍스트 제거
                    ai_html = _re.sub(r'```json\s*\[.*?\]\s*```', '', ai_text, flags=_re.DOTALL)
                    ai_html = _re.sub(r'```python\s*.*?```', '', ai_html, flags=_re.DOTALL)
                    ai_html = _re.sub(r'```.*?```', '', ai_html, flags=_re.DOTALL)
                    ai_html = _re.sub(r'\[\s*\{\s*"query".*?\}\s*\]', '', ai_html, flags=_re.DOTALL)
                    ai_html = _re.sub(r'print\s*\(.*?\)\s*', '', ai_html, flags=_re.DOTALL)
                    ai_html = _re.sub(r'google_search\.\w+\(.*?\)', '', ai_html, flags=_re.DOTALL)
                    ai_html = _re.sub(r'(?i)disclaimer.*?(?=\n\n|🌍)', '', ai_html, flags=_re.DOTALL)
                    ai_html = _re.sub(r'(?i)I am an AI.*?(?=\n\n|🌍)', '', ai_html, flags=_re.DOTALL)
                    ai_html = _re.sub(r'(?i)(?:Note|Warning|Caution)\s*:?\s*(?:I am|This is|The following).*?(?=\n\n|🌍)', '', ai_html, flags=_re.DOTALL)
                    # 🌍 이전의 모든 텍스트 제거 (프롬프트에서 🌍로 시작하도록 지시함)
                    _globe_idx = ai_html.find('🌍')
                    if _globe_idx > 0:
                        ai_html = ai_html[_globe_idx:]
                    # 🌍가 없으면 Gemini가 제대로 응답하지 않은 것 → 폴백
                    if '🌍' not in ai_html:
                        ai_html = ""
                    ai_html = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', ai_html)
                    ai_html = _re.sub(r'<(tool_code|function_call|tool_result|code_execution)[^>]*>.*?</\1>', '', ai_html, flags=_re.DOTALL)
                    ai_html = ai_html.replace("\n", "<br>")                
                    if ai_html.strip():
                        src_label = "🔍 Google 검색 포함" if _used_search else ("📰 네이버 뉴스 기반" if news_txt else "📊 Prophet 데이터만")
                        source_tag = f'<div style="font-size:0.68rem;color:#4a5568;margin-top:8px">🤖 {_used_model} · {src_label}</div>'
                        st.session_state["cached_ai_html"] = ai_html
                        st.session_state["cached_ai_src"]  = source_tag
                        st.markdown(f'<div class="ai-box">{ai_html}{source_tag}</div>', unsafe_allow_html=True)
                    else:
                        # Gemini가 코드만 출력하고 분석을 안 한 경우 → 폴백
                        st.session_state["cached_ai_html"] = None
                        st.session_state["cached_ai_src"]  = None
                        _show_fallback_briefing(display_name, cp, pred_days, pred_end, pred_pct, pred_lower, pred_upper)
                else:
                    # 에러 시 캐시에 저장 → 재호출 방지
                    err_msg = _ai_error or "알 수 없는 오류"
                    st.session_state["cached_ai_html"] = f'<span style="color:#fc5c5c">⚠️ {err_msg}</span>'
                    st.session_state["cached_ai_src"]  = ""
                    st.warning(f"⚠️ Gemini API 오류: {err_msg}")
                    _show_fallback_briefing(display_name, cp, pred_days, pred_end, pred_pct, pred_lower, pred_upper)

            else:
                st.info("💡 Streamlit Secrets에 `GEMINI_API_KEY`를 등록하면 AI 브리핑이 활성화됩니다.")
                _show_fallback_briefing(display_name, cp, pred_days, pred_end, pred_pct, pred_lower, pred_upper)

            st.markdown('</div><div class="bottom-right">', unsafe_allow_html=True)

            st.markdown('<div class="section-header">📰 네이버 뉴스</div>', unsafe_allow_html=True)

            if news_status == "ok" and news_raw:
                for n in news_raw:
                    title = n.get("title","").strip()
                    link  = n.get("link","#")
                    pub   = n.get("pub","")
                    if not title: continue
                    st.markdown(
                        f'<div class="news-card">'
                        f'<span style="color:#4a5568;font-size:0.68rem">{pub}&nbsp;&nbsp;</span>'
                        f'<a href="{link}" target="_blank" style="color:#a0aec0;text-decoration:none;line-height:1.6">{title}</a>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            elif news_status == "no_key":
                st.markdown("""
                <div class="news-card">
                    <div style="color:#f9a825;font-size:0.8rem;font-weight:600;margin-bottom:6px">🔑 Naver API 키 미등록</div>
                    <div style="color:#718096;font-size:0.78rem;line-height:1.7">
                        ① <a href="https://developers.naver.com/apps/#/register" target="_blank" style="color:#4d9fff">developers.naver.com</a> 접속<br>
                        ② 애플리케이션 등록 → <b>검색 API</b> 선택<br>
                        ③ Client ID / Secret 복사<br>
                        ④ Streamlit Secrets에 추가:
                    </div>
                    <div style="background:#0d1117;border-radius:6px;padding:8px 10px;margin-top:8px;font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#79c0ff;line-height:1.8">
                        NAVER_CLIENT_ID = "여기에_ID"<br>
                        NAVER_CLIENT_SECRET = "여기에_SECRET"
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif news_status.startswith("empty:"):
                st.markdown(f'<div class="news-card" style="color:#a0aec0;font-size:0.8rem">🔍 검색 결과 없음</div>', unsafe_allow_html=True)
            elif news_status.startswith("api_error"):
                st.markdown(f'<div class="news-card" style="color:#fc5c5c;font-size:0.8rem">⚠️ {news_status.replace("api_error:","")}</div>', unsafe_allow_html=True)

            st.markdown('</div></div>', unsafe_allow_html=True)
            st.markdown('<div style="height:2rem"></div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
            st.exception(e)
