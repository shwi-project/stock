import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import GradientBoostingRegressor
import plotly.graph_objects as go
from streamlit_lightweight_charts import renderLightweightCharts
from streamlit_autorefresh import st_autorefresh
import requests
import json
import base64
from datetime import datetime, timedelta

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
    
    .stButton > button {
        height: 32px !important;
        min-height: 32px !important;
        max-height: 32px !important;
        width: 36px !important;        /* 고정 너비 */
        min-width: 36px !important;
        padding: 0 !important;
        line-height: 32px !important;
        font-size: 0.9rem !important;
        margin-bottom: 5px !important;
        margin-left: 0px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: #2d3a55 !important;
        border: 1px solid #3d4f6e !important;
        color: #e2e8f0 !important;
    }
    .stButton > button:hover {
        background: #3d4f6e !important;
        border-color: #4d9fff !important;
        color: #e2e8f0 !important;
    }
    .stButton > button:active {
        background: #1e2d45 !important;
        border-color: #4d9fff !important;
    }
    [data-testid="stButton"] > div {
        display: flex !important;
        align-items: flex-end !important;
         margin-left: -12px !important;
        height: 100% !important;
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
        min-height: 32px !important; max-height: 32px !important;
        padding-top: 0 !important; padding-bottom: 0 !important; line-height: 32px !important;
    }
    [data-baseweb="select"] * { color: #e2e8f0 !important; }
    [data-baseweb="select"] input { height: 32px !important; }    
    .stSelectbox { margin-bottom: 0.3rem !important; }
    [data-baseweb="popover"], [data-baseweb="menu"] { background: #131929 !important; }
    [data-baseweb="option"]       { color: #e2e8f0 !important; background: #131929 !important; }
    [data-baseweb="option"]:hover { background: #1e2d45 !important; }

    [data-testid="stSpinner"] p, [data-testid="stSpinner"] span, .stSpinner p { font-size: 0.72rem !important; color: #4a5568 !important; }

    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    .stDeployButton { display: none; }
    [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; flex-direction: row !important; align-items: flex-start !important; gap: 0.3rem !important; }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width: 0 !important; overflow: hidden; }
    .bottom-wrap { display: grid; grid-template-columns: 3fr 2fr; gap: 0.8rem; margin-top: 0.5rem; }
    .bottom-left  { min-width: 0; }
    .bottom-right { min-width: 0; }
    @media (max-width: 768px) { .bottom-wrap { grid-template-columns: 1fr; } }
    @media (max-width: 640px) {
        .block-container { padding-left: 0.3rem !important; padding-right: 0.3rem !important; }
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
                direction = nxt.get("compareToPreviousPrice", {})
                is_rising = direction.get("name") in ("RISING", "UPPER_LIMIT")
                return {
                    "price": int(price_str),
                    "change": int(change_str) if is_rising else -int(change_str),
                    "change_pct": float(nxt.get("fluctuationsRatio", 0)) if is_rising else -float(nxt.get("fluctuationsRatio", 0)),
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
# 상단 컨트롤: 종목 검색만 (예측기간 2일 고정)
# ─────────────────────────────────────────────
st.markdown(f"""<div style='display:flex;align-items:center;margin-bottom:2px'>
            {logo_html}
            Castle Stock AI &nbsp;<span style='font-size:0.75rem;color:#4a5568;font-weight:400'>믿지 못할 주식 예측</span>"""
            ,unsafe_allow_html=True
           )
st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)

_col1, _col2, _col3 = st.columns([2.0, 0.4, 0.6], vertical_alignment="bottom")
with _col1:
    st.markdown('<p style="font-size:0.72rem;color:#a0aec0;margin-bottom:2px">종목 검색</p>', unsafe_allow_html=True)
    selected_label = st.selectbox(
        "종목 검색",
        [None] + labels,
        index=0,
        label_visibility="collapsed",
        format_func=lambda x: "종목명 또는 코드를 입력하세요" if x is None else x,
    )

with _col2:
    _query_btn = st.button("🔍", use_container_width=True)

with _col3:
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
    st.markdown(
        f'<div style="font-size:0.68rem;color:#4a5568;text-align:center;'
        f'padding-top:22px">{_mkt_status}<br>{_now.strftime("%H:%M")} KST</div>',
        unsafe_allow_html=True
    )

# 예측기간 1영업일 고정
pred_days = 1

# 장중 자동 갱신 (60초 간격, 장중에만)
_is_market_open = (_wd < 5 and 800 <= _hm <= 2000)
if _is_market_open:
    st_autorefresh(interval=60_000, limit=None, key="market_refresh")

if selected_label is None:
    st.markdown('<div style="height:2rem"></div>', unsafe_allow_html=True)
    st.stop()

# 조회 버튼 클릭 시 캐시 초기화 (재조회 가능)
if _query_btn:
    for _k in ["cached_data_key", "cached_df", "cached_news_raw",
               "cached_news_status", "cached_ai_html", "cached_ai_src",
               "cached_fc_future", "cached_sentiment", "cached_backtest"]:
        st.session_state[_k] = None

if not _query_btn:
    if st.session_state.get("cached_data_key") is None:
        st.markdown('<div style="height:2rem"></div>', unsafe_allow_html=True)
        st.stop()
    else:
        # 캐시 있으면 이전 결과 유지하면서 대기
        ticker_cached = st.session_state["cached_data_key"].split("__")[0]
        sel_row_tmp = all_stocks[all_stocks["label"] == selected_label].iloc[0]
        if ticker_cached != sel_row_tmp["ticker"]:
            # 종목 바뀌었으면 캐시 종목으로 되돌려서 표시
            selected_label = all_stocks[all_stocks["ticker"] == ticker_cached]["label"].values[0]

sel_row      = all_stocks[all_stocks["label"] == selected_label].iloc[0]
ticker       = sel_row["ticker"]
stock_code   = sel_row["종목코드"]
market       = sel_row["시장"]
display_name = sel_row["종목명"]

st.markdown("---")

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

        # 장중 당일 데이터 보완 (pykrx 스냅샷)
        try:
            from pykrx import stock as _krx
            _today_str = now_kst().strftime("%Y%m%d")
            _today_ohlcv = _krx.get_market_ohlcv_by_date(_today_str, _today_str, stock_code)
            if _today_ohlcv is not None and len(_today_ohlcv) > 0:
                _tr = _today_ohlcv.iloc[-1]
                _today_date = pd.Timestamp(_today_ohlcv.index[-1])
                if _today_date.tz is not None:
                    _today_date = _today_date.tz_localize(None)
                _last_date = pd.to_datetime(df["Date"]).dt.tz_localize(None).max()
                if _today_date > _last_date and float(_tr["종가"]) > 0:
                    _new = pd.DataFrame([{
                        "Date": _today_date,
                        "Open": float(_tr["시가"]), "High": float(_tr["고가"]),
                        "Low": float(_tr["저가"]), "Close": float(_tr["종가"]),
                        "Volume": float(_tr["거래량"]),
                    }])
                    df = pd.concat([df, _new], ignore_index=True)
        except Exception:
            pass

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
        # Prophet 예측
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
        }

        def get_kr_trading_days(start_date, count):
            days = []
            cur = start_date + timedelta(days=1)
            while len(days) < count:
                if cur.weekday() < 5 and cur not in KR_HOLIDAYS:
                    days.append(cur)
                cur += timedelta(days=1)
            return days

        _need_prophet = (st.session_state["cached_fc_future"] is None)

        if _need_prophet:
          with st.spinner("🤖 기술지표 계산 및 AI 예측 모델 실행 중..."):
            close = df["Close"]
            delta = close.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / loss.replace(0, 1e-9)
            df["RSI"] = (100 - 100 / (1 + rs)).fillna(50)

            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line   = ema12 - ema26
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

            # 거래량 비율 (당일 / 20일 평균)
            df["vol_ratio"] = (df["Volume"] / df["Volume"].rolling(20).mean().replace(0, 1)).fillna(1.0)
            # MA20 이격도
            df["ma20_dist"] = ((close - close.rolling(20).mean()) / (close.rolling(20).mean() + 1e-9)).fillna(0)

            POS_WORDS = {"상승","급등","신고가","호실적","흑자","수주","계약","승인","성장","돌파","매수","강세","반등","기대"}
            NEG_WORDS = {"하락","급락","적자","부진","리스크","매도","약세","손실","취소","조사","소송","경고","우려","악화"}

            sentiment_score = 0.0
            if news_raw:
                scores = []
                for i, n in enumerate(news_raw):
                    title = n.get("title", "")
                    pos = sum(1 for w in POS_WORDS if w in title)
                    neg = sum(1 for w in NEG_WORDS if w in title)
                    raw_s = (pos - neg) / max(pos + neg, 1) if (pos + neg) > 0 else 0.0
                    recency_w = 1.0 / (1 + i * 0.2)
                    scores.append(raw_s * recency_w)
                sentiment_score = sum(scores) / sum(1.0 / (1 + i * 0.2) for i in range(len(scores))) if scores else 0.0

            # 감성 점수를 컬럼으로 추가 (Prophet regressor)
            df["sentiment"] = sentiment_score

            REGRESSOR_COLS = ["RSI", "MACD_hist", "BB_pct", "OBV_norm", "sentiment"]
            df_f = df[["Date", "Close"] + REGRESSOR_COLS].rename(
                columns={"Date": "ds", "Close": "y"}
            ).dropna()

            # ── 적응형 Prophet 하이퍼파라미터 ──
            atr_14 = (df["High"] - df["Low"]).rolling(14).mean()
            atr_60 = (df["High"] - df["Low"]).rolling(60).mean()
            if len(atr_14.dropna()) > 0 and len(atr_60.dropna()) > 0:
                vol_ratio_now = float(atr_14.iloc[-1]) / max(float(atr_60.iloc[-1]), 1e-9)
                if vol_ratio_now > 2.0:
                    cps = 0.1
                elif vol_ratio_now < 0.5:
                    cps = 0.02
                else:
                    cps = 0.05
            else:
                cps = 0.05

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

            # ── 평균회귀 기반 지표 외삽 ──
            last_row = df_f.iloc[-1]
            MEAN_TARGETS = {"RSI": 50.0, "BB_pct": 0.5, "MACD_hist": 0.0, "OBV_norm": 0.5, "sentiment": 0.0}
            REVERT_SPEED = {"RSI": 0.15, "BB_pct": 0.2, "MACD_hist": 0.25, "OBV_norm": 0.1, "sentiment": 0.3}

            for col in REGRESSOR_COLS:
                vals, curr_val = [], float(last_row[col])
                mu = MEAN_TARGETS[col]
                theta = REVERT_SPEED[col]
                for step in range(1, pred_days + 1):
                    curr_val = curr_val + theta * (mu - curr_val)
                    vals.append(curr_val)
                future_dates[col] = vals

            future_dates["RSI"]    = future_dates["RSI"].clip(0, 100)
            future_dates["BB_pct"] = future_dates["BB_pct"].clip(-0.2, 1.2)

            future_all = pd.concat(
                [df_f[["ds"] + REGRESSOR_COLS], future_dates],
                ignore_index=True
            )
            fc = model.predict(future_all)

            last_date_tmp = df["Date"].max()
            fc_future_tmp = fc[fc["ds"] > last_date_tmp].copy()
            prophet_pred = float(fc_future_tmp.iloc[-1]["yhat"])

            # ── GradientBoosting 앙상블 ──
            GBR_FEATURES = ["RSI", "MACD_hist", "BB_pct", "OBV_norm", "vol_ratio", "ma20_dist"]
            df_gb = df.dropna(subset=GBR_FEATURES + ["Close"]).copy()
            df_gb["target"] = df_gb["Close"].shift(-1)
            df_gb = df_gb.dropna(subset=["target"])

            gbr_pred = prophet_pred
            if len(df_gb) > 30:
                try:
                    X_gb = df_gb[GBR_FEATURES].values
                    y_gb = df_gb["target"].values
                    gbr = GradientBoostingRegressor(
                        n_estimators=100, max_depth=4, learning_rate=0.1,
                        subsample=0.8, random_state=42
                    )
                    gbr.fit(X_gb, y_gb)
                    last_features = df[GBR_FEATURES].iloc[-1:].values
                    gbr_pred = float(gbr.predict(last_features)[0])
                except Exception:
                    gbr_pred = prophet_pred

            # 가중 평균 (Prophet 60%, GBR 40%)
            ensemble_pred = prophet_pred * 0.6 + gbr_pred * 0.4

            # ── 예측값 상한/하한 클램프 (일일 상하한가 ±30%) ──
            _max_daily_pct = 0.15  # 현실적 1일 변동 상한 ±15%
            _hard_limit_pct = 0.30  # 한국 주식 일일 상하한가
            _current_close = float(cp)

            # 예측값이 ±15% 초과 시 현재가 쪽으로 당김
            _pred_pct_raw = (ensemble_pred - _current_close) / _current_close
            if abs(_pred_pct_raw) > _max_daily_pct:
                ensemble_pred = _current_close * (1 + np.sign(_pred_pct_raw) * _max_daily_pct)

            # 절대 상하한가 ±30% 클램프
            ensemble_pred = np.clip(ensemble_pred,
                                    _current_close * (1 - _hard_limit_pct),
                                    _current_close * (1 + _hard_limit_pct))

            fc_future_tmp["yhat"] = ensemble_pred
            spread = float(fc_future_tmp.iloc[-1]["yhat_upper"] - fc_future_tmp.iloc[-1]["yhat_lower"])
            # 신뢰구간도 ±30% 내로 클램프
            _max_spread = _current_close * _hard_limit_pct * 2
            spread = min(spread, _max_spread)
            fc_future_tmp["yhat_upper"] = min(ensemble_pred + spread * 0.5,
                                              _current_close * (1 + _hard_limit_pct))
            fc_future_tmp["yhat_lower"] = max(ensemble_pred - spread * 0.5,
                                              _current_close * (1 - _hard_limit_pct))

            # ── 백테스팅 (최근 10영업일 MAPE) ──
            backtest_mape = None
            if len(df_gb) > 40:
                try:
                    X_bt = df_gb[GBR_FEATURES].values
                    y_bt = df_gb["target"].values
                    bt_actual = y_bt[-10:]
                    bt_pred_gbr = []
                    for i in range(10):
                        idx = len(X_bt) - 10 + i
                        gbr_bt = GradientBoostingRegressor(
                            n_estimators=100, max_depth=4, learning_rate=0.1,
                            subsample=0.8, random_state=42
                        )
                        gbr_bt.fit(X_bt[:idx], y_bt[:idx])
                        bt_pred_gbr.append(gbr_bt.predict(X_bt[idx:idx+1])[0])
                    bt_pred_arr = np.array(bt_pred_gbr)
                    bt_actual_arr = np.array(bt_actual)
                    backtest_mape = float(np.mean(np.abs((bt_actual_arr - bt_pred_arr) / bt_actual_arr)) * 100)
                except Exception:
                    backtest_mape = None

            st.session_state["cached_fc_future"]  = fc_future_tmp
            st.session_state["cached_sentiment"]   = sentiment_score
            st.session_state["cached_backtest"]    = backtest_mape

        else:
            fc = None

        last_date = df["Date"].max()
        fc_future = st.session_state["cached_fc_future"] if st.session_state["cached_fc_future"] is not None else fc[fc["ds"] > last_date].copy()
        sentiment_score = st.session_state["cached_sentiment"] or 0.0
        backtest_mape   = st.session_state.get("cached_backtest")

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
                <div class="metric-label">모델 정확도</div>
                <div class="metric-value" style="font-size:0.95rem;color:{'#4dc98f' if backtest_mape is not None and backtest_mape < 3 else '#f9a825' if backtest_mape is not None and backtest_mape < 5 else '#fc5c5c' if backtest_mape is not None else '#4a5568'}">{f'{backtest_mape:.1f}%' if backtest_mape is not None else '-'}</div>
                <div class="metric-sub" style="color:#4a5568">{'MAPE 10일 백테스트' if backtest_mape is not None else '데이터 부족'}</div>
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
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
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
                ai_html = _re.sub(r'\[\s*\{\s*"query".*?\}\s*\]', '', ai_html, flags=_re.DOTALL)
                ai_html = _re.sub(r'(?i)disclaimer.*?(?=\n\n|🌍)', '', ai_html, flags=_re.DOTALL)
                ai_html = _re.sub(r'(?i)I am an AI.*?(?=\n\n|🌍)', '', ai_html, flags=_re.DOTALL)
                ai_html = _re.sub(r'(?i)(?:Note|Warning|Caution)\s*:?\s*(?:I am|This is|The following).*?(?=\n\n|🌍)', '', ai_html, flags=_re.DOTALL)
                # 🌍 이전의 모든 텍스트 제거 (프롬프트에서 🌍로 시작하도록 지시함)
                _globe_idx = ai_html.find('🌍')
                if _globe_idx > 0:
                    ai_html = ai_html[_globe_idx:]
                ai_html = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', ai_html)
                ai_html = _re.sub(r'<(tool_code|function_call|tool_result|code_execution)[^>]*>.*?</\1>', '', ai_html, flags=_re.DOTALL)
                ai_html = ai_html.replace("\n", "<br>")                
                src_label = "🔍 Google 검색 포함" if _used_search else ("📰 네이버 뉴스 기반" if news_txt else "📊 Prophet 데이터만")
                source_tag = f'<div style="font-size:0.68rem;color:#4a5568;margin-top:8px">🤖 {_used_model} · {src_label}</div>'
                st.session_state["cached_ai_html"] = ai_html
                st.session_state["cached_ai_src"]  = source_tag
                st.markdown(f'<div class="ai-box">{ai_html}{source_tag}</div>', unsafe_allow_html=True)
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
