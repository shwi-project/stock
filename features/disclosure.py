"""F. DART/거래소 공시 리스트."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from features._async import run_sync
from sources import naver


@st.cache_data(ttl=300, show_spinner=False)
def _disclosures(code: str, page: int) -> list[dict]:
    return run_sync(naver.get_disclosure_list(code, page))


def render(code: str) -> None:
    if not code:
        st.info("종목코드를 입력하세요.")
        return
    try:
        rows = _disclosures(code, 1)
    except Exception as e:
        st.error(f"공시 조회 실패: {e}")
        return

    if not rows:
        st.info("최근 공시 없음.")
        return

    df = pd.DataFrame(rows)

    def _link(row):
        link = row["link"] or ""
        return f"[{row['title']}]({link})" if link else row["title"]

    df["제목"] = df.apply(_link, axis=1)
    df = df[["date", "source", "제목"]]
    df.columns = ["날짜", "출처", "제목"]
    st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
