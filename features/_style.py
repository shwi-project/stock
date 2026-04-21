"""다크 테마에서 가독성 보강용 공통 스타일 주입.

Streamlit 기본 `st.caption`은 저대비 회색이라 다크 모드에서 잘 안 보인다.
feature 탭들에서 최초 1회 호출하면 해당 런 동안 스타일이 적용된다.
"""

from __future__ import annotations

import streamlit as st

_CSS = """
<style>
/* caption · help 텍스트 대비 상향 */
.stCaption, [data-testid="stCaptionContainer"], small {
    color: #c9d1d9 !important;
    opacity: 0.95 !important;
}

/* metric 라벨과 값 강조 */
[data-testid="stMetricLabel"] p {
    font-size: 0.88rem !important;
    color: #c9d1d9 !important;
    opacity: 0.95 !important;
}
[data-testid="stMetricValue"] {
    font-weight: 600 !important;
}

/* dataframe 내부 글자 크기 살짝 키움 */
[data-testid="stDataFrame"] {
    font-size: 0.95rem !important;
}

/* info/warning/error 박스 글자 진하게 */
[data-testid="stAlertContentInfo"] p,
[data-testid="stAlertContentWarning"] p,
[data-testid="stAlertContentError"] p {
    font-weight: 500 !important;
}
</style>
"""


def inject() -> None:
    """한 번만 주입. 세션 상태로 중복 주입 방지."""
    if st.session_state.get("_ext_style_injected"):
        return
    st.markdown(_CSS, unsafe_allow_html=True)
    st.session_state["_ext_style_injected"] = True
