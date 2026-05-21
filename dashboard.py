from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="GridSense", page_icon="🔌", layout="wide")

st.markdown(
    "<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;} .css-18e3th9 {padding: 0rem 0rem 0rem 0rem;}</style>",
    unsafe_allow_html=True,
)

page_html = Path(__file__).with_name("gridsense_redesigned.html").read_text(encoding="utf-8")
components.html(page_html, height=5200, scrolling=True)
