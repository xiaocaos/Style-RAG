"""Streamlit 主入口。"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

st.set_page_config(
    page_title="服装智能客服",
    page_icon="👔",
    layout="wide",
)

# 侧边栏导航
page = st.sidebar.radio(
    "导航",
    ["在线问答", "知识库管理"],
    index=0,
)

if page == "在线问答":
    from src.ui.app_qa import run_app
    run_app()
elif page == "知识库管理":
    from src.ui.app_uploader import run_app
    run_app()
