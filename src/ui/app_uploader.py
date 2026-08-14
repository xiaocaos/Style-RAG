"""知识库上传界面：支持多格式文件上传。"""

from __future__ import annotations

import streamlit as st

from src.knowledge.base import KnowledgeBase
from src.logging_config import logger


def run_app() -> None:
    st.title("知识库管理")
    st.caption("上传文档更新知识库，支持 .txt 格式")
    st.divider()

    if "kb" not in st.session_state:
        st.session_state["kb"] = KnowledgeBase()

    kb: KnowledgeBase = st.session_state["kb"]

    # 知识库状态
    stats = kb.get_stats()
    col1, col2 = st.columns(2)
    col1.metric("文档片段总数", stats["total_chunks"])
    col2.metric("集合名称", stats["collection"])

    st.divider()

    # 文件上传
    uploaded_files = st.file_uploader(
        "上传文档",
        type=["txt"],
        accept_multiple_files=True,
        help="支持 .txt 格式，多个文件可同时上传",
    )

    if uploaded_files:
        for file in uploaded_files:
            st.subheader(f"📄 {file.name}")
            file_size = file.size / 1024
            st.write(f"大小: {file_size:.2f} KB")

            try:
                text = file.getvalue().decode("utf-8")
                with st.spinner("正在导入知识库..."):
                    count = kb.add_texts([text], source=file.name)
                    if count > 0:
                        st.success(f"导入成功，新增 {count} 个文档片段")
                    else:
                        st.info("内容已存在，跳过导入")
            except UnicodeDecodeError:
                st.error("文件编码错误，请确保文件为 UTF-8 格式")
            except Exception as e:
                st.error(f"导入失败: {e}")
                logger.error("文件导入失败: %s - %s", file.name, e)

    st.divider()

    # 手动添加文本
    st.subheader("手动添加知识")
    with st.form("manual_add"):
        source_name = st.text_input("来源名称", placeholder="例如：春季新品说明")
        content = st.text_area("内容", placeholder="在此输入文本内容...", height=200)
        submitted = st.form_submit_button("添加到知识库")

        if submitted and content:
            with st.spinner("正在添加..."):
                count = kb.add_texts([content], source=source_name or "manual_input")
                if count > 0:
                    st.success(f"添加成功，新增 {count} 个文档片段")
                else:
                    st.info("内容已存在，跳过添加")
        elif submitted:
            st.warning("请输入内容")


if __name__ == "__main__":
    run_app()
