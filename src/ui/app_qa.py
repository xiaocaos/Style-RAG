"""聊天界面：支持流式输出、来源展示、反馈机制。"""

from __future__ import annotations

import streamlit as st

from src.config import settings
from src.knowledge.base import KnowledgeBase
from src.rag.advanced_chain import AdvancedRAGChain
from src.rag.chain import RAGChain
from src.logging_config import logger


def _init_session() -> None:
    """初始化 Streamlit session state。"""
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "你好！我是 AI 客服小爽，有什么可以帮助你的吗？"}
        ]
    if "kb" not in st.session_state:
        st.session_state["kb"] = KnowledgeBase()
    if "mode" not in st.session_state:
        st.session_state["mode"] = "高级检索"
    if "feedback" not in st.session_state:
        st.session_state["feedback"] = {}


def _render_sidebar() -> None:
    """侧边栏：知识库状态 + 模式切换。"""
    with st.sidebar:
        st.header("设置")

        # 检索模式切换
        mode = st.radio(
            "检索模式",
            ["基础检索", "高级检索"],
            index=1,
            help="基础检索：直接向量检索；高级检索：Rewriting + Hybrid + Reranking",
        )
        st.session_state["mode"] = mode

        st.divider()

        # 知识库统计
        kb: KnowledgeBase = st.session_state["kb"]
        stats = kb.get_stats()
        st.metric("知识库文档片段", stats["total_chunks"])

        st.divider()

        # 清空对话
        if st.button("清空对话"):
            st.session_state["messages"] = [
                {"role": "assistant", "content": "对话已清空，有什么新问题吗？"}
            ]
            st.rerun()


def _display_sources(sources: list[dict]) -> None:
    """展示检索来源。"""
    if not sources:
        return
    with st.expander("参考来源", expanded=False):
        for i, src in enumerate(sources, 1):
            st.markdown(f"**来源 {i}** ({src['source']})")
            st.text(src["content"][:200] + ("..." if len(src["content"]) > 200 else ""))
            st.divider()


def _render_feedback(msg_idx: int) -> None:
    """渲染反馈按钮。"""
    cols = st.columns([1, 1, 8])
    with cols[0]:
        if st.button("👍", key=f"like_{msg_idx}"):
            st.session_state["feedback"][msg_idx] = "positive"
            st.toast("感谢反馈！")
    with cols[1]:
        if st.button("👎", key=f"dislike_{msg_idx}"):
            st.session_state["feedback"][msg_idx] = "negative"
            st.toast("感谢反馈，我们会持续优化！")


def run_app() -> None:
    """主应用入口。"""
    st.title("AI 小爽客服")
    st.caption("基于高级 RAG 流水线的服装智能客服")
    st.divider()

    _init_session()
    _render_sidebar()

    # 显示历史消息
    for i, msg in enumerate(st.session_state["messages"]):
        st.chat_message(msg["role"]).write(msg["content"])
        if msg["role"] == "assistant" and i > 0:
            _render_feedback(i)

    # 用户输入
    prompt = st.chat_input("请输入你的问题...")
    if not prompt:
        return

    # 显示用户消息
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # 生成回答
    with st.spinner("正在思考..."):
        mode = st.session_state["mode"]
        try:
            if mode == "高级检索":
                chain = AdvancedRAGChain(knowledge_base=st.session_state["kb"])
                result = chain.invoke_with_sources(prompt)
                answer = result["answer"]
                sources = result["sources"]
            else:
                chain = RAGChain(knowledge_base=st.session_state["kb"])
                answer = chain.invoke(prompt)
                sources = []
        except Exception as e:
            logger.error("生成回答失败: %s", e)
            answer = f"抱歉，处理时出现错误：{e}"
            sources = []

    # 显示回答
    st.chat_message("assistant").write(answer)
    st.session_state["messages"].append({"role": "assistant", "content": answer})

    # 展示来源
    _display_sources(sources)

    # 反馈
    _render_feedback(len(st.session_state["messages"]) - 1)


if __name__ == "__main__":
    run_app()
