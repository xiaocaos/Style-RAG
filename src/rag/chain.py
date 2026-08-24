"""RAG 核心链：检索增强生成。"""

from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnablePassthrough,
    RunnableWithMessageHistory,
)

from src.config import settings
from src.knowledge.base import KnowledgeBase
from src.logging_config import logger
from src.ui.history import get_history


# ── Prompt ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是服装客服助手"小爽"。请严格根据以下参考资料回答用户问题。
如果参考资料中没有相关信息，请坦诚告知，不要编造。

【参考资料】
{context}"""

HISTORY_PLACEHOLDER = MessagesPlaceholder("history")

USER_PROMPT = "{input}"


def _format_docs(docs: list) -> str:
    """将检索到的文档格式化为字符串。"""
    if not docs:
        return "暂无相关参考资料"
    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知")
        parts.append(f"[文档{i}] (来源: {source})\n{doc.page_content}")
    return "\n\n".join(parts)


def _extract_input(value: dict[str, Any]) -> str:
    return value["input"]


def _build_prompt_inputs(value: dict[str, Any]) -> dict[str, Any]:
    """重组 Runnable 输入，适配 PromptTemplate。"""
    return {
        "input": value["input"]["input"],
        "context": value["context"],
        "history": value["input"]["history"],
    }


class RAGChain:
    """封装 RAG 链，支持流式输出和多轮对话。"""

    def __init__(self, knowledge_base: KnowledgeBase | None = None) -> None:
        self._kb = knowledge_base or KnowledgeBase()
        self._llm = ChatOpenAI(
            model=settings.chat_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            streaming=True,
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            HISTORY_PLACEHOLDER,
            ("user", USER_PROMPT),
        ])
        self._parser = StrOutputParser()
        self._chain = self._build_chain()

    @property
    def retriever(self):
        return self._kb.get_retriever()

    def _build_chain(self):
        """构建 LangChain 链。"""
        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(_extract_input) | self._kb.get_retriever() | _format_docs,
            }
            | RunnableLambda(_build_prompt_inputs)
            | self._prompt
            | self._llm
            | self._parser
        )

        return RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def invoke(self, question: str, session_id: str | None = None) -> str:
        """同步调用。"""
        config = self._session_config(session_id)
        result = self._chain.invoke({"input": question}, config)
        return result

    def stream(self, question: str, session_id: str | None = None):
        """流式调用，yield 每个 token。"""
        config = self._session_config(session_id)
        return self._chain.stream({"input": question}, config)

    def invoke_with_docs(self, question: str, session_id: str | None = None) -> dict:
        """调用并返回回答 + 检索到的文档，用于 UI 展示来源。"""
        # 先检索
        docs = self._kb.get_retriever().invoke(question)
        # 再生成
        config = self._session_config(session_id)
        answer = self.invoke(question, session_id)
        return {
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "未知"),
                    "score": doc.metadata.get("score", None),
                }
                for doc in docs
            ],
        }

    @staticmethod
    def _session_config(session_id: str | None = None) -> dict:
        return {
            "configurable": {
                "session_id": session_id or settings.default_session_id,
            }
        }
