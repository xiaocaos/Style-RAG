"""高级 RAG 流水线：Query Rewriting → Hybrid Search → Reranking → Generation。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
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
from src.rag.hybrid import HybridRetriever
from src.rag.reranker import Reranker
from src.rag.rewriting import QueryRewriter
from src.ui.history import get_history


# ── 数据结构 ───────────────────────────────────────────────────
@dataclass
class RetrievalResult:
    """检索结果。"""
    documents: list[Document] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    mode: str = "basic"


# ── Prompt ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是服装客服助手"小爽"。请严格根据以下参考资料回答用户问题。
回答要求：
1. 简洁专业，直击要点
2. 如果参考资料中没有相关信息，请坦诚告知
3. 适当引用资料中的关键数据（如具体温度、尺码范围等）

【参考资料】
{context}"""

USER_PROMPT = "{input}"


def _format_docs(docs: list[Document]) -> str:
    if not docs:
        return "暂无相关参考资料"
    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知")
        parts.append(f"[文档{i}] (来源: {source})\n{doc.page_content}")
    return "\n\n".join(parts)


# ── 高级 RAG Chain ─────────────────────────────────────────────
class AdvancedRAGChain:
    """完整高级 RAG 流水线。

    流程: Query Rewriting → Hybrid Search → Reranking → LLM Generation
    """

    def __init__(self, knowledge_base: KnowledgeBase | None = None) -> None:
        self._kb = knowledge_base or KnowledgeBase()
        self._rewriter = QueryRewriter(n_variants=3)
        self._hybrid = HybridRetriever(self._kb.vector_store)
        self._reranker = Reranker()
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
            MessagesPlaceholder("history"),
            ("user", USER_PROMPT),
        ])
        self._parser = StrOutputParser()
        self._chain = self._build_chain()

    def _retrieve(self, question: str) -> list[Document]:
        """执行检索流水线：Rewriting → Hybrid → Reranking。"""
        # 1. Query Rewriting
        queries = self._rewriter.rewrite(question)
        logger.info("改写查询: %s", queries)

        # 2. Hybrid Search（对每个变体检索，合并去重）
        all_docs: list[Document] = []
        for q in queries:
            docs = self._hybrid.retrieve(q, vector_k=5, bm25_k=5)
            all_docs.extend(docs)

        # 去重（基于内容前100字符）
        seen: set[str] = set()
        unique_docs: list[Document] = []
        for doc in all_docs:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        logger.info("Hybrid Search 合并后: %d 文档", len(unique_docs))

        # 3. Reranking
        reranked = self._reranker.rerank(question, unique_docs, top_n=settings.rerank_top_n)
        return reranked

    def _build_chain(self):
        def retrieve_fn(input_data: dict) -> str:
            question = input_data["input"] if isinstance(input_data, dict) else str(input_data)
            docs = self._retrieve(question)
            return _format_docs(docs)

        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(retrieve_fn),
            }
            | RunnableLambda(lambda v: {"input": v["input"]["input"], "context": v["context"], "history": v["input"]["history"]})
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
        config = self._session_config(session_id)
        return self._chain.invoke({"input": question}, config)

    def stream(self, question: str, session_id: str | None = None):
        config = self._session_config(session_id)
        return self._chain.stream({"input": question}, config)

    def invoke_with_sources(self, question: str, session_id: str | None = None) -> dict:
        """调用并返回回答 + 检索来源。"""
        docs = self._retrieve(question)
        answer = self.invoke(question, session_id)
        return {
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "未知"),
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
