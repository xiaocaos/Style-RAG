"""Query Rewriting：将用户口语化问题改写为多个精确检索查询。"""

from __future__ import annotations

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate

from src.config import settings
from src.logging_config import logger

_REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """你是一个查询改写专家。请将用户的口语化问题改写为 {n} 个更精确、更适合向量检索的查询。
每个查询占一行，不要编号，不要解释。

用户问题: {question}"""
)


class QueryRewriter:
    """将用户问题改写为多个检索查询变体。"""

    def __init__(self, n_variants: int = 3) -> None:
        self._n = n_variants
        self._llm = ChatTongyi(
            model=settings.chat_model,
            temperature=0.3,
            max_tokens=256,
        )

    def rewrite(self, question: str) -> list[str]:
        """返回改写后的查询列表（包含原始查询）。"""
        try:
            chain = _REWRITE_PROMPT | self._llm
            result = chain.invoke({"question": question, "n": self._n})
            variants = [line.strip() for line in result.content.strip().split("\n") if line.strip()]
            # 去重并确保原始问题在内
            seen: set[str] = set()
            unique: list[str] = []
            for q in [question] + variants:
                if q not in seen:
                    seen.add(q)
                    unique.append(q)
            logger.info("Query Rewriting: %d -> %d 变体", 1, len(unique))
            return unique
        except Exception as e:
            logger.warning("Query Rewriting 失败，使用原始查询: %s", e)
            return [question]
