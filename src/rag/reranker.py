"""Reranking：使用交叉编码器对检索结果精细重排序。"""

from __future__ import annotations

from langchain_core.documents import Document

from src.config import settings
from src.logging_config import logger


class Reranker:
    """交叉编码器重排序器。使用 sentence-transformers 的交叉编码器模型。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self._model_name = model_name
        self._model = None

    def _load_model(self):
        """延迟加载模型（首次调用时加载）。"""
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name, max_length=512)
            logger.info("Reranker 模型加载成功: %s", self._model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers 未安装，跳过 Reranking。"
                "运行: pip install sentence-transformers"
            )
        except Exception as e:
            logger.warning("Reranker 模型加载失败: %s", e)

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int | None = None,
    ) -> list[Document]:
        """对文档列表重排序，返回 top_n 结果。"""
        n = top_n or settings.rerank_top_n
        if not documents:
            return []

        self._load_model()
        if self._model is None:
            # 模型不可用时直接截断返回
            return documents[:n]

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs)

        scored_docs = sorted(
            zip(documents, scores, strict=False),
            key=lambda x: x[1],
            reverse=True,
        )
        result = [doc for doc, _ in scored_docs[:n]]
        logger.info("Reranking: %d -> %d", len(documents), len(result))
        return result
