"""Hybrid Search：语义检索 + BM25 关键词检索 + RRF 融合。"""

from __future__ import annotations

from typing import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.config import settings
from src.logging_config import logger


def _tokenize_cn(text: str) -> list[str]:
    """简单的中文分词（按字符 + 标点分割）。"""
    import re
    return [t for t in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+|\d+", text)]


def _bm25_search(
    query: str,
    corpus: list[Document],
    top_k: int = 8,
) -> list[tuple[Document, float]]:
    """BM25 关键词检索。"""
    tokenized_corpus = [_tokenize_cn(doc.page_content) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = _tokenize_cn(query)
    scores = bm25.get_scores(tokenized_query)

    scored_docs = sorted(
        zip(corpus, scores, strict=False),
        key=lambda x: x[1],
        reverse=True,
    )
    return [(doc, score) for doc, score in scored_docs[:top_k] if score > 0]


def _rrf_fusion(
    results_list: list[list[tuple[Document, float]]],
    k: int = 60,
) -> list[Document]:
    """Reciprocal Rank Fusion：融合多路检索结果。"""
    doc_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for results in results_list:
        for rank, (doc, _) in enumerate(results, 1):
            doc_id = doc.page_content[:100]  # 用内容前100字符作为去重key
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (k + rank)
            doc_map[doc_id] = doc

    ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_id] for doc_id, _ in ranked]


class HybridRetriever:
    """混合检索器：语义检索 + BM25 + RRF 融合。"""

    def __init__(self, vector_store: Chroma) -> None:
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        vector_k: int | None = None,
        bm25_k: int | None = None,
    ) -> list[Document]:
        """执行混合检索，返回融合后的文档列表。"""
        vk = vector_k or settings.retrieval_top_k
        bk = bm25_k or settings.bm25_top_k

        # 1. 语义检索
        vector_docs = self._vector_store.similarity_search_with_score(query, k=vk)
        vector_results = [(doc, score) for doc, score in vector_docs]

        # 2. 获取全量语料用于 BM25
        all_docs = self._vector_store.get()
        if not all_docs or not all_docs.get("documents"):
            logger.warning("向量库为空，仅使用语义检索")
            return [doc for doc, _ in vector_results]

        # 构建 Document 对象
        corpus: list[Document] = []
        metadatas = all_docs.get("metadatas", [])
        for i, content in enumerate(all_docs["documents"]):
            meta = metadatas[i] if i < len(metadatas) else {}
            corpus.append(Document(page_content=content, metadata=meta))

        # 3. BM25 检索
        bm25_results = _bm25_search(query, corpus, top_k=bk)

        # 4. RRF 融合
        fused = _rrf_fusion([vector_results, bm25_results], k=settings.rrf_k)
        logger.info(
            "Hybrid Search: vector=%d, bm25=%d, fused=%d",
            len(vector_results),
            len(bm25_results),
            len(fused),
        )
        return fused
