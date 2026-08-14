"""知识库管理：文档分片、去重、入库。"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from typing import Sequence

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.logging_config import logger


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _already_ingested(md5_str: str) -> bool:
    """检查内容是否已入库。"""
    path = settings.md5_path
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        return md5_str in (line.strip() for line in f)


def _save_md5(md5_str: str) -> None:
    with open(settings.md5_path, "a", encoding="utf-8") as f:
        f.write(md5_str + "\n")


class KnowledgeBase:
    """知识库服务：负责文档的分片、去重和向量化存储。"""

    def __init__(self) -> None:
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        self._embedding = DashScopeEmbeddings(model=settings.embedding_model)
        self._chroma = Chroma(
            collection_name=settings.chroma_collection,
            embedding_function=self._embedding,
            persist_directory=settings.chroma_persist_dir,
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=settings.separators,
            length_function=len,
        )

    @property
    def embedding(self) -> DashScopeEmbeddings:
        return self._embedding

    @property
    def vector_store(self) -> Chroma:
        return self._chroma

    def add_texts(
        self,
        texts: Sequence[str],
        source: str = "unknown",
        metadata: dict | None = None,
    ) -> int:
        """将文本入库，返回实际新增的 chunk 数量。"""
        combined = "\n".join(texts)
        md5_str = _md5(combined)
        if _already_ingested(md5_str):
            logger.info("内容已存在，跳过: %s", source)
            return 0

        # 分片
        if len(combined) > settings.chunk_size:
            chunks = self._splitter.split_text(combined)
        else:
            chunks = list(texts)

        meta = {
            "source": source,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **(metadata or {}),
        }
        metadatas = [meta] * len(chunks)

        self._chroma.add_texts(chunks, metadatas=metadatas)
        _save_md5(md5_str)
        logger.info("入库成功: %s, chunks=%d", source, len(chunks))
        return len(chunks)

    def add_documents(self, documents: Sequence[Document], source: str = "unknown") -> int:
        """将 Document 对象列表入库。"""
        combined = "\n".join(doc.page_content for doc in documents)
        md5_str = _md5(combined)
        if _already_ingested(md5_str):
            logger.info("文档已存在，跳过: %s", source)
            return 0

        self._chroma.add_documents(list(documents))
        _save_md5(md5_str)
        logger.info("文档入库成功: %s, docs=%d", source, len(documents))
        return len(documents)

    def get_retriever(self, k: int | None = None):
        """返回向量检索器。"""
        return self._chroma.as_retriever(
            search_kwargs={"k": k or settings.retrieval_top_k}
        )

    def get_stats(self) -> dict:
        """返回知识库统计信息。"""
        collection = self._chroma._collection
        count = collection.count()
        return {"total_chunks": count, "collection": settings.chroma_collection}
