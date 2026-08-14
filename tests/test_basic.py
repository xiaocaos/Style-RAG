"""基础单元测试。"""

from src.config import settings


def test_settings_load():
    """配置能正常加载。"""
    assert settings.chroma_collection == "RAG"
    assert settings.chunk_size > 0
    assert settings.rerank_top_n > 0


def test_format_docs():
    """文档格式化函数。"""
    from src.rag.chain import _format_docs
    from langchain_core.documents import Document

    docs = [
        Document(page_content="测试内容1", metadata={"source": "test.txt"}),
        Document(page_content="测试内容2", metadata={"source": "test2.txt"}),
    ]
    result = _format_docs(docs)
    assert "测试内容1" in result
    assert "test.txt" in result

    # 空文档
    assert _format_docs([]) == "暂无相关参考资料"


def test_query_rewriter():
    """Query Rewriting 基础测试。"""
    from src.rag.rewriting import QueryRewriter
    rw = QueryRewriter(n_variants=2)
    # 不调用 LLM，仅测试结构
    assert rw._n == 2


def test_bm25_tokenizer():
    """BM25 中文分词。"""
    from src.rag.hybrid import _tokenize_cn
    tokens = _tokenize_cn("真丝连衣裙怎么洗")
    assert "真丝" in tokens
    assert "连衣裙" in tokens
