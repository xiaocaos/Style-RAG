"""多格式文档加载器。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document


def load_txt(file_path: str | Path) -> list[Document]:
    """加载 .txt 文件。"""
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": path.name})]


def load_documents(file_path: str | Path) -> list[Document]:
    """根据文件扩展名自动选择加载器。"""
    path = Path(file_path)
    suffix = path.suffix.lower()
    loaders = {
        ".txt": load_txt,
    }
    loader = loaders.get(suffix)
    if loader is None:
        raise ValueError(f"不支持的文件格式: {suffix}")
    return loader(path)
