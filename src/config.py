"""集中配置管理，基于 pydantic-settings，支持 .env 环境变量覆盖。"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── DashScope ──────────────────────────────────────────────
    dashscope_api_key: str = ""
    embedding_model: str = "text-embedding-v1"
    chat_model: str = "qwen3.8-max"

    # ── ChromaDB ───────────────────────────────────────────────
    chroma_collection: str = "RAG"
    chroma_persist_dir: str = str(BASE_DIR / "chroma-db")

    # ── Text Splitting ─────────────────────────────────────────
    chunk_size: int = 500
    chunk_overlap: int = 80
    separators: list[str] = ["\n\n", "\n", "。", "？", "！", ".", "?", "!", " "]

    # ── Retrieval ──────────────────────────────────────────────
    retrieval_top_k: int = 8
    rerank_top_n: int = 3
    bm25_top_k: int = 8
    rrf_k: int = 60  # RRF 常数

    # ── LLM Generation ────────────────────────────────────────
    temperature: float = 0.1
    max_tokens: int = 1024

    # ── Knowledge Graph ────────────────────────────────────────
    kg_enabled: bool = True

    # ── Paths ──────────────────────────────────────────────────
    data_dir: str = str(BASE_DIR / "data")
    chat_history_dir: str = str(BASE_DIR / "chat_history")
    md5_path: str = str(BASE_DIR / "md5.text")

    # ── Session ────────────────────────────────────────────────
    default_session_id: str = "user_001"


settings = Settings()
