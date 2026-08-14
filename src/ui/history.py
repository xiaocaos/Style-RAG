"""基于文件的聊天历史持久化。"""

from __future__ import annotations

import json
import os
from pathlib import Path

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

from src.config import settings


class FileChatMessageHistory(BaseChatMessageHistory):
    """将聊天记录以 JSON 文件形式持久化到磁盘。"""

    def __init__(self, session_id: str, storage_dir: str | None = None) -> None:
        self.session_id = session_id
        self._dir = Path(storage_dir or settings.chat_history_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / f"{session_id}.json"

    def _load(self) -> list[BaseMessage]:
        if not self._file.exists():
            return []
        with open(self._file, "r", encoding="utf-8") as f:
            return messages_from_dict(json.load(f))

    def _save(self, messages: list[BaseMessage]) -> None:
        with open(self._file, "w", encoding="utf-8") as f:
            json.dump([message_to_dict(m) for m in messages], f, ensure_ascii=False)

    @property
    def messages(self) -> list[BaseMessage]:
        return self._load()

    def add_messages(self, messages: list[BaseMessage]) -> None:
        all_msgs = self._load()
        all_msgs.extend(messages)
        self._save(all_msgs)

    def clear(self) -> None:
        self._save([])


_history_store: dict[str, FileChatMessageHistory] = {}


def get_history(session_id: str | None = None) -> FileChatMessageHistory:
    """获取或创建聊天历史实例（单例缓存）。"""
    sid = session_id or settings.default_session_id
    if sid not in _history_store:
        _history_store[sid] = FileChatMessageHistory(sid)
    return _history_store[sid]
