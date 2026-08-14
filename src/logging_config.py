"""日志配置模块。"""

import logging
import sys
from pathlib import Path

from src.config import settings


def setup_logger(name: str = "rag", level: int = logging.INFO) -> logging.Logger:
    """创建并配置 logger，同时输出到控制台和文件。"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # 文件
    log_dir = Path(settings.chat_history_dir).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()
