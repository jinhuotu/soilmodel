# src/utils/logger.py
import logging
import sys
import os
import json
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# 日志配置常量
LOG_DIR = Path("logs")
LOG_FORMAT = "%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s"
JSON_LOG_FORMAT = {
    "timestamp": "%(asctime)s",
    "logger": "%(name)s",
    "level": "%(levelname)s",
    "message": "%(message)s",
    "module": "%(module)s",
    "function": "%(funcName)s",
    "line": "%(lineno)d"
}


class JSONFormatter(logging.Formatter):
    """结构化JSON日志格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            key: value % record.__dict__
            for key, value in JSON_LOG_FORMAT.items()
        }
        log_record["timestamp"] = datetime.utcnow().isoformat()
        return json.dumps(log_record)


class ColorFormatter(logging.Formatter):
    """带颜色输出的控制台日志格式化器"""

    COLORS = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[31;1m"  # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, "")
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


def setup_logging(
        name: str = "app",
        level: int = logging.INFO,
        enable_file_log: bool = True,
        enable_console_log: bool = True,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
) -> logging.Logger:
    """初始化并配置全局日志记录器

    Args:
        name: 日志记录器名称
        level: 日志级别
        enable_file_log: 是否启用文件日志
        enable_console_log: 是否启用控制台日志
        max_bytes: 单个日志文件最大大小
        backup_count: 保留的备份文件数量
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除已有处理器
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    # 环境检测
    is_production = os.getenv("ENVIRONMENT") == "production"
    is_ci = os.getenv("CI") == "true"

    # 控制台处理器
    if enable_console_log and not is_ci:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        if is_production:
            formatter = logging.Formatter(LOG_FORMAT)
        else:
            formatter = ColorFormatter(LOG_FORMAT)

        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件处理器（生产环境使用JSON格式）
    if enable_file_log:
        LOG_DIR.mkdir(exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=LOG_DIR / f"{name}.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(level)

        if is_production:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(LOG_FORMAT)

        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # 处理Unicode编码问题（Windows环境）
    if sys.platform == "win32":
        from logging import StreamHandler
        class SafeStreamHandler(StreamHandler):
            def emit(self, record):
                try:
                    super().emit(record)
                except UnicodeEncodeError:
                    self.handleError(record)

        console_handler = SafeStreamHandler(sys.stdout)
        console_handler.setFormatter(ColorFormatter(LOG_FORMAT))
        logger.addHandler(console_handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取配置好的日志记录器

    Args:
        name: 模块名称（默认自动获取调用模块名称）
    """
    if name is None:
        import inspect
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        name = module.__name__ if module else "__main__"

    return logging.getLogger(name)


# 初始化全局默认日志记录器
logger = setup_logging(
    level=logging.DEBUG if os.getenv("DEBUG") else logging.INFO,
    enable_file_log=os.getenv("DISABLE_FILE_LOG") != "true"
)