"""
Structured logging module using structlog

Features
--------
• Structured logging with automatic context
• Console colour support
• Optional file logging with rotation
• Automatic method entry/exit tracing
"""
from __future__ import annotations
import json
import logging
import os
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Any, Dict
from functools import wraps

import structlog


def _supports_colour() -> bool:
    """True if stdout seems to handle ANSI colour codes."""
    if os.getenv("NO_COLOR"):
        return False
    if sys.platform == "win32" and os.getenv("TERM") != "xterm":
        return False
    return sys.stdout.isatty()


def _read_cfg(path: str | Path | None) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Logging config file not found: {p}")
    try:
        return json.loads(p.read_text()).get("logging", {})
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in logging config: {e}") from e


def init_logger(config_path: str | Path | None = None) -> None:
    """Configure structlog with console and optional file output."""
    cfg = _read_cfg(config_path)
    
    # Configure standard library logging first
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, cfg.get("level", "INFO").upper(), logging.INFO),
    )

    # Configure structlog processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Add console renderer
    if _supports_colour():
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=False))

    # Configure structlog
    structlog.configure(
        processors=processors,  # type: ignore[arg-type]
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Setup file logging if enabled
    file_cfg = cfg.get("file", {})
    if file_cfg.get("enabled", False):
        path = Path(file_cfg.get("path", "logs/app.log"))
        path.parent.mkdir(parents=True, exist_ok=True)

        if file_cfg.get("rotation", {}).get("enabled", True):
            handler = RotatingFileHandler(
                path,
                maxBytes=file_cfg.get("rotation", {}).get("max_bytes", 10_000_000),
                backupCount=file_cfg.get("rotation", {}).get("backup_count", 5),
            )
        else:
            handler = logging.FileHandler(path)  # type: ignore[assignment]

        handler.setLevel(getattr(logging, file_cfg.get("level", "DEBUG").upper(), logging.DEBUG))
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        
        # Add to root logger
        logging.getLogger().addHandler(handler)


def get_logger(name: str):
    """Get a structlog logger instance."""
    return structlog.get_logger(name)


def trace_method(func):
    """Decorator to automatically trace method entry/exit at debug level."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        logger = get_logger(func.__module__)
        method_name = f"{self.__class__.__name__}.{func.__name__}"
        
        logger.debug("method_entry", method=method_name)
        try:
            result = func(self, *args, **kwargs)
            logger.debug("method_exit", method=method_name, success=True)
            return result
        except Exception as e:
            logger.debug("method_exit", method=method_name, success=False, error=str(e))
            raise
    return wrapper