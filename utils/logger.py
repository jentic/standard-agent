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

# Structured logging constants
TIME_KEY = "@timestamp"

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


def _base_processors() -> list:
    return [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="ISO", key=TIME_KEY),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

def _pre_chain() -> list:
    """Processors to normalize stdlib LogRecord into structlog shape before rendering."""
    return [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="ISO", key=TIME_KEY),
    ]

def _build_console_handler(level: int, renderer: str) -> logging.Handler:
    """Build a console handler with either pretty or JSON rendering."""

    processors: list[Any]
    if renderer == "json":
        processors = [
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.EventRenamer(to="message"),
            structlog.processors.JSONRenderer(),
        ]
    else:  # pretty (default)
        processors = [
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=_supports_colour(), timestamp_key=TIME_KEY),
        ]

    formatter = structlog.stdlib.ProcessorFormatter(foreign_pre_chain=_pre_chain(), processors=processors,)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler


def _build_file_handler(file_cfg: Dict[str, Any]) -> logging.Handler:
    path = Path(file_cfg.get("path", "logs/app.log"))
    path.parent.mkdir(parents=True, exist_ok=True)

    if file_cfg.get("rotation", {}).get("enabled", True):
        handler: logging.Handler = RotatingFileHandler(
            path,
            maxBytes=file_cfg.get("rotation", {}).get("max_bytes", 10_000_000),
            backupCount=file_cfg.get("rotation", {}).get("backup_count", 5),
        )
    else:
        handler = logging.FileHandler(path)  # type: ignore[assignment]

    handler.setLevel(getattr(logging, file_cfg.get("level", "DEBUG").upper(), logging.DEBUG))
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_pre_chain(),
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.EventRenamer(to="message"),
            structlog.processors.JSONRenderer(),
        ],
    )
    handler.setFormatter(formatter)
    return handler


def init_logger(config_path: str | Path | None = None) -> None:
    """Structured logging setup.

    Console renderer is configurable via config, with env override support when needed:
    - Config: logging.console.renderer = "json" | "pretty" (default: pretty)
    - Env (optional): LOG_CONSOLE_RENDERER=pretty|json (wins over config)

    Files always use JSON with optional rotation.
    """
    cfg = _read_cfg(config_path)

    root_level = getattr(logging, cfg.get("level", "INFO").upper(), logging.INFO)

    # Defer rendering to handlers
    structlog.configure(
        processors=_base_processors() + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],  # type: ignore[arg-type]
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    handlers: list[logging.Handler] = []

    # Console handler enabled by default
    console_enabled = cfg.get("console", {}).get("enabled", True)
    if console_enabled:
        renderer = os.getenv("LOG_CONSOLE_RENDERER") or cfg.get("console", {}).get("renderer", "pretty").strip().lower()
        if renderer not in {"json", "pretty"}:
            raise ValueError(f"Invalid console logging renderer option: {renderer!r}. Allowed: json, pretty")
        handlers.append(_build_console_handler(root_level, renderer))

    # File handler (JSON) if enabled
    file_cfg = cfg.get("file", {})
    if file_cfg.get("enabled", False):
        handlers.append(_build_file_handler(file_cfg))

    logging.basicConfig(level=root_level, handlers=handlers, force=True)


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