"""
Lightweight logging module

Features
--------
• Console colour (auto-detect; honour NO_COLOR)
• Optional rotating file handler
• One-shot initialiser – respects existing root config
• Tiny (≈ 80 LOC), stdlib-only
"""
from __future__ import annotations
import json, logging, os, sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List
from .load_config import load_config
import dataclasses

# ------------------------------------------------------------------- colour
_ANSI = {
    "DEBUG": "\033[36m",     # cyan
    "INFO": "\033[32m",      # green
    "WARNING": "\033[33m",   # yellow
    "ERROR": "\033[31m",     # red
    "CRITICAL": "\033[35m",  # magenta
}
_RESET = "\033[0m"


def _supports_colour() -> bool:
    """True if stdout seems to handle ANSI colour codes."""
    if os.getenv("NO_COLOR"):
        return False
    if sys.platform == "win32" and os.getenv("TERM") != "xterm":
        return False
    return sys.stdout.isatty()


class _ColourFilter(logging.Filter):
    """Adds levelname_coloured to each record."""
    def __init__(self, enable: bool):
        super().__init__()
        self._enable = enable

    def filter(self, record: logging.LogRecord) -> bool:
        colour = _ANSI.get(record.levelname, "")
        record.levelname_coloured = (
            f"{colour}{record.levelname}{_RESET}" if self._enable and colour else record.levelname
        )
        return True


# ------------------------------------------------------------------- config helpers
_DEFAULT_FMT = "%(asctime)s | %(levelname_coloured)s | %(name)s: %(message)s"
_DEFAULT_FILE_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def init_logger() -> None:
    """
    Configure the *root* logger.

    If root already has handlers, we respect the existing configuration.
    """
    root = logging.getLogger()
    if root.handlers:
        return  # app/framework configured logging already – do nothing

    config = load_config()
    logging_cfg = config.logging

    # Set root level
    level = getattr(logging, logging_cfg.console.level.upper(), logging.INFO)
    root.setLevel(level)

    # ---------------- console ----------------
    if logging_cfg.console.enabled:
        h = logging.StreamHandler(stream=sys.stdout)
        h.addFilter(_ColourFilter(logging_cfg.console.colour_enabled))
        fmt = logging_cfg.console.format or _DEFAULT_FMT
        h.setFormatter(logging.Formatter(fmt))
        root.addHandler(h)

    # ---------------- file -------------------
    file_cfg = logging_cfg.file
    if file_cfg.enabled:
        path = Path(file_cfg.path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if file_cfg.file_rotation:
            h = RotatingFileHandler(
                path,
                maxBytes=file_cfg.max_bytes,
                backupCount=file_cfg.backup_count,
            )
        else:
            h = logging.FileHandler(path)

        h.setLevel(getattr(logging, file_cfg.level.upper(), logging.DEBUG))
        h.setFormatter(logging.Formatter(file_cfg.format or _DEFAULT_FILE_FMT))
        root.addHandler(h)

    # ---------------- library level loggers -------------------
    libraries_config = logging_cfg.libraries
    for field in dataclasses.fields(libraries_config):
        lib_name = field.name
        level = getattr(libraries_config, lib_name)
        logging.getLogger(lib_name).setLevel(level.upper())


def reload_logger() -> None:
    """
    Reconfigure logging at runtime (clears current root handlers).
    """
    logging.getLogger().handlers.clear()
    init_logger()


def get_logger(name: str) -> logging.Logger:
    """Shortcut to `logging.getLogger(name)`."""
    return logging.getLogger(name)