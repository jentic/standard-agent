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
    _enable = _supports_colour()

    def filter(self, record: logging.LogRecord) -> bool:
        colour = _ANSI.get(record.levelname, "")
        record.levelname_coloured = (
            f"{colour}{record.levelname}{_RESET}" if self._enable and colour else record.levelname
        )
        return True


# ------------------------------------------------------------------- config helpers
_DEFAULT_FMT = "%(asctime)s | %(levelname_coloured)s | %(name)s: %(message)s"
_DEFAULT_FILE_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


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


# ------------------------------------------------------------------- public API
def init_logger(config_path: str | Path | None = None) -> None:
    """
    Configure the *root* logger.

    If root already has handlers, we respect the existing configuration.
    """
    root = logging.getLogger()
    if root.handlers:
        return  # app/framework configured logging already – do nothing

    cfg = _read_cfg(config_path)

    level = getattr(logging, cfg.get("level", "INFO").upper(), logging.INFO)
    root.setLevel(level)

    # ---------------- console ----------------
    if cfg.get("console", {}).get("enabled", True):
        h = logging.StreamHandler(stream=sys.stdout)
        h.addFilter(_ColourFilter())
        fmt = cfg.get("console", {}).get("format", _DEFAULT_FMT)
        h.setFormatter(logging.Formatter(fmt))
        root.addHandler(h)

    # ---------------- file -------------------
    file_cfg = cfg.get("file", {})
    if file_cfg.get("enabled", False):
        path = Path(file_cfg.get("path", "logs/app.log"))
        path.parent.mkdir(parents=True, exist_ok=True)

        if file_cfg.get("rotation", {}).get("enabled", True):
            h = RotatingFileHandler(
                path,
                maxBytes=file_cfg.get("rotation", {}).get("max_bytes", 10_000_000),
                backupCount=file_cfg.get("rotation", {}).get("backup_count", 5),
            )
        else:
            h = logging.FileHandler(path)

        h.setLevel(getattr(logging, file_cfg.get("level", "DEBUG").upper(), logging.DEBUG))
        h.setFormatter(logging.Formatter(file_cfg.get("format", _DEFAULT_FILE_FMT)))
        root.addHandler(h)


def reload_logger(config_path: str | Path) -> None:
    """
    Reconfigure logging at runtime (clears current root handlers).
    """
    logging.getLogger().handlers.clear()
    init_logger(config_path)


def get_logger(name: str) -> logging.Logger:
    """Shortcut to `logging.getLogger(name)`."""
    return logging.getLogger(name)


init_logger()