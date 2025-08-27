import io
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import pytest
import structlog

from utils.logger import init_logger, trace_method


def _write_cfg(tmp_path: Path, cfg: dict) -> Path:
    p = tmp_path / "logcfg.json"
    p.write_text(json.dumps(cfg))
    return p


def _find_handler(handlers, klass):
    return next((h for h in handlers if isinstance(h, klass)), None)


def test_init_logger_honors_env_console_renderer_json(monkeypatch, tmp_path):
    cfg = {
        "logging": {
            "level": "DEBUG",
            "console": {"enabled": True, "renderer": "pretty"},
            "file": {"enabled": False},
        }
    }
    cfg_path = _write_cfg(tmp_path, cfg)

    # Force env override to json
    monkeypatch.setenv("LOG_CONSOLE_RENDERER", "json")

    fake_stdout = io.StringIO()
    monkeypatch.setattr("sys.stdout", fake_stdout)

    init_logger(cfg_path)

    root = logging.getLogger()
    stream_h = _find_handler(root.handlers, logging.StreamHandler)
    assert stream_h is not None
    fmt = getattr(stream_h, "formatter", None)
    assert isinstance(fmt, structlog.stdlib.ProcessorFormatter)
    # JSON renderer should be the last processor
    assert isinstance(fmt.processors[-1], structlog.processors.JSONRenderer)


def test_init_logger_honors_config_console_renderer_pretty(monkeypatch, tmp_path):
    cfg = {
        "logging": {
            "level": "INFO",
            "console": {"enabled": True, "renderer": "pretty"},
            "file": {"enabled": False},
        }
    }
    cfg_path = _write_cfg(tmp_path, cfg)

    # Ensure no env override is present
    monkeypatch.delenv("LOG_CONSOLE_RENDERER", raising=False)

    # Avoid depending on TTY; force colour detection off
    monkeypatch.setattr("utils.logger._supports_colour", lambda: False)

    init_logger(cfg_path)

    root = logging.getLogger()
    stream_h = _find_handler(root.handlers, logging.StreamHandler)
    assert stream_h is not None
    fmt = getattr(stream_h, "formatter", None)
    assert isinstance(fmt, structlog.stdlib.ProcessorFormatter)
    # Pretty renderer should be ConsoleRenderer as the last processor
    assert isinstance(fmt.processors[-1], structlog.dev.ConsoleRenderer)


def test_init_logger_honors_config_console_renderer_json(monkeypatch, tmp_path):
    cfg = {
        "logging": {
            "level": "INFO",
            "console": {"enabled": True, "renderer": "json"},
            "file": {"enabled": False},
        }
    }
    cfg_path = _write_cfg(tmp_path, cfg)

    monkeypatch.delenv("LOG_CONSOLE_RENDERER", raising=False)

    # Keep consistent behavior
    monkeypatch.setattr("utils.logger._supports_colour", lambda: False)

    init_logger(cfg_path)

    root = logging.getLogger()
    stream_h = _find_handler(root.handlers, logging.StreamHandler)
    assert stream_h is not None
    fmt = getattr(stream_h, "formatter", None)
    assert isinstance(fmt, structlog.stdlib.ProcessorFormatter)
    assert isinstance(fmt.processors[-1], structlog.processors.JSONRenderer)


def test_init_logger_defaults_pretty_console_and_info_level(monkeypatch, tmp_path):
    # Empty config file (no console config) should default to console enabled + pretty renderer and INFO level
    cfg = {"logging": {}}
    cfg_path = _write_cfg(tmp_path, cfg)

    monkeypatch.delenv("LOG_CONSOLE_RENDERER", raising=False)
    monkeypatch.setattr("utils.logger._supports_colour", lambda: False)

    init_logger(cfg_path)

    root = logging.getLogger()
    # Level default is INFO
    assert root.level == logging.INFO
    stream_h = _find_handler(root.handlers, logging.StreamHandler)
    assert stream_h is not None
    fmt = getattr(stream_h, "formatter", None)
    assert isinstance(fmt, structlog.stdlib.ProcessorFormatter)
    assert isinstance(fmt.processors[-1], structlog.dev.ConsoleRenderer)


def test_init_logger_file_defaults_when_enabled_minimally(tmp_path):
    # With file enabled but minimal config, defaults should apply: path logs/app.log, rotation on, level DEBUG
    cfg = {
        "logging": {
            "console": {"enabled": False},
            "file": {"enabled": True},
        }
    }
    cfg_path = _write_cfg(tmp_path, cfg)

    init_logger(cfg_path)

    root = logging.getLogger()
    file_h = _find_handler(root.handlers, logging.FileHandler)
    assert file_h is not None
    # Default path
    assert Path(file_h.baseFilename).name == "app.log"
    assert Path(file_h.baseFilename).parent.name == "logs"
    # Rotation is enabled by default, so handler may be RotatingFileHandler
    assert isinstance(file_h, RotatingFileHandler)
    # Default file level is DEBUG per implementation
    assert file_h.level == logging.DEBUG

def test_init_logger_adds_rotating_file_handler(tmp_path, monkeypatch):
    log_path = tmp_path / "logs" / "app.log"
    cfg = {
        "logging": {
            "level": "INFO",
            "console": {"enabled": False},
            "file": {
                "enabled": True,
                "path": str(log_path),
                "rotation": {"enabled": True, "max_bytes": 1024, "backup_count": 1},
            },
        }
    }
    cfg_path = _write_cfg(tmp_path, cfg)

    init_logger(cfg_path)

    root = logging.getLogger()
    file_h = _find_handler(root.handlers, RotatingFileHandler)
    assert file_h is not None
    # Verify path
    assert Path(file_h.baseFilename) == log_path


def test_init_logger_adds_plain_file_handler_when_rotation_disabled(tmp_path):
    log_path = tmp_path / "plain.log"
    cfg = {
        "logging": {
            "level": "INFO",
            "console": {"enabled": False},
            "file": {
                "enabled": True,
                "path": str(log_path),
                "rotation": {"enabled": False},
            },
        }
    }
    cfg_path = _write_cfg(tmp_path, cfg)

    init_logger(cfg_path)

    root = logging.getLogger()
    # Should be a FileHandler but not RotatingFileHandler
    file_h = _find_handler(root.handlers, logging.FileHandler)
    assert file_h is not None and not isinstance(file_h, RotatingFileHandler)
    assert Path(file_h.baseFilename) == log_path


def test_init_logger_raises_for_invalid_console_renderer(tmp_path):
    cfg = {
        "logging": {
            "console": {"enabled": True, "renderer": "invalid"},
        }
    }
    cfg_path = _write_cfg(tmp_path, cfg)

    with pytest.raises(ValueError, match=r"Invalid console logging renderer option: 'invalid'.*Allowed: json, pretty"):
        init_logger(cfg_path)


def test_init_logger_missing_config_path_raises():
    with pytest.raises(FileNotFoundError):
        init_logger("/nonexistent/path/config.json")


def test_init_logger_invalid_json_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not: valid }")
    with pytest.raises(ValueError):
        init_logger(bad)


def test_trace_method_logs_entry_exit_and_errors(monkeypatch):
    calls = []

    class Capture:
        def debug(self, event, **kwargs):  # type: ignore[no-untyped-def]
            calls.append((event, kwargs))

    # Ensure our decorator uses the capture logger
    monkeypatch.setattr("utils.logger.get_logger", lambda name: Capture())

    class Sample:
        @trace_method
        def ok(self):  # type: ignore[no-untyped-def]
            return "OK"

        @trace_method
        def fail(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("X")

    s = Sample()
    # Success path
    out = s.ok()
    assert out == "OK"
    assert calls[0][0] == "method_entry"
    assert calls[0][1]["method"] == "Sample.ok"
    assert calls[1][0] == "method_exit"
    assert calls[1][1]["method"] == "Sample.ok"
    assert calls[1][1]["success"] is True

    # Error path
    calls.clear()
    with pytest.raises(RuntimeError):
        s.fail()
    assert calls[0][0] == "method_entry"
    assert calls[1][0] == "method_exit"
    assert calls[1][1]["method"] == "Sample.fail"
    assert calls[1][1]["success"] is False
    assert "error" in calls[1][1]


