from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional


SCHEMA_VERSION = "1.0"


@dataclass
class RunRecord:
    schema_version: str = SCHEMA_VERSION
    run_id: str = ""
    dataset_id: Optional[str] = None
    item_id: Optional[str] = None
    agent_name: str = ""
    agent_version: Optional[str] = None
    config_hash: str = ""
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    goal: str = ""
    expected: Optional[Any] = None
    result: Optional[Any] = None
    success: bool = False
    time_ms: int = 0
    tokens_prompt: Optional[int] = None
    tokens_completion: Optional[int] = None
    tokens_total: Optional[int] = None
    trace_ids: Optional[list[str]] = None
    agent_config: Optional[Dict[str, Any]] = None
    errors: Optional[list[str]] = None
    runtime_env: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def validate_record(r: RunRecord) -> None:
    assert isinstance(r.run_id, str) and r.run_id, "run_id required"
    assert isinstance(r.agent_name, str) and r.agent_name, "agent_name required"
    assert isinstance(r.config_hash, str) and r.config_hash, "config_hash required"
    assert isinstance(r.time_ms, int) and r.time_ms >= 0, "time_ms must be >= 0"
    if r.tokens_prompt is not None:
        assert isinstance(r.tokens_prompt, int) and r.tokens_prompt >= 0, "tokens_prompt >= 0"
    if r.tokens_completion is not None:
        assert isinstance(r.tokens_completion, int) and r.tokens_completion >= 0, "tokens_completion >= 0"
    if r.tokens_total is not None:
        assert isinstance(r.tokens_total, int) and r.tokens_total >= 0, "tokens_total >= 0"


class JsonlStorage:
    def __init__(self, output_path: str | Path) -> None:
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: RunRecord) -> None:
        validate_record(record)
        payload = asdict(record)
        # Ensure tokens_total consistency
        if payload.get("tokens_prompt") is not None and payload.get("tokens_completion") is not None:
            payload["tokens_total"] = payload.get("tokens_prompt", 0) + payload.get("tokens_completion", 0)
        line = json.dumps(payload, ensure_ascii=False)
        fd = os.open(self.path, os.O_CREAT | os.O_APPEND | os.O_WRONLY)
        try:
            with os.fdopen(fd, "a", encoding="utf-8", closefd=True) as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
        finally:
            pass

    def iter_records(self) -> Iterator[Dict[str, Any]]:
        if not self.path.exists():
            return iter(())
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def make_output_path(base_dir: str | Path, dataset_id: str, short_cfg: str) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    stamp = datetime.utcnow().strftime("%Y-%m-%d")
    filename = f"{dataset_id}__{stamp}__{short_cfg}.jsonl"
    return base / filename


__all__ = [
    "RunRecord",
    "JsonlStorage",
    "validate_record",
    "make_output_path",
    "SCHEMA_VERSION",
]


