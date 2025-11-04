#!/usr/bin/env python3
"""Trace collector for workflow mining (Langfuse → filtered signals).

Usage:
  python workflow_miner/trace_collector.py <TRACE_ID>

Env:
  LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
  (Optional .env at repo root will be loaded if python-dotenv is installed.)

Outputs (per-trace directory):
  workflow_miner/logs/<TRACE_ID>/trace.json
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from urllib import request, error


# Optional .env loader
try:
    from dotenv import load_dotenv  # type: ignore
    _HAS_DOTENV = True
except Exception:
    _HAS_DOTENV = False


REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "workflow_miner" / "logs"


def _load_env_if_available() -> None:
    if not _HAS_DOTENV:
        return
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _env(name: str) -> str:
    val = os.getenv(name)
    if not (isinstance(val, str) and val.strip()):
        _die(f"Missing required environment variable: {name}")
    return val.strip()


def _die(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    sys.exit(code)


def _auth_header(pub: str, sec: str) -> str:
    token = base64.b64encode(f"{pub}:{sec}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def fetch_trace(host: str, public_key: str, secret_key: str, trace_id: str) -> Dict[str, Any]:
    host = host.rstrip("/")
    url = f"{host}/api/public/traces/{trace_id}"
    headers = {
        "Authorization": _auth_header(public_key, secret_key),
        "Accept": "application/json",
        "User-Agent": "standard-agent/workflow-miner",
    }
    req = request.Request(url, headers=headers, method="GET")
    try:
        with request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            return json.loads(data.decode("utf-8"))
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        _die(f"HTTP error from Langfuse: {e.code} {e.reason}\n{detail}")
    except error.URLError as e:
        _die(f"Failed to reach Langfuse host: {e.reason}")
    except json.JSONDecodeError as e:
        _die(f"Langfuse returned non-JSON response: {e}")


def ensure_log_dir() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR


def get_trace_dir(trace_id: str) -> Path:
    """Return per-trace directory path and ensure it exists."""
    d = LOG_DIR / trace_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_spans_generic(trace_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """(Deprecated extraction helper retained for future use.)"""
    return []


def _get_or_none(d: Dict[str, Any], key: str) -> Any:
    return d.get(key) if isinstance(d, dict) else None


def project_trace_minimal(trace_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Project a Langfuse trace object into a strict minimal shape without heuristics.

    Top-level: id, projectId, name, timestamp (fallback to createdAt only if timestamp missing),
    input, output, observations (projected list).
    Observations items: id, traceId, startTime, endTime, name, input, output, createdAt, updatedAt.
    Missing fields are set to null; no derivations from other fields.
    """
    top_id = _get_or_none(trace_obj, "id")
    top_project = _get_or_none(trace_obj, "projectId")
    top_name = _get_or_none(trace_obj, "name")
    top_timestamp = _get_or_none(trace_obj, "timestamp") or _get_or_none(trace_obj, "createdAt")
    top_input = _get_or_none(trace_obj, "input")
    top_output = _get_or_none(trace_obj, "output")

    raw_obs = trace_obj.get("observations")
    obs_list: List[Dict[str, Any]] = raw_obs if isinstance(raw_obs, list) else []

    projected_obs: List[Dict[str, Any]] = []
    for o in obs_list:
        if not isinstance(o, dict):
            continue
        projected_obs.append({
            "id": _get_or_none(o, "id"),
            "traceId": _get_or_none(o, "traceId"),
            "startTime": _get_or_none(o, "startTime"),
            "endTime": _get_or_none(o, "endTime"),
            "name": _get_or_none(o, "name"),
            "input": _get_or_none(o, "input"),
            "output": _get_or_none(o, "output"),
            "createdAt": _get_or_none(o, "createdAt"),
            "updatedAt": _get_or_none(o, "updatedAt"),
        })

    # Order observations oldest-first by startTime; fallback to createdAt then endTime
    def _obs_sort_key(obs: Dict[str, Any]) -> str:
        t = (
            (obs.get("startTime") or "")
            or (obs.get("createdAt") or "")
        )
        return str(t)

    projected_obs.sort(key=_obs_sort_key)

    return {
        "id": top_id,
        "projectId": top_project,
        "name": top_name,
        "timestamp": top_timestamp,
        "input": top_input,
        "output": top_output,
        "observations": projected_obs,
    }


def main(argv: List[str]) -> None:
    if len(argv) != 2 or argv[1] in {"-h", "--help"}:
        print(__doc__.strip())
        sys.exit(0)

    _load_env_if_available()
    ensure_log_dir()

    trace_id = argv[1].strip()
    if not trace_id:
        _die("TRACE_ID must be a non-empty string")

    host = _env("LANGFUSE_HOST")
    pub = _env("LANGFUSE_PUBLIC_KEY")
    sec = _env("LANGFUSE_SECRET_KEY")

    trace_obj = fetch_trace(host, pub, sec, trace_id)

    trace_dir = get_trace_dir(trace_id)
    raw_path = trace_dir / "trace.json"
    save_json(raw_path, trace_obj)

    # Minimal filtered projection (no heuristics)
    filtered_obj = project_trace_minimal(trace_obj)
    filtered_path = trace_dir / "filtered.json"
    save_json(filtered_path, filtered_obj)

    print(f"Saved: {raw_path}")
    print(f"Saved: {filtered_path}")


if __name__ == "__main__":
    main(sys.argv)


