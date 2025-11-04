#!/usr/bin/env python3
"""Fetch a Langfuse trace by trace ID using the public API.

Environment:
  - LANGFUSE_PUBLIC_KEY: required
  - LANGFUSE_SECRET_KEY: required
  - LANGFUSE_HOST: required (e.g., https://cloud.langfuse.com)

Usage:
  python langfuse_trace_puller.py <TRACE_ID>

Notes:
  - Uses Basic auth against `${LANGFUSE_HOST}/api/public/traces/<TRACE_ID>`
  - Prints the JSON response to stdout, pretty-formatted
"""

from __future__ import annotations

import base64
import json
import os
import sys
from urllib import request, error

# Optional: load environment variables from a local .env file
try:
    from dotenv import load_dotenv  # type: ignore
    _DOTENV_AVAILABLE = True
except Exception:
    _DOTENV_AVAILABLE = False

if _DOTENV_AVAILABLE:
    # Attempt to load .env from the repository root (same directory as this script)
    from pathlib import Path
    _ENV_PATH = Path(__file__).parent / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)


def _env(name: str) -> str:
    val = os.getenv(name)
    if not (isinstance(val, str) and val.strip()):
        _die(f"Missing required environment variable: {name}")
    return val.strip()


def _die(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    sys.exit(code)


def _build_auth_header(public_key: str, secret_key: str) -> str:
    token = base64.b64encode(f"{public_key}:{secret_key}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def fetch_trace(host: str, public_key: str, secret_key: str, trace_id: str) -> dict:
    host = host.rstrip("/")
    url = f"{host}/api/public/traces/{trace_id}"

    headers = {
        "Authorization": _build_auth_header(public_key, secret_key),
        "Accept": "application/json",
        "User-Agent": "standard-agent/langfuse-trace-puller",
    }

    req = request.Request(url, headers=headers, method="GET")
    try:
        with request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            try:
                return json.loads(data.decode("utf-8"))
            except json.JSONDecodeError as e:
                _die(f"Langfuse returned non-JSON response (status {resp.status}): {e}")
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        _die(f"HTTP error from Langfuse: {e.code} {e.reason}\n{detail}")
    except error.URLError as e:
        _die(f"Failed to reach Langfuse host: {e.reason}")


def main(argv: list[str]) -> None:
    if len(argv) != 2 or argv[1] in {"-h", "--help"}:
        print(__doc__.strip())
        sys.exit(0 if len(argv) != 2 else 0)

    trace_id = argv[1].strip()
    if not trace_id:
        _die("TRACE_ID must be a non-empty string")

    host = _env("LANGFUSE_HOST")
    public_key = _env("LANGFUSE_PUBLIC_KEY")
    secret_key = _env("LANGFUSE_SECRET_KEY")

    result = fetch_trace(host, public_key, secret_key, trace_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv)