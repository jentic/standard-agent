#!/usr/bin/env python3
"""
scripts/validate_config.py

Validate agent configuration files and environment before runtime.

Features:
- Validate JSON or YAML configuration files
- Check required environment variables
- Support agent-type–specific validation
- Optional network connectivity checks
- CLI usage with helpful output

Usage:
    python scripts/validate_config.py --config path/to/config.yaml
"""

from __future__ import annotations
import argparse
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional YAML and JSON Schema support
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

try:
    import jsonschema  # type: ignore
except Exception:
    jsonschema = None  # type: ignore


# -----------------------------
# Load Config (JSON/YAML)
# -----------------------------
def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text(encoding="utf-8")

    # Try JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try YAML
    if yaml is not None:
        try:
            return yaml.safe_load(text) or {}
        except Exception as e:
            raise ValueError(f"Failed to parse YAML: {e}") from e

    # Very basic fallback (not recommended)
    data: Dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, val = line.split(":", 1)
            data[key.strip()] = val.strip().strip("\"'")

    if data:
        return data

    raise ValueError("Unable to parse config file. Install PyYAML for proper YAML support.")


# -----------------------------
# Environment Variable Checks
# -----------------------------
def check_env_vars(names: List[str]) -> Tuple[bool, List[str]]:
    missing = [name for name in names if not os.environ.get(name)]
    return (len(missing) == 0, missing)


# -----------------------------
# Schema Checks (Lightweight)
# -----------------------------
def basic_validate_config_schema(cfg: Dict[str, Any], agent_type: Optional[str] = None) -> List[str]:
    errors: List[str] = []

    # Common required fields
    if "agent_name" not in cfg:
        errors.append("Missing required key: 'agent_name'.")

    # Agent-type–specific logic
    at = agent_type or cfg.get("agent_type")
    if at:
        if at == "discord_bot":
            if "discord" not in cfg:
                errors.append("agent_type 'discord_bot' requires a 'discord' section.")
            else:
                if "token" not in cfg["discord"] and "DISCORD_TOKEN" not in os.environ:
                    errors.append("Discord bot requires 'discord.token' or DISCORD_TOKEN env var.")

        elif at == "http_agent":
            if "endpoint" not in cfg:
                errors.append("agent_type 'http_agent' requires an 'endpoint' key.")

        # Here you can extend to other agent types
    else:
        errors.append("No 'agent_type' specified. Some validations were skipped.")

    return errors


# -----------------------------
# Network Check
# -----------------------------
def try_connect_host(url_or_host: str, timeout: float = 3.0) -> Tuple[bool, str]:
    # Extract host:port
    if "://" in url_or_host:
        try:
            host = url_or_host.split("://", 1)[1].split("/", 1)[0]
        except Exception:
            host = url_or_host
    else:
        host = url_or_host

    if ":" in host:
        h, p = host.split(":", 1)
        try:
            port = int(p)
        except Exception:
            port = 443
    else:
        h, port = host, 443

    try:
        addr_info = socket.getaddrinfo(h, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except Exception as e:
        return False, f"DNS/resolve error for {h}:{port} — {e}"

    for fam, socktype, proto, canonname, sa in addr_info:
        try:
            s = socket.socket(fam, socktype, proto)
            s.settimeout(timeout)
            s.connect(sa)
            s.close()
            return True, f"Connected to {h}:{port} ({sa})"
        except Exception:
            continue

    return False, f"Failed to connect to {h}:{port}."


# -----------------------------
# CLI Entry Point
# -----------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate agent configuration and environment."
    )

    parser.add_argument("--config", "-c", type=Path, required=True,
                        help="Path to JSON/YAML config file.")
    parser.add_argument("--env-vars", "-e", type=str,
                        help="Comma-separated list of environment vars to check.")
    parser.add_argument("--agent-type", "-t", type=str,
                        help="Force agent type (overrides config).")
    parser.add_argument("--check-network", action="store_true",
                        help="Check network connectivity for endpoints found in config.")
    parser.add_argument("--verbose", "-v", action="count", default=0,
                        help="Increase verbosity.")

    args = parser.parse_args(argv)

    problems: List[str] = []
    exit_code = 0

    # Load config
    try:
        cfg = load_config(args.config)
    except Exception as e:
        print(f"ERROR: Unable to load config: {e}", file=sys.stderr)
        return 2

    # Env var list
    if args.env_vars:
        env_list = [x.strip() for x in args.env_vars.split(",") if x.strip()]
    else:
        env_list = [
            "JENTIC_AGENT_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
        ]

    ok, missing = check_env_vars(env_list)
    if not ok:
        problems.append(f"Missing environment variables: {', '.join(missing)}")
        exit_code = max(exit_code, 3)

    # Schema validation
    schema_errors = basic_validate_config_schema(cfg, args.agent_type)
    if schema_errors:
        problems.extend(schema_errors)
        exit_code = max(exit_code, 4)

    # Optional jsonschema validation
    if jsonschema is not None and isinstance(cfg, dict) and "$schema" in cfg:
        if args.verbose:
            print("jsonschema detected — strict validation available (schema must be provided).")

    # Network checks
    if args.check_network:
        def discover_urls(obj):
            if isinstance(obj, dict):
                for v in obj.values():
                    yield from discover_urls(v)
            elif isinstance(obj, list):
                for v in obj:
                    yield from discover_urls(v)
            elif isinstance(obj, str):
                if "http://" in obj or "https://" in obj or ":" in obj:
                    yield obj

        endpoints = list(discover_urls(cfg))

        for ep in endpoints:
            ok, msg = try_connect_host(ep)
            if not ok:
                problems.append(f"Network check failed for {ep}: {msg}")
                exit_code = max(exit_code, 6)
            elif args.verbose:
                print(msg)

    # Final Output
    if problems:
        print("\nValidation FAILED:\n", file=sys.stderr)
        for p in problems:
            print(f" - {p}", file=sys.stderr)
        return exit_code

    print("Validation OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
