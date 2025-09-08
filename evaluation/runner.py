from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import uuid
from datetime import datetime, timezone
import platform
import sys
from importlib import metadata as importlib_metadata
from typing import Any, Dict

# Load env vars from .env if present (for LANGFUSE_*, LLM creds, etc.)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from .dataset import load_dataset
from .instrumentation import instrument_agent, get_current_run_metrics, current_run
from .metrics import aggregate, group_by
from .storage import JsonlStorage, RunRecord, make_output_path
from .otel_setup import setup_telemetry


def compute_config_hash(config: Dict[str, Any]) -> str:
    payload = json.dumps(config or {}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def short_hash(h: str) -> str:
    return h[:8]


def build_agent(profile: str, model: str) -> Any:
    # Lazy import to avoid core coupling
    from agents.prebuilt import ReWOOAgent, ReACTAgent
    if profile.lower() == "rewoo":
        return ReWOOAgent(model=model)
    if profile.lower() == "react":
        return ReACTAgent(model=model)
    raise ValueError(f"Unknown profile: {profile}")


def _likely_secret_string(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if len(value) < 32:
        return False
    safe_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.+/=")
    return all(c in safe_chars for c in value)


def redact_config(obj: Any) -> Any:
    """Recursively redact likely secret values by key name or shape."""
    sensitive_keys = {"key", "token", "secret", "password", "auth", "credential", "api_key", "apikey"}
    if isinstance(obj, dict):
        redacted: Dict[str, Any] = {}
        for k, v in obj.items():
            if any(kw in k.lower() for kw in sensitive_keys):
                redacted[k] = "***"
            else:
                redacted[k] = redact_config(v)
        return redacted
    if isinstance(obj, list):
        return [redact_config(v) for v in obj]
    if _likely_secret_string(obj):
        return "***"
    return obj


def collect_runtime_env(agent: Any) -> Dict[str, Any]:
    pkg_versions: Dict[str, str] = {}
    for pkg in ("litellm", "standard-agent", "jentic", "structlog"):
        try:
            ver = importlib_metadata.version(pkg)
            pkg_versions[pkg] = ver
        except Exception:
            continue
    llm_model = getattr(getattr(agent, "llm", None), "model", None)
    return {
        "model": llm_model,
        "llm_wrapper": getattr(getattr(agent, "llm", None), "__class__", type("", (), {})).__name__,
        "agent_name": agent.__class__.__name__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(terse=True),
        "package_versions": pkg_versions or None,
    }


def cmd_run(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset)
    config = {}
    if args.config and Path(args.config).exists():
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model = config.get("model") or os.getenv("LLM_MODEL")
    if not model:
        raise ValueError("No model specified")
    cfg_hash = compute_config_hash(config)
    out_path = Path(args.output) if args.output else make_output_path("./eval_runs", dataset_path.stem, short_hash(cfg_hash))
    spans_path = Path("./eval_runs/spans.jsonl")

    # Configure OpenTelemetry + Langfuse
    setup_telemetry("standard-agent-eval")
    storage = JsonlStorage(out_path)

    agent = build_agent(args.agent, model=model)
    # Best-effort: pass llm if present
    llm = getattr(agent, "llm", None)
    instrument_agent(agent, llm)

    processed = 0
    successes = 0

    for item_id, goal, expected, metadata in load_dataset(dataset_path):
        if args.limit and processed >= args.limit:
            break
        processed += 1
        try:
            # reset per-run context and pre-populate metadata for span attributes
            current_run.set({
                "dataset_id": dataset_path.stem,
                "item_id": str(item_id),
                "agent_name": agent.__class__.__name__,
                "config_hash": cfg_hash,
            })
            result = agent.solve(goal)
            m = get_current_run_metrics()
            tokens_prompt = m.get("tokens_prompt")
            tokens_completion = m.get("tokens_completion")
            tokens_total = m.get("tokens_total")
            time_ms = int(m.get("duration_ms", 0))
            trace_ids = []
            trace_id = current_run.get().get("trace_id") if current_run.get() else None
            if trace_id:
                trace_ids = [trace_id]
            success = bool(getattr(result, "success", False))
            successes += int(success)
            rec = RunRecord(
                run_id=str(uuid.uuid4()),
                dataset_id=dataset_path.stem,
                item_id=str(item_id),
                agent_name=agent.__class__.__name__,
                config_hash=cfg_hash,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                goal=goal,
                expected=expected,
                result=str(result),
                success=success,
                time_ms=time_ms,
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion,
                tokens_total=tokens_total,
                agent_config=redact_config(config) if config else None,
                runtime_env=collect_runtime_env(agent),
                trace_ids=trace_ids if trace_ids else None,
            )
            storage.append(rec)
        except Exception as e:
            rec = RunRecord(
                run_id=str(uuid.uuid4()),
                dataset_id=dataset_path.stem,
                item_id=str(item_id),
                agent_name=agent.__class__.__name__,
                config_hash=cfg_hash,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                goal=goal,
                expected=expected,
                result=str(e),
                success=False,
                time_ms=0,
                tokens_prompt=None,
                tokens_completion=None,
                tokens_total=None,
                errors=[str(e)],
                agent_config=redact_config(config) if config else None,
                runtime_env=collect_runtime_env(agent),
            )
            storage.append(rec)

    print(f"Wrote records to: {out_path}")
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    storage = JsonlStorage(args.runs)
    rows = list(storage.iter_records())
    stats = aggregate(rows)
    print(f"count={stats.count} success_rate={stats.success_rate:.2%} avg_time_ms={stats.avg_time_ms:.1f} avg_tokens={stats.avg_tokens_total:.1f}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="eval")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--dataset", required=True)
    r.add_argument("--agent", required=True, choices=["rewoo", "react"])
    r.add_argument("--config", required=False)
    r.add_argument("--output", required=False)
    r.add_argument("--limit", type=int, required=False)

    a = sub.add_parser("aggregate")
    a.add_argument("--runs", required=True)

    return ap


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    ap = build_argparser()
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "aggregate":
        return cmd_aggregate(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


