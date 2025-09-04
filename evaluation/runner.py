from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from .dataset import load_dataset
from .hooks import enable_instrumentation
from .metrics import aggregate, group_by
from .storage import JsonlStorage, RunRecord, make_output_path
from .tracing import JsonTracer


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


def cmd_run(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset)
    config = {}
    if args.config and Path(args.config).exists():
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model = config.get("model") or os.getenv("LLM_MODEL") or "gpt-4o"
    cfg_hash = compute_config_hash(config)
    out_path = Path(args.output) if args.output else make_output_path("./eval_runs", dataset_path.stem, short_hash(cfg_hash))
    spans_path = Path("./eval_runs/spans.jsonl")

    tracer = JsonTracer(spans_path)
    storage = JsonlStorage(out_path)

    agent = build_agent(args.agent, model=model)
    # Best-effort: pass llm if present
    llm = getattr(agent, "llm", None)
    enable_instrumentation(agent, llm=llm, tracer=tracer)

    processed = 0
    successes = 0

    for item_id, goal, expected, metadata in load_dataset(dataset_path):
        if args.limit and processed >= args.limit:
            break
        processed += 1
        try:
            result = agent.solve(goal)
            success = bool(getattr(result, "success", False))
            successes += int(success)
            rec = RunRecord(
                run_id=f"run-{processed}",
                dataset_id=dataset_path.stem,
                item_id=str(item_id),
                agent_name=agent.__class__.__name__,
                config_hash=cfg_hash,
                goal=goal,
                expected=expected,
                result=str(result),
                success=success,
                # time_ms and tokens require deeper integration; stub 0 for now
                time_ms=0,
                tokens_prompt=0,
                tokens_completion=0,
            )
            storage.append(rec)
        except Exception as e:
            rec = RunRecord(
                run_id=f"run-{processed}",
                dataset_id=dataset_path.stem,
                item_id=str(item_id),
                agent_name=agent.__class__.__name__,
                config_hash=cfg_hash,
                goal=goal,
                expected=expected,
                result=str(e),
                success=False,
                time_ms=0,
                tokens_prompt=0,
                tokens_completion=0,
                errors=[str(e)],
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


