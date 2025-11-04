#!/usr/bin/env python3
"""Compose an Arazzo workflow YAML from a mined trace using an LLM (LiteLLM).

Usage:
  python workflow_miner/arazzo_composer.py <TRACE_ID> [OUTPUT_YAML]

Inputs/Assumptions:
  - Expects per-trace directory created by trace_collector:
      workflow_miner/logs/<TRACE_ID>/filtered.json
    (This script will proactively run the trace collector for the given TRACE_ID
     before composing, so a single command completes the full flow.)
  - Loads .env if available (for LLM_MODEL and provider credentials).
  - Uses agents.llm.litellm.LiteLLM for the LLM call.

Output:
  - Writes Arazzo YAML to OUTPUT_YAML or
      workflow_miner/logs/<TRACE_ID>/workflow.arazzo.yaml

Notes:
  - First-pass, generalizable: prompts the LLM to infer steps from selected
    observations without overfitting to a specific API. The prompt encourages
    Arazzo 1.0.1 structure while allowing placeholders if details are missing.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure repo root for local imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Optional .env loader
try:
    from dotenv import load_dotenv  # type: ignore
    _HAS_DOTENV = True
except Exception:
    _HAS_DOTENV = False

from agents.llm.litellm import LiteLLM
from workflow_miner import trace_collector as collector
import yaml


def _load_env_if_available() -> None:
    if not _HAS_DOTENV:
        return
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _read_filtered(trace_id: str) -> Dict[str, Any]:
    p = PROJECT_ROOT / "workflow_miner" / "logs" / trace_id / "filtered.json"
    if not p.is_file():
        raise FileNotFoundError(f"Filtered trace not found at: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _truncate_for_prompt(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated] ..."


def _build_prompt(filtered: Dict[str, Any]) -> str:
    # Keep the prompt generic and generalizable.
    observations_json = json.dumps(filtered.get("observations", []), ensure_ascii=False, indent=2)
    observations_json = _truncate_for_prompt(observations_json, 12000)

    top_goal_hint = ""
    top_input = filtered.get("input")
    if isinstance(top_input, dict) and top_input:
        # Provide a light hint for context if there is a goal-like field
        goal_text = top_input.get("goal") or ""
        if isinstance(goal_text, str) and goal_text.strip():
            top_goal_hint = f"Goal: {goal_text.strip()}\n"

    # Load prompt template from workflow_miner/arazzo.yaml (same folder as this script)
    prompt_path = PROJECT_ROOT / "workflow_miner" / "arazzo.yaml"
    data = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    template = data.get("compose", "")
    return template.format(goal_hint=top_goal_hint, observations_json=observations_json)


def _resolve_output_path(trace_id: str, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    # default path in trace folder
    d = PROJECT_ROOT / "workflow_miner" / "logs" / trace_id
    d.mkdir(parents=True, exist_ok=True)
    return d / "workflow.arazzo.yaml"


def main(argv: list[str]) -> None:
    if len(argv) < 2 or argv[1] in {"-h", "--help"}:
        print(__doc__.strip())
        sys.exit(0)

    trace_id = argv[1].strip()
    out_path_arg = argv[2].strip() if len(argv) >= 3 else None

    _load_env_if_available()

    # Construct LLM
    model = os.getenv("LLM_MODEL")
    if not (isinstance(model, str) and model.strip()):
        raise SystemExit("LLM_MODEL is not set in environment (.env).")
    llm = LiteLLM(model=model)

    # Ensure filtered trace exists by running the collector for this trace id
    # This will fetch from Langfuse and write trace.json and filtered.json
    collector.main(["trace_collector.py", trace_id])

    # Read filtered observations
    filtered = _read_filtered(trace_id)
    prompt = _build_prompt(filtered)

    # Call LLM to generate the Arazzo YAML
    yaml_text = llm.prompt(prompt)
    out_path = _resolve_output_path(trace_id, out_path_arg)
    out_path.write_text(yaml_text, encoding="utf-8")

    print(f"Saved Arazzo: {out_path}")


if __name__ == "__main__":
    main(sys.argv)


