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
from typing import Any, Dict, List
import re

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


def _extract_plan_steps(filtered: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract ReWOO plan steps from observations, if present."""
    plan_obs = [o for o in filtered.get("observations", []) if isinstance(o, dict) and o.get("name") == "agents.reasoner.rewoo.ReWOOReasoner._plan"]
    if not plan_obs:
        return []
    out = plan_obs[-1].get("output")
    if not isinstance(out, str):
        return []
    steps: List[Dict[str, Any]] = []
    step_re = re.compile(r"Step\(text='(.*?)'.*?output_key='(.*?)'.*?input_keys=\[(.*?)\]", re.DOTALL)
    for m in step_re.finditer(out):
        text = m.group(1).strip()
        output_key = m.group(2).strip() or None
        raw_inputs = m.group(3).strip()
        input_keys: List[str] = []
        if raw_inputs:
            for tok in raw_inputs.split(','):
                v = tok.strip().strip("'\"")
                if v:
                    input_keys.append(v)
        steps.append({"text": text, "output_key": output_key, "input_keys": input_keys})
    return steps


def _extract_executions(filtered: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract successful execute calls (order preserved as in filtered observations).

    Success if: level != "ERROR" and statusMessage empty/absent and output present.
    """
    execs: List[Dict[str, Any]] = []
    for o in filtered.get("observations", []):
        if not isinstance(o, dict):
            continue
        if o.get("name") != "agents.tools.jentic.JenticClient.execute":
            continue
        level = str(o.get("level") or "").upper()
        status_msg = o.get("statusMessage")
        output = o.get("output")
        if level == "ERROR":
            continue
        if isinstance(status_msg, str) and status_msg.strip():
            continue
        if not output:
            continue
        tool_id = None
        tool_field = None
        inp = o.get("input")
        if isinstance(inp, dict):
            tool_field = inp.get("tool")
        elif isinstance(inp, str):
            tool_field = inp
        if isinstance(tool_field, str):
            m = re.search(r"JenticTool\('([^']+)'", tool_field)
            if m:
                tool_id = m.group(1)
        execs.append({
            "tool_id": tool_id,
            "startTime": o.get("startTime"),
            "endTime": o.get("endTime"),
        })
    return execs


def _extract_goal(filtered: Dict[str, Any]) -> str:
    # Prefer top-level input.goal
    top_inp = filtered.get("input")
    if isinstance(top_inp, dict):
        g = top_inp.get("goal")
        if isinstance(g, str) and g.strip():
            return g.strip()
    # Fallback: find StandardAgent.solve observation input.goal
    for o in filtered.get("observations", []):
        if isinstance(o, dict) and o.get("name") == "agents.standard_agent.StandardAgent.solve":
            inp = o.get("input")
            if isinstance(inp, dict):
                g = inp.get("goal")
                if isinstance(g, str) and g.strip():
                    return g.strip()
    return ""


def _extract_tools(filtered: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tool metadata keyed by tool_id (best-effort).

    Attempts sources in order:
      - statusMessage containing JenticTool(... api_name=..., method=..., path=..., required=[...], parameters={...})
      - load observation input.tool string for tool_id
    """
    tools: Dict[str, Dict[str, Any]] = {}

    def _ensure(tid: str) -> Dict[str, Any]:
        if tid not in tools:
            tools[tid] = {"tool_id": tid}
        return tools[tid]

    obs_list = filtered.get("observations", [])
    for o in obs_list:
        if not isinstance(o, dict):
            continue
        # Parse statusMessage rich detail if present
        sm = o.get("statusMessage")
        if isinstance(sm, str) and "JenticTool(" in sm:
            # id
            m_id = re.search(r"JenticTool\(id=([^,\s]+)", sm)
            if m_id:
                tid = m_id.group(1)
                ref = _ensure(tid)
                m_api = re.search(r"api_name=([^,\s]+)", sm)
                if m_api:
                    ref["api_name"] = m_api.group(1)
                m_method = re.search(r"method=([^,\s]+)", sm)
                if m_method:
                    ref["method"] = m_method.group(1)
                m_path = re.search(r"path=([^,\s]+)", sm)
                if m_path:
                    ref["path"] = m_path.group(1)
                # required=[...] keys
                m_req = re.search(r"required=\[(.*?)\]", sm)
                if m_req:
                    keys = [k.strip().strip("'\"") for k in m_req.group(1).split(',') if k.strip()]
                    # nest under parameters.required
                    params = ref.setdefault("parameters", {})
                    params["required"] = keys
                # parameters={...} keys
                m_params = re.search(r"parameters=\{(.*?)\}\)?:", sm, re.DOTALL)
                if m_params:
                    # naive key extraction
                    keys = re.findall(r"'([^']+)'\s*:", m_params.group(1))
                    if keys:
                        params = ref.setdefault("parameters", {})
                        params["allowed"] = sorted(set(keys))

        # Parse load observation to capture tool_id if not seen
        if o.get("name") == "agents.tools.jentic.JenticClient.load":
            inp = o.get("input")
            tool_field = None
            if isinstance(inp, dict):
                tool_field = inp.get("tool")
            elif isinstance(inp, str):
                tool_field = inp
            if isinstance(tool_field, str):
                m = re.search(r"JenticTool\('([^']+)'", tool_field)
                if m:
                    tid = m.group(1)
                    _ensure(tid)

            # Also parse rich metadata from load.output when present
            out_field = o.get("output")
            if isinstance(out_field, str) and "JenticTool(" in out_field:
                # tool id
                m_id = re.search(r"JenticTool\(id=([^,\s]+)", out_field)
                if m_id:
                    tid = m_id.group(1)
                    ref = _ensure(tid)
                    m_api = re.search(r"api_name=([^,\s]+)", out_field)
                    if m_api:
                        ref["api_name"] = m_api.group(1)
                    m_method = re.search(r"method=([^,\s]+)", out_field)
                    if m_method:
                        ref["method"] = m_method.group(1)
                    m_path = re.search(r"path=([^,\s]+)", out_field)
                    if m_path:
                        ref["path"] = m_path.group(1)
                    m_req = re.search(r"required=\[(.*?)\]", out_field, re.DOTALL)
                    if m_req:
                        keys_req = [k.strip().strip("'\"") for k in m_req.group(1).split(',') if k.strip()]
                        if keys_req:
                            params = ref.setdefault("parameters", {})
                            params["required"] = keys_req
                    m_params = re.search(r"parameters=\{(.*?)\}\)", out_field, re.DOTALL)
                    if m_params:
                        keys_allowed = re.findall(r"'([^']+)'\s*:\s*", m_params.group(1))
                        if not keys_allowed:
                            keys_allowed = re.findall(r'"([^"]+)"\s*:\s*', m_params.group(1))
                        if keys_allowed:
                            params = ref.setdefault("parameters", {})
                            params["allowed"] = sorted(set(keys_allowed))

    return list(tools.values())


def _extract_params_generated(filtered: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract generated parameter keys per tool from _generate_params observations."""
    results: List[Dict[str, Any]] = []
    gen_names = {
        "agents.reasoner.rewoo.ReWOOReasoner._generate_params",
        "agents.reasoner.react.ReACTReasoner._generate_params",
    }
    for o in filtered.get("observations", []):
        if not isinstance(o, dict):
            continue
        if o.get("name") not in gen_names:
            continue
        # tool id from input.tool
        tool_id = None
        inp = o.get("input")
        if isinstance(inp, dict):
            tool_field = inp.get("tool")
            if isinstance(tool_field, str):
                m = re.search(r"JenticTool\('([^']+)'", tool_field)
                if m:
                    tool_id = m.group(1)
        # keys from output
        keys: List[str] = []
        out = o.get("output")
        if isinstance(out, dict):
            keys = list(out.keys())
        elif isinstance(out, str):
            # naive key extraction from dict-like text
            keys = re.findall(r"'([^']+)'\s*:\s*", out)
            if not keys:
                keys = re.findall(r'"([^"]+)"\s*:\s*', out)
        if tool_id and keys:
            results.append({"tool_id": tool_id, "keys": sorted(set(keys))})
    return results


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

    # Phase 1 structured extraction (plan + successful executions + tools + params + goal)
    plan_steps = _extract_plan_steps(filtered)
    executions = _extract_executions(filtered)
    tools = _extract_tools(filtered)
    params = _extract_params_generated(filtered)
    goal = _extract_goal(filtered)
    # Persist for debugging/iteration
    try:
        trace_id = str(filtered.get("id") or "")
        trace_dir = PROJECT_ROOT / "workflow_miner" / "logs" / (trace_id if trace_id else "")
        # fall back to unknown if top-level id missing
        if not trace_dir.exists():
            trace_dir = PROJECT_ROOT / "workflow_miner" / "logs"
        ctx = {"plan_steps": plan_steps, "executions": executions, "tools": tools, "paramsGenerated": params, "goal": goal}
        (trace_dir / "phase1_context.json").write_text(json.dumps(ctx, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    # Load prompt template from workflow_miner/arazzo.yaml (same folder as this script)
    prompt_path = PROJECT_ROOT / "workflow_miner" / "arazzo.yaml"
    data = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    template = data.get("compose", "")
    few_shot_examples = data.get("example", "")
    return template.format(
        goal_hint=top_goal_hint,
        observations_json=observations_json,
        goal_text=goal,
        tools_json=json.dumps(tools, ensure_ascii=False, indent=2),
        params_json=json.dumps(params, ensure_ascii=False, indent=2),
        executions_json=json.dumps(executions, ensure_ascii=False, indent=2),
        plan_steps_json=json.dumps(plan_steps, ensure_ascii=False, indent=2),
    ) + few_shot_examples


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

    # Call LLM to generate the Arazzo (JSON-only per prompt)
    llm_text = llm.prompt(prompt)

    # Sanitize: strip code fences if present and trim whitespace
    try:
        import re as _re
        fence = _re.search(r"```(?:json|yaml)?\s*([\s\S]*?)\s*```", llm_text)
        if fence:
            llm_text = fence.group(1)
        llm_text = llm_text.replace("```", "").strip()
    except Exception:
        llm_text = llm_text.strip()

    # Try to parse as JSON and convert to desired output format
    final_text = llm_text
    try:
        obj = json.loads(llm_text)
        if out_path_arg and (out_path_arg.endswith(".yaml") or out_path_arg.endswith(".yml")):
            final_text = yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)
        else:
            final_text = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        # Leave as-is if JSON parsing fails
        pass
    out_path = _resolve_output_path(trace_id, out_path_arg)
    out_path.write_text(final_text, encoding="utf-8")

    print(f"Saved Arazzo: {out_path}")


if __name__ == "__main__":
    main(sys.argv)


