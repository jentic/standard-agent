import json
import re
from copy import deepcopy
from typing import Any, Dict, List

import jentic_agents.reasoners.rewoo_reasoner._prompts as prompts
from jentic_agents.memory.base_memory import BaseMemory
from jentic_agents.reasoners.models import ReasonerState, Step
from jentic_agents.tools.models import Tool
from jentic_agents.reasoners.rewoo_reasoner.exceptions import (
    MissingInputError,
    ReasoningStepError,
)
from jentic_agents.tools.exceptions import ToolExecutionError
from jentic_agents.reasoners.rewoo_reasoner._parser import parse_bullet_plan
from jentic_agents.reasoners.rewoo_reasoner_contract import BaseReWOOReasoner
from jentic_agents.tools.interface import ToolInterface
from jentic_agents.utils.llm import BaseLLM


class ReWOOReasoner(BaseReWOOReasoner):
    """Reasoner implementing ReWOO + Reflection on top of Jentic tools."""

    def __init__(
        self,
        *,
        tool: ToolInterface,
        memory: BaseMemory,
        llm: BaseLLM,
    ) -> None:
        super().__init__(tool=tool, memory=memory, llm=llm)
        self._tool_cache: Dict[str, Tool] = {}

    def run(self, goal: str, max_iterations: int = 20):  # noqa: D401
        return super().run(goal, max_iterations)

    def _generate_plan(self, state: ReasonerState) -> None:
        """Generate initial plan from goal using the LLM."""
        prompt = prompts.PLAN_GENERATION_PROMPT.replace("{goal}", state.goal)
        plan_md = self._call_llm(prompt)
        self._logger.info(f"phase=PLAN_GENERATED plan={plan_md}")
        state.plan = parse_bullet_plan(plan_md)

    def _execute_step(self, step: Step, state: ReasonerState) -> Dict[str, Any]:  # noqa: D401
        """Execute a single plan step with retry bookkeeping."""
        step.status = "running"
        try:
            inputs = self._fetch_inputs(step)
        except MissingInputError as exc:
            self._reflect_on_failure(exc, step, state)
            return None

        if step.step_type == Step.StepType.REASONING:
            result = self._execute_reasoning_step(step, inputs)
            step.status = "done"
            step.result = result
            self._store_step_output(step, state)
            return None

        tool_id = self._select_tool(step)
        params = self._generate_params(step, tool_id, inputs)
        try:
            result = self.tool.execute(tool_id, params)
            self._logger.info("phase=EXECUTE_OK run_id=%s tool_id=%s", getattr(self, "_run_id", "NA"), tool_id)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "phase=EXECUTE_FAIL run_id=%s tool_id=%s error=%s",
                getattr(self, "_run_id", "NA"),
                tool_id,
                exc,
            )
            self._reflect_on_failure(ToolExecutionError(str(exc)), step, state, failed_tool_id=tool_id)
            return None

        step.status = "done"
        step.result = result["result"].output
        # Persist in-memory for downstream steps
        self._store_step_output(step, state)
        return {"tool_id": tool_id, "params": params, "result": result}

    def _reflect_on_failure(
        self,
        error: Exception,
        step: Step,
        state: ReasonerState,
        failed_tool_id: str = None,
    ) -> None:  # noqa: D401
        """Invoke reflection logic and possibly modify the plan."""
        step.status = "failed"
        step.error = str(error)

        if step.retry_count >= 2:
            state.history.append(f"Giving up on step after retries: {step.text}")
            return

        tool_schema = {}
        failed_tool_name = failed_tool_id or "unknown"

        # If we know which tool failed, get its schema directly.
        if failed_tool_id and failed_tool_id != "none":
            tool_execution_info = self._get_tool(failed_tool_id)
            if tool_execution_info:
                tool_schema = tool_execution_info.parameters


        error_type = error.__class__.__name__
        prompt = prompts.BASE_REFLECTION_PROMPT.format(
            goal=state.goal,
            step=step.text,
            failed_tool_name=failed_tool_name,
            error_type=error_type,
            error_message=str(error),
            tool_schema=json.dumps(tool_schema),
        )

        if isinstance(error, ToolExecutionError):
            try:
                alternative_tools = self.tool.search(step.text, top_k=15)
                if alternative_tools:
                    # Create a minimal representation of the tools for the prompt
                    tools_for_prompt = [
                        {"id": t.id, "name": t.name, "api_name": t.api_name, "description": t.description}
                        for t in alternative_tools
                    ]
                    alternative_tools_json = json.dumps(tools_for_prompt, indent=2)
                    prompt += prompts.ALTERNATIVE_TOOLS_SECTION.format(
                        alternative_tools=alternative_tools_json
                    )
            except Exception as exc:
                self._logger.info(
                    "Could not search for alternative tools during reflection: %s", exc
                )

        raw = self._call_llm(prompt).strip()
        decision = self._parse_json_or_retry(raw, prompt)

        action = decision.get("action")
        state.history.append(f"Reflection decision: {decision}")
        if action == "give_up":
            return

        new_step = deepcopy(step)
        new_step.retry_count += 1
        new_step.status = "pending"
        # Handle actions
        if action == "rephrase_step" and "step" in decision:
            new_step.text = str(decision["step"])
        elif action == "change_tool" and "tool_id" in decision:
            # store chosen tool_id in memory for later stages if needed
            self._memory.store(f"forced_tool:{new_step.text}", decision["tool_id"])
        elif action == "retry_params" and "params" in decision:
            # stash params so _generate_params can skip LLM call next time
            self._memory.store(f"forced_params:{new_step.text}", decision["params"])
        # push the modified step to the front of the deque for immediate retry
        state.plan.appendleft(new_step)

    def _synthesize_final_answer(self, state: ReasonerState) -> str:  # noqa: D401
        """Combine successful step results into a final answer."""
        prompt = prompts.FINAL_ANSWER_SYNTHESIS_PROMPT.format(
            goal=state.goal,
            history="\n".join(state.history),
        )
        state.is_complete = True
        return self._call_llm(prompt)

    def _classify_step(self, step: Step, state: ReasonerState) -> Step.StepType:  # noqa: D401
        """Heuristic LLM classifier deciding TOOL vs REASONING."""
        mem_keys = getattr(self._memory, "keys", lambda: [])()
        keys_list = ", ".join(mem_keys)
        prompt = prompts.STEP_CLASSIFICATION_PROMPT.format(step_text=step.text, keys_list=keys_list)
        reply = self._call_llm(prompt).lower()
        print("Step Classified as :", reply)
        if "reason" in reply:
            return Step.StepType.REASONING
        return Step.StepType.TOOL

    def _execute_reasoning_step(self, step: Step, inputs: Dict[str, Any]) -> Any:  # noqa: D401
        """Execute a reasoning-only step via the LLM and return its output."""
        try:
            mem_snippet = json.dumps(inputs, ensure_ascii=False)
            prompt = prompts.REASONING_STEP_PROMPT.format(step_text=step.text, mem_snippet=mem_snippet)
            reply = self._call_llm(prompt).strip()
            _JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```")
            m = _JSON_FENCE_RE.search(reply)
            if m:
                reply = m.group(1).strip()
            return reply
        except Exception as exc:
            raise ReasoningStepError(str(exc)) from exc

    def _fetch_inputs(self, step: Step) -> Dict[str, Any]:
        """Retrieve all required inputs from memory or raise ``ge``."""
        inputs: Dict[str, Any] = {}
        for key in step.input_keys:
            try:
                inputs[key] = self._memory.retrieve(key)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                self._logger.warning("Missing required input key: %s", key)
                raise MissingInputError(key)
        return inputs

    def _store_step_output(self, step: Step, state: ReasonerState) -> None:
        """Persist a successful step's result under its `output_key`.

        Respect strict typing: only store if both key and result exist.
        The value is stored *as is*; if callers require serialisable data
        they must ensure the tool returns JSON-serialisable results.
        """
        if step.output_key and step.result is not None:
            # Unwrap OperationResult-like objects to their payload for JSON safety
            value_to_store = step.result["result"].output if hasattr(step.result, "result") else step.result
            try:
                self._memory.store(step.output_key, value_to_store)
                snippet = str(value_to_store).replace("\n", " ")
                state.history.append(f"stored {step.output_key}: {snippet}")
                self._logger.info(
                    "phase=MEM_STORE run_id=%s key=%s",
                    getattr(self, "_run_id", "NA"),
                    step.output_key,
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("Could not store result for key '%s': %s", step.output_key, exc)

    def _call_llm(self, prompt: str, **kwargs) -> str:
        """Send a single-turn user prompt to the LLM and return assistant content."""
        messages = [
            {"role": "user", "content": prompt},
        ]
        return self._llm.chat(messages, **kwargs).strip()


    def _select_tool(self, step: Step) -> str:
        """Search for tools and ask the LLM to pick the best one for *step*."""
        # First, check if reflection has forced a tool choice
        memory_key = f"forced_tool:{step.text}"

        forced_tool_id = self._memory.retrieve(memory_key)
        if forced_tool_id:
            self._logger.info("Using forced tool from reflection: %s", forced_tool_id)
            self._memory.delete(memory_key)  # Ensure this is a one-time override
            return forced_tool_id

        tools = self.tool.search(step.text, top_k=20)
        self._logger.info(
            "phase=SELECT_SEARCH run_id=%s step_text=%s hits=%s",
            getattr(self, "_run_id", "NA"),
            step.text,
            [f"{t.id}:{t.name}" for t in tools],
        )

        tools_json = json.dumps(
            [
                {
                    "id": t.id,
                    "name": t.name,
                    "description": t.description,
                    "api_name": t.api_name,
                }
                for t in tools
            ],
            ensure_ascii=False,
        )

        prompt = prompts.TOOL_SELECTION_PROMPT.format(step=step.text, tools_json=tools_json)
        reply = self._call_llm(prompt).strip()

        if self._is_valid_tool_reply(reply, tools):
            return reply

        # Simple retry
        retry_prompt = (
            f"Your reply '{reply}' was not in the list of valid tool IDs. "
            f"Please select from this list or reply 'none'.\n"
            f"List: {[t.id for t in tools]}"
        )
        reply = self._call_llm(retry_prompt).strip()
        if self._is_valid_tool_reply(reply, tools):
            return reply

        raise ValueError(f"Could not obtain valid tool id for step '{step.text}'. Last reply: {reply}")


    def _get_tool(self, tool_id: str) -> Tool:
        """Load and cache full tool execution info via the tool interface."""
        if tool_id in self._tool_cache:
            return self._tool_cache[tool_id]

        try:
            tool_execution_info = self.tool.load(tool_id)
            self._tool_cache[tool_id] = tool_execution_info
            return tool_execution_info
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Could not load tool execution info Tool: %s Error: %s", tool_id, exc)
            return None

    def _generate_params(
        self,
        step: Step,
        tool_id: str,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Use the LLM to propose parameters for *tool_id*."""
        tool_execution_info = self._get_tool(tool_id)

        forced_key = f"forced_params:{step.text}"
        forced = self._memory.retrieve(forced_key) if hasattr(self, "_memory") else None
        if forced:
            return forced

        tool_params = tool_execution_info.parameters or {}
        allowed_keys = ",".join(tool_params.keys())
        prompt = prompts.PARAMETER_GENERATION_PROMPT.format(
            step=step.text,
            tool_schema=json.dumps(tool_params, ensure_ascii=False),
            step_inputs=json.dumps(inputs, ensure_ascii=False),
            allowed_keys=allowed_keys,
        )
        raw = self._call_llm(prompt).strip()
        params = self._parse_json_or_retry(raw, prompt)

        # Keep only parameters that the tool schema recognises to avoid 400s.
        params = {k: v for k, v in params.items() if k in tool_params}
        return params

    def _parse_json_or_retry(self, raw: str, original_prompt: str) -> Dict[str, Any]:
        """Best-effort JSON parse with a single retry on failure."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            self._logger.warning("phase=JSON_PARSE_FAIL raw='%s'", raw)
            # Ask the LLM to fix the JSON, this is a single-shot correction
            prompt = prompts.JSON_CORRECTION_PROMPT.format(bad_json=raw, original_prompt=original_prompt)
            raw = self._call_llm(prompt).strip()
            return json.loads(raw)

    @staticmethod
    def _is_valid_tool_reply(reply: str, tools: List[Tool]) -> bool:
        """Return *True* iff *reply* is a valid tool id from *tools* or 'none'."""
        if reply == "none":
            return True
        return any(t.id == reply for t in tools)