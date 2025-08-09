## Implicit (ReACT-style) Reasoner

This module provides a composable, readable implementation of an implicit (ReACT-style) agent. It is designed for OSS users to plug in their own components while keeping the orchestration loop simple and clear.

### What “Implicit” means

- No upfront global plan. The agent iterates through short steps: THOUGHT → (optional) ACTION → OBSERVATION, until it produces a FINAL answer.
- Components are pure and side-effect free; the main loop owns state and logging.


### Components (swap any of these)

- Think
  - Input: ImplicitState
  - Output: ReasonNode(kind, text)
  - Guidance:
    - THOUGHT.text: brief reasoning that advances toward an actionable step
    - ACTION.text: a single, clear instruction in plain language
    - FINAL.text: concise, user-facing answer

- Act
  - Input: ImplicitState with latest node kind == ACTION
  - Output: (tool_id: str, params: dict, observation: Any)
  - May raise ToolSelectionError (no suitable tool). The loop records it in the transcript for the next THOUGHT to react to.


- Policy (DecidePolicy)
  - Input: ImplicitState
  - Output: Decision (REASON | TOOL | HALT)

- Summarizer
  - Input: ImplicitState
  - Output: final answer (str)
  - Uses FINAL if present; otherwise synthesizes from observations.

### Core data contracts

- ReasonNode
  - kind: one of THOUGHT | ACTION | FINAL
  - text: content of the node
- Decision
  - REASON | TOOL | HALT (policy output)
- Turn
  - thought: ReasonNode | None
  - action: {tool_id, params} | None
  - observation: Any | None
- ImplicitState
  - goal: str
  - turns: list[Turn]
  - is_complete: bool
  - final_answer: str | None
  - helper: get_reasoning_transcript() → str (rendered transcript used in prompts)

### Minimal wiring example

```python
from collections.abc import MutableMapping
from agents.llm.base_llm import BaseLLM
from agents.tools.base import JustInTimeToolingBase

from agents.reasoner.implicit.reasoner import ImplicitReasoner
from agents.reasoner.implicit.policy.react import ReACTPolicy
from agents.reasoner.implicit.think.react import ReACTThink
from agents.reasoner.implicit.act.react import ReACTAct
from agents.reasoner.implicit.summarizer.summarize import DefaultImplicitSummarizer


class ReACTReasoner(ImplicitReasoner):
    """Pre-wired ImplicitReasoner configured for ReACT-style operation."""

    def __init__(
        self,
        llm: BaseLLM,
        tools: JustInTimeToolingBase,
        memory: MutableMapping,
        max_turns: int = 20,
    ) -> None:
        super().__init__(
            llm=llm,
            tools=tools,
            memory=memory,
            decide=ReACTPolicy(),
            think=ReACTThink(llm=llm),
            act=ReACTAct(llm=llm, tools=tools),
            summarize=DefaultImplicitSummarizer(llm=llm),
            max_turns=max_turns,
        )
```



