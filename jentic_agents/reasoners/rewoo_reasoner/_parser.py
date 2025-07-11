"""Utilities for parsing LLM-generated markdown plans.

Currently supports simple indented bullet lists, e.g.::

    - Do X
        - Substep
    - Do Y

Leading bullet symbols (*, -, +) or enumerated lists (1., 2.) are removed and
indent level is determined from leading whitespace (2 spaces == 1 indent).
"""
from __future__ import annotations

import re
from collections import deque
from typing import Deque, List
from jentic_agents.utils.logger import get_logger


from ..models import Step

# Two spaces per indent level keeps markdown compatibility.
_INDENT_SIZE = 2
_BULLET_PATTERN = re.compile(r"^\s*(?:[-*+]\s|\d+\.\s)(.*)$")
# Capture (input: a, b) and (output: c) sections
_IO_DIRECTIVE_PATTERN = re.compile(r"\((input|output):\s*([^)]*)\)")
logger = get_logger(__name__)

def _line_indent(text: str) -> int:
    """Return indent level (0-based) from leading spaces."""
    spaces = len(text) - len(text.lstrip(" "))
    return spaces // _INDENT_SIZE


def _strip_bullet(text: str) -> str:
    """Remove leading bullet/number and extra whitespace."""
    match = _BULLET_PATTERN.match(text)
    return match.group(1).rstrip() if match else text.strip()


def parse_bullet_plan(markdown: str) -> Deque[Step]:
    """Parse an indented markdown bullet list into a queue of ``Step`` objects.

    The function intentionally ignores any non-list lines. It also **only**
    extracts structure — higher-level orchestration (nesting relationships,
    container steps, etc.) is left for the caller to interpret.
    """
    steps: Deque[Step] = deque()
    for raw_line in markdown.splitlines():
        # Skip empty lines
        if not raw_line.strip():
            continue
        # Only process lines that look like list items (start with bullet or number)
        if not _BULLET_PATTERN.match(raw_line):
            continue

        indent = _line_indent(raw_line)
        stripped = _strip_bullet(raw_line)

        # Extract (input: ...) and (output: ...) directives
        input_keys: List[str] = []
        output_key = None
        for io_match in _IO_DIRECTIVE_PATTERN.finditer(stripped):
            kind, payload = io_match.groups()
            if kind == "output":
                output_key = payload.strip()
            else:  # input
                input_keys = [k.strip() for k in payload.split(',') if k.strip()]
        # Remove directives from the visible text
        cleaned_text = _IO_DIRECTIVE_PATTERN.sub("", stripped).strip()

        steps.append(
            Step(
                text=cleaned_text,
                indent=indent,
                output_key=output_key,
                input_keys=input_keys,
            )
        )

    logger.info(f"Parsed steps")
    for step in steps:
        logger.info(f"Step => step.text: {step.text}, step.indent: {step.indent}, step.output_key: {step.output_key}, step.input_keys: {step.input_keys}")

    return steps
