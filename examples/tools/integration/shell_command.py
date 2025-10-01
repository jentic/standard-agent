"""Command execution tool example with a configurable allow-list."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Protocol, Sequence

import subprocess

from agents.tools.base import JustInTimeToolingBase, ToolBase


class _SubprocessRunner(Protocol):
    """Callable protocol mirroring `subprocess.run`."""

    def __call__(
        self,
        args: Sequence[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        ...


class ShellCommandTool(ToolBase):
    """Tool metadata for executing shell commands on an allow-list."""

    TOOL_ID = "shell.run"

    def __init__(self) -> None:
        super().__init__(id=self.TOOL_ID)
        self.name = "Shell Command"
        self.description = "Execute a pre-approved shell command and capture the output."
        self._parameters: Dict[str, Any] = {
            "command": {
                "type": "string",
                "description": "Command name that must be present in the allow-list.",
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional command arguments.",
            },
            "timeout": {
                "type": "number",
                "description": "Optional timeout in seconds (defaults to 10).",
            },
        }

    def get_summary(self) -> str:
        return f"{self.id}: {self.name} - {self.description}"

    def get_details(self) -> str:
        return (
            "Runs commands using subprocess with capture_output=True. Only commands present in the allow-list "
            "configured on the provider are permitted."
        )

    def get_parameters(self) -> Dict[str, Any]:
        return self._parameters


@dataclass
class ShellCommandTools(JustInTimeToolingBase):
    """Provider that safely exposes shell commands to the agent."""

    allow_list: Iterable[str] = field(default_factory=lambda: ("echo",))
    runner: _SubprocessRunner = subprocess.run

    def __post_init__(self) -> None:
        self._tool = ShellCommandTool()
        self._allowed = {cmd.strip() for cmd in self.allow_list}

    def search(self, query: str, *, top_k: int = 10) -> List[ToolBase]:
        if any(word in query.lower() for word in ("run", "shell", "command")):
            return [self._tool]
        return []

    def load(self, tool: ToolBase) -> ToolBase:
        if tool.id != self._tool.id:
            raise ValueError(f"Unknown tool requested: {tool.id}")
        return self._tool

    def execute(self, tool: ToolBase, parameters: Dict[str, Any]) -> Dict[str, Any]:
        if tool.id != self._tool.id:
            raise ValueError(f"Unexpected tool invocation: {tool.id}")

        command = str(parameters.get("command", "")).strip()
        if command not in self._allowed:
            raise ValueError(f"Command '{command}' is not permitted")

        args = parameters.get("args", []) or []
        if not isinstance(args, list):
            raise ValueError("args must be an array of strings")
        str_args = [str(item) for item in args]

        timeout = parameters.get("timeout")
        timeout_f = float(timeout) if timeout is not None else 10.0

        completed = self.runner(
            [command, *str_args],
            capture_output=True,
            text=True,
            timeout=timeout_f,
        )

        return {
            "command": command,
            "args": str_args,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }

    @staticmethod
    def format_command(command: str, args: Iterable[str]) -> str:
        """Utility helper to show the command in shell form for documentation snippets."""
        parts = [command, *args]
        return " ".join(shlex.quote(part) for part in parts)
