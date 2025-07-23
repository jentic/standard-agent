"""
CLI-based inbox that reads goals from standard input.

Each line typed by the user becomes a goal.  Typing nothing, or any of
the quit aliases (“quit” / “exit” / “q”), closes the inbox.
"""
from __future__ import annotations

import sys
from typing import Optional, TextIO

from .base_inbox import BaseInbox


class CLIInbox(BaseInbox):
    """
    Interactive stdin inbox.

    Examples
    --------
    >>> inbox = CLIInbox(prompt="Enter a Goal please: ")
    >>> while (goal := inbox.get_next_goal()) is not None:
    ...     print("received:", goal)
    """

    def __init__(
        self,
        *,
        input_stream: Optional[TextIO] = None,
        prompt: str = "Enter goal: ",
    ) -> None:
        self.input_stream: TextIO = input_stream or sys.stdin
        self.prompt = prompt
        self._closed = False

    def get_next_goal(self) -> str | None:
        """Read the next line from the input stream; return None on EOF/quit."""
        if self._closed:
            return None

        try:
            # Show prompt only for interactive stdin
            if self.input_stream is sys.stdin:
                print(self.prompt, end="", flush=True)

            line = self.input_stream.readline()
            if not line:                       # EOF
                self._closed = True
                return None

            goal = line.strip()
            if goal.lower() in {"bye", "quit", "exit", "q"}:
                raise KeyboardInterrupt

            return goal

        except (EOFError, KeyboardInterrupt):
            self._closed = True
            raise

    def close(self) -> None:                   # override default pass‐through
        if self._closed:
            return
        self._closed = True
        if self.input_stream is not sys.stdin:
            self.input_stream.close()