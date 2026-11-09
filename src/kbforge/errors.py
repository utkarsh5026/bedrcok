"""The error kbforge raises for problems the user can fix, and helpers for wording them.

Everything here is about how a mistake *reads*. A colleague who mistypes an option or whose
AWS session has expired should see one clear sentence that says what to do next, not a
40-line Python stack trace.
"""

from __future__ import annotations

import difflib
from typing import Iterable, List


class KBForgeError(RuntimeError):
    """A problem the user can fix, with a message that says how.

    In Jupyter/IPython this shows as just the message instead of a full stack trace: the
    trace points into kbforge's internals, which only gets in the way. Tracebacks of the
    underlying cause are still available with ``%debug`` or ``exc.__cause__``.
    """

    def _render_traceback_(self) -> List[str]:
        """What IPython prints for this error, in place of the usual traceback."""
        red, bold, reset = "\x1b[31m", "\x1b[1m", "\x1b[0m"
        lines = str(self).splitlines() or [""]
        return [f"{red}{bold}kbforge:{reset} {lines[0]}", *lines[1:]]


def did_you_mean(word: object, options: Iterable[str]) -> str:
    """A " Did you mean 'x'?" hint for a likely typo, or "" if nothing is close.

    Meant to be appended to an error message that already lists the valid choices.
    """
    if not isinstance(word, str):
        return ""
    choices = list(options)
    # Compare case-insensitively ("Titan-V2" is still a typo of "titan-v2") but suggest the real spelling.
    lowered = {choice.lower(): choice for choice in choices}
    match = difflib.get_close_matches(word.lower(), list(lowered), n=1, cutoff=0.6)
    return f" Did you mean {lowered[match[0]]!r}?" if match else ""
