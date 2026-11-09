"""Live progress while a plan is applied.

In a notebook, a single card is displayed and then updated in place as each step starts,
reports progress, and finishes — so the user sees a spinner and "Waiting for ingestion ·
40 scanned (120s)" instead of a cell that looks frozen for ten minutes.

In a terminal it prints one line per event instead.

The wizard (and tests) can redirect the HTML somewhere else with ``progress_target``.
"""

from __future__ import annotations

import contextlib
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Generator, List, Optional, Tuple

from .theme import card, esc

_target: Optional[Callable[[str], None]] = None


@contextlib.contextmanager
def progress_target(sink: Callable[[str], None]) -> Generator[None, None, None]:
    """Send progress HTML to ``sink(html)`` instead of a notebook output, while inside this block."""
    global _target
    previous, _target = _target, sink
    try:
        yield
    finally:
        _target = previous


def in_notebook() -> bool:
    """True when running inside Jupyter / SageMaker Studio (rather than a terminal or script)."""
    try:
        from IPython.core.getipython import get_ipython

        shell = get_ipython()
        return shell is not None and type(shell).__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


@dataclass
class _Row:
    title: str
    resource: str
    state: str = "pending"  # pending | running | done | skipped | failed | not_run
    message: str = ""
    started: float = 0.0
    seconds: float = 0.0


class Progress:
    """Tracks each step's state and redraws the progress view whenever something changes."""

    def __init__(self, title: str, rows: List[Tuple[str, str]], kb_name: str):
        self.title = title
        self.kb_name = kb_name
        self.rows = [_Row(t, r) for t, r in rows]
        self.started = time.time()
        self.finished = False
        self.failed = False
        self._last_draw = 0.0
        self._last_text_log = 0.0
        self._sink = _target
        self._handle: Any = None
        self._html_mode = self._sink is not None or in_notebook()
        if self._html_mode and self._sink is None:
            from IPython.display import HTML, display

            self._handle = display(HTML(self.render()), display_id=True)
        elif not self._html_mode:
            print(f"\n{title}", flush=True)

    def start(self, i: int) -> None:
        """Mark step ``i`` as running."""
        row = self.rows[i]
        row.state, row.started, row.message = "running", time.time(), ""
        self._text(f"  … {row.title}")
        self._draw()

    def log(self, i: int, message: str) -> None:
        """Show a progress message under the running step (throttled so we don't flood the notebook)."""
        self.rows[i].message = message
        now = time.time()
        if not self._html_mode and now - self._last_text_log > 15:
            self._last_text_log = now
            self._text(f"      {message}")
        if now - self._last_draw > 0.5:
            self._draw()

    def done(self, i: int, result: str = "") -> None:
        """Mark step ``i`` as finished successfully, with an optional short result."""
        row = self.rows[i]
        row.state, row.message, row.seconds = "done", result, time.time() - row.started
        self._text(f"  ✓ {row.title} ({row.seconds:.0f}s){'  ' + result if result else ''}")
        self._draw()

    def skip(self, i: int, note: str) -> None:
        """Mark step ``i`` as needing no work (e.g. it already exists)."""
        row = self.rows[i]
        row.state, row.message = "skipped", note
        self._text(f"  – {row.title}: {note}")
        self._draw()

    def fail(self, i: int, message: str) -> None:
        """Mark step ``i`` as failed, and every later step as not run."""
        row = self.rows[i]
        row.state, row.message, row.seconds = "failed", message, time.time() - (row.started or time.time())
        for later in self.rows[i + 1 :]:
            later.state = "not_run"
        self.failed = self.finished = True
        self._text(f"  ✗ {row.title}\n\n{message}\n")
        self._draw()

    def close(self) -> None:
        """Final redraw once everything succeeded."""
        self.finished = True
        self._text(f"Done in {time.time() - self.started:.0f}s")
        self._draw()

    def _text(self, line: str) -> None:
        if not self._html_mode:
            print(line, file=sys.stdout, flush=True)

    def _draw(self) -> None:
        if not self._html_mode:
            return
        self._last_draw = time.time()
        html = self.render()
        if self._sink is not None:
            self._sink(html)
        elif self._handle is not None:
            from IPython.display import HTML

            self._handle.update(HTML(html))

    def render(self) -> str:
        """The progress card as HTML."""
        elapsed = time.time() - self.started
        if self.failed:
            status = '<span class="badge c-red">failed</span>'
        elif self.finished:
            status = '<span class="badge c-green">done</span>'
        else:
            status = '<span class="badge c-blue">working</span>'
        rows_html = "".join(self._render_row(row) for row in self.rows)
        footer = ""
        if self.failed:
            footer = (
                '<div class="section next">Fix the problem above, then run <code>kb.apply()</code> again. '
                "Steps that already finished are detected and skipped.</div>"
            )
        return card(
            f'<div class="head"><div class="eyebrow">kbforge · {esc(self.kb_name)}</div>'
            f"<h2>{esc(self.title)} {status}</h2>"
            f'<div class="muted">{elapsed:.0f}s elapsed</div></div>'
            f'<div class="section"><ol class="steps prog">{rows_html}</ol></div>{footer}'
        )

    def _render_row(self, row: _Row) -> str:
        icons = {
            "pending": '<span class="num" style="width:18px;height:18px;font-size:10px">·</span>',
            "running": '<span class="spin"></span>',
            "done": '<span class="icon" style="--c:var(--kbf-green)">✓</span>',
            "skipped": '<span class="icon" style="--c:var(--kbf-grey)">–</span>',
            "failed": '<span class="icon" style="--c:var(--kbf-red)">✕</span>',
            "not_run": '<span class="icon" style="--c:var(--kbf-border)">·</span>',
        }
        detail = ""
        if row.state == "running" and row.message:
            detail = f'<div class="log mono">{esc(row.message)}</div>'
        elif row.state in ("done", "skipped") and row.message:
            detail = f'<div class="note muted mono">{esc(row.message)}</div>'
        elif row.state == "failed":
            detail = f'<div class="warn c-red">{esc(row.message)}</div>'
        elif row.state == "not_run":
            detail = '<div class="note muted">not run</div>'
        if row.state == "running":
            timing = f"{time.time() - row.started:.0f}s"
        elif row.state in ("done", "failed"):
            timing = f"{row.seconds:.0f}s"
        else:
            timing = ""
        dim = ' style="opacity:.55"' if row.state in ("pending", "not_run") else ""
        return (
            f"<li{dim}>{icons[row.state]}<div><div class='row'><span class='title'>{esc(row.title)}</span>"
            f"<span class='res mono'>{esc(row.resource)}</span></div>{detail}</div>"
            f"<span class='time'>{timing}</span></li>"
        )
