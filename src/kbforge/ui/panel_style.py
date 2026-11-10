"""The look of ``kb.sync_panel()``: its header, its list of syncs, and a little extra CSS.

The panel is built from the same pieces as the wizard (buttons, status line, option cards),
so it reuses the wizard's stylesheet, scoped to the ``kbf-wiz`` class, and adds only what's
new here. Like ``wizard_style``, everything is a plain function returning HTML so it can be
tested without a browser.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from .theme import CSS, esc
from .wizard_style import WIZARD_CSS, icon

if TYPE_CHECKING:
    from ..ingestion import IngestionJob

PANEL_CSS = """
<style>
.kbf-wiz .hero .state { flex: 0 0 auto; display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 600; padding: 5px 10px;
  border-radius: 999px; color: var(--c); background: color-mix(in srgb, var(--c) 10%, var(--kbf-card));
  border: 1px solid color-mix(in srgb, var(--c) 30%, transparent); }
.kbf-wiz .hero .state svg { width: 14px; height: 14px; }
.kbf-wiz .hero .state .spin { width: 12px; height: 12px; margin: 0; border-width: 2px; border-top-color: var(--c); }
.kbf-wiz .hero .chips { margin-top: 10px; }
.kbf-wiz .kbf-jobs { padding: 16px 26px 20px; width: 100%; }
.kbf-wiz .kbf-jobs .jobs-h { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; margin-bottom: 10px; }
.kbf-wiz .kbf-jobs .jobs-h h3 { margin: 0; }
.kbf-wiz .kbf-jobs .jobs-h span { font-size: 12px; color: var(--kbf-muted); }
.kbf-wiz .kbf-jobs .empty { color: var(--kbf-muted); padding: 6px 0; }
.kbf-wiz .kbf-jobs .kbf-opt .opt .ico { width: 28px; height: 28px; }
.kbf-wiz .kbf-jobs .kbf-opt.bad .opt .ico { color: var(--kbf-red); }
@media (max-width: 860px) { .kbf-wiz .kbf-jobs .kbf-opts { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; } }
</style>
"""


def state_pill(latest: Optional[IngestionJob], error: str = "") -> str:
    """A pill summing up where the knowledge base stands: up to date, syncing, failed, never synced."""
    from ..ingestion import time_ago

    if error:
        return f'<span class="state" style="--c:var(--kbf-grey)">{icon("info", 2)}Not available</span>'
    if latest is None:
        return f'<span class="state" style="--c:var(--kbf-grey)">{icon("dot", 3)}Never synced</span>'
    if latest.running:
        return '<span class="state" style="--c:var(--kbf-blue)"><span class="spin"></span>Sync running</span>'
    when = time_ago(latest.started_at)
    if latest.status == "FAILED":
        return (
            f'<span class="state" style="--c:var(--kbf-red)">{icon("cross", 2.4)}Last sync failed · {esc(when)}</span>'
        )
    if latest.failed:
        return (
            f'<span class="state" style="--c:var(--kbf-amber)">{icon("alert", 2)}'
            f"{latest.failed:,} file(s) failed · {esc(when)}</span>"
        )
    if latest.status == "STOPPED":
        return f'<span class="state" style="--c:var(--kbf-grey)">{icon("dot", 3)}Last sync stopped · {esc(when)}</span>'
    return f'<span class="state" style="--c:var(--kbf-green)">{icon("check", 2.4)}Synced {esc(when)}</span>'


def panel_header(name: str, region: str, source: str, pill: str) -> str:
    """The panel's header: brand mark, what it's for, where the documents are, and the state pill.

    It also carries the stylesheets, so it must stay the first thing inside the panel.
    """
    chips = "".join(
        f'<span class="chip">{esc(k)} <b>{esc(v)}</b></span>' for k, v in (("region", region), ("source", source)) if v
    )
    return (
        f"{CSS}{WIZARD_CSS}{PANEL_CSS}"
        '<div class="hero"><div class="hero-top">'
        f'<div class="mark">{icon("logo", 2)}</div>'
        '<div class="hero-text"><div class="eyebrow">kbforge · syncs</div>'
        f"<h2>Keep {esc(name)} up to date</h2>"
        '<div class="lede">A sync re-reads your documents after they change. Bedrock only processes new and '
        "changed files, and removes deleted ones. <b>Check</b> looks for problems first without changing "
        "anything.</div>"
        f'<div class="chips">{chips}</div></div>{pill}</div></div>'
    )


def jobs_heading(count: int, more: bool) -> str:
    """The title above the list of syncs."""
    if not count:
        return '<div class="jobs-h"><h3>Recent syncs</h3></div><div class="empty">No syncs yet.</div>'
    hint = "click one for details" + (" · older ones aren't shown" if more else "")
    return f'<div class="jobs-h"><h3>Recent syncs</h3><span>{hint}</span></div>'


def job_choice(job: IngestionJob) -> Tuple[str, str, str, Tuple[str, str], str]:
    """How one sync appears in the panel's list: (title, body, icon, tag, meta) for an option card."""
    from ..ingestion import clock, human_duration, time_ago

    label, colour = job.outcome
    if job.running:
        glyph = "sparkle"
    elif job.status == "FAILED":
        glyph = "cross"
    elif job.failed:
        glyph = "alert"
    elif job.status == "COMPLETE":
        glyph = "check"
    else:
        glyph = "dot"
    took = f"running {human_duration(job.seconds)}" if job.running else f"took {human_duration(job.seconds)}"
    return (
        time_ago(job.started_at) or job.job_id,
        job.stats_text,
        glyph,
        (label, colour),
        f"{clock(job.started_at)} · {took}",
    )


def picker_header(region: str) -> str:
    """The header of ``kbforge.sync_panel()`` before a knowledge base is picked. Carries the stylesheets."""
    chips = f'<div class="chips"><span class="chip">region <b>{esc(region)}</b></span></div>' if region else ""
    return (
        f"{CSS}{WIZARD_CSS}{PANEL_CSS}"
        '<div class="hero"><div class="hero-top">'
        f'<div class="mark">{icon("logo", 2)}</div>'
        '<div class="hero-text"><div class="eyebrow">kbforge · syncs</div>'
        "<h2>Which knowledge base?</h2>"
        '<div class="lede">Pick one to see its recent syncs, check that a sync would work, and sync it. '
        "Opening one only reads from AWS.</div>"
        f"{chips}</div></div></div>"
    )


def picker_heading(count: int) -> str:
    """The title above the knowledge bases to pick from."""
    return f'<div class="jobs-h"><h3>Built by kbforge</h3><span>{count} · click one</span></div>'


def message_card(title: str, text: str, colour: str = "c-red") -> str:
    """A small card with one message, e.g. an error or "nothing to show yet"."""
    return (
        f'{CSS}<div class="kbf"><div class="card"><div class="head"><div class="eyebrow">kbforge · syncs</div>'
        f'<h2>{esc(title)}</h2><div class="warn {colour}">{esc(text)}</div></div></div></div>'
    )
