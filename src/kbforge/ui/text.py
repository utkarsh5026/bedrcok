"""Plain-text versions of the cards, for the CLI and for ``print(plan)``.

Colours are added only when writing to a real terminal, so piping to a file stays clean.
"""

from __future__ import annotations

import os
import sys
import textwrap
from typing import TYPE_CHECKING, List

from ..checks import FAIL, OK, WARN
from ..spec import VECTOR_STORES, KBSpec
from ..steps import Action, describe_stats
from .html import source_location

if TYPE_CHECKING:
    from ..deployed import KnowledgeBases, Status
    from ..knowledge_base import Answers
    from ..parsers import ParsingModels
    from ..plan import ApplyResult, Plan

_COLOURS = {"green": "32", "amber": "33", "red": "31", "blue": "34", "grey": "90", "bold": "1"}

ACTION_TEXT = {
    Action.CREATE: ("+ create", "green"),
    Action.UPDATE: ("~ update", "amber"),
    Action.EXISTS: ("= exists", "grey"),
    Action.USE: ("= yours ", "grey"),
    Action.RUN: ("> run   ", "blue"),
    Action.SKIP: ("- skip  ", "grey"),
    Action.CONFLICT: ("! BLOCKED", "red"),
    Action.DELETE: ("- delete", "red"),
    Action.ABSENT: ("  gone  ", "grey"),
    Action.KEEP: ("= keep  ", "amber"),
}

CHECK_TEXT = {OK: ("✓", "green"), WARN: ("!", "amber"), FAIL: ("✗", "red")}


def _use_colour() -> bool:
    """Colour on a terminal; ``NO_COLOR`` turns it off and ``FORCE_COLOR`` forces it on."""
    if "NO_COLOR" in os.environ:
        return False
    return "FORCE_COLOR" in os.environ or sys.stdout.isatty()


def paint(text: str, colour: str) -> str:
    """Wrap text in an ANSI colour code (no-op when not on a terminal)."""
    if not _use_colour():
        return text
    return f"\033[{_COLOURS[colour]}m{text}\033[0m"


def _wrap(text: str, indent: int) -> str:
    pad = " " * indent
    return textwrap.fill(
        text, width=100, initial_indent=pad, subsequent_indent=pad, break_on_hyphens=False, break_long_words=False
    )


def _parsing_step(spec: KBSpec) -> str:
    """The parser in the plan's one-line data flow, or nothing for the default parser."""
    return "" if spec.parsing.strategy == "default" else f"{spec.parsing.describe()} → "


def render_plan(plan: Plan) -> str:
    """The plan as text: header, checks, then one line per step."""
    spec = plan.spec
    lines: List[str] = [
        "",
        paint(f"{'Destroy plan' if plan.mode == 'destroy' else 'Plan'} for knowledge base '{spec.name}'", "bold"),
        paint(f"  {plan.aws.region} · {VECTOR_STORES[spec.vector_store]} · {spec.model.label} ({spec.dims}d)", "grey"),
        paint(f"  {spec.source.describe()} → {_parsing_step(spec)}{spec.chunking.describe()}", "grey"),
        "",
    ]
    if plan.checks:
        lines.append(paint("Preflight checks", "bold"))
        for check in plan.checks:
            glyph, colour = CHECK_TEXT[check.status]
            lines.append(f"  {paint(glyph, colour)} {check.title}")
            lines.append(_wrap(check.detail, 4))
        lines.append("")
    if plan.items:
        lines.append(paint("Steps", "bold"))
        for item in plan.items:
            label, colour = ACTION_TEXT[item.inspection.action]
            lines.append(f"  {paint(label, colour)}  {item.step.title}  {paint(item.resource, 'grey')}")
            if item.inspection.note:
                lines.append(paint(_wrap(item.inspection.note, 13), "grey"))
            if item.inspection.warning:
                lines.append(paint(_wrap(item.inspection.warning, 13), "amber"))
        lines.append("")
    if plan.blockers:
        lines.append(paint("Blocked — fix these and plan again:", "red"))
        lines += [_wrap(f"- {b}", 2) for b in plan.blockers]
    elif not plan.has_work:
        lines.append(paint("Nothing to do — everything already matches.", "green"))
    else:
        lines.append(paint("Ready. Nothing has been changed yet.", "green"))
    return "\n".join(lines)


def render_apply_result(result: ApplyResult) -> str:
    """One short summary after apply finishes or fails."""
    if not result.ok:
        where = f" at '{result.failed_step}'" if result.failed_step else ""
        return paint(f"\nStopped{where}.", "red") + f"\n{result.error}\n\nFix it and run apply again."
    if result.mode == "destroy":
        return paint(f"\nDeleted '{result.spec.name}' in {result.seconds:.0f}s.", "green")
    lines = [
        paint(f"\n'{result.spec.name}' is ready ({result.seconds:.0f}s).", "green"),
        f"  Knowledge base ID: {result.out.get('kb_id', '')}",
        f"  Data source ID:    {result.out.get('ds_id', '')}",
    ]
    if result.out.get("last_job"):
        lines.append(f"  First sync:        {describe_stats(result.out['last_job'].get('statistics', {}))}")
    return "\n".join(lines)


def render_answers(answers: Answers) -> str:
    """The written answer with [n] citation markers (if asked for), then the chunks, best first."""
    lines = [paint(f"\n{answers.question}", "bold")]
    if answers.answer is not None:
        text = "".join(piece + "".join(f"[{n}]" for n in numbers) for piece, numbers in answers.answer_with_markers())
        lines.extend(_wrap(paragraph, 2) for paragraph in text.splitlines() if paragraph.strip())
        lines.append(paint(f"\n  Answered by {answers.model}. Sources:", "grey"))
        if not answers.results:
            lines.append("  None cited: the answer may not come from your documents.")
    elif not answers.results:
        lines.append("  No matching chunks. Has the knowledge base been synced?")
    for i, r in enumerate(answers.results, 1):
        text = " ".join(r.get("content", {}).get("text", "").split())
        score = paint(f"{r['score']:.3f}", "blue") + "  " if "score" in r else ""
        lines.append(f"\n  {i}. {score}{paint(source_location(r.get('location', {})), 'grey')}")
        lines.append(_wrap(text[:400] + (" …" if len(text) > 400 else ""), 5))
    return "\n".join(lines)


def render_knowledge_bases(found: KnowledgeBases) -> str:
    """Every knowledge base kbforge built in a region, one per line."""
    lines = [paint(f"\nKnowledge bases built by kbforge in {found.region}: {len(found)}", "bold")]
    if not found.items:
        lines.append("  None yet. Run `kbforge init` to describe one.")
    for s in found:
        colour = "green" if s.kb_status == "ACTIVE" else "amber"
        lines.append(f"\n  {paint(s.name, 'bold')}  {paint(s.kb_status.lower(), colour)}  {paint(s.source, 'grey')}")
        lines.append(f"    last sync: {s.last_sync_text}")
        lines.append(f"    {s.model} · {s.store}")
    if found.unreadable:
        lines.append(
            paint(f"\n  {found.unreadable} other knowledge base(s) couldn't be checked (no access to tags).", "amber")
        )
    return "\n".join(lines)


def render_status(status: Status) -> str:
    """Key facts about a deployed knowledge base."""
    if not status.exists:
        return f"'{status.name}' has not been created yet. Run `kbforge plan` to see what would be built."
    width = max(len(k) for k, _ in status.rows())
    lines = [paint(f"\n{status.name}", "bold")]
    lines += [f"  {k.ljust(width)}  {v}" for k, v in status.rows() if v]
    return "\n".join(lines)


def render_parsing_models(models: ParsingModels) -> str:
    """The models that can parse documents, as a table with the value to pass as ``parsing=``."""
    where = f"in {models.region}" if models.region else ""
    lines = [paint(f"\nModels that can parse documents {where}".rstrip(), "bold")]
    if models.error:
        lines.append(paint(_wrap(models.error, 2), "amber"))
    if not models.options:
        lines.append("  None here. Try another region, or use parsing: data_automation.")
        return "\n".join(lines)
    width = max(len(o.value) for o in models) + 2
    label_width = max(len(o.label) for o in models) + 2
    lines.append(paint(f"  {'parsing:'.ljust(width)}{'model'.ljust(label_width)}called via", "grey"))
    for o in models:
        mark = paint(" (recommended)", "green") if o.recommended else ""
        lines.append(f"  {o.value.ljust(width)}{o.label.ljust(label_width)}{o.route}{mark}")
        if o.note:
            lines.append(paint(f"  {' ' * width}{o.note}", "grey"))
    lines.append("")
    lines.append(paint("Use one in kb.yaml:", "grey"))
    lines.append(f"  parsing:\n    strategy: foundation_model\n    model: {models.values[0]}")
    return "\n".join(lines)
