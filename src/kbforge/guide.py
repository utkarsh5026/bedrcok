"""``kbforge.guide()``: a getting-started card for the first cell of a notebook.

It explains the whole journey on one screen — what you need, then describe → plan → build →
try — with copy-pasteable code, so nobody has to go hunting for a wiki page.
"""

from __future__ import annotations

from typing import Optional

import boto3

from .ui.theme import card, esc

_STEPS = [
    (
        "Describe it",
        "Name it and point it at your documents. Everything else has a sensible default.",
        'from kbforge import KnowledgeBase\n\nkb = KnowledgeBase("hr-policies", source="s3://my-bucket/hr/")',
    ),
    (
        "Review the plan",
        "Checks your access, the embedding model and your files, then lists every AWS resource it would "
        "create. Nothing is changed.",
        "kb.plan()",
    ),
    (
        "Build it",
        "Creates the resources and runs the first sync, with live progress. Safe to re-run if anything fails.",
        "kb.apply()",
    ),
    (
        "Try it",
        "See which chunks come back for a real question. Re-sync whenever documents change.",
        'kb.ask("How many days of parental leave do we get?")\nkb.sync()',
    ),
]


class Guide:
    """Displays the getting-started card. Create it with ``kbforge.guide()``."""

    def __init__(self, region: Optional[str] = None):
        self.region = region or boto3.Session().region_name

    def _repr_html_(self) -> str:
        steps = "".join(
            f'<li><span class="num">{i}</span><div><div class="title">{esc(title)}</div>'
            f'<div class="explain">{esc(text)}</div><pre><code>{esc(code)}</code></pre></div></li>'
            for i, (title, text, code) in enumerate(_STEPS, 1)
        )
        region = (
            f'<span class="chip">region <b>{esc(self.region)}</b></span>'
            if self.region
            else '<span class="chip">region <b>not set — pass region="us-east-1"</b></span>'
        )
        needs = (
            "<ul style='margin:0;padding-left:18px'>"
            "<li>Documents in S3 (in the same region as the knowledge base) or a website you may crawl.</li>"
            "<li>A notebook role allowed to use Bedrock, S3 Vectors and IAM — or an existing Bedrock role "
            "to pass as <code>role_arn=</code> if IAM is locked down.</li>"
            "</ul>"
        )
        return card(
            '<div class="head"><div class="eyebrow">kbforge · getting started</div>'
            "<h2>From documents to a Bedrock knowledge base in four cells</h2>"
            f'<div class="chips">{region}</div></div>'
            f'<div class="section"><h3>You need</h3>{needs}</div>'
            f'<div class="section"><ol class="steps">{steps}</ol></div>'
            '<div class="section next"><h3>Prefer clicking?</h3><pre><code>import kbforge\n'
            "kbforge.wizard()</code></pre></div>"
        )

    def __repr__(self) -> str:
        lines = ["kbforge — getting started", ""]
        for i, (title, text, code) in enumerate(_STEPS, 1):
            lines += [f"{i}. {title}: {text}", *("     " + line for line in code.splitlines()), ""]
        return "\n".join(lines)


def guide(region: Optional[str] = None) -> Guide:
    """Show the getting-started card (put it as the last line of a notebook cell)."""
    return Guide(region)
