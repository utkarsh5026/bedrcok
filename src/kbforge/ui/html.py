"""HTML cards for notebooks.

Each ``render_*`` function takes one kbforge object and returns a self-contained HTML string.
Objects hook these up through ``_repr_html_``, which is what makes Jupyter draw a card when
the object is the last expression in a cell.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List

from ..checks import FAIL, OK, WARN, Check, human_bytes
from ..spec import VECTOR_STORES, KBSpec
from ..steps import Action
from .theme import SERIES, card, esc

if TYPE_CHECKING:
    from ..deployed import KnowledgeBases, Status
    from ..knowledge_base import Answers
    from ..parsers import ParsingModels
    from ..plan import ApplyResult, Plan, PlanItem

#: How each action is labelled and coloured in the plan.
ACTION_STYLE = {
    Action.CREATE: ("create", "c-green"),
    Action.UPDATE: ("update", "c-amber"),
    Action.EXISTS: ("exists", "c-grey"),
    Action.USE: ("use yours", "c-grey"),
    Action.RUN: ("run", "c-blue"),
    Action.SKIP: ("skip", "c-grey"),
    Action.CONFLICT: ("blocked", "c-red"),
    Action.DELETE: ("delete", "c-red"),
    Action.ABSENT: ("gone", "c-grey"),
    Action.KEEP: ("keep", "c-amber"),
}

CHECK_STYLE = {OK: ("✓", "var(--kbf-green)"), WARN: ("!", "var(--kbf-amber)"), FAIL: ("✕", "var(--kbf-red)")}


def console_url(region: str) -> str:
    """Link to the Bedrock knowledge-base list in the AWS console."""
    return f"https://{region}.console.aws.amazon.com/bedrock/home?region={region}#/knowledge-bases"


def _badge(action: Action) -> str:
    label, colour = ACTION_STYLE[action]
    return f'<span class="badge {colour}">{label}</span>'


def _chips(items: List[tuple]) -> str:
    return (
        '<div class="chips">'
        + "".join(f'<span class="chip">{esc(k)} <b>{esc(v)}</b></span>' for k, v in items)
        + "</div>"
    )


def _code(text: str) -> str:
    return f"<pre><code>{esc(text)}</code></pre>"


def _source_stats(checks: List[Check]) -> str:
    """'132 files · 48 MB' from the source check, if we scanned an S3 bucket."""
    for check in checks:
        if check.title == "Source documents" and check.data.get("files"):
            return f"{check.data['files']:,} files · {human_bytes(check.data['bytes'])}"
    return ""


def render_pipeline(spec: KBSpec, source_stats: str = "") -> str:
    """The data flow, left to right: source → (parsing) → chunking → embeddings → vector store → knowledge base."""
    parsing = spec.parsing.describe().split(" · ", 1)
    stages = [
        (
            "Source",
            spec.source.describe(),
            source_stats or ("S3 bucket" if spec.source.kind == "s3" else "Web crawler"),
        ),
        # The default parser is left out to keep the common case short.
        *([("Parsing", parsing[0], parsing[-1])] if spec.parsing.strategy != "default" else []),
        ("Chunking", spec.chunking.describe().split(" · ")[0], spec.chunking.describe().split(" · ", 1)[-1]),
        ("Embeddings", spec.model.label, f"{spec.dims} dimensions"),
        (
            "Vector store",
            VECTOR_STORES[spec.vector_store],
            "serverless" if spec.vector_store == "s3_vectors" else "collection",
        ),
        ("Knowledge base", spec.name, "Bedrock"),
    ]
    parts = []
    for i, (label, value, sub) in enumerate(stages):
        if i:
            parts.append('<div class="arrow">→</div>')
        parts.append(
            f'<div class="stage"><div class="label">{label}</div>'
            f'<div class="value">{esc(value)}</div><div class="sub">{esc(sub)}</div></div>'
        )
    return f'<div class="pipeline">{"".join(parts)}</div>'


def _render_check(check: Check) -> str:
    glyph, colour = CHECK_STYLE[check.status]
    extra = ""
    by_ext = check.data.get("by_ext")
    if by_ext:
        total = sum(n for _, n in by_ext) or 1
        supported = set(check.data.get("supported_exts", []))
        segments = "".join(
            f'<i style="width:{100 * n / total:.2f}%;background:{SERIES[i % len(SERIES)]}" title="{esc(ext)}: {n}"></i>'
            for i, (ext, n) in enumerate(by_ext)
        )
        legend = "".join(
            f'<span><i style="background:{SERIES[i % len(SERIES)]}"></i>{esc(ext)} {n:,}'
            f"{'' if ext in supported else ' (skipped)'}</span>"
            for i, (ext, n) in enumerate(by_ext)
        )
        extra = f'<div class="bar">{segments}</div><div class="legend">{legend}</div>'
    return (
        f'<li><span class="icon" style="--c:{colour}">{glyph}</span>'
        f'<div><span class="title">{esc(check.title)}</span>'
        f'<div class="muted">{esc(check.detail)}</div>{extra}</div></li>'
    )


def _render_item(number: int, item: PlanItem) -> str:
    insp = item.inspection
    note = f'<div class="note muted">{esc(insp.note)}</div>' if insp.note else ""
    colour = "c-red" if insp.action == Action.CONFLICT else "c-amber"
    warning = f'<div class="warn {colour}">{esc(insp.warning)}</div>' if insp.warning else ""
    details = ""
    if item.details:
        label, text = item.details
        details = f"<details><summary>{esc(label)}</summary>{_code(text)}</details>"
    return (
        f'<li><span class="num">{number}</span><div>'
        f'<div class="row"><span class="title">{esc(item.step.title)}</span>{_badge(insp.action)}'
        f'<span class="res mono">{esc(item.resource)}</span></div>'
        f'<div class="explain">{esc(item.explain)}</div>{note}{warning}{details}</div></li>'
    )


def _summary(plan: Plan) -> str:
    wording = {
        Action.CREATE: "to create",
        Action.UPDATE: "to update",
        Action.RUN: "to run",
        Action.EXISTS: "already exist",
        Action.DELETE: "to delete",
        Action.KEEP: "kept",
        Action.ABSENT: "already gone",
        Action.CONFLICT: "blocked",
    }
    parts = [f"<span><b>{plan.count(a)}</b> {w}</span>" for a, w in wording.items() if plan.count(a)]
    return f'<div class="summary">{"".join(parts)}</div>' if parts else ""


def _banner(plan: Plan) -> str:
    if plan.blockers:
        items = "".join(f"<li>{esc(b)}</li>" for b in plan.blockers)
        return (
            f'<div class="banner c-red"><b>Blocked — nothing will be changed.</b> Fix these, then plan again:'
            f'<ul style="margin:6px 0 0 18px;padding:0">{items}</ul></div>'
        )
    if not plan.has_work:
        return '<div class="banner c-grey"><b>Nothing to do.</b> Everything already matches this spec.</div>'
    if plan.mode == "destroy":
        return (
            '<div class="banner c-red"><b>Ready to delete.</b> '
            "This removes the vectors; your source documents are untouched.</div>"
        )
    return '<div class="banner c-green"><b>Ready.</b> Nothing has been created yet — review the steps below.</div>'


def _next(plan: Plan) -> str:
    if plan.blockers:
        text = "# after fixing the blockers above\nkb.plan()"
    elif not plan.has_work:
        text = 'kb.ask("a question your documents answer")\nkb.sync()   # after adding or changing documents'
    elif plan.mode == "destroy":
        text = "kb.destroy(confirm=True)" + ("" if not plan.force else "  # with force=True")
    else:
        text = "kb.apply()"
    return f'<div class="section next"><h3>Next</h3>{_code(text)}</div>'


def render_plan(plan: Plan) -> str:
    """The plan card: header, data-flow pipeline, preflight checks, numbered steps, next action."""
    spec = plan.spec
    try:
        account = plan.aws.account_id
    except Exception:
        account = "unknown"
    eyebrow = "kbforge · destroy plan" if plan.mode == "destroy" else "kbforge · plan"
    head = (
        f'<div class="head"><div class="eyebrow">{eyebrow}</div><h2>{esc(spec.name)}</h2>'
        + _chips([("account", account), ("region", plan.aws.region), ("store", VECTOR_STORES[spec.vector_store])])
        + _summary(plan)
        + _banner(plan)
        + "</div>"
    )
    sections = []
    if plan.mode == "create":
        sections.append(
            f'<div class="section"><h3>Data flow</h3>{render_pipeline(spec, _source_stats(plan.checks))}</div>'
        )
    if plan.checks:
        checks = "".join(_render_check(c) for c in plan.checks)
        sections.append(f'<div class="section"><h3>Preflight checks</h3><ul class="checks">{checks}</ul></div>')
    if plan.items:
        title = "What will be deleted" if plan.mode == "destroy" else "What will be built"
        items = "".join(_render_item(i + 1, item) for i, item in enumerate(plan.items))
        sections.append(f'<div class="section"><h3>{title}</h3><ol class="steps">{items}</ol></div>')
    sections.append(_next(plan))
    return card(head + "".join(sections))


def render_apply_result(result: ApplyResult) -> str:
    """The card shown after apply: what was built (and how to use it) or where it stopped."""
    name = result.spec.name
    if not result.ok:
        where = f"Stopped at <b>{esc(result.failed_step)}</b>." if result.failed_step else ""
        retry = _code("# fix the problem, then run again — finished steps are skipped\nkb.apply()")
        return card(
            f'<div class="head"><div class="eyebrow">kbforge</div><h2>{esc(name)} '
            f'<span class="badge c-red">not finished</span></h2><div class="muted">{where}</div>'
            f'<div class="warn c-red">{esc(result.error)}</div></div>'
            f'<div class="section next"><h3>Next</h3>{retry}</div>'
        )
    if result.mode == "destroy":
        return card(
            f'<div class="head"><div class="eyebrow">kbforge</div><h2>{esc(name)} '
            f'<span class="badge c-grey">deleted</span></h2>'
            f'<div class="muted">Finished in {result.seconds:.0f}s. Your source documents were not touched.</div></div>'
        )
    out = result.out
    rows = [
        ("Knowledge base ID", out.get("kb_id", "")),
        ("Data source ID", out.get("ds_id", "")),
        ("Region", result.region),
    ]
    job = out.get("last_job")
    if job:
        from ..steps import describe_stats

        rows.append(("First sync", describe_stats(job.get("statistics", {}))))
    grid = "".join(f'<span class="k">{esc(k)}</span><span class="mono">{esc(v)}</span>' for k, v in rows)
    usage = (
        'kb.ask("a question your documents answer")\n'
        "kb.sync()      # after adding or changing documents\n\n"
        "# From any application:\n"
        "import boto3\n"
        f'runtime = boto3.client("bedrock-agent-runtime", region_name="{result.region}")\n'
        f'runtime.retrieve(knowledgeBaseId="{out.get("kb_id", "")}", retrievalQuery={{"text": "..."}})'
    )
    return card(
        f'<div class="head"><div class="eyebrow">kbforge</div><h2>{esc(name)} '
        f'<span class="badge c-green">ready</span></h2>'
        f'<div class="muted">Built in {result.seconds:.0f}s · <a href="{esc(console_url(result.region))}" '
        f'target="_blank" rel="noopener">open in the AWS console</a></div></div>'
        f'<div class="section"><div class="grid2">{grid}</div></div>'
        f'<div class="section next"><h3>Use it</h3>{_code(usage)}</div>'
    )


def source_location(loc: Dict[str, Any]) -> str:
    """Pull a human-readable source (S3 URI or web URL) out of a retrieval result location."""
    for key in ("s3Location", "webLocation", "confluenceLocation", "sharePointLocation", "salesforceLocation"):
        if key in loc:
            value = loc[key]
            return value.get("uri") or value.get("url") or json.dumps(value)
    return loc.get("type", "unknown source")


def _render_chunks(answers: Answers) -> str:
    """The retrieved (or cited) chunks, numbered, each with its score when there is one and its source."""
    scores = [r["score"] for r in answers.results if "score" in r]
    top = max(scores, default=0) or 1
    items = []
    for i, r in enumerate(answers.results, 1):
        text = r.get("content", {}).get("text", "")
        short = text if len(text) <= 600 else text[:600].rsplit(" ", 1)[0] + " …"
        more = f"<details><summary>show full chunk</summary>{_code(text)}</details>" if short != text else ""
        score = ""
        if "score" in r:
            score = (
                f'<span class="score"><i style="width:{100 * r["score"] / top:.0f}%"></i></span>'
                f'<span class="mono muted">{r["score"]:.3f}</span>'
            )
        items.append(
            f'<div class="result"><div class="row"><span class="num">{i}</span>{score}'
            f'<span class="res mono">{esc(source_location(r.get("location", {})))}</span></div>'
            f'<div class="snippet">{esc(short)}</div>{more}</div>'
        )
    return "".join(items)


def render_answers(answers: Answers) -> str:
    """The card for ``kb.ask``: the written answer with citation markers (if asked for), then the chunks."""
    head = (
        f'<div class="head"><div class="eyebrow">kbforge · ask · {esc(answers.kb_name)}</div>'
        f"<h2>{esc(answers.question)}</h2>"
    )
    if answers.answer is None:
        if not answers.results:
            body = '<div class="section muted">No matching chunks. Has the knowledge base been synced?</div>'
        else:
            body = f'<div class="section">{_render_chunks(answers)}</div>'
        return card(f'{head}<div class="muted">{len(answers.results)} chunk(s) retrieved</div></div>{body}')

    text = "".join(
        esc(piece) + "".join(f'<sup class="cite">[{n}]</sup>' for n in numbers)
        for piece, numbers in answers.answer_with_markers()
    )
    if answers.results:
        sources = f'<div class="section"><h3>Sources</h3>{_render_chunks(answers)}</div>'
    else:
        sources = (
            '<div class="section muted">The answer cites no passages, so it may not come from your documents. '
            "Run <code>kb.ask(question)</code> without <code>answer=True</code> to see what the search finds.</div>"
        )
    return card(
        f'{head}<div class="muted">Answered by <span class="mono">{esc(answers.model)}</span> from '
        f"{len(answers.results)} cited chunk(s)</div></div>"
        f'<div class="section"><div class="answer">{text}</div></div>{sources}'
    )


def render_knowledge_bases(found: KnowledgeBases) -> str:
    """The table for ``kbforge.knowledge_bases()``: every KB kbforge built in a region."""
    note = (
        f'<div class="warn c-amber">{found.unreadable} other knowledge base(s) couldn\'t be checked because your '
        "role can't read their tags (bedrock:ListTagsForResource).</div>"
        if found.unreadable
        else ""
    )
    if not found.items:
        body = (
            f'<div class="section muted">kbforge hasn\'t built any knowledge bases in {esc(found.region)} yet. '
            "Run <code>kbforge.guide()</code> to start one, or pass <code>region=...</code> to look elsewhere.</div>"
        )
        example = ""
    else:
        rows = "".join(
            f'<tr><td class="mono nw">{esc(s.name)}</td>'
            f'<td><span class="badge {"c-green" if s.kb_status == "ACTIVE" else "c-amber"}">'
            f"{esc(s.kb_status.lower())}</span></td>"
            f'<td class="mono">{esc(s.source)}</td><td>{esc(s.last_sync_text)}</td>'
            f'<td class="muted">{esc(s.model)} · {esc(s.store)}</td></tr>'
            for s in found
        )
        body = (
            '<div class="section"><div class="tbl"><table><thead><tr><th>Name</th><th>Status</th><th>Source</th>'
            f"<th>Last sync</th><th>Model · store</th></tr></thead><tbody>{rows}</tbody></table></div></div>"
        )
        example = (
            '<div class="section next"><h3>Pick one up</h3>'
            + _code(
                f'kb = KnowledgeBase.open("{found.names[0]}")\nkb.ask("…")   # or kb.sync(), kb.status(), kb.destroy()'
            )
            + "</div>"
        )
    return card(
        '<div class="head"><div class="eyebrow">kbforge · knowledge bases</div>'
        f"<h2>{len(found)} knowledge base(s) built by kbforge</h2>"
        + _chips([("region", found.region)])
        + f"{note}</div>{body}{example}"
    )


def render_status(status: Status) -> str:
    """A snapshot of a deployed knowledge base: ids, health and the last sync."""
    if not status.exists:
        return card(
            f'<div class="head"><div class="eyebrow">kbforge · status</div><h2>{esc(status.name)} '
            f'<span class="badge c-grey">not created</span></h2>'
            f'<div class="muted">Run <code>kb.plan()</code> to see what would be built.</div></div>'
        )
    colour = "c-green" if status.kb_status == "ACTIVE" else "c-amber"
    grid = "".join(f'<span class="k">{esc(k)}</span><span class="mono">{esc(v)}</span>' for k, v in status.rows() if v)
    return card(
        f'<div class="head"><div class="eyebrow">kbforge · status</div><h2>{esc(status.name)} '
        f'<span class="badge {colour}">{esc(status.kb_status.lower())}</span></h2>'
        f'<div class="muted"><a href="{esc(console_url(status.region))}" target="_blank" rel="noopener">'
        f"open in the AWS console</a></div></div>"
        f'<div class="section"><div class="grid2">{grid}</div></div>'
    )


def render_spec(spec: KBSpec) -> str:
    """What a ``KnowledgeBase`` object shows before any AWS calls: just the intended design."""
    next_step = _code("kb.plan()   # checks your data and shows what will be built")
    return card(
        f'<div class="head"><div class="eyebrow">kbforge · knowledge base</div><h2>{esc(spec.name)}</h2>'
        f'<div class="muted">Not checked against AWS yet.</div></div>'
        f'<div class="section"><h3>Data flow</h3>{render_pipeline(spec)}</div>'
        f'<div class="section next"><h3>Next</h3>{next_step}</div>'
    )


def render_parsing_models(models: ParsingModels) -> str:
    """The models that can parse documents in a region, with the value to pass as ``parsing=``."""
    if models.live:
        intro = (
            f"{len(models)} model(s) in {esc(models.region)} can read your documents' tables, charts and scans. "
            "The plan checks you can actually call the one you pick."
        )
    else:
        intro = "The parsers kbforge knows. Availability depends on your region."
    warning = f'<div class="warn c-amber">{esc(models.error)}</div>' if models.error else ""
    rows = "".join(
        f'<tr><td class="mono">{esc(o.value)}'
        + (' <span class="badge c-green">recommended</span>' if o.recommended else "")
        + f"</td><td>{esc(o.label)}</td><td>{esc(o.provider)}</td><td>{esc(o.route)}</td>"
        f'<td class="muted">{esc(o.note)}</td></tr>'
        for o in models
    )
    table = (
        '<div class="tbl"><table><thead><tr><th>Pass as parsing=</th><th>Model</th><th>Provider</th>'
        f"<th>Called via</th><th>Notes</th></tr></thead><tbody>{rows}</tbody></table></div>"
        if models.options
        else '<div class="muted">No parsing models are available in this region. Try another region, '
        'or use parsing="data_automation".</div>'
    )
    example = models.values[0] if models.options else "nova-lite"
    usage = _code(
        f'kb = KnowledgeBase("hr-policies", source="s3://my-bucket/hr/", parsing="{example}")\n\n'
        "# with your own instructions for the model:\n"
        "from kbforge import Parsing\n"
        f'parsing = Parsing.foundation_model("{example}", prompt="Transcribe tables as Markdown.")'
    )
    return card(
        '<div class="head"><div class="eyebrow">kbforge · parsing models</div>'
        "<h2>Models that can parse documents</h2>"
        + _chips([("region", models.region or "not set"), ("source", "your account" if models.live else "built-in")])
        + f'<div class="muted">{intro}</div>{warning}</div>'
        f'<div class="section">{table}</div>'
        f'<div class="section next"><h3>Use one</h3>{usage}</div>'
    )
