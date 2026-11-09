"""A point-and-click form for creating a knowledge base inside a notebook.

    import kbforge
    kbforge.wizard()

The form walks through three stages — describe, review the plan, build — and can show the
equivalent Python code / YAML at any time, so people can graduate from clicking to a
committed config file.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .knowledge_base import KnowledgeBase
from .models import EMBEDDING_MODELS
from .plan import ApplyResult, Plan
from .ui.html import render_spec
from .ui.progress import progress_target
from .ui.theme import card, esc

VECTOR_STORE_CHOICES = [
    ("Amazon S3 Vectors — serverless, lowest cost (recommended)", "s3_vectors"),
    ("OpenSearch Serverless — always-on collection, hybrid search", "opensearch_serverless"),
]

CHUNKING_CHOICES = [
    ("Fixed size — 300 tokens, 20% overlap (recommended)", "fixed"),
    ("Hierarchical — long, structured documents", "hierarchical"),
    ("Semantic — split where the topic changes", "semantic"),
    ("None — each file is one chunk", "none"),
]


def _stage_bar(active: int) -> str:
    """The "1 Describe → 2 Review plan → 3 Build" header, highlighting the current stage."""
    names = ["Describe", "Review plan", "Build & try it"]
    parts = []
    for i, label in enumerate(names, 1):
        colour = "var(--kbf-blue)" if i == active else ("var(--kbf-green)" if i < active else "var(--kbf-border)")
        weight = "600" if i == active else "400"
        parts.append(
            f'<span style="display:inline-flex;align-items:center;gap:6px;font-weight:{weight}">'
            f'<span class="icon" style="--c:{colour}">{"✓" if i < active else i}</span>{label}</span>'
        )
    arrows = '<span class="muted" style="margin:0 10px">→</span>'.join(parts)
    return card(
        '<div class="head"><div class="eyebrow">kbforge</div><h2>Create a Bedrock knowledge base</h2>'
        f'<div style="display:flex;flex-wrap:wrap;align-items:center;margin-top:4px">{arrows}</div></div>'
    )


class Wizard:
    """Holds the form widgets and reacts to button clicks."""

    def __init__(self, region: Optional[str] = None, session: Any = None):
        try:
            import ipywidgets as w
        except ImportError as exc:
            raise ImportError(
                "The wizard needs ipywidgets: pip install 'kbforge[notebook]', then restart the kernel."
            ) from exc
        self.w = w
        self.session = session
        self.plan: Optional[Plan] = None
        self.kb: Optional[KnowledgeBase] = None

        wide = w.Layout(width="560px")
        style = {"description_width": "130px"}

        self.header = w.HTML(_stage_bar(1))
        self.name = w.Text(description="Name", placeholder="hr-policies", layout=wide, style=style)
        self.source_kind = w.ToggleButtons(
            description="Documents in",
            options=[("S3 bucket", "s3"), ("Website", "web")],
            style={**style, "button_width": "120px"},
        )
        self.s3_uri = w.Text(description="S3 location", placeholder="s3://my-bucket/folder/", layout=wide, style=style)
        self.urls = w.Textarea(
            description="Seed URLs",
            placeholder="https://docs.example.com\n(one per line)",
            layout=w.Layout(width="560px", height="80px", display="none"),
            style=style,
        )
        self.embedding = w.Dropdown(
            description="Embedding model",
            options=[(f"{m.label} — {m.note}" if m.note else m.label, m.alias) for m in EMBEDDING_MODELS.values()],
            value="titan-v2",
            layout=wide,
            style=style,
        )
        self.dimensions = w.Dropdown(description="Dimensions", layout=w.Layout(width="260px"), style=style)
        self.vector_store = w.Dropdown(
            description="Vector store", options=VECTOR_STORE_CHOICES, value="s3_vectors", layout=wide, style=style
        )
        self.chunking = w.Dropdown(
            description="Chunking", options=CHUNKING_CHOICES, value="fixed", layout=wide, style=style
        )
        self.region = w.Text(
            description="Region",
            value=region or "",
            placeholder="default from your session",
            layout=w.Layout(width="360px"),
            style=style,
        )
        self.role_arn = w.Text(
            description="Existing role", placeholder="optional — leave empty to create one", layout=wide, style=style
        )
        self.description = w.Text(description="Description", placeholder="optional", layout=wide, style=style)
        self.sync = w.Checkbox(description="Run the first sync right after building", value=True, indent=False)
        advanced = w.Accordion(children=[w.VBox([self.region, self.role_arn, self.description, self.sync])])
        advanced.set_title(0, "Advanced (region, existing IAM role, …)")
        advanced.selected_index = None

        self.preview_btn = w.Button(description="Review plan", button_style="primary", icon="search")
        self.build_btn = w.Button(description="Build it", button_style="success", icon="play", disabled=True)
        self.code_btn = w.Button(description="Show as code", icon="code")
        self.view = w.HTML()
        self.result = w.HTML()
        self.code = w.HTML()

        self.question = w.Text(placeholder="Ask your new knowledge base something…", layout=w.Layout(width="440px"))
        self.ask_btn = w.Button(description="Ask", icon="comment")
        self.ask_row = w.HBox([self.question, self.ask_btn], layout=w.Layout(display="none", margin="8px 0"))
        self.answers = w.HTML()

        self.source_kind.observe(self._on_source_kind, names="value")
        self.embedding.observe(self._on_embedding, names="value")
        self.preview_btn.on_click(self._on_preview)
        self.build_btn.on_click(self._on_build)
        self.code_btn.on_click(self._on_code)
        self.ask_btn.on_click(self._on_ask)
        self.question.on_submit(lambda _: self._on_ask(None))
        self._on_embedding({"new": "titan-v2"})

        self.widget = w.VBox(
            [
                self.header,
                self.name,
                self.source_kind,
                self.s3_uri,
                self.urls,
                self.embedding,
                self.dimensions,
                self.vector_store,
                self.chunking,
                advanced,
                w.HBox([self.preview_btn, self.build_btn, self.code_btn], layout=w.Layout(margin="10px 0")),
                self.code,
                self.view,
                self.result,
                self.ask_row,
                self.answers,
            ]
        )

    def _on_source_kind(self, change: Dict[str, Any]) -> None:
        is_s3 = change["new"] == "s3"
        self.s3_uri.layout.display = None if is_s3 else "none"
        self.urls.layout.display = "none" if is_s3 else None

    def _on_embedding(self, change: Dict[str, Any]) -> None:
        model = EMBEDDING_MODELS[change["new"]]
        self.dimensions.options = [(str(d), d) for d in model.dimensions]
        self.dimensions.value = model.default_dimensions
        self.dimensions.disabled = not model.configurable_dimensions

    def _source(self) -> Any:
        if self.source_kind.value == "s3":
            return self.s3_uri.value.strip()
        urls = [u.strip() for u in self.urls.value.splitlines() if u.strip()]
        return {"web": urls}

    def _options(self) -> Dict[str, Any]:
        """The form's values as KnowledgeBase keyword arguments, leaving out defaults."""
        options: Dict[str, Any] = {}
        if self.embedding.value != "titan-v2":
            options["embedding"] = self.embedding.value
        model = EMBEDDING_MODELS[self.embedding.value]
        if model.configurable_dimensions and self.dimensions.value != model.default_dimensions:
            options["dimensions"] = self.dimensions.value
        if self.vector_store.value != "s3_vectors":
            options["vector_store"] = self.vector_store.value
        if self.chunking.value != "fixed":
            options["chunking"] = self.chunking.value
        for key in ("region", "role_arn", "description"):
            value = getattr(self, key).value.strip()
            if value:
                options[key] = value
        if not self.sync.value:
            options["sync"] = False
        return options

    def _make_kb(self) -> KnowledgeBase:
        return KnowledgeBase(self.name.value.strip(), self._source(), session=self.session, **self._options())

    def _show_error(self, message: str) -> None:
        self.view.value = card(
            f'<div class="head"><div class="eyebrow">kbforge</div><h2>Check the form</h2>'
            f'<div class="warn c-red">{esc(message)}</div></div>'
        )

    def _on_preview(self, _: Any) -> None:
        self.build_btn.disabled = True
        self.result.value = self.answers.value = ""
        self.ask_row.layout.display = "none"
        try:
            kb = self.kb = self._make_kb()
        except ValueError as exc:
            self._show_error(str(exc))
            return
        self.view.value = render_spec(kb.spec).replace(
            "Not checked against AWS yet.", "Checking your account, data and permissions…"
        )
        try:
            plan = self.plan = kb.plan()
        except Exception as exc:
            self._show_error(str(exc))
            return
        self.header.value = _stage_bar(2)
        self.view.value = plan._repr_html_()
        self.build_btn.disabled = not (plan.ready and plan.has_work)

    def _on_build(self, _: Any) -> None:
        if self.kb is None:
            return
        self.build_btn.disabled = self.preview_btn.disabled = True
        self.header.value = _stage_bar(3)
        try:
            with progress_target(lambda html: setattr(self.view, "value", html)):
                outcome = self.kb.apply()
        finally:
            self.preview_btn.disabled = False
        self.result.value = outcome._repr_html_()
        if isinstance(outcome, ApplyResult) and outcome.ok:
            self.ask_row.layout.display = None

    def _on_code(self, _: Any) -> None:
        if self.code.value:
            self.code.value = ""
            self.code_btn.description = "Show as code"
            return
        try:
            kb = self._make_kb()
        except ValueError as exc:
            self._show_error(str(exc))
            return
        args: List[str] = [f"    {kb.name!r},", f"    source={self._source()!r},"]
        args += [f"    {k}={v!r}," for k, v in self._options().items()]
        python = "from kbforge import KnowledgeBase\n\nkb = KnowledgeBase(\n" + "\n".join(args) + "\n)\nkb.plan()"
        self.code.value = card(
            '<div class="section"><h3>Python</h3><pre><code>' + esc(python) + "</code></pre></div>"
            '<div class="section"><h3>Or save as kb.yaml</h3><pre><code>' + esc(kb.to_yaml()) + "</code></pre></div>"
        )
        self.code_btn.description = "Hide code"

    def _on_ask(self, _: Any) -> None:
        if self.kb is None or not self.question.value.strip():
            return
        try:
            self.answers.value = self.kb.ask(self.question.value.strip())._repr_html_()
        except Exception as exc:
            self.answers.value = card(f'<div class="head"><div class="warn c-red">{esc(exc)}</div></div>')


def wizard(region: Optional[str] = None, session: Any = None) -> Any:
    """Show an interactive form that plans and builds a knowledge base. Needs ipywidgets."""
    return Wizard(region=region, session=session).widget
