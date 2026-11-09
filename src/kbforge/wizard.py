"""A point-and-click form for creating a knowledge base inside a notebook.

    import kbforge
    kbforge.wizard()

The form walks through three stages — describe, review the plan, build — and can show the
equivalent Python code / YAML at any time, so people can graduate from clicking to a
committed config file.

While you fill it in, the form checks each answer as you type and a side panel shows the
knowledge base taking shape (data flow and the AWS names it will use). Choices with
trade-offs, like the vector store, are shown as cards with a sentence on when to pick each,
because the people using this usually aren't AWS experts. When a model is to parse the
documents, the form can ask Bedrock which models the account can actually use.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .aws import Aws
from .knowledge_base import KnowledgeBase
from .models import DEFAULT_PARSING_MODEL, EMBEDDING_MODELS, EmbeddingModel
from .parsers import ParsingModels, known_parsing_models, list_parsing_models
from .plan import ApplyResult, Plan
from .spec import KB_NAME_RE, VECTOR_STORES, KBSpec, S3Source, WebSource, coerce_chunking, coerce_parsing
from .steps import Action
from .ui import wizard_style as ui
from .ui.progress import progress_target
from .ui.theme import card, esc

#: Vector store cards: (value, title, explanation, icon, tag).
VECTOR_STORE_CHOICES = [
    (
        "s3_vectors",
        "Amazon S3 Vectors",
        "Serverless: pay per GB stored and per query, with nothing running in between. Lowest cost.",
        "database",
        ("Recommended", "c-green"),
    ),
    (
        "opensearch_serverless",
        "OpenSearch Serverless",
        "An always-on collection that adds hybrid (keyword + meaning) search. Billed hourly, even when idle.",
        "search",
        ("", ""),
    ),
]

#: Chunking cards: (value, title, explanation, icon, tag).
CHUNKING_CHOICES = [
    (
        "fixed",
        "Fixed size",
        "300-token pieces that overlap by 20%. A safe default.",
        "scissors",
        ("Recommended", "c-green"),
    ),
    (
        "hierarchical",
        "Hierarchical",
        "Searches small pieces but returns their larger parent. For long, structured documents.",
        "tree",
        ("", ""),
    ),
    (
        "semantic",
        "Semantic",
        "Splits where the topic changes. Better matches; slower, with extra model calls during sync.",
        "wave",
        ("", ""),
    ),
    ("none", "No chunking", "Each file is one piece. Only for files that are already small.", "file", ("", "")),
]

#: Parsing cards: (value, title, explanation, icon, tag).
PARSING_CHOICES = [
    (
        "default",
        "Default parser",
        "Pulls the plain text out of each file. Free and fast; images, charts and scanned pages are skipped.",
        "file",
        ("Recommended", "c-green"),
    ),
    (
        "foundation_model",
        "A model reads each page",
        "A vision model turns tables, charts, images and scans in PDFs into text. Every page is a model call.",
        "sparkle",
        ("", ""),
    ),
    (
        "data_automation",
        "Bedrock Data Automation",
        "A managed service extracts text, tables and figures, with no model to choose. Billed per page.",
        "shield",
        ("", ""),
    ),
]

#: Embedding models that get a tag on their card.
MODEL_TAGS = {"titan-v2": ("Recommended", "c-green"), "titan-v1": ("Legacy", "c-grey")}


def _dimension_notes(model: EmbeddingModel) -> Dict[int, str]:
    """A few words on each vector size a model offers: bigger is more precise, smaller is cheaper."""
    sizes = sorted(model.dimensions)
    notes = {size: "Balanced" for size in sizes}
    notes[sizes[0]] = "Smallest, cheapest to store"
    notes[sizes[-1]] = "Most precise"
    return notes


def _default_region(region: Optional[str], session: Any) -> str:
    """The region to show in the header: the one passed in, else the session's default (if any)."""
    if region:
        return region
    try:
        import boto3

        return (session or boto3.Session()).region_name or ""
    except Exception:
        return ""


class Wizard:
    """Holds the form widgets, keeps the preview and hints in step with them, and reacts to clicks.

    The stages are: 1 Describe (fill in the form), 2 Review (a read-only plan is shown) and
    3 Build (resources are created, then you can ask questions). Changing the form after
    reviewing drops back to stage 1, so you never build something you haven't reviewed.
    """

    def __init__(self, region: Optional[str] = None, session: Any = None):
        try:
            import ipywidgets as w

            from .ui.choices import Choice, OptionCards
        except ImportError as exc:
            raise ImportError(
                "The wizard needs ipywidgets: pip install 'kbforge[notebook]', then restart the kernel."
            ) from exc
        self.w = w
        self.OptionCards = OptionCards
        self.Choice = Choice
        self.session = session
        self.region_label = _default_region(region, session)
        self.plan: Optional[Plan] = None
        self.kb: Optional[KnowledgeBase] = None
        self.stage = 1
        self.busy = False
        self.built = False

        self.header = w.HTML(ui.hero(1, self.region_label))

        self.name = self._text("hr-policies")
        self.name_hint = w.HTML()

        self.source_kind = OptionCards(
            [
                Choice("s3", "S3 bucket", "Files in a bucket, or one folder of it", "folder"),
                Choice("web", "Website", "Crawl pages starting from a few URLs", "globe"),
            ],
            value="s3",
        )
        self.source_label = w.HTML(ui.label("S3 location"))
        self.s3_uri = self._text("s3://my-bucket/folder/", mono=True)
        self.urls = w.Textarea(placeholder="https://docs.example.com\nhttps://wiki.example.com/handbook")
        self.urls.add_class("kbf-input")
        self.urls.layout = w.Layout(width="100%", display="none")
        self.source_hint = w.HTML()

        self.embedding = OptionCards(
            [
                Choice(
                    m.alias,
                    m.label,
                    m.note,
                    "sparkle",
                    MODEL_TAGS.get(m.alias, ("", "")),
                    " · ".join(str(d) for d in m.dimensions) + " dimensions",
                )
                for m in EMBEDDING_MODELS.values()
            ],
            value="titan-v2",
        )
        self.dimensions: Any = None
        self.dims_box = w.VBox(layout=w.Layout(width="100%"))

        self.parsing = OptionCards(
            [Choice(v, t, b, i, tag) for v, t, b, i, tag in PARSING_CHOICES], value="default", columns=3
        )
        self.parser: Any = None
        self.parser_cards = w.VBox(layout=w.Layout(width="100%"))
        self.parser_list_btn = self._button("Show the models my account can use", "search", "ghost")
        self.parser_hint = w.HTML()
        self.parser_prompt = w.Textarea(
            placeholder="Optional: your own instructions for the model, e.g. "
            "“Transcribe every table as Markdown and describe each chart in one sentence.”"
        )
        self.parser_prompt.add_class("kbf-input")
        self.parser_prompt.layout = w.Layout(width="100%")
        self.parser_box = w.VBox(
            [
                w.HTML(ui.label("Parsing model")),
                self.parser_cards,
                self.parser_list_btn,
                self.parser_hint,
                w.HTML(ui.label("Instructions for the model")),
                self.parser_prompt,
            ],
            layout=w.Layout(width="100%", display="none"),
        )
        self._show_parsers(known_parsing_models(self.region_label))

        self.vector_store = OptionCards(
            [Choice(v, t, b, i, tag) for v, t, b, i, tag in VECTOR_STORE_CHOICES], value="s3_vectors"
        )
        self.chunking = OptionCards([Choice(v, t, b, i, tag) for v, t, b, i, tag in CHUNKING_CHOICES], value="fixed")

        self.region = self._text("default from your session", value=region or "")
        self.role_arn = self._text("optional — leave empty to create one", mono=True)
        self.description = self._text("optional")
        self.sync = w.Checkbox(description="Run the first sync right after building", value=True, indent=False)
        self.sync.add_class("kbf-switch")
        self.adv_btn = w.Button(description="Advanced options — region, existing IAM role, description", icon="plus")
        self.adv_btn.add_class("kbf-disclose")
        self.adv_box = w.VBox(
            [
                self._grid(
                    [
                        w.VBox([w.HTML(ui.label("Region")), self.region]),
                        w.VBox([w.HTML(ui.label("Description")), self.description]),
                    ]
                ),
                self._grid([w.VBox([w.HTML(ui.label("Existing IAM role ARN")), self.role_arn])]),
                self.sync,
            ],
            layout=w.Layout(display="none", width="100%"),
        )

        self.preview = w.HTML()
        self.preview.add_class("kbf-aside")
        self.status = w.HTML()
        self.code_btn = self._button("Show as code", "code", "ghost")
        self.preview_btn = self._button("Review plan", "search", "primary")
        self.build_btn = self._button("Build it", "play", "go", disabled=True)
        self.recap_html = w.HTML()
        self.edit_btn = self._button("Edit answers", "pencil", "ghost")

        self.code = w.HTML()
        self.view = w.HTML()
        self.result = w.HTML()
        self.question = self._text("Ask your new knowledge base something…")
        self.ask_btn = self._button("Ask", "search", "primary")
        self.try_title = w.HTML()
        ask_row = w.HBox([self.question, self.ask_btn], layout=w.Layout(width="100%"))
        ask_row.add_class("ask-row")
        self.try_box = w.VBox([self.try_title, ask_row], layout=w.Layout(display="none"))
        self.try_box.add_class("try")
        self.answers = w.HTML()

        for widget in (self.name, self.s3_uri, self.urls, self.region, self.role_arn, self.description):
            widget.observe(self._on_change, names="value")
        for widget in (self.source_kind, self.vector_store, self.chunking, self.sync, self.parser_prompt):
            widget.observe(self._on_change, names="value")
        self.source_kind.observe(self._on_source_kind, names="value")
        self.parsing.observe(self._on_parsing, names="value")
        self.parser_list_btn.on_click(self._on_list_parsers)
        self.embedding.observe(self._on_embedding, names="value")
        self.adv_btn.on_click(self._on_advanced)
        self.edit_btn.on_click(lambda _: self._fold(False))
        self.preview_btn.on_click(self._on_preview)
        self.build_btn.on_click(self._on_build)
        self.code_btn.on_click(self._on_code)
        self.ask_btn.on_click(self._on_ask)
        self.question.on_submit(lambda _: self._on_ask(None))
        self._on_embedding({"new": "titan-v2"})

        self.widget = self._layout()
        self._refresh()

    def _text(self, placeholder: str, value: str = "", mono: bool = False) -> Any:
        """A full-width text box with the wizard's input style."""
        box = self.w.Text(value=value, placeholder=placeholder, layout=self.w.Layout(width="100%"))
        box.add_class("kbf-input")
        if mono:
            box.add_class("mono")
        return box

    def _button(self, label: str, icon: str, kind: str, disabled: bool = False) -> Any:
        """A button in the wizard's style; ``kind`` is primary, go (green) or ghost."""
        button = self.w.Button(description=label, icon=icon, disabled=disabled)
        button.add_class("kbf-btn")
        button.add_class(kind)
        return button

    def _grid(self, children: List[Any]) -> Any:
        """A responsive row of fields that wraps onto new lines in a narrow notebook."""
        grid = self.w.Box(children)
        grid.add_class("adv-grid")
        return grid

    def _section(self, number: object, title: str, hint: str, *children: Any) -> Any:
        """One numbered part of the form: a heading, then its fields indented beneath it."""
        w = self.w
        body = w.VBox(list(children), layout=w.Layout(width="100%"))
        body.add_class("sec-body")
        section = w.VBox([w.HTML(ui.section_head(number, title, hint)), body], layout=w.Layout(width="100%"))
        section.add_class("kbf-sec")
        return section

    def _layout(self) -> Any:
        """Put every widget in place: header, form + preview side by side, action bar, then results."""
        w = self.w
        form = w.VBox(
            [
                self._section(
                    1,
                    "Name it",
                    "Used for the knowledge base itself and to name every AWS resource kbforge creates.",
                    self.name,
                    self.name_hint,
                ),
                self._section(
                    2,
                    "Point it at your documents",
                    "PDF, Word, Excel, HTML, Markdown, CSV and text files are read; other types are skipped.",
                    self.source_kind,
                    self.source_label,
                    self.s3_uri,
                    self.urls,
                    self.source_hint,
                ),
                self._section(
                    3,
                    "How to read them",
                    "The default parser handles text documents. If your PDFs hold important tables, charts or "
                    "scanned pages, let a model read them. Changing this later means rebuilding.",
                    self.parsing,
                    self.parser_box,
                ),
                self._section(
                    4,
                    "Embedding model",
                    "Turns each piece of text into numbers so similar meanings sit close together. "
                    "Can't be changed later.",
                    self.embedding,
                    self.dims_box,
                ),
                self._section(
                    5,
                    "Vector store",
                    "Where those numbers are kept and searched. Can't be changed later.",
                    self.vector_store,
                ),
                self._section(
                    6,
                    "Chunking",
                    "How documents are cut into pieces before embedding. Smaller pieces give more precise matches.",
                    self.chunking,
                ),
                w.VBox([self.adv_btn, self.adv_box], layout=w.Layout(width="100%")),
            ],
            layout=w.Layout(width="auto"),
        )
        form.children[-1].add_class("kbf-sec")
        form.add_class("kbf-form")
        body = self.body = w.Box([form, self.preview], layout=w.Layout(width="100%"))
        body.add_class("kbf-body")
        recap = self.recap = w.Box([self.recap_html, self.edit_btn], layout=w.Layout(width="100%", display="none"))
        recap.add_class("kbf-recap")
        actions = w.Box([self.status, self.code_btn, self.preview_btn, self.build_btn], layout=w.Layout(width="100%"))
        actions.add_class("kbf-actions")
        shell = w.VBox([self.header, body, recap, actions], layout=w.Layout(width="100%"))
        shell.add_class("kbf-shell")
        out = w.VBox([self.code, self.view, self.result, self.try_box, self.answers], layout=w.Layout(width="100%"))
        out.add_class("kbf-out")
        root = w.VBox([shell, out])
        root.add_class("kbf")
        root.add_class("kbf-wiz")
        return root

    def _on_source_kind(self, change: Dict[str, Any]) -> None:
        is_s3 = change["new"] == "s3"
        self.s3_uri.layout.display = None if is_s3 else "none"
        self.urls.layout.display = "none" if is_s3 else None
        self.source_label.value = ui.label("S3 location" if is_s3 else "Seed URLs — one per line")

    def _on_parsing(self, change: Dict[str, Any]) -> None:
        """Show the model choice and instructions only when a model will do the parsing."""
        self.parser_box.layout.display = None if change["new"] == "foundation_model" else "none"
        self._on_change(None)

    def _show_parsers(self, models: ParsingModels) -> None:
        """Offer ``models`` as cards, keeping the current pick when it's still on the list."""
        current = self.parser.value if self.parser is not None else DEFAULT_PARSING_MODEL
        if not models.options:
            self.parser = None
            self.parser_cards.children = []
        else:
            self.parser = self.OptionCards(
                [
                    self.Choice(
                        o.value,
                        o.label,
                        o.note or f"{o.provider} model.",
                        "sparkle",
                        ("Recommended", "c-green") if o.recommended else ("", ""),
                        f"{o.value} · {o.route}",
                    )
                    for o in models
                ],
                value=current if current in models.values else models.values[0],
            )
            self.parser.observe(self._on_change, names="value")
            self.parser_cards.children = [self.parser]
        if models.error:
            self.parser_hint.value = ui.field_hint(esc(models.error), "bad")
        elif models.live:
            where = f" in {esc(models.region)}" if models.region else ""
            self.parser_hint.value = ui.field_hint(
                f"{len(models)} model(s) your account can use{where}. <b>Review plan</b> checks you can call the "
                "one you pick.",
                "ok",
            )
        else:
            self.parser_hint.value = ui.field_hint(
                "The parsers kbforge knows. Click the button to see exactly what your account offers."
            )

    def _on_list_parsers(self, _: Any) -> None:
        """Ask Bedrock which parsing models this account and region offer, and show them as cards."""
        self.parser_hint.value = ui.field_hint("Asking Bedrock which models you can use…")
        region = self.region.value.strip() or self.region_label or None
        try:
            models = list_parsing_models(Aws(region=region, session=self.session))
        except Exception as exc:
            models = known_parsing_models(region or "", f"Couldn't connect to AWS ({exc}).")
        self._show_parsers(models)
        self._on_change(None)

    def _parsing_option(self) -> Any:
        """The parsing choice as a ``parsing=`` value, or None for the default parser."""
        if self.parsing.value == "data_automation":
            return "data_automation"
        if self.parsing.value != "foundation_model" or self.parser is None:
            return None
        prompt = self.parser_prompt.value.strip()
        if not prompt:
            return self.parser.value
        return {"strategy": "foundation_model", "model": self.parser.value, "prompt": prompt}

    def _on_embedding(self, change: Dict[str, Any]) -> None:
        """Offer the new model's vector sizes (as small cards) when it has more than one."""
        model = EMBEDDING_MODELS[change["new"]]
        if model.configurable_dimensions:
            notes = _dimension_notes(model)
            self.dimensions = self.OptionCards(
                [self.Choice(d, f"{d} dimensions", notes[d]) for d in sorted(model.dimensions, reverse=True)],
                value=model.default_dimensions,
                columns=len(model.dimensions),
                pills=True,
            )
            self.dimensions.observe(self._on_change, names="value")
            self.dims_box.children = [self.w.HTML(ui.label("Vector size")), self.dimensions]
        else:
            self.dimensions = None
            self.dims_box.children = []
        self._on_change(None)

    def _on_advanced(self, _: Any) -> None:
        opening = self.adv_box.layout.display == "none"
        self.adv_box.layout.display = None if opening else "none"
        self.adv_btn.icon = "minus" if opening else "plus"

    def _fold(self, folded: bool) -> None:
        """Swap the full form for a one-line recap (after a plan is shown), or back again to edit."""
        if folded:
            self.recap_html.value = ui.recap(self._flow_rows())
        self.body.layout.display = "none" if folded else None
        self.recap.layout.display = None if folded else "none"

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
        if self.dimensions is not None and self.dimensions.value != model.default_dimensions:
            options["dimensions"] = self.dimensions.value
        if self.vector_store.value != "s3_vectors":
            options["vector_store"] = self.vector_store.value
        if self.chunking.value != "fixed":
            options["chunking"] = self.chunking.value
        parsing = self._parsing_option()
        if parsing is not None:
            options["parsing"] = parsing
        for key in ("region", "role_arn", "description"):
            value = getattr(self, key).value.strip()
            if value:
                options[key] = value
        if not self.sync.value:
            options["sync"] = False
        return options

    def _make_kb(self) -> KnowledgeBase:
        return KnowledgeBase(self.name.value.strip(), self._source(), session=self.session, **self._options())

    def _validate(self) -> Tuple[Optional[KBSpec], Dict[str, str]]:
        """Check the form without calling AWS.

        Returns the spec (or None) and the problems found, keyed by ``name``, ``source`` or
        ``other``. An empty problem text means "not filled in yet", which is shown as advice
        rather than an error. Problem texts are HTML, already escaped.
        """
        problems: Dict[str, str] = {}
        name = self.name.value.strip()
        if not name:
            problems["name"] = ""
        elif not KB_NAME_RE.match(name):
            problems["name"] = (
                "Use letters, digits, <code>-</code> or <code>_</code> (never two in a row), up to 100 characters."
            )

        if self.source_kind.value == "s3":
            uri = self.s3_uri.value.strip()
            if not uri:
                problems["source"] = ""
            elif not uri.startswith("s3://"):
                problems["source"] = "S3 locations start with <code>s3://</code>, like <code>s3://my-bucket/hr/</code>."
            else:
                try:
                    S3Source(uri)
                except ValueError as exc:
                    problems["source"] = esc(exc)
        else:
            urls = self._source()["web"]
            if not urls:
                problems["source"] = ""
            else:
                try:
                    WebSource(urls)
                except ValueError as exc:
                    problems["source"] = esc(exc)

        if problems:
            return None, problems
        try:
            return KBSpec(name=name, source=self._source(), **self._options()), {}
        except ValueError as exc:
            return None, {"other": esc(exc)}

    def _on_change(self, _: Any) -> None:
        """Any edit: re-check the form, redraw the preview, and forget a plan that no longer matches."""
        if self.plan is not None or self.built:
            self.plan = None
            self.built = False
            self.build_btn.disabled = True
            for output in (self.view, self.result):
                output.remove_class("stale")
                output.add_class("stale")
            self.try_box.layout.display = "none"
            self._set_stage(1)
        if hasattr(self, "widget"):
            self._refresh()

    def _refresh(self) -> None:
        spec, problems = self._validate()
        self._show_hints(problems)
        self.preview.value = ui.preview(self._flow_rows(), self._aws_names())
        self.preview_btn.disabled = self.busy or spec is None
        if self.busy:
            return
        if self.view.value and self.plan is None and not self.built and "stale" in self.view._dom_classes:
            self.status.value = ui.status(
                "warn", "You changed the form — <b>review the plan again</b> before building."
            )
        elif "name" in problems and not problems["name"]:
            self.status.value = ui.status("todo", "Next: <b>give your knowledge base a name</b>.")
        elif "source" in problems and not problems["source"]:
            what = "the S3 location" if self.source_kind.value == "s3" else "at least one URL"
            self.status.value = ui.status("todo", f"Next: <b>add {what}</b> of your documents.")
        elif problems:
            self.status.value = ui.status("bad", "<b>Check the form</b> — see the message in red above.")
        elif self.plan is None:
            self.status.value = ui.status(
                "info", "Looks good. <b>Review plan</b> checks your account and files — it changes nothing."
            )

    def _show_hints(self, problems: Dict[str, str]) -> None:
        """The advice / validation line under the name and source fields, and red borders on bad input."""
        name_problem = problems.get("name")
        if name_problem:
            self.name_hint.value = ui.field_hint(name_problem, "bad")
        elif name_problem == "":
            self.name_hint.value = ui.field_hint("Short and descriptive, like <code>hr-policies</code>.")
        else:
            self.name_hint.value = ui.field_hint("Looks good.", "ok")
        self._mark(self.name, bool(name_problem))

        source_problem = problems.get("source")
        is_s3 = self.source_kind.value == "s3"
        if source_problem:
            self.source_hint.value = ui.field_hint(source_problem, "bad")
        elif source_problem == "":
            advice = (
                "Everything under this folder is indexed. Keep it in the same region as the knowledge base."
                if is_s3
                else "The crawler follows links on the same host. Only crawl sites you're allowed to."
            )
            self.source_hint.value = ui.field_hint(advice)
        elif "other" in problems:
            self.source_hint.value = ui.field_hint(problems["other"], "bad")
        else:
            what = "That location looks" if is_s3 else "Those URLs look"
            self.source_hint.value = ui.field_hint(
                f"{what} right. <b>Review plan</b> will check you can read them.", "ok"
            )
        self._mark(self.s3_uri, bool(source_problem) and is_s3)
        self._mark(self.urls, bool(source_problem) and not is_s3)

    @staticmethod
    def _mark(widget: Any, bad: bool) -> None:
        widget.remove_class("bad")
        if bad:
            widget.add_class("bad")

    def _flow_rows(self) -> List[ui.FlowRow]:
        """The preview's data-flow rows, built from the raw form so they show even before it validates."""
        model = EMBEDDING_MODELS[self.embedding.value]
        dims = self.dimensions.value if self.dimensions is not None else model.default_dimensions
        chunk_title, _, chunk_detail = coerce_chunking(self.chunking.value).describe().partition(" · ")
        if self.source_kind.value == "s3":
            source, source_sub, source_icon = self.s3_uri.value.strip(), "S3 bucket", "folder"
        else:
            urls = self._source()["web"]
            source = urls[0] if urls else ""
            source_sub = "Web crawler" + (f" · +{len(urls) - 1} more" if len(urls) > 1 else "")
            source_icon = "globe"
        store = self.vector_store.value
        region = self.region.value.strip() or self.region_label
        try:
            parse_title, _, parse_detail = coerce_parsing(self._parsing_option()).describe().partition(" · ")
        except ValueError:  # e.g. instructions that are too long; the form shows the error
            parse_title, parse_detail = PARSING_CHOICES[1][1], ""
        return [
            ("Source", source, source_sub, source_icon),
            ("Parsing", parse_title, parse_detail, "file"),
            ("Chunking", chunk_title, chunk_detail, "scissors"),
            ("Embeddings", model.label, f"{dims} dimensions", "sparkle"),
            ("Vector store", VECTOR_STORES[store], "serverless" if store == "s3_vectors" else "collection", "database"),
            ("Knowledge base", self.name.value.strip(), "Bedrock" + (f" · {region}" if region else ""), "book"),
        ]

    def _aws_names(self) -> Optional[List[Tuple[str, str]]]:
        """The resource names kbforge will use. They only depend on the KB name, so show them as soon as it's valid."""
        name = self.name.value.strip()
        if not KB_NAME_RE.match(name):
            return None
        # Names come from the KB name alone; the source here is a stand-in so the spec can be built.
        names = KBSpec(name=name, source=S3Source("s3://placeholder/")).names
        role = self.role_arn.value.strip()
        rows = [("Knowledge base", names.kb), ("IAM role", role or names.role)]
        if self.vector_store.value == "s3_vectors":
            rows += [("Vector bucket", names.vector_bucket), ("Vector index", names.vector_index)]
        else:
            rows += [("OpenSearch collection", names.collection)]
        return rows

    def _set_stage(self, stage: int) -> None:
        """Move the header's stepper, and make "Review plan" the main button only while describing."""
        self.stage = stage
        self.header.value = ui.hero(stage, self.region_label)
        self.preview_btn.remove_class("primary")
        self.preview_btn.remove_class("ghost")
        self.preview_btn.add_class("primary" if stage == 1 else "ghost")

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        self.code_btn.disabled = busy
        if busy:
            self.preview_btn.disabled = self.build_btn.disabled = True

    def _show_error(self, message: str) -> None:
        self.view.value = card(
            '<div class="head"><div class="eyebrow">kbforge · wizard</div><h2>Something needs fixing</h2>'
            f'<div class="warn c-red">{esc(message)}</div></div>'
        )
        self.status.value = ui.status("bad", "<b>Couldn't make a plan</b> — see the message below.")

    def _on_preview(self, _: Any) -> None:
        self.result.value = self.answers.value = ""
        self.try_box.layout.display = "none"
        self.view.remove_class("stale")
        self.result.remove_class("stale")
        self.built = False
        try:
            kb = self.kb = self._make_kb()
        except ValueError as exc:
            self._show_error(str(exc))
            return
        self._set_busy(True)
        self.status.value = ui.status("busy", "Checking your account, data and permissions…")
        self.view.value = ui.loading(
            "Making a plan for " + kb.name, "Read-only checks: credentials, embedding model, your files, vector store."
        )
        try:
            plan = self.plan = kb.plan()
        except Exception as exc:
            self._set_busy(False)
            self._refresh()
            self._show_error(str(exc))
            return
        self._set_busy(False)
        self._refresh()
        self._set_stage(2)
        self._fold(plan.ready)  # keep the form open when blocked: the fix is often a typo in it
        self.view.value = plan._repr_html_()
        self.build_btn.disabled = not (plan.ready and plan.has_work)
        if plan.blockers:
            self.status.value = ui.status("bad", "<b>Blocked.</b> The plan below says what to fix; then review again.")
        elif not plan.has_work:
            self.status.value = ui.status("ok", "<b>Already built</b> — nothing to change. Try it below.")
            self.built = True
            self._set_stage(4)
            self._show_try_box()
        else:
            counts = [(Action.CREATE, "to create"), (Action.UPDATE, "to update"), (Action.RUN, "to run")]
            summary = ", ".join(f"{plan.count(a)} {words}" for a, words in counts if plan.count(a))
            self.status.value = ui.status(
                "ok", f"Plan ready: <b>{esc(summary)}</b>. Read it below, then click <b>Build it</b>."
            )

    def _on_build(self, _: Any) -> None:
        if self.kb is None:
            return
        self._set_busy(True)
        self._set_stage(3)
        self.status.value = ui.status("busy", "Building… large document sets can take several minutes to sync.")
        try:
            with progress_target(lambda html: setattr(self.view, "value", html)):
                outcome = self.kb.apply()
        except Exception as exc:
            outcome = None
            self._show_error(str(exc))
        finally:
            self._set_busy(False)
            self.preview_btn.disabled = False
        if outcome is None:
            return
        self.result.value = outcome._repr_html_()
        if isinstance(outcome, ApplyResult) and outcome.ok:
            self.plan = None
            self.built = True
            self._set_stage(4)
            self.status.value = ui.status("ok", f"<b>{esc(self.kb.name)}</b> is ready. Ask it something below.")
            self._show_try_box()
        else:
            where = f" at <b>{esc(outcome.failed_step)}</b>" if isinstance(outcome, ApplyResult) else ""
            self.build_btn.disabled = False
            self.status.value = ui.status(
                "bad",
                f"Stopped{where}. Fix the problem, then click <b>Build it</b> again — finished steps are skipped.",
            )

    def _show_try_box(self) -> None:
        if self.kb is not None:
            self.try_title.value = ui.try_it_header(self.kb.name)
            self.try_box.layout.display = None

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
            '<div class="head"><div class="eyebrow">kbforge · as code</div><h2>The same knowledge base, in code</h2>'
            '<div class="muted">Paste it into a cell, or commit the YAML or JSON file and run '
            "<code>kbforge plan kb.yaml</code> (or <code>kb.json</code>).</div></div>"
            '<div class="code2"><div class="section"><h3>Python</h3><pre><code>'
            + esc(python)
            + '</code></pre></div><div class="section"><h3>Or save as kb.yaml</h3><pre><code>'
            + esc(kb.to_yaml())
            + "</code></pre><h3>Or as kb.json</h3><pre><code>"
            + esc(kb.to_json())
            + "</code></pre></div></div>"
        )
        self.code_btn.description = "Hide code"

    def _on_ask(self, _: Any) -> None:
        if self.kb is None or not self.question.value.strip():
            return
        self.answers.value = ui.loading("Searching " + self.kb.name, self.question.value.strip())
        try:
            self.answers.value = self.kb.ask(self.question.value.strip())._repr_html_()
        except Exception as exc:
            self.answers.value = card(f'<div class="head"><div class="warn c-red">{esc(exc)}</div></div>')


def wizard(region: Optional[str] = None, session: Any = None) -> Any:
    """Show an interactive form that plans and builds a knowledge base. Needs ipywidgets."""
    return Wizard(region=region, session=session).widget
