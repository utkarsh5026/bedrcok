"""``KnowledgeBase``: the one object most people need.

Typical notebook session::

    from kbforge import KnowledgeBase

    kb = KnowledgeBase("hr-policies", source="s3://my-company-docs/hr/")
    kb.plan()       # 1. look before you leap: checks + what will be created
    kb.apply()      # 2. build it, with live progress
    kb.ask("How many days of parental leave do we get?")   # 3. try it
    kb.sync()       # later: re-ingest after documents change
    kb.destroy()    # shows what would be deleted; destroy(confirm=True) deletes it
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from .aws import Aws
from .deployed import Status, find_data_source, make_status, open_spec
from .errors import KBForgeError, did_you_mean
from .models import DEFAULT_ANSWER_MODEL, resolve_answer_model
from .parsers import ParsingModels, list_parsing_models
from .plan import ApplyResult, Plan, make_plan
from .spec import Chunking, KBSpec, Parsing, Source
from .steps import Ctx, describe_stats, explain_error, run_ingestion
from .steps.bedrock import DataSourceStep, KnowledgeBaseStep

__all__ = ["Answers", "KBForgeError", "KnowledgeBase", "Status"]


@dataclass
class Answers:
    """What a knowledge base returned for a question.

    ``results`` are the retrieved chunks, best match first. When an answer was asked for
    (``kb.ask(..., answer=True)``), ``answer`` holds the model's reply and ``citations`` says
    which chunks back up which part of it; ``results`` are then the chunks it cited.
    """

    question: str
    kb_name: str
    results: List[Dict[str, Any]] = field(default_factory=list)
    #: The generated answer, or None when only retrieval was asked for.
    answer: Optional[str] = None
    #: Bedrock's citations: each says which part of the answer (``span``) which chunks support.
    citations: List[Dict[str, Any]] = field(default_factory=list)
    #: The model that wrote the answer, e.g. ``us.amazon.nova-lite-v1:0``.
    model: str = ""

    @property
    def texts(self) -> List[str]:
        """Just the chunk texts, e.g. to paste into a prompt."""
        return [r.get("content", {}).get("text", "") for r in self.results]

    def answer_with_markers(self) -> List[Tuple[str, List[int]]]:
        """The answer split into pieces, each with the numbers of the chunks (in ``results``) it cites.

        Used to draw ``[1]``-style markers after each cited passage. Text no citation covers
        comes back as a piece with no numbers.
        """
        if self.answer is None:
            return []
        keys = [_chunk_key(r) for r in self.results]
        pieces: List[Tuple[str, List[int]]] = []
        position = 0
        for citation in self.citations:
            part = citation.get("generatedResponsePart", {}).get("textResponsePart", {})
            span = part.get("span", {})
            end = span.get("end")
            if end is None or end < position:
                continue
            end = min(end + 1, len(self.answer))  # Bedrock's span end is inclusive
            numbers = sorted({keys.index(_chunk_key(ref)) + 1 for ref in citation.get("retrievedReferences", [])})
            pieces.append((self.answer[position:end], numbers))
            position = end
        if position < len(self.answer):
            pieces.append((self.answer[position:], []))
        return pieces

    def _repr_html_(self) -> str:
        from .ui.html import render_answers

        return render_answers(self)

    def __str__(self) -> str:
        from .ui.text import render_answers

        return render_answers(self)


def _chunk_key(result: Dict[str, Any]) -> Tuple[str, str]:
    """What makes a retrieved chunk the same chunk: its text and where it came from."""
    return (result.get("content", {}).get("text", ""), repr(result.get("location", {})))


def _cited_chunks(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every chunk the citations refer to, once each, in the order they're first cited."""
    seen, chunks = set(), []
    for citation in citations:
        for ref in citation.get("retrievedReferences", []):
            key = _chunk_key(ref)
            if key not in seen:
                seen.add(key)
                chunks.append(ref)
    return chunks


class KnowledgeBase:
    """A Bedrock knowledge base you can plan, build, query, sync and delete from a notebook.

    Only ``name`` and ``source`` are required. ``source`` can be ``"s3://bucket/prefix/"``,
    ``"https://site"``, an ``S3Source`` or a ``WebSource``. Everything else has a sensible
    default; see ``KBSpec`` for what each option means.

    Creating this object doesn't call AWS. Nothing is created until you call ``apply()``.
    """

    def __init__(
        self,
        name: str,
        source: Union[str, Source],
        *,
        embedding: str = "titan-v2",
        dimensions: Optional[int] = None,
        vector_store: str = "s3_vectors",
        chunking: Union[str, Chunking, None] = None,
        parsing: Union[str, Dict[str, Any], Parsing, None] = None,
        description: str = "",
        role_arn: Optional[str] = None,
        permissions_boundary: Optional[str] = None,
        region: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        sync: bool = True,
        session: Optional[boto3.Session] = None,
        **unknown: Any,
    ):
        if unknown:
            raise TypeError(_unknown_options_message(unknown))
        spec = KBSpec(
            name=name,
            source=source,  # type: ignore[arg-type]  # coerced in KBSpec.__post_init__
            embedding=embedding,
            dimensions=dimensions,
            vector_store=vector_store,
            chunking=chunking,  # type: ignore[arg-type]
            parsing=parsing,  # type: ignore[arg-type]  # coerced in KBSpec.__post_init__
            description=description,
            role_arn=role_arn,
            permissions_boundary=permissions_boundary,
            region=region,
            tags=dict(tags or {}),
            sync=sync,
        )
        self._init(spec, session)

    def _init(self, spec: KBSpec, session: Optional[boto3.Session]) -> None:
        self.spec = spec
        self._session = session
        self._aws: Optional[Aws] = None

    @classmethod
    def from_spec(cls, spec: KBSpec, session: Optional[boto3.Session] = None) -> KnowledgeBase:
        """Wrap an existing ``KBSpec``."""
        kb = cls.__new__(cls)
        kb._init(spec, session)
        return kb

    @classmethod
    def from_file(
        cls, path: Union[str, Path], region: Optional[str] = None, session: Optional[boto3.Session] = None
    ) -> KnowledgeBase:
        """Load a knowledge base definition from a ``.yaml``/``.yml`` or ``.json`` file.

        The format comes from the extension; see ``examples/kb.yaml`` and ``examples/kb.json``.
        ``region`` overrides the one in the file, handy for building the same KB elsewhere.
        """
        return cls._from_loaded(KBSpec.from_file(path), region, session)

    @classmethod
    def from_yaml(
        cls, path: Union[str, Path], region: Optional[str] = None, session: Optional[boto3.Session] = None
    ) -> KnowledgeBase:
        """Load a knowledge base definition from YAML (see ``examples/kb.yaml``)."""
        return cls._from_loaded(KBSpec.from_yaml(path), region, session)

    @classmethod
    def from_json(
        cls, path: Union[str, Path], region: Optional[str] = None, session: Optional[boto3.Session] = None
    ) -> KnowledgeBase:
        """Load a knowledge base definition from JSON (see ``examples/kb.json``)."""
        return cls._from_loaded(KBSpec.from_json(path), region, session)

    @classmethod
    def _from_loaded(cls, spec: KBSpec, region: Optional[str], session: Optional[boto3.Session]) -> KnowledgeBase:
        """Apply an optional region override to a spec read from a file, then wrap it."""
        if region:
            spec.region = region
        return cls.from_spec(spec, session)

    @classmethod
    def open(cls, name: str, region: Optional[str] = None, session: Optional[boto3.Session] = None) -> KnowledgeBase:
        """Pick up a knowledge base that's already deployed, without retyping its settings.

        Reads the knowledge base and its data source from AWS (read-only) and rebuilds the
        definition: source, embedding model, vector store, parsing and chunking. Use it in a
        fresh notebook to ``sync()``, ``ask()`` or ``destroy()`` a knowledge base built earlier,
        possibly by someone else. ``kb.save("kb.yaml")`` then writes the definition to a file.
        Run ``kbforge.knowledge_bases()`` to see the names you can open.
        """
        aws = Aws(region=region, session=session)
        with friendly_aws_errors():
            spec = open_spec(aws, name)
        kb = cls.from_spec(spec, session)
        kb._aws = aws
        return kb

    @property
    def name(self) -> str:
        """The knowledge base's name."""
        return self.spec.name

    @property
    def aws(self) -> Aws:
        """The AWS connection, created the first time something needs it."""
        if self._aws is None:
            self._aws = Aws(region=self.spec.region, session=self._session)
        return self._aws

    def plan(self) -> Plan:
        """Check everything and show what ``apply()`` would do. Never changes anything in AWS."""
        with friendly_aws_errors():
            return make_plan(self.spec, self.aws)

    def apply(self) -> Union[ApplyResult, Plan]:
        """Build (or finish building) the knowledge base, showing live progress.

        If the plan has blockers, nothing is changed and the plan is returned instead so you
        can see what needs fixing.
        """
        plan = self.plan()
        if not plan.ready:
            return plan
        return plan.apply()

    def sync(self) -> Status:
        """Re-ingest the data source (after documents were added, changed or removed)."""
        from .ui.progress import Progress

        with friendly_aws_errors():
            ctx = self._locate()
        if not ctx.out.get("ds_id"):
            raise KBForgeError(f"'{self.name}' hasn't been created yet. Run kb.plan() and kb.apply() first.")
        progress = Progress(f"Syncing {self.name}", [("Ingestion", self.spec.source.describe())], self.name)
        progress.start(0)
        try:
            job = run_ingestion(self.aws, ctx.out["kb_id"], ctx.out["ds_id"], lambda msg: progress.log(0, msg))
        except Exception as exc:
            progress.fail(0, explain_error(exc))
            raise KBForgeError(explain_error(exc)) from exc
        progress.done(0, describe_stats(job.get("statistics", {})))
        progress.close()
        return self.status()

    def ask(self, question: str, top_k: int = 5, answer: bool = False, model: Optional[str] = None) -> Answers:
        """Search the knowledge base for a question and show what comes back.

        By default this is retrieval only: the best-matching chunks with their scores and
        sources, which is the quickest way to judge whether chunking and embeddings work for
        your documents. With ``answer=True`` a model also reads those chunks and writes an
        answer, with numbered citations pointing at the chunks it used. ``model`` picks that
        model (an alias such as ``"nova-pro"``, or a Bedrock model id) and implies ``answer=True``.
        """
        if model is not None:
            answer = True
        writer = resolve_answer_model(model or DEFAULT_ANSWER_MODEL) if answer else None
        with friendly_aws_errors():
            ctx = self._locate(data_source=False)
            if not ctx.out.get("kb_id"):
                raise KBForgeError(f"'{self.name}' hasn't been created yet. Run kb.apply() first.")
            runtime = self.aws.client("bedrock-agent-runtime")
            search = {"vectorSearchConfiguration": {"numberOfResults": top_k}}
            if writer is None:
                response = runtime.retrieve(
                    knowledgeBaseId=ctx.out["kb_id"], retrievalQuery={"text": question}, retrievalConfiguration=search
                )
                return Answers(question, self.name, response.get("retrievalResults", []))
            response = runtime.retrieve_and_generate(
                input={"text": question},
                retrieveAndGenerateConfiguration={
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": ctx.out["kb_id"],
                        "modelArn": writer.arn(self.aws.region, self.aws.account_id),
                        "retrievalConfiguration": search,
                    },
                },
            )
        citations = response.get("citations", [])
        return Answers(
            question,
            self.name,
            _cited_chunks(citations),
            answer=response.get("output", {}).get("text", ""),
            citations=citations,
            model=writer.invoke_id(self.aws.region),
        )

    def status(self) -> Status:
        """Where things stand: ids, health, and how the last sync went."""
        with friendly_aws_errors():
            return self._status()

    def _status(self) -> Status:
        """``status()`` without the error translation, so the lookups read top to bottom."""
        kb = KnowledgeBaseStep().find(Ctx(self.spec, self.aws))
        if kb is None:
            return Status(self.name, self.aws.region, exists=False)
        ds = find_data_source(self.aws, kb["knowledgeBaseId"], self.spec.names.data_source)
        return make_status(self.aws, self.name, kb, ds, self.spec.source.describe())

    def parsing_models(self) -> ParsingModels:
        """The models in this knowledge base's region that can parse documents, and what to pass to use them.

        Read-only. Pick one and pass it as ``parsing=`` (e.g. ``parsing="nova-pro"``).
        """
        return list_parsing_models(self.aws)

    def destroy(self, confirm: bool = False, force: bool = False) -> Union[Plan, ApplyResult]:
        """Delete everything kbforge created for this knowledge base.

        Without ``confirm=True`` this only *shows* what would be deleted. Resources that
        weren't created by kbforge are kept unless ``force=True``. Your source documents are
        never touched.
        """
        with friendly_aws_errors():
            plan = make_plan(self.spec, self.aws, mode="destroy", force=force)
        if not confirm or not plan.ready:
            return plan
        return plan.apply()

    def to_yaml(self) -> str:
        """This knowledge base's definition as YAML."""
        return self.spec.to_yaml()

    def to_json(self) -> str:
        """This knowledge base's definition as JSON."""
        return self.spec.to_json()

    def save(self, path: Union[str, Path]) -> Path:
        """Write the definition to a file so it can be committed and reused.

        A ``.json`` path is written as JSON; ``.yaml``/``.yml`` as YAML. Load it again with
        ``KnowledgeBase.from_file(path)``.
        """
        path = Path(path)
        path.write_text(self.spec.to_text(path))
        return path

    def _locate(self, data_source: bool = True) -> Ctx:
        """Find the existing KB (and data source) ids without changing anything."""
        ctx = Ctx(self.spec, self.aws)
        kb = KnowledgeBaseStep().find(ctx)
        if kb:
            ctx.out["kb_id"] = kb["knowledgeBaseId"]
            if data_source:
                ds = DataSourceStep().find(ctx)
                if ds:
                    ctx.out["ds_id"] = ds["dataSourceId"]
        return ctx

    def __repr__(self) -> str:
        return f"KnowledgeBase({self.name!r}, source={self.spec.source.describe()!r})"

    def _repr_html_(self) -> str:
        from .ui.html import render_spec

        return render_spec(self.spec)


@contextmanager
def friendly_aws_errors() -> Iterator[None]:
    """Turn AWS/botocore exceptions raised inside the block into a ``KBForgeError`` that says what to do.

    Without this, an expired session or a missing permission shows up in a notebook as a long
    botocore stack trace. The original exception stays attached as ``__cause__`` for debugging.
    """
    try:
        yield
    except (ClientError, BotoCoreError) as exc:
        raise KBForgeError(explain_error(exc)) from exc


def _unknown_options_message(unknown: Dict[str, Any]) -> str:
    """Explain keyword arguments ``KnowledgeBase()`` doesn't take, suggesting the likely intended ones."""
    valid = [*KBSpec.__dataclass_fields__, "session"]
    hints = "".join(did_you_mean(key, valid) for key in sorted(unknown))
    names = ", ".join(sorted(unknown))
    return f"KnowledgeBase() doesn't take: {names}.{hints} Valid options: {', '.join(valid)}."
