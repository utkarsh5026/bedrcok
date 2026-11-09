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

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import boto3

from .aws import Aws
from .plan import ApplyResult, Plan, make_plan
from .spec import VECTOR_STORES, Chunking, KBSpec, Source
from .steps import Ctx, describe_stats, explain_error, latest_ingestion_job, run_ingestion
from .steps.bedrock import DataSourceStep, KnowledgeBaseStep

#: Bedrock's storage type names mapped to the labels kbforge shows.
STORE_LABELS = {
    "S3_VECTORS": VECTOR_STORES["s3_vectors"],
    "OPENSEARCH_SERVERLESS": VECTOR_STORES["opensearch_serverless"],
}


class KBForgeError(RuntimeError):
    """A problem the user can fix, with a message that says how."""


@dataclass
class Answers:
    """The chunks a knowledge base returned for a question, best match first."""

    question: str
    kb_name: str
    results: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def texts(self) -> List[str]:
        """Just the chunk texts, e.g. to paste into a prompt."""
        return [r.get("content", {}).get("text", "") for r in self.results]

    def _repr_html_(self) -> str:
        from .ui.html import render_answers

        return render_answers(self)

    def __str__(self) -> str:
        from .ui.text import render_answers

        return render_answers(self)


@dataclass
class Status:
    """A snapshot of a deployed knowledge base and its most recent sync."""

    name: str
    region: str
    exists: bool
    kb_id: str = ""
    kb_status: str = ""
    model: str = ""
    store: str = ""
    ds_id: str = ""
    source: str = ""
    last_sync: Optional[Dict[str, Any]] = None

    def rows(self) -> List[Tuple[str, str]]:
        """Label/value pairs, in display order."""
        sync = "never"
        if self.last_sync:
            job = self.last_sync
            started = job.get("startedAt")
            when = started.strftime("%Y-%m-%d %H:%M") if isinstance(started, datetime) else str(started or "")
            sync = f"{job['status'].lower()} · {when} · {describe_stats(job.get('statistics', {}))}"
        return [
            ("Knowledge base ID", self.kb_id),
            ("Region", self.region),
            ("Embedding model", self.model),
            ("Vector store", self.store),
            ("Data source ID", self.ds_id),
            ("Source", self.source),
            ("Last sync", sync),
        ]

    def _repr_html_(self) -> str:
        from .ui.html import render_status

        return render_status(self)

    def __str__(self) -> str:
        from .ui.text import render_status

        return render_status(self)


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
        description: str = "",
        role_arn: Optional[str] = None,
        permissions_boundary: Optional[str] = None,
        region: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        sync: bool = True,
        session: Optional[boto3.Session] = None,
    ):
        spec = KBSpec(
            name=name,
            source=source,  # type: ignore[arg-type]  # coerced in KBSpec.__post_init__
            embedding=embedding,
            dimensions=dimensions,
            vector_store=vector_store,
            chunking=chunking,  # type: ignore[arg-type]
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

    def ask(self, question: str, top_k: int = 5) -> Answers:
        """Search the knowledge base and show the best-matching chunks with their sources.

        This is retrieval only (no LLM answer), which is the quickest way to judge whether
        chunking and embeddings are working for your documents.
        """
        ctx = self._locate(data_source=False)
        if not ctx.out.get("kb_id"):
            raise KBForgeError(f"'{self.name}' hasn't been created yet. Run kb.apply() first.")
        response = self.aws.client("bedrock-agent-runtime").retrieve(
            knowledgeBaseId=ctx.out["kb_id"],
            retrievalQuery={"text": question},
            retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": top_k}},
        )
        return Answers(question, self.name, response.get("retrievalResults", []))

    def status(self) -> Status:
        """Where things stand: ids, health, and how the last sync went."""
        kb = KnowledgeBaseStep().find(Ctx(self.spec, self.aws))
        if kb is None:
            return Status(self.name, self.aws.region, exists=False)
        ctx = Ctx(self.spec, self.aws, out={"kb_id": kb["knowledgeBaseId"]})
        ds = DataSourceStep().find(ctx)
        model = kb.get("knowledgeBaseConfiguration", {}).get("vectorKnowledgeBaseConfiguration", {})
        store_type = kb.get("storageConfiguration", {}).get("type", "")
        return Status(
            name=self.name,
            region=self.aws.region,
            exists=True,
            kb_id=kb["knowledgeBaseId"],
            kb_status=kb["status"],
            model=model.get("embeddingModelArn", "").split("/")[-1],
            store=STORE_LABELS.get(store_type, store_type),
            ds_id=ds["dataSourceId"] if ds else "",
            source=self.spec.source.describe(),
            last_sync=latest_ingestion_job(self.aws, kb["knowledgeBaseId"], ds["dataSourceId"]) if ds else None,
        )

    def destroy(self, confirm: bool = False, force: bool = False) -> Union[Plan, ApplyResult]:
        """Delete everything kbforge created for this knowledge base.

        Without ``confirm=True`` this only *shows* what would be deleted. Resources that
        weren't created by kbforge are kept unless ``force=True``. Your source documents are
        never touched.
        """
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
