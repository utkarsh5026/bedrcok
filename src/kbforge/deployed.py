"""Reading knowledge bases that already exist in AWS back into kbforge.

kbforge keeps no state file: every AWS name comes from the KB name. That makes two everyday
questions easy to answer straight from AWS:

* "What did we build?" — ``kbforge.knowledge_bases()`` lists every knowledge base kbforge made
  in a region (found by its ``managed-by: kbforge`` tag), with its health and last sync.
* "How do I get my ``kb`` object back?" — ``KnowledgeBase.open("hr-policies")`` reads the
  deployed knowledge base and its data source and rebuilds the definition, so ``sync``, ``ask``
  and ``destroy`` work in a fresh notebook without retyping every setting.

Everything here is read-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple

from botocore.exceptions import ClientError

from .aws import Aws, error_code
from .errors import KBForgeError, did_you_mean
from .models import EMBEDDING_MODELS, PARSING_MODELS
from .spec import VECTOR_STORES, Chunking, KBSpec, Parsing, S3Source, Source, WebSource
from .steps.base import ACCESS_DENIED, KB_TAG, MANAGED_TAG, tags_if_allowed
from .steps.bedrock import describe_parsing, describe_stats, latest_ingestion_job

#: Bedrock's storage type names mapped to the labels kbforge shows.
STORE_LABELS = {
    "S3_VECTORS": VECTOR_STORES["s3_vectors"],
    "OPENSEARCH_SERVERLESS": VECTOR_STORES["opensearch_serverless"],
}

#: Bedrock's storage type names mapped to kbforge's ``vector_store`` values.
STORE_NAMES = {"S3_VECTORS": "s3_vectors", "OPENSEARCH_SERVERLESS": "opensearch_serverless"}


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
    parsing: str = ""
    last_sync: Optional[Dict[str, Any]] = None

    @property
    def last_sync_text(self) -> str:
        """The last sync as one line, e.g. ``complete · 2026-11-09 14:02 · 3 scanned · 3 indexed``."""
        if not self.last_sync:
            return "never"
        job = self.last_sync
        started = job.get("startedAt")
        when = started.strftime("%Y-%m-%d %H:%M") if isinstance(started, datetime) else str(started or "")
        parts = [job["status"].lower(), when, describe_stats(job.get("statistics", {}))]
        return " · ".join(part for part in parts if part)

    def rows(self) -> List[Tuple[str, str]]:
        """Label/value pairs, in display order."""
        return [
            ("Knowledge base ID", self.kb_id),
            ("Region", self.region),
            ("Embedding model", self.model),
            ("Vector store", self.store),
            ("Data source ID", self.ds_id),
            ("Source", self.source),
            ("Parsing", self.parsing),
            ("Last sync", self.last_sync_text),
        ]

    def _repr_html_(self) -> str:
        from .ui.html import render_status

        return render_status(self)

    def __str__(self) -> str:
        from .ui.text import render_status

        return render_status(self)


@dataclass
class KnowledgeBases:
    """The knowledge bases kbforge built in one region, as returned by ``kbforge.knowledge_bases()``.

    Shows as a table in a notebook. Iterate over it for each one's ``Status``, or use ``names``
    with ``KnowledgeBase.open(name)`` to work with one.
    """

    region: str
    items: List[Status] = field(default_factory=list)
    #: Knowledge bases whose tags couldn't be read, so kbforge can't tell whether it made them.
    unreadable: int = 0

    @property
    def names(self) -> List[str]:
        """The knowledge base names, in the order shown."""
        return [item.name for item in self.items]

    def __iter__(self) -> Iterator[Status]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def _repr_html_(self) -> str:
        from .ui.html import render_knowledge_bases

        return render_knowledge_bases(self)

    def __str__(self) -> str:
        from .ui.text import render_knowledge_bases

        return render_knowledge_bases(self)


def all_knowledge_bases(aws: Aws) -> List[Dict[str, str]]:
    """Every knowledge base in the region, as Bedrock's short summaries (id and name)."""
    agent = aws.client("bedrock-agent")
    return [
        s for page in agent.get_paginator("list_knowledge_bases").paginate() for s in page["knowledgeBaseSummaries"]
    ]


def get_knowledge_base(aws: Aws, kb_id: str) -> Dict[str, Any]:
    """The full description of one knowledge base."""
    return aws.client("bedrock-agent").get_knowledge_base(knowledgeBaseId=kb_id)["knowledgeBase"]


def find_data_source(aws: Aws, kb_id: str, name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The knowledge base's data source called ``name`` (or its first one, if no name is given)."""
    agent = aws.client("bedrock-agent")
    for page in agent.get_paginator("list_data_sources").paginate(knowledgeBaseId=kb_id):
        for summary in page.get("dataSourceSummaries", []):
            if name is None or summary["name"] == name:
                return agent.get_data_source(knowledgeBaseId=kb_id, dataSourceId=summary["dataSourceId"])["dataSource"]
    return None


def parser_label(ds: Optional[Dict[str, Any]]) -> str:
    """How a data source parses documents, for the status card ("" if there's no data source)."""
    if not ds:
        return ""
    parser = describe_parsing(ds.get("vectorIngestionConfiguration", {}).get("parsingConfiguration"))
    return "Default parser" if parser == "the default parser" else parser


def make_status(aws: Aws, name: str, kb: Dict[str, Any], ds: Optional[Dict[str, Any]], source: str) -> Status:
    """Build the ``Status`` of a deployed knowledge base from its description and its data source."""
    vector = kb.get("knowledgeBaseConfiguration", {}).get("vectorKnowledgeBaseConfiguration", {})
    store_type = kb.get("storageConfiguration", {}).get("type", "")
    return Status(
        name=name,
        region=aws.region,
        exists=True,
        kb_id=kb["knowledgeBaseId"],
        kb_status=kb["status"],
        model=vector.get("embeddingModelArn", "").split("/")[-1],
        store=STORE_LABELS.get(store_type, store_type),
        ds_id=ds["dataSourceId"] if ds else "",
        source=source,
        parsing=parser_label(ds),
        last_sync=latest_ingestion_job(aws, kb["knowledgeBaseId"], ds["dataSourceId"]) if ds else None,
    )


def list_knowledge_bases(aws: Aws) -> KnowledgeBases:
    """Every knowledge base kbforge made in this region, with its health and last sync (read-only)."""
    found = KnowledgeBases(aws.region)
    for summary in all_knowledge_bases(aws):
        kb = get_knowledge_base(aws, summary["knowledgeBaseId"])
        tags = _kb_tags(aws, kb)
        if tags is None:
            found.unreadable += 1
            continue
        if tags.get(MANAGED_TAG[0]) != MANAGED_TAG[1]:
            continue
        name = tags.get(KB_TAG) or kb["name"]
        ds = find_data_source(aws, kb["knowledgeBaseId"])
        source = _describe_source(ds)
        found.items.append(make_status(aws, name, kb, ds, source))
    found.items.sort(key=lambda status: status.name.lower())
    return found


def _kb_tags(aws: Aws, kb: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """A knowledge base's tags, or None if this role isn't allowed to read them."""
    agent = aws.client("bedrock-agent")
    return tags_if_allowed(lambda: agent.list_tags_for_resource(resourceArn=kb["knowledgeBaseArn"]).get("tags", {}))


def _describe_source(ds: Optional[Dict[str, Any]]) -> str:
    """Where a data source reads documents from, as shown in the UI ("" if unknown)."""
    if not ds:
        return ""
    try:
        return source_from_api(ds["dataSourceConfiguration"]).describe()
    except (KeyError, ValueError):
        return ds.get("dataSourceConfiguration", {}).get("type", "")


def open_spec(aws: Aws, name: str) -> KBSpec:
    """Rebuild the ``KBSpec`` of the deployed knowledge base called ``name``.

    Raises ``KBForgeError`` (suggesting the closest names) if there's no such knowledge base.
    """
    summaries = all_knowledge_bases(aws)
    match = next((s for s in summaries if s["name"] == name), None)
    if match is None:
        names = [s["name"] for s in summaries]
        raise KBForgeError(
            f"There's no knowledge base named '{name}' in {aws.region}.{did_you_mean(name, names)} "
            "Run kbforge.knowledge_bases() to see the ones kbforge built there, or pass region=... "
            "if it's in another region."
        )
    kb = get_knowledge_base(aws, match["knowledgeBaseId"])
    tags = _kb_tags(aws, kb)
    spec_name = (tags or {}).get(KB_TAG) or kb["name"]
    ds = find_data_source(aws, kb["knowledgeBaseId"], f"{spec_name}-source") or find_data_source(
        aws, kb["knowledgeBaseId"]
    )
    if ds is None:
        raise KBForgeError(
            f"'{name}' exists but has no data source, so kbforge can't tell where its documents come from. "
            "Describe it again with KnowledgeBase(name, source=...) and run kb.apply() to add one."
        )
    return spec_from_aws(aws, spec_name, kb, ds, tags or {})


def spec_from_aws(aws: Aws, name: str, kb: Dict[str, Any], ds: Dict[str, Any], tags: Dict[str, str]) -> KBSpec:
    """Turn a knowledge base's and data source's AWS descriptions back into a ``KBSpec``.

    This is the reverse of what the steps send to Bedrock, so planning the result shows
    everything as already existing.
    """
    vector = kb.get("knowledgeBaseConfiguration", {}).get("vectorKnowledgeBaseConfiguration", {})
    embedding, dimensions = _embedding_from_api(vector)
    ingestion = ds.get("vectorIngestionConfiguration", {})
    spec = KBSpec(
        name=name,
        source=source_from_api(ds["dataSourceConfiguration"]),
        embedding=embedding,
        dimensions=dimensions,
        vector_store=STORE_NAMES.get(kb.get("storageConfiguration", {}).get("type", ""), "s3_vectors"),
        chunking=chunking_from_api(ingestion.get("chunkingConfiguration")),
        parsing=parsing_from_api(ingestion.get("parsingConfiguration"), aws.region),
        description=_user_description(kb.get("description", "")),
        region=aws.region,
        tags={k: v for k, v in tags.items() if k not in (MANAGED_TAG[0], KB_TAG)},
    )
    role_arn = kb.get("roleArn", "")
    if role_arn and not role_arn.endswith(f"/{spec.names.role}"):
        spec.role_arn = role_arn  # the user's own role, not one kbforge generated
    elif role_arn:
        spec.permissions_boundary = _permissions_boundary(aws, spec.names.role)
    return spec


def _embedding_from_api(vector: Dict[str, Any]) -> Tuple[str, Optional[int]]:
    """The ``embedding`` alias (or raw model id) and non-default ``dimensions`` of a knowledge base."""
    model_id = vector.get("embeddingModelArn", "").split("/")[-1]
    dims = vector.get("embeddingModelConfiguration", {}).get("bedrockEmbeddingModelConfiguration", {}).get("dimensions")
    known = next((m for m in EMBEDDING_MODELS.values() if m.model_id == model_id), None)
    if known is None:
        if not dims:
            raise KBForgeError(
                f"This knowledge base uses {model_id}, which kbforge doesn't know, and its vector size isn't "
                "recorded. Describe it with KnowledgeBase(name, source=..., embedding=..., dimensions=...) instead."
            )
        return model_id, dims
    return known.alias, dims if dims and dims != known.default_dimensions else None


def source_from_api(config: Dict[str, Any]) -> Source:
    """The reverse of ``S3Source.to_api`` / ``WebSource.to_api``."""
    if config.get("type") == "S3":
        s3 = config["s3Configuration"]
        bucket = s3["bucketArn"].split(":::", 1)[-1]
        prefixes = s3.get("inclusionPrefixes") or [""]
        return S3Source(f"s3://{bucket}/{prefixes[0]}")
    if config.get("type") == "WEB":
        web = config["webConfiguration"]
        crawler = web.get("crawlerConfiguration", {})
        limits = crawler.get("crawlerLimits", {})
        return WebSource(
            urls=[seed["url"] for seed in web["sourceConfiguration"]["urlConfiguration"]["seedUrls"]],
            scope=crawler.get("scope", "HOST_ONLY"),
            max_pages=limits.get("maxPages"),
            rate_limit=limits.get("rateLimit", 300),
            include=list(crawler.get("inclusionFilters", [])),
            exclude=list(crawler.get("exclusionFilters", [])),
        )
    raise KBForgeError(
        f"This knowledge base reads from a {config.get('type', 'unknown')} source, which kbforge doesn't manage. "
        "kbforge can open knowledge bases whose source is S3 or a website."
    )


def chunking_from_api(config: Optional[Dict[str, Any]]) -> Chunking:
    """The reverse of ``Chunking.to_api``."""
    strategy = (config or {}).get("chunkingStrategy", "FIXED_SIZE")
    if strategy == "FIXED_SIZE":
        fixed = config.get("fixedSizeChunkingConfiguration", {}) if config else {}
        return Chunking.fixed(fixed.get("maxTokens", 300), fixed.get("overlapPercentage", 20))
    if strategy == "HIERARCHICAL":
        hier = config.get("hierarchicalChunkingConfiguration", {}) if config else {}
        levels = hier.get("levelConfigurations", [{}, {}])
        return Chunking.hierarchical(levels[0]["maxTokens"], levels[1]["maxTokens"], hier.get("overlapTokens", 60))
    if strategy == "SEMANTIC":
        sem = config.get("semanticChunkingConfiguration", {}) if config else {}
        return Chunking.semantic(
            sem.get("maxTokens", 300), sem.get("bufferSize", 0), sem.get("breakpointPercentileThreshold", 95)
        )
    return Chunking.none()


def parsing_from_api(config: Optional[Dict[str, Any]], region: str) -> Parsing:
    """The reverse of ``Parsing.to_api``, naming the model by its alias when kbforge knows it."""
    if not config:
        return Parsing.default()
    if config.get("parsingStrategy") == "BEDROCK_DATA_AUTOMATION":
        return Parsing.data_automation()
    fm = config.get("bedrockFoundationModelConfiguration", {})
    model = fm.get("modelArn", "").split("/")[-1]
    alias = next((m.alias for m in PARSING_MODELS.values() if model == m.invoke_id(region)), model)
    prompt = fm.get("parsingPrompt", {}).get("parsingPromptText")
    return Parsing.foundation_model(alias, prompt=prompt)


def _user_description(description: str) -> str:
    """The description the user gave, or "" if it's the one kbforge writes when none was given."""
    return "" if description.startswith("Created with kbforge from ") else description


def _permissions_boundary(aws: Aws, role_name: str) -> Optional[str]:
    """The permissions boundary on kbforge's role, if any (None when it can't be read)."""
    try:
        role = aws.client("iam").get_role(RoleName=role_name)["Role"]
    except ClientError as exc:
        if error_code(exc) in (*ACCESS_DENIED, "NoSuchEntity"):
            return None
        raise
    return role.get("PermissionsBoundary", {}).get("PermissionsBoundaryArn")
