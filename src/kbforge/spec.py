"""What the user wants, written down as data.

A ``KBSpec`` describes the knowledge base you'd like to end up with — its name, where the
documents are, which embedding model and vector store to use, how to chunk. It says nothing
about *how* to build it; that's the job of the steps in ``kbforge.steps``.

Specs can be built in Python or loaded from a YAML or JSON file (``KBSpec.from_file("kb.yaml")``),
so a team can keep its knowledge-base definitions in git.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .models import EmbeddingModel, resolve_model

#: Vector stores kbforge can set up, with the name shown in the UI.
VECTOR_STORES = {
    "s3_vectors": "Amazon S3 Vectors",
    "opensearch_serverless": "OpenSearch Serverless",
}

#: Friendly spellings people are likely to type, mapped to the canonical name.
_VECTOR_STORE_ALIASES = {
    "s3_vectors": "s3_vectors",
    "s3-vectors": "s3_vectors",
    "s3vectors": "s3_vectors",
    "opensearch_serverless": "opensearch_serverless",
    "opensearch-serverless": "opensearch_serverless",
    "opensearch": "opensearch_serverless",
    "aoss": "opensearch_serverless",
}

#: The settings each chunking strategy uses. A config file may only set the ones listed for
#: its strategy, so a typo or a setting borrowed from another strategy is reported, not ignored.
CHUNKING_SETTINGS = {
    "fixed": ("max_tokens", "overlap_percent"),
    "hierarchical": ("parent_tokens", "child_tokens", "overlap_tokens"),
    "semantic": ("max_tokens", "buffer_size", "breakpoint_threshold"),
    "none": (),
}

#: The range Bedrock's CreateDataSource API accepts for each chunking setting (``None`` means no
#: upper limit). Checking them here means a bad value stops the plan instead of failing apply
#: halfway through, after the vector store, role and knowledge base were already created.
CHUNKING_LIMITS: Dict[str, Tuple[int, Optional[int]]] = {
    "max_tokens": (1, 8192),
    "overlap_percent": (1, 99),
    "parent_tokens": (1, 8192),
    "child_tokens": (1, 8192),
    "overlap_tokens": (1, None),
    "buffer_size": (0, 1),
    "breakpoint_threshold": (50, 99),
}

#: File extensions kbforge reads and writes as config, mapped to their format.
CONFIG_FORMATS = {".yaml": "yaml", ".yml": "yaml", ".json": "json"}

#: Bedrock's rule for knowledge-base names: letters/digits, single '-' or '_' between them.
KB_NAME_RE = re.compile(r"^([0-9a-zA-Z][_-]?){1,100}$")


@dataclass
class S3Source:
    """Documents stored in S3, optionally limited to one folder (prefix).

    Example: ``S3Source("s3://my-company-docs/hr/")`` ingests everything under ``hr/``.
    """

    uri: str

    kind = "s3"

    def __post_init__(self) -> None:
        if not self.uri.startswith("s3://") or len(self.uri) <= 5:
            raise ValueError(f"S3 source must look like s3://bucket/optional/prefix, got {self.uri!r}")

    @property
    def bucket(self) -> str:
        """The bucket name, e.g. ``my-company-docs``."""
        return self.uri[5:].split("/", 1)[0]

    @property
    def prefix(self) -> str:
        """The folder inside the bucket, e.g. ``hr/`` (empty string means the whole bucket)."""
        parts = self.uri[5:].split("/", 1)
        return parts[1] if len(parts) > 1 else ""

    def describe(self) -> str:
        """Short text for the UI."""
        return self.uri

    def to_api(self) -> Dict[str, Any]:
        """The ``dataSourceConfiguration`` block Bedrock's CreateDataSource expects."""
        config: Dict[str, Any] = {"bucketArn": f"arn:aws:s3:::{self.bucket}"}
        if self.prefix:
            config["inclusionPrefixes"] = [self.prefix]
        return {"type": "S3", "s3Configuration": config}

    def to_dict(self) -> Dict[str, Any]:
        """How this source is written in a YAML config."""
        return {"s3": self.uri}


@dataclass
class WebSource:
    """Public web pages, fetched by Bedrock's managed web crawler.

    Example: ``WebSource(["https://docs.example.com"], max_pages=500)``.
    Only crawl sites you own or are allowed to crawl.
    """

    urls: List[str]
    #: ``HOST_ONLY`` stays on the same host; ``SUBDOMAINS`` also follows links to subdomains.
    scope: str = "HOST_ONLY"
    max_pages: Optional[int] = None
    #: Pages per minute per host. Keep it polite.
    rate_limit: int = 300
    #: Regex patterns; only URLs matching ``include`` (and not ``exclude``) are crawled.
    include: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)

    kind = "web"

    def __post_init__(self) -> None:
        if isinstance(self.urls, str):
            self.urls = [self.urls]
        if not self.urls:
            raise ValueError("WebSource needs at least one seed URL")
        for url in self.urls:
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"Seed URL must start with http(s)://, got {url!r}")
        if self.scope not in ("HOST_ONLY", "SUBDOMAINS"):
            raise ValueError("scope must be 'HOST_ONLY' or 'SUBDOMAINS'")

    def describe(self) -> str:
        """Short text for the UI: the first URL plus a count of the rest."""
        more = f" (+{len(self.urls) - 1} more)" if len(self.urls) > 1 else ""
        return self.urls[0] + more

    def to_api(self) -> Dict[str, Any]:
        """The ``dataSourceConfiguration`` block Bedrock's CreateDataSource expects."""
        limits: Dict[str, Any] = {"rateLimit": self.rate_limit}
        if self.max_pages:
            limits["maxPages"] = self.max_pages
        crawler: Dict[str, Any] = {"crawlerLimits": limits, "scope": self.scope}
        if self.include:
            crawler["inclusionFilters"] = self.include
        if self.exclude:
            crawler["exclusionFilters"] = self.exclude
        return {
            "type": "WEB",
            "webConfiguration": {
                "sourceConfiguration": {"urlConfiguration": {"seedUrls": [{"url": u} for u in self.urls]}},
                "crawlerConfiguration": crawler,
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """How this source is written in a YAML config."""
        data: Dict[str, Any] = {"web": list(self.urls), "scope": self.scope}
        if self.max_pages:
            data["max_pages"] = self.max_pages
        return data


Source = Union[S3Source, WebSource]


def coerce_source(value: Any) -> Source:
    """Accept the many ways people write a source and return a proper Source object.

    All of these work: ``"s3://bucket/prefix"``, ``"https://site"``, ``{"s3": "..."}``,
    ``{"web": [...]}``, or an ``S3Source``/``WebSource`` instance.
    """
    if isinstance(value, (S3Source, WebSource)):
        return value
    if isinstance(value, str):
        if value.startswith("s3://"):
            return S3Source(value)
        if value.startswith(("http://", "https://")):
            return WebSource([value])
    if isinstance(value, dict):
        if "s3" in value:
            return S3Source(value["s3"])
        if "web" in value:
            options = {k: v for k, v in value.items() if k != "web"}
            return WebSource(urls=value["web"], **options)
    raise ValueError(
        f"Don't know how to read data source {value!r}. "
        "Use 's3://bucket/prefix', 'https://site', S3Source(...) or WebSource(...)."
    )


@dataclass
class Chunking:
    """How documents are cut into pieces before each piece is embedded.

    Use the helpers rather than the constructor:

    * ``Chunking.fixed(300, 20)`` — equal-sized pieces with some overlap. A safe default.
    * ``Chunking.hierarchical()`` — small pieces for search, returned with their larger parent
      for context. Good for long, structured documents.
    * ``Chunking.semantic()`` — split where the meaning changes. Better quality, slower and
      costs extra model calls during sync.
    * ``Chunking.none()`` — each file is one piece. Only for files that are already small.
    """

    strategy: str = "fixed"
    max_tokens: int = 300
    overlap_percent: int = 20
    parent_tokens: int = 1500
    child_tokens: int = 300
    overlap_tokens: int = 60
    buffer_size: int = 0
    breakpoint_threshold: int = 95

    @classmethod
    def fixed(cls, max_tokens: int = 300, overlap_percent: int = 20) -> Chunking:
        """Equal-sized chunks of ``max_tokens`` that overlap by ``overlap_percent``."""
        return cls("fixed", max_tokens=max_tokens, overlap_percent=overlap_percent)

    @classmethod
    def hierarchical(cls, parent_tokens: int = 1500, child_tokens: int = 300, overlap_tokens: int = 60) -> Chunking:
        """Search over small child chunks, but return their bigger parent chunk as context."""
        return cls(
            "hierarchical", parent_tokens=parent_tokens, child_tokens=child_tokens, overlap_tokens=overlap_tokens
        )

    @classmethod
    def semantic(cls, max_tokens: int = 300, buffer_size: int = 0, breakpoint_threshold: int = 95) -> Chunking:
        """Split at points where consecutive sentences stop being about the same thing."""
        return cls(
            "semantic", max_tokens=max_tokens, buffer_size=buffer_size, breakpoint_threshold=breakpoint_threshold
        )

    @classmethod
    def none(cls) -> Chunking:
        """Don't split: every file becomes exactly one chunk."""
        return cls("none")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Chunking:
        """Build chunking from a config-file block such as ``{"strategy": "semantic", "max_tokens": 400}``.

        Settings left out keep their defaults. Settings that don't belong to the chosen strategy
        (``overlap_percent`` on semantic chunking, say) raise an error naming the ones that do,
        because Bedrock would otherwise ignore them without telling anyone.
        """
        data = dict(data)
        strategy = data.pop("strategy", "fixed")
        allowed = CHUNKING_SETTINGS.get(strategy)
        if allowed is None:
            raise ValueError(f"chunking strategy must be one of: {', '.join(CHUNKING_SETTINGS)}, got {strategy!r}")
        unexpected = sorted(set(data) - set(allowed))
        if unexpected:
            hint = f"its settings are: {', '.join(allowed)}" if allowed else "it takes no other settings"
            raise ValueError(f"{', '.join(unexpected)} can't be used with {strategy} chunking; {hint}.")
        return cls(strategy, **data)

    def __post_init__(self) -> None:
        if self.strategy not in CHUNKING_SETTINGS:
            raise ValueError(f"chunking strategy must be one of: {', '.join(CHUNKING_SETTINGS)}")
        for key in CHUNKING_SETTINGS[self.strategy]:
            value = getattr(self, key)
            low, high = CHUNKING_LIMITS[key]
            allowed = f"between {low} and {high}" if high is not None else f"at least {low}"
            # bool is a subclass of int, but `max_tokens: true` is a mistake, not a number.
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"chunking {key} must be a whole number {allowed}, got {value!r}.")
            if value < low or (high is not None and value > high):
                raise ValueError(f"chunking {key} must be {allowed}, got {value}.")
        if self.strategy == "hierarchical":
            if self.child_tokens >= self.parent_tokens:
                raise ValueError(
                    f"child_tokens ({self.child_tokens}) must be smaller than parent_tokens ({self.parent_tokens}): "
                    "the small child chunks are searched and their bigger parent is returned."
                )
            if self.overlap_tokens >= self.child_tokens:
                raise ValueError(
                    f"overlap_tokens ({self.overlap_tokens}) must be smaller than child_tokens ({self.child_tokens}), "
                    "or neighbouring chunks would be all overlap."
                )

    @property
    def embedded_tokens(self) -> Optional[int]:
        """The largest chunk that gets sent to the embedding model, or None if there's no limit.

        For hierarchical chunking only the small child chunks are embedded; the parent is just
        returned as context. With no chunking a whole file is one chunk, so there's no fixed size.
        """
        if self.strategy == "hierarchical":
            return self.child_tokens
        if self.strategy == "none":
            return None
        return self.max_tokens

    def describe(self) -> str:
        """Short text for the UI, e.g. ``Fixed size · 300 tokens, 20% overlap``."""
        if self.strategy == "fixed":
            return f"Fixed size · {self.max_tokens} tokens, {self.overlap_percent}% overlap"
        if self.strategy == "hierarchical":
            return f"Hierarchical · parent {self.parent_tokens} / child {self.child_tokens} tokens"
        if self.strategy == "semantic":
            return f"Semantic · up to {self.max_tokens} tokens, breakpoint p{self.breakpoint_threshold}"
        return "None · each file is one chunk"

    def to_api(self) -> Dict[str, Any]:
        """The ``chunkingConfiguration`` block Bedrock's CreateDataSource expects."""
        if self.strategy == "fixed":
            return {
                "chunkingStrategy": "FIXED_SIZE",
                "fixedSizeChunkingConfiguration": {
                    "maxTokens": self.max_tokens,
                    "overlapPercentage": self.overlap_percent,
                },
            }
        if self.strategy == "hierarchical":
            return {
                "chunkingStrategy": "HIERARCHICAL",
                "hierarchicalChunkingConfiguration": {
                    "levelConfigurations": [{"maxTokens": self.parent_tokens}, {"maxTokens": self.child_tokens}],
                    "overlapTokens": self.overlap_tokens,
                },
            }
        if self.strategy == "semantic":
            return {
                "chunkingStrategy": "SEMANTIC",
                "semanticChunkingConfiguration": {
                    "maxTokens": self.max_tokens,
                    "bufferSize": self.buffer_size,
                    "breakpointPercentileThreshold": self.breakpoint_threshold,
                },
            }
        return {"chunkingStrategy": "NONE"}

    def to_dict(self) -> Dict[str, Any]:
        """How this chunking is written in a config file (only the settings its strategy uses)."""
        relevant = CHUNKING_SETTINGS[self.strategy]
        return {"strategy": self.strategy, **{key: getattr(self, key) for key in relevant}}


def coerce_chunking(value: Any) -> Chunking:
    """Accept ``None``, a strategy name like ``"semantic"``, a dict from a config file, or a Chunking."""
    if value is None:
        return Chunking.fixed()
    if isinstance(value, Chunking):
        return value
    if isinstance(value, str):
        presets: Dict[str, Callable[[], Chunking]] = {
            "fixed": Chunking.fixed,
            "default": Chunking.fixed,
            "hierarchical": Chunking.hierarchical,
            "semantic": Chunking.semantic,
            "none": Chunking.none,
        }
        if value not in presets:
            raise ValueError(f"Unknown chunking {value!r}. Use one of: {', '.join(presets)}")
        return presets[value]()
    if isinstance(value, dict):
        return Chunking.from_dict(value)
    raise ValueError(f"Can't read chunking config {value!r}")


def _fit(name: str, limit: int) -> str:
    """Shorten a generated name to an AWS length limit, adding a short hash so it stays unique."""
    if len(name) <= limit:
        return name
    digest = hashlib.sha1(name.encode()).hexdigest()[:6]
    return name[: limit - 7].rstrip("-_") + "-" + digest


@dataclass
class Names:
    """Every AWS resource name kbforge will use for one knowledge base.

    They're all derived from the KB name, so running the same spec twice finds the same
    resources instead of creating duplicates. No state file needed.
    """

    kb: str
    data_source: str
    role: str
    role_policy: str
    vector_bucket: str
    vector_index: str
    collection: str
    enc_policy: str
    net_policy: str
    access_policy: str
    aoss_index: str = "bedrock-kb-index"


@dataclass
class KBSpec:
    """Everything needed to describe one knowledge base.

    Only ``name`` and ``source`` are required; everything else has a sensible default:
    Titan v2 embeddings, S3 Vectors storage, fixed 300-token chunks, and a first sync
    right after creation.
    """

    name: str
    source: Source
    embedding: str = "titan-v2"
    dimensions: Optional[int] = None
    vector_store: str = "s3_vectors"
    chunking: Chunking = field(default_factory=Chunking.fixed)
    description: str = ""
    #: Reuse an existing IAM role instead of creating one (common when IAM is locked down).
    role_arn: Optional[str] = None
    #: Attach this permissions boundary to the role we create, if your org requires one.
    permissions_boundary: Optional[str] = None
    region: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    #: Run the first ingestion as part of ``apply()``.
    sync: bool = True

    def __post_init__(self) -> None:
        self.source = coerce_source(self.source)
        self.chunking = coerce_chunking(self.chunking)
        store = _VECTOR_STORE_ALIASES.get(str(self.vector_store).lower())
        if store is None:
            raise ValueError(f"vector_store must be one of {list(VECTOR_STORES)}, got {self.vector_store!r}")
        self.vector_store = store
        if not KB_NAME_RE.match(self.name):
            raise ValueError(
                f"Knowledge base name {self.name!r} is invalid: use letters, digits, '-' or '_' "
                "(no two separators in a row, max 100 chars)."
            )
        model = self.model
        if self.dimensions is not None and self.dimensions not in model.dimensions:
            raise ValueError(f"{model.label} supports dimensions {list(model.dimensions)}, not {self.dimensions}")

    @property
    def model(self) -> EmbeddingModel:
        """The embedding model this spec uses (raises if the name isn't recognised)."""
        return resolve_model(self.embedding, self.dimensions)

    @property
    def dims(self) -> int:
        """Vector size: what the user asked for, or the model's default."""
        return self.dimensions or self.model.default_dimensions

    @property
    def slug(self) -> str:
        """The KB name made safe for resources that only allow lowercase letters, digits and '-'."""
        slug = re.sub(r"[^a-z0-9-]+", "-", self.name.lower()).strip("-")
        return slug if slug[:1].isalpha() else "kb-" + slug

    @property
    def names(self) -> Names:
        """The concrete AWS names for every resource, each trimmed to its service's length limit."""
        slug = self.slug
        return Names(
            kb=self.name,
            data_source=_fit(f"{self.name}-source", 100),
            # The "AmazonBedrock" prefix matters: AWS's managed Bedrock policies only allow
            # iam:PassRole on roles whose name contains it.
            role=_fit(f"AmazonBedrockExecutionRoleForKnowledgeBase_kbf-{slug}", 64),
            role_policy="kbforge-access",
            vector_bucket=_fit(f"kbforge-{slug}", 63),
            vector_index=_fit(f"{slug}-index", 63),
            collection=_fit(f"kbf-{slug}", 32),
            enc_policy=_fit(f"kbf-{slug}-enc", 32),
            net_policy=_fit(f"kbf-{slug}-net", 32),
            access_policy=_fit(f"kbf-{slug}-access", 32),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KBSpec:
        """Build a spec from a plain dict (e.g. a parsed YAML or JSON file), with a clear error for typos."""
        data = dict(data)
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}. Valid keys: {sorted(known)}")
        return cls(**data)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> KBSpec:
        """Load a spec from a ``.yaml``/``.yml`` or ``.json`` file, picking the format from the extension.

        See ``examples/kb.yaml`` and ``examples/kb.json``; both formats take exactly the same keys.
        """
        return cls._load(path, config_format(path))

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> KBSpec:
        """Load a spec from a YAML file, whatever its extension. See ``examples/kb.yaml``."""
        return cls._load(path, "yaml")

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> KBSpec:
        """Load a spec from a JSON file, whatever its extension. See ``examples/kb.json``."""
        return cls._load(path, "json")

    @classmethod
    def _load(cls, path: Union[str, Path], fmt: str) -> KBSpec:
        """Read and parse a config file, turning parse problems into errors that point at the line."""
        path = Path(path)
        text = path.read_text()
        if fmt == "json":
            try:
                data = json.loads(text) if text.strip() else {}
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path} isn't valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})."
                ) from exc
        else:
            import yaml

            try:
                data = yaml.safe_load(text) or {}
            except yaml.YAMLError as exc:
                raise ValueError(f"{path} isn't valid YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{path} should hold one object with keys such as name and source.")
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """The spec as a plain dict, leaving out values that are just defaults."""
        data: Dict[str, Any] = {
            "name": self.name,
            "source": self.source.to_dict(),
            "embedding": self.embedding,
            "vector_store": self.vector_store,
            "chunking": self.chunking.to_dict(),
        }
        for key in ("dimensions", "description", "role_arn", "permissions_boundary", "region"):
            if getattr(self, key):
                data[key] = getattr(self, key)
        if self.tags:
            data["tags"] = dict(self.tags)
        if not self.sync:
            data["sync"] = False
        return data

    def to_yaml(self) -> str:
        """The spec as YAML text, ready to save next to your project."""
        import yaml

        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def to_json(self) -> str:
        """The spec as pretty-printed JSON text, for teams and tools that prefer JSON to YAML."""
        return json.dumps(self.to_dict(), indent=2) + "\n"

    def to_text(self, path: Union[str, Path]) -> str:
        """The spec in whichever format ``path``'s extension asks for (YAML or JSON)."""
        return self.to_json() if config_format(path) == "json" else self.to_yaml()


def config_format(path: Union[str, Path]) -> str:
    """``"yaml"`` or ``"json"`` for a config path, judged by its extension.

    Raises a friendly error for anything else, rather than guessing and failing on the parse.
    """
    suffix = Path(path).suffix.lower()
    if suffix not in CONFIG_FORMATS:
        raise ValueError(f"Can't tell the format of {str(path)!r}: config files must end in .yaml, .yml or .json.")
    return CONFIG_FORMATS[suffix]
