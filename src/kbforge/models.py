"""The models kbforge knows how to set up: embedding models and parsing models.

An embedding model turns a chunk of text into a list of numbers (a vector) so that chunks
with similar meaning end up close together. The knowledge base uses the same model for your
documents and for every question, so it can't be changed after the KB is created.

A parsing model is optional. Bedrock's default parser only pulls the plain text out of each
file; with a parsing model, a vision-capable model reads every PDF page instead, so tables,
charts, images and scanned pages become searchable text too.

An answer model is also optional. ``kb.ask(question, answer=True)`` has it read the retrieved
chunks and write an answer that cites them, instead of just showing the chunks.

Users pick a model by a short alias (``"titan-v2"``, ``"nova-lite"``) instead of a long model id.
To see which parsing models your own account can use, call ``kbforge.parsing_models()``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .errors import did_you_mean


@dataclass(frozen=True)
class EmbeddingModel:
    """One embedding model: its id, a friendly label, and the vector sizes it supports."""

    alias: str
    model_id: str
    label: str
    dimensions: Tuple[int, ...]
    default_dimensions: int
    note: str = ""
    #: The most tokens the model reads from one chunk; anything longer is cut off before
    #: embedding. ``None`` when we don't know (a raw model id kbforge has no entry for).
    max_input_tokens: Optional[int] = None

    @property
    def configurable_dimensions(self) -> bool:
        """True if you can choose the vector size (only Titan v2 today)."""
        return len(self.dimensions) > 1

    def arn(self, region: str) -> str:
        """The model's ARN in a given region, as Bedrock's APIs expect it."""
        return f"arn:aws:bedrock:{region}::foundation-model/{self.model_id}"

    def probe_body(self) -> str:
        """A tiny request body used by the preflight check to prove the model is invocable."""
        if self.model_id.startswith("cohere."):
            return json.dumps({"texts": ["hello"], "input_type": "search_document"})
        body: dict = {"inputText": "hello"}
        if self.model_id == "amazon.titan-embed-text-v2:0":
            body["dimensions"] = 256
        return json.dumps(body)


#: Every model kbforge supports out of the box, keyed by alias.
EMBEDDING_MODELS = {
    m.alias: m
    for m in [
        EmbeddingModel(
            "titan-v2",
            "amazon.titan-embed-text-v2:0",
            "Amazon Titan Text Embeddings V2",
            (256, 512, 1024),
            1024,
            "Good default. Cheap, multilingual, 8k token input.",
            max_input_tokens=8192,
        ),
        EmbeddingModel(
            "cohere-en",
            "cohere.embed-english-v3",
            "Cohere Embed English v3",
            (1024,),
            1024,
            "Strong retrieval quality for English-only corpora.",
            max_input_tokens=512,
        ),
        EmbeddingModel(
            "cohere-multi",
            "cohere.embed-multilingual-v3",
            "Cohere Embed Multilingual v3",
            (1024,),
            1024,
            "Use when documents mix many languages.",
            max_input_tokens=512,
        ),
        EmbeddingModel(
            "titan-v1",
            "amazon.titan-embed-text-v1",
            "Amazon Titan Embeddings G1 - Text",
            (1536,),
            1536,
            "Legacy. Prefer titan-v2 for new knowledge bases.",
            max_input_tokens=8192,
        ),
    ]
}


def resolve_model(name: str, dimensions: Optional[int] = None) -> EmbeddingModel:
    """Accept an alias ("titan-v2") or a raw model id ("amazon.titan-embed-text-v2:0")."""
    if name in EMBEDDING_MODELS:
        return EMBEDDING_MODELS[name]
    for m in EMBEDDING_MODELS.values():
        if m.model_id == name:
            return m
    if dimensions is None:
        known = ", ".join(EMBEDDING_MODELS)
        raise ValueError(
            f"Unknown embedding model {name!r}. Use one of: {known}, "
            f"or pass a raw model id together with dimensions=....{did_you_mean(name, EMBEDDING_MODELS)}"
        )
    return EmbeddingModel(name, name, name, (dimensions,), dimensions)


#: Region prefixes of Bedrock's cross-region inference profiles, as in ``us.amazon.nova-lite-v1:0``.
PROFILE_PREFIXES = ("us", "eu", "apac", "global", "us-gov", "jp", "au", "ca")

#: What a model id has to look like: ``provider.model-name`` with an optional ``:version``.
_MODEL_ID_RE = re.compile(r"^[a-z0-9-]+(\.[A-Za-z0-9-]+)+(:[A-Za-z0-9-]+)*$")


def profile_geo(region: str) -> str:
    """The inference-profile prefix for a region's geography, e.g. ``eu`` for ``eu-west-1``."""
    if region.startswith("us-gov"):
        return "us-gov"
    if region.startswith("eu-"):
        return "eu"
    if region.startswith("ap-"):
        return "apac"
    return "us"


@dataclass(frozen=True)
class ParsingModel:
    """A model Bedrock can use to read documents, page by page, before they're chunked.

    Most newer models can only be called through a cross-region *inference profile*: an id
    like ``us.amazon.nova-lite-v1:0`` that lets Bedrock run each request in whichever region of
    that geography has capacity. ``profile`` says how this model is called.
    """

    alias: str
    #: The foundation model id, without any profile prefix.
    model_id: str
    label: str
    note: str = ""
    #: ``"auto"`` uses the profile for your region's geography (``us.``, ``eu.``, ``apac.``);
    #: ``""`` calls the model directly, without a profile; anything else is that exact prefix.
    profile: str = "auto"

    def invoke_id(self, region: str) -> str:
        """The id to pass as ``modelId``: the profile id, or the plain model id."""
        if not self.profile:
            return self.model_id
        prefix = profile_geo(region) if self.profile == "auto" else self.profile
        return f"{prefix}.{self.model_id}"

    def arn(self, region: str, account: str) -> str:
        """The ARN Bedrock's data source expects for this parser in a given region and account."""
        if not self.profile:
            return f"arn:aws:bedrock:{region}::foundation-model/{self.model_id}"
        return f"arn:aws:bedrock:{region}:{account}:inference-profile/{self.invoke_id(region)}"

    def iam_resources(self, region: str, account: str) -> List[str]:
        """What the knowledge base's role must be allowed to invoke to use this parser.

        A profile sends each request to one of several regions, so the role needs the model
        itself in every region as well as the profile.
        """
        if not self.profile:
            return [self.arn(region, account)]
        return [self.arn(region, account), f"arn:aws:bedrock:*::foundation-model/{self.model_id}"]


#: Parsing models kbforge knows by alias. Bedrock accepts Claude, Nova and Llama 4 vision models;
#: any of them can also be given by its model id (see ``kbforge.parsing_models()``).
PARSING_MODELS = {
    m.alias: m
    for m in [
        ParsingModel(
            "nova-lite",
            "amazon.nova-lite-v1:0",
            "Amazon Nova Lite",
            "Good default. Cheap and quick; handles tables and simple charts.",
        ),
        ParsingModel(
            "nova-pro",
            "amazon.nova-pro-v1:0",
            "Amazon Nova Pro",
            "More accurate on dense layouts and charts. Costs more than Lite.",
        ),
        ParsingModel(
            "claude-haiku-4.5",
            "anthropic.claude-haiku-4-5-20251001-v1:0",
            "Claude Haiku 4.5",
            "Fast Claude model. Strong on tables and forms.",
        ),
        ParsingModel(
            "claude-sonnet-4",
            "anthropic.claude-sonnet-4-20250514-v1:0",
            "Claude Sonnet 4",
            "Best on complex tables, charts and scans. The most expensive here.",
        ),
        ParsingModel(
            "llama4-maverick",
            "meta.llama4-maverick-17b-instruct-v1:0",
            "Llama 4 Maverick 17B",
            "Open-weight alternative from Meta.",
        ),
    ]
}

DEFAULT_PARSING_MODEL = "nova-lite"


def is_parser_family(model_id: str) -> bool:
    """True for the model families Bedrock accepts as a knowledge-base parser (Claude, Nova, Llama 4)."""
    return any(family in model_id for family in ("anthropic.claude", "amazon.nova", "meta.llama4"))


def resolve_parsing_model(name: str) -> ParsingModel:
    """Accept an alias (``"nova-lite"``), a model id, or an inference profile id (``"us.amazon.nova-lite-v1:0"``)."""
    if name in PARSING_MODELS:
        return PARSING_MODELS[name]
    # A plain model id means "call the model directly"; a prefixed one names a profile.
    if not _MODEL_ID_RE.match(name or ""):
        raise ValueError(
            f"Unknown parsing model {name!r}. Use one of: {', '.join(PARSING_MODELS)}, or a Bedrock model id "
            "such as 'anthropic.claude-3-haiku-20240307-v1:0'. Run kbforge.parsing_models() to see what your "
            f"account can use.{did_you_mean(name, PARSING_MODELS)}"
        )
    prefix, _, rest = name.partition(".")
    profile = prefix if prefix in PROFILE_PREFIXES else ""
    base = rest if profile else name
    if not is_parser_family(base):
        raise ValueError(
            f"{name!r} can't parse documents: Bedrock only accepts Claude, Amazon Nova and Llama 4 vision "
            "models as parsers. Run kbforge.parsing_models() to see which ones your account can use."
        )
    known = next((m for m in PARSING_MODELS.values() if m.model_id == base), None)
    label = known.label if known else name
    return ParsingModel(name, base, label, known.note if known else "", profile=profile)


#: Models kbforge knows by alias for writing an answer from retrieved chunks (``kb.ask(..., answer=True)``).
#: They're called through the region's cross-region inference profile, like parsing models.
ANSWER_MODELS = {
    m.alias: m
    for m in [
        ParsingModel("nova-lite", "amazon.nova-lite-v1:0", "Amazon Nova Lite", "Good default. Cheap and quick."),
        ParsingModel("nova-pro", "amazon.nova-pro-v1:0", "Amazon Nova Pro", "Better at long or tricky questions."),
        ParsingModel(
            "claude-haiku-4.5",
            "anthropic.claude-haiku-4-5-20251001-v1:0",
            "Claude Haiku 4.5",
            "Fast, careful answers that stick to the sources.",
        ),
        ParsingModel(
            "claude-sonnet-4",
            "anthropic.claude-sonnet-4-20250514-v1:0",
            "Claude Sonnet 4",
            "The strongest writer here, and the most expensive.",
        ),
    ]
}

#: The model ``kb.ask(..., answer=True)`` uses unless told otherwise.
DEFAULT_ANSWER_MODEL = "nova-lite"


def resolve_answer_model(name: str) -> ParsingModel:
    """Accept an alias (``"nova-pro"``), a model id, or an inference profile id for writing answers.

    Any Bedrock text model works here, so beyond the aliases this only checks the id's shape;
    Bedrock itself says if the model can't be used.
    """
    if name in ANSWER_MODELS:
        return ANSWER_MODELS[name]
    if not _MODEL_ID_RE.match(name or ""):
        raise ValueError(
            f"Unknown answer model {name!r}. Use one of: {', '.join(ANSWER_MODELS)}, or a Bedrock model or "
            f"inference profile id such as 'us.amazon.nova-pro-v1:0'.{did_you_mean(name, ANSWER_MODELS)}"
        )
    prefix, _, rest = name.partition(".")
    profile = prefix if prefix in PROFILE_PREFIXES else ""
    base = rest if profile else name
    known = next((m for m in ANSWER_MODELS.values() if m.model_id == base), None)
    return ParsingModel(name, base, known.label if known else name, profile=profile)
