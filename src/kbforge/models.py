"""The embedding models kbforge knows how to set up.

An embedding model turns a chunk of text into a list of numbers (a vector) so that chunks
with similar meaning end up close together. The knowledge base uses the same model for your
documents and for every question, so it can't be changed after the KB is created.

Users pick a model by a short alias (``"titan-v2"``) instead of a long model id.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Tuple


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
            "or pass a raw model id together with dimensions=..."
        )
    return EmbeddingModel(name, name, name, (dimensions,), dimensions)
