"""kbforge — build Amazon Bedrock Knowledge Bases from a notebook, plan first.

    from kbforge import KnowledgeBase

    kb = KnowledgeBase("hr-policies", source="s3://my-bucket/hr/")
    kb.plan()          # shows what will be created, checks your data and permissions
    kb.apply()         # builds it, with live progress
    kb.ask("How many days of leave do I get?")

New here? Run ``kbforge.guide()`` in a notebook cell, or ``kbforge.wizard()`` for a form.
Built one before? ``kbforge.knowledge_bases()`` lists them and ``KnowledgeBase.open(name)`` picks one up.
"""

from typing import Any, Optional

from .deployed import KnowledgeBases
from .guide import guide
from .knowledge_base import Answers, KBForgeError, KnowledgeBase, Status
from .models import ANSWER_MODELS, EMBEDDING_MODELS, PARSING_MODELS
from .parsers import ParsingModels
from .plan import ApplyResult, Plan
from .spec import Chunking, KBSpec, Parsing, S3Source, WebSource

__version__ = "0.1.0"


def wizard(region: Optional[str] = None, session: Any = None) -> Any:
    """Show an interactive form that plans and builds a knowledge base (needs ipywidgets)."""
    from .wizard import wizard as _wizard

    return _wizard(region=region, session=session)


def parsing_models(region: Optional[str] = None, session: Any = None) -> ParsingModels:
    """List the models your account can use to parse documents (read-only; shows a card in a notebook).

    Pass one as ``KnowledgeBase(..., parsing="nova-pro")``. Falls back to kbforge's built-in list,
    with a note, if your role can't list models.
    """
    from .aws import Aws
    from .parsers import known_parsing_models, list_parsing_models

    try:
        aws = Aws(region=region, session=session)
    except Exception as exc:
        return known_parsing_models(region or "", f"Couldn't connect to AWS ({exc}).")
    return list_parsing_models(aws)


def knowledge_bases(region: Optional[str] = None, session: Any = None) -> KnowledgeBases:
    """List the knowledge bases kbforge built in a region, with their health and last sync (read-only).

    Shows a table in a notebook. Open one with ``KnowledgeBase.open(name)`` to sync, query or
    delete it without retyping its settings. Knowledge bases kbforge didn't create aren't listed.
    """
    from .aws import Aws
    from .deployed import list_knowledge_bases
    from .knowledge_base import friendly_aws_errors

    aws = Aws(region=region, session=session)
    with friendly_aws_errors():
        return list_knowledge_bases(aws)


__all__ = [
    "ANSWER_MODELS",
    "EMBEDDING_MODELS",
    "PARSING_MODELS",
    "Answers",
    "ApplyResult",
    "Chunking",
    "KBForgeError",
    "KBSpec",
    "KnowledgeBase",
    "KnowledgeBases",
    "Parsing",
    "ParsingModels",
    "Plan",
    "S3Source",
    "Status",
    "WebSource",
    "__version__",
    "guide",
    "knowledge_bases",
    "parsing_models",
    "wizard",
]
