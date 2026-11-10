"""kbforge — build Amazon Bedrock Knowledge Bases from a notebook, plan first.

    from kbforge import KnowledgeBase

    kb = KnowledgeBase("hr-policies", source="s3://my-bucket/hr/")
    kb.plan()          # shows what will be created, checks your data and permissions
    kb.apply()         # builds it, with live progress
    kb.ask("How many days of leave do I get?")

New here? Run ``kbforge.guide()`` in a notebook cell, or ``kbforge.wizard()`` for a form.
Built one before? ``kbforge.knowledge_bases()`` lists them and ``KnowledgeBase.open(name)`` picks one up.
Keeping one up to date? ``kbforge.sync_panel()`` checks, syncs and shows how every sync went.
"""

from typing import Any, Optional

# Loaded up front on purpose. Python sets ``kbforge.wizard`` to the submodule the first time it is
# imported, so importing it lazily inside ``wizard()`` below would replace that function with the
# module, and running ``kbforge.wizard()`` a second time would fail. (ipywidgets itself is still
# only imported when the form is shown.)
from . import wizard as _wizard_module
from .deployed import KnowledgeBases
from .guide import guide
from .ingestion import IngestionJob, IngestionJobs
from .knowledge_base import Answers, KBForgeError, KnowledgeBase, Status
from .models import ANSWER_MODELS, EMBEDDING_MODELS, PARSING_MODELS
from .parsers import ParsingModels
from .plan import ApplyResult, Plan
from .spec import Chunking, KBSpec, Parsing, S3Source, WebSource
from .sync_check import SyncCheck

__version__ = "0.1.0"


def wizard(region: Optional[str] = None, session: Any = None) -> Any:
    """Show an interactive form that plans and builds a knowledge base (needs ipywidgets)."""
    return _wizard_module.wizard(region=region, session=session)


def sync_panel(name: Optional[str] = None, region: Optional[str] = None, session: Any = None) -> Any:
    """Show buttons to check, sync and stop a knowledge base, with every recent sync (needs ipywidgets).

    Pass the knowledge base's ``name``, or leave it out to pick from the ones kbforge built in
    the region. Same as ``KnowledgeBase.open(name).sync_panel()``.
    """
    from .panel import SyncPanel, SyncPicker

    if name is None:
        return SyncPicker(region=region, session=session).widget
    return SyncPanel(KnowledgeBase.open(name, region=region, session=session)).widget


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
    "IngestionJob",
    "IngestionJobs",
    "KBForgeError",
    "KBSpec",
    "KnowledgeBase",
    "KnowledgeBases",
    "Parsing",
    "ParsingModels",
    "Plan",
    "S3Source",
    "Status",
    "SyncCheck",
    "WebSource",
    "__version__",
    "guide",
    "knowledge_bases",
    "parsing_models",
    "sync_panel",
    "wizard",
]
