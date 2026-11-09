"""kbforge — build Amazon Bedrock Knowledge Bases from a notebook, plan first.

    from kbforge import KnowledgeBase

    kb = KnowledgeBase("hr-policies", source="s3://my-bucket/hr/")
    kb.plan()          # shows what will be created, checks your data and permissions
    kb.apply()         # builds it, with live progress
    kb.ask("How many days of leave do I get?")

New here? Run ``kbforge.guide()`` in a notebook cell, or ``kbforge.wizard()`` for a form.
"""

from typing import Any, Optional

from .guide import guide
from .knowledge_base import Answers, KBForgeError, KnowledgeBase, Status
from .models import EMBEDDING_MODELS
from .plan import ApplyResult, Plan
from .spec import Chunking, KBSpec, S3Source, WebSource

__version__ = "0.1.0"


def wizard(region: Optional[str] = None, session: Any = None) -> Any:
    """Show an interactive form that plans and builds a knowledge base (needs ipywidgets)."""
    from .wizard import wizard as _wizard

    return _wizard(region=region, session=session)


__all__ = [
    "EMBEDDING_MODELS",
    "Answers",
    "ApplyResult",
    "Chunking",
    "KBForgeError",
    "KBSpec",
    "KnowledgeBase",
    "Plan",
    "S3Source",
    "Status",
    "WebSource",
    "__version__",
    "guide",
    "wizard",
]
