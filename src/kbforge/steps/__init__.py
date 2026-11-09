"""Steps: one class per AWS resource that makes up a knowledge base.

``build_steps(spec)`` returns them in the order they must be created. Destroying walks the
same list backwards.
"""

from __future__ import annotations

from typing import List

from ..spec import KBSpec
from .base import Action, Ctx, Inspection, Log, Step, explain_error
from .bedrock import DataSourceStep, KnowledgeBaseStep, SyncStep, describe_stats, latest_ingestion_job, run_ingestion
from .iam import RoleStep
from .opensearch import AccessPolicyStep, CollectionStep, IndexStep, SecurityPoliciesStep
from .s3_vectors import VectorBucketStep, VectorIndexStep


def build_steps(spec: KBSpec) -> List[Step]:
    """The ordered recipe for this spec.

    Order matters: the vector store comes before the role (so the role's policy can name the
    exact index/collection), and the role comes before the knowledge base that uses it.
    """
    if spec.vector_store == "s3_vectors":
        store: List[Step] = [VectorBucketStep(), VectorIndexStep(), RoleStep()]
    else:
        store = [SecurityPoliciesStep(), CollectionStep(), RoleStep(), AccessPolicyStep(), IndexStep()]
    return store + [KnowledgeBaseStep(), DataSourceStep(), SyncStep()]


__all__ = [
    "Action",
    "Ctx",
    "Inspection",
    "Log",
    "Step",
    "build_steps",
    "describe_stats",
    "explain_error",
    "latest_ingestion_job",
    "run_ingestion",
]
