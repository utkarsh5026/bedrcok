"""The Bedrock pieces: the knowledge base, its data source, and the ingestion (sync) job.

* The **knowledge base** says *which* embedding model to use and *where* vectors are stored.
* The **data source** says *where documents come from* and *how to chunk them*.
* An **ingestion job** actually reads, chunks, embeds and stores the documents. You run one
  after creating the KB, and again whenever documents change (``kb.sync()``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from botocore.exceptions import ClientError

from ..aws import Aws, error_code, retry_on, wait_for
from .base import Action, Ctx, Inspection, Log, Step, is_ours
from .opensearch import METADATA_FIELD, TEXT_FIELD, VECTOR_FIELD


class KnowledgeBaseStep(Step):
    """The Bedrock knowledge base itself."""

    title = "Bedrock knowledge base"

    def resource(self, ctx: Ctx) -> str:
        """The knowledge base name."""
        return ctx.names.kb

    def explain(self, ctx: Ctx) -> str:
        """Which embedding model the KB uses."""
        return f"Ties it together: {ctx.spec.model.label} embeddings stored in the index above."

    def _storage_type(self, ctx: Ctx) -> str:
        return "S3_VECTORS" if ctx.spec.vector_store == "s3_vectors" else "OPENSEARCH_SERVERLESS"

    def _storage(self, ctx: Ctx) -> Dict[str, Any]:
        """The ``storageConfiguration`` block: where Bedrock writes vectors."""
        if ctx.spec.vector_store == "s3_vectors":
            return {"type": "S3_VECTORS", "s3VectorsConfiguration": {"indexArn": ctx.index_arn}}
        return {
            "type": "OPENSEARCH_SERVERLESS",
            "opensearchServerlessConfiguration": {
                "collectionArn": ctx.out["collection_arn"],
                "vectorIndexName": ctx.names.aoss_index,
                "fieldMapping": {"vectorField": VECTOR_FIELD, "textField": TEXT_FIELD, "metadataField": METADATA_FIELD},
            },
        }

    def _embedding(self, ctx: Ctx) -> Dict[str, Any]:
        """The ``vectorKnowledgeBaseConfiguration`` block: which model and how many dimensions."""
        model = ctx.spec.model
        config: Dict[str, Any] = {"embeddingModelArn": model.arn(ctx.aws.region)}
        if model.configurable_dimensions:
            config["embeddingModelConfiguration"] = {
                "bedrockEmbeddingModelConfiguration": {"dimensions": ctx.spec.dims}
            }
        return config

    def _dimensions(self, ctx: Ctx, vector_config: Dict[str, Any]) -> int:
        """The vector size an existing KB was created with.

        Bedrock only records it for models where you can choose; otherwise (or if it's left
        out) the model's default size applies.
        """
        configured = (
            vector_config.get("embeddingModelConfiguration", {})
            .get("bedrockEmbeddingModelConfiguration", {})
            .get("dimensions")
        )
        return configured or ctx.spec.model.default_dimensions

    def find(self, ctx: Ctx) -> Optional[Dict[str, Any]]:
        """Look up a knowledge base by name (Bedrock only lets you fetch by id, so we list)."""
        agent = ctx.aws.client("bedrock-agent")
        for page in agent.get_paginator("list_knowledge_bases").paginate():
            for summary in page.get("knowledgeBaseSummaries", []):
                if summary["name"] == ctx.names.kb:
                    return agent.get_knowledge_base(knowledgeBaseId=summary["knowledgeBaseId"])["knowledgeBase"]
        return None

    def inspect(self, ctx: Ctx) -> Inspection:
        """Find the KB by name and check its model and vector store match the spec."""
        kb = self.find(ctx)
        if kb is None:
            return Inspection(Action.CREATE)
        ctx.out.update(kb_id=kb["knowledgeBaseId"], kb_arn=kb["knowledgeBaseArn"])

        if kb["status"] in ("FAILED", "DELETE_UNSUCCESSFUL"):
            reasons = "; ".join(kb.get("failureReasons", [])) or kb["status"]
            return Inspection(Action.CONFLICT, warning=f"Existing knowledge base is {kb['status']}: {reasons}")

        # Embedding model, vector size and vector store are fixed once a KB exists, so a mismatch
        # is a blocker.
        problems = []
        vector_config = kb.get("knowledgeBaseConfiguration", {}).get("vectorKnowledgeBaseConfiguration", {})
        current_model = vector_config.get("embeddingModelArn", "?")
        if current_model != ctx.spec.model.arn(ctx.aws.region):
            problems.append(f"it uses {current_model.split('/')[-1]}")
        else:
            current_dims = self._dimensions(ctx, vector_config)
            if current_dims != ctx.spec.dims:
                problems.append(f"it makes {current_dims}-dimension vectors, not {ctx.spec.dims}")
        current_store = kb.get("storageConfiguration", {}).get("type")
        if current_store != self._storage_type(ctx):
            problems.append(f"it stores vectors in {current_store}")
        if problems:
            return Inspection(
                Action.CONFLICT,
                warning=(
                    f"A knowledge base named '{ctx.names.kb}' already exists but {' and '.join(problems)}. "
                    "Embedding model, vector size and vector store can't change in place — pick a new name "
                    "or destroy it first."
                ),
            )
        return Inspection(Action.EXISTS, kb["knowledgeBaseId"])

    def create(self, ctx: Ctx, log: Log, inspection: Inspection) -> str:
        """Create the KB (retrying while a new role/index propagates) and wait until ACTIVE."""
        agent = ctx.aws.client("bedrock-agent")
        request = dict(
            name=ctx.names.kb,
            description=ctx.spec.description or f"Created with kbforge from {ctx.spec.source.describe()}",
            roleArn=ctx.role_arn,
            knowledgeBaseConfiguration={"type": "VECTOR", "vectorKnowledgeBaseConfiguration": self._embedding(ctx)},
            storageConfiguration=self._storage(ctx),
            tags=ctx.tags,
        )
        # A brand-new IAM role or index can take a little while to become visible to Bedrock.
        kb = retry_on(
            lambda: agent.create_knowledge_base(**request)["knowledgeBase"],
            codes=("ValidationException",),
            hints=("role", "assume", "index", "collection", "not authorized"),
            log=log,
            why="Bedrock can't see the new role/index yet",
        )
        ctx.out.update(kb_id=kb["knowledgeBaseId"], kb_arn=kb["knowledgeBaseArn"])
        wait_for(
            lambda: agent.get_knowledge_base(knowledgeBaseId=kb["knowledgeBaseId"])["knowledgeBase"],
            lambda k: k["status"] == "ACTIVE",
            failed=lambda k: k["status"] == "FAILED" and ("; ".join(k.get("failureReasons", [])) or "FAILED"),
            describe=lambda k: k["status"].lower(),
            what="knowledge base to become active",
            log=log,
        )
        return kb["knowledgeBaseId"]

    def delete(self, ctx: Ctx, log: Log) -> str:
        """Delete the KB and wait until it's gone."""
        agent = ctx.aws.client("bedrock-agent")
        kb_id = ctx.out["kb_id"]
        agent.delete_knowledge_base(knowledgeBaseId=kb_id)

        def current_status() -> Optional[str]:
            """The resource's status, or None once it's gone."""
            try:
                return agent.get_knowledge_base(knowledgeBaseId=kb_id)["knowledgeBase"]["status"]
            except ClientError as exc:
                if error_code(exc) == "ResourceNotFoundException":
                    return None
                raise

        wait_for(
            current_status,
            lambda s: s is None,
            failed=lambda s: s == "DELETE_UNSUCCESSFUL" and s,
            what="knowledge base deletion",
            log=log,
        )
        return "deleted"

    def owned(self, ctx: Ctx) -> Optional[bool]:
        """True if the KB carries kbforge's tags for this knowledge base."""
        tags = ctx.aws.client("bedrock-agent").list_tags_for_resource(resourceArn=ctx.out["kb_arn"]).get("tags", {})
        return is_ours(tags, ctx)


class DataSourceStep(Step):
    """Connects the knowledge base to your documents and sets the chunking strategy."""

    title = "Data source"

    def resource(self, ctx: Ctx) -> str:
        """The data source name."""
        return ctx.names.data_source

    def explain(self, ctx: Ctx) -> str:
        """Where documents come from and how they're chunked."""
        return f"Points the knowledge base at {ctx.spec.source.describe()}. Chunking: {ctx.spec.chunking.describe()}."

    def find(self, ctx: Ctx) -> Optional[Dict[str, Any]]:
        """Look up our data source by name inside the knowledge base (None if the KB doesn't exist yet)."""
        kb_id = ctx.out.get("kb_id")
        if not kb_id:
            return None
        agent = ctx.aws.client("bedrock-agent")
        for page in agent.get_paginator("list_data_sources").paginate(knowledgeBaseId=kb_id):
            for summary in page.get("dataSourceSummaries", []):
                if summary["name"] == ctx.names.data_source:
                    found = agent.get_data_source(knowledgeBaseId=kb_id, dataSourceId=summary["dataSourceId"])
                    return found["dataSource"]
        return None

    def inspect(self, ctx: Ctx) -> Inspection:
        """Find the data source; warn if its chunking differs from the spec."""
        ds = self.find(ctx)
        if ds is None:
            return Inspection(Action.CREATE)
        ctx.out["ds_id"] = ds["dataSourceId"]
        chunking = ds.get("vectorIngestionConfiguration", {}).get("chunkingConfiguration", {})
        current, wanted = chunking.get("chunkingStrategy"), ctx.spec.chunking.to_api()["chunkingStrategy"]
        warning = ""
        if current and current != wanted:
            warning = (
                f"Existing data source chunks with {current}, but the spec says {wanted}. Chunking is fixed "
                "when a data source is created — destroy and recreate to change it."
            )
        return Inspection(Action.EXISTS, ds["dataSourceId"], warning=warning)

    def create(self, ctx: Ctx, log: Log, inspection: Inspection) -> str:
        """Create the data source with the spec's source and chunking."""
        ds = ctx.aws.client("bedrock-agent").create_data_source(
            knowledgeBaseId=ctx.out["kb_id"],
            name=ctx.names.data_source,
            description=f"{ctx.spec.source.describe()} (kbforge)",
            dataSourceConfiguration=ctx.spec.source.to_api(),
            vectorIngestionConfiguration={"chunkingConfiguration": ctx.spec.chunking.to_api()},
            # When the data source is deleted, also delete its vectors from the store.
            dataDeletionPolicy="DELETE",
        )["dataSource"]
        ctx.out["ds_id"] = ds["dataSourceId"]
        return ds["dataSourceId"]

    def delete(self, ctx: Ctx, log: Log) -> str:
        """Delete the data source (and, by its deletion policy, its vectors)."""
        agent = ctx.aws.client("bedrock-agent")
        kb_id, ds_id = ctx.out["kb_id"], ctx.out["ds_id"]
        agent.delete_data_source(knowledgeBaseId=kb_id, dataSourceId=ds_id)

        def current_status() -> Optional[str]:
            """The resource's status, or None once it's gone."""
            try:
                return agent.get_data_source(knowledgeBaseId=kb_id, dataSourceId=ds_id)["dataSource"]["status"]
            except ClientError as exc:
                if error_code(exc) == "ResourceNotFoundException":
                    return None
                raise

        wait_for(
            current_status,
            lambda s: s is None,
            failed=lambda s: s == "DELETE_UNSUCCESSFUL" and s,
            what="data source deletion",
            log=log,
        )
        return "deleted"

    def owned(self, ctx: Ctx) -> Optional[bool]:
        """A data source can't be tagged, so it counts as ours when its knowledge base is."""
        return KnowledgeBaseStep().owned(ctx)


class SyncStep(Step):
    """Runs the first ingestion job so the knowledge base is usable as soon as apply finishes."""

    title = "First sync (ingestion)"
    deletable = False

    def resource(self, ctx: Ctx) -> str:
        """Not a resource: shown as 'ingestion job'."""
        return "ingestion job"

    def explain(self, ctx: Ctx) -> str:
        """What an ingestion job does."""
        return "Reads every document, chunks it, embeds each chunk and stores the vectors. Time grows with corpus size."

    def inspect(self, ctx: Ctx) -> Inspection:
        """Run a sync unless one already completed or is in progress."""
        if not ctx.spec.sync:
            return Inspection(Action.SKIP, "sync disabled — run kb.sync() when ready")
        if not ctx.out.get("ds_id"):
            return Inspection(Action.RUN)
        job = latest_ingestion_job(ctx.aws, ctx.out["kb_id"], ctx.out["ds_id"])
        if job is None:
            return Inspection(Action.RUN)
        if job["status"] in ("FAILED", "STOPPED"):
            return Inspection(Action.RUN, "previous sync did not finish")
        if job["status"] == "COMPLETE":
            return Inspection(Action.SKIP, "already synced — run kb.sync() to pick up new documents")
        return Inspection(Action.SKIP, f"a sync is already {job['status'].lower().replace('_', ' ')}")

    def create(self, ctx: Ctx, log: Log, inspection: Inspection) -> str:
        """Run the ingestion job to completion and report its statistics."""
        job = run_ingestion(ctx.aws, ctx.out["kb_id"], ctx.out["ds_id"], log)
        ctx.out["last_job"] = job
        return describe_stats(job.get("statistics", {}))

    def delete(self, ctx: Ctx, log: Log) -> str:
        """Nothing to delete: an ingestion job isn't a resource."""
        return ""


def latest_ingestion_job(aws: Aws, kb_id: str, ds_id: str) -> Optional[Dict[str, Any]]:
    """The most recent ingestion job for a data source, or None if it has never synced."""
    jobs = aws.client("bedrock-agent").list_ingestion_jobs(
        knowledgeBaseId=kb_id,
        dataSourceId=ds_id,
        sortBy={"attribute": "STARTED_AT", "order": "DESCENDING"},
        maxResults=1,
    )
    summaries = jobs.get("ingestionJobSummaries", [])
    return summaries[0] if summaries else None


def describe_stats(stats: Dict[str, int]) -> str:
    """Turn ingestion statistics into a one-liner like ``120 scanned · 118 indexed · 2 failed``."""
    indexed = stats.get("numberOfNewDocumentsIndexed", 0) + stats.get("numberOfModifiedDocumentsIndexed", 0)
    parts = [f"{stats.get('numberOfDocumentsScanned', 0)} scanned", f"{indexed} indexed"]
    if stats.get("numberOfDocumentsDeleted"):
        parts.append(f"{stats['numberOfDocumentsDeleted']} removed")
    if stats.get("numberOfDocumentsFailed"):
        parts.append(f"{stats['numberOfDocumentsFailed']} failed")
    return " · ".join(parts)


def run_ingestion(aws: Aws, kb_id: str, ds_id: str, log: Log) -> Dict[str, Any]:
    """Start an ingestion job and wait for it to finish, reporting progress through ``log``."""
    agent = aws.client("bedrock-agent")
    job_id = agent.start_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=ds_id)["ingestionJob"]["ingestionJobId"]
    log("Ingestion job started")
    return wait_for(
        lambda: agent.get_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=ds_id, ingestionJobId=job_id)[
            "ingestionJob"
        ],
        lambda j: j["status"] in ("COMPLETE", "STOPPED"),
        failed=lambda j: j["status"] == "FAILED" and ("; ".join(j.get("failureReasons", [])) or "FAILED"),
        describe=lambda j: f"{j['status'].lower().replace('_', ' ')} · {describe_stats(j.get('statistics', {}))}",
        what="ingestion",
        log=log,
        timeout=6 * 3600,
        interval=10,
    )
