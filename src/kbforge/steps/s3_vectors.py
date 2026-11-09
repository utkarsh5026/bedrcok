"""Amazon S3 Vectors: the default, serverless place to store embeddings.

Two resources: a *vector bucket* (a container) and a *vector index* inside it (where the
vectors actually live). No cluster, no minimum hourly cost — you pay for storage and queries.
"""

from __future__ import annotations

from typing import Optional

from botocore.exceptions import ClientError

from ..aws import error_code
from .base import Action, Ctx, Inspection, Log, Step, claimed_by, is_ours, shared_name_conflict, tags_if_allowed

#: Bedrock stores each chunk's text and metadata next to its vector under these keys.
#: Marking them "non-filterable" avoids S3 Vectors' 2 KB limit on filterable metadata,
#: which large chunks would otherwise exceed.
BEDROCK_NON_FILTERABLE_KEYS = ["AMAZON_BEDROCK_TEXT", "AMAZON_BEDROCK_METADATA"]


class VectorBucketStep(Step):
    """The S3 vector bucket that holds the index."""

    title = "S3 vector bucket"

    def resource(self, ctx: Ctx) -> str:
        """The vector bucket name."""
        return ctx.names.vector_bucket

    def explain(self, ctx: Ctx) -> str:
        """What S3 Vectors is and how it's billed."""
        return "Serverless vector storage. Pay per GB stored and per query, with no cluster to run."

    def inspect(self, ctx: Ctx) -> Inspection:
        """Check whether the vector bucket exists and isn't another knowledge base's."""
        s3v = ctx.aws.client("s3vectors")
        try:
            bucket = s3v.get_vector_bucket(vectorBucketName=ctx.names.vector_bucket)
        except ClientError as exc:
            if error_code(exc) == "NotFoundException":
                return Inspection(Action.CREATE)
            raise
        arn = bucket["vectorBucket"]["vectorBucketArn"]
        ctx.out["vector_bucket_arn"] = arn
        tags = tags_if_allowed(lambda: s3v.list_tags_for_resource(resourceArn=arn).get("tags", {}))
        owner = claimed_by(tags or {}, ctx)
        if owner:
            return shared_name_conflict(owner, ctx)
        return Inspection(Action.EXISTS)

    def create(self, ctx: Ctx, log: Log, inspection: Inspection) -> str:
        """Create the vector bucket with kbforge's tags."""
        resp = ctx.aws.client("s3vectors").create_vector_bucket(vectorBucketName=ctx.names.vector_bucket, tags=ctx.tags)
        ctx.out["vector_bucket_arn"] = resp["vectorBucketArn"]
        return resp["vectorBucketArn"]

    def delete(self, ctx: Ctx, log: Log) -> str:
        # A vector bucket can only be deleted once it has no indexes left.
        """Delete any leftover indexes, then the bucket."""
        s3v = ctx.aws.client("s3vectors")
        for index in s3v.list_indexes(vectorBucketName=ctx.names.vector_bucket).get("indexes", []):
            log(f"Deleting leftover index {index['indexName']}")
            s3v.delete_index(vectorBucketName=ctx.names.vector_bucket, indexName=index["indexName"])
        s3v.delete_vector_bucket(vectorBucketName=ctx.names.vector_bucket)
        return "deleted"

    def owned(self, ctx: Ctx) -> Optional[bool]:
        """True if the bucket carries kbforge's tags for this knowledge base."""
        arn = ctx.out.get("vector_bucket_arn")
        if not arn:
            return None
        tags = ctx.aws.client("s3vectors").list_tags_for_resource(resourceArn=arn).get("tags", {})
        return is_ours(tags, ctx)


class VectorIndexStep(Step):
    """The index inside the vector bucket. Its dimension must match the embedding model."""

    title = "Vector index"

    def resource(self, ctx: Ctx) -> str:
        """bucket/index."""
        return f"{ctx.names.vector_bucket}/{ctx.names.vector_index}"

    def explain(self, ctx: Ctx) -> str:
        """The index's size and distance metric."""
        return f"{ctx.spec.dims}-dimension float32 index, cosine distance. Holds one vector per chunk."

    def inspect(self, ctx: Ctx) -> Inspection:
        """Check the index exists and its dimension matches the embedding model."""
        try:
            index = ctx.aws.client("s3vectors").get_index(
                vectorBucketName=ctx.names.vector_bucket, indexName=ctx.names.vector_index
            )["index"]
        except ClientError as exc:
            # NotFound covers both "no index" and "no bucket yet".
            if error_code(exc) == "NotFoundException":
                return Inspection(Action.CREATE)
            raise
        ctx.out["index_arn"] = index["indexArn"]
        if index.get("dimension") != ctx.spec.dims:
            return Inspection(
                Action.CONFLICT,
                warning=(
                    f"Existing index has {index.get('dimension')} dimensions but this spec needs "
                    f"{ctx.spec.dims}. Destroy the knowledge base first or pick another name."
                ),
            )
        return Inspection(Action.EXISTS)

    def create(self, ctx: Ctx, log: Log, inspection: Inspection) -> str:
        """Create the index (float32, cosine, Bedrock keys non-filterable)."""
        s3v = ctx.aws.client("s3vectors")
        s3v.create_index(
            vectorBucketName=ctx.names.vector_bucket,
            indexName=ctx.names.vector_index,
            dataType="float32",
            dimension=ctx.spec.dims,
            distanceMetric="cosine",
            metadataConfiguration={"nonFilterableMetadataKeys": BEDROCK_NON_FILTERABLE_KEYS},
        )
        index = s3v.get_index(vectorBucketName=ctx.names.vector_bucket, indexName=ctx.names.vector_index)["index"]
        ctx.out["index_arn"] = index["indexArn"]
        return index["indexArn"]

    def delete(self, ctx: Ctx, log: Log) -> str:
        """Delete the index."""
        ctx.aws.client("s3vectors").delete_index(
            vectorBucketName=ctx.names.vector_bucket, indexName=ctx.names.vector_index
        )
        return "deleted"

    def owned(self, ctx: Ctx) -> Optional[bool]:
        """An index can't be tagged, so it counts as ours when the bucket around it is."""
        return VectorBucketStep().owned(ctx)
