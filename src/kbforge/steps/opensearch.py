"""OpenSearch Serverless: the classic vector store for Bedrock knowledge bases.

It needs more moving parts than S3 Vectors:

* an encryption policy and a network policy (must exist *before* the collection),
* the collection itself (takes a few minutes to become ACTIVE),
* a data access policy saying who may read/write it (Bedrock's role, and you),
* a vector index with the exact field names Bedrock expects.

The index has to be created through the OpenSearch API rather than boto3, which is why
this store uses the ``opensearch-py`` dependency (installed with kbforge).
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from botocore.exceptions import ClientError

from ..aws import error_code, wait_for
from .base import (
    Action,
    Ctx,
    Inspection,
    Log,
    Step,
    claimed_by,
    is_ours,
    shared_name_conflict,
    tag_dict,
    tags_if_allowed,
)

#: The description kbforge puts on the policies it creates. Policies can't be tagged, so this is
#: how ``destroy`` recognises them.
POLICY_DESCRIPTION = "kbforge"

#: Field names Bedrock reads and writes inside the index. They must match ``fieldMapping``
#: in the knowledge base's storage configuration.
VECTOR_FIELD = "bedrock-knowledge-base-default-vector"
TEXT_FIELD = "AMAZON_BEDROCK_TEXT_CHUNK"
METADATA_FIELD = "AMAZON_BEDROCK_METADATA"


class SecurityPoliciesStep(Step):
    """The encryption and network policies a collection needs before it can be created."""

    title = "Encryption & network policies"

    def resource(self, ctx: Ctx) -> str:
        """Both policy names."""
        return f"{ctx.names.enc_policy}, {ctx.names.net_policy}"

    def explain(self, ctx: Ctx) -> str:
        """What the two policies allow."""
        return "Encrypts the collection with an AWS-owned key and exposes an HTTPS endpoint (IAM auth still required)."

    def _policies(self, ctx: Ctx) -> Dict[str, Tuple[str, str]]:
        """Map of policy type -> (policy name, policy JSON)."""
        collection = f"collection/{ctx.names.collection}"
        encryption = {"Rules": [{"ResourceType": "collection", "Resource": [collection]}], "AWSOwnedKey": True}
        network = [
            {
                "Rules": [
                    {"ResourceType": "collection", "Resource": [collection]},
                    {"ResourceType": "dashboard", "Resource": [collection]},
                ],
                "AllowFromPublic": True,
            }
        ]
        return {
            "encryption": (ctx.names.enc_policy, json.dumps(encryption)),
            "network": (ctx.names.net_policy, json.dumps(network)),
        }

    def _existing(self, ctx: Ctx) -> Dict[str, Dict[str, Any]]:
        """The policies that already exist, as policy type -> AWS's details for it."""
        aoss = ctx.aws.client("opensearchserverless")
        found = {}
        for policy_type, (name, _) in self._policies(ctx).items():
            try:
                found[policy_type] = aoss.get_security_policy(name=name, type=policy_type)["securityPolicyDetail"]
            except ClientError as exc:
                if error_code(exc) != "ResourceNotFoundException":
                    raise
        return found

    def _missing(self, ctx: Ctx) -> List[str]:
        """Which of the two policy types don't exist yet."""
        existing = self._existing(ctx)
        return [policy_type for policy_type in self._policies(ctx) if policy_type not in existing]

    def inspect(self, ctx: Ctx) -> Inspection:
        """Create whichever of the two policies is missing."""
        missing = self._missing(ctx)
        if not missing:
            return Inspection(Action.EXISTS)
        return Inspection(Action.CREATE, "" if len(missing) == 2 else f"{missing[0]} policy missing")

    def create(self, ctx: Ctx, log: Log, inspection: Inspection) -> str:
        """Create the missing encryption/network policies."""
        aoss = ctx.aws.client("opensearchserverless")
        policies = self._policies(ctx)
        for policy_type in self._missing(ctx):
            name, document = policies[policy_type]
            log(f"Creating {policy_type} policy")
            aoss.create_security_policy(name=name, type=policy_type, policy=document, description=POLICY_DESCRIPTION)
        return "ready"

    def delete(self, ctx: Ctx, log: Log) -> str:
        """Delete both policies (ignoring ones already gone)."""
        aoss = ctx.aws.client("opensearchserverless")
        for policy_type, (name, _) in self._policies(ctx).items():
            try:
                aoss.delete_security_policy(name=name, type=policy_type)
            except ClientError as exc:
                if error_code(exc) != "ResourceNotFoundException":
                    raise
        return "deleted"

    def owned(self, ctx: Ctx) -> Optional[bool]:
        """Ours if kbforge's description is on every policy, and the collection (if still there) is ours too."""
        if any(d.get("description") != POLICY_DESCRIPTION for d in self._existing(ctx).values()):
            return False
        return _collection_owned(ctx, default=True)


class CollectionStep(Step):
    """The OpenSearch Serverless collection (think: a small managed search cluster)."""

    title = "OpenSearch Serverless collection"

    def resource(self, ctx: Ctx) -> str:
        """The collection name."""
        return ctx.names.collection

    def explain(self, ctx: Ctx) -> str:
        """What the collection is, and that it bills while it exists."""
        return "Vector search collection. Takes a few minutes to become active and bills per OCU-hour while it exists."

    def _get(self, ctx: Ctx) -> Optional[Dict[str, Any]]:
        """Fetch the collection's details, or None if it doesn't exist."""
        found = ctx.aws.client("opensearchserverless").batch_get_collection(names=[ctx.names.collection])
        details = found.get("collectionDetails", [])
        return details[0] if details else None

    def _remember(self, ctx: Ctx, collection: Dict[str, Any]) -> None:
        """Save the id/ARN/endpoint so the role, access policy and index steps can use them."""
        ctx.out.update(
            collection_id=collection["id"],
            collection_arn=collection["arn"],
            collection_endpoint=collection.get("collectionEndpoint"),
        )

    def inspect(self, ctx: Ctx) -> Inspection:
        """Find the collection and remember its id, ARN and endpoint."""
        collection = self._get(ctx)
        if collection is None:
            return Inspection(Action.CREATE)
        self._remember(ctx, collection)
        aoss = ctx.aws.client("opensearchserverless")
        tags = tags_if_allowed(lambda: tag_dict(aoss.list_tags_for_resource(resourceArn=collection["arn"])["tags"]))
        owner = claimed_by(tags or {}, ctx)
        if owner:
            return shared_name_conflict(owner, ctx)
        if collection["status"] == "FAILED":
            reason = collection.get("failureMessage", "")
            return Inspection(Action.CONFLICT, warning=f"Collection is FAILED: {reason}")
        return Inspection(Action.EXISTS, collection["status"].lower())

    def create(self, ctx: Ctx, log: Log, inspection: Inspection) -> str:
        """Create the collection and wait until it's ACTIVE (a few minutes)."""
        ctx.aws.client("opensearchserverless").create_collection(
            name=ctx.names.collection,
            type="VECTORSEARCH",
            description=f"Bedrock KB {ctx.spec.name} (kbforge)",
            tags=[{"key": k, "value": v} for k, v in ctx.tags.items()],
        )
        collection = wait_for(
            lambda: self._get(ctx),
            lambda c: c is not None and c["status"] == "ACTIVE",
            failed=lambda c: c and c["status"] == "FAILED" and c.get("failureMessage", "FAILED"),
            describe=lambda c: c["status"].lower() if c else "pending",
            what="collection to become active",
            log=log,
            interval=15,
        )
        self._remember(ctx, collection)
        return collection["arn"]

    def delete(self, ctx: Ctx, log: Log) -> str:
        """Delete the collection and wait until it's gone."""
        ctx.aws.client("opensearchserverless").delete_collection(id=ctx.out["collection_id"])
        wait_for(lambda: self._get(ctx), lambda c: c is None, what="collection deletion", log=log, interval=10)
        return "deleted"

    def owned(self, ctx: Ctx) -> Optional[bool]:
        """True if the collection carries kbforge's tags for this knowledge base."""
        arn = ctx.out.get("collection_arn")
        if not arn:
            return None
        tags = ctx.aws.client("opensearchserverless").list_tags_for_resource(resourceArn=arn)["tags"]
        return is_ours(tag_dict(tags), ctx)


def _collection_owned(ctx: Ctx, default: Optional[bool]) -> Optional[bool]:
    """The collection's ownership verdict, for policies that can't be tagged themselves.

    ``default`` is used when the collection doesn't exist (any more), so there's nothing to ask.
    """
    if not ctx.out.get("collection_arn"):
        return default
    return CollectionStep().owned(ctx)


class AccessPolicyStep(Step):
    """Grants Bedrock's role (to sync and query) and you (to create the index) access to the data."""

    title = "Data access policy"

    def resource(self, ctx: Ctx) -> str:
        """The data access policy name."""
        return ctx.names.access_policy

    def explain(self, ctx: Ctx) -> str:
        """Who the policy lets in."""
        return "Lets the Bedrock role, and you (to create the index), read and write the collection."

    def details(self, ctx: Ctx) -> Optional[Tuple[str, str]]:
        """The principals the policy grants access to."""
        return ("Principals", "\n".join(self._principals(ctx)))

    def _principals(self, ctx: Ctx) -> List[str]:
        """The principals this run needs: Bedrock's role and whoever is running kbforge."""
        return sorted({ctx.role_arn, ctx.aws.principal_arn})

    def _policy(self, ctx: Ctx, principals: List[str]) -> str:
        """The policy document granting ``principals`` access to the collection and its indexes."""
        collection = ctx.names.collection
        rules = [
            {
                "ResourceType": "collection",
                "Resource": [f"collection/{collection}"],
                "Permission": [
                    "aoss:CreateCollectionItems",
                    "aoss:DeleteCollectionItems",
                    "aoss:UpdateCollectionItems",
                    "aoss:DescribeCollectionItems",
                ],
            },
            {
                "ResourceType": "index",
                "Resource": [f"index/{collection}/*"],
                "Permission": [
                    "aoss:CreateIndex",
                    "aoss:DeleteIndex",
                    "aoss:UpdateIndex",
                    "aoss:DescribeIndex",
                    "aoss:ReadDocument",
                    "aoss:WriteDocument",
                ],
            },
        ]
        return json.dumps([{"Rules": rules, "Principal": principals}])

    def inspect(self, ctx: Ctx) -> Inspection:
        """Check the policy exists and includes both principals."""
        try:
            detail = ctx.aws.client("opensearchserverless").get_access_policy(
                name=ctx.names.access_policy, type="data"
            )["accessPolicyDetail"]
        except ClientError as exc:
            if error_code(exc) == "ResourceNotFoundException":
                return Inspection(Action.CREATE)
            raise
        document = detail.get("policy", [])
        # boto3 normally decodes the document for us, but be tolerant of a raw JSON string.
        if isinstance(document, str):
            document = json.loads(document)
        current = {p for rule in document for p in rule.get("Principal", [])}
        ctx.out.update(
            access_policy_version=detail["policyVersion"],
            access_policy_principals=sorted(current),
            access_policy_description=detail.get("description", ""),
        )
        if set(self._principals(ctx)) <= current:
            return Inspection(Action.EXISTS)
        return Inspection(Action.UPDATE, "adding missing principals")

    def create(self, ctx: Ctx, log: Log, inspection: Inspection) -> str:
        """Create the policy, or update it to add missing principals.

        An update keeps every principal already in the policy. Teammates who ran kbforge before
        you were granted access the same way, and replacing the list would lock them out.
        """
        aoss = ctx.aws.client("opensearchserverless")
        if inspection.action == Action.UPDATE:
            principals = sorted(set(ctx.out.get("access_policy_principals", [])) | set(self._principals(ctx)))
            aoss.update_access_policy(
                name=ctx.names.access_policy,
                type="data",
                policyVersion=ctx.out["access_policy_version"],
                policy=self._policy(ctx, principals),
            )
        else:
            aoss.create_access_policy(
                name=ctx.names.access_policy,
                type="data",
                policy=self._policy(ctx, self._principals(ctx)),
                description=POLICY_DESCRIPTION,
            )
        return "ready"

    def delete(self, ctx: Ctx, log: Log) -> str:
        """Delete the data access policy."""
        ctx.aws.client("opensearchserverless").delete_access_policy(name=ctx.names.access_policy, type="data")
        return "deleted"

    def owned(self, ctx: Ctx) -> Optional[bool]:
        """Ours when its collection is; with no collection left, go by kbforge's description.

        Older kbforge versions created this policy without a description, so a missing one
        isn't proof either way (None).
        """
        marked = ctx.out.get("access_policy_description") == POLICY_DESCRIPTION
        return _collection_owned(ctx, default=True if marked else None)


class IndexStep(Step):
    """The k-NN vector index inside the collection, with the field names Bedrock expects."""

    title = "Vector index"
    deletable = False

    def resource(self, ctx: Ctx) -> str:
        """collection/index."""
        return f"{ctx.names.collection}/{ctx.names.aoss_index}"

    def explain(self, ctx: Ctx) -> str:
        """The index's shape."""
        return f"{ctx.spec.dims}-dimension HNSW (faiss) index. Removed automatically with the collection."

    def _client(self, ctx: Ctx) -> Any:
        """An OpenSearch client that signs requests with your notebook's AWS credentials."""
        return ctx.aws.opensearch(ctx.out["collection_endpoint"])

    def _mapping(self, ctx: Ctx) -> Dict[str, Any]:
        return {
            "settings": {"index": {"knn": True, "knn.algo_param.ef_search": 512}},
            "mappings": {
                "properties": {
                    VECTOR_FIELD: {
                        "type": "knn_vector",
                        "dimension": ctx.spec.dims,
                        "method": {
                            "name": "hnsw",
                            "engine": "faiss",
                            "space_type": "l2",
                            "parameters": {"ef_construction": 512, "m": 16},
                        },
                    },
                    TEXT_FIELD: {"type": "text"},
                    METADATA_FIELD: {"type": "text", "index": False},
                }
            },
        }

    def inspect(self, ctx: Ctx) -> Inspection:
        """Check whether the index exists (if we can reach the collection yet) and has the right size."""
        if not ctx.out.get("collection_endpoint"):
            return Inspection(Action.CREATE)
        name = ctx.names.aoss_index
        try:
            client = self._client(ctx)
            if not client.indices.exists(index=name):
                return Inspection(Action.CREATE)
            mapping = client.indices.get_mapping(index=name)
        except Exception:
            # Usually means the access policy doesn't include us yet. Apply will sort it out.
            return Inspection(Action.CREATE, "couldn't check yet — will create if missing")
        fields = mapping.get(name, {}).get("mappings", {}).get("properties", {})
        dims = fields.get(VECTOR_FIELD, {}).get("dimension")
        if dims != ctx.spec.dims:
            found = f"{dims}-dimension vectors" if dims else f"no '{VECTOR_FIELD}' field"
            return Inspection(
                Action.CONFLICT,
                warning=(
                    f"Existing index has {found} but this spec needs {ctx.spec.dims}. "
                    "Destroy the knowledge base first or pick another name."
                ),
            )
        return Inspection(Action.EXISTS)

    def create(self, ctx: Ctx, log: Log, inspection: Inspection) -> str:
        """Create the index, retrying while the access policy takes effect."""
        client = self._client(ctx)
        attempts = 18
        # A new data access policy can take a minute or two to apply; until then we get 403s.
        for attempt in range(1, attempts + 1):
            try:
                if client.indices.exists(index=ctx.names.aoss_index):
                    return "already present"
                client.indices.create(index=ctx.names.aoss_index, body=self._mapping(ctx))
                break
            except Exception as exc:
                if getattr(exc, "status_code", None) != 403 or attempt == attempts:
                    raise
                log(f"Waiting for the access policy to take effect (attempt {attempt}/{attempts})")
                time.sleep(10)
        log("Letting the index settle")
        time.sleep(20)
        return ctx.names.aoss_index

    def delete(self, ctx: Ctx, log: Log) -> str:
        """Nothing to do: the index goes away with the collection."""
        return "removed with collection"
