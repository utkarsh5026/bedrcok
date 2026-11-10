"""An in-memory stand-in for the AWS services kbforge uses.

It's deliberately small: just enough behaviour (create → exists → delete, not-found errors,
paginators) to run a full plan/apply/destroy cycle in tests without an AWS account.
Everything becomes ACTIVE/COMPLETE immediately, so tests never wait.
"""

from __future__ import annotations

import fnmatch
import hashlib
import itertools
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

from botocore.exceptions import ClientError

ACCOUNT = "123456789012"
REGION = "us-east-1"
_counter = itertools.count(1)


def new_id() -> str:
    """A 10-character id that looks like the ones AWS hands out (e.g. ``Q7K2M9XPLA``)."""
    digest = hashlib.sha256(str(next(_counter)).encode()).digest()
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(alphabet[b % 36] for b in digest[:10])


def aws_error(code: str, message: str = "", operation: str = "Op") -> ClientError:
    """Build the same kind of exception boto3 raises."""
    return ClientError({"Error": {"Code": code, "Message": message or code}}, operation)


class _Paginator:
    def __init__(self, pages: List[Dict[str, Any]]):
        self._pages = pages

    def paginate(self, **_: Any) -> List[Dict[str, Any]]:
        return self._pages


class FakeSts:
    def get_caller_identity(self) -> Dict[str, str]:
        return {"Account": ACCOUNT, "Arn": f"arn:aws:sts::{ACCOUNT}:assumed-role/SageMakerRole/SageMaker"}


class FakeIam:
    def __init__(self) -> None:
        self.roles: Dict[str, Dict[str, Any]] = {}
        self.policies: Dict[str, Dict[str, str]] = {}
        #: True plays a notebook role without iam:SimulatePrincipalPolicy.
        self.simulator_denied = False
        #: Actions the notebook's own role isn't allowed, as the simulator reports them.
        self.caller_denied: Set[str] = set()

    def get_role(self, RoleName: str) -> Dict[str, Any]:
        if RoleName not in self.roles:
            raise aws_error("NoSuchEntity")
        return {"Role": self.roles[RoleName]}

    def create_role(
        self, RoleName: str, AssumeRolePolicyDocument: str, Tags: list, PermissionsBoundary: str = "", **_: Any
    ) -> Dict[str, Any]:
        role: Dict[str, Any] = {
            "RoleName": RoleName,
            "Arn": f"arn:aws:iam::{ACCOUNT}:role/{RoleName}",
            "AssumeRolePolicyDocument": AssumeRolePolicyDocument,
            "Tags": Tags,
        }
        if PermissionsBoundary:
            role["PermissionsBoundary"] = {
                "PermissionsBoundaryType": "Policy",
                "PermissionsBoundaryArn": PermissionsBoundary,
            }
        self.roles[RoleName] = role
        self.policies[RoleName] = {}
        return {"Role": role}

    def put_role_policy(self, RoleName: str, PolicyName: str, PolicyDocument: str) -> None:
        self.policies.setdefault(RoleName, {})[PolicyName] = PolicyDocument

    def get_role_policy(self, RoleName: str, PolicyName: str) -> Dict[str, Any]:
        if PolicyName not in self.policies.get(RoleName, {}):
            raise aws_error("NoSuchEntity")
        return {"PolicyDocument": json.loads(self.policies[RoleName][PolicyName])}

    def list_role_policies(self, RoleName: str) -> Dict[str, Any]:
        return {"PolicyNames": list(self.policies.get(RoleName, {}))}

    def delete_role_policy(self, RoleName: str, PolicyName: str) -> None:
        del self.policies[RoleName][PolicyName]

    def list_attached_role_policies(self, RoleName: str) -> Dict[str, Any]:
        return {"AttachedPolicies": []}

    def delete_role(self, RoleName: str) -> None:
        if self.policies.get(RoleName):
            raise aws_error("DeleteConflict", "role still has policies")
        self.roles.pop(RoleName)

    def simulate_principal_policy(
        self, PolicySourceArn: str, ActionNames: List[str], ResourceArns: List[str], **_: Any
    ) -> Dict[str, Any]:
        """IAM's policy simulator. Roles in this fake are judged by their inline policies (conditions
        are ignored); the notebook's own role isn't modelled, so it may do anything not in ``caller_denied``.
        """
        if self.simulator_denied:
            raise aws_error("AccessDenied", "not authorized to perform: iam:SimulatePrincipalPolicy")
        name = PolicySourceArn.rsplit("/", 1)[-1]
        results = []
        for action in ActionNames:
            if name in self.roles:
                allowed = any(
                    self._allows(json.loads(doc), action, ResourceArns[0]) for doc in self.policies[name].values()
                )
            else:
                allowed = action not in self.caller_denied
            results.append({"EvalActionName": action, "EvalDecision": "allowed" if allowed else "implicitDeny"})
        return {"EvaluationResults": results, "IsTruncated": False}

    @staticmethod
    def _allows(policy: Dict[str, Any], action: str, resource: str) -> bool:
        for statement in policy.get("Statement", []):
            actions = statement.get("Action", [])
            resources = statement.get("Resource", [])
            actions = [actions] if isinstance(actions, str) else actions
            resources = [resources] if isinstance(resources, str) else resources
            allows_action = statement.get("Effect") == "Allow" and any(fnmatch.fnmatch(action, a) for a in actions)
            if allows_action and any(fnmatch.fnmatch(resource, r) for r in resources):
                return True
        return False


class FakeS3:
    def __init__(self) -> None:
        self.buckets: Dict[str, Dict[str, int]] = {}
        #: Default encryption per bucket; buckets not listed use S3's own keys (AES256), like new buckets in AWS.
        self.encryption: Dict[str, Dict[str, str]] = {}

    def get_bucket_encryption(self, Bucket: str) -> Dict[str, Any]:
        default = self.encryption.get(Bucket, {"SSEAlgorithm": "AES256"})
        return {"ServerSideEncryptionConfiguration": {"Rules": [{"ApplyServerSideEncryptionByDefault": default}]}}

    def get_bucket_location(self, Bucket: str) -> Dict[str, Any]:
        if Bucket not in self.buckets:
            raise aws_error("NoSuchBucket")
        return {"LocationConstraint": None}

    def get_paginator(self, name: str) -> Any:
        fake = self

        class Pager:
            def paginate(self, Bucket: str, Prefix: str = "") -> List[Dict[str, Any]]:
                objects = fake.buckets.get(Bucket, {})
                contents = [{"Key": k, "Size": v} for k, v in objects.items() if k.startswith(Prefix)]
                return [{"Contents": contents}]

        return Pager()


class FakeBedrockRuntime:
    def __init__(self) -> None:
        self.denied = False
        #: Model ids ``converse`` refuses, to play a parsing model you don't have access to.
        self.denied_models: List[str] = []
        self.conversed: List[str] = []

    def invoke_model(self, **_: Any) -> Dict[str, Any]:
        if self.denied:
            raise aws_error("AccessDeniedException", "You don't have access to the model")
        return {"body": b"{}"}

    def converse(self, modelId: str, messages: list, **_: Any) -> Dict[str, Any]:
        self.conversed.append(modelId)
        if modelId in self.denied_models:
            raise aws_error("AccessDeniedException", "You don't have access to the model")
        return {"output": {"message": {"role": "assistant", "content": [{"text": "Hi"}]}}, "stopReason": "max_tokens"}


def _model(model_id: str, name: str, provider: str, inputs: List[str], on_demand: bool = False) -> Dict[str, Any]:
    return {
        "modelId": model_id,
        "modelName": name,
        "providerName": provider,
        "inputModalities": inputs,
        "outputModalities": ["TEXT"],
        "inferenceTypesSupported": ["ON_DEMAND"] if on_demand else ["INFERENCE_PROFILE"],
        "modelLifecycle": {"status": "ACTIVE"},
    }


class FakeBedrock:
    """Bedrock's control plane: just the model and inference-profile listings."""

    def __init__(self) -> None:
        self.denied = False
        self.models: List[Dict[str, Any]] = [
            _model("amazon.titan-embed-text-v2:0", "Titan Text Embeddings V2", "Amazon", ["TEXT"], on_demand=True),
            _model("amazon.nova-micro-v1:0", "Nova Micro", "Amazon", ["TEXT"]),
            _model("amazon.nova-lite-v1:0", "Nova Lite", "Amazon", ["TEXT", "IMAGE", "VIDEO"], on_demand=True),
            _model("amazon.nova-pro-v1:0", "Nova Pro", "Amazon", ["TEXT", "IMAGE", "VIDEO"]),
            _model("anthropic.claude-3-haiku-20240307-v1:0", "Claude 3 Haiku", "Anthropic", ["TEXT", "IMAGE"], True),
            _model("anthropic.claude-sonnet-4-20250514-v1:0", "Claude Sonnet 4", "Anthropic", ["TEXT", "IMAGE"]),
            _model("mistral.pixtral-large-2502-v1:0", "Pixtral Large", "Mistral AI", ["TEXT", "IMAGE"]),
            # No profile in this region, and not callable on demand: unusable here.
            _model("meta.llama4-maverick-17b-instruct-v1:0", "Llama 4 Maverick", "Meta", ["TEXT", "IMAGE"]),
        ]
        self.profiles: List[Dict[str, Any]] = [
            self._profile(f"us.{m}", m)
            for m in ("amazon.nova-lite-v1:0", "amazon.nova-pro-v1:0", "anthropic.claude-sonnet-4-20250514-v1:0")
        ]

    @staticmethod
    def _profile(profile_id: str, model_id: str) -> Dict[str, Any]:
        return {
            "inferenceProfileId": profile_id,
            "status": "ACTIVE",
            "models": [
                {"modelArn": f"arn:aws:bedrock:{r}::foundation-model/{model_id}"} for r in ("us-east-1", "us-west-2")
            ],
        }

    def list_foundation_models(self, byOutputModality: str = "", **_: Any) -> Dict[str, Any]:
        if self.denied:
            raise aws_error("AccessDeniedException", "not authorized to perform: bedrock:ListFoundationModels")
        return {"modelSummaries": [m for m in self.models if byOutputModality in m["outputModalities"]]}

    def list_inference_profiles(self, nextToken: str = "", **_: Any) -> Dict[str, Any]:
        # Two pages, so callers have to follow nextToken.
        if not nextToken:
            return {"inferenceProfileSummaries": self.profiles[:1], "nextToken": "page-2"}
        return {"inferenceProfileSummaries": self.profiles[1:]}


class FakeS3Vectors:
    def __init__(self) -> None:
        self.buckets: Dict[str, Dict[str, Any]] = {}
        self.indexes: Dict[tuple, Dict[str, Any]] = {}

    def list_vector_buckets(self, **_: Any) -> Dict[str, Any]:
        return {"vectorBuckets": []}

    def get_vector_bucket(self, vectorBucketName: str) -> Dict[str, Any]:
        if vectorBucketName not in self.buckets:
            raise aws_error("NotFoundException")
        return {"vectorBucket": self.buckets[vectorBucketName]}

    def create_vector_bucket(self, vectorBucketName: str, tags: Optional[dict] = None) -> Dict[str, Any]:
        arn = f"arn:aws:s3vectors:{REGION}:{ACCOUNT}:bucket/{vectorBucketName}"
        self.buckets[vectorBucketName] = {"vectorBucketArn": arn, "tags": tags or {}}
        return {"vectorBucketArn": arn}

    def list_tags_for_resource(self, resourceArn: str) -> Dict[str, Any]:
        for bucket in self.buckets.values():
            if bucket["vectorBucketArn"] == resourceArn:
                return {"tags": bucket["tags"]}
        raise aws_error("NotFoundException")

    def get_index(self, vectorBucketName: str, indexName: str) -> Dict[str, Any]:
        if (vectorBucketName, indexName) not in self.indexes:
            raise aws_error("NotFoundException")
        return {"index": self.indexes[(vectorBucketName, indexName)]}

    def create_index(self, vectorBucketName: str, indexName: str, dimension: int, **_: Any) -> Dict[str, Any]:
        if vectorBucketName not in self.buckets:
            raise aws_error("NotFoundException")
        arn = f"arn:aws:s3vectors:{REGION}:{ACCOUNT}:bucket/{vectorBucketName}/index/{indexName}"
        self.indexes[(vectorBucketName, indexName)] = {"indexArn": arn, "dimension": dimension}
        return {"indexArn": arn}

    def list_indexes(self, vectorBucketName: str) -> Dict[str, Any]:
        return {"indexes": [{"indexName": i} for (b, i) in self.indexes if b == vectorBucketName]}

    def delete_index(self, vectorBucketName: str, indexName: str) -> None:
        del self.indexes[(vectorBucketName, indexName)]

    def delete_vector_bucket(self, vectorBucketName: str) -> None:
        if any(b == vectorBucketName for b, _ in self.indexes):
            raise aws_error("ConflictException", "bucket not empty")
        del self.buckets[vectorBucketName]


class FakeBedrockAgent:
    def __init__(self) -> None:
        self.kbs: Dict[str, Dict[str, Any]] = {}
        self.data_sources: Dict[str, Dict[str, Any]] = {}
        self.jobs: List[Dict[str, Any]] = []
        self.fail_create_kb: Optional[ClientError] = None
        #: How the next syncs end, and their counts and failure reasons.
        self.job_status = "COMPLETE"
        self.job_statistics: Dict[str, int] = {"numberOfDocumentsScanned": 3, "numberOfNewDocumentsIndexed": 3}
        self.job_failure_reasons: List[str] = []
        #: How many times a new sync is looked at while IN_PROGRESS before it ends (0: ends at once).
        self.job_polls = 0
        #: Per-file results (``documentDetails`` entries), and whether listing them is denied.
        self.documents: List[Dict[str, Any]] = []
        self.documents_denied = False

    def get_paginator(self, name: str) -> _Paginator:
        if name == "list_knowledge_bases":
            summaries = [{"knowledgeBaseId": k, "name": v["name"]} for k, v in self.kbs.items()]
            return _Paginator([{"knowledgeBaseSummaries": summaries}])
        fake = self

        class DsPager:
            def paginate(self, knowledgeBaseId: str) -> List[Dict[str, Any]]:
                found = [
                    {"dataSourceId": d, "name": v["name"]}
                    for d, v in fake.data_sources.items()
                    if v["knowledgeBaseId"] == knowledgeBaseId
                ]
                return [{"dataSourceSummaries": found}]

        return DsPager()  # type: ignore[return-value]

    def create_knowledge_base(self, **kwargs: Any) -> Dict[str, Any]:
        if self.fail_create_kb:
            raise self.fail_create_kb
        kb_id = new_id()
        kb = {
            "knowledgeBaseId": kb_id,
            "knowledgeBaseArn": f"arn:aws:bedrock:{REGION}:{ACCOUNT}:knowledge-base/{kb_id}",
            "name": kwargs["name"],
            "description": kwargs.get("description", ""),
            "status": "ACTIVE",
            "knowledgeBaseConfiguration": kwargs["knowledgeBaseConfiguration"],
            "storageConfiguration": kwargs["storageConfiguration"],
            "roleArn": kwargs["roleArn"],
            "tags": kwargs.get("tags", {}),
        }
        self.kbs[kb_id] = kb
        return {"knowledgeBase": kb}

    def get_knowledge_base(self, knowledgeBaseId: str) -> Dict[str, Any]:
        if knowledgeBaseId not in self.kbs:
            raise aws_error("ResourceNotFoundException")
        return {"knowledgeBase": self.kbs[knowledgeBaseId]}

    def list_tags_for_resource(self, resourceArn: str) -> Dict[str, Any]:
        for kb in self.kbs.values():
            if kb["knowledgeBaseArn"] == resourceArn:
                return {"tags": kb["tags"]}
        return {"tags": {}}

    def delete_knowledge_base(self, knowledgeBaseId: str) -> None:
        if any(d["knowledgeBaseId"] == knowledgeBaseId for d in self.data_sources.values()):
            raise aws_error("ValidationException", "delete data sources first")
        del self.kbs[knowledgeBaseId]

    def create_data_source(self, knowledgeBaseId: str, name: str, **kwargs: Any) -> Dict[str, Any]:
        ds_id = new_id()
        ds = {"dataSourceId": ds_id, "knowledgeBaseId": knowledgeBaseId, "name": name, "status": "AVAILABLE", **kwargs}
        self.data_sources[ds_id] = ds
        return {"dataSource": ds}

    def get_data_source(self, knowledgeBaseId: str, dataSourceId: str) -> Dict[str, Any]:
        if dataSourceId not in self.data_sources:
            raise aws_error("ResourceNotFoundException")
        return {"dataSource": self.data_sources[dataSourceId]}

    def delete_data_source(self, knowledgeBaseId: str, dataSourceId: str) -> None:
        del self.data_sources[dataSourceId]

    def start_ingestion_job(self, knowledgeBaseId: str, dataSourceId: str, **kwargs: Any) -> Dict[str, Any]:
        if any(j["dataSourceId"] == dataSourceId and j["status"] in ("STARTING", "IN_PROGRESS") for j in self.jobs):
            raise aws_error("ConflictException", "There is already an ongoing ingestion job for this data source")
        started = datetime.now(timezone.utc) - timedelta(seconds=90)
        job = {
            "ingestionJobId": new_id(),
            "knowledgeBaseId": knowledgeBaseId,
            "dataSourceId": dataSourceId,
            "description": kwargs.get("description", ""),
            "status": "IN_PROGRESS" if self.job_polls else self.job_status,
            "statistics": dict(self.job_statistics),
            "failureReasons": list(self.job_failure_reasons) if not self.job_polls else [],
            "startedAt": started,
            "updatedAt": started + timedelta(seconds=75),
            "_polls_left": self.job_polls,
        }
        self.jobs.append(job)
        return {"ingestionJob": self._public(job)}

    @staticmethod
    def _public(job: Dict[str, Any]) -> Dict[str, Any]:
        """A job as Bedrock returns it, without this fake's bookkeeping."""
        return {k: v for k, v in job.items() if not k.startswith("_")}

    def get_ingestion_job(self, ingestionJobId: str, **_: Any) -> Dict[str, Any]:
        job = next((j for j in self.jobs if j["ingestionJobId"] == ingestionJobId), None)
        if job is None:
            raise aws_error("ResourceNotFoundException", f"Ingestion job {ingestionJobId} not found")
        if job["_polls_left"] > 0 and job["status"] == "IN_PROGRESS":
            job["_polls_left"] -= 1
            if job["_polls_left"] == 0:
                job.update(status=self.job_status, failureReasons=list(self.job_failure_reasons))
        return {"ingestionJob": self._public(job)}

    def list_ingestion_jobs(
        self, knowledgeBaseId: str, dataSourceId: str, maxResults: int = 1000, **_: Any
    ) -> Dict[str, Any]:
        mine = [j for j in reversed(self.jobs) if j["dataSourceId"] == dataSourceId]
        # Like Bedrock, summaries leave out failureReasons.
        summaries = [{k: v for k, v in self._public(j).items() if k != "failureReasons"} for j in mine[:maxResults]]
        response: Dict[str, Any] = {"ingestionJobSummaries": summaries}
        if len(mine) > maxResults:
            response["nextToken"] = "more"
        return response

    def stop_ingestion_job(self, knowledgeBaseId: str, dataSourceId: str, ingestionJobId: str) -> Dict[str, Any]:
        job = next(j for j in self.jobs if j["ingestionJobId"] == ingestionJobId)
        if job["status"] not in ("STARTING", "IN_PROGRESS"):
            raise aws_error("ConflictException", "The ingestion job is not in progress")
        job.update(status="STOPPED", _polls_left=0)
        return {"ingestionJob": self._public(job)}

    def list_knowledge_base_documents(self, knowledgeBaseId: str, dataSourceId: str, **_: Any) -> Dict[str, Any]:
        if self.documents_denied:
            raise aws_error("AccessDeniedException", "not authorized to perform: bedrock:ListKnowledgeBaseDocuments")
        return {"documentDetails": [d for d in self.documents if d["dataSourceId"] == dataSourceId]}


class FakeOpenSearchServerless:
    """OpenSearch Serverless's control plane: security/access policies, collections and their tags.

    Like the real service, a collection can only be created once an encryption policy covers it,
    and an access policy update must quote the current ``policyVersion``.
    """

    def __init__(self) -> None:
        self.security_policies: Dict[tuple, Dict[str, Any]] = {}
        self.access_policies: Dict[str, Dict[str, Any]] = {}
        self.collections: Dict[str, Dict[str, Any]] = {}
        #: The vector indexes inside each collection, keyed by endpoint (see ``FakeOpenSearch``).
        self.indexes: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def get_security_policy(self, name: str, type: str) -> Dict[str, Any]:
        if (type, name) not in self.security_policies:
            raise aws_error("ResourceNotFoundException")
        return {"securityPolicyDetail": self.security_policies[(type, name)]}

    def create_security_policy(self, name: str, type: str, policy: str, description: str = "") -> Dict[str, Any]:
        if (type, name) in self.security_policies:
            raise aws_error("ConflictException", "policy already exists")
        detail = {"name": name, "type": type, "policy": json.loads(policy), "description": description}
        self.security_policies[(type, name)] = detail
        return {"securityPolicyDetail": detail}

    def delete_security_policy(self, name: str, type: str) -> None:
        if (type, name) not in self.security_policies:
            raise aws_error("ResourceNotFoundException")
        del self.security_policies[(type, name)]

    def _encrypted(self, collection: str) -> bool:
        for (kind, _), detail in self.security_policies.items():
            rules = detail["policy"]["Rules"] if kind == "encryption" else []
            if any(f"collection/{collection}" in rule["Resource"] for rule in rules):
                return True
        return False

    def batch_get_collection(self, names: List[str]) -> Dict[str, Any]:
        return {"collectionDetails": [self.collections[n] for n in names if n in self.collections]}

    def create_collection(self, name: str, type: str, description: str = "", tags: Optional[list] = None) -> Any:
        if not self._encrypted(name):
            raise aws_error("ValidationException", f"No matching security policy of encryption type found for {name}")
        cid = new_id().lower()
        endpoint = f"https://{cid}.{REGION}.aoss.amazonaws.com"
        self.collections[name] = {
            "id": cid,
            "name": name,
            "arn": f"arn:aws:aoss:{REGION}:{ACCOUNT}:collection/{cid}",
            "status": "ACTIVE",
            "collectionEndpoint": endpoint,
            "tags": list(tags or []),
        }
        self.indexes[endpoint] = {}
        return {"createCollectionDetail": {"id": cid, "name": name, "status": "CREATING"}}

    def list_tags_for_resource(self, resourceArn: str) -> Dict[str, Any]:
        for collection in self.collections.values():
            if collection["arn"] == resourceArn:
                return {"tags": collection["tags"]}
        raise aws_error("ResourceNotFoundException")

    def delete_collection(self, id: str) -> None:
        name = next(n for n, c in self.collections.items() if c["id"] == id)
        self.indexes.pop(self.collections.pop(name)["collectionEndpoint"], None)

    def get_access_policy(self, name: str, type: str) -> Dict[str, Any]:
        if name not in self.access_policies:
            raise aws_error("ResourceNotFoundException")
        return {"accessPolicyDetail": self.access_policies[name]}

    def create_access_policy(self, name: str, type: str, policy: str, description: str = "") -> Dict[str, Any]:
        if name in self.access_policies:
            raise aws_error("ConflictException", "policy already exists")
        detail = {"name": name, "type": type, "policy": json.loads(policy), "policyVersion": new_id()}
        if description:
            detail["description"] = description
        self.access_policies[name] = detail
        return {"accessPolicyDetail": detail}

    def update_access_policy(self, name: str, type: str, policyVersion: str, policy: str, **_: Any) -> Any:
        detail = self.access_policies[name]
        if detail["policyVersion"] != policyVersion:
            raise aws_error("ConflictException", "policy version is out of date")
        detail.update(policy=json.loads(policy), policyVersion=new_id())
        return {"accessPolicyDetail": detail}

    def delete_access_policy(self, name: str, type: str) -> None:
        if name not in self.access_policies:
            raise aws_error("ResourceNotFoundException")
        del self.access_policies[name]


class FakeOpenSearch:
    """The data-plane client ``Aws.opensearch(endpoint)`` returns: just the index calls kbforge makes."""

    def __init__(self, indexes: Dict[str, Dict[str, Any]]):
        self.indices = self
        self._indexes = indexes

    def exists(self, index: str) -> bool:
        return index in self._indexes

    def create(self, index: str, body: Dict[str, Any]) -> Dict[str, Any]:
        self._indexes[index] = body
        return {"acknowledged": True, "index": index}

    def get_mapping(self, index: str) -> Dict[str, Any]:
        return {index: {"mappings": self._indexes[index]["mappings"]}}


class FakeAgentRuntime:
    def __init__(self) -> None:
        #: The last retrieve_and_generate request, so tests can check the model and settings sent.
        self.last_generate: Dict[str, Any] = {}

    def retrieve_and_generate(self, input: Dict[str, str], retrieveAndGenerateConfiguration: Dict[str, Any], **_: Any):
        self.last_generate = {"input": input, **retrieveAndGenerateConfiguration}
        leave = {
            "content": {"text": "Employees get <b>26 weeks</b> of parental leave."},
            "location": {"type": "S3", "s3Location": {"uri": "s3://docs/hr/leave.pdf"}},
        }
        notes = {
            "content": {"text": "Leave can be split into two blocks."},
            "location": {"type": "S3", "s3Location": {"uri": "s3://docs/hr/notes.md"}},
        }
        answer = "You get 26 weeks of parental leave. It can be taken in two blocks."
        first = answer.index(".")
        return {
            "output": {"text": answer},
            "citations": [
                {
                    "generatedResponsePart": {
                        "textResponsePart": {"text": answer[: first + 1], "span": {"start": 0, "end": first}}
                    },
                    "retrievedReferences": [leave],
                },
                {
                    "generatedResponsePart": {
                        "textResponsePart": {
                            "text": answer[first + 2 :],
                            "span": {"start": first + 2, "end": len(answer) - 1},
                        }
                    },
                    "retrievedReferences": [notes, leave],
                },
            ],
        }

    def retrieve(self, knowledgeBaseId: str, retrievalQuery: Dict[str, str], **_: Any) -> Dict[str, Any]:
        return {
            "retrievalResults": [
                {
                    "content": {"text": "Employees get <b>26 weeks</b> of parental leave."},
                    "location": {"type": "S3", "s3Location": {"uri": "s3://docs/hr/leave.pdf"}},
                    "score": 0.82,
                }
            ]
        }


class FakeAws:
    """Drop-in replacement for ``kbforge.aws.Aws`` backed by the fakes above."""

    def __init__(self) -> None:
        self.region = REGION
        self.session = None
        self.services: Dict[str, Any] = {
            "sts": FakeSts(),
            "iam": FakeIam(),
            "s3": FakeS3(),
            "bedrock-runtime": FakeBedrockRuntime(),
            "bedrock": FakeBedrock(),
            "s3vectors": FakeS3Vectors(),
            "bedrock-agent": FakeBedrockAgent(),
            "bedrock-agent-runtime": FakeAgentRuntime(),
            "opensearchserverless": FakeOpenSearchServerless(),
        }
        #: Who is "running the notebook". Tests change it to play a second teammate.
        self.principal = f"arn:aws:iam::{ACCOUNT}:role/SageMakerRole"
        self.services["s3"].buckets["my-company-docs"] = {
            "hr/leave.pdf": 120_000,
            "hr/handbook.docx": 340_000,
            "hr/notes.md": 2_000,
            "hr/leave.pdf.metadata.json": 100,
        }

    def client(self, service: str) -> Any:
        return self.services[service]

    @property
    def identity(self) -> Dict[str, str]:
        return self.services["sts"].get_caller_identity()

    @property
    def account_id(self) -> str:
        return ACCOUNT

    @property
    def caller_arn(self) -> str:
        return self.identity["Arn"]

    @property
    def principal_arn(self) -> str:
        return self.principal

    def opensearch(self, endpoint: str) -> FakeOpenSearch:
        return FakeOpenSearch(self.services["opensearchserverless"].indexes[endpoint])
