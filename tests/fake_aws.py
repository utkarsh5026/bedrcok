"""An in-memory stand-in for the AWS services kbforge uses.

It's deliberately small: just enough behaviour (create → exists → delete, not-found errors,
paginators) to run a full plan/apply/destroy cycle in tests without an AWS account.
Everything becomes ACTIVE/COMPLETE immediately, so tests never wait.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any, Dict, List, Optional

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

    def get_role(self, RoleName: str) -> Dict[str, Any]:
        if RoleName not in self.roles:
            raise aws_error("NoSuchEntity")
        return {"Role": self.roles[RoleName]}

    def create_role(self, RoleName: str, AssumeRolePolicyDocument: str, Tags: list, **_: Any) -> Dict[str, Any]:
        role = {
            "RoleName": RoleName,
            "Arn": f"arn:aws:iam::{ACCOUNT}:role/{RoleName}",
            "AssumeRolePolicyDocument": AssumeRolePolicyDocument,
            "Tags": Tags,
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


class FakeS3:
    def __init__(self) -> None:
        self.buckets: Dict[str, Dict[str, int]] = {}

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

    def invoke_model(self, **_: Any) -> Dict[str, Any]:
        if self.denied:
            raise aws_error("AccessDeniedException", "You don't have access to the model")
        return {"body": b"{}"}


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

    def start_ingestion_job(self, knowledgeBaseId: str, dataSourceId: str) -> Dict[str, Any]:
        job = {
            "ingestionJobId": new_id(),
            "knowledgeBaseId": knowledgeBaseId,
            "dataSourceId": dataSourceId,
            "status": "COMPLETE",
            "statistics": {"numberOfDocumentsScanned": 3, "numberOfNewDocumentsIndexed": 3},
        }
        self.jobs.append(job)
        return {"ingestionJob": job}

    def get_ingestion_job(self, ingestionJobId: str, **_: Any) -> Dict[str, Any]:
        return {"ingestionJob": next(j for j in self.jobs if j["ingestionJobId"] == ingestionJobId)}

    def list_ingestion_jobs(self, knowledgeBaseId: str, dataSourceId: str, **_: Any) -> Dict[str, Any]:
        mine = [j for j in self.jobs if j["dataSourceId"] == dataSourceId]
        return {"ingestionJobSummaries": list(reversed(mine))[:1]}


class FakeAgentRuntime:
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
            "s3vectors": FakeS3Vectors(),
            "bedrock-agent": FakeBedrockAgent(),
            "bedrock-agent-runtime": FakeAgentRuntime(),
        }
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
        return f"arn:aws:iam::{ACCOUNT}:role/SageMakerRole"
