"""The IAM role Bedrock uses to do its job.

When a knowledge base syncs, it is *Bedrock* — not you — that reads your documents, calls
the embedding model and writes vectors. It does that by assuming this role, so the role
needs exactly those three permissions and nothing more (plus the parsing model or Bedrock
Data Automation, if the spec uses one to read documents).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from botocore.exceptions import ClientError

from ..aws import error_code
from ..spec import S3Source
from .base import Action, Ctx, Inspection, Log, Step, claimed_by, is_ours, shared_name_conflict, tag_dict


class RoleStep(Step):
    """Create (or reuse) the service role and keep its permissions in sync with the spec."""

    title = "IAM service role"

    def resource(self, ctx: Ctx) -> str:
        """The role name, or the ARN you supplied."""
        return ctx.spec.role_arn or ctx.names.role

    def explain(self, ctx: Ctx) -> str:
        """What the role is for (or what your own role must allow)."""
        if ctx.spec.role_arn:
            return (
                "Uses the role you supplied. It must trust bedrock.amazonaws.com and have the permissions shown below."
            )
        models = "the embedding model" if ctx.spec.parsing.strategy == "default" else "the embedding model and parser"
        return f"The identity Bedrock assumes to read your documents, call {models} and write vectors."

    def details(self, ctx: Ctx) -> Optional[Tuple[str, str]]:
        """The permissions policy as JSON, for the plan's fold-out."""
        return ("Permissions policy", json.dumps(self.permissions_policy(ctx), indent=2))

    def trust_policy(self, ctx: Ctx) -> Dict[str, Any]:
        """Who may assume the role: only Bedrock, only for knowledge bases in this account and region."""
        acct, region = ctx.aws.account_id, ctx.aws.region
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "bedrock.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                    "Condition": {
                        "StringEquals": {"aws:SourceAccount": acct},
                        "ArnLike": {"aws:SourceArn": f"arn:aws:bedrock:{region}:{acct}:knowledge-base/*"},
                    },
                }
            ],
        }

    def permissions_policy(self, ctx: Ctx) -> Dict[str, Any]:
        """What the role may do: invoke the embedding model (and parser), read the source, use the vector store."""
        spec, region, acct = ctx.spec, ctx.aws.region, ctx.aws.account_id
        statements: List[Dict[str, Any]] = [
            {
                "Sid": "InvokeEmbeddingModel",
                "Effect": "Allow",
                "Action": ["bedrock:InvokeModel"],
                "Resource": [spec.model.arn(region)],
            }
        ]
        statements += self.parsing_statements(ctx)
        if isinstance(spec.source, S3Source):
            bucket = f"arn:aws:s3:::{spec.source.bucket}"
            statements.append(
                {
                    "Sid": "ReadSourceDocuments",
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:ListBucket"],
                    "Resource": [bucket, f"{bucket}/*"],
                    "Condition": {"StringEquals": {"aws:ResourceAccount": acct}},
                }
            )
        if spec.vector_store == "s3_vectors":
            statements.append(
                {
                    "Sid": "ReadWriteVectors",
                    "Effect": "Allow",
                    "Action": [
                        "s3vectors:PutVectors",
                        "s3vectors:GetVectors",
                        "s3vectors:DeleteVectors",
                        "s3vectors:QueryVectors",
                        "s3vectors:GetIndex",
                    ],
                    "Resource": [ctx.index_arn],
                }
            )
        else:
            # The collection ARN contains a random id, so until it exists we scope to the account.
            collection = ctx.out.get("collection_arn") or f"arn:aws:aoss:{region}:{acct}:collection/*"
            statements.append(
                {
                    "Sid": "OpenSearchServerless",
                    "Effect": "Allow",
                    "Action": ["aoss:APIAccessAll"],
                    "Resource": [collection],
                }
            )
        return {"Version": "2012-10-17", "Statement": statements}

    def parsing_statements(self, ctx: Ctx) -> List[Dict[str, Any]]:
        """What the role needs to read documents with an advanced parser (nothing for the default one)."""
        parsing, region, acct = ctx.spec.parsing, ctx.aws.region, ctx.aws.account_id
        parser = parsing.parser
        if parser is not None:
            return [
                {
                    "Sid": "InvokeParsingModel",
                    "Effect": "Allow",
                    "Action": ["bedrock:InvokeModel"],
                    "Resource": parser.iam_resources(region, acct),
                }
            ]
        if parsing.strategy == "data_automation":
            # Bedrock's documented statements for Data Automation; "*" in the account field also
            # covers the account-less ARNs AWS-owned profiles use.
            return [
                {
                    "Sid": "DataAutomationInvoke",
                    "Effect": "Allow",
                    "Action": ["bedrock:InvokeDataAutomationAsync"],
                    "Resource": [
                        f"arn:aws:bedrock:{region}:aws:data-automation-project/public-rag-default",
                        f"arn:aws:bedrock:{region}:*:data-automation-profile/*",
                    ],
                },
                {
                    "Sid": "DataAutomationStatus",
                    "Effect": "Allow",
                    "Action": ["bedrock:GetDataAutomationStatus"],
                    "Resource": [f"arn:aws:bedrock:{region}:*:data-automation-invocation/*"],
                },
            ]
        return []

    def inspect(self, ctx: Ctx) -> Inspection:
        """Use your role, or check ours exists, belongs to this knowledge base and has up-to-date permissions."""
        if ctx.spec.role_arn:
            ctx.out["role_arn"] = ctx.spec.role_arn
            return Inspection(Action.USE, "your role")
        try:
            role = ctx.aws.client("iam").get_role(RoleName=ctx.names.role)["Role"]
        except ClientError as exc:
            if error_code(exc) == "NoSuchEntity":
                return Inspection(Action.CREATE)
            raise
        # Updating another knowledge base's role would swap its permissions for ours.
        owner = claimed_by(tag_dict(role.get("Tags", [])), ctx)
        if owner:
            return shared_name_conflict(owner, ctx)
        ctx.out["role_arn"] = role["Arn"]
        if self._current_policy(ctx) == self.permissions_policy(ctx):
            return Inspection(Action.EXISTS)
        return Inspection(Action.UPDATE, "exists · permissions will be brought up to date")

    def _current_policy(self, ctx: Ctx) -> Optional[Dict[str, Any]]:
        """The role's kbforge policy as it is in AWS today, or None if it's missing."""
        try:
            doc = ctx.aws.client("iam").get_role_policy(RoleName=ctx.names.role, PolicyName=ctx.names.role_policy)
        except ClientError as exc:
            if error_code(exc) == "NoSuchEntity":
                return None
            raise
        document = doc["PolicyDocument"]
        # boto3 normally decodes the document for us, but be tolerant of a raw JSON string.
        return json.loads(document) if isinstance(document, str) else document

    def create(self, ctx: Ctx, log: Log, inspection: Inspection) -> str:
        """Create the role if needed, then attach the current permissions policy."""
        iam = ctx.aws.client("iam")
        if inspection.action == Action.CREATE:
            log("Creating role")
            kwargs: Dict[str, Any] = dict(
                RoleName=ctx.names.role,
                AssumeRolePolicyDocument=json.dumps(self.trust_policy(ctx)),
                Description=f"Bedrock knowledge base '{ctx.spec.name}' (managed by kbforge)",
                Tags=[{"Key": k, "Value": v} for k, v in ctx.tags.items()],
            )
            if ctx.spec.permissions_boundary:
                kwargs["PermissionsBoundary"] = ctx.spec.permissions_boundary
            ctx.out["role_arn"] = iam.create_role(**kwargs)["Role"]["Arn"]
        log("Attaching permissions")
        iam.put_role_policy(
            RoleName=ctx.names.role,
            PolicyName=ctx.names.role_policy,
            PolicyDocument=json.dumps(self.permissions_policy(ctx)),
        )
        return ctx.out["role_arn"]

    def delete(self, ctx: Ctx, log: Log) -> str:
        # IAM refuses to delete a role that still has policies, so remove those first.
        """Remove the role's policies, then the role."""
        iam = ctx.aws.client("iam")
        name = ctx.names.role
        for policy in iam.list_role_policies(RoleName=name)["PolicyNames"]:
            iam.delete_role_policy(RoleName=name, PolicyName=policy)
        for attached in iam.list_attached_role_policies(RoleName=name)["AttachedPolicies"]:
            iam.detach_role_policy(RoleName=name, PolicyArn=attached["PolicyArn"])
        iam.delete_role(RoleName=name)
        return "deleted"

    def owned(self, ctx: Ctx) -> Optional[bool]:
        """True if the role carries kbforge's tags for this knowledge base."""
        role = ctx.aws.client("iam").get_role(RoleName=ctx.names.role)["Role"]
        return is_ours(tag_dict(role.get("Tags", [])), ctx)
