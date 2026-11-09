"""Preflight checks: catch the usual "why did my knowledge base fail?" problems up front.

Each check returns a ``Check`` with a status (ok / warn / fail) and a plain-English detail.
A single ``fail`` blocks ``apply()``; warnings are shown but don't block.

What we check:

* your AWS credentials work (and which account/region you're in),
* the embedding model can actually be invoked (catches "model access not enabled"),
* the source bucket exists, is in the right region, and contains files Bedrock can read,
* the chosen vector store is available,
* a role you supplied trusts Bedrock.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from botocore.exceptions import ClientError

from .aws import Aws, error_code, error_message
from .spec import KBSpec, S3Source, WebSource

SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".doc", ".docx", ".csv", ".xls", ".xlsx", ".pdf"}
MAX_FILE_BYTES = 50 * 1024 * 1024
SCAN_LIMIT = 20_000

OK, WARN, FAIL = "ok", "warn", "fail"


@dataclass
class Check:
    """The result of one preflight check."""

    status: str
    title: str
    detail: str = ""
    # Optional structured payload for the renderer (e.g. file-type histogram).
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceScan:
    """What we found when listing the source bucket."""

    files: int = 0
    bytes: int = 0
    by_ext: Counter = field(default_factory=Counter)
    unsupported: int = 0
    too_large: List[str] = field(default_factory=list)
    metadata_files: int = 0
    truncated: bool = False

    @property
    def supported(self) -> int:
        """How many files Bedrock will actually be able to ingest."""
        return self.files - self.unsupported - len(self.too_large)


def human_bytes(n: float) -> str:
    """Format a byte count as e.g. ``48.2 MB``."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _check_identity(aws: Aws) -> Check:
    """Can we call AWS at all? Everything else depends on this."""
    try:
        return Check(OK, "AWS credentials", f"{aws.caller_arn} · account {aws.account_id} · {aws.region}")
    except Exception as exc:
        return Check(FAIL, "AWS credentials", f"Couldn't call STS: {error_message(exc)}")


def _check_model(spec: KBSpec, aws: Aws) -> Check:
    """Embed the word "hello" — the only reliable way to know the model is enabled for you."""
    model = spec.model
    title = f"Embedding model · {model.label}"
    try:
        aws.client("bedrock-runtime").invoke_model(
            modelId=model.model_id, body=model.probe_body(), contentType="application/json", accept="application/json"
        )
        return Check(OK, title, f"{model.model_id} responded · {spec.dims} dimensions")
    except ClientError as exc:
        code = error_code(exc)
        if code == "AccessDeniedException":
            hint = (
                "Your role can't invoke this model, or model access isn't enabled for the account. "
                "Check Bedrock → Model access, and that your role allows bedrock:InvokeModel."
            )
        elif code in ("ValidationException", "ResourceNotFoundException"):
            hint = f"The model isn't available in {aws.region}. Try another region or model."
        else:
            hint = error_message(exc)
        return Check(FAIL, title, hint)


def scan_s3(aws: Aws, bucket: str, prefix: str) -> SourceScan:
    """List the source files and tally them by type and size (stops after 20,000 files)."""
    scan = SourceScan()
    paginator = aws.client("s3").get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            if key.endswith(".metadata.json"):
                scan.metadata_files += 1
                continue
            scan.files += 1
            scan.bytes += obj["Size"]
            ext = os.path.splitext(key)[1].lower() or "(none)"
            scan.by_ext[ext] += 1
            if ext not in SUPPORTED_EXTENSIONS:
                scan.unsupported += 1
            elif obj["Size"] > MAX_FILE_BYTES:
                scan.too_large.append(key)
            if scan.files >= SCAN_LIMIT:
                scan.truncated = True
                return scan
    return scan


def _check_s3_source(src: S3Source, aws: Aws) -> List[Check]:
    """Does the bucket exist, is it in our region, and does it hold files Bedrock can read?"""
    s3 = aws.client("s3")
    checks: List[Check] = []
    try:
        loc = s3.get_bucket_location(Bucket=src.bucket).get("LocationConstraint") or "us-east-1"
    except ClientError as exc:
        code = error_code(exc)
        if code == "NoSuchBucket":
            return [Check(FAIL, "Source bucket", f"Bucket {src.bucket!r} doesn't exist.")]
        return [Check(FAIL, "Source bucket", f"Can't read bucket {src.bucket!r}: {error_message(exc)}")]
    if loc != aws.region:
        checks.append(
            Check(
                WARN,
                "Source bucket region",
                f"Bucket is in {loc} but the knowledge base will be in {aws.region}. "
                "Bedrock requires them to match — use region='" + loc + "' or copy the data.",
            )
        )
    try:
        scan = scan_s3(aws, src.bucket, src.prefix)
    except ClientError as exc:
        return checks + [Check(FAIL, "Source documents", f"Can't list {src.uri}: {error_message(exc)}")]

    data = {
        "files": scan.files,
        "bytes": scan.bytes,
        "by_ext": scan.by_ext.most_common(8),
        "supported_exts": sorted(SUPPORTED_EXTENSIONS),
    }
    if scan.files == 0:
        checks.append(Check(FAIL, "Source documents", f"No files found under {src.uri}.", data))
        return checks
    more = "+" if scan.truncated else ""
    summary = f"{scan.files:,}{more} files · {human_bytes(scan.bytes)}{more}"
    if scan.metadata_files:
        summary += f" · {scan.metadata_files} metadata sidecars"
    status, notes = OK, []
    if scan.unsupported:
        status = WARN
        notes.append(f"{scan.unsupported} file(s) have types Bedrock will skip")
    if scan.too_large:
        status = WARN
        notes.append(f"{len(scan.too_large)} file(s) exceed 50 MB and will fail (e.g. {scan.too_large[0]})")
    if scan.supported == 0:
        status = FAIL
        notes.append("nothing here can be ingested")
    detail = summary + ("" if not notes else " — " + "; ".join(notes))
    checks.append(Check(status, "Source documents", detail, data))
    return checks


def _check_web_source(src: WebSource) -> List[Check]:
    """We can't test-crawl from here, so just summarise the seeds and remind about permission."""
    return [
        Check(
            OK,
            "Web crawler seeds",
            f"{len(src.urls)} seed URL(s), scope {src.scope}. Only crawl sites you're allowed to.",
        )
    ]


def _check_vector_store(spec: KBSpec, aws: Aws) -> Check:
    """Is the chosen vector store reachable in this region (and its client library installed)?"""
    if spec.vector_store == "s3_vectors":
        try:
            aws.client("s3vectors").list_vector_buckets(maxResults=1)
            return Check(OK, "Vector store · S3 Vectors", f"Available in {aws.region}")
        except ClientError as exc:
            if error_code(exc) == "AccessDeniedException":
                return Check(FAIL, "Vector store · S3 Vectors", "Your role lacks s3vectors:* permissions.")
            return Check(FAIL, "Vector store · S3 Vectors", error_message(exc))
        except Exception as exc:  # endpoint missing in region
            return Check(
                FAIL,
                "Vector store · S3 Vectors",
                f"S3 Vectors isn't reachable in {aws.region} ({exc}). Use vector_store='opensearch_serverless'.",
            )
    try:
        import opensearchpy  # noqa: F401
    except ImportError:
        return Check(
            FAIL,
            "Vector store · OpenSearch Serverless",
            "Needs the OpenSearch client: pip install 'kbforge[opensearch]', then restart the kernel.",
        )
    return Check(
        OK,
        "Vector store · OpenSearch Serverless",
        f"Index will be created as {aws.principal_arn}. Note: a collection bills ~2+ OCUs continuously.",
    )


def _check_role(spec: KBSpec, aws: Aws) -> Optional[Check]:
    """If the user brought their own role, make sure Bedrock is allowed to assume it."""
    if not spec.role_arn:
        return None
    name = spec.role_arn.rsplit("/", 1)[-1]
    try:
        role = aws.client("iam").get_role(RoleName=name)["Role"]
    except ClientError as exc:
        if error_code(exc) == "NoSuchEntity":
            return Check(FAIL, "Your IAM role", f"{spec.role_arn} doesn't exist.")
        return Check(WARN, "Your IAM role", f"Couldn't inspect it ({error_code(exc)}); assuming it's set up correctly.")
    trust = str(role.get("AssumeRolePolicyDocument", ""))
    if "bedrock.amazonaws.com" not in trust:
        return Check(FAIL, "Your IAM role", "Its trust policy doesn't allow bedrock.amazonaws.com to assume it.")
    return Check(OK, "Your IAM role", "Trusts bedrock.amazonaws.com. Make sure it has the permissions listed below.")


def run_checks(spec: KBSpec, aws: Aws, identity_only: bool = False) -> List[Check]:
    """Run every relevant check, in the order they're shown.

    ``identity_only`` skips everything but the credentials check (used when destroying,
    where the model and source don't matter).
    """
    ident = _check_identity(aws)
    if ident.status == FAIL or identity_only:
        return [ident]
    checks = [ident, _check_model(spec, aws)]
    if isinstance(spec.source, S3Source):
        checks += _check_s3_source(spec.source, aws)
    else:
        checks += _check_web_source(spec.source)
    checks.append(_check_vector_store(spec, aws))
    role = _check_role(spec, aws)
    if role:
        checks.append(role)
    return checks
