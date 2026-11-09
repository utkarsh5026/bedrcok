"""A small wrapper around boto3 that every other module uses to talk to AWS.

Keeping all AWS access behind one ``Aws`` object means:

* clients are created once and reused,
* "who am I?" (account, ARN) is looked up once,
* tests can hand in a fake instead of real AWS.

It also holds two helpers for the slow, eventually-consistent parts of AWS: ``wait_for``
(poll until something is ready) and ``retry_on`` (retry a call that fails only because a
just-created resource isn't visible yet).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Tuple

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

_CLIENT_CONFIG = Config(retries={"max_attempts": 8, "mode": "adaptive"}, user_agent_extra="kbforge")


def error_code(exc: BaseException) -> str:
    """The AWS error code (e.g. ``AccessDeniedException``) or the exception's class name."""
    if isinstance(exc, ClientError):
        return exc.response.get("Error", {}).get("Code", "")
    return type(exc).__name__


def error_message(exc: BaseException) -> str:
    """The human-readable part of an AWS error."""
    if isinstance(exc, ClientError):
        return exc.response.get("Error", {}).get("Message", str(exc))
    return str(exc)


class Aws:
    """One AWS session for one region, plus cached identity information."""

    def __init__(self, region: Optional[str] = None, session: Optional[boto3.Session] = None):
        self.session = session or boto3.Session(region_name=region)
        self.region: str = region or self.session.region_name or ""
        if not self.region:
            raise RuntimeError("No AWS region configured. Pass region='us-east-1' (or set AWS_DEFAULT_REGION).")
        self._clients: Dict[str, Any] = {}
        self._identity: Optional[Dict[str, str]] = None
        self._principal_arn: Optional[str] = None

    def client(self, service: str) -> Any:
        """A boto3 client for ``service`` in our region, created on first use."""
        if service not in self._clients:
            self._clients[service] = self.session.client(service, region_name=self.region, config=_CLIENT_CONFIG)
        return self._clients[service]

    @property
    def identity(self) -> Dict[str, str]:
        """``{"Account": ..., "Arn": ...}`` for whoever is running this code."""
        if self._identity is None:
            resp = self.client("sts").get_caller_identity()
            self._identity = {"Account": resp["Account"], "Arn": resp["Arn"]}
        return self._identity

    @property
    def account_id(self) -> str:
        """The 12-digit AWS account id."""
        return self.identity["Account"]

    @property
    def caller_arn(self) -> str:
        """The ARN AWS sees for the current credentials."""
        return self.identity["Arn"]

    @property
    def principal_arn(self) -> str:
        """The IAM role/user ARN of whoever is running this notebook.

        In SageMaker you run as an *assumed role*, whose ARN looks like
        ``arn:aws:sts::123:assumed-role/MyRole/SageMaker``. Resource policies (like the
        OpenSearch data access policy) need the underlying *role* ARN instead — including its
        path, which is often ``service-role/`` for SageMaker roles — so we look it up.
        """
        if self._principal_arn is not None:
            return self._principal_arn
        arn = self.caller_arn
        if ":assumed-role/" in arn:
            role_name = arn.split(":assumed-role/", 1)[1].split("/", 1)[0]
            try:
                arn = self.client("iam").get_role(RoleName=role_name)["Role"]["Arn"]
            except ClientError:
                # No iam:GetRole permission: assume the role has no path. Right most of the time.
                arn = f"arn:aws:iam::{self.account_id}:role/{role_name}"
        self._principal_arn = arn
        return arn


def wait_for(
    poll: Callable[[], Any],
    done: Callable[[Any], bool],
    *,
    what: str,
    log: Callable[[str], None],
    timeout: float = 900,
    interval: float = 5,
    failed: Optional[Callable[[Any], Any]] = None,
    describe: Optional[Callable[[Any], str]] = None,
) -> Any:
    """Call ``poll()`` repeatedly until ``done(result)`` is true, then return the result.

    ``failed(result)`` can return a reason string to stop early with an error.
    ``describe(result)`` adds the current state to the progress message, so the user sees
    e.g. "Waiting for ingestion · in progress · 40 scanned (120s)" instead of a frozen cell.
    """
    start = time.time()
    while True:
        result = poll()
        if done(result):
            return result
        if failed is not None:
            reason = failed(result)
            if reason:
                raise RuntimeError(f"{what} failed: {reason}")
        elapsed = int(time.time() - start)
        if elapsed > timeout:
            raise TimeoutError(f"Timed out after {elapsed}s waiting for {what}")
        state = describe(result) if describe else ""
        log(f"Waiting for {what}{' · ' + state if state else ''} ({elapsed}s)")
        time.sleep(interval)


def retry_on(
    fn: Callable[[], Any],
    *,
    codes: Tuple[str, ...],
    log: Callable[[str], None],
    why: str,
    hints: Tuple[str, ...] = (),
    attempts: int = 12,
    delay: float = 10,
) -> Any:
    """Call ``fn()``, retrying when it fails with one of ``codes``.

    Used for AWS's eventual consistency: e.g. a role created seconds ago may not be usable
    by Bedrock yet. ``why`` is shown to the user while we wait.

    If ``hints`` is given, we only retry when the error message contains one of those words,
    so a genuine mistake fails straight away instead of after minutes of retrying.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except ClientError as exc:
            message = error_message(exc).lower()
            retryable = error_code(exc) in codes and (not hints or any(h in message for h in hints))
            if not retryable or attempt == attempts:
                raise
            log(f"{why} — retrying in {int(delay)}s (attempt {attempt}/{attempts})")
            time.sleep(delay)
