"""Asking IAM "would this be allowed?" without trying it.

Some permissions can be checked by simply calling the API: if ``ListIngestionJobs`` works, you
may list syncs. Others can't, because trying them *does* the thing: the only way to test
``StartIngestionJob`` by calling it is to start a sync. IAM's policy simulator answers the
question from the policies alone (including permissions boundaries and organisation SCPs), so
kbforge uses it before a sync, both for your own role and for the role Bedrock uses to read your
documents.

The simulator needs ``iam:SimulatePrincipalPolicy``, which plenty of roles lack. Then every
answer is "unknown", and callers say so plainly instead of guessing.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from botocore.exceptions import ClientError

from .aws import Aws, error_code
from .steps.base import ACCESS_DENIED

ALLOWED, DENIED, UNKNOWN = "allowed", "denied", "unknown"

#: One question for the simulator: may the principal do this action on this resource?
Question = Tuple[str, str]


def simulate(
    aws: Aws, principal_arn: str, questions: Iterable[Question], context: Optional[Dict[str, str]] = None
) -> Dict[Question, str]:
    """Ask IAM whether ``principal_arn`` may do each (action, resource) pair. Never changes anything.

    Each answer is ``ALLOWED``, ``DENIED`` or ``UNKNOWN``. Unknown means the simulator couldn't
    be used (no permission, or a principal it can't simulate such as the root user), or that a
    policy condition depends on something only known at request time. ``context`` fills in
    condition keys we do know, such as ``aws:ResourceAccount``.
    """
    asked = list(questions)
    by_resource: Dict[str, List[str]] = defaultdict(list)
    for action, resource in asked:
        by_resource[resource].append(action)
    entries = [
        {"ContextKeyName": key, "ContextKeyValues": [value], "ContextKeyType": "string"}
        for key, value in (context or {}).items()
    ]
    answers: Dict[Question, str] = {}
    iam = aws.client("iam")
    try:
        for resource, actions in by_resource.items():
            marker = ""
            while True:
                kwargs: Dict[str, Any] = {
                    "PolicySourceArn": principal_arn,
                    "ActionNames": actions,
                    "ResourceArns": [resource],
                }
                if entries:
                    kwargs["ContextEntries"] = entries
                if marker:
                    kwargs["Marker"] = marker
                response = iam.simulate_principal_policy(**kwargs)
                for result in response.get("EvaluationResults", []):
                    answers[(result["EvalActionName"], resource)] = _verdict(result)
                marker = response.get("Marker", "") if response.get("IsTruncated") else ""
                if not marker:
                    break
    except ClientError as exc:
        if error_code(exc) in (*ACCESS_DENIED, "NoSuchEntity", "InvalidInput"):
            return {question: UNKNOWN for question in asked}
        raise
    return {question: answers.get(question, UNKNOWN) for question in asked}


def _verdict(result: Dict[str, Any]) -> str:
    """Read one simulator result: allowed, denied, or unknown when a condition couldn't be evaluated."""
    decision = result.get("EvalDecision")
    if decision == "allowed":
        return ALLOWED
    if decision == "explicitDeny":
        return DENIED
    # An implicit deny with missing context means "allowed if the condition holds", which we can't know.
    return UNKNOWN if result.get("MissingContextValues") else DENIED
