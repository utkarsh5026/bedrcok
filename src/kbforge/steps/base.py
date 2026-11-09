"""The building blocks every step shares.

A knowledge base is really 4-8 separate AWS resources (an IAM role, a vector index, the
KB itself, ...). Each one is modelled as a ``Step``. Every step can answer two questions:

1. ``inspect()`` — "What would you do right now?"  It looks at AWS and returns an
   ``Inspection`` such as CREATE, EXISTS or CONFLICT. It never changes anything, which is
   why ``kb.plan()`` is always safe to run.
2. ``create()`` / ``delete()`` — "Do it."  Called by ``Plan.apply()`` only for steps whose
   inspection said there is work to do.

Steps pass information to each other (e.g. the role ARN the knowledge base needs) through
the shared ``Ctx.out`` dictionary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from botocore.exceptions import ClientError

from ..aws import Aws, error_code, error_message
from ..spec import KBSpec, Names

#: A callback a step uses to report progress, e.g. ``log("Waiting for index (30s)")``.
Log = Callable[[str], None]

#: Every resource kbforge creates carries this tag, so ``destroy`` can tell ours from yours.
MANAGED_TAG = ("managed-by", "kbforge")

#: Tag naming the knowledge base a resource was made for. Resource names come from a lowercase
#: form of the KB name, so ``HR_Policies`` and ``hr-policies`` map to the same bucket, role and
#: collection. This tag is how a step tells "ours" apart from "another knowledge base's".
KB_TAG = "kbforge-kb"

#: Error codes that mean "you're not allowed to look", as opposed to "it isn't there".
ACCESS_DENIED = ("AccessDenied", "AccessDeniedException", "UnauthorizedOperation")

#: Error codes that mean the notebook's AWS session has run out and needs refreshing.
EXPIRED_CREDENTIALS = ("ExpiredToken", "ExpiredTokenException", "RequestExpired", "TokenRefreshRequired")

#: Error codes (and botocore exception names) that mean there are no usable AWS credentials at all.
BAD_CREDENTIALS = (
    "UnrecognizedClientException",
    "InvalidClientTokenId",
    "InvalidSignatureException",
    "SignatureDoesNotMatch",
    "NoCredentialsError",
    "PartialCredentialsError",
    "CredentialRetrievalError",
)

#: Error codes that mean "AWS is busy, slow down"; botocore already retried before we see these.
THROTTLED = ("ThrottlingException", "TooManyRequestsException", "Throttling", "RequestLimitExceeded")


class Action(str, Enum):
    """What a step is going to do (or did). Shown as a coloured badge in the plan."""

    # Used when building:
    CREATE = "create"  # resource is missing and will be made
    UPDATE = "update"  # resource exists but will be adjusted (e.g. role permissions refreshed)
    EXISTS = "exists"  # resource is already there and correct — nothing to do
    USE = "use"  # user supplied it (e.g. role_arn=...) — we just use it
    RUN = "run"  # an action rather than a resource (e.g. start an ingestion job)
    SKIP = "skip"  # nothing to do, with a note explaining why
    CONFLICT = "conflict"  # something exists but doesn't match the spec — blocks apply
    # Used when destroying:
    DELETE = "delete"  # will be deleted
    ABSENT = "absent"  # already gone
    KEEP = "keep"  # exists but we won't delete it (not ours, or user-supplied)

    @property
    def does_work(self) -> bool:
        """True if applying the plan will call AWS to change something for this step."""
        return self in (Action.CREATE, Action.UPDATE, Action.RUN, Action.DELETE)


@dataclass
class Inspection:
    """The answer a step gives to "what would you do?".

    ``note`` is a short friendly remark shown next to the badge.
    ``warning`` is something the user should read (shown highlighted).
    """

    action: Action
    note: str = ""
    warning: str = ""
    #: True when the step couldn't look at AWS at all (throttling, a missing permission), so
    #: the action is only a placeholder. Destroy refuses to go ahead on a guess.
    error: bool = False


@dataclass
class Ctx:
    """Shared state for one plan/apply run.

    ``out`` collects facts discovered along the way — ARNs and IDs such as ``role_arn``,
    ``index_arn``, ``kb_id`` and ``ds_id`` — so later steps can use what earlier steps found
    or created.
    """

    spec: KBSpec
    aws: Aws
    out: Dict[str, Any] = field(default_factory=dict)

    @property
    def names(self) -> Names:
        """The AWS resource names derived from the KB name (see ``KBSpec.names``)."""
        return self.spec.names

    @property
    def tags(self) -> Dict[str, str]:
        """Tags put on everything we create: the kbforge marker plus the user's own tags."""
        return {MANAGED_TAG[0]: MANAGED_TAG[1], KB_TAG: self.spec.name, **self.spec.tags}

    @property
    def role_arn(self) -> str:
        """The role Bedrock will use: the user's own, the one we found/created, or the one we will create."""
        if self.spec.role_arn:
            return self.spec.role_arn
        return self.out.get("role_arn") or f"arn:aws:iam::{self.aws.account_id}:role/{self.names.role}"

    @property
    def index_arn(self) -> str:
        """The S3 Vectors index ARN. We can compute it before the index exists, which lets the plan show it."""
        return self.out.get("index_arn") or (
            f"arn:aws:s3vectors:{self.aws.region}:{self.aws.account_id}:"
            f"bucket/{self.names.vector_bucket}/index/{self.names.vector_index}"
        )


def is_managed(tags: Dict[str, str]) -> bool:
    """True if a resource's tags show kbforge created it (for any knowledge base)."""
    return tags.get(MANAGED_TAG[0]) == MANAGED_TAG[1]


def tag_dict(tags: List[Dict[str, str]]) -> Dict[str, str]:
    """Turn AWS's list-of-pairs tag format into a plain dict.

    IAM uses ``{"Key", "Value"}`` and OpenSearch Serverless ``{"key", "value"}``; both work.
    """
    return {t.get("Key", t.get("key", "")): t.get("Value", t.get("value", "")) for t in tags}


def claimed_by(tags: Dict[str, str], ctx: Ctx) -> Optional[str]:
    """The name of a *different* knowledge base this resource was made for, or None.

    None means the resource is free for this knowledge base to use: it was made for this one,
    or kbforge didn't make it (that case is handled by ``owned`` when destroying).
    """
    owner = tags.get(KB_TAG)
    if is_managed(tags) and owner and owner != ctx.spec.name:
        return owner
    return None


def is_ours(tags: Dict[str, str], ctx: Ctx) -> bool:
    """True if kbforge created this resource *for this knowledge base*, so destroy may remove it."""
    return is_managed(tags) and claimed_by(tags, ctx) is None


def shared_name_conflict(owner: str, ctx: Ctx) -> Inspection:
    """The blocker shown when a resource kbforge would use already belongs to another knowledge base."""
    return Inspection(
        Action.CONFLICT,
        warning=(
            f"This already belongs to knowledge base '{owner}'. kbforge names resources from a lowercase "
            f"form of the KB name, so '{ctx.spec.name}' and '{owner}' would share it. Pick a name that "
            "differs by more than upper/lower case or '-' versus '_'."
        ),
    )


def tags_if_allowed(fetch: Callable[[], Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Read a resource's tags, or return None if we aren't allowed to.

    Used by ``inspect`` to spot a resource that belongs to another knowledge base. Lacking
    permission to read tags shouldn't stop a plan, so the check is skipped instead.
    """
    try:
        return fetch()
    except ClientError as exc:
        if error_code(exc) in ACCESS_DENIED:
            return None
        raise


class Step:
    """One AWS resource (or action) in the knowledge-base recipe.

    Subclasses set ``title`` and implement ``inspect``, ``create`` and ``delete``.
    The display helpers (``resource``, ``explain``, ``details``) feed the plan UI.
    """

    #: Short human name shown in the plan, e.g. "IAM service role".
    title = ""
    #: False for things that disappear with their parent (an index inside a collection)
    #: or that aren't resources at all (an ingestion run). ``destroy`` skips these.
    deletable = True

    def resource(self, ctx: Ctx) -> str:
        """The concrete name/ARN this step manages, e.g. ``kbforge-hr-policies``."""
        return ""

    def explain(self, ctx: Ctx) -> str:
        """One sentence, in plain English, about what this resource is for."""
        return ""

    def details(self, ctx: Ctx) -> Optional[Tuple[str, str]]:
        """Optional ``(label, text)`` shown in a fold-out, e.g. the IAM policy JSON."""
        return None

    def inspect(self, ctx: Ctx) -> Inspection:
        """Look at AWS (read-only) and decide what this step would do."""
        raise NotImplementedError

    def create(self, ctx: Ctx, log: Log, inspection: Inspection) -> str:
        """Do the work promised by ``inspection``. Returns a short result shown in the UI."""
        raise NotImplementedError

    def delete(self, ctx: Ctx, log: Log) -> str:
        """Remove the resource. Only called if ``inspect`` found it."""
        raise NotImplementedError

    def owned(self, ctx: Ctx) -> Optional[bool]:
        """Did kbforge create this for this knowledge base?

        True/False from tags (or from a parent resource's tags, for things that can't be
        tagged), or None when there is nothing to go on. Raising means "couldn't tell", and
        destroy then leaves the resource alone.
        """
        return None


def explain_error(exc: BaseException) -> str:
    """Turn the most common AWS failures into a message a notebook user can act on."""
    code, msg = error_code(exc), error_message(exc)
    if code in ACCESS_DENIED:
        return (
            f"{msg}\n\nYour notebook's role is missing a permission. Ask your AWS admin to grant the "
            "action named above, or pass role_arn=... to reuse an existing Bedrock role."
        )
    if code == "ServiceQuotaExceededException":
        return f"{msg}\n\nYou've hit an account quota. Delete unused resources or request a quota increase."
    if code in EXPIRED_CREDENTIALS:
        return (
            "Your AWS session has expired. Refresh your credentials (in SageMaker, restart the kernel; "
            "elsewhere, log in again, e.g. `aws sso login`), then run the same command again."
        )
    if code in BAD_CREDENTIALS:
        return (
            f"AWS didn't accept your credentials ({msg.rstrip('.')}).\n\n"
            "Check that this notebook or terminal is signed in to the right AWS account (`aws sts get-caller-identity` shows who you are), then try again."
        )
    if code in THROTTLED:
        return f"{msg}\n\nAWS is busy right now. Wait a minute and run the same command again."
    if code == "NoRegionError":
        return "No AWS region configured. Pass region='us-east-1' (or set AWS_DEFAULT_REGION)."
    if code in ("EndpointConnectionError", "ConnectTimeoutError", "ConnectionClosedError"):
        return (
            f"Couldn't reach AWS ({msg}).\n\nCheck the region name is right (e.g. 'us-east-1') and that "
            "this machine has internet access, then try again."
        )
    if code == "ResourceNotFoundException" and "model" in msg.lower():
        return (
            f"{msg}\n\nThe model isn't available to your account in this region. Turn it on in the Bedrock "
            "console under Model access, or pick another model."
        )
    return f"{code}: {msg}" if code and code not in msg else msg
