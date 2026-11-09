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
from typing import Any, Callable, Dict, Optional, Tuple

from ..aws import Aws, error_code, error_message
from ..spec import KBSpec, Names

#: A callback a step uses to report progress, e.g. ``log("Waiting for index (30s)")``.
Log = Callable[[str], None]

#: Every resource kbforge creates carries this tag, so ``destroy`` can tell ours from yours.
MANAGED_TAG = ("managed-by", "kbforge")


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
        return {MANAGED_TAG[0]: MANAGED_TAG[1], "kbforge-kb": self.spec.name, **self.spec.tags}

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
    """True if a resource's tags show kbforge created it."""
    return tags.get(MANAGED_TAG[0]) == MANAGED_TAG[1]


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
        """Did kbforge create this? True/False from tags, or None if the resource can't be tagged."""
        return None


def explain_error(exc: BaseException) -> str:
    """Turn the most common AWS failures into a message a notebook user can act on."""
    code, msg = error_code(exc), error_message(exc)
    if code in ("AccessDenied", "AccessDeniedException", "UnauthorizedOperation"):
        return (
            f"{msg}\n\nYour notebook's role is missing a permission. Ask your AWS admin to grant the "
            "action named above, or pass role_arn=... to reuse an existing Bedrock role."
        )
    if code == "ServiceQuotaExceededException":
        return f"{msg}\n\nYou've hit an account quota. Delete unused resources or request a quota increase."
    return f"{code}: {msg}" if code and code not in msg else msg
