"""``kb.check_sync()``: will a sync work? Answered before anything is started.

A sync can run for an hour and then fail for a reason that was knowable up front: your role
isn't allowed to start one, the role Bedrock uses can't read the bucket or its KMS key, model
access was switched off, the folder is empty. ``check_sync`` looks at all of that in a few
seconds without changing anything, and ``kb.sync()`` runs it first so a sync that's bound to
fail is never started.

What it checks, in the order shown:

* your AWS credentials;
* the knowledge base and its data source exist and are ready;
* whether a sync is already running, and how the last one went;
* your own permissions to start, follow and stop syncs and to see per-file results;
* the knowledge base's role (the identity Bedrock uses during a sync): it exists, trusts
  Bedrock, and may read your documents, call the models and write vectors;
* the KMS key that encrypts the source bucket, if it has one;
* the embedding model (and parsing model, if any) respond;
* the source still holds files Bedrock can read.

Everything is checked against what's deployed, not just what the ``KnowledgeBase`` object
says, because that's what the sync will use. Permissions that can't be tried without side
effects are put to IAM's policy simulator (see ``permissions.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from botocore.exceptions import ClientError

from .aws import Aws, error_code
from .checks import (
    FAIL,
    INFO,
    OK,
    WARN,
    Check,
    check_identity,
    check_model,
    check_parser,
    check_s3_source,
    check_web_source,
)
from .deployed import find_data_source, parser_label, spec_from_aws
from .errors import KBForgeError
from .ingestion import IngestionJob, list_jobs, time_ago
from .permissions import ALLOWED, DENIED, UNKNOWN, simulate
from .spec import KBSpec, S3Source
from .steps.base import ACCESS_DENIED, Ctx
from .steps.bedrock import KnowledgeBaseStep
from .steps.iam import RoleStep

#: What your notebook role needs for each part of syncing: (IAM action, what it lets you do, essential?).
#: Without an essential one a sync can't be started or followed; the rest only switch features off.
CALLER_ACTIONS = [
    ("bedrock:StartIngestionJob", "start a sync", True),
    ("bedrock:GetIngestionJob", "follow a sync's progress", True),
    ("bedrock:ListIngestionJobs", "list syncs", False),
    ("bedrock:StopIngestionJob", "stop a running sync", False),
    ("bedrock:ListKnowledgeBaseDocuments", "see which files failed and why", False),
]


def and_join(items: List[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


@dataclass
class SyncCheck:
    """Whether a knowledge base is ready to sync, from read-only checks. Returned by ``kb.check_sync()``.

    Shows as a card in a notebook. ``ready`` is False when something would make the sync fail;
    ``blockers`` says what. ``kb.sync()`` returns this instead of starting a sync in that case.
    """

    kb_name: str
    region: str
    source: str
    checks: List[Check] = field(default_factory=list)
    account: str = ""
    kb_id: str = ""
    ds_id: str = ""
    #: The sync that's running right now, if any (``kb.sync()`` follows it rather than starting another).
    running: Optional[IngestionJob] = None
    #: The most recent sync, running or not.
    last: Optional[IngestionJob] = None

    @property
    def blockers(self) -> List[str]:
        """Everything that would make a sync fail. Empty means it's fine to sync."""
        return [f"{c.title}: {c.detail}" for c in self.checks if c.status == FAIL]

    @property
    def warnings(self) -> List[str]:
        """Things worth reading that don't stop a sync (some files may fail, say)."""
        return [f"{c.title}: {c.detail}" for c in self.checks if c.status == WARN]

    @property
    def ready(self) -> bool:
        """True when nothing found would make the sync fail."""
        return not self.blockers

    @property
    def summary(self) -> str:
        """The outcome in a few words, e.g. ``9 passed · 1 warning``."""
        counts = {status: sum(c.status == status for c in self.checks) for status in (OK, INFO, WARN, FAIL)}
        parts = [f"{counts[OK] + counts[INFO]} passed"]
        if counts[WARN]:
            parts.append(f"{counts[WARN]} warning{'s' if counts[WARN] > 1 else ''}")
        if counts[FAIL]:
            parts.append(f"{counts[FAIL]} problem{'s' if counts[FAIL] > 1 else ''}")
        return " · ".join(parts)

    def _repr_html_(self) -> str:
        from .ui.html import render_sync_check

        return render_sync_check(self)

    def __str__(self) -> str:
        from .ui.text import render_sync_check

        return render_sync_check(self)


def check_sync(spec: KBSpec, aws: Aws) -> SyncCheck:
    """Run every pre-sync check for the knowledge base ``spec`` describes. Never changes anything."""
    report = SyncCheck(spec.name, aws.region, spec.source.describe())
    identity = check_identity(aws)
    report.checks.append(identity)
    if identity.status == FAIL:
        return report
    report.account = aws.account_id

    kb = KnowledgeBaseStep().find(Ctx(spec, aws))
    report.checks.append(_kb_check(spec, aws, kb))
    if kb is None:
        return report
    report.kb_id = kb["knowledgeBaseId"]
    ds = find_data_source(aws, report.kb_id, spec.names.data_source)
    report.checks.append(_data_source_check(spec, ds))
    if ds is None:
        return report
    report.ds_id = ds["dataSourceId"]

    live = _deployed_spec(spec, aws, kb, ds)
    report.source = live.source.describe()
    report.checks.append(_activity_check(aws, report))
    report.checks.append(_caller_check(aws, kb))
    report.checks.append(_role_check(live, aws, kb, ds))
    if isinstance(live.source, S3Source):
        encryption = _encryption_check(live.source, aws, kb)
        if encryption:
            report.checks.append(encryption)
    report.checks.append(check_model(live, aws))
    parser = check_parser(live, aws)
    if parser:
        report.checks.append(parser)
    if isinstance(live.source, S3Source):
        report.checks += check_s3_source(live.source, aws)
    else:
        report.checks += check_web_source(live.source)
    return report


def _deployed_spec(spec: KBSpec, aws: Aws, kb: Dict[str, Any], ds: Dict[str, Any]) -> KBSpec:
    """The knowledge base as deployed (source, models, parsing), which is what a sync uses.

    Usually identical to ``spec``. Falls back to ``spec`` if the deployed settings are ones
    kbforge can't describe (a model it doesn't know, say).
    """
    try:
        return spec_from_aws(aws, spec.name, kb, ds, spec.tags)
    except (KBForgeError, ValueError, KeyError, ClientError):
        return spec


def _kb_check(spec: KBSpec, aws: Aws, kb: Optional[Dict[str, Any]]) -> Check:
    """The knowledge base exists and is ACTIVE (Bedrock won't sync one that isn't)."""
    title = "Knowledge base"
    if kb is None:
        return Check(
            FAIL,
            title,
            f"'{spec.name}' doesn't exist in {aws.region} yet, so there's nothing to sync. "
            "Run kb.plan() and kb.apply() to build it.",
        )
    status = kb.get("status", "")
    if status == "ACTIVE":
        return Check(OK, title, f"{kb['knowledgeBaseId']} is active.")
    if status in ("CREATING", "UPDATING"):
        return Check(FAIL, title, f"It's still {status.lower()}. Wait a minute, then try again.")
    reasons = "; ".join(kb.get("failureReasons", []))
    if status == "FAILED":
        return Check(
            FAIL, title, f"It's in a failed state{': ' + reasons if reasons else ''}. Run kb.plan() to see what to fix."
        )
    return Check(FAIL, title, f"It's {status.lower().replace('_', ' ')}, so it can't sync.")


def _data_source_check(spec: KBSpec, ds: Optional[Dict[str, Any]]) -> Check:
    """The data source exists and is AVAILABLE."""
    title = "Data source"
    if ds is None:
        return Check(
            FAIL,
            title,
            f"The knowledge base has no data source called {spec.names.data_source}, so it doesn't know where "
            "your documents are. Run kb.apply() to add it.",
        )
    status = ds.get("status", "AVAILABLE")
    if status != "AVAILABLE":
        return Check(FAIL, title, f"{ds['dataSourceId']} is {status.lower().replace('_', ' ')}, so it can't sync.")
    return Check(OK, title, f"{ds['dataSourceId']} is available · {parser_label(ds)}.")


def _activity_check(aws: Aws, report: SyncCheck) -> Check:
    """Is a sync already running, and how did the last one go?"""
    title = "Sync activity"
    try:
        jobs = list_jobs(aws, report.kb_name, report.kb_id, report.ds_id, report.source, limit=1)
    except ClientError as exc:
        if error_code(exc) in ACCESS_DENIED:
            return Check(
                WARN,
                title,
                "Your role can't list syncs (bedrock:ListIngestionJobs), so kbforge can't tell whether one is "
                "already running.",
            )
        raise
    latest = report.last = jobs.latest
    if latest is None:
        return Check(OK, title, "Never synced. The first sync reads every document, so it takes the longest.")
    when = time_ago(latest.started_at)
    if latest.running:
        report.running = latest
        return Check(
            WARN,
            title,
            f"A sync started {when} is still running ({latest.stats_text}). kb.sync() follows it instead of "
            "starting another; kb.stop_sync() stops it.",
        )
    if latest.status == "FAILED":
        reason = latest.failure_reasons[0] if latest.failure_reasons else "Bedrock didn't say why"
        return Check(WARN, title, f"No sync running, but the last one failed {when}: {reason}")
    return Check(OK, title, f"No sync running. The last one, {when}: {latest.outcome[0]} · {latest.stats_text}.")


def _caller_check(aws: Aws, kb: Dict[str, Any]) -> Check:
    """May you start, follow and stop syncs, and see per-file results? Asked of IAM's policy simulator."""
    title = "Your permissions"
    principal = aws.principal_arn
    who = principal.rsplit("/", 1)[-1] or principal
    arn = kb["knowledgeBaseArn"]
    answers = simulate(aws, principal, [(action, arn) for action, _, _ in CALLER_ACTIONS])
    verdict = {action: answers[(action, arn)] for action, _, _ in CALLER_ACTIONS}
    if all(v == UNKNOWN for v in verdict.values()):
        return Check(
            INFO,
            title,
            "kbforge couldn't check them in advance (that needs IAM's policy simulator, "
            "iam:SimulatePrincipalPolicy). If AWS refuses the sync, the message names the permission to add.",
        )
    denied = [(action, why, essential) for action, why, essential in CALLER_ACTIONS if verdict[action] == DENIED]
    missing = [(action, why) for action, why, essential in denied if essential]
    optional = [(action, why) for action, why, essential in denied if not essential]
    if missing:
        return Check(
            FAIL,
            title,
            f"{who} isn't allowed to {and_join([why for _, why in missing])} "
            f"({', '.join(action for action, _ in missing)}), so the sync would be refused. Ask your AWS admin to "
            "add it to your role; kbforge's reference guide lists every permission it uses.",
        )
    if optional:
        return Check(
            WARN,
            title,
            f"You can start and follow syncs, but {who} isn't allowed to {and_join([why for _, why in optional])} "
            f"({', '.join(action for action, _ in optional)}). Ask your AWS admin to add "
            f"{'it' if len(optional) == 1 else 'them'} if you need {'it' if len(optional) == 1 else 'them'}.",
        )
    unsure = [action for action, v in verdict.items() if v == UNKNOWN]
    note = f" Couldn't confirm {', '.join(unsure)}." if unsure else ""
    return Check(
        OK,
        title,
        f"{who} may start, follow and stop syncs and see per-file results (checked with IAM's policy simulator).{note}",
    )


def _role_needs(live: KBSpec, kb: Dict[str, Any], ds: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """What the knowledge base's role must be allowed during a sync: (action, resource, what it's for)."""
    needs: List[Tuple[str, str, str]] = []
    if isinstance(live.source, S3Source):
        bucket = f"arn:aws:s3:::{live.source.bucket}"
        prefix = live.source.prefix
        folder = prefix if not prefix or prefix.endswith("/") else prefix + "/"
        what = f"read {live.source.uri}"
        needs += [("s3:ListBucket", bucket, what), ("s3:GetObject", f"{bucket}/{folder}example.pdf", what)]
    vector = kb.get("knowledgeBaseConfiguration", {}).get("vectorKnowledgeBaseConfiguration", {})
    if vector.get("embeddingModelArn"):
        needs.append(("bedrock:InvokeModel", vector["embeddingModelArn"], "call the embedding model"))
    parsing = ds.get("vectorIngestionConfiguration", {}).get("parsingConfiguration") or {}
    parser_arn = parsing.get("bedrockFoundationModelConfiguration", {}).get("modelArn")
    if parser_arn:
        needs.append(("bedrock:InvokeModel", parser_arn, "call the parsing model"))
    storage = kb.get("storageConfiguration", {})
    index_arn = storage.get("s3VectorsConfiguration", {}).get("indexArn")
    collection_arn = storage.get("opensearchServerlessConfiguration", {}).get("collectionArn")
    if index_arn:
        needs.append(("s3vectors:PutVectors", index_arn, "write vectors to the index"))
    elif collection_arn:
        needs.append(("aoss:APIAccessAll", collection_arn, "write vectors to the collection"))
    return needs


def _role_check(live: KBSpec, aws: Aws, kb: Dict[str, Any], ds: Dict[str, Any]) -> Check:
    """The role Bedrock uses during a sync exists, trusts Bedrock and has the permissions a sync needs."""
    title = "Knowledge base's role"
    name = kb.get("roleArn", "").rsplit("/", 1)[-1]
    ours = name == live.names.role
    try:
        role = aws.client("iam").get_role(RoleName=name)["Role"]
    except ClientError as exc:
        if error_code(exc) == "NoSuchEntity":
            fix = "Run kb.apply() to create it again." if ours else "Recreate it, or ask your AWS admin."
            return Check(
                FAIL,
                title,
                f"The role Bedrock uses to read your documents ({name}) no longer exists, so every sync would fail. "
                + fix,
            )
        if error_code(exc) in ACCESS_DENIED:
            return Check(INFO, title, f"Couldn't look at {name} (your role lacks iam:GetRole); assuming it's fine.")
        raise
    if "bedrock.amazonaws.com" not in str(role.get("AssumeRolePolicyDocument", "")):
        return Check(
            FAIL,
            title,
            f"{name}'s trust policy doesn't let Bedrock use it, so every sync would fail. Ask your AWS admin to "
            "allow bedrock.amazonaws.com to assume it.",
        )
    needs = _role_needs(live, kb, ds)
    answers = simulate(aws, role["Arn"], [(a, r) for a, r, _ in needs], {"aws:ResourceAccount": aws.account_id})
    denied = list(dict.fromkeys(what for a, r, what in needs if answers[(a, r)] == DENIED))
    if denied:
        fix = (
            "Run kb.apply() to put kbforge's permissions back."
            if ours
            else 'Give it the permissions kb.plan() lists under "IAM service role".'
        )
        return Check(
            WARN,
            title,
            f"{name} isn't allowed to {and_join(denied)}, so the sync would fail unless a resource policy (a bucket "
            f"policy, say) allows it. {fix}",
        )
    everything = and_join(list(dict.fromkeys(what for _, _, what in needs)))
    unsure = list(dict.fromkeys(what for a, r, what in needs if answers[(a, r)] == UNKNOWN))
    if not unsure:
        return Check(OK, title, f"{name} trusts Bedrock and may {everything}.")
    if any(answers[(a, r)] == ALLOWED for a, r, _ in needs):
        return Check(OK, title, f"{name} trusts Bedrock. Couldn't confirm it may {and_join(unsure)}.")
    if ours:
        return _compare_with_kbforge_policy(live, aws, kb, role, title, name)
    return Check(
        INFO,
        title,
        f"{name} trusts Bedrock. kbforge couldn't check its permissions in advance (that needs "
        'iam:SimulatePrincipalPolicy); it must allow what kb.plan() lists under "IAM service role".',
    )


def _compare_with_kbforge_policy(
    live: KBSpec, aws: Aws, kb: Dict[str, Any], role: Dict[str, Any], title: str, name: str
) -> Check:
    """Without the simulator, check a role kbforge made still has exactly the policy kbforge gave it."""
    ctx = Ctx(live, aws)
    storage = kb.get("storageConfiguration", {})
    ctx.out["role_arn"] = role["Arn"]
    ctx.out["index_arn"] = storage.get("s3VectorsConfiguration", {}).get("indexArn")
    ctx.out["collection_arn"] = storage.get("opensearchServerlessConfiguration", {}).get("collectionArn")
    step = RoleStep()
    try:
        current = step.current_policy(ctx)
    except ClientError as exc:
        if error_code(exc) in ACCESS_DENIED:
            return Check(INFO, title, f"{name} trusts Bedrock. Couldn't read its permissions to check them.")
        raise
    if current == step.permissions_policy(ctx):
        return Check(OK, title, f"{name} trusts Bedrock and has the permissions kbforge gave it.")
    return Check(
        WARN,
        title,
        f"{name}'s permissions differ from what this knowledge base needs, so the sync may fail. "
        "Run kb.apply() to bring them up to date.",
    )


def _encryption_check(source: S3Source, aws: Aws, kb: Dict[str, Any]) -> Optional[Check]:
    """If the bucket encrypts files with your own KMS key, the knowledge base's role must be allowed to use it.

    This is the classic "every file failed" sync: the role can read the bucket but not decrypt
    what's in it. Silent when the bucket uses S3's own encryption, or its settings can't be read.
    """
    try:
        config = aws.client("s3").get_bucket_encryption(Bucket=source.bucket)["ServerSideEncryptionConfiguration"]
    except ClientError:
        return None  # no default encryption configured, or not allowed to look
    key = ""
    for rule in config.get("Rules", []):
        default = rule.get("ApplyServerSideEncryptionByDefault", {})
        if default.get("SSEAlgorithm", "").startswith("aws:kms"):
            key = default.get("KMSMasterKeyID", "")
            break
    if not key or key.endswith("alias/aws/s3"):
        return None  # S3's own key works for any role that may read the bucket
    title = "Source encryption"
    role_arn = kb.get("roleArn", "")
    name = role_arn.rsplit("/", 1)[-1]
    key_arn = key if key.startswith("arn:") else f"arn:aws:kms:{aws.region}:{aws.account_id}:key/{key}"
    short = key_arn.rsplit("/", 1)[-1]
    verdict = UNKNOWN
    if ":alias/" not in key_arn and not key.startswith("alias/"):
        context = {"kms:ViaService": f"s3.{aws.region}.amazonaws.com"}
        verdict = simulate(aws, role_arn, [("kms:Decrypt", key_arn)], context)[("kms:Decrypt", key_arn)]
    if verdict == ALLOWED:
        return Check(OK, title, f"Files are encrypted with KMS key {short}, and the knowledge base's role may use it.")
    if verdict == DENIED:
        return Check(
            WARN,
            title,
            f"Files are encrypted with KMS key {short}, and {name} isn't allowed kms:Decrypt on it, so every file "
            "would fail unless the key's own policy allows the role. Add kms:Decrypt on this key to the role, or "
            "ask the key's owner to allow it.",
        )
    return Check(
        INFO,
        title,
        f"Files are encrypted with KMS key {key}. The knowledge base's role ({name}) needs kms:Decrypt on it, or "
        "every file fails to sync; kbforge couldn't check this in advance.",
    )
