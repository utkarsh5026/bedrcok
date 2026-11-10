"""Syncs (Bedrock calls them *ingestion jobs*): how they went, why they failed, and starting or stopping one.

A sync reads the documents in the source, cuts them into chunks, embeds each chunk and writes the
vectors to the store. Bedrock only re-processes files that are new or changed since the last
sync, and removes the vectors of files that were deleted, so later syncs are much quicker than
the first.

Bedrock reports a sync as a status plus a handful of counters, which are hard to read on their
own ("numberOfModifiedDocumentsIndexed: 0"?). This module turns them into objects that explain
themselves, as a card in a notebook or as text in a terminal:

* ``IngestionJob``: one sync. What happened in a sentence, the counts, why it failed if it did,
  and which files Bedrock couldn't index with what to do about each.
* ``IngestionJobs``: the recent syncs of one knowledge base, newest first (``kb.syncs()``).

It also holds the calls that start, follow and stop a sync, so ``kb.sync()``, the ``kbforge``
command and the sync panel all behave the same way. Starting a sync while one is already running
follows the running one instead (Bedrock allows one at a time per data source).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from botocore.exceptions import ClientError

from .aws import Aws, error_code, error_message
from .errors import KBForgeError
from .steps.base import ACCESS_DENIED
from .steps.bedrock import describe_stats, latest_ingestion_job

#: Job statuses that mean the sync hasn't finished yet.
RUNNING = ("STARTING", "IN_PROGRESS", "STOPPING")

#: Job statuses that mean the sync is over, one way or another.
FINISHED = ("COMPLETE", "FAILED", "STOPPED")

#: How each job status is labelled and coloured. A complete sync where some files failed gets
#: its own look (see ``IngestionJob.outcome``).
STATUS_STYLE = {
    "STARTING": ("starting", "c-blue"),
    "IN_PROGRESS": ("syncing", "c-blue"),
    "STOPPING": ("stopping", "c-amber"),
    "COMPLETE": ("complete", "c-green"),
    "FAILED": ("failed", "c-red"),
    "STOPPED": ("stopped", "c-grey"),
}

#: Per-file statuses worth showing (everything else means the file is fine or on its way),
#: with how each is labelled and coloured.
DOCUMENT_STYLE = {
    "FAILED": ("failed", "c-red"),
    "PARTIALLY_INDEXED": ("partly indexed", "c-amber"),
    "METADATA_PARTIALLY_INDEXED": ("metadata partly indexed", "c-amber"),
    "METADATA_UPDATE_FAILED": ("metadata failed", "c-amber"),
    "IGNORED": ("skipped", "c-grey"),
}

#: The description kbforge gives the syncs it starts, so they're recognisable in the console.
JOB_DESCRIPTION = "Started with kbforge"

#: Stop reading the per-file list after this many pages (of up to 1,000 files each).
MAX_DOCUMENT_PAGES = 20


def now() -> datetime:
    """The current time, with a time zone so it can be compared with AWS's timestamps."""
    return datetime.now(timezone.utc)


def _aware(value: Any) -> Optional[datetime]:
    """A timestamp from AWS as a time-zone-aware datetime (None if there isn't one)."""
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def human_duration(seconds: float) -> str:
    """A length of time as people say it: ``45s``, ``3 min 20s``, ``2 h 5 min``."""
    seconds = max(int(round(seconds)), 0)
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {secs}s" if secs and minutes < 10 else f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min" if minutes else f"{hours} h"


def time_ago(when: Optional[datetime], reference: Optional[datetime] = None) -> str:
    """How long ago something happened: ``just now``, ``12 min ago``, ``yesterday``."""
    if when is None:
        return ""
    seconds = ((reference or now()) - when).total_seconds()
    if seconds < 45:
        return "just now"
    if seconds < 3600:
        return f"{max(1, round(seconds / 60))} min ago"
    if seconds < 86400:
        return f"{round(seconds / 3600)} h ago"
    days = round(seconds / 86400)
    return "yesterday" if days == 1 else f"{days} days ago"


def clock(when: Optional[datetime]) -> str:
    """A timestamp in this machine's time zone, e.g. ``2026-11-09 14:02``."""
    return when.astimezone().strftime("%Y-%m-%d %H:%M") if when else ""


def plural(count: int, word: str) -> str:
    """``1 document``, ``3 documents``."""
    return f"{count:,} {word}{'' if count == 1 else 's'}"


@dataclass
class DocumentProblem:
    """A file Bedrock couldn't (fully) index, why, and what to do about it."""

    uri: str
    status: str
    reason: str = ""
    updated_at: Optional[datetime] = None

    @property
    def label(self) -> str:
        """The status in a word or two, e.g. ``failed``."""
        return DOCUMENT_STYLE.get(self.status, (self.status.lower().replace("_", " "), ""))[0]

    @property
    def colour(self) -> str:
        """The badge colour class for the status."""
        return DOCUMENT_STYLE.get(self.status, ("", "c-amber"))[1]

    @property
    def advice(self) -> str:
        """A suggestion for fixing it, from the reason Bedrock gave ("" when there's nothing to add)."""
        return document_advice(self.status, self.reason, self.uri)


def document_advice(status: str, reason: str, uri: str = "") -> str:
    """Turn Bedrock's reason for a file failing into what to do next.

    Bedrock's reasons are written for AWS experts ("ValidationException: ... exceeds ...").
    The common causes are few, so they're matched by keyword; anything else gets no advice
    and the reason is shown on its own.
    """
    text = reason.lower()
    if "metadata" in text or uri.endswith(".metadata.json"):
        return (
            "Its .metadata.json sidecar file couldn't be used. Check it's valid JSON in Bedrock's format "
            '({"metadataAttributes": {...}}) and not too large, then sync again.'
        )
    if any(word in text for word in ("access denied", "accessdenied", "not authorized", "forbidden", "kms", "decrypt")):
        return (
            "The knowledge base's role couldn't read this file. If the bucket uses a KMS key, the role needs "
            "kms:Decrypt on it. Run kb.check_sync() to check the role's permissions."
        )
    if any(word in text for word in ("size", "too large", "exceeds", "larger than")):
        return "Bedrock reads files up to 50 MB. Split the file or shrink its images, then sync again."
    if any(word in text for word in ("unsupported", "not supported", "file type", "format", "extension")):
        return (
            "Bedrock can't read this kind of file. Save it as PDF, Word, Excel, HTML, Markdown, CSV or plain "
            "text, or remove it from the source."
        )
    if any(word in text for word in ("password", "encrypted", "protected")):
        return "The file is password-protected. Upload an unprotected copy and sync again."
    if any(word in text for word in ("throttl", "rate exceeded", "too many requests", "timed out", "timeout")):
        return (
            "Bedrock was too busy to finish it. Sync again later; if it keeps failing, upload the file again "
            "so Bedrock treats it as changed."
        )
    if any(word in text for word in ("empty", "no text", "no content")):
        return (
            "No text could be pulled out of it (a scan or an image, perhaps). A parsing model can read those: "
            'see kbforge.parsing_models() and parsing="nova-lite".'
        )
    if "pars" in text:
        return "The file couldn't be read. Check it opens normally, or try a parsing model for complex PDFs."
    if status == "IGNORED":
        return "Bedrock skipped it, usually because it isn't a type Bedrock reads or it's excluded by a filter."
    return ""


def job_advice(reasons: List[str]) -> str:
    """What to do about a sync that failed as a whole, from Bedrock's failure reasons."""
    text = " ".join(reasons).lower()
    if any(word in text for word in ("access denied", "accessdenied", "not authorized", "assume", "permission")):
        return (
            "The knowledge base's role is missing a permission, or Bedrock can't use the role. "
            "Run kb.check_sync() to see which, and kb.apply() to repair a role kbforge created."
        )
    if "model" in text and any(word in text for word in ("access", "not available", "not found", "invalid")):
        return (
            "The embedding or parsing model isn't available to your account any more. "
            "Check Bedrock → Model access in the AWS console, then sync again."
        )
    if any(word in text for word in ("quota", "limit", "exceeded")):
        return (
            "An AWS account limit was reached. Wait for other syncs to finish, remove unneeded files, "
            "or ask your AWS admin to raise the Bedrock quota."
        )
    if any(word in text for word in ("bucket", "nosuchkey", "no such")):
        return "The source can't be read any more. Run kb.check_sync() to check the bucket and folder still exist."
    if any(word in text for word in ("index", "vector", "collection", "storage", "dimension")):
        return (
            "Bedrock couldn't write to the vector store. Run kb.plan() to check it still exists and matches, "
            "and kb.apply() to repair it."
        )
    return "Run kb.check_sync() to look for the cause, then kb.sync() to try again."


@dataclass
class IngestionJob:
    """One sync of a knowledge base: how it went, its counts, and why it failed if it did.

    Shows as a card in a notebook. ``problems`` lists the files Bedrock couldn't index (with
    advice) when they were looked up, which kbforge does for the latest sync.
    """

    kb_name: str
    region: str
    kb_id: str
    ds_id: str
    job_id: str
    status: str
    started_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    statistics: Dict[str, int] = field(default_factory=dict)
    failure_reasons: List[str] = field(default_factory=list)
    description: str = ""
    #: Where the documents come from, e.g. ``s3://my-bucket/hr/``.
    source: str = ""
    #: True when kbforge followed a sync that was already running instead of starting one.
    followed: bool = False
    #: Files Bedrock couldn't (fully) index right now, or None when they weren't looked up.
    problems: Optional[List[DocumentProblem]] = None
    #: Why ``problems`` is missing or incomplete, e.g. a permission your role lacks.
    problems_note: str = ""
    #: When this snapshot was read from AWS; a running sync's duration counts up to here.
    checked_at: datetime = field(default_factory=now)

    @classmethod
    def from_api(cls, job: Dict[str, Any], kb_name: str, region: str, source: str = "") -> IngestionJob:
        """Wrap a job (or job summary) as returned by Bedrock."""
        return cls(
            kb_name=kb_name,
            region=region,
            kb_id=job.get("knowledgeBaseId", ""),
            ds_id=job.get("dataSourceId", ""),
            job_id=job.get("ingestionJobId", ""),
            status=job.get("status", ""),
            started_at=_aware(job.get("startedAt")),
            updated_at=_aware(job.get("updatedAt")),
            statistics=dict(job.get("statistics") or {}),
            failure_reasons=list(job.get("failureReasons") or []),
            description=job.get("description", ""),
            source=source,
        )

    def _count(self, key: str) -> int:
        return int(self.statistics.get(key) or 0)

    @property
    def scanned(self) -> int:
        """Documents Bedrock looked at (every file in the source, changed or not)."""
        return self._count("numberOfDocumentsScanned")

    @property
    def new(self) -> int:
        """Documents indexed for the first time."""
        return self._count("numberOfNewDocumentsIndexed")

    @property
    def updated(self) -> int:
        """Documents that had changed and were indexed again."""
        return self._count("numberOfModifiedDocumentsIndexed")

    @property
    def deleted(self) -> int:
        """Documents gone from the source, whose vectors were removed."""
        return self._count("numberOfDocumentsDeleted")

    @property
    def failed(self) -> int:
        """Documents Bedrock couldn't index."""
        return self._count("numberOfDocumentsFailed")

    @property
    def skipped(self) -> int:
        """Documents Bedrock passed over (for example a file type it doesn't read)."""
        return self._count("numberOfDocumentsSkipped")

    @property
    def metadata_files(self) -> int:
        """``.metadata.json`` sidecar files read alongside the documents."""
        return self._count("numberOfMetadataDocumentsScanned")

    @property
    def unchanged(self) -> int:
        """Documents that were already indexed and hadn't changed, so needed no work."""
        return max(self.scanned - self.new - self.updated - self.failed - self.skipped, 0)

    @property
    def running(self) -> bool:
        """True while the sync hasn't finished."""
        return self.status in RUNNING

    @property
    def ok(self) -> bool:
        """True for a sync that finished with every document indexed."""
        return self.status == "COMPLETE" and not self.failed

    @property
    def seconds(self) -> float:
        """How long it ran (so far, if it's still running)."""
        if self.started_at is None:
            return 0.0
        end = self.checked_at if self.running else (self.updated_at or self.checked_at)
        return max((end - self.started_at).total_seconds(), 0.0)

    @property
    def outcome(self) -> Tuple[str, str]:
        """The status as a short label and a colour class, e.g. ``("some files failed", "c-amber")``."""
        if self.status == "COMPLETE" and self.failed:
            return ("some files failed", "c-amber")
        return STATUS_STYLE.get(self.status, (self.status.lower().replace("_", " "), "c-grey"))

    @property
    def timing(self) -> str:
        """When it started and how long it took, e.g. ``started 2026-11-09 14:02 (2 h ago) · took 3 min 20s``."""
        if self.started_at is None:
            return ""
        duration = human_duration(self.seconds)
        how_long = f"running for {duration}" if self.running else f"took {duration}"
        return f"started {clock(self.started_at)} ({time_ago(self.started_at, self.checked_at)}) · {how_long}"

    @property
    def stats_text(self) -> str:
        """The counts in one line, e.g. ``97 scanned · 97 indexed``."""
        return describe_stats(self.statistics)

    @property
    def headline(self) -> str:
        """What happened, in one plain sentence."""
        if self.status == "STARTING":
            return "Bedrock is getting ready to read your documents."
        if self.status == "IN_PROGRESS":
            done = f" {plural(self.scanned, 'document')} looked at so far." if self.scanned else ""
            return "Reading, chunking and embedding your documents." + done
        if self.status == "STOPPING":
            return "Stopping. Bedrock finishes the documents it's working on first."
        if self.status == "STOPPED":
            return (
                f"The sync was stopped after looking at {plural(self.scanned, 'document')}. "
                "Run kb.sync() to finish indexing."
            )
        if self.status == "FAILED":
            done = f" after looking at {plural(self.scanned, 'document')}" if self.scanned else ""
            return f"The sync failed{done}, before finishing. The reason is below."
        if self.scanned == 0:
            return "Bedrock found no documents to read. Run kb.check_sync() to look at the source."
        if self.failed:
            return (
                f"{self.failed:,} of {plural(self.scanned, 'document')} couldn't be indexed; the rest are "
                "searchable. The list below says why."
            )
        changes = []
        if self.new:
            changes.append(f"{self.new:,} new")
        if self.updated:
            changes.append(f"{self.updated:,} changed")
        if not changes and not self.deleted:
            return f"Nothing had changed since the last sync: {self._all_documents} {self._are} already up to date."
        sentence = (
            f"Indexed {' and '.join(changes)} {'document' if self.new + self.updated == 1 else 'documents'}"
            if changes
            else "No new or changed documents"
        )
        removed = f"; removed {plural(self.deleted, 'deleted document')}" if self.deleted else ""
        return f"{sentence}{removed}. {self._all_documents[0].upper()}{self._all_documents[1:]} {self._are} searchable."

    @property
    def _all_documents(self) -> str:
        """``the 1 document`` or ``all 97 documents``, for sentences about the whole source."""
        return "the 1 document" if self.scanned == 1 else f"all {plural(self.scanned, 'document')}"

    @property
    def _are(self) -> str:
        return "is" if self.scanned == 1 else "are"

    @property
    def advice(self) -> str:
        """What to do next about a failed or stopped sync ("" when nothing needs doing)."""
        if self.status == "FAILED":
            return job_advice(self.failure_reasons)
        return ""

    def _repr_html_(self) -> str:
        from .ui.html import render_ingestion_job

        return render_ingestion_job(self)

    def __str__(self) -> str:
        from .ui.text import render_ingestion_job

        return render_ingestion_job(self)


@dataclass
class IngestionJobs:
    """The recent syncs of one knowledge base, newest first, as returned by ``kb.syncs()``.

    Shows as a table in a notebook. Index it (``syncs[0]`` is the latest) or iterate over it for
    each ``IngestionJob``; ``kb.sync_job(job_id)`` shows one in detail.
    """

    kb_name: str
    region: str
    kb_id: str
    ds_id: str
    source: str = ""
    items: List[IngestionJob] = field(default_factory=list)
    #: True when there are older syncs than the ones listed.
    more: bool = False

    @property
    def latest(self) -> Optional[IngestionJob]:
        """The most recent sync, or None if it has never synced."""
        return self.items[0] if self.items else None

    @property
    def running(self) -> Optional[IngestionJob]:
        """The sync that's running now, if any."""
        return next((job for job in self.items if job.running), None)

    @property
    def last_success(self) -> Optional[IngestionJob]:
        """The most recent sync that completed."""
        return next((job for job in self.items if job.status == "COMPLETE"), None)

    def __iter__(self) -> Iterator[IngestionJob]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> IngestionJob:
        return self.items[index]

    def _repr_html_(self) -> str:
        from .ui.html import render_ingestion_jobs

        return render_ingestion_jobs(self)

    def __str__(self) -> str:
        from .ui.text import render_ingestion_jobs

        return render_ingestion_jobs(self)


def list_jobs(aws: Aws, kb_name: str, kb_id: str, ds_id: str, source: str = "", limit: int = 10) -> IngestionJobs:
    """The data source's most recent syncs, newest first (read-only)."""
    agent = aws.client("bedrock-agent")
    response = agent.list_ingestion_jobs(
        knowledgeBaseId=kb_id,
        dataSourceId=ds_id,
        sortBy={"attribute": "STARTED_AT", "order": "DESCENDING"},
        maxResults=max(1, min(limit, 1000)),
    )
    found = IngestionJobs(kb_name, aws.region, kb_id, ds_id, source, more=bool(response.get("nextToken")))
    for summary in response.get("ingestionJobSummaries", [])[:limit]:
        job = IngestionJob.from_api(summary, kb_name, aws.region, source)
        if job.status == "FAILED":
            # Summaries leave out why a sync failed, and that's the first thing people want to know.
            try:
                full = agent.get_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=ds_id, ingestionJobId=job.job_id)
                job.failure_reasons = list(full["ingestionJob"].get("failureReasons") or [])
            except ClientError:
                pass
        found.items.append(job)
    return found


def get_job(aws: Aws, kb_name: str, kb_id: str, ds_id: str, job_id: str, source: str = "") -> IngestionJob:
    """One sync in full, including why it failed (read-only)."""
    agent = aws.client("bedrock-agent")
    try:
        job = agent.get_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=ds_id, ingestionJobId=job_id)
    except ClientError as exc:
        if error_code(exc) in ("ResourceNotFoundException", "ValidationException"):
            raise KBForgeError(
                f"'{kb_name}' has no sync with the id {job_id!r}. Run kb.syncs() to see the recent ones and their ids."
            ) from exc
        raise
    return IngestionJob.from_api(job["ingestionJob"], kb_name, aws.region, source)


def running_job(aws: Aws, kb_id: str, ds_id: str) -> Optional[Dict[str, Any]]:
    """The sync that's running now, if any. Only one can run at a time, so it's always the latest."""
    latest = latest_ingestion_job(aws, kb_id, ds_id)
    return latest if latest and latest.get("status") in RUNNING else None


def start_job(aws: Aws, kb_id: str, ds_id: str, description: str = JOB_DESCRIPTION) -> Tuple[Dict[str, Any], bool]:
    """Start a sync, or pick up the one that's already running.

    Returns ``(job, followed)``: ``followed`` is True when a sync was already running, which
    Bedrock reports as a conflict. Following it is what the user wants: their documents are
    being synced either way.
    """
    agent = aws.client("bedrock-agent")
    try:
        job = agent.start_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=ds_id, description=description)
        return job["ingestionJob"], False
    except ClientError as exc:
        code = error_code(exc)
        if code == "ConflictException":
            running = running_job(aws, kb_id, ds_id)
            if running:
                return running, True
        if code == "ServiceQuotaExceededException":
            raise KBForgeError(
                f"{error_message(exc)}\n\nBedrock only runs a few syncs at once in an account. Wait for another "
                "knowledge base's sync to finish, then run kb.sync() again (or ask your AWS admin to raise the "
                "quota)."
            ) from exc
        raise


def follow(
    aws: Aws,
    kb_id: str,
    ds_id: str,
    job_id: str,
    on_update: Callable[[Dict[str, Any]], None],
    *,
    interval: float = 10,
    timeout: float = 6 * 3600,
    sleep: Optional[Callable[[float], Any]] = None,
    keep_going: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Check on a sync every ``interval`` seconds until it finishes, and return its last state.

    ``on_update(job)`` gets every fresh snapshot, so a progress view can show the counts
    climbing. ``keep_going()`` returning False stops watching early (the sync itself carries
    on in AWS), which is how the sync panel stops its background watcher. ``sleep`` defaults to
    ``time.sleep``, looked up on each call so tests can patch it out.
    """
    agent = aws.client("bedrock-agent")
    started = time.time()
    while True:
        job = agent.get_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=ds_id, ingestionJobId=job_id)["ingestionJob"]
        on_update(job)
        if job["status"] in FINISHED or (keep_going is not None and not keep_going()):
            return job
        if time.time() - started > timeout:
            raise KBForgeError(
                f"The sync is still running after {human_duration(timeout)}, so kbforge stopped watching it. It "
                "carries on in AWS: kb.syncs() shows how it's going and kb.stop_sync() stops it."
            )
        (sleep or time.sleep)(interval)


def stop_job(aws: Aws, kb_id: str, ds_id: str, job_id: str) -> Dict[str, Any]:
    """Ask Bedrock to stop a running sync. It finishes the documents in hand, then stops."""
    agent = aws.client("bedrock-agent")
    try:
        return agent.stop_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=ds_id, ingestionJobId=job_id)[
            "ingestionJob"
        ]
    except ClientError as exc:
        if error_code(exc) == "ConflictException":
            raise KBForgeError(
                f"{error_message(exc)}\n\nA sync can only be stopped while it's running. It may be just starting "
                "or already finishing: wait a few seconds and try again, or run kb.syncs() to see where it is."
            ) from exc
        raise


def document_problems(aws: Aws, kb_id: str, ds_id: str) -> Tuple[List[DocumentProblem], str]:
    """The files Bedrock couldn't (fully) index right now, and a note when the list is missing or partial.

    This is the current state of each file, not a record of one particular sync, so kbforge
    only shows it next to the latest sync. Never raises for a missing permission or an old
    boto3: the note explains instead, because the sync itself is fine either way.
    """
    agent = aws.client("bedrock-agent")
    if not hasattr(agent, "list_knowledge_base_documents"):
        return [], (
            "This version of boto3 can't list per-file results. Run %pip install --upgrade boto3 and restart "
            "the kernel to see which files failed and why."
        )
    problems: List[DocumentProblem] = []
    token = ""
    for _ in range(MAX_DOCUMENT_PAGES):
        kwargs: Dict[str, Any] = {"knowledgeBaseId": kb_id, "dataSourceId": ds_id, "maxResults": 1000}
        if token:
            kwargs["nextToken"] = token
        try:
            page = agent.list_knowledge_base_documents(**kwargs)
        except ClientError as exc:
            if error_code(exc) in ACCESS_DENIED:
                return problems, (
                    "Your role can't list the knowledge base's files (bedrock:ListKnowledgeBaseDocuments), so "
                    "kbforge can't say which ones failed. Ask your AWS admin to add that permission."
                )
            return problems, f"Bedrock didn't list this data source's files: {error_message(exc)}"
        for doc in page.get("documentDetails", []):
            if doc.get("status") in DOCUMENT_STYLE:
                identifier = doc.get("identifier", {})
                uri = identifier.get("s3", {}).get("uri") or identifier.get("custom", {}).get("id") or "unknown file"
                problems.append(
                    DocumentProblem(uri, doc["status"], doc.get("statusReason", ""), _aware(doc.get("updatedAt")))
                )
        token = page.get("nextToken", "")
        if not token:
            return problems, ""
    return problems, f"Only the first {MAX_DOCUMENT_PAGES * 1000:,} files were checked."


def add_problems(aws: Aws, job: IngestionJob) -> IngestionJob:
    """Look up the files Bedrock couldn't index and attach them to ``job`` (meant for the latest sync)."""
    job.problems, job.problems_note = document_problems(aws, job.kb_id, job.ds_id)
    return job
