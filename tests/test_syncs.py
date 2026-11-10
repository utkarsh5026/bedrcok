"""Syncs (ingestion jobs): the pre-sync checks, starting and following a sync, its history and details."""

import json

import pytest

from kbforge import IngestionJob, IngestionJobs, KBForgeError, SyncCheck
from kbforge.checks import FAIL, INFO, OK, WARN
from kbforge.cli import main
from kbforge.ingestion import document_advice, human_duration, job_advice, time_ago
from kbforge.steps import Action

from .fake_aws import ACCOUNT

KMS_KEY = f"arn:aws:kms:us-east-1:{ACCOUNT}:key/1234abcd-5678"


def check(report: SyncCheck, title: str):
    """The check called ``title`` in a sync check (fails the test if it isn't there)."""
    return next(c for c in report.checks if c.title == title)


def agent(fake_aws):
    return fake_aws.services["bedrock-agent"]


def data_source_id(fake_aws) -> str:
    (ds_id,) = agent(fake_aws).data_sources
    return ds_id


def document(fake_aws, uri: str, status: str, reason: str = "") -> dict:
    """A per-file result as ListKnowledgeBaseDocuments returns it."""
    return {
        "dataSourceId": data_source_id(fake_aws),
        "status": status,
        "identifier": {"dataSourceType": "S3", "s3": {"uri": uri}},
        "statusReason": reason,
    }


def kbforge_role(fake_aws) -> str:
    (name,) = fake_aws.services["iam"].roles
    return name


@pytest.fixture
def built(kb):
    """A knowledge base that has been built (and synced once)."""
    kb.apply()
    return kb


def test_sync_check_before_building_says_to_build_first(kb):
    report = kb.check_sync()

    assert not report.ready
    assert check(report, "Knowledge base").status == FAIL
    assert "kb.apply()" in report.blockers[0]


def test_sync_check_on_a_healthy_knowledge_base_passes_every_check(built):
    report = built.check_sync()

    assert report.ready and not report.warnings
    assert {c.status for c in report.checks} == {OK}
    titles = [c.title for c in report.checks]
    for title in ("Sync activity", "Your permissions", "Knowledge base's role", "Source documents"):
        assert title in titles
    assert "read s3://my-company-docs/hr/" in check(report, "Knowledge base's role").detail
    assert report.last is not None and report.running is None


def test_missing_permission_to_start_blocks_the_sync_and_starts_nothing(built, fake_aws):
    fake_aws.services["iam"].caller_denied = {"bedrock:StartIngestionJob"}
    before = len(agent(fake_aws).jobs)

    result = built.sync()

    assert isinstance(result, SyncCheck) and not result.ready
    assert "bedrock:StartIngestionJob" in check(result, "Your permissions").detail
    assert len(agent(fake_aws).jobs) == before


def test_missing_optional_permissions_only_warn(built, fake_aws):
    fake_aws.services["iam"].caller_denied = {"bedrock:StopIngestionJob", "bedrock:ListKnowledgeBaseDocuments"}

    report = built.check_sync()

    permissions = check(report, "Your permissions")
    assert report.ready and permissions.status == WARN
    assert "stop a running sync and see which files failed" in permissions.detail


def test_without_the_policy_simulator_kbforge_says_so_and_compares_its_own_policy(built, fake_aws):
    fake_aws.services["iam"].simulator_denied = True

    report = built.check_sync()

    assert report.ready
    assert check(report, "Your permissions").status == INFO
    role = check(report, "Knowledge base's role")
    assert role.status == OK and "permissions kbforge gave it" in role.detail

    iam = fake_aws.services["iam"]
    iam.policies[kbforge_role(fake_aws)]["kbforge-access"] = json.dumps({"Version": "2012-10-17", "Statement": []})
    drifted = check(built.check_sync(), "Knowledge base's role")
    assert drifted.status == WARN and "kb.apply()" in drifted.detail


def test_a_role_that_cant_read_the_bucket_is_flagged(built, fake_aws):
    iam = fake_aws.services["iam"]
    name = kbforge_role(fake_aws)
    policy = json.loads(iam.policies[name]["kbforge-access"])
    policy["Statement"] = [s for s in policy["Statement"] if s["Sid"] != "ReadSourceDocuments"]
    iam.policies[name]["kbforge-access"] = json.dumps(policy)

    role = check(built.check_sync(), "Knowledge base's role")

    assert role.status == WARN
    assert "isn't allowed to read s3://my-company-docs/hr/" in role.detail
    assert "kb.apply()" in role.detail


def test_a_deleted_role_blocks_the_sync(built, fake_aws):
    iam = fake_aws.services["iam"]
    name = kbforge_role(fake_aws)
    iam.policies[name] = {}
    iam.delete_role(RoleName=name)

    report = built.check_sync()

    assert not report.ready
    assert "no longer exists" in check(report, "Knowledge base's role").detail


def encrypt_bucket(fake_aws, key_id: str) -> None:
    """Make the source bucket encrypt new files with ``key_id`` by default."""
    fake_aws.services["s3"].encryption["my-company-docs"] = {"SSEAlgorithm": "aws:kms", "KMSMasterKeyID": key_id}


def decrypt_statement(fake_aws):
    """The role's KMS statement as kbforge wrote it, or None."""
    policy = json.loads(fake_aws.services["iam"].policies[kbforge_role(fake_aws)]["kbforge-access"])
    return next((s for s in policy["Statement"] if s["Sid"] == "DecryptSourceDocuments"), None)


def role_item(plan):
    return next(item for item in plan.items if item.step.title == "IAM service role")


def test_the_role_kbforge_creates_may_decrypt_a_kms_encrypted_bucket(kb, fake_aws):
    encrypt_bucket(fake_aws, KMS_KEY)

    plan = kb.plan()
    assert '"kms:Decrypt"' in role_item(plan).details[1]
    assert "decrypt them with your KMS key" in role_item(plan).explain
    kb.apply()

    statement = decrypt_statement(fake_aws)
    assert statement["Resource"] == [KMS_KEY]
    assert statement["Condition"] == {"StringEquals": {"kms:ViaService": "s3.us-east-1.amazonaws.com"}}
    assert check(kb.check_sync(), "Source encryption").status == OK


def test_a_key_added_after_building_is_flagged_and_apply_grants_it(built, fake_aws):
    encrypt_bucket(fake_aws, KMS_KEY)

    encryption = check(built.check_sync(), "Source encryption")
    assert encryption.status == WARN and "Run kb.apply() to give it access" in encryption.detail
    assert role_item(built.plan()).inspection.action == Action.UPDATE

    built.apply()
    assert check(built.check_sync(), "Source encryption").status == OK
    assert role_item(built.plan()).inspection.action == Action.EXISTS


@pytest.mark.parametrize(
    ("key_id", "alias", "expected"),
    [
        ("1234abcd-5678", None, KMS_KEY),
        ("alias/hr-docs", "alias/hr-docs", KMS_KEY),
        (f"arn:aws:kms:us-east-1:{ACCOUNT}:alias/hr-docs", "alias/hr-docs", KMS_KEY),
        ("alias/aws/s3", None, None),
    ],
)
def test_every_way_a_bucket_names_its_key(kb, fake_aws, key_id, alias, expected):
    if alias:
        fake_aws.services["kms"].aliases[alias] = {"Arn": KMS_KEY, "KeyManager": "CUSTOMER"}
    encrypt_bucket(fake_aws, key_id)

    kb.apply()

    statement = decrypt_statement(fake_aws)
    assert (statement["Resource"] if statement else [None]) == [expected]


def test_an_alias_that_cant_be_looked_up_is_explained_not_guessed(kb, fake_aws):
    fake_aws.services["kms"].denied = True
    encrypt_bucket(fake_aws, "alias/hr-docs")

    plan = kb.plan()

    assert plan.ready
    assert "kms:DescribeKey" in role_item(plan).inspection.warning
    kb.apply()
    assert decrypt_statement(fake_aws) is None
    assert check(kb.check_sync(), "Source encryption").status == INFO


def test_s3_managed_encryption_needs_no_check(built, fake_aws):
    assert "Source encryption" not in [c.title for c in built.check_sync().checks]
    assert decrypt_statement(fake_aws) is None


def test_sync_reports_counts_and_failed_files_with_advice(built, fake_aws):
    jobs = agent(fake_aws)
    jobs.job_statistics = {
        "numberOfDocumentsScanned": 97,
        "numberOfNewDocumentsIndexed": 2,
        "numberOfModifiedDocumentsIndexed": 1,
        "numberOfDocumentsFailed": 2,
    }
    jobs.documents = [
        document(fake_aws, "s3://my-company-docs/hr/<huge>.pdf", "FAILED", "The file size exceeds the limit"),
        document(fake_aws, "s3://my-company-docs/hr/notes.md", "INDEXED"),
        document(fake_aws, "s3://my-company-docs/hr/scan.bin", "IGNORED"),
    ]

    job = built.sync()

    assert isinstance(job, IngestionJob) and job.status == "COMPLETE" and not job.ok
    assert (job.new, job.updated, job.failed, job.unchanged) == (2, 1, 2, 92)
    assert "2 of 97 documents couldn't be indexed" in job.headline
    assert [p.uri for p in job.problems] == ["s3://my-company-docs/hr/<huge>.pdf", "s3://my-company-docs/hr/scan.bin"]
    assert "50 MB" in job.problems[0].advice
    html = job._repr_html_()
    assert "&lt;huge&gt;.pdf" in html and "<huge>" not in html
    assert "some files failed" in html
    assert "Files that need attention" in str(job)


def test_sync_follows_a_running_sync_instead_of_starting_another(built, fake_aws):
    jobs = agent(fake_aws)
    jobs.job_polls = 3
    running = built.sync(wait=False)
    assert isinstance(running, IngestionJob) and running.running
    before = len(jobs.jobs)

    job = built.sync()

    assert isinstance(job, IngestionJob)
    assert job.followed and job.job_id == running.job_id and job.status == "COMPLETE"
    assert len(jobs.jobs) == before
    assert "already running" in job._repr_html_()


def test_a_failed_sync_comes_back_with_its_reason_and_what_to_do(built, fake_aws):
    jobs = agent(fake_aws)
    jobs.job_status = "FAILED"
    jobs.job_failure_reasons = ["Role is not authorized to perform: s3:ListBucket"]

    job = built.sync(check=False)

    assert job.status == "FAILED"
    assert "kb.check_sync()" in job.advice
    html = job._repr_html_()
    assert "Why it failed" in html and "s3:ListBucket" in html


def test_history_lists_syncs_newest_first_with_failure_reasons(built, fake_aws):
    jobs = agent(fake_aws)
    jobs.job_status, jobs.job_failure_reasons = "FAILED", ["Something broke"]
    built.sync(check=False)
    jobs.job_status, jobs.job_failure_reasons = "COMPLETE", []
    built.sync(check=False)

    history = built.syncs()

    assert isinstance(history, IngestionJobs) and len(history) == 3
    assert [job.status for job in history] == ["COMPLETE", "FAILED", "COMPLETE"]
    assert history[1].failure_reasons == ["Something broke"]
    assert history.last_success is history.latest and history.running is None
    assert "Something broke" in history._repr_html_()
    assert built.syncs(limit=1).more


def test_sync_job_shows_the_latest_in_detail_and_older_ones_without_files(built, fake_aws):
    first = built.syncs()[0].job_id
    agent(fake_aws).job_statistics = {"numberOfDocumentsScanned": 3, "numberOfDocumentsFailed": 1}
    agent(fake_aws).documents = [document(fake_aws, "s3://my-company-docs/hr/a.pdf", "FAILED", "Unsupported format")]
    built.sync(check=False)

    latest = built.sync_job()
    assert latest.problems and latest.problems[0].uri.endswith("a.pdf")

    older = built.sync_job(first)
    assert older.job_id == first and older.problems is None

    with pytest.raises(KBForgeError, match="kb.syncs"):
        built.sync_job("NOSUCHJOB0")


def test_per_file_results_need_a_permission_and_say_so(built, fake_aws):
    agent(fake_aws).job_statistics = {"numberOfDocumentsScanned": 3, "numberOfDocumentsFailed": 1}
    agent(fake_aws).documents_denied = True

    job = built.sync(check=False)

    assert job.problems == [] and "bedrock:ListKnowledgeBaseDocuments" in job.problems_note
    assert "bedrock:ListKnowledgeBaseDocuments" in job._repr_html_()


def test_stop_sync(built, fake_aws):
    with pytest.raises(KBForgeError, match="nothing to stop"):
        built.stop_sync()

    agent(fake_aws).job_polls = 5
    built.sync(wait=False)
    stopped = built.stop_sync()

    assert stopped.status == "STOPPED"
    assert "kb.sync()" in stopped.headline


def test_methods_explain_when_the_knowledge_base_doesnt_exist(kb):
    for method in (kb.syncs, kb.sync_job, kb.stop_sync):
        with pytest.raises(KBForgeError, match="kb.apply"):
            method()


def test_cli_sync_commands(built, fake_aws, tmp_path, monkeypatch, capsys):
    config = built.save(tmp_path / "kb.yaml")
    monkeypatch.setattr("kbforge.knowledge_base.Aws", lambda region=None, session=None: fake_aws)
    monkeypatch.setenv("NO_COLOR", "1")  # CI sets FORCE_COLOR; compare plain text

    assert main(["check-sync", str(config)]) == 0
    assert "Ready to sync" in capsys.readouterr().out
    assert main(["sync", str(config)]) == 0
    assert main(["syncs", str(config)]) == 0
    assert "Recent syncs of 'hr-policies'" in capsys.readouterr().out
    assert main(["syncs", str(config), "latest"]) == 0
    assert "Sync of 'hr-policies': complete" in capsys.readouterr().out

    fake_aws.services["iam"].caller_denied = {"bedrock:StartIngestionJob"}
    assert main(["sync", str(config)]) == 2
    assert main(["check-sync", str(config)]) == 2
    assert main(["stop-sync", str(config)]) == 1  # nothing running: a friendly error


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("The file size exceeds the maximum", "50 MB"),
        ("Unsupported file format", "can't read this kind of file"),
        ("AccessDenied when calling GetObject", "kms:Decrypt"),
        ("Invalid metadata file", ".metadata.json"),
        ("Rate exceeded", "too busy"),
        ("Something nobody has seen before", ""),
    ],
)
def test_document_advice(reason, expected):
    advice = document_advice("FAILED", reason)
    assert expected in advice if expected else advice == ""


def test_job_advice_and_time_wording():
    assert "kb.check_sync()" in job_advice(["User is not authorized to perform: s3:GetObject"])
    assert "quota" in job_advice(["Service quota exceeded"])
    assert human_duration(45) == "45s"
    assert human_duration(200) == "3 min 20s"
    assert human_duration(3900) == "1 h 5 min"
    assert time_ago(None) == ""
