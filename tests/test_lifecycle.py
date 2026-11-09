"""End-to-end behaviour against the in-memory fake AWS: plan → apply → re-plan → destroy."""

import json

from kbforge import ApplyResult, KnowledgeBase, Plan
from kbforge.checks import FAIL, WARN
from kbforge.steps import Action

from .fake_aws import aws_error


def actions(plan: Plan) -> dict:
    return {item.step.title: item.inspection.action for item in plan.items}


def test_fresh_account_plans_to_create_everything(kb):
    plan = kb.plan()

    assert plan.ready and plan.has_work
    assert set(actions(plan).values()) == {Action.CREATE, Action.RUN}
    assert [c.status for c in plan.checks].count(FAIL) == 0


def test_source_scan_counts_files_and_ignores_metadata_sidecars(kb):
    source = next(c for c in kb.plan().checks if c.title == "Source documents")

    assert source.data["files"] == 3
    assert "metadata sidecars" in source.detail


def test_apply_builds_then_second_plan_has_nothing_to_do(kb, fake_aws):
    result = kb.apply()

    assert isinstance(result, ApplyResult) and result.ok
    assert result.kb_id in fake_aws.services["bedrock-agent"].kbs
    assert result.out["last_job"]["status"] == "COMPLETE"

    again = kb.plan()
    assert not again.has_work
    assert actions(again)["Bedrock knowledge base"] == Action.EXISTS
    assert actions(again)["First sync (ingestion)"] == Action.SKIP


def test_role_policy_is_scoped_to_this_kb(kb, fake_aws):
    kb.apply()
    iam = fake_aws.services["iam"]
    (role_name,) = iam.roles
    policy = json.loads(iam.policies[role_name]["kbforge-access"])
    resources = [r for s in policy["Statement"] for r in s["Resource"]]

    assert "arn:aws:s3:::my-company-docs/*" in resources
    assert any(r.endswith("/index/hr-policies-index") for r in resources)
    assert any("amazon.titan-embed-text-v2:0" in r for r in resources)


def test_existing_kb_with_different_model_blocks_apply(kb, spec, fake_aws):
    kb.apply()
    changed = KnowledgeBase.from_spec(spec.__class__(**{**spec.__dict__, "embedding": "cohere-en"}))
    changed._aws = fake_aws

    plan = changed.plan()
    assert not plan.ready
    assert actions(plan)["Bedrock knowledge base"] == Action.CONFLICT
    assert any("cohere" not in b and "titan" in b for b in plan.blockers)
    assert changed.apply() is not None and not changed.plan().ready


def test_missing_bucket_is_a_blocker(fake_aws):
    kb = KnowledgeBase("docs", source="s3://no-such-bucket/", region="us-east-1")
    kb._aws = fake_aws

    plan = kb.plan()
    assert not plan.ready
    assert "doesn't exist" in plan.blockers[0]
    assert isinstance(kb.apply(), Plan)  # blocked: the plan comes back, nothing is built
    assert not fake_aws.services["s3vectors"].buckets


def test_model_access_problem_is_explained(kb, fake_aws):
    fake_aws.services["bedrock-runtime"].denied = True
    check = next(c for c in kb.plan().checks if c.title.startswith("Embedding model"))

    assert check.status == FAIL
    assert "Model access" in check.detail


def test_unsupported_files_warn_but_do_not_block(kb, fake_aws):
    fake_aws.services["s3"].buckets["my-company-docs"]["hr/diagram.png"] = 5_000
    plan = kb.plan()
    source = next(c for c in plan.checks if c.title == "Source documents")

    assert source.status == WARN and plan.ready


def test_failure_midway_reports_step_and_rerun_resumes(kb, fake_aws):
    agent = fake_aws.services["bedrock-agent"]
    agent.fail_create_kb = aws_error("AccessDeniedException", "not allowed: bedrock:CreateKnowledgeBase")

    failed = kb.apply()
    assert isinstance(failed, ApplyResult) and not failed.ok
    assert failed.failed_step == "Bedrock knowledge base"
    assert "missing a permission" in failed.error
    assert fake_aws.services["s3vectors"].indexes  # earlier steps did finish

    agent.fail_create_kb = None
    plan = kb.plan()
    assert actions(plan)["Vector index"] == Action.EXISTS
    assert kb.apply().ok


def test_destroy_previews_then_deletes(kb, fake_aws):
    kb.apply()

    preview = kb.destroy()
    assert isinstance(preview, Plan) and preview.mode == "destroy"
    assert fake_aws.services["bedrock-agent"].kbs  # nothing deleted yet
    assert preview.items[0].step.title == "Data source"  # reverse order

    done = kb.destroy(confirm=True)
    assert isinstance(done, ApplyResult) and done.ok
    assert not fake_aws.services["bedrock-agent"].kbs
    assert not fake_aws.services["s3vectors"].buckets
    assert not fake_aws.services["iam"].roles
    assert set(actions(kb.plan()).values()) == {Action.CREATE, Action.RUN}


def test_destroy_keeps_a_role_the_user_supplied(fake_aws):
    role = "arn:aws:iam::123456789012:role/TeamBedrockRole"
    fake_aws.services["iam"].roles["TeamBedrockRole"] = {
        "Arn": role,
        "AssumeRolePolicyDocument": '{"Principal": {"Service": "bedrock.amazonaws.com"}}',
    }
    kb = KnowledgeBase("hr", source="s3://my-company-docs/hr/", role_arn=role, region="us-east-1")
    kb._aws = fake_aws
    assert kb.apply().ok

    preview = kb.destroy()
    assert actions(preview)["IAM service role"] == Action.KEEP
    kb.destroy(confirm=True)
    assert "TeamBedrockRole" in fake_aws.services["iam"].roles


def test_destroy_leaves_untagged_kb_alone_unless_forced(kb, fake_aws):
    kb.apply()
    for item in fake_aws.services["bedrock-agent"].kbs.values():
        item["tags"] = {}

    assert actions(kb.destroy())["Bedrock knowledge base"] == Action.KEEP
    assert actions(kb.destroy(force=True))["Bedrock knowledge base"] == Action.DELETE


def test_ask_and_status_after_build(kb):
    kb.apply()

    answers = kb.ask("How long is parental leave?")
    assert answers.texts == ["Employees get <b>26 weeks</b> of parental leave."]

    status = kb.status()
    assert status.exists and status.kb_status == "ACTIVE"
    assert dict(status.rows())["Last sync"].startswith("complete")


def test_sync_runs_a_new_ingestion_job(kb, fake_aws):
    kb.apply()
    before = len(fake_aws.services["bedrock-agent"].jobs)

    kb.sync()
    assert len(fake_aws.services["bedrock-agent"].jobs) == before + 1


def test_role_policy_drift_is_detected_and_fixed(kb, fake_aws):
    kb.apply()
    iam = fake_aws.services["iam"]
    (role_name,) = iam.roles
    iam.policies[role_name]["kbforge-access"] = json.dumps({"Version": "2012-10-17", "Statement": []})

    assert actions(kb.plan())["IAM service role"] == Action.UPDATE
    kb.apply()
    assert actions(kb.plan())["IAM service role"] == Action.EXISTS
