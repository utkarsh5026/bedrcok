"""OpenSearch Serverless against the in-memory fake: build, re-plan, size changes, teammates, destroy."""

from typing import Any

from kbforge import KnowledgeBase, Plan
from kbforge.steps import Action
from kbforge.steps.opensearch import VECTOR_FIELD

from .fake_aws import ACCOUNT, FakeAws

TEAMMATE = f"arn:aws:iam::{ACCOUNT}:role/TeammateRole"


def actions(plan: Plan) -> dict:
    return {item.step.title: item.inspection.action for item in plan.items}


def make(fake_aws: FakeAws, name: str = "hr-policies", **options: Any) -> KnowledgeBase:
    """An OpenSearch-backed knowledge base wired to the fake."""
    kb = KnowledgeBase(
        name, source="s3://my-company-docs/hr/", vector_store="opensearch_serverless", region="us-east-1", **options
    )
    kb._aws = fake_aws  # type: ignore[assignment]
    return kb


def test_build_then_second_plan_has_nothing_to_do(fake_aws):
    kb = make(fake_aws)
    plan = kb.plan()
    assert plan.ready
    assert set(actions(plan).values()) == {Action.CREATE, Action.RUN}

    assert kb.apply().ok
    aoss = fake_aws.services["opensearchserverless"]
    (collection,) = aoss.collections.values()
    index = aoss.indexes[collection["collectionEndpoint"]]["bedrock-kb-index"]
    assert index["mappings"]["properties"][VECTOR_FIELD]["dimension"] == 1024
    (stored,) = fake_aws.services["bedrock-agent"].kbs.values()
    assert stored["storageConfiguration"]["opensearchServerlessConfiguration"]["collectionArn"] == collection["arn"]

    assert not kb.plan().has_work


def test_changing_the_vector_size_is_a_blocker(fake_aws):
    make(fake_aws).apply()

    plan = make(fake_aws, dimensions=256).plan()
    assert not plan.ready
    assert actions(plan)["Vector index"] == Action.CONFLICT
    assert actions(plan)["Bedrock knowledge base"] == Action.CONFLICT
    assert any("1024-dimension" in b for b in plan.blockers)


def test_a_teammate_is_added_without_removing_whoever_was_there(fake_aws):
    make(fake_aws).apply()
    first = fake_aws.principal_arn

    fake_aws.principal = TEAMMATE
    kb = make(fake_aws)
    assert actions(kb.plan())["Data access policy"] == Action.UPDATE
    assert kb.apply().ok

    (policy,) = fake_aws.services["opensearchserverless"].access_policies.values()
    principals = policy["policy"][0]["Principal"]
    assert first in principals and TEAMMATE in principals
    assert any("AmazonBedrockExecutionRoleForKnowledgeBase_" in p for p in principals)


def test_destroy_removes_everything_kbforge_made(fake_aws):
    kb = make(fake_aws)
    kb.apply()

    assert kb.destroy(confirm=True).ok
    aoss = fake_aws.services["opensearchserverless"]
    assert not aoss.collections and not aoss.security_policies and not aoss.access_policies
    assert not fake_aws.services["iam"].roles
    assert not fake_aws.services["bedrock-agent"].kbs


def test_destroy_keeps_an_untagged_collection_and_its_policies(fake_aws):
    kb = make(fake_aws)
    kb.apply()
    aoss = fake_aws.services["opensearchserverless"]
    for collection in aoss.collections.values():
        collection["tags"] = []

    preview = actions(kb.destroy())
    assert preview["OpenSearch Serverless collection"] == Action.KEEP
    assert preview["Encryption & network policies"] == Action.KEEP
    assert preview["Data access policy"] == Action.KEEP

    assert kb.destroy(confirm=True).ok
    assert aoss.collections and aoss.security_policies and aoss.access_policies


def test_a_collection_belonging_to_another_kb_is_a_blocker(fake_aws):
    make(fake_aws, name="HR_Policies").apply()

    plan = make(fake_aws).plan()
    assert not plan.ready
    assert actions(plan)["OpenSearch Serverless collection"] == Action.CONFLICT
    assert any("HR_Policies" in b for b in plan.blockers)
