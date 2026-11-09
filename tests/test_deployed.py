"""Listing and reopening deployed knowledge bases, and asking for a written answer."""

import pytest

import kbforge
from kbforge import Chunking, KBForgeError, KBSpec, KnowledgeBase, Parsing, S3Source, WebSource
from kbforge.cli import main
from kbforge.plan import make_plan
from kbforge.steps import Action

from .fake_aws import aws_error


@pytest.fixture
def connect(fake_aws, monkeypatch):
    """Make every new AWS connection (``KnowledgeBase.open``, ``kbforge.knowledge_bases``) use the fake."""
    monkeypatch.setattr("kbforge.knowledge_base.Aws", lambda region=None, session=None: fake_aws)
    monkeypatch.setattr("kbforge.aws.Aws", lambda region=None, session=None: fake_aws)
    return fake_aws


def build(spec: KBSpec, fake_aws) -> KnowledgeBase:
    """Apply a spec against the fake and return its KnowledgeBase."""
    kb = KnowledgeBase.from_spec(spec)
    kb._aws = fake_aws
    result = kb.apply()
    assert getattr(result, "ok", False), result
    return kb


SPECS = [
    KBSpec(name="hr-policies", source=S3Source("s3://my-company-docs/hr/"), region="us-east-1"),
    KBSpec(
        name="Handbook_v2",
        source=S3Source("s3://my-company-docs/"),
        embedding="cohere-multi",
        chunking=Chunking.hierarchical(1200, 200, 40),
        parsing=Parsing.foundation_model("nova-pro", prompt="Transcribe tables as Markdown."),
        description="The staff handbook",
        tags={"team": "people"},
        permissions_boundary="arn:aws:iam::123456789012:policy/Boundary",
        region="us-east-1",
    ),
    KBSpec(
        name="docs-site",
        source=WebSource(["https://docs.example.com"], scope="SUBDOMAINS", max_pages=50, exclude=[".*\\.pdf"]),
        embedding="titan-v2",
        dimensions=512,
        vector_store="opensearch_serverless",
        chunking=Chunking.semantic(400, 1, 90),
        parsing=Parsing.data_automation(),
        region="us-east-1",
    ),
]


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_open_rebuilds_the_same_definition(spec, connect):
    build(spec, connect)

    opened = KnowledgeBase.open(spec.name)

    assert opened.spec == spec
    plan = make_plan(opened.spec, connect)
    assert plan.ready and not plan.has_work
    assert {item.inspection.action for item in plan.items} <= {Action.EXISTS, Action.SKIP, Action.USE}


def test_open_keeps_a_users_own_role(connect):
    role = "arn:aws:iam::123456789012:role/AmazonBedrockExecutionRoleForKnowledgeBase_mine"
    connect.services["iam"].roles["AmazonBedrockExecutionRoleForKnowledgeBase_mine"] = {
        "Arn": role,
        "AssumeRolePolicyDocument": '{"Principal": {"Service": "bedrock.amazonaws.com"}}',
    }
    spec = KBSpec(name="hr-policies", source=S3Source("s3://my-company-docs/hr/"), role_arn=role, region="us-east-1")
    build(spec, connect)

    assert KnowledgeBase.open("hr-policies").spec.role_arn == role


def test_opened_knowledge_base_can_be_used_straight_away(connect, tmp_path):
    build(SPECS[0], connect)
    kb = KnowledgeBase.open("hr-policies")

    assert kb.status().exists
    assert kb.ask("parental leave").results
    saved = KnowledgeBase.from_file(kb.save(tmp_path / "kb.yaml"))
    assert saved.spec == kb.spec


def test_open_suggests_the_closest_name(connect):
    build(SPECS[0], connect)
    with pytest.raises(KBForgeError, match="Did you mean 'hr-policies'"):
        KnowledgeBase.open("hr-policy")


def test_knowledge_bases_lists_only_what_kbforge_built(connect):
    for spec in SPECS[:2]:
        build(spec, connect)
    connect.services["bedrock-agent"].create_knowledge_base(
        name="someone-elses",
        knowledgeBaseConfiguration={},
        storageConfiguration={"type": "S3_VECTORS"},
        roleArn="arn:aws:iam::123456789012:role/x",
        tags={"team": "other"},
    )

    found = kbforge.knowledge_bases()

    assert found.names == ["Handbook_v2", "hr-policies"]
    hr = next(s for s in found if s.name == "hr-policies")
    assert hr.kb_status == "ACTIVE" and hr.source == "s3://my-company-docs/hr/"
    assert hr.last_sync_text.startswith("complete")
    html = found._repr_html_()
    assert "KnowledgeBase.open(&quot;Handbook_v2&quot;)" in html
    assert "hr-policies" in str(found)


def test_knowledge_bases_counts_ones_it_cannot_check(connect, monkeypatch):
    build(SPECS[0], connect)

    def denied(resourceArn):
        raise aws_error("AccessDeniedException", "not allowed")

    monkeypatch.setattr(connect.services["bedrock-agent"], "list_tags_for_resource", denied)
    found = kbforge.knowledge_bases()
    assert len(found) == 0 and found.unreadable == 1
    assert "couldn't be checked" in found._repr_html_()


def test_empty_region_says_how_to_start(connect):
    found = kbforge.knowledge_bases()
    assert len(found) == 0
    assert "kbforge.guide()" in found._repr_html_()


def test_cli_list(connect, capsys):
    build(SPECS[0], connect)
    assert main(["list"]) == 0
    assert "hr-policies" in capsys.readouterr().out


def test_ask_with_answer_cites_its_sources(kb, fake_aws):
    kb.apply()

    answers = kb.ask("How much parental leave?", answer=True)

    assert answers.answer.startswith("You get 26 weeks")
    assert answers.model == "us.amazon.nova-lite-v1:0"
    sent = fake_aws.services["bedrock-agent-runtime"].last_generate
    assert sent["knowledgeBaseConfiguration"]["modelArn"].endswith("inference-profile/us.amazon.nova-lite-v1:0")
    # Two distinct chunks, numbered in the order they're first cited.
    assert [r["location"]["s3Location"]["uri"] for r in answers.results] == [
        "s3://docs/hr/leave.pdf",
        "s3://docs/hr/notes.md",
    ]
    pieces = answers.answer_with_markers()
    assert "".join(text for text, _ in pieces) == answers.answer
    assert [numbers for _, numbers in pieces] == [[1], [1, 2]]

    html = answers._repr_html_()
    assert '<sup class="cite">[2]</sup>' in html and "Sources" in html
    assert "&lt;b&gt;26 weeks&lt;/b&gt;" in html  # chunk text is escaped
    assert "leave.[1]" in str(answers)


def test_naming_a_model_implies_an_answer(kb, fake_aws):
    kb.apply()
    answers = kb.ask("How much parental leave?", model="claude-haiku-4.5")
    assert answers.answer is not None
    assert answers.model == "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def test_unknown_answer_model_gets_a_suggestion(kb):
    with pytest.raises(ValueError, match="Did you mean 'nova-pro'"):
        kb.ask("anything", model="nova-por")


def test_plain_ask_is_unchanged(kb):
    kb.apply()
    answers = kb.ask("How much parental leave?")
    assert answers.answer is None and answers.results[0]["score"] == 0.82


def test_cli_ask_with_answer(kb, fake_aws, tmp_path, monkeypatch, capsys):
    kb.apply()
    config = kb.save(tmp_path / "kb.yaml")
    monkeypatch.setattr("kbforge.knowledge_base.Aws", lambda region=None, session=None: fake_aws)
    assert main(["ask", str(config), "How much leave?", "--answer"]) == 0
    assert "Answered by us.amazon.nova-lite-v1:0" in capsys.readouterr().out
