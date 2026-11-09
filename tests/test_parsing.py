"""Parsing: choosing how documents are read, listing the models that can do it, and building with one."""

import json

import pytest

from kbforge import KBSpec, KnowledgeBase, Parsing
from kbforge.checks import FAIL
from kbforge.cli import main
from kbforge.models import resolve_parsing_model
from kbforge.parsers import list_parsing_models
from kbforge.steps import Action


def data_source(fake_aws) -> dict:
    (ds,) = fake_aws.services["bedrock-agent"].data_sources.values()
    return ds


def build(fake_aws, **options) -> KnowledgeBase:
    kb = KnowledgeBase("hr-policies", source="s3://my-company-docs/hr/", region="us-east-1", **options)
    kb._aws = fake_aws
    return kb


def test_default_parsing_sends_no_parsing_configuration():
    spec = KBSpec(name="a", source="s3://b/")

    assert spec.parsing == Parsing.default()
    assert spec.parsing.to_api("us-east-1", "123") is None
    assert "parsing" not in spec.to_dict()


@pytest.mark.parametrize(
    "value, strategy, model",
    [
        (None, "default", None),
        ("data_automation", "data_automation", None),
        ("bda", "data_automation", None),
        ("foundation_model", "foundation_model", "nova-lite"),
        ("claude-sonnet-4", "foundation_model", "claude-sonnet-4"),
        ({"strategy": "foundation_model", "model": "nova-pro"}, "foundation_model", "nova-pro"),
        (Parsing.foundation_model("us.amazon.nova-pro-v1:0"), "foundation_model", "us.amazon.nova-pro-v1:0"),
    ],
)
def test_parsing_can_be_written_many_ways(value, strategy, model):
    parsing = KBSpec(name="a", source="s3://b/", parsing=value).parsing

    assert (parsing.strategy, parsing.model) == (strategy, model)


@pytest.mark.parametrize(
    "value, message",
    [
        ("magic", "Unknown parsing 'magic'"),
        ("amazon.titan-embed-text-v2:0", "Unknown parsing"),
        ({"strategy": "ocr"}, "parsing strategy must be one of"),
        ({"strategy": "default", "model": "nova-lite"}, "model can only be set with the foundation_model"),
        ({"strategy": "foundation_model", "temperature": 0}, "Unknown parsing settings: temperature"),
        ({"strategy": "foundation_model", "prompt": "x" * 10_001}, "Bedrock accepts up to 10,000"),
        ({"strategy": "foundation_model", "prompt": "  "}, "parsing prompt must be some text"),
    ],
)
def test_bad_parsing_is_rejected_before_anything_is_built(value, message):
    with pytest.raises(ValueError, match=message):
        KBSpec(name="a", source="s3://b/", parsing=value)


def test_only_parser_families_are_accepted_as_raw_model_ids():
    with pytest.raises(ValueError, match="only accepts Claude, Amazon Nova and Llama 4"):
        resolve_parsing_model("mistral.pixtral-large-2502-v1:0")


def test_aliases_use_the_profile_for_the_regions_geography():
    model = resolve_parsing_model("nova-lite")

    assert model.invoke_id("us-west-2") == "us.amazon.nova-lite-v1:0"
    assert model.invoke_id("eu-central-1") == "eu.amazon.nova-lite-v1:0"
    assert model.invoke_id("ap-southeast-2") == "apac.amazon.nova-lite-v1:0"
    assert model.arn("eu-west-1", "123") == "arn:aws:bedrock:eu-west-1:123:inference-profile/eu.amazon.nova-lite-v1:0"


def test_raw_ids_keep_the_route_they_name():
    direct = resolve_parsing_model("anthropic.claude-3-haiku-20240307-v1:0")
    profile = resolve_parsing_model("global.anthropic.claude-sonnet-4-20250514-v1:0")

    assert direct.arn("us-east-1", "123") == "arn:aws:bedrock:us-east-1::foundation-model/" + direct.model_id
    assert direct.iam_resources("us-east-1", "123") == [direct.arn("us-east-1", "123")]
    assert profile.invoke_id("eu-west-1") == "global.anthropic.claude-sonnet-4-20250514-v1:0"
    assert profile.label == "Claude Sonnet 4"


def test_foundation_model_parsing_api_block_with_prompt():
    parsing = Parsing.foundation_model("nova-pro", prompt="Transcribe tables as Markdown.")

    assert parsing.to_api("us-east-1", "123") == {
        "parsingStrategy": "BEDROCK_FOUNDATION_MODEL",
        "bedrockFoundationModelConfiguration": {
            "modelArn": "arn:aws:bedrock:us-east-1:123:inference-profile/us.amazon.nova-pro-v1:0",
            "parsingModality": "MULTIMODAL",
            "parsingPrompt": {"parsingPromptText": "Transcribe tables as Markdown."},
        },
    }
    assert "custom instructions" in parsing.describe()


def test_parsing_survives_a_yaml_round_trip(tmp_path):
    kb = KnowledgeBase("a", source="s3://b/", parsing=Parsing.foundation_model("claude-sonnet-4", prompt="Be exact."))
    path = kb.save(tmp_path / "kb.yaml")

    loaded = KnowledgeBase.from_file(path).spec.parsing
    assert loaded == Parsing("foundation_model", "claude-sonnet-4", "Be exact.")


def test_listing_shows_usable_parsers_with_what_to_pass(fake_aws):
    models = list_parsing_models(fake_aws)

    assert models.live and not models.error
    # Known aliases first in kbforge's order; text-only, non-parser and unreachable models are left out.
    assert models.values == ["nova-lite", "nova-pro", "claude-sonnet-4", "anthropic.claude-3-haiku-20240307-v1:0"]
    by_value = {o.value: o for o in models}
    assert by_value["nova-lite"].recommended and by_value["nova-lite"].route == "us. inference profile"
    assert by_value["anthropic.claude-3-haiku-20240307-v1:0"].route == "on-demand"
    # Every listed value is accepted as parsing=.
    for value in models.values:
        KBSpec(name="a", source="s3://b/", parsing=value)


def test_listing_outside_the_profiles_geography_names_the_profile(fake_aws):
    fake_aws.region = "eu-west-1"

    values = list_parsing_models(fake_aws).values
    assert "us.amazon.nova-pro-v1:0" in values and "nova-pro" not in values


def test_listing_falls_back_to_known_parsers_when_not_allowed(fake_aws):
    fake_aws.services["bedrock"].denied = True

    models = list_parsing_models(fake_aws)
    assert not models.live and "bedrock:ListFoundationModels" in models.error
    assert "nova-lite" in models.values


def test_listing_renders_as_card_and_text(fake_aws):
    models = list_parsing_models(fake_aws)

    html = models._repr_html_()
    assert "Pass as parsing=" in html and "nova-lite" in html and "recommended" in html
    assert "strategy: foundation_model" in str(models)


def test_kb_lists_parsers_for_its_own_region(fake_aws):
    assert "nova-lite" in build(fake_aws).parsing_models().values


def test_apply_with_a_parsing_model(fake_aws):
    kb = build(fake_aws, parsing="claude-sonnet-4")
    plan = kb.plan()

    parser_check = next(c for c in plan.checks if c.title.startswith("Parsing model"))
    assert parser_check.status == "ok"
    assert fake_aws.services["bedrock-runtime"].conversed == ["us.anthropic.claude-sonnet-4-20250514-v1:0"]
    assert "Claude Sonnet 4" in plan._repr_html_()  # shown in the data flow

    assert kb.apply().ok
    config = data_source(fake_aws)["vectorIngestionConfiguration"]["parsingConfiguration"]
    assert config["parsingStrategy"] == "BEDROCK_FOUNDATION_MODEL"
    assert config["bedrockFoundationModelConfiguration"]["modelArn"].endswith(
        "inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0"
    )
    assert kb.status().parsing == "us.anthropic.claude-sonnet-4-20250514-v1:0"

    iam = fake_aws.services["iam"]
    (role_name,) = iam.roles
    statements = json.loads(iam.policies[role_name]["kbforge-access"])["Statement"]
    parser = next(s for s in statements if s["Sid"] == "InvokeParsingModel")
    assert parser["Resource"] == [
        "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0",
        "arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-4-20250514-v1:0",
    ]


def test_apply_with_data_automation(fake_aws):
    kb = build(fake_aws, parsing="data_automation")

    assert any(c.title == "Parser · Bedrock Data Automation" for c in kb.plan().checks)
    assert kb.apply().ok
    config = data_source(fake_aws)["vectorIngestionConfiguration"]["parsingConfiguration"]
    assert config["parsingStrategy"] == "BEDROCK_DATA_AUTOMATION"

    iam = fake_aws.services["iam"]
    (role_name,) = iam.roles
    sids = {s["Sid"] for s in json.loads(iam.policies[role_name]["kbforge-access"])["Statement"]}
    assert {"DataAutomationInvoke", "DataAutomationStatus"} <= sids


def test_a_parsing_model_you_cant_call_blocks_the_plan(fake_aws):
    fake_aws.services["bedrock-runtime"].denied_models = ["us.amazon.nova-lite-v1:0"]

    plan = build(fake_aws, parsing="nova-lite").plan()
    check = next(c for c in plan.checks if c.title.startswith("Parsing model"))
    assert check.status == FAIL and "kbforge.parsing_models()" in check.detail
    assert not plan.ready


def test_changing_parsing_after_the_build_warns_and_updates_the_role(fake_aws):
    assert build(fake_aws).apply().ok

    plan = build(fake_aws, parsing="nova-lite").plan()
    items = {item.step.title: item.inspection for item in plan.items}
    assert items["Data source"].action == Action.EXISTS
    assert "parses with the default parser" in items["Data source"].warning
    assert "us.amazon.nova-lite-v1:0" in items["Data source"].warning
    # The role would gain the parser permission, so it's brought up to date.
    assert items["IAM service role"].action == Action.UPDATE


def test_cli_lists_parsers(fake_aws, monkeypatch, capsys):
    monkeypatch.setattr("kbforge.cli.Aws", lambda region=None: fake_aws)

    assert main(["parsers"]) == 0
    out = capsys.readouterr().out
    assert "nova-lite" in out and "us. inference profile" in out
