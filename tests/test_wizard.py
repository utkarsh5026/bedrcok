"""The wizard form: live validation, the describe → review → build flow, and escaping."""

import importlib

import pytest

pytest.importorskip("ipywidgets")

import kbforge.knowledge_base  # noqa: E402
from kbforge.ui import wizard_style  # noqa: E402
from kbforge.wizard import Wizard  # noqa: E402

#: The wizard's module. ``kbforge.wizard`` is the function that opens the form, so get the module itself.
wizard_module = importlib.import_module("kbforge.wizard")


@pytest.fixture
def wiz(fake_aws, monkeypatch):
    """A wizard whose knowledge bases talk to the in-memory fake instead of AWS."""
    monkeypatch.setattr(kbforge.knowledge_base, "Aws", lambda region=None, session=None: fake_aws)
    return Wizard(region="us-east-1")


def fill(wiz, name="hr-policies", uri="s3://my-company-docs/hr/"):
    wiz.name.value = name
    wiz.s3_uri.value = uri


def test_empty_form_asks_for_a_name_and_cannot_be_reviewed(wiz):
    assert wiz.preview_btn.disabled
    assert "give your knowledge base a name" in wiz.status.value
    assert "not set yet" in wiz.preview.value


def test_bad_input_is_flagged_next_to_the_field(wiz):
    fill(wiz, name="bad--name", uri="my-bucket/hr")

    assert "bad" in wiz.name._dom_classes and "fhint bad" in wiz.name_hint.value
    assert "start with <code>s3://</code>" in wiz.source_hint.value
    assert wiz.preview_btn.disabled


def test_valid_form_shows_live_preview_and_aws_names(wiz):
    fill(wiz)

    assert not wiz.preview_btn.disabled
    assert "kbforge-hr-policies" in wiz.preview.value  # vector bucket name
    wiz.vector_store.value = "opensearch_serverless"
    assert "OpenSearch collection" in wiz.preview.value
    assert wiz._options() == {"vector_store": "opensearch_serverless", "region": "us-east-1"}


def test_option_cards_highlight_the_selected_choice(wiz):
    wiz.chunking.value = "semantic"

    selected = [box for box in wiz.chunking.children if "on" in box._dom_classes]
    assert len(selected) == 1 and "Semantic" in selected[0].children[0].value


def test_only_configurable_models_offer_vector_sizes(wiz):
    assert wiz.dimensions is not None
    wiz.dimensions.value = 256
    assert wiz._options()["dimensions"] == 256

    wiz.embedding.value = "cohere-en"
    assert wiz.dimensions is None and wiz.dims_box.children == ()
    assert "dimensions" not in wiz._options()


def test_review_build_and_ask(wiz):
    fill(wiz)
    wiz.preview_btn.click()

    assert wiz.stage == 2 and "Ready." in wiz.view.value
    assert not wiz.build_btn.disabled
    assert wiz.body.layout.display == "none" and "hr-policies" in wiz.recap_html.value

    wiz.build_btn.click()
    assert wiz.stage == 4 and "ready" in wiz.result.value
    assert wiz.try_box.layout.display is None

    wiz.question.value = "leave?"
    wiz.ask_btn.click()
    assert "&lt;b&gt;26 weeks&lt;/b&gt;" in wiz.answers.value


def test_editing_after_review_requires_a_new_plan(wiz):
    fill(wiz)
    wiz.preview_btn.click()
    wiz.edit_btn.click()
    wiz.chunking.value = "none"

    assert wiz.plan is None and wiz.build_btn.disabled and wiz.stage == 1
    assert "stale" in wiz.view._dom_classes
    assert "review the plan again" in wiz.status.value


def test_show_as_code_matches_the_form(wiz):
    fill(wiz)
    wiz.embedding.value = "cohere-multi"
    wiz.code_btn.click()

    assert "KnowledgeBase(" in wiz.code.value and "embedding=&#x27;cohere-multi&#x27;" in wiz.code.value
    assert "kb.json" in wiz.code.value and "&quot;embedding&quot;: &quot;cohere-multi&quot;" in wiz.code.value
    wiz.code_btn.click()
    assert wiz.code.value == ""


def test_user_text_is_escaped_in_the_preview(wiz):
    wiz.s3_uri.value = "s3://<script>x</script>/"

    assert "<script>" not in wiz.preview.value + wiz.source_hint.value
    assert "&lt;script&gt;" in wiz.preview.value


def test_stepper_marks_finished_stages():
    html = wizard_style.stepper(4)
    assert html.count('class="stp done"') == 3


def test_parsing_model_choice_appears_only_for_foundation_model(wiz):
    fill(wiz)
    assert wiz.parser_box.layout.display == "none" and "parsing" not in wiz._options()

    wiz.parsing.value = "foundation_model"
    assert wiz.parser_box.layout.display is None
    assert wiz._options()["parsing"] == "nova-lite"
    assert "Amazon Nova Lite" in wiz.preview.value

    wiz.parser.value = "claude-sonnet-4"
    wiz.parser_prompt.value = "Transcribe tables as Markdown."
    assert wiz._options()["parsing"] == {
        "strategy": "foundation_model",
        "model": "claude-sonnet-4",
        "prompt": "Transcribe tables as Markdown.",
    }

    wiz.parsing.value = "data_automation"
    assert wiz._options()["parsing"] == "data_automation"


def test_wizard_lists_the_parsers_the_account_offers(wiz, fake_aws, monkeypatch):
    monkeypatch.setattr(wizard_module, "Aws", lambda region=None, session=None: fake_aws)
    fill(wiz)
    wiz.parsing.value = "foundation_model"
    wiz.parser.value = "nova-pro"

    wiz.parser_list_btn.click()
    titles = [box.children[0].value for box in wiz.parser.children]
    assert any("Claude 3 Haiku" in t for t in titles)
    assert wiz.parser.value == "nova-pro"  # the pick survives the refresh
    assert "your account can use" in wiz.parser_hint.value


def test_wizard_parser_listing_failure_is_explained(wiz, fake_aws, monkeypatch):
    fake_aws.services["bedrock"].denied = True
    monkeypatch.setattr(wizard_module, "Aws", lambda region=None, session=None: fake_aws)
    wiz.parsing.value = "foundation_model"

    wiz.parser_list_btn.click()
    assert "bedrock:ListFoundationModels" in wiz.parser_hint.value
    assert wiz.parser.value == "nova-lite"


def test_wizard_can_be_opened_twice():
    """Opening the form must not replace kbforge.wizard() with the module it lives in."""
    import kbforge

    first, second = kbforge.wizard(region="us-east-1"), kbforge.wizard(region="us-east-1")
    assert first is not second and callable(kbforge.wizard)
