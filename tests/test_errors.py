"""How mistakes read: friendly AWS errors, short notebook errors, a clean CLI, and typo hints."""

import pytest
from botocore.exceptions import EndpointConnectionError, NoCredentialsError

from kbforge import KBForgeError, KBSpec, KnowledgeBase
from kbforge.cli import main
from kbforge.errors import did_you_mean
from kbforge.steps import explain_error

from .fake_aws import aws_error


@pytest.mark.parametrize(
    "exc, expected",
    [
        (aws_error("ExpiredTokenException", "The security token included in the request is expired"), "expired"),
        (aws_error("UnrecognizedClientException", "The security token is invalid"), "credentials"),
        (NoCredentialsError(), "credentials"),
        (aws_error("ThrottlingException", "Rate exceeded"), "Wait a minute"),
        (EndpointConnectionError(endpoint_url="https://bedrock.us-esat-1.amazonaws.com"), "region name"),
        (aws_error("ResourceNotFoundException", "Could not resolve the foundation model"), "Model access"),
        (aws_error("AccessDeniedException", "not authorized to perform: bedrock:Retrieve"), "missing a permission"),
    ],
)
def test_common_aws_errors_say_what_to_do(exc, expected):
    assert expected in explain_error(exc)


def test_unknown_errors_keep_their_code():
    assert explain_error(aws_error("WeirdException", "something odd")) == "WeirdException: something odd"


def test_notebook_calls_turn_aws_errors_into_kbforge_errors(kb, fake_aws, monkeypatch):
    kb.apply()

    def expired(**_):
        raise aws_error("ExpiredTokenException", "The security token included in the request is expired")

    monkeypatch.setattr(fake_aws.services["bedrock-agent-runtime"], "retrieve", expired)
    with pytest.raises(KBForgeError, match="session has expired") as info:
        kb.ask("anything")
    assert info.value.__cause__ is not None  # the original error is kept for debugging

    monkeypatch.setattr(fake_aws.services["bedrock-agent"], "get_paginator", lambda name: expired())
    with pytest.raises(KBForgeError, match="session has expired"):
        kb.status()


def test_kbforge_errors_show_without_a_traceback_in_ipython():
    lines = KBForgeError("Something is wrong.\n\nDo this next.")._render_traceback_()
    assert "Something is wrong." in lines[0]
    assert lines[-1] == "Do this next."
    assert not any("Traceback" in line for line in lines)


def test_missing_region_is_a_friendly_error(monkeypatch):
    for var in ("AWS_DEFAULT_REGION", "AWS_REGION", "AWS_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/nonexistent")
    kb = KnowledgeBase("hr-policies", source="s3://docs/hr/")
    with pytest.raises(KBForgeError, match="region"):
        kb.aws  # noqa: B018  # creating the connection is what fails


def test_cli_prints_aws_errors_without_a_traceback(tmp_path, monkeypatch, capsys):
    config = tmp_path / "kb.yaml"
    assert main(["init", str(config)]) == 0

    def broken(*_, **__):
        raise aws_error("UnrecognizedClientException", "The security token included in the request is invalid.")

    monkeypatch.setattr("kbforge.cli.KnowledgeBase.status", broken)
    assert main(["status", str(config), "--region", "us-east-1"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: AWS didn't accept your credentials")
    assert "Traceback" not in err


def test_cli_explains_a_missing_region(tmp_path, monkeypatch, capsys):
    for var in ("AWS_DEFAULT_REGION", "AWS_REGION", "AWS_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/nonexistent")
    config = tmp_path / "kb.yaml"
    main(["init", str(config)])
    capsys.readouterr()
    assert main(["status", str(config)]) == 1
    assert "No AWS region configured" in capsys.readouterr().err


def test_did_you_mean():
    assert did_you_mean("embeding", ["embedding", "dimensions"]) == " Did you mean 'embedding'?"
    assert did_you_mean("Titan-V2", ["titan-v2"]) == " Did you mean 'titan-v2'?"
    assert did_you_mean("zzz", ["embedding"]) == ""


@pytest.mark.parametrize(
    "kwargs, hint",
    [
        ({"embedding": "titan2"}, "'titan-v2'"),
        ({"vector_store": "opensearch_serverles"}, "'opensearch_serverless'"),
        ({"chunking": "semantik"}, "'semantic'"),
        ({"parsing": "nova-lit"}, "'nova-lite'"),
        ({"parsing": {"strategy": "foundation_model", "modle": "nova-lite"}}, "'model'"),
        ({"chunking": {"strategy": "fixed", "max_token": 300}}, "'max_tokens'"),
    ],
)
def test_typos_get_a_suggestion(kwargs, hint):
    with pytest.raises(ValueError, match=f"Did you mean {hint}"):
        KnowledgeBase("hr-policies", source="s3://docs/hr/", **kwargs)


def test_mistyped_keyword_argument_is_explained():
    with pytest.raises(TypeError, match="Did you mean 'embedding'"):
        KnowledgeBase("hr-policies", source="s3://docs/hr/", embeding="titan-v2")


def test_mistyped_config_key_is_explained():
    with pytest.raises(ValueError, match="Did you mean 'embedding'"):
        KBSpec.from_dict({"name": "a", "source": "s3://b/", "embeddings": "titan-v2"})
