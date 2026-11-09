from pathlib import Path

import pytest

from kbforge import Chunking, KBSpec, S3Source, WebSource

EXAMPLE = Path(__file__).parent.parent / "examples" / "kb.yaml"


def test_string_sources_are_coerced():
    assert isinstance(KBSpec(name="a", source="s3://b/p/").source, S3Source)
    assert isinstance(KBSpec(name="a", source="https://docs.example.com").source, WebSource)


def test_s3_source_parts():
    src = S3Source("s3://bucket/some/prefix/")
    assert (src.bucket, src.prefix) == ("bucket", "some/prefix/")
    assert src.to_api()["s3Configuration"] == {
        "bucketArn": "arn:aws:s3:::bucket",
        "inclusionPrefixes": ["some/prefix/"],
    }
    assert "inclusionPrefixes" not in S3Source("s3://bucket").to_api()["s3Configuration"]


@pytest.mark.parametrize("bad", ["my kb", "a--b", "", "x" * 101])
def test_invalid_kb_names_rejected(bad):
    with pytest.raises(ValueError):
        KBSpec(name=bad, source="s3://b/")


def test_vector_store_aliases():
    assert KBSpec(name="a", source="s3://b/", vector_store="aoss").vector_store == "opensearch_serverless"
    with pytest.raises(ValueError):
        KBSpec(name="a", source="s3://b/", vector_store="pinecone")


def test_dimensions_validated_against_model():
    assert KBSpec(name="a", source="s3://b/", dimensions=512).dims == 512
    with pytest.raises(ValueError):
        KBSpec(name="a", source="s3://b/", embedding="cohere-en", dimensions=512)


def test_generated_names_fit_aws_limits():
    spec = KBSpec(name="a-very-long-knowledge-base-name-for-the-finance-team-2026", source="s3://b/")
    n = spec.names
    assert len(n.role) <= 64 and n.role.startswith("AmazonBedrockExecutionRoleForKnowledgeBase_")
    assert len(n.collection) <= 32
    assert len(n.access_policy) <= 32
    assert n.collection == n.collection.lower()


def test_chunking_api_shapes():
    assert Chunking.fixed(500, 10).to_api()["fixedSizeChunkingConfiguration"] == {
        "maxTokens": 500,
        "overlapPercentage": 10,
    }
    h = Chunking.hierarchical().to_api()
    assert h["chunkingStrategy"] == "HIERARCHICAL"
    assert len(h["hierarchicalChunkingConfiguration"]["levelConfigurations"]) == 2
    assert Chunking.none().to_api() == {"chunkingStrategy": "NONE"}


def test_example_yaml_round_trips():
    spec = KBSpec.from_yaml(EXAMPLE)
    assert spec.name == "hr-policies"
    assert KBSpec.from_dict(spec.to_dict()) == spec


def test_unknown_config_keys_are_reported():
    with pytest.raises(ValueError, match="Unknown config keys"):
        KBSpec.from_dict({"name": "a", "source": "s3://b/", "embeddings": "titan-v2"})
