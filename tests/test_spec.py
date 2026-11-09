from pathlib import Path

import pytest

from kbforge import Chunking, KBSpec, KnowledgeBase, S3Source, WebSource

EXAMPLES = Path(__file__).parent.parent / "examples"
EXAMPLE = EXAMPLES / "kb.yaml"


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


def test_example_json_matches_example_yaml():
    assert KBSpec.from_file(EXAMPLES / "kb.json") == KBSpec.from_file(EXAMPLE)
    assert KBSpec.from_json(EXAMPLES / "kb.json") == KBSpec.from_yaml(EXAMPLE)


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
def test_save_and_load_in_either_format(tmp_path, suffix):
    kb = KnowledgeBase("a", source="s3://b/docs/", chunking=Chunking.semantic(400, 1, 90), tags={"t": "1"})
    path = kb.save(tmp_path / f"kb{suffix}")
    assert KnowledgeBase.from_file(path).spec == kb.spec
    if suffix == ".json":
        assert path.read_text().lstrip().startswith("{")


def test_unknown_config_extension_is_explained(tmp_path):
    with pytest.raises(ValueError, match=r"\.yaml, \.yml or \.json"):
        KBSpec.from_file(tmp_path / "kb.txt")


def test_broken_json_points_at_the_line(tmp_path):
    path = tmp_path / "kb.json"
    path.write_text('{\n  "name": "a",\n  "source": "s3://b/",\n}')
    with pytest.raises(ValueError, match="isn't valid JSON.*line 4"):
        KBSpec.from_file(path)


def test_config_must_be_an_object(tmp_path):
    path = tmp_path / "kb.json"
    path.write_text('["a", "b"]')
    with pytest.raises(ValueError, match="one object"):
        KBSpec.from_file(path)


@pytest.mark.parametrize(
    "block",
    [
        {"strategy": "hierarchical", "parent_tokens": 2000, "child_tokens": 400, "overlap_tokens": 50},
        {"strategy": "semantic", "max_tokens": 500, "buffer_size": 1, "breakpoint_threshold": 90},
        {"strategy": "fixed", "max_tokens": 500},
        {"strategy": "none"},
    ],
)
def test_chunking_blocks_round_trip(block):
    chunking = Chunking.from_dict(block)
    assert chunking.strategy == block["strategy"]
    assert {k: v for k, v in chunking.to_dict().items() if k in block} == block


def test_chunking_settings_from_another_strategy_are_rejected():
    with pytest.raises(ValueError, match="overlap_percent can't be used with semantic chunking.*breakpoint_threshold"):
        KBSpec.from_dict(
            {"name": "a", "source": "s3://b/", "chunking": {"strategy": "semantic", "overlap_percent": 10}}
        )
    with pytest.raises(ValueError, match="takes no other settings"):
        Chunking.from_dict({"strategy": "none", "max_tokens": 300})
    with pytest.raises(ValueError, match="must be one of"):
        Chunking.from_dict({"strategy": "paragraph"})


@pytest.mark.parametrize(
    ("block", "message"),
    [
        ({"strategy": "fixed", "max_tokens": 10000}, "max_tokens must be between 1 and 8192, got 10000"),
        ({"strategy": "fixed", "overlap_percent": 0}, "overlap_percent must be between 1 and 99"),
        ({"strategy": "semantic", "buffer_size": 2}, "buffer_size must be between 0 and 1"),
        ({"strategy": "semantic", "breakpoint_threshold": 40}, "breakpoint_threshold must be between 50 and 99"),
        ({"strategy": "hierarchical", "overlap_tokens": 0}, "overlap_tokens must be at least 1"),
        ({"strategy": "hierarchical", "parent_tokens": 9000}, "parent_tokens must be between 1 and 8192"),
        ({"strategy": "fixed", "max_tokens": "300"}, "must be a whole number"),
        ({"strategy": "fixed", "max_tokens": True}, "must be a whole number"),
    ],
)
def test_out_of_range_chunking_is_rejected_before_anything_is_built(block, message):
    with pytest.raises(ValueError, match=message):
        Chunking.from_dict(block)


def test_hierarchical_layers_must_get_smaller():
    with pytest.raises(ValueError, match="child_tokens .* must be smaller than parent_tokens"):
        Chunking.hierarchical(parent_tokens=300, child_tokens=300)
    with pytest.raises(ValueError, match="overlap_tokens .* must be smaller than child_tokens"):
        Chunking.hierarchical(child_tokens=60, overlap_tokens=60)


def test_settings_of_other_strategies_are_not_range_checked():
    # A semantic chunking keeps the (unused) fixed-size defaults; only its own settings matter.
    assert Chunking.semantic(max_tokens=8192, buffer_size=1, breakpoint_threshold=99).embedded_tokens == 8192
    assert Chunking.hierarchical().embedded_tokens == 300
    assert Chunking.none().embedded_tokens is None
