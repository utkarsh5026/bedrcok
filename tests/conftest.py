import pytest

from kbforge import KBSpec, KnowledgeBase, S3Source

from .fake_aws import FakeAws


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    """Polling helpers sleep between checks; tests shouldn't."""
    monkeypatch.setattr("kbforge.aws.time.sleep", lambda _: None)


@pytest.fixture
def spec() -> KBSpec:
    return KBSpec(name="hr-policies", source=S3Source("s3://my-company-docs/hr/"), region="us-east-1")


@pytest.fixture
def fake_aws() -> FakeAws:
    return FakeAws()


@pytest.fixture
def kb(spec, fake_aws) -> KnowledgeBase:
    """A KnowledgeBase wired to the in-memory fake AWS."""
    kb = KnowledgeBase.from_spec(spec)
    kb._aws = fake_aws  # type: ignore[assignment]
    return kb
