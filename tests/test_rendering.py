"""The notebook cards and terminal text show the right things — and escape user content."""

import kbforge
from kbforge.cli import main


def test_plan_card_shows_pipeline_checks_and_next_step(kb):
    html = kb.plan()._repr_html_()

    for expected in ("Data flow", "Preflight checks", "What will be built", "kb.apply()", "Amazon S3 Vectors"):
        assert expected in html
    assert "3 files" in html


def test_blocked_plan_card_says_nothing_will_change(fake_aws):
    kb = kbforge.KnowledgeBase("x", source="s3://missing/", region="us-east-1")
    kb._aws = fake_aws

    assert "Blocked" in kb.plan()._repr_html_()


def test_answers_card_escapes_document_text(kb):
    kb.apply()
    html = kb.ask("leave?")._repr_html_()

    assert "&lt;b&gt;26 weeks&lt;/b&gt;" in html
    assert "<b>26 weeks</b>" not in html
    assert "s3://docs/hr/leave.pdf" in html


def test_text_rendering(kb):
    text = str(kb.plan())
    assert "Preflight checks" in text and "+ create" in text

    result = kb.apply()
    assert "is ready" in str(result)


def test_kb_object_and_guide_render_without_aws():
    kb = kbforge.KnowledgeBase("demo", source="https://docs.example.com", region="us-east-1")
    assert "Not checked against AWS yet" in kb._repr_html_()
    assert "kbforge.wizard()" in kbforge.guide(region="us-east-1")._repr_html_()


def test_cli_models_and_init(tmp_path, capsys):
    assert main(["models"]) == 0
    assert "titan-v2" in capsys.readouterr().out

    target = tmp_path / "kb.yaml"
    assert main(["init", str(target)]) == 0
    assert kbforge.KBSpec.from_yaml(target).name == "my-knowledge-base"
    assert main(["init", str(target)]) == 1  # refuses to overwrite
