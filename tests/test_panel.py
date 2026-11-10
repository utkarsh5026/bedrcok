"""The sync panel: buttons for check / sync / stop, the list of syncs, and following a running sync."""

import pytest

pytest.importorskip("ipywidgets")

import kbforge  # noqa: E402
from kbforge.panel import SyncPanel, SyncPicker  # noqa: E402


def panel(kb) -> SyncPanel:
    """A panel that follows syncs in the test's own thread, so everything happens in order."""
    return SyncPanel(kb, background=False, interval=0)


def test_before_building_the_panel_explains_and_only_offers_check(kb):
    p = panel(kb)

    assert p.sync_btn.disabled and p.stop_btn.disabled and not p.check_btn.disabled
    assert "hasn&#x27;t been created yet" in p.view.value

    p.check_btn.click()
    assert "A sync would fail" in p.status.value
    assert "kbforge · sync check" in p.view.value


def test_after_building_it_lists_syncs_and_shows_the_latest(kb):
    kb.apply()
    p = panel(kb)

    assert not p.sync_btn.disabled and p.stop_btn.disabled
    assert "Last sync: <b>complete</b>" in p.status.value
    assert "Synced" in p.header.value
    (cards,) = p.list_box.children
    assert len(cards.children) == 1 and cards.value == p.selected
    assert "Sync of hr-policies" in p.view.value


def test_check_button_shows_the_read_only_checks(kb):
    kb.apply()
    p = panel(kb)

    p.check_btn.click()

    assert "Ready to sync" in p.status.value
    assert "Your permissions" in p.view.value and "Knowledge base&#x27;s role" in p.view.value
    assert p.selected is None  # so clicking a sync card shows it again


def test_sync_now_runs_a_sync_and_follows_it_to_the_end(kb, fake_aws):
    kb.apply()
    fake_aws.services["bedrock-agent"].job_polls = 3
    p = panel(kb)

    p.sync_btn.click()

    assert "Sync complete" in p.status.value
    assert len(p.list_box.children[0].children) == 2
    assert p.following is None and not p.sync_btn.disabled and p.stop_btn.disabled


def test_sync_now_stops_at_the_checks_when_a_sync_would_fail(kb, fake_aws):
    kb.apply()
    fake_aws.services["iam"].caller_denied = {"bedrock:StartIngestionJob"}
    jobs = len(fake_aws.services["bedrock-agent"].jobs)
    p = panel(kb)

    p.sync_btn.click()

    assert "Didn&#x27;t start" in p.status.value or "Didn't start" in p.status.value
    assert "bedrock:StartIngestionJob" in p.view.value
    assert len(fake_aws.services["bedrock-agent"].jobs) == jobs


def test_a_sync_started_elsewhere_is_followed_when_the_panel_opens(kb, fake_aws):
    kb.apply()
    fake_aws.services["bedrock-agent"].job_polls = 2
    kb.sync(wait=False)

    p = panel(kb)

    assert "Sync complete" in p.status.value
    assert fake_aws.services["bedrock-agent"].jobs[-1]["status"] == "COMPLETE"


def test_clicking_a_sync_shows_its_details(kb, fake_aws):
    kb.apply()
    kb.sync(check=False)
    p = panel(kb)
    oldest = fake_aws.services["bedrock-agent"].jobs[0]["ingestionJobId"]

    p.list_box.children[0].value = oldest

    assert p.selected == oldest and oldest in p.view.value


def test_picker_lists_kbforge_knowledge_bases(kb, fake_aws, monkeypatch):
    monkeypatch.setattr("kbforge.aws.Aws", lambda region=None, session=None: fake_aws)
    assert "Nothing to sync yet" in SyncPicker(region="us-east-1").widget.value

    kb.apply()
    picker = SyncPicker(region="us-east-1")
    assert "Which knowledge base?" in picker.widget.children[0].children[0].value


def test_sync_panel_entry_points_return_a_widget(kb, fake_aws, monkeypatch):
    kb.apply()
    monkeypatch.setattr("kbforge.knowledge_base.Aws", lambda region=None, session=None: fake_aws)
    monkeypatch.setattr("kbforge.panel.SyncPanel.__init__.__defaults__", (False, 0))

    assert "kbf-wiz" in kbforge.sync_panel("hr-policies", region="us-east-1")._dom_classes
    assert "kbf-wiz" in kb.sync_panel()._dom_classes
