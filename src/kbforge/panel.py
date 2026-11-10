"""``kb.sync_panel()``: keep a knowledge base up to date with buttons instead of code.

    kb = KnowledgeBase.open("hr-policies")
    kb.sync_panel()

    import kbforge
    kbforge.sync_panel()          # first pick one of the knowledge bases kbforge built

The panel shows where the knowledge base stands (synced 2 h ago, a sync running, the last one
failed), its recent syncs as cards you can click for the details, and four buttons:

* **Check** runs the read-only pre-sync checks (``kb.check_sync()``): your permissions, the
  knowledge base's role, the source and the models.
* **Sync now** checks first, then starts a sync, or follows the one already running.
* **Stop** stops the running sync.
* **Refresh** reads the latest state from AWS.

While a sync runs, the panel follows it from a background thread and redraws itself every few
seconds, so the notebook stays free: you can keep working in other cells, or click Stop. It
shows the same cards as the code API (``IngestionJob``, ``SyncCheck``), so what you see here
matches ``kb.syncs()`` and ``kb.sync()``.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

from botocore.exceptions import BotoCoreError, ClientError

from .ingestion import RUNNING, IngestionJob, IngestionJobs, follow, human_duration, start_job, stop_job, time_ago
from .steps import explain_error
from .sync_check import SyncCheck
from .ui import panel_style as look
from .ui import wizard_style as ui
from .ui.theme import esc

if TYPE_CHECKING:
    from .knowledge_base import KnowledgeBase


def _message(exc: BaseException) -> str:
    """An exception as a sentence for the panel: AWS errors get kbforge's usual explanation."""
    if isinstance(exc, (ClientError, BotoCoreError)):
        return explain_error(exc)
    return str(exc)


def _widgets() -> Tuple[Any, Any, Any]:
    """ipywidgets and the option-card classes, or a friendly error if ipywidgets is missing."""
    try:
        import ipywidgets as w

        from .ui.choices import Choice, OptionCards
    except ImportError as exc:
        raise ImportError(
            "The sync panel needs ipywidgets, which is missing. Run %pip install --upgrade kbforge (it includes "
            "ipywidgets), then restart the kernel. kb.sync() and kb.syncs() work without it."
        ) from exc
    return w, Choice, OptionCards


class SyncPanel:
    """The widgets of one knowledge base's sync panel, and what each button does.

    ``background=False`` follows a running sync in the calling thread instead of a background
    one; tests use it so everything happens in order.
    """

    #: How many recent syncs the list shows.
    LIMIT = 8

    def __init__(self, kb: KnowledgeBase, background: bool = True, interval: float = 10.0):
        w, self.Choice, self.OptionCards = _widgets()
        self.w = w
        self.kb = kb
        self.background = background
        self.interval = interval
        self.jobs: Optional[IngestionJobs] = None
        #: The sync whose details are shown below the list.
        self.selected: Optional[str] = None
        #: The sync being followed live, if any.
        self.following: Optional[str] = None
        self.busy = False
        self._live_html = ""
        #: Set by ``close()`` so a background watcher stops (the sync itself carries on in AWS).
        self._closed = threading.Event()

        self.header = w.HTML(look.panel_header(kb.name, self._region(), kb.spec.source.describe(), ""))
        self.list_html = w.HTML()
        self.list_box = w.Box([], layout=w.Layout(width="100%"))
        self.status = w.HTML()
        self.check_btn = self._button("Check", "shield", "ghost")
        self.refresh_btn = self._button("Refresh", "refresh", "ghost")
        self.stop_btn = self._button("Stop", "stop", "ghost", disabled=True)
        self.sync_btn = self._button("Sync now", "play", "go")
        self.view = w.HTML()

        self.check_btn.on_click(self._on_check)
        self.refresh_btn.on_click(lambda _: self.refresh())
        self.stop_btn.on_click(self._on_stop)
        self.sync_btn.on_click(self._on_sync)

        self.widget = self._layout()
        self.refresh()

    def _region(self) -> str:
        """The region to show, without failing if no AWS session can be made yet."""
        try:
            return self.kb.aws.region
        except Exception:
            return self.kb.spec.region or ""

    def _button(self, label: str, icon: str, kind: str, disabled: bool = False) -> Any:
        """A button in the wizard's style; ``kind`` is primary, go (green) or ghost."""
        button = self.w.Button(description=label, icon=icon, disabled=disabled)
        button.add_class("kbf-btn")
        button.add_class(kind)
        return button

    def _layout(self) -> Any:
        """Header, the list of recent syncs, the action bar, then the details card underneath."""
        w = self.w
        jobs = w.VBox([self.list_html, self.list_box], layout=w.Layout(width="100%"))
        jobs.add_class("kbf-jobs")
        actions = w.Box(
            [self.status, self.check_btn, self.refresh_btn, self.stop_btn, self.sync_btn],
            layout=w.Layout(width="100%"),
        )
        actions.add_class("kbf-actions")
        shell = w.VBox([self.header, jobs, actions], layout=w.Layout(width="100%"))
        shell.add_class("kbf-shell")
        out = w.VBox([self.view], layout=w.Layout(width="100%"))
        out.add_class("kbf-out")
        root = w.VBox([shell, out])
        root.add_class("kbf")
        root.add_class("kbf-wiz")
        return root

    def close(self) -> None:
        """Stop following a running sync in the background (the sync itself carries on in AWS)."""
        self._closed.set()

    def refresh(self, show: Optional[str] = None) -> None:
        """Read the recent syncs from AWS and redraw everything; ``show`` picks the sync to detail."""
        try:
            jobs = self.jobs = self.kb.syncs(limit=self.LIMIT)
        except Exception as exc:
            self._unavailable(_message(exc))
            return
        latest = jobs.latest
        self.header.value = look.panel_header(self.kb.name, jobs.region, jobs.source, look.state_pill(latest))
        if latest is not None and latest.running and self.following is None:
            # Someone (perhaps you, in another cell) started a sync: follow it.
            self._follow(latest.job_id, followed=True)
            return
        self.selected = show or self.selected or (latest.job_id if latest else None)
        self._draw_list()
        if self.selected:
            self._show_job(self.selected)
        else:
            self.view.value = look.message_card(
                "Never synced",
                "Click Sync now to make the documents searchable. kbforge checks everything first, and Check "
                "does just the checks if you'd like to look before syncing.",
                "c-blue",
            )
        if not self.busy:
            self.status.value = self._idle_status()
            self._set_busy(False)

    def _unavailable(self, message: str) -> None:
        """Show why the syncs can't be listed (e.g. the knowledge base doesn't exist yet)."""
        self.header.value = look.panel_header(
            self.kb.name, self._region(), self.kb.spec.source.describe(), look.state_pill(None, error=message)
        )
        self.list_html.value = ""
        self.list_box.children = []
        self.view.value = look.message_card("Can't show the syncs", message)
        self.status.value = ui.status("bad", "Couldn't read the syncs. <b>Check</b> shows what's wrong.")
        self.sync_btn.disabled = self.stop_btn.disabled = True
        self.check_btn.disabled = False

    def _draw_list(self) -> None:
        """The recent syncs as clickable cards, the selected one highlighted."""
        jobs = self.jobs
        if jobs is None:
            return
        self.list_html.value = look.jobs_heading(len(jobs), jobs.more)
        if not jobs.items:
            self.list_box.children = []
            return
        choices = [self.Choice(job.job_id, *look.job_choice(job)) for job in jobs]
        cards = self.OptionCards(choices, value=self.selected, columns=4)
        cards.observe(self._on_select, names="value")
        self.list_box.children = [cards]

    def _clear_selection(self) -> None:
        """Un-highlight the list, so clicking any sync (even the last one shown) shows it again."""
        self.selected = None
        for child in self.list_box.children:
            child.value = None

    def _on_select(self, change: Dict[str, Any]) -> None:
        if change["new"] is not None:
            self.selected = change["new"]
            self._show_job(change["new"])

    def _show_job(self, job_id: str) -> None:
        """Show one sync in detail: the live view if it's the one being followed, else read from AWS."""
        if job_id == self.following and self._live_html:
            self.view.value = self._live_html
            return
        self.view.value = ui.loading("Loading the sync", f"Job {job_id}, with the files that need attention")
        try:
            self.view.value = self.kb.sync_job(job_id)._repr_html_()
        except Exception as exc:
            self.view.value = look.message_card("Couldn't load this sync", _message(exc))

    def _set_busy(self, busy: bool) -> None:
        """Grey out the buttons that mustn't be clicked while something is running."""
        self.busy = busy
        self.sync_btn.disabled = self.check_btn.disabled = busy
        self.stop_btn.disabled = self.following is None

    def _idle_status(self) -> str:
        """The status line when nothing is happening: how the last sync went, and what to click."""
        latest = self.jobs.latest if self.jobs else None
        if latest is None:
            return ui.status("todo", "Never synced. Click <b>Sync now</b>; kbforge checks everything first.")
        label = esc(latest.outcome[0])
        when = esc(time_ago(latest.started_at))
        return ui.status(
            "todo",
            f"Last sync: <b>{label}</b>, {when}. Click <b>Sync now</b> after documents change, or <b>Check</b> "
            "to look for problems first.",
        )

    def _on_check(self, _: Any) -> None:
        """Run the read-only checks and show their card."""
        self._set_busy(True)
        self.status.value = ui.status("busy", "Checking permissions, roles, the source and models (read-only)…")
        self.view.value = ui.loading("Checking " + self.kb.name, "Read-only: nothing is started or changed.")
        try:
            report = self.kb.check_sync()
        except Exception as exc:
            self._set_busy(False)
            self.status.value = ui.status("bad", esc(_message(exc)))
            return
        self._set_busy(False)
        self._clear_selection()
        self.view.value = report._repr_html_()
        self.status.value = self._check_status(report)

    @staticmethod
    def _check_status(report: SyncCheck) -> str:
        """The status line after the checks."""
        if report.blockers:
            count = len(report.blockers)
            return ui.status("bad", f"<b>A sync would fail.</b> The {count} problem(s) below say what to fix.")
        if report.running:
            return ui.status("info", "<b>A sync is running.</b> <b>Sync now</b> follows it and <b>Stop</b> ends it.")
        if report.warnings:
            return ui.status("warn", "<b>Ready, with warnings.</b> Read them below: some files may fail.")
        return ui.status("ok", "<b>Ready to sync.</b> Nothing was started. Click <b>Sync now</b> when you like.")

    def _on_sync(self, _: Any) -> None:
        """Check, then start a sync (or follow the running one) and watch it."""
        self._set_busy(True)
        self.status.value = ui.status("busy", "Checking before starting (read-only)…")
        self.view.value = ui.loading("Checking before the sync", "Permissions, roles, the source and models.")
        try:
            report = self.kb.check_sync()
            if not report.ready:
                self._set_busy(False)
                self._clear_selection()
                self.view.value = report._repr_html_()
                self.status.value = ui.status(
                    "bad", "<b>Didn't start:</b> a sync would fail. Fix the problems below, then try again."
                )
                return
            started, followed = start_job(self.kb.aws, report.kb_id, report.ds_id)
        except Exception as exc:
            self._set_busy(False)
            self.view.value = look.message_card("Couldn't start the sync", _message(exc))
            self.status.value = ui.status("bad", "Couldn't start the sync; the message below says why.")
            return
        self._follow(started["ingestionJobId"], followed)

    def _on_stop(self, _: Any) -> None:
        """Ask Bedrock to stop the running sync; the watcher reports when it has."""
        job_id = self.following or (self.jobs.running.job_id if self.jobs and self.jobs.running else None)
        if job_id is None or self.jobs is None:
            self.status.value = ui.status("info", "No sync is running.")
            return
        try:
            stop_job(self.kb.aws, self.jobs.kb_id, self.jobs.ds_id, job_id)
        except Exception as exc:
            self.status.value = ui.status("bad", esc(_message(exc)))
            return
        self.stop_btn.disabled = True
        self.status.value = ui.status("busy", "Stopping… Bedrock finishes the documents in hand, then stops.")
        if self.following is None:
            self._follow(job_id, followed=True)

    def _follow(self, job_id: str, followed: bool) -> None:
        """Show ``job_id`` live until it finishes, in the background unless ``background=False``."""
        self.following = self.selected = job_id
        self._live_html = ""
        try:
            self.jobs = self.kb.syncs(limit=self.LIMIT)
        except Exception as exc:
            self.following = None
            self._set_busy(False)
            self.status.value = ui.status("bad", esc(_message(exc)))
            return
        self._draw_list()
        self._set_busy(True)
        self.status.value = ui.status(
            "busy", "<b>Syncing.</b> This panel updates itself; you can keep working in other cells."
        )
        args = (self.jobs.kb_id, self.jobs.ds_id, job_id, followed)
        if self.background:
            threading.Thread(target=self._watch, args=args, daemon=True, name=f"kbforge-sync-{job_id}").start()
        else:
            self._watch(*args)

    def _sleep(self) -> Callable[[float], Any]:
        """How the watcher waits: interruptible by ``close()`` in the background, plain sleep otherwise."""
        return self._closed.wait if self.background else time.sleep

    def _watch(self, kb_id: str, ds_id: str, job_id: str, followed: bool) -> None:
        """Poll the sync, redrawing the panel each time, then show how it ended."""
        source = self.jobs.source if self.jobs else self.kb.spec.source.describe()
        region = self.kb.aws.region

        def update(raw: Dict[str, Any]) -> None:
            job = IngestionJob.from_api(raw, self.kb.name, region, source)
            job.followed = followed
            self._live_html = job._repr_html_()
            if self.selected == job_id:
                self.view.value = self._live_html
            self.header.value = look.panel_header(self.kb.name, region, source, look.state_pill(job))
            if job.running:
                counts = f" · {esc(job.stats_text)}" if job.scanned else ""
                self.status.value = ui.status(
                    "busy",
                    f"<b>{esc(job.outcome[0].capitalize())}</b>{counts} · {human_duration(job.seconds)}. "
                    "This panel updates itself; you can keep working in other cells.",
                )

        try:
            final = follow(
                self.kb.aws,
                kb_id,
                ds_id,
                job_id,
                update,
                interval=self.interval,
                sleep=self._sleep(),
                keep_going=lambda: not self._closed.is_set(),
            )
        except Exception as exc:
            self.following = None
            self._set_busy(False)
            self.status.value = ui.status("bad", esc(_message(exc)))
            return
        self.following = None
        self._set_busy(False)
        if final["status"] in RUNNING:
            return  # close() was called; the sync carries on in AWS
        self.refresh(show=job_id)
        self.status.value = self._finished_status(IngestionJob.from_api(final, self.kb.name, region, source))

    @staticmethod
    def _finished_status(job: IngestionJob) -> str:
        """The status line once a followed sync has ended."""
        if job.status == "FAILED":
            return ui.status("bad", "<b>The sync failed.</b> See why below; <b>Check</b> looks for the cause.")
        if job.status == "STOPPED":
            return ui.status("info", "<b>Sync stopped.</b> Click <b>Sync now</b> to finish indexing.")
        if job.failed:
            return ui.status(
                "warn", f"<b>Sync finished, but {job.failed:,} file(s) failed.</b> The list below says why."
            )
        return ui.status("ok", f"<b>Sync complete.</b> {esc(job.stats_text)}. Try it: <code>kb.ask(...)</code>")


class SyncPicker:
    """Pick one of the knowledge bases kbforge built in a region, then show its sync panel."""

    def __init__(self, region: Optional[str] = None, session: Any = None):
        from . import knowledge_bases

        w, Choice, OptionCards = _widgets()
        self.w = w
        self.region, self.session = region, session
        self.out = w.VBox([], layout=w.Layout(width="100%"))
        try:
            found = knowledge_bases(region=region, session=session)
        except Exception as exc:
            self.widget = w.HTML(look.message_card("Couldn't list your knowledge bases", _message(exc)))
            return
        if not found.items:
            self.widget = w.HTML(
                look.message_card(
                    "Nothing to sync yet",
                    f"kbforge hasn't built any knowledge bases in {found.region}. Run kbforge.wizard() to build one, "
                    "or pass region=... to look elsewhere.",
                    "c-blue",
                )
            )
            return
        choices = [Choice(s.name, s.name, s.source, "database", meta=f"last sync: {s.last_sync_text}") for s in found]
        cards = OptionCards(choices, value=None, columns=3)
        cards.observe(self._on_pick, names="value")
        pick = w.VBox([w.HTML(look.picker_heading(len(found))), cards], layout=w.Layout(width="100%"))
        pick.add_class("kbf-jobs")
        header = w.HTML(look.picker_header(found.region))
        shell = w.VBox([header, pick], layout=w.Layout(width="100%"))
        shell.add_class("kbf-shell")
        root = w.VBox([shell, self.out])
        root.add_class("kbf")
        root.add_class("kbf-wiz")
        self.widget = root

    def _on_pick(self, change: Dict[str, Any]) -> None:
        """Open the chosen knowledge base (read-only) and show its panel below the list."""
        from .knowledge_base import KnowledgeBase

        name = change["new"]
        self.out.children = [self.w.HTML(ui.loading(f"Opening {name}", "Reading its settings from AWS."))]
        try:
            kb = KnowledgeBase.open(name, region=self.region, session=self.session)
        except Exception as exc:
            self.out.children = [self.w.HTML(look.message_card(f"Couldn't open {name}", _message(exc)))]
            return
        self.out.children = [SyncPanel(kb).widget]
