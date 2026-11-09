"""Regenerate the screenshots used by the GitHub Pages guide in ``docs/``.

How it works:

1. Writes a demo notebook to a temporary folder. Its first (hidden) cell swaps AWS for the
   in-memory fake from ``tests/fake_aws.py``, so no AWS account is needed and the output is
   the same every time.
2. Starts a real JupyterLab (the same UI SageMaker Studio uses) with clean settings.
3. Uses Playwright to run every cell, fill in the wizard, and screenshot each cell into
   ``docs/assets/screens/``.
4. Renders the CLI's coloured output into a terminal-style screenshot.

Usage:

    uv run --group docs python scripts/docs_screenshots.py
    uv run --group docs python scripts/docs_screenshots.py --chromium /path/to/chrome
"""

from __future__ import annotations

import argparse
import contextlib
import html
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Generator, Optional

import nbformat
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets" / "screens"

SETUP = f"""
# Hidden setup: replace AWS with the in-memory fake used by the test-suite.
import sys
sys.path.insert(0, {str(ROOT)!r})
import kbforge.aws, kbforge.knowledge_base
from tests.fake_aws import FakeAws

fake = FakeAws()
fake.services["s3"].buckets["my-company-docs"] = {{
    **{{f"hr/policies/policy-{{i:02d}}.pdf": 180_000 + i * 9_000 for i in range(48)}},
    **{{f"hr/handbook/chapter-{{i:02d}}.docx": 95_000 + i * 4_000 for i in range(31)}},
    **{{f"hr/faq/topic-{{i:02d}}.md": 6_000 for i in range(12)}},
    **{{f"hr/intranet/page-{{i:02d}}.html": 22_000 for i in range(6)}},
    **{{f"hr/diagrams/org-{{i}}.png": 410_000 for i in range(3)}},
}}
agent = fake.services["bedrock-agent"]
_start = agent.start_ingestion_job
def _realistic_job(**kw):
    job = _start(**kw)["ingestionJob"]
    job["statistics"] = {{"numberOfDocumentsScanned": 97, "numberOfNewDocumentsIndexed": 97}}
    return {{"ingestionJob": job}}
agent.start_ingestion_job = _realistic_job
kbforge.knowledge_base.Aws = lambda region=None, session=None: fake
"""

#: (screenshot name or None, cell source). Names become docs/assets/screens/<name>.png.
CELLS = [
    (None, SETUP),
    ("guide", 'import kbforge\n\nkbforge.guide(region="us-east-1")'),
    (
        "describe",
        'from kbforge import KnowledgeBase\n\nkb = KnowledgeBase("hr-policies", source="s3://my-company-docs/hr/")\nkb',
    ),
    ("plan", "kb.plan()"),
    (
        "blocked",
        'typo = KnowledgeBase("finance-docs", source="s3://finanse-reports/2026/")\ntypo.plan()',
    ),
    (
        "progress",
        "# What kb.apply() looks like while it works (a snapshot mid-run)\n"
        "from kbforge.ui.progress import Progress\n"
        "p = Progress('Building hr-policies', [\n"
        "    ('S3 vector bucket', 'kbforge-hr-policies'),\n"
        "    ('Vector index', 'kbforge-hr-policies/hr-policies-index'),\n"
        "    ('IAM service role', 'AmazonBedrockExecutionRoleForKnowledgeBase_kbf-hr-policies'),\n"
        "    ('Bedrock knowledge base', 'hr-policies'),\n"
        "    ('Data source', 'hr-policies-source'),\n"
        "    ('First sync (ingestion)', 'ingestion job'),\n"
        "], 'hr-policies')\n"
        "for i, r in enumerate(['arn:aws:s3vectors:us-east-1:123456789012:bucket/kbforge-hr-policies',\n"
        "                       'arn:aws:s3vectors:…/index/hr-policies-index',\n"
        "                       'arn:aws:iam::123456789012:role/AmazonBedrockExecutionRoleForKnowledgeBase_kbf-…',\n"
        "                       'Q7K2M9XPLA', 'B3RT8WZK1D']):\n"
        "    p.start(i); p.rows[i].started -= [1, 2, 4, 38, 1][i]; p.done(i, r)\n"
        "p.start(5); p.rows[5].started -= 74; p.started -= 122\n"
        "import time; time.sleep(0.6)  # let the progress view redraw\n"
        "p.log(5, 'Waiting for ingestion · in progress · 61 scanned · 58 indexed (70s)')",
    ),
    ("apply", "kb.apply()"),
    ("ask", 'kb.ask("How many weeks of parental leave do we get?")'),
    ("status", "kb.status()"),
    ("destroy", "kb.destroy()"),
    ("wizard", 'kbforge.wizard(region="us-east-1")'),
]

ASK_RESULTS = [
    (
        0.71,
        "s3://my-company-docs/hr/policies/policy-07.pdf",
        "Parental leave. All employees are entitled to 26 weeks of fully paid parental leave, which can be "
        "taken in up to three blocks within 18 months of the birth or adoption. Leave may start up to four "
        "weeks before the expected due date.",
    ),
    (
        0.64,
        "s3://my-company-docs/hr/handbook/chapter-04.docx",
        "Chapter 4 — Time off. Besides annual leave, the company offers parental leave (see Policy 07), "
        "bereavement leave of up to 10 days, and 3 volunteering days per year.",
    ),
    (
        0.52,
        "s3://my-company-docs/hr/faq/topic-03.md",
        "Q: Can I split my parental leave? A: Yes — up to three blocks. Tell your manager and People Ops at "
        "least 8 weeks before each block starts.",
    ),
]


#: The fake answers every question with these hits, so the "ask" screenshot looks real.
SETUP += "\nfake.services['bedrock-agent-runtime'].retrieve = lambda **kw: {'retrievalResults': %r}\n" % [
    {"content": {"text": text}, "location": {"type": "S3", "s3Location": {"uri": uri}}, "score": score}
    for score, uri, text in ASK_RESULTS
]
CELLS[0] = (None, SETUP)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def write_lab_settings(settings_dir: Path) -> None:
    """Quiet, light, non-virtualised JupyterLab so every cell is rendered for screenshots."""
    files = {
        "@jupyterlab/apputils-extension/notification.jupyterlab-settings": {
            "fetchNews": "false",
            "checkForUpdates": False,
        },
        "@jupyterlab/apputils-extension/themes.jupyterlab-settings": {"theme": "JupyterLab Light"},
        "@jupyterlab/notebook-extension/tracker.jupyterlab-settings": {"windowingMode": "none"},
    }
    for rel, content in files.items():
        path = settings_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content))


def build_notebook(folder: Path) -> Path:
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(src) for _, src in CELLS]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    path = folder / "kbforge-demo.ipynb"
    nbformat.write(nb, path)
    return path


@contextlib.contextmanager
def jupyter_lab(folder: Path) -> Generator[str, None, None]:
    port = free_port()
    env = {
        **os.environ,
        "JUPYTERLAB_SETTINGS_DIR": str(folder / "settings"),
        "JUPYTERLAB_WORKSPACES_DIR": str(folder / "workspaces"),
    }
    write_lab_settings(folder / "settings")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "jupyter",
            "lab",
            "--no-browser",
            f"--port={port}",
            "--IdentityProvider.token=",
            "--ServerApp.password=",
            f"--ServerApp.root_dir={folder}",
            "--ServerApp.disable_check_xsrf=True",
            # Containers often run as root; Jupyter refuses to start as root without this.
            *(["--allow-root"] if hasattr(os, "geteuid") and os.geteuid() == 0 else []),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never send localhost via a proxy
    try:
        for _ in range(120):
            with contextlib.suppress(Exception):
                direct.open(f"{url}/api", timeout=1)
                break
            time.sleep(0.5)
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=20)


def cell(page: Page, index: int):  # type: ignore[no-untyped-def]
    return page.locator(".jp-Notebook .jp-Cell").nth(index)


def shoot(page: Page, index: int, name: str, output_only: bool = False) -> None:
    target = cell(page, index)
    if output_only:
        target = target.locator(".jp-OutputArea-output").first
    target.scroll_into_view_if_needed()
    page.wait_for_timeout(300)
    target.screenshot(path=str(OUT / f"{name}.png"), animations="disabled")
    print(f"  saved {name}.png")


def run_all_cells(page: Page, attempts: int = 4) -> None:
    """Click Run → Run All Cells, retrying if the kernel wasn't ready to accept it yet."""
    first_prompt = page.locator(".jp-Notebook .jp-Cell .jp-InputPrompt").first
    for _ in range(attempts):
        page.get_by_role("menuitem", name="Run", exact=True).click()
        page.get_by_text("Run All Cells", exact=True).click()
        try:
            page.wait_for_function(
                "el => /\\[(\\*|\\d+)\\]/.test(el.textContent)", arg=first_prompt.element_handle(), timeout=15_000
            )
            return
        except Exception:
            page.keyboard.press("Escape")
    raise RuntimeError("JupyterLab never started running the notebook")


def wait_until_executed(page: Page, count: int) -> None:
    """Wait until every code cell shows an execution number (not [*] or [ ])."""
    page.wait_for_function(
        """(n) => {
            const prompts = [...document.querySelectorAll('.jp-Notebook .jp-Cell .jp-InputPrompt')];
            return prompts.length >= n && prompts.slice(0, n).every(p => /\\[\\d+\\]/.test(p.textContent));
        }""",
        arg=count,
        timeout=60_000,
    )


def notebook_screens(url: str, chromium: Optional[str]) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chromium) if chromium else p.chromium.launch()
        page = browser.new_page(viewport={"width": 1360, "height": 3000}, device_scale_factor=2)
        page.goto(f"{url}/lab/tree/kbforge-demo.ipynb")
        page.wait_for_selector(".jp-Notebook .jp-Cell", timeout=60_000)
        page.wait_for_selector(".jp-Notebook-ExecutionIndicator[data-status='idle']", timeout=60_000)
        page.keyboard.press("Control+b")  # hide the file browser for a cleaner window
        run_all_cells(page)
        try:
            wait_until_executed(page, len(CELLS))
        except Exception:
            page.screenshot(path=str(OUT.parent / "debug-timeout.png"))
            raise
        page.wait_for_timeout(1500)

        for index, (name, _) in enumerate(CELLS):
            if name and name != "wizard":
                shoot(page, index, name, output_only=(name == "progress"))

        page.set_viewport_size({"width": 1360, "height": 900})
        plan_cell = cell(page, [name for name, _ in CELLS].index("plan"))
        plan_cell.evaluate("el => el.scrollIntoView({block: 'start'})")
        page.wait_for_timeout(400)
        page.screenshot(path=str(OUT / "jupyterlab.png"))
        print("  saved jupyterlab.png")
        page.set_viewport_size({"width": 1360, "height": 3000})

        wizard_index = len(CELLS) - 1
        wizard = cell(page, wizard_index)
        wizard.locator("input[placeholder='hr-policies']").fill("benefits-faq")
        wizard.locator("input[placeholder='s3://my-bucket/folder/']").fill("s3://my-company-docs/hr/")
        page.keyboard.press("Tab")
        shoot(page, wizard_index, "wizard-form")
        wizard.get_by_role("button", name="Review plan").click()
        wizard.get_by_text("Ready.", exact=False).wait_for(timeout=30_000)
        shoot(page, wizard_index, "wizard-plan")
        wizard.get_by_role("button", name="Build it").click()
        wizard.locator(".badge", has_text="ready").wait_for(timeout=30_000)
        shoot(page, wizard_index, "wizard-built")
        browser.close()


def ansi_to_html(text: str) -> str:
    colours = {"31": "#ff7b72", "32": "#7ee787", "33": "#e3b341", "34": "#79c0ff", "90": "#8b949e", "1": None}
    out, open_span = [], False
    for part in re.split(r"(\x1b\[[0-9;]*m)", text):
        match = re.fullmatch(r"\x1b\[([0-9;]*)m", part)
        if not match:
            out.append(html.escape(part))
            continue
        if open_span:
            out.append("</span>")
            open_span = False
        code = match.group(1)
        if code and code != "0":
            style = "font-weight:700" if code == "1" else f"color:{colours.get(code, '#e6edf3')}"
            out.append(f'<span style="{style}">')
            open_span = True
    return "".join(out) + ("</span>" if open_span else "")


def cli_screen(chromium: Optional[str]) -> None:
    sys.path.insert(0, str(ROOT))
    os.environ["FORCE_COLOR"] = "1"
    from tests.fake_aws import FakeAws

    from kbforge import KnowledgeBase

    kb = KnowledgeBase.from_yaml(ROOT / "examples" / "kb.yaml")
    kb._aws = FakeAws()  # type: ignore[assignment]
    body = ansi_to_html("$ kbforge plan examples/kb.yaml\n" + str(kb.plan()))
    page_html = (
        "<html><body style='margin:0;background:#0d1117'>"
        "<div id='t' style='display:inline-block;padding:0 0 18px;background:#0d1117;border-radius:10px;"
        "font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:#e6edf3;min-width:880px'>"
        "<div style='padding:10px 14px;display:flex;gap:7px'>"
        + "".join(
            f"<i style='width:12px;height:12px;border-radius:50%;background:{c}'></i>"
            for c in ("#ff5f57", "#febc2e", "#28c840")
        )
        + f"</div><pre style='margin:0;padding:0 22px'>{body}</pre></div></body></html>"
    )
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chromium) if chromium else p.chromium.launch()
        page = browser.new_page(device_scale_factor=2)
        page.set_content(page_html)
        page.locator("#t").screenshot(path=str(OUT / "cli-plan.png"))
        print("  saved cli-plan.png")
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the GitHub Pages screenshots.")
    parser.add_argument("--chromium", help="path to a Chromium binary (default: Playwright's own)")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        build_notebook(folder)
        print("Running the demo notebook in JupyterLab…")
        with jupyter_lab(folder) as url:
            notebook_screens(url, args.chromium)
    print("Rendering the CLI screenshot…")
    cli_screen(args.chromium)


if __name__ == "__main__":
    main()
