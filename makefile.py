"""Developer task runner for kbforge: the code behind every ``make`` target.

The ``Makefile`` is deliberately thin: each target just calls ``uv run python makefile.py <task>``.
All the real logic lives here, in Python, because that gives us three things a plain Makefile can't:

* readable, colourful output (via `rich`) that tells you what's running, how long it took and what to do
  if it failed, which matters because many contributors are not build-tool experts;
* a ``check`` task that runs every gate (lint, format, types, tests) and shows one summary table, instead of
  stopping at the first failure and hiding the rest;
* the same behaviour on every OS, since nothing depends on shell quirks.

You can also call it directly, e.g. ``uv run python makefile.py test -k plan`` (extra arguments are passed
straight through to the underlying tool).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parent

#: Folders that hold Python we lint, format and test. Kept in one place so every task agrees.
PY_PATHS = ["src", "tests", "scripts", "makefile.py"]

console = Console(highlight=False)


@dataclass
class Task:
    """One runnable task, as listed by ``make help``.

    ``group`` decides which section of the help table it appears in, and ``emoji`` is the little icon shown
    beside it in the help table and in the run banner, so a glance at the terminal tells you which step you're on.
    """

    name: str
    emoji: str
    group: str
    summary: str
    func: Callable[[List[str]], int]
    usage: Optional[str] = None


@dataclass
class StepResult:
    """The outcome of one command, kept so ``check`` can draw its summary table at the end."""

    label: str
    emoji: str
    code: int
    seconds: float
    hint: str = ""

    @property
    def ok(self) -> bool:
        """True when the command exited with status 0."""
        return self.code == 0


@dataclass
class Registry:
    """The collection of tasks, filled in by the ``@task`` decorator below.

    Using a registry (rather than a big if/else) means adding a task is just writing one decorated function,
    and the help screen can never drift out of sync with what actually exists.
    """

    tasks: Dict[str, Task] = field(default_factory=dict)

    def groups(self) -> List[str]:
        """Group names in the order their first task was declared, so help reads top-down like a workflow."""
        seen: List[str] = []
        for t in self.tasks.values():
            if t.group not in seen:
                seen.append(t.group)
        return seen


REGISTRY = Registry()


def task(name: str, emoji: str, group: str, summary: str, usage: Optional[str] = None):
    """Register the decorated function as a task called ``name``.

    The function receives any extra command-line arguments and returns a process exit code (0 = success).
    """

    def wrap(func: Callable[[List[str]], int]) -> Callable[[List[str]], int]:
        REGISTRY.tasks[name] = Task(name, emoji, group, summary, func, usage)
        return func

    return wrap


def fmt_seconds(seconds: float) -> str:
    """Turn a duration into a short human string like ``820ms`` or ``3.4s``."""
    return f"{seconds * 1000:.0f}ms" if seconds < 1 else f"{seconds:.1f}s"


def banner(emoji: str, title: str, cmd: Optional[List[str]] = None) -> None:
    """Print a coloured header before a step so it's easy to find in long output."""
    console.print()
    console.rule(f"[bold cyan]{emoji}  {title}[/]", align="left", style="cyan")
    if cmd:
        console.print(Text("$ " + " ".join(cmd), style="dim"))


def run(label: str, emoji: str, cmd: List[str], hint: str = "", quiet_banner: bool = False) -> StepResult:
    """Run one external command, streaming its output, and report how it went.

    The command inherits our terminal, so tools like pytest and ruff keep their own colours. On failure we
    print ``hint`` (what to try next) rather than only an exit code, in line with the project's "friendly
    errors" rule.
    """
    if not quiet_banner:
        banner(emoji, label, cmd)
    start = time.perf_counter()
    try:
        code = subprocess.call(cmd, cwd=ROOT)
    except FileNotFoundError:
        console.print(f"[bold red]✗ Couldn't find [white]{cmd[0]}[/]. Is it installed and on your PATH?[/]")
        code = 127
    except KeyboardInterrupt:
        console.print("[yellow]⏹  Interrupted.[/]")
        code = 130
    took = time.perf_counter() - start
    if code == 0:
        console.print(f"[bold green]✅ {label} passed[/] [dim]({fmt_seconds(took)})[/]")
    else:
        console.print(f"[bold red]❌ {label} failed[/] [dim](exit {code}, {fmt_seconds(took)})[/]")
        if hint:
            console.print(f"   [yellow]💡 {hint}[/]")
    return StepResult(label, emoji, code, took, hint)


def uv(*args: str) -> List[str]:
    """Build a ``uv run ...`` command so tools always come from the project's locked environment."""
    return ["uv", "run", *args]


@task("help", "📖", "General", "Show this list of tasks")
def help_task(_: List[str]) -> int:
    """Print every task, grouped, as a table. This is also what a bare ``make`` shows."""
    table = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 2))
    table.add_column("", no_wrap=True)
    table.add_column("Command", style="bold cyan", no_wrap=True)
    table.add_column("What it does")
    for group in REGISTRY.groups():
        table.add_row("", f"[bold yellow]{group}[/]", "")
        for t in REGISTRY.tasks.values():
            if t.group == group:
                desc = t.summary + (f"\n[dim]{t.usage}[/]" if t.usage else "")
                table.add_row(t.emoji, f"make {t.name}", desc)
        table.add_row("", "", "")
    console.print(
        Panel(
            table,
            title="[bold]🔨 kbforge developer tasks[/]",
            subtitle='[dim]pass extra args with ARGS="..."  e.g. make test ARGS="-k plan"[/]',
            border_style="blue",
            padding=(1, 2),
        )
    )
    return 0


@task("info", "🔎", "General", "Show tool versions and where things live")
def info(_: List[str]) -> int:
    """Print a small environment report; handy when asking a teammate "why does it work for you?"."""

    def version(cmd: List[str]) -> str:
        try:
            out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
            return (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr) else "?"
        except FileNotFoundError:
            return "[red]not installed[/]"

    table = Table(box=None, show_header=False, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("🐍 Python", sys.version.split()[0] + f"  [dim]{sys.executable}[/]")
    table.add_row("⚡ uv", version(["uv", "--version"]))
    table.add_row("🧹 ruff", version(uv("ruff", "--version")))
    table.add_row("🔬 mypy", version(uv("mypy", "--version")))
    table.add_row("🧪 pytest", version(uv("pytest", "--version")))
    table.add_row("📁 Project", str(ROOT))
    table.add_row(
        "🌍 AWS region", os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "[dim]unset[/]"
    )
    console.print(Panel(table, title="[bold]Environment[/]", border_style="blue"))
    return 0


@task("install", "📦", "Setup", "Install the package with all extras plus dev tools")
def install(args: List[str]) -> int:
    """Create/refresh ``.venv`` from ``uv.lock`` so everyone runs identical tool versions."""
    return run(
        "Install", "📦", ["uv", "sync", "--all-extras", *args], "Is uv installed? See https://docs.astral.sh/uv/"
    ).code


@task("install-docs", "🖼️ ", "Setup", "Also install the docs/screenshot tools (JupyterLab, Playwright)")
def install_docs(args: List[str]) -> int:
    """Add the ``docs`` dependency group, only needed to regenerate the GitHub Pages screenshots."""
    return run("Install docs tools", "🖼️ ", ["uv", "sync", "--all-extras", "--group", "docs", *args]).code


@task("test", "🧪", "Quality", "Run the test-suite (offline, against tests/fake_aws.py)", 'make test ARGS="-k plan -x"')
def test(args: List[str]) -> int:
    """Run pytest. No AWS account is needed: every service is replaced by the in-memory fake."""
    hint = 'Re-run just the failing test with: make test ARGS="-k <name> -x"'
    return run("Tests", "🧪", uv("pytest", *args), hint).code


@task("cov", "📊", "Quality", "Run tests with a coverage report (needs pytest-cov)")
def cov(args: List[str]) -> int:
    """Run pytest with coverage, pulling in pytest-cov on the fly so it isn't a permanent dependency."""
    cmd = ["uv", "run", "--with", "pytest-cov", "pytest", "--cov=kbforge", "--cov-report=term-missing", *args]
    return run("Coverage", "📊", cmd).code


@task("lint", "🧹", "Quality", "Lint with ruff")
def lint(args: List[str]) -> int:
    """Check for bugs and style problems; ``make fix`` can repair most of them automatically."""
    return run("Lint", "🧹", uv("ruff", "check", *PY_PATHS, *args), "Try `make fix` to auto-fix what ruff can.").code


@task("format", "🎨", "Quality", "Format code with ruff (rewrites files)")
def format_(args: List[str]) -> int:
    """Rewrite files into the project's standard layout."""
    return run("Format", "🎨", uv("ruff", "format", *PY_PATHS, *args)).code


@task("format-check", "🎨", "Quality", "Check formatting without changing files")
def format_check(args: List[str]) -> int:
    """The read-only version of ``format``; used by ``check`` so CI never edits your files."""
    return run("Format check", "🎨", uv("ruff", "format", "--check", *PY_PATHS, *args), "Run `make format`.").code


@task("typecheck", "🔬", "Quality", "Type-check src/ with mypy")
def typecheck(args: List[str]) -> int:
    """Run mypy over ``src`` (configured in pyproject.toml)."""
    return run(
        "Type check", "🔬", uv("mypy", *args), "Fix the reported lines, or add a narrow `# type: ignore[code]`."
    ).code


@task("fix", "🔧", "Quality", "Auto-fix lint issues and format everything")
def fix(args: List[str]) -> int:
    """Apply ruff's safe fixes, then format: the quickest way to a clean ``make lint``."""
    first = run("Lint fixes", "🔧", uv("ruff", "check", "--fix", *PY_PATHS, *args))
    second = run("Format", "🎨", uv("ruff", "format", *PY_PATHS))
    return first.code or second.code


@task("check", "🚦", "Quality", "Run every gate (lint, format, types, tests) and summarise")
def check(_: List[str]) -> int:
    """Run all quality gates, keep going after a failure, and show one table at the end.

    Running everything (instead of stopping early) means one ``make check`` tells you every problem at once,
    which is what you want before pushing.
    """
    steps = [
        ("Lint", "🧹", uv("ruff", "check", *PY_PATHS), "make fix"),
        ("Format", "🎨", uv("ruff", "format", "--check", *PY_PATHS), "make format"),
        ("Types", "🔬", uv("mypy"), "make typecheck"),
        ("Tests", "🧪", uv("pytest"), "make test"),
    ]
    results = [run(label, emoji, cmd, f"Details: {hint}") for label, emoji, cmd, hint in steps]

    table = Table(header_style="bold magenta", border_style="dim")
    table.add_column("Step")
    table.add_column("Result", justify="center")
    table.add_column("Time", justify="right", style="dim")
    for r in results:
        status = "[bold green]✅ pass[/]" if r.ok else f"[bold red]❌ fail ({r.code})[/]"
        table.add_row(f"{r.emoji} {r.label}", status, fmt_seconds(r.seconds))

    failed = [r for r in results if not r.ok]
    console.print()
    if failed:
        names = ", ".join(r.label for r in failed)
        console.print(Panel(table, title=f"[bold red]🚨 {len(failed)} check(s) failed: {names}[/]", border_style="red"))
        console.print(
            "[yellow]💡 Run the matching make target to see just that output, fix, then `make check` again.[/]"
        )
        return 1
    console.print(Panel(table, title="[bold green]🎉 All checks passed. Ready to push![/]", border_style="green"))
    return 0


@task("build", "🏗️ ", "Release", "Build the sdist and wheel into dist/")
def build(args: List[str]) -> int:
    """Produce the files you'd upload to PyPI or ``pip install`` on SageMaker."""
    result = run("Build", "🏗️ ", ["uv", "build", *args])
    if result.ok:
        for f in sorted((ROOT / "dist").glob("kbforge-*")):
            console.print(f"   [green]📦 {f.relative_to(ROOT)}[/] [dim]({f.stat().st_size / 1024:.0f} KB)[/]")
    return result.code


@task("cli", "💻", "Release", "Run the kbforge command-line tool", 'make cli ARGS="--help"')
def cli(args: List[str]) -> int:
    """Run the installed ``kbforge`` entry point from the dev environment, for trying out CLI changes."""
    return run("kbforge CLI", "💻", uv("kbforge", *(args or ["--help"]))).code


@task("docs-screens", "📸", "Docs", "Regenerate light and dark docs screenshots (CHROMIUM=/path for your own browser)")
def docs_screens(args: List[str]) -> int:
    """Rebuild the GitHub Pages screenshots (light and dark) by running the docs notebook against the fake AWS."""
    cmd = ["uv", "run", "--group", "docs", "python", "scripts/docs_screenshots.py"]
    if os.environ.get("CHROMIUM"):
        cmd += ["--chromium", os.environ["CHROMIUM"]]
    return run("Docs screenshots", "📸", [*cmd, *args], "Run `make install-docs` first.").code


@task("docs-serve", "🌐", "Docs", "Preview the docs site at http://localhost:8000 (PORT=... to change)")
def docs_serve(args: List[str]) -> int:
    """Serve ``docs/`` locally so you can check the GitHub Pages site before pushing. Ctrl+C to stop."""
    port = os.environ.get("PORT", "8000")
    banner("🌐", f"Serving docs at http://localhost:{port}  [dim](Ctrl+C to stop)[/]")
    result = run(
        "Docs server", "🌐", [sys.executable, "-m", "http.server", port, "-d", "docs", *args], quiet_banner=True
    )
    return 0 if result.code in (0, 130) else result.code


@task("clean", "🧽", "Housekeeping", "Remove caches and build output (keeps .venv)")
def clean(_: List[str]) -> int:
    """Delete generated files so the next run starts fresh. The virtualenv is kept; use ``clean-all`` for that."""
    banner("🧽", "Clean")
    targets = [
        ROOT / p for p in ("dist", "build", ".pytest_cache", ".mypy_cache", ".ruff_cache", "htmlcov", ".coverage")
    ]
    targets += list(ROOT.glob("src/*.egg-info"))
    targets += [p for p in ROOT.rglob("__pycache__") if ".venv" not in p.parts]
    return _remove(targets)


@task("clean-all", "💣", "Housekeeping", "Like clean, but also delete .venv (run `make install` after)")
def clean_all(args: List[str]) -> int:
    """Full reset, for when the environment itself seems broken."""
    code = clean(args)
    return code or _remove([ROOT / ".venv"])


def _remove(paths: List[Path]) -> int:
    """Delete each existing path and list what went, so ``clean`` is never a silent mystery."""
    removed = 0
    for p in paths:
        if not p.exists():
            continue
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
        console.print(f"   [dim]🗑️  {p.relative_to(ROOT)}[/]")
        removed += 1
    console.print(f"[bold green]✅ Removed {removed} item(s)[/]" if removed else "[green]✨ Already clean[/]")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Look up the requested task and run it, suggesting close matches for typos."""
    argv = list(sys.argv[1:] if argv is None else argv)
    name = argv.pop(0) if argv else "help"
    if name in ("-h", "--help"):
        name = "help"
    found = REGISTRY.tasks.get(name)
    if found is None:
        import difflib

        console.print(f"[bold red]❓ Unknown task [white]{name!r}[/].[/]")
        close = difflib.get_close_matches(name, REGISTRY.tasks, n=3)
        if close:
            console.print("[yellow]💡 Did you mean: " + ", ".join(f"[bold]make {c}[/]" for c in close) + "?[/]")
        console.print("[dim]Run `make help` to see every task.[/]")
        return 2
    try:
        return found.func(argv)
    except KeyboardInterrupt:
        console.print("\n[yellow]⏹  Interrupted.[/]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
