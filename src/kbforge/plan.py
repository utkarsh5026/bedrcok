"""Plans: "here's what I'm about to do" — and then doing it.

``make_plan`` runs the preflight checks and asks every step what it would do, without
changing anything in AWS. The resulting ``Plan`` shows itself as a card in a notebook (or as
text in a terminal). When it looks right, ``plan.apply()`` carries it out with live progress.

Apply is safe to re-run: every step re-checks AWS right before acting, so anything already
created is detected and skipped. If a run fails halfway, fix the cause and apply again.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .aws import Aws
from .checks import FAIL, Check, run_checks
from .spec import KBSpec
from .steps import Action, Ctx, Inspection, Log, Step, build_steps, explain_error


@dataclass
class PlanItem:
    """One row of the plan: a step plus what it decided to do."""

    step: Step
    inspection: Inspection
    resource: str
    explain: str
    details: Optional[Tuple[str, str]] = None


@dataclass
class ApplyResult:
    """What happened when a plan was applied. Displays as a summary card in notebooks."""

    ok: bool
    mode: str
    spec: KBSpec
    region: str
    out: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    failed_step: str = ""
    seconds: float = 0.0

    @property
    def kb_id(self) -> Optional[str]:
        """The knowledge base id, once it exists."""
        return self.out.get("kb_id")

    def _repr_html_(self) -> str:
        from .ui.html import render_apply_result

        return render_apply_result(self)

    def __str__(self) -> str:
        from .ui.text import render_apply_result

        return render_apply_result(self)


@dataclass
class Plan:
    """The full picture before anything changes: checks, plus one ``PlanItem`` per resource.

    ``mode`` is ``"create"`` for building and ``"destroy"`` for tearing down.
    """

    spec: KBSpec
    aws: Aws
    mode: str
    checks: List[Check]
    items: List[PlanItem]
    error: str = ""
    force: bool = False

    @property
    def blockers(self) -> List[str]:
        """Everything that must be fixed before ``apply()`` is allowed. Empty means good to go."""
        problems = [f"{c.title}: {c.detail}" for c in self.checks if c.status == FAIL]
        problems += [
            f"{item.step.title}: {item.inspection.warning}"
            for item in self.items
            if item.inspection.action == Action.CONFLICT
        ]
        if self.error:
            problems.append(self.error)
        return problems

    @property
    def ready(self) -> bool:
        """True when there are no blockers and ``apply()`` may run."""
        return not self.blockers

    def count(self, action: Action) -> int:
        """How many steps will take ``action``."""
        return sum(1 for item in self.items if item.inspection.action == action)

    @property
    def has_work(self) -> bool:
        """False when everything already exists and there's nothing to do."""
        return any(item.inspection.action.does_work for item in self.items)

    def apply(self) -> ApplyResult:
        """Carry out the plan, showing live progress. Refuses to start if there are blockers."""
        if not self.ready:
            return ApplyResult(
                ok=False,
                mode=self.mode,
                spec=self.spec,
                region=self.aws.region,
                error="The plan has blockers, so nothing was changed:\n- " + "\n- ".join(self.blockers),
            )
        if self.mode == "destroy":
            return _run_destroy(self)
        return _run_create(self)

    def _repr_html_(self) -> str:
        from .ui.html import render_plan

        return render_plan(self)

    def __str__(self) -> str:
        from .ui.text import render_plan

        return render_plan(self)


def make_plan(spec: KBSpec, aws: Aws, mode: str = "create", force: bool = False) -> Plan:
    """Build a plan by running preflight checks and inspecting every step (read-only)."""
    checks = run_checks(spec, aws, identity_only=(mode == "destroy"))
    if any(c.status == FAIL and c.title == "AWS credentials" for c in checks):
        return Plan(spec, aws, mode, checks, items=[])

    ctx = Ctx(spec, aws)
    steps = build_steps(spec)
    inspections = _inspect_all(steps, ctx)
    if mode == "destroy":
        items = [
            _item(step, _destroy_decision(step, insp, ctx, force), ctx)
            for step, insp in reversed(list(zip(steps, inspections, strict=True)))
            if step.deletable
        ]
    else:
        items = [_item(step, insp, ctx) for step, insp in zip(steps, inspections, strict=True)]
    return Plan(spec, aws, mode, checks, items, force=force)


def _inspect_all(steps: List[Step], ctx: Ctx) -> List[Inspection]:
    """Inspect steps in order. A step that can't be inspected becomes a blocker, not a crash."""
    results = []
    for step in steps:
        try:
            results.append(step.inspect(ctx))
        except Exception as exc:
            results.append(Inspection(Action.CONFLICT, warning=f"Couldn't check this: {explain_error(exc)}"))
    return results


def _item(step: Step, inspection: Inspection, ctx: Ctx) -> PlanItem:
    return PlanItem(step, inspection, step.resource(ctx), step.explain(ctx), step.details(ctx))


def _destroy_decision(step: Step, found: Inspection, ctx: Ctx, force: bool) -> Inspection:
    """Turn "what exists?" into "what will destroy do?", protecting things we didn't create."""
    if found.action == Action.CREATE:
        return Inspection(Action.ABSENT, "nothing to delete")
    if found.action == Action.USE:
        return Inspection(Action.KEEP, "your role — left alone")
    try:
        owned = step.owned(ctx)
    except Exception:
        owned = None
    if owned is False and not force:
        return Inspection(
            Action.KEEP,
            warning="Not created by kbforge (no managed-by tag), so it's left alone. "
            "Use destroy(force=True) to delete it anyway.",
        )
    return Inspection(Action.DELETE)


def _logger(progress: Any, i: int) -> Log:
    """A ``log`` callback that shows messages under row ``i`` of the progress view."""

    def log(message: str) -> None:
        """Show ``message`` under this step's row."""
        progress.log(i, message)

    return log


def _run_create(plan: Plan) -> ApplyResult:
    from .ui.progress import Progress

    ctx = Ctx(plan.spec, plan.aws)
    steps = build_steps(plan.spec)
    progress = Progress(f"Building {plan.spec.name}", [(s.title, s.resource(ctx)) for s in steps], plan.spec.name)
    started = time.time()
    for i, step in enumerate(steps):
        try:
            inspection = step.inspect(ctx)
            if inspection.action == Action.CONFLICT:
                raise RuntimeError(inspection.warning)
            if not inspection.action.does_work:
                progress.skip(i, inspection.note or inspection.action.value)
                continue
            progress.start(i)
            progress.done(i, step.create(ctx, _logger(progress, i), inspection))
        except Exception as exc:
            message = explain_error(exc)
            progress.fail(i, message)
            return ApplyResult(
                False, "create", plan.spec, plan.aws.region, ctx.out, message, step.title, time.time() - started
            )
    result = ApplyResult(True, "create", plan.spec, plan.aws.region, ctx.out, seconds=time.time() - started)
    progress.close()
    return result


def _run_destroy(plan: Plan) -> ApplyResult:
    from .ui.progress import Progress

    ctx = Ctx(plan.spec, plan.aws)
    steps = build_steps(plan.spec)
    found = _inspect_all(steps, ctx)  # forward pass fills ctx.out with ids we need to delete
    targets = [(s, f) for s, f in reversed(list(zip(steps, found, strict=True))) if s.deletable]
    progress = Progress(f"Deleting {plan.spec.name}", [(s.title, s.resource(ctx)) for s, _ in targets], plan.spec.name)
    started = time.time()
    for i, (step, inspection) in enumerate(targets):
        decision = _destroy_decision(step, inspection, ctx, plan.force)
        if decision.action != Action.DELETE:
            progress.skip(i, decision.note or "kept")
            continue
        try:
            progress.start(i)
            progress.done(i, step.delete(ctx, _logger(progress, i)))
        except Exception as exc:
            message = explain_error(exc)
            progress.fail(i, message)
            return ApplyResult(
                False, "destroy", plan.spec, plan.aws.region, ctx.out, message, step.title, time.time() - started
            )
    result = ApplyResult(True, "destroy", plan.spec, plan.aws.region, ctx.out, seconds=time.time() - started)
    progress.close()
    return result
