# kbforge

A pip-installable Python package for building Amazon Bedrock Knowledge Bases from a SageMaker/Jupyter notebook. The flow is **describe → plan (read-only, shown as an HTML card) → apply (live progress) → ask / sync / destroy**. The users are colleagues who aren't AWS experts, so friendly errors and a readable plan matter as much as correctness.

## Commands

```bash
uv sync --all-extras                 # install package + dev deps (Python 3.10+, dev uses 3.11)
uv run pytest                        # all tests, against tests/fake_aws.py; no AWS needed
uv run ruff check src tests scripts makefile.py  # lint (ruff format to format)
uv run mypy                          # type-check src/
uv run --group docs python scripts/docs_screenshots.py --chromium <path>   # regenerate docs/assets/screens
```

The same tasks are wrapped by `make` (see `make help`): `make check` runs lint, format, types and tests and prints one summary table. The `Makefile` is a thin shim; the logic and `rich` output live in `makefile.py`, so add new tasks there with the `@task` decorator.

## Layout

- `src/kbforge/spec.py`: `KBSpec` (what the user wants), `S3Source`/`WebSource`, `Chunking`, and `Names` (every AWS name, derived deterministically from the KB name; this is why there's no state file).
- `src/kbforge/models.py`: catalogue of embedding models (alias → model id, dimensions).
- `src/kbforge/aws.py`: `Aws` wrapper (cached clients, identity, SageMaker assumed-role → role ARN), plus `wait_for` and `retry_on` for eventual consistency.
- `src/kbforge/checks.py`: preflight checks (credentials, model invoke, S3 scan, vector store, user-supplied role). A `fail` blocks apply.
- `src/kbforge/steps/`: one `Step` per AWS resource. `base.py` defines `Step`/`Action`/`Inspection`/`Ctx`; `iam.py`, `s3_vectors.py`, `opensearch.py` and `bedrock.py` implement them; `build_steps()` sets the order.
- `src/kbforge/plan.py`: `make_plan()` (checks + `inspect()` every step, read-only) and `Plan.apply()` (re-inspects each step just before acting, so re-runs resume).
- `src/kbforge/knowledge_base.py`: `KnowledgeBase`, the user-facing facade (`plan/apply/sync/ask/status/destroy`), plus `Answers` and `Status`.
- `src/kbforge/ui/`: `theme.py` (scoped CSS using JupyterLab variables, so it follows light/dark), `html.py` (cards via `_repr_html_`), `text.py` (CLI output), `progress.py` (live-updating display; `progress_target` redirects it for the wizard).
- `src/kbforge/wizard.py` (ipywidgets form), `guide.py` (getting-started card), `cli.py` (`kbforge` command).
- `tests/fake_aws.py`: in-memory fake of every service used. Add any new boto3 call here as well.
- `docs/`: GitHub Pages site (static `index.html`, served from `/docs` on `main`). Screenshots are generated, not hand-made.

## How a step works

`inspect(ctx)` must never change anything. It returns CREATE / EXISTS / UPDATE / CONFLICT / … and stores discovered ids in `ctx.out` (e.g. `kb_id`, `index_arn`) for later steps. `create()` does only what the inspection promised. `delete()` is called in reverse order. `owned()` checks the `managed-by: kbforge` tag so `destroy` never removes resources kbforge didn't create (unless `force=True`). Mismatches that can't change in place (embedding model, vector store, index dimension) are CONFLICTs, which block apply.

## Code style (the user feels strongly about this)

- Every module, class and public function gets a **plain-English docstring** a newcomer can follow: say what it's for and why, not just what it does.
- **No banner/divider comments** (`# ---- section ----`, `# ====`). Organise with files and classes instead. Short inline comments that explain *why* are fine.
- Prefer clear names over cleverness; keep `typing.Optional`/`List` style (ruff UP006/UP007 are ignored on purpose).
- User-facing messages say what to do next ("Run kb.apply() again — finished steps are skipped"), not just what went wrong. AWS errors go through `explain_error()`.
- All HTML output must escape user/AWS content with `esc()` and live inside `card()` so styles stay scoped to `.kbf`.

## Gotchas

- SageMaker runs as an assumed role; resource policies (the OpenSearch data access policy) need the real role ARN with its path. Use `Aws.principal_arn`.
- The generated IAM role name starts with `AmazonBedrockExecutionRoleForKnowledgeBase_` because AWS's managed Bedrock policies only allow `iam:PassRole` on roles with that prefix. Don't rename it.
- S3 Vectors indexes mark `AMAZON_BEDROCK_TEXT`/`AMAZON_BEDROCK_METADATA` as non-filterable; without this, large chunks hit the 2 KB filterable-metadata limit.
- `retry_on` only retries when the error message hints at propagation (role/index not visible yet), so real permission errors fail fast.
- In tests, `time.sleep` is patched out by an autouse fixture; the fake makes everything ACTIVE/COMPLETE immediately.
