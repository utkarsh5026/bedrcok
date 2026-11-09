# kbforge

**From documents in S3 to an Amazon Bedrock Knowledge Base in four notebook cells, with a plan you can read before anything is created.**

📖 **Guide with screenshots:** <https://utkarsh5026.github.io/bedrcok/>

![A kbforge plan rendered in JupyterLab](docs/assets/screens/jupyterlab.png)

## Why

Creating a Bedrock knowledge base by hand means an S3 Vectors bucket (or an OpenSearch collection with three policies), a vector index, an IAM role with the right trust and permissions, the knowledge base, a data source and an ingestion job. Usually that's a notebook of cells you run in order and hope.

kbforge turns that into:

```python
%pip install -q "kbforge[notebook] @ git+https://github.com/utkarsh5026/bedrcok.git"

from kbforge import KnowledgeBase

kb = KnowledgeBase("hr-policies", source="s3://my-company-docs/hr/")
kb.plan()     # read-only: checks your access, model and files; lists what will be built
kb.apply()    # builds it with live progress (safe to re-run if anything fails)
kb.ask("How many weeks of parental leave do we get?")
```

- **The plan comes first.** Each AWS resource is shown with its name, its purpose, and whether it will be created, reused or updated.
- **Problems surface up front.** No model access, a missing or empty bucket, files Bedrock will skip, or a region mismatch all appear before anything is built.
- **Re-runs are safe.** Resource names are derived from the KB name, so a second run finds what already exists. There's no state file.
- **Teardown is safe.** `kb.destroy()` only shows what would be deleted, and `destroy(confirm=True)` deletes it. Your documents, your own role, and anything kbforge didn't create are left alone.

Prefer a form? Run `kbforge.wizard()`. Prefer a terminal or CI? Use `kbforge plan kb.yaml` / `kbforge apply kb.yaml`.

## Getting started

1. Upload [`notebooks/getting_started.ipynb`](notebooks/getting_started.ipynb) to SageMaker and run it top to bottom.
2. Make sure your notebook role has the permissions in [`docs/notebook-role-policy.json`](docs/notebook-role-policy.json). If IAM is locked down, pass an existing Bedrock role as `role_arn=` instead.

The [guide](https://utkarsh5026.github.io/bedrcok/) covers every option, how to choose settings, and troubleshooting.

## Options at a glance

| Option | Default | Choices |
| --- | --- | --- |
| `source` | required | `s3://bucket/prefix/`, `https://site`, `S3Source`, `WebSource` |
| `embedding` | `titan-v2` | `titan-v2`, `cohere-en`, `cohere-multi`, `titan-v1` |
| `vector_store` | `s3_vectors` | `s3_vectors`, `opensearch_serverless` (needs `kbforge[opensearch]`) |
| `chunking` | `fixed` (300 tokens, 20%) | `fixed`, `hierarchical`, `semantic`, `none`, or `Chunking.fixed(500, 10)` |
| `role_arn` | create one | an existing role that trusts `bedrock.amazonaws.com` |
| `region`, `tags`, `description`, `permissions_boundary`, `sync` | | see the guide |

## Development

```bash
uv sync --all-extras          # package + dev tools
uv run pytest                 # tests run against an in-memory fake AWS, no account needed
uv run ruff check src tests && uv run mypy

# regenerate the screenshots in docs/ (runs a real JupyterLab with Playwright)
uv run --group docs python scripts/docs_screenshots.py
```

See [CLAUDE.md](CLAUDE.md) for how the code is organised.

## License

Apache-2.0
