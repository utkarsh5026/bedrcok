# Example configs

A knowledge base can be described in **YAML** or **JSON**. Both formats take exactly the same keys, and kbforge picks the format from the file extension (`.yaml`, `.yml` or `.json`).

| File | What it is |
| --- | --- |
| [`kb.yaml`](kb.yaml) | Every option, commented, including all four chunking strategies and the web-crawler source. **Start here.** |
| [`kb.json`](kb.json) | The same knowledge base as JSON. JSON has no comments, so the choices are listed below. |

```bash
kbforge init kb.json        # write a starter (kb.yaml by default)
kbforge plan kb.json        # every command takes either format
kbforge apply kb.json
```

```python
from kbforge import KnowledgeBase

kb = KnowledgeBase.from_file("kb.json")   # or "kb.yaml"
kb.save("kb.yaml")                        # write it back out in either format
```

## Chunking in JSON

Pick **one** strategy and give only its settings. Any setting you leave out keeps the default shown here. A setting that belongs to another strategy (say `overlap_percent` with `semantic`) is reported as an error, so it can't be ignored by mistake. Values outside the ranges in the table below are rejected too, before anything is built.

**Fixed size** (default): equal-sized chunks with some overlap. A safe choice for most document sets.

```json
"chunking": {
  "strategy": "fixed",
  "max_tokens": 300,
  "overlap_percent": 20
}
```

**Hierarchical**: search runs over small child chunks, and the larger parent chunk around each match is returned. Good for long, structured documents such as policies and manuals.

```json
"chunking": {
  "strategy": "hierarchical",
  "parent_tokens": 1500,
  "child_tokens": 300,
  "overlap_tokens": 60
}
```

**Semantic**: splits where the topic changes. Gives better chunks, but each sync makes extra model calls.

```json
"chunking": {
  "strategy": "semantic",
  "max_tokens": 300,
  "buffer_size": 0,
  "breakpoint_threshold": 95
}
```

**None**: each file is one chunk. Only for files that are already small.

```json
"chunking": { "strategy": "none" }
```

| Setting | Strategy | Default | Meaning |
| --- | --- | --- | --- |
| `max_tokens` | fixed, semantic | 300 | 1–8,192. Largest chunk size. Should fit the embedding model: titan-v2 reads up to 8,192 tokens, the cohere models 512 (the plan warns if it doesn't) |
| `overlap_percent` | fixed | 20 | 1–99. How much of each chunk is repeated at the start of the next |
| `parent_tokens` | hierarchical | 1500 | 1–8,192. The context returned with each match |
| `child_tokens` | hierarchical | 300 | Smaller than `parent_tokens`. What is embedded and searched, so it's the size checked against the model |
| `overlap_tokens` | hierarchical | 60 | 1 or more, smaller than `child_tokens`. Tokens repeated between neighbouring chunks |
| `buffer_size` | semantic | 0 | 0 or 1. Sentences either side included when comparing each sentence |
| `breakpoint_threshold` | semantic | 95 | 50–99. Higher means fewer splits and bigger chunks |

## Every other key

| Key | Required | Default | Values |
| --- | --- | --- | --- |
| `name` | yes | | Letters, digits, `-` or `_`; up to 100 characters |
| `source` | yes | | `{"s3": "s3://bucket/prefix/"}` or `{"web": ["https://…"], "scope": "HOST_ONLY", "max_pages": 500, "rate_limit": 300, "include": [], "exclude": []}` |
| `region` | | your profile's region | e.g. `"us-east-1"` |
| `description` | | `""` | Free text |
| `embedding` | | `"titan-v2"` | `"titan-v2"`, `"cohere-en"`, `"cohere-multi"`, `"titan-v1"`, or a raw model id together with `dimensions` |
| `dimensions` | | model default | titan-v2 only: `256`, `512` or `1024` |
| `vector_store` | | `"s3_vectors"` | `"s3_vectors"` or `"opensearch_serverless"` |
| `parsing` | | `{"strategy": "default"}` | `{"strategy": "foundation_model", "model": "nova-lite", "prompt": "…"}` (`prompt` optional), `{"strategy": "data_automation"}`, or just a model name such as `"nova-lite"`. `kbforge parsers` lists the models your account can use |
| `role_arn` | | kbforge creates a role | An existing role that trusts `bedrock.amazonaws.com` |
| `permissions_boundary` | | none | Policy ARN to attach to the role kbforge creates |
| `sync` | | `true` | `false` to build now and run the first ingestion later |
| `tags` | | `{}` | Extra tags; `managed-by: kbforge` is always added |

Embedding model, vector store, dimensions, parsing and chunking can't be changed once the knowledge base exists. To try different settings, give it a new `name`, compare the two with `kb.ask()`, then `destroy` the one you don't keep.
