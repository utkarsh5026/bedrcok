"""The ``kbforge`` command, for people who prefer a terminal (or CI) to a notebook.

kbforge init                    # write a starter kb.yaml (or: kbforge init kb.json)
kbforge plan kb.yaml            # checks + what would be built (kb.json works everywhere too)
kbforge apply kb.yaml           # build it (asks before changing anything)
kbforge sync kb.yaml            # re-ingest after documents change (checks first; --no-wait to not wait)
kbforge check-sync kb.yaml      # would a sync work? permissions, roles, source, models (read-only)
kbforge syncs kb.yaml           # recent syncs; `kbforge syncs kb.yaml latest` (or a job id) for details
kbforge stop-sync kb.yaml       # stop the running sync
kbforge ask kb.yaml "question"  # try a query (add --answer for a written answer with citations)
kbforge list                    # the knowledge bases kbforge built in this region
kbforge status kb.yaml
kbforge destroy kb.yaml         # delete it (asks first)
kbforge models                  # list supported embedding models
kbforge parsers                 # list the models your account can use to parse documents
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from botocore.exceptions import BotoCoreError, ClientError

from .aws import Aws
from .knowledge_base import KBForgeError, KnowledgeBase
from .models import EMBEDDING_MODELS
from .parsers import known_parsing_models, list_parsing_models
from .plan import ApplyResult, make_plan
from .spec import config_format
from .steps import explain_error
from .sync_check import SyncCheck

STARTER_YAML = """\
# Describe your knowledge base, then run:  kbforge plan kb.yaml
# Every option, with all four chunking strategies spelled out, is in examples/kb.yaml.
name: my-knowledge-base
# region: us-east-1           # defaults to your AWS profile's region

source:
  s3: s3://your-bucket/your-folder/

embedding: titan-v2          # titan-v2 | cohere-en | cohere-multi | titan-v1
vector_store: s3_vectors     # s3_vectors | opensearch_serverless

# How text is read out of each file. "default" pulls out plain text (free). To have a
# vision model read tables, charts and scans in PDFs, list the models with `kbforge parsers`:
# parsing:
#   strategy: foundation_model    # default | foundation_model | data_automation
#   model: nova-lite

# Pick ONE strategy and keep only its settings:
#   fixed:        max_tokens, overlap_percent
#   hierarchical: parent_tokens, child_tokens, overlap_tokens
#   semantic:     max_tokens, buffer_size, breakpoint_threshold
#   none:         (no settings)
chunking:
  strategy: fixed
  max_tokens: 300
  overlap_percent: 20

# role_arn: arn:aws:iam::123456789012:role/AmazonBedrockExecutionRoleForKnowledgeBase_x
sync: true
"""

#: The JSON starter holds the same settings. JSON has no comments, so the choices are listed in
#: examples/kb.json and examples/README.md instead.
STARTER_JSON = (
    json.dumps(
        {
            "name": "my-knowledge-base",
            "source": {"s3": "s3://your-bucket/your-folder/"},
            "embedding": "titan-v2",
            "vector_store": "s3_vectors",
            "chunking": {"strategy": "fixed", "max_tokens": 300, "overlap_percent": 20},
            "sync": True,
        },
        indent=2,
    )
    + "\n"
)


def _confirm(question: str) -> bool:
    """Ask a yes/no question on the terminal. Anything but 'y' means no."""
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _load(args: argparse.Namespace) -> KnowledgeBase:
    """Read the config file named on the command line (YAML or JSON, by extension)."""
    return KnowledgeBase.from_file(args.config, region=args.region)


def cmd_init(args: argparse.Namespace) -> int:
    """``kbforge init``: write a starter kb.yaml, or kb.json if that's the name given (never overwrites)."""
    path = Path(args.path)
    if path.exists():
        print(f"{path} already exists — not overwriting it.")
        return 1
    path.write_text(STARTER_JSON if config_format(path) == "json" else STARTER_YAML)
    print(f"Wrote {path}. Edit the source and name, then run: kbforge plan {path}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """``kbforge plan``: print the plan; exit code 2 if it's blocked."""
    plan = _load(args).plan()
    print(plan)
    return 0 if plan.ready else 2


def cmd_apply(args: argparse.Namespace) -> int:
    """``kbforge apply``: show the plan, ask for confirmation, then build."""
    kb = _load(args)
    plan = kb.plan()
    print(plan)
    if not plan.ready:
        return 2
    if not plan.has_work:
        return 0
    if not args.yes and not _confirm("\nBuild it?"):
        print("Nothing was changed.")
        return 1
    result = plan.apply()
    print(result)
    return 0 if result.ok else 1


def cmd_sync(args: argparse.Namespace) -> int:
    """``kbforge sync``: check, then re-ingest the data source; exit code 2 if blocked, 1 if the sync failed."""
    result = _load(args).sync(wait=not args.no_wait, check=not args.skip_checks)
    print(result)
    if isinstance(result, SyncCheck):
        return 2
    return 1 if result.status in ("FAILED", "STOPPED") else 0


def cmd_check_sync(args: argparse.Namespace) -> int:
    """``kbforge check-sync``: the read-only pre-sync checks; exit code 2 if a sync would fail."""
    report = _load(args).check_sync()
    print(report)
    return 0 if report.ready else 2


def cmd_syncs(args: argparse.Namespace) -> int:
    """``kbforge syncs``: recent syncs, or one in detail (``latest`` or a job id)."""
    kb = _load(args)
    if args.job:
        print(kb.sync_job(None if args.job == "latest" else args.job))
    else:
        print(kb.syncs(limit=args.limit))
    return 0


def cmd_stop_sync(args: argparse.Namespace) -> int:
    """``kbforge stop-sync``: stop the sync that's running."""
    print(_load(args).stop_sync())
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """``kbforge ask``: print the best-matching chunks for a question."""
    print(_load(args).ask(args.question, top_k=args.top_k, answer=args.answer, model=args.model))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """``kbforge list``: the knowledge bases kbforge built in a region."""
    from . import knowledge_bases

    print(knowledge_bases(region=args.region))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """``kbforge status``: print ids and the last sync."""
    print(_load(args).status())
    return 0


def cmd_destroy(args: argparse.Namespace) -> int:
    """``kbforge destroy``: show what would be deleted, confirm, then delete."""
    kb = _load(args)
    plan = make_plan(kb.spec, kb.aws, mode="destroy", force=args.force)
    print(plan)
    if not plan.ready or not plan.has_work:
        return 0 if plan.ready else 2
    if not args.yes and not _confirm(f"\nDelete '{kb.name}'? This can't be undone."):
        print("Nothing was deleted.")
        return 1
    result: ApplyResult = plan.apply()
    print(result)
    return 0 if result.ok else 1


def cmd_models(args: argparse.Namespace) -> int:
    """``kbforge models``: list the embedding models kbforge supports."""
    for m in EMBEDDING_MODELS.values():
        dims = "/".join(str(d) for d in m.dimensions)
        print(f"  {m.alias:<13} {m.label:<36} {dims:<14} {m.note}")
    return 0


def cmd_parsers(args: argparse.Namespace) -> int:
    """``kbforge parsers``: list the models this account can use to parse documents."""
    try:
        aws = Aws(region=args.region)
    except Exception as exc:
        print(known_parsing_models(args.region or "", f"Couldn't connect to AWS ({exc})."))
        return 0
    print(list_parsing_models(aws))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The argparse definition for every subcommand."""
    parser = argparse.ArgumentParser(prog="kbforge", description="Plan and build Amazon Bedrock Knowledge Bases.")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_config(name: str, help_text: str) -> argparse.ArgumentParser:
        """Add a subcommand that takes a YAML/JSON config path and an optional --region."""
        p = sub.add_parser(name, help=help_text)
        p.add_argument("config", help="path to the knowledge base config (.yaml, .yml or .json)")
        p.add_argument("--region", help="override the region in the config")
        return p

    p = sub.add_parser("init", help="write a starter kb.yaml (name it kb.json for JSON)")
    p.add_argument("path", nargs="?", default="kb.yaml", help="where to write it; the extension picks the format")
    p.set_defaults(func=cmd_init)

    with_config("plan", "show checks and what would be built").set_defaults(func=cmd_plan)

    p = with_config("apply", "build the knowledge base")
    p.add_argument("-y", "--yes", action="store_true", help="don't ask for confirmation")
    p.set_defaults(func=cmd_apply)

    p = with_config("sync", "re-ingest the data source (runs the pre-sync checks first)")
    p.add_argument("--no-wait", action="store_true", help="start the sync and return without waiting for it")
    p.add_argument("--skip-checks", action="store_true", help="don't run the pre-sync checks first")
    p.set_defaults(func=cmd_sync)
    with_config("check-sync", "check a sync would work, without starting one").set_defaults(func=cmd_check_sync)
    p = with_config("syncs", "list recent syncs, or show one in detail")
    p.add_argument("job", nargs="?", help="'latest' or a job id, to show that sync in detail with failed files")
    p.add_argument("-n", "--limit", type=int, default=10, help="how many recent syncs to list (default 10)")
    p.set_defaults(func=cmd_syncs)
    with_config("stop-sync", "stop the sync that's running").set_defaults(func=cmd_stop_sync)
    with_config("status", "show ids and the last sync").set_defaults(func=cmd_status)

    p = with_config("ask", "retrieve the best-matching chunks for a question")
    p.add_argument("question")
    p.add_argument("-k", "--top-k", type=int, default=5)
    p.add_argument("-a", "--answer", action="store_true", help="have a model write an answer that cites the chunks")
    p.add_argument("-m", "--model", help="model for --answer, e.g. nova-pro (implies --answer; default nova-lite)")
    p.set_defaults(func=cmd_ask)

    p = with_config("destroy", "delete the knowledge base and what kbforge created for it")
    p.add_argument("-y", "--yes", action="store_true", help="don't ask for confirmation")
    p.add_argument("--force", action="store_true", help="also delete resources without the kbforge tag")
    p.set_defaults(func=cmd_destroy)

    p = sub.add_parser("list", help="list the knowledge bases kbforge built in a region")
    p.add_argument("--region", help="region to look in (default: your AWS profile's region)")
    p.set_defaults(func=cmd_list)

    sub.add_parser("models", help="list supported embedding models").set_defaults(func=cmd_models)
    p = sub.add_parser("parsers", help="list the models your account can use to parse documents")
    p.add_argument("--region", help="region to look in (default: your AWS profile's region)")
    p.set_defaults(func=cmd_parsers)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the ``kbforge`` command. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (KBForgeError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (ClientError, BotoCoreError) as exc:
        print(f"error: {explain_error(exc)}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped. Anything already built is kept; run the same command again to carry on.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
