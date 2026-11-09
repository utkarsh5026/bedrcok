"""The ``kbforge`` command, for people who prefer a terminal (or CI) to a notebook.

kbforge init                    # write a starter kb.yaml
kbforge plan kb.yaml            # checks + what would be built
kbforge apply kb.yaml           # build it (asks before changing anything)
kbforge sync kb.yaml            # re-ingest after documents change
kbforge ask kb.yaml "question"  # try a query
kbforge status kb.yaml
kbforge destroy kb.yaml         # delete it (asks first)
kbforge models                  # list supported embedding models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .knowledge_base import KBForgeError, KnowledgeBase
from .models import EMBEDDING_MODELS
from .plan import ApplyResult, make_plan

STARTER_YAML = """\
# Describe your knowledge base, then run:  kbforge plan kb.yaml
name: my-knowledge-base
# region: us-east-1           # defaults to your AWS profile's region

source:
  s3: s3://your-bucket/your-folder/

embedding: titan-v2          # titan-v2 | cohere-en | cohere-multi | titan-v1
vector_store: s3_vectors     # s3_vectors | opensearch_serverless
chunking:
  strategy: fixed            # fixed | hierarchical | semantic | none
  max_tokens: 300
  overlap_percent: 20

# role_arn: arn:aws:iam::123456789012:role/AmazonBedrockExecutionRoleForKnowledgeBase_x
sync: true
"""


def _confirm(question: str) -> bool:
    """Ask a yes/no question on the terminal. Anything but 'y' means no."""
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _load(args: argparse.Namespace) -> KnowledgeBase:
    return KnowledgeBase.from_yaml(args.config, region=args.region)


def cmd_init(args: argparse.Namespace) -> int:
    """``kbforge init``: write a starter kb.yaml (never overwrites)."""
    path = Path(args.path)
    if path.exists():
        print(f"{path} already exists — not overwriting it.")
        return 1
    path.write_text(STARTER_YAML)
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
    """``kbforge sync``: re-ingest the data source."""
    print(_load(args).sync())
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """``kbforge ask``: print the best-matching chunks for a question."""
    print(_load(args).ask(args.question, top_k=args.top_k))
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


def build_parser() -> argparse.ArgumentParser:
    """The argparse definition for every subcommand."""
    parser = argparse.ArgumentParser(prog="kbforge", description="Plan and build Amazon Bedrock Knowledge Bases.")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_config(name: str, help_text: str) -> argparse.ArgumentParser:
        """Add a subcommand that takes a YAML config path and an optional --region."""
        p = sub.add_parser(name, help=help_text)
        p.add_argument("config", help="path to the knowledge base YAML file")
        p.add_argument("--region", help="override the region in the config")
        return p

    p = sub.add_parser("init", help="write a starter kb.yaml")
    p.add_argument("path", nargs="?", default="kb.yaml")
    p.set_defaults(func=cmd_init)

    with_config("plan", "show checks and what would be built").set_defaults(func=cmd_plan)

    p = with_config("apply", "build the knowledge base")
    p.add_argument("-y", "--yes", action="store_true", help="don't ask for confirmation")
    p.set_defaults(func=cmd_apply)

    with_config("sync", "re-ingest the data source").set_defaults(func=cmd_sync)
    with_config("status", "show ids and the last sync").set_defaults(func=cmd_status)

    p = with_config("ask", "retrieve the best-matching chunks for a question")
    p.add_argument("question")
    p.add_argument("-k", "--top-k", type=int, default=5)
    p.set_defaults(func=cmd_ask)

    p = with_config("destroy", "delete the knowledge base and what kbforge created for it")
    p.add_argument("-y", "--yes", action="store_true", help="don't ask for confirmation")
    p.add_argument("--force", action="store_true", help="also delete resources without the kbforge tag")
    p.set_defaults(func=cmd_destroy)

    sub.add_parser("models", help="list supported embedding models").set_defaults(func=cmd_models)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the ``kbforge`` command. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (KBForgeError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
