"""
Offline reconstruction of the answer context (fix pack J2a).

This script runs ONLY retrieval and context building for one question: the real
embedding model, the local Chroma collection, no LLM, no API key, no quota. It
prints the exact blocks the answer model would receive, plus the alias -> report
map the backend keeps private.

    python scripts/show_context.py --question "Bandingkan pembelajaran tim juara
        dan tim yang belum juara." --compare-by achievement_group

Dummy corpus only. Never point it at real evaluation documents.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from core.context_builder import ContextBuilder
from core.knowledge_base import KnowledgeBase, KnowledgeBaseError
from core.retriever import Retriever
from core.tokens import E5TokenCounter
from enums import QueryMode
from ingestion.embedder import E5Embedder


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the exact answer context for one question (no LLM call)."
    )
    parser.add_argument("--question", required=True)
    parser.add_argument("--chroma-dir", default=str(config.CHROMA_PERSIST_DIR))
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--mode",
        choices=[QueryMode.SPECIFIC.value, QueryMode.CROSS_REPORT.value],
        default=QueryMode.CROSS_REPORT.value,
    )
    parser.add_argument(
        "--compare-by",
        choices=config.COMPARE_BY_OPTIONS,
        default=None,
        help="Grouped comparison; matches what the analyzer would set.",
    )
    parser.add_argument(
        "--competition-type",
        action="append",
        default=[],
        help="Filter value; repeat for several values.",
    )
    parser.add_argument(
        "--achievement-level",
        action="append",
        default=[],
        help="Filter value; repeat for several values.",
    )
    parser.add_argument(
        "--year",
        action="append",
        type=int,
        default=[],
        help="Competition year filter; repeat for several values.",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="Override the context budget (default: the mode's configured budget).",
    )
    parser.add_argument(
        "--no-context-text",
        action="store_true",
        help="Print the bookkeeping only, not the context blocks.",
    )
    return parser.parse_args(argv)


def build_filters(args: argparse.Namespace) -> Dict[str, List]:
    filters: Dict[str, List] = {}
    if args.competition_type:
        filters["competition_type"] = sorted(set(args.competition_type))
    if args.achievement_level:
        filters["achievement_level"] = sorted(set(args.achievement_level))
    if args.year:
        filters["competition_year"] = sorted(set(args.year))
    return filters


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)

    try:
        knowledge_base = KnowledgeBase(
            persist_dir=Path(args.chroma_dir),
            manifest_path=Path(args.manifest) if args.manifest else None,
        )
    except KnowledgeBaseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    embedder = E5Embedder()
    retriever = Retriever(knowledge_base, embedder=embedder)
    filters = build_filters(args)
    mode = QueryMode(args.mode)

    groups: List[dict] = []
    if args.compare_by:
        chunks, groups = retriever.retrieve_grouped(
            args.question, mode, filters, args.compare_by
        )
    else:
        chunks = retriever.retrieve(args.question, mode, filters)

    if not chunks:
        print("retrieved: (none)")
        return 1

    if args.max_context_tokens is not None:
        budget = args.max_context_tokens
    elif args.compare_by:
        budget = config.MAX_CONTEXT_TOKENS_COMPARE
    else:
        budget = config.MAX_CONTEXT_TOKENS

    builder = ContextBuilder()
    counter = E5TokenCounter()
    bundle, dropped = builder.build_chunks_within_budget(chunks, counter, budget)

    print(f"question={args.question}")
    print(f"mode={mode.value} compare_by={args.compare_by} filters={json.dumps(filters)}")
    print(f"groups={json.dumps(groups)}")
    print(f"embedding_latency_ms={retriever.last_embedding_latency_ms}")
    print(f"context_budget_tokens={budget}")
    print(f"context_tokens={counter.count_tokens(bundle.text)}")
    print(f"context_chars={len(bundle.text)}")

    print("retrieved:")
    for chunk in chunks:
        score = "n/a" if chunk.score is None else f"{chunk.score:.4f}"
        kept = "kept" if chunk.chunk_id not in dropped else "dropped"
        label = getattr(chunk, "group_label", None) or "-"
        print(
            f"  - {chunk.chunk_id} report={chunk.report_id} "
            f"group={label} section={chunk.metadata.get('section_type')} "
            f"score={score} {kept}"
        )

    print("alias_map:")
    for alias, report_id in bundle.alias_map.items():
        label = "-"
        for chunk in chunks:
            if chunk.report_id == report_id and getattr(chunk, "group_label", None):
                label = chunk.group_label
                break
        print(f"  - {alias} -> {report_id} group={label}")

    if not args.no_context_text:
        print("--- context sent to the answer model ---")
        print(bundle.text)
        print("--- end context ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
