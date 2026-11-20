"""
Real end-to-end smoke run against Groq (dummy corpus only).

This script talks to the real Groq API with the configured model, so it needs a
key and a knowledge base built from the dummy corpus:

    export GROQ_API_KEY=...            # PowerShell: $env:GROQ_API_KEY="..."
    python scripts/make_dummy_corpus.py
    python scripts/run_ingestion.py --source local
    python scripts/smoke_groq.py --chroma-dir chroma_db

It prints status, mode, validated filters, citation count and the answer for each
question. It never fabricates a result: without a key it stops with instructions.
Never point this at real evaluation documents: ZDR for Groq is not confirmed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from core.knowledge_base import KnowledgeBase, KnowledgeBaseError
from core.llm_client import GroqLLMClient
from core.pipeline import RagPipeline, configure_file_logging
from ingestion.embedder import E5Embedder

# PRD section 10 questions, adapted to the dummy corpus.
DUMMY_QUESTIONS: List[str] = [
    "Apa saja pengalaman tim Lab Prestasi ketika mengikuti kompetisi business plan?",
    "Carikan pengalaman kompetisi yang memiliki tema limbah atau energi terbarukan.",
    "Apakah ada pengalaman tim dalam business case bertema fintech?",
    "Apa saja kendala yang sering dialami tim ketika mengikuti kompetisi business plan?",
    "Apa tantangan terbesar yang pernah dihadapi tim saat menyusun proposal kompetisi?",
    "Apa pembelajaran dari tim yang pernah mengikuti kompetisi tingkat internasional?",
    "Apa rekomendasi pengembangan yang pernah diberikan kepada tim setelah kompetisi?",
    "Bandingkan kendala yang dialami beberapa tim dalam kompetisi business plan.",
    "Apa perbedaan pembelajaran dari tim yang memperoleh juara dan tim yang belum juara?",
    "Apa saja saran pengembangan yang tercantum pada RPT0002?",
]


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the core RAG pipeline on Groq.")
    parser.add_argument("--chroma-dir", default=str(config.CHROMA_PERSIST_DIR))
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--limit", type=int, default=len(DUMMY_QUESTIONS), help="How many questions to run."
    )
    parser.add_argument(
        "--log-dir",
        default=str(config.LOG_DIR),
        help="Operational log directory (metadata only).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)

    if not os.environ.get(config.LLM_API_KEY_ENV_VAR):
        print(
            f"{config.LLM_API_KEY_ENV_VAR} is not set.\n"
            "Set it in your shell (never in code or git) and run again:\n"
            '  PowerShell: $env:GROQ_API_KEY="<your key>"\n'
            "  Bash:       export GROQ_API_KEY='<your key>'\n"
            "Then: python scripts/smoke_groq.py --chroma-dir chroma_db",
            file=sys.stderr,
        )
        return 2

    try:
        knowledge_base = KnowledgeBase(
            persist_dir=Path(args.chroma_dir),
            manifest_path=Path(args.manifest) if args.manifest else None,
        )
    except KnowledgeBaseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    configure_file_logging(Path(args.log_dir))

    pipeline = RagPipeline(
        knowledge_base=knowledge_base,
        llm=GroqLLMClient(),
        embedder=E5Embedder(),
    )

    for index, question in enumerate(DUMMY_QUESTIONS[: args.limit], start=1):
        result = pipeline.run(question)
        filters = json.dumps(result.debug.filters, ensure_ascii=False)
        print(f"\n=== [{index}] {question}")
        print(
            f"status={result.status.value} mode={result.debug.mode} "
            f"route={result.debug.route} filters={filters} "
            f"chunks={len(result.debug.retrieved)} "
            f"cited={result.cited_report_ids} latency_ms={result.debug.latency_ms}"
        )
        print(result.answer)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
