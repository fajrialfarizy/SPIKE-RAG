"""
Command-line entry point: python -m core.cli "pertanyaan" [--debug]

Prints the answer; with --debug it also prints the backend debug object (status,
mode, validated filters, retrieved chunks with scores, alias map and incident
counts). The API key is read from the environment and never printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import config
from core.knowledge_base import KnowledgeBase, KnowledgeBaseError
from core.llm_client import GroqLLMClient, LLMConfigurationError, get_api_key
from core.pipeline import AnswerStatus, RagPipeline, configure_file_logging
from ingestion.embedder import E5Embedder


def build_pipeline(
    chroma_dir: Path,
    manifest_path: Optional[Path] = None,
    log_dir: Optional[Path] = None,
) -> RagPipeline:
    if not get_api_key():
        raise LLMConfigurationError(
            f"{config.LLM_API_KEY_ENV_VAR} is not set. Export it before running the "
            "chat pipeline; the value is never logged."
        )

    knowledge_base = KnowledgeBase(
        persist_dir=chroma_dir,
        manifest_path=manifest_path,
    )
    if log_dir is not None:
        configure_file_logging(log_dir)

    return RagPipeline(
        knowledge_base=knowledge_base,
        llm=GroqLLMClient(),
        embedder=E5Embedder(),
    )


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask the Lab Prestasi knowledge base a question."
    )
    parser.add_argument("question", help="Pertanyaan mahasiswa (bahasa Indonesia).")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print the backend debug object (never shown to students).",
    )
    parser.add_argument(
        "--chroma-dir", default=str(config.CHROMA_PERSIST_DIR), help="Chroma persist dir."
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest.json path to validate the artifact against.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Write an operational chat log file into this directory.",
    )
    args = parser.parse_args(argv)

    try:
        pipeline = build_pipeline(
            chroma_dir=Path(args.chroma_dir),
            manifest_path=Path(args.manifest) if args.manifest else None,
            log_dir=Path(args.log_dir) if args.log_dir else None,
        )
    except (LLMConfigurationError, KnowledgeBaseError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    result = pipeline.run(args.question)
    print(result.answer)

    if args.debug:
        print("\n--- DEBUG (backend only) ---")
        print(f"status: {result.status.value}")
        print(f"cited_report_ids: {result.cited_report_ids}")
        print(json.dumps(result.debug.to_dict(), indent=2, ensure_ascii=False))

    return 0 if result.status is AnswerStatus.ANSWERED else 1


if __name__ == "__main__":
    raise SystemExit(main())
