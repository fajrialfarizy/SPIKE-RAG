"""
Chat pipeline (Brief 5.5). Every question is independent: no conversation memory.

Order of operations:
1. input validation, 2. privacy request detector, 3. exact report-id route,
4. analyzer, 5. out-of-scope short-circuit, 6. retrieval (semantic or catalog),
7. not-found short-circuit, 8. alias context, 9. answer LLM, 10. citations with
one stricter retry, 11. output guardrails, 12. backend-built Sumber block.

Any unexpected failure becomes TECHNICAL_ERROR with a fixed user-facing text. Raw
exception text is never shown to the user and never logged; only the error type is.
Operational logging carries timestamp, status, mode, chunk count, latency and error
type — never question text, answers, document text or the API key.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from core import prompts
from core.analyzer import Analyzer
from core.citations import build_source_block, compose_answer, process_citations
from core.context_builder import ContextBuilder
from core.guardrails import OutputGuard, detect_privacy_request
from core.llm_client import LLMClient
from core.retriever import Retriever, RetrievedChunk
from enums import QueryMode
from ingestion.embedder import E5Embedder

logger = logging.getLogger("core_pipeline")


class AnswerStatus(str, Enum):
    ANSWERED = "ANSWERED"
    NOT_FOUND = "NOT_FOUND"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    PRIVACY_REFUSED = "PRIVACY_REFUSED"
    NO_SOURCE = "NO_SOURCE"
    TECHNICAL_ERROR = "TECHNICAL_ERROR"


@dataclass(frozen=True)
class DebugInfo:
    """Developer-only diagnostics. Never rendered to a student."""

    mode: Optional[str] = None
    filters: Dict[str, List[Any]] = field(default_factory=dict)
    route: str = ""
    retrieved: List[Dict[str, Any]] = field(default_factory=list)
    alias_map: Dict[str, str] = field(default_factory=dict)
    incidents: Dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    error_type: Optional[str] = None
    missing_report_ids: List[str] = field(default_factory=list)
    used_fallback: bool = False
    input_truncated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "filters": self.filters,
            "route": self.route,
            "retrieved": self.retrieved,
            "alias_map": self.alias_map,
            "incidents": self.incidents,
            "latency_ms": self.latency_ms,
            "error_type": self.error_type,
            "missing_report_ids": self.missing_report_ids,
            "used_fallback": self.used_fallback,
            "input_truncated": self.input_truncated,
        }


@dataclass(frozen=True)
class RagResult:
    status: AnswerStatus
    answer: str
    cited_report_ids: List[str] = field(default_factory=list)
    debug: DebugInfo = field(default_factory=DebugInfo)


def configure_file_logging(log_dir: Path = config.LOG_DIR) -> Path:
    """Attach a per-run chat log file that holds operational metadata only."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = log_dir / f"{config.CORE_LOG_FILE_PREFIX}_{stamp}.log"

    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(config.LOG_LEVEL)
    return path


class RagPipeline:
    """Wires retrieval, prompting, citations and guardrails for one answer."""

    def __init__(
        self,
        knowledge_base: Any,
        llm: LLMClient,
        embedder: Optional[Any] = None,
        retriever: Optional[Retriever] = None,
        analyzer: Optional[Analyzer] = None,
        context_builder: Optional[ContextBuilder] = None,
        output_guard: Optional[OutputGuard] = None,
    ):
        self.knowledge_base = knowledge_base
        self.llm = llm
        self.embedder = embedder or E5Embedder()
        self.retriever = retriever or Retriever(knowledge_base, embedder=self.embedder)
        self.analyzer = analyzer or Analyzer(llm)
        self.context_builder = context_builder or ContextBuilder()
        self.output_guard = output_guard or OutputGuard()

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _retrieved_debug(chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
        return [
            {
                "chunk_id": chunk.chunk_id,
                "report_id": chunk.report_id,
                "section_type": chunk.metadata.get("section_type"),
                "score": None if chunk.score is None else round(chunk.score, 4),
            }
            for chunk in chunks
        ]

    def _answer_with_llm(
        self, context_text: str, question: str, mode: QueryMode
    ) -> str:
        user_prompt = prompts.ANSWER_USER_TEMPLATE.format(
            context=context_text,
            hint=prompts.MODE_HINTS[mode.value],
            question=question,
        )
        return self.llm.complete(
            prompts.SYSTEM_PROMPT,
            user_prompt,
            temperature=config.LLM_TEMPERATURE_CHAT,
            max_tokens=config.LLM_MAX_OUTPUT_TOKENS,
            reasoning_effort=config.LLM_REASONING_EFFORT_ANSWER,
            json_mode=False,
        )

    def _answer_user_prompt(
        self, context_text: str, question: str, mode: QueryMode
    ) -> str:
        return prompts.ANSWER_USER_TEMPLATE.format(
            context=context_text,
            hint=prompts.MODE_HINTS[mode.value],
            question=question,
        )

    # -- main entry point --------------------------------------------------
    def run(self, question: str) -> RagResult:
        started = time.perf_counter()
        state: Dict[str, Any] = {
            "mode": None,
            "filters": {},
            "route": "",
            "retrieved": [],
            "alias_map": {},
            "incidents": {"unknown_alias": 0, "fabricated_report_id": 0,
                          "pii_masked": 0, "prompt_leak": 0},
            "missing_report_ids": [],
            "used_fallback": False,
            "input_truncated": False,
        }

        def result(
            status: AnswerStatus,
            answer: str,
            cited: Optional[List[str]] = None,
            error_type: Optional[str] = None,
        ) -> RagResult:
            latency_ms = round((time.perf_counter() - started) * 1000, 1)
            debug = DebugInfo(
                mode=state["mode"],
                filters=state["filters"],
                route=state["route"],
                retrieved=state["retrieved"],
                alias_map=state["alias_map"],
                incidents=state["incidents"],
                latency_ms=latency_ms,
                error_type=error_type,
                missing_report_ids=state["missing_report_ids"],
                used_fallback=state["used_fallback"],
                input_truncated=state["input_truncated"],
            )
            logger.info(
                "chat_request status=%s mode=%s route=%s chunks=%d latency_ms=%.1f "
                "error_type=%s",
                status.value,
                debug.mode,
                debug.route or "-",
                len(debug.retrieved),
                latency_ms,
                error_type or "-",
            )
            return RagResult(
                status=status,
                answer=answer,
                cited_report_ids=cited or [],
                debug=debug,
            )

        try:
            text = (question or "").strip()
            if not text:
                logger.info("chat_input_rejected reason=empty")
                return result(AnswerStatus.NOT_FOUND, prompts.NOT_FOUND_RESPONSE)

            if len(text) > config.CHAT_MAX_INPUT_CHARS:
                text = text[: config.CHAT_MAX_INPUT_CHARS]
                state["input_truncated"] = True

            # 1. Privacy request: refused before any model call.
            privacy_category = detect_privacy_request(text)
            if privacy_category:
                logger.info("chat_privacy_refused category=%s", privacy_category)
                state["route"] = "privacy"
                state["incidents"]["privacy_request"] = 1
                return result(
                    AnswerStatus.PRIVACY_REFUSED, prompts.PRIVACY_REFUSAL_RESPONSE
                )

            # 2. Exact report-id route.
            report_ids = list(
                dict.fromkeys(config.PATTERN_REPORT_ID.findall(text))
            )[: config.MAX_REPORT_IDS_BYPASS]

            if report_ids:
                state["route"] = "report_id"
                state["mode"] = QueryMode.SPECIFIC.value
                chunks = self.retriever.retrieve_by_report_ids(report_ids)
                found = {chunk.report_id for chunk in chunks}
                state["missing_report_ids"] = [
                    report_id for report_id in report_ids if report_id not in found
                ]
                if not chunks:
                    return result(AnswerStatus.NOT_FOUND, prompts.NOT_FOUND_RESPONSE)
                return self._compose_answer(
                    result, state, chunks, text, QueryMode.SPECIFIC, catalog=None
                )

            # 3. Analyzer.
            analysis = self.analyzer.analyze(text)
            state["mode"] = analysis.mode.value
            state["filters"] = analysis.filters
            state["used_fallback"] = analysis.used_fallback

            if not analysis.in_scope:
                state["route"] = "out_of_scope"
                return result(AnswerStatus.OUT_OF_SCOPE, prompts.OUT_OF_SCOPE_RESPONSE)

            # 4. Retrieval.
            if analysis.mode is QueryMode.CATALOG:
                state["route"] = "catalog"
                reports = self.retriever.list_catalog(analysis.filters)
                if not reports:
                    return result(AnswerStatus.NOT_FOUND, prompts.NOT_FOUND_RESPONSE)
                return self._compose_answer(
                    result, state, [], text, analysis.mode, catalog=reports
                )

            state["route"] = "semantic"
            chunks = self.retriever.retrieve(
                analysis.search_query, analysis.mode, analysis.filters
            )
            if not chunks:
                return result(AnswerStatus.NOT_FOUND, prompts.NOT_FOUND_RESPONSE)
            return self._compose_answer(
                result, state, chunks, text, analysis.mode, catalog=None
            )

        except Exception as exc:  # provider, storage or programming failure
            error_type = type(exc).__name__
            logger.warning("chat_failed error_type=%s", error_type)
            return result(
                AnswerStatus.TECHNICAL_ERROR,
                prompts.TECHNICAL_ERROR_RESPONSE,
                error_type=error_type,
            )

    # -- answer construction ----------------------------------------------
    def _compose_answer(
        self,
        result,
        state: Dict[str, Any],
        chunks: List[RetrievedChunk],
        question: str,
        mode: QueryMode,
        catalog: Optional[List[Dict[str, Any]]],
    ) -> RagResult:
        if catalog is not None:
            bundle = self.context_builder.build_catalog(catalog)
        else:
            bundle = self.context_builder.build_chunks(chunks)

        state["alias_map"] = bundle.alias_map
        state["retrieved"] = self._retrieved_debug(chunks)
        retrieved_ids = list(bundle.report_metadata)

        answer_text = self._answer_with_llm(bundle.text, question, mode)

        def resolve(raw_answer: str):
            return process_citations(raw_answer, bundle.alias_map, retrieved_ids)

        citations = resolve(answer_text)
        if not citations.cited_report_ids:
            # One stricter attempt before giving up (Brief 5.6).
            retry_prompt = (
                f"{self._answer_user_prompt(bundle.text, question, mode)}"
                f"\n\n{prompts.CITATION_RETRY_INSTRUCTION}"
            )
            answer_text = self.llm.complete(
                prompts.SYSTEM_PROMPT,
                retry_prompt,
                temperature=config.LLM_TEMPERATURE_CHAT,
                max_tokens=config.LLM_MAX_OUTPUT_TOKENS,
                reasoning_effort=config.LLM_REASONING_EFFORT_ANSWER,
                json_mode=False,
            )
            citations = resolve(answer_text)

        for key, value in citations.incidents.items():
            state["incidents"][key] = state["incidents"].get(key, 0) + value

        if not citations.cited_report_ids:
            return result(AnswerStatus.NO_SOURCE, prompts.NO_SOURCE_RESPONSE)

        guarded = self.output_guard.apply(citations.text)
        for key, value in guarded.incidents.items():
            state["incidents"][key] = state["incidents"].get(key, 0) + value

        if guarded.blocked:
            # A blocked answer carries no source block, so it claims no sources.
            return result(AnswerStatus.PRIVACY_REFUSED, guarded.text, cited=[])

        source_block = build_source_block(
            citations.cited_report_ids, bundle.report_metadata
        )
        final_answer = compose_answer(guarded.text, source_block)

        return result(
            AnswerStatus.ANSWERED, final_answer, cited=citations.cited_report_ids
        )
