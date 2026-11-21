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
from core import compare_answer, prompts
from core.analyzer import Analyzer
from core.catalog import render_catalog_answer
from core.citations import (
    build_source_block,
    citation_style_metrics,
    compose_answer,
    find_uncited_mentions,
    process_citations,
)
from core.context_builder import ContextBuilder
from core.coverage import (
    build_coverage_line,
    build_cross_report_coverage_line,
    comparison_totals,
    count_reports,
)
from core.deadline import Deadline, DeadlineExceeded
from core.guardrails import (
    OutputGuard,
    detect_instruction_override,
    detect_privacy_request,
)
from core.llm_client import LLMClient, LLMRateLimitError
from core.negation import negation_sentences, point_negation_counts
from core.tokens import EstimatorTokenCounter
from core.retriever import Retriever, RetrievedChunk
from core.sanitizer import sanitize_answer
from core.truncation import trim_to_last_complete_unit
from enums import QueryMode
from ingestion.embedder import E5Embedder

logger = logging.getLogger("core_pipeline")


class AnswerStatus(str, Enum):
    ANSWERED = "ANSWERED"
    NOT_FOUND = "NOT_FOUND"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    PRIVACY_REFUSED = "PRIVACY_REFUSED"
    SECURITY_REFUSED = "SECURITY_REFUSED"
    NO_SOURCE = "NO_SOURCE"
    RATE_LIMITED = "RATE_LIMITED"
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
    calls: List[Dict[str, Any]] = field(default_factory=list)
    token_usage: Dict[str, int] = field(default_factory=dict)
    context_dropped_chunk_ids: List[str] = field(default_factory=list)
    analyzer: Dict[str, Any] = field(default_factory=dict)
    compare_by: Optional[str] = None
    groups: List[Dict[str, Any]] = field(default_factory=list)
    embedding_latency_ms: float = 0.0
    deadline_budget_seconds: float = 0.0
    # Raw model output per answer attempt. Developer-only evidence for the smoke
    # run: never rendered, never written to a log line.
    raw_answers: List[Dict[str, Any]] = field(default_factory=list)
    style: Dict[str, int] = field(default_factory=dict)
    # Backend-written coverage line for comparison answers (fix pack H3b).
    coverage: str = ""
    # Exact context text the answer model received (developer evidence only).
    context_text: str = ""
    # Non-blocking negation metric (fix pack K2). Samples stay developer-only.
    negation: Dict[str, Any] = field(default_factory=dict)

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
            "calls": self.calls,
            "token_usage": self.token_usage,
            "context_dropped_chunk_ids": self.context_dropped_chunk_ids,
            "analyzer": self.analyzer,
            "compare_by": self.compare_by,
            "groups": self.groups,
            "embedding_latency_ms": self.embedding_latency_ms,
            "deadline_budget_seconds": self.deadline_budget_seconds,
            "raw_answers": self.raw_answers,
            "style": self.style,
            "coverage": self.coverage,
            "context_text": self.context_text,
            "negation": self.negation,
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
        token_counter: Optional[Any] = None,
        max_context_tokens: int = config.MAX_CONTEXT_TOKENS,
        answer_reasoning_effort: str = config.ANSWER_REASONING_EFFORT,
        deadline_budget_seconds: float = config.PIPELINE_TOTAL_DEADLINE_SECONDS,
        clock: Any = time.monotonic,
    ):
        if answer_reasoning_effort not in config.ANSWER_REASONING_EFFORT_VALUES:
            raise ValueError(
                f"answer reasoning effort '{answer_reasoning_effort}' is not supported"
            )
        self.knowledge_base = knowledge_base
        self.llm = llm
        self.embedder = embedder or E5Embedder()
        self.deadline_budget_seconds = float(deadline_budget_seconds)
        self._clock = clock
        self.retriever = retriever or Retriever(knowledge_base, embedder=self.embedder)
        self.analyzer = analyzer or Analyzer(llm)
        self.context_builder = context_builder or ContextBuilder()
        self.output_guard = output_guard or OutputGuard()
        self.token_counter = token_counter or EstimatorTokenCounter()
        self.max_context_tokens = int(max_context_tokens)
        self.answer_reasoning_effort = answer_reasoning_effort

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _sum_usage(calls: List[Dict[str, Any]]) -> Dict[str, int]:
        totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }
        cached = 0
        cached_reported = False
        for call in calls:
            usage = call.get("usage") or {}
            for key in totals:
                totals[key] += int(usage.get(key, 0) or 0)
            # Fix pack H5b: keep "reported zero" apart from "not reported".
            if usage.get("cached_tokens") is not None:
                cached_reported = True
                cached += int(usage.get("cached_tokens") or 0)
        if cached_reported:
            totals["cached_tokens"] = cached
        return totals

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
        self,
        context_text: str,
        question: str,
        mode: QueryMode,
        deadline: Optional[Deadline] = None,
        hint_extra: str = "",
    ):
        user_prompt = self._answer_user_prompt(
            context_text, question, mode, hint_extra=hint_extra
        )
        return self.llm.complete(
            prompts.SYSTEM_PROMPT,
            user_prompt,
            temperature=config.LLM_TEMPERATURE_CHAT,
            max_tokens=config.LLM_MAX_OUTPUT_TOKENS,
            reasoning_effort=self.answer_reasoning_effort,
            json_mode=False,
            deadline=deadline,
        )

    def _answer_user_prompt(
        self,
        context_text: str,
        question: str,
        mode: QueryMode,
        hint_extra: str = "",
    ) -> str:
        hint = prompts.MODE_HINTS[mode.value]
        if hint_extra:
            hint = f"{hint}\n\n{hint_extra}"
        return prompts.ANSWER_USER_TEMPLATE.format(
            context=context_text,
            hint=hint,
            question=question,
        )

    # -- main entry point --------------------------------------------------
    def run(self, question: str) -> RagResult:
        started = time.perf_counter()
        deadline = Deadline(self.deadline_budget_seconds, clock=self._clock)
        state: Dict[str, Any] = {
            "mode": None,
            "filters": {},
            "route": "",
            "retrieved": [],
            "alias_map": {},
            "incidents": {"unknown_alias": 0, "fabricated_report_id": 0,
                          "pii_masked": 0, "prompt_leak": 0,
                          "security_request": 0},
            "missing_report_ids": [],
            "used_fallback": False,
            "input_truncated": False,
            "calls": [],
            "context_dropped_chunk_ids": [],
            "analyzer": {},
            "compare_by": None,
            "groups": [],
            "embedding_latency_ms": 0.0,
            "raw_answers": [],
            "style": {},
            "coverage": "",
            "context_text": "",
            "negation": {},
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
                calls=list(state["calls"]),
                token_usage=self._sum_usage(state["calls"]),
                context_dropped_chunk_ids=list(state["context_dropped_chunk_ids"]),
                analyzer=state["analyzer"],
                compare_by=state["compare_by"],
                groups=list(state["groups"]),
                embedding_latency_ms=state["embedding_latency_ms"],
                deadline_budget_seconds=self.deadline_budget_seconds,
                raw_answers=list(state["raw_answers"]),
                style=dict(state["style"]),
                coverage=state["coverage"],
                context_text=state["context_text"],
                negation=dict(state["negation"]),
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

            # 2. Instruction attack: refused before any model call.
            security_category = detect_instruction_override(text)
            if security_category:
                logger.info("chat_security_refused category=%s", security_category)
                state["route"] = "security"
                state["incidents"]["security_request"] = 1
                return result(
                    AnswerStatus.SECURITY_REFUSED, prompts.SECURITY_REFUSAL_RESPONSE
                )

            # 3. Exact report-id route.
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
                    result, state, chunks, text, QueryMode.SPECIFIC, catalog=None,
                    deadline=deadline, id_route=True,
                )

            # 4. Analyzer.
            deadline.require_remaining()
            analysis = self.analyzer.analyze(text, deadline=deadline)
            state["mode"] = analysis.mode.value
            state["filters"] = analysis.filters
            state["used_fallback"] = analysis.used_fallback
            state["analyzer"] = {
                "raw_output": analysis.raw_output,
                "payload": analysis.payload,
                "filters_before_validation": analysis.filters_before_validation,
                "filters": analysis.filters,
                "dropped_reasons": analysis.dropped_reasons,
                "deterministic_filters": analysis.deterministic_filters,
                "used_fallback": analysis.used_fallback,
                "error_type": analysis.error_type,
                "latency_ms": analysis.latency_ms,
                "usage": analysis.usage,
            }
            state["calls"].append(
                {
                    "name": "analyzer",
                    "latency_ms": analysis.latency_ms,
                    "usage": analysis.usage,
                    "attempts": 1,
                    "used_fallback": analysis.used_fallback,
                    "error_type": analysis.error_type,
                }
            )

            if not analysis.in_scope:
                state["route"] = "out_of_scope"
                return result(AnswerStatus.OUT_OF_SCOPE, prompts.OUT_OF_SCOPE_RESPONSE)

            # 5. Retrieval.
            if analysis.mode is QueryMode.CATALOG:
                state["route"] = "catalog"
                reports = self.retriever.list_catalog(analysis.filters)
                if not reports:
                    return result(AnswerStatus.NOT_FOUND, prompts.NOT_FOUND_RESPONSE)
                return self._compose_answer(
                    result, state, [], text, analysis.mode, catalog=reports,
                    deadline=deadline,
                )

            state["route"] = "semantic_grouped" if analysis.compare_by else "semantic"
            hint_extra = ""
            if analysis.compare_by:
                state["compare_by"] = analysis.compare_by
                chunks, groups = self.retriever.retrieve_grouped(
                    analysis.search_query,
                    analysis.mode,
                    analysis.filters,
                    analysis.compare_by,
                )
                state["groups"] = groups
                hint_extra = prompts.build_group_hint(groups, analysis.compare_by)
                logger.info(
                    "chat_compare_by grouping=%s groups=%d chunks=%d",
                    analysis.compare_by,
                    len(groups),
                    len(chunks),
                )
            else:
                chunks = self.retriever.retrieve(
                    analysis.search_query, analysis.mode, analysis.filters
                )

            state["embedding_latency_ms"] = getattr(
                self.retriever, "last_embedding_latency_ms", 0.0
            )

            if not chunks:
                return result(AnswerStatus.NOT_FOUND, prompts.NOT_FOUND_RESPONSE)
            return self._compose_answer(
                result, state, chunks, text, analysis.mode, catalog=None,
                deadline=deadline, hint_extra=hint_extra,
                max_context_tokens=(
                    config.MAX_CONTEXT_TOKENS_COMPARE
                    if analysis.compare_by
                    else None
                ),
            )

        except LLMRateLimitError as exc:
            is_daily = bool(getattr(exc, "is_daily_limit", False))
            if is_daily:
                state["incidents"]["daily_limit"] = 1
            logger.warning(
                "chat_failed error_type=%s daily_limit=%s",
                type(exc).__name__,
                is_daily,
            )
            return result(
                AnswerStatus.RATE_LIMITED,
                prompts.DAILY_QUOTA_RESPONSE
                if is_daily
                else prompts.RATE_LIMITED_RESPONSE,
                error_type=type(exc).__name__,
            )
        except DeadlineExceeded as exc:
            logger.warning("chat_failed error_type=DeadlineExceeded")
            return result(
                AnswerStatus.TECHNICAL_ERROR,
                prompts.TECHNICAL_ERROR_RESPONSE,
                error_type=type(exc).__name__,
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
    @staticmethod
    def _declared_not_found(result, state: Dict[str, Any], id_route: bool) -> RagResult:
        """Map a model-declared refusal to its decisive status (fix pack H2)."""
        if id_route:
            # Documents WERE retrieved, so the model contradicted the context.
            # The user gets NO_SOURCE; the incident counts only in logs/debug.
            state["incidents"]["model_declared_not_found_with_context"] = 1
            logger.info("chat_model_declared_not_found route=report_id")
            return result(AnswerStatus.NO_SOURCE, prompts.NO_SOURCE_RESPONSE)
        state["incidents"]["model_declared_not_found"] = 1
        logger.info("chat_model_declared_not_found route=semantic")
        return result(AnswerStatus.NOT_FOUND, prompts.NOT_FOUND_RESPONSE)

    def _coverage_line(
        self,
        state: Dict[str, Any],
        bundle: Any,
        chunks: List[RetrievedChunk],
        cited_report_ids: Optional[List[str]] = None,
    ) -> str:
        """
        Backend coverage sentence for cross_report answers (fix pack H3b).

        Fix pack J3b: only reports actually cited in the final answer are
        counted, so the sentence and the Sumber block always agree.
        """
        if state.get("mode") != QueryMode.CROSS_REPORT.value:
            return ""

        filters = state.get("filters") or {}
        cited = list(cited_report_ids or [])
        compare_by = state.get("compare_by")
        if compare_by:
            group_by_report: Dict[str, str] = {}
            for chunk in chunks:
                label = getattr(chunk, "group_label", None)
                if label is not None:
                    group_by_report.setdefault(chunk.report_id, label)
            shown_counts: Dict[str, int] = {}
            for report_id in cited:
                if report_id not in bundle.report_metadata:
                    continue
                label = group_by_report.get(report_id)
                if label is not None:
                    shown_counts[label] = shown_counts.get(label, 0) + 1
            totals = comparison_totals(self.knowledge_base, filters, compare_by)
            return build_coverage_line(totals, shown_counts)

        shown = len(
            [report_id for report_id in cited if report_id in bundle.report_metadata]
        )
        return build_cross_report_coverage_line(
            shown, count_reports(self.knowledge_base, filters)
        )

    @staticmethod
    def _raw_answer(attempt: int, completion: Any) -> Dict[str, Any]:
        """Debug-only evidence of one answer attempt (never logged, never shown)."""
        usage = completion.usage
        return {
            "attempt": attempt,
            "content": completion.content,
            "finish_reason": completion.finish_reason,
            "completion_tokens": usage.completion_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "content_chars": len(completion.content or ""),
        }

    def _structured_technical_error(
        self, result, state: Dict[str, Any], error_type: str
    ) -> RagResult:
        """Invalid structured output is technical, never a partial answer."""
        state["incidents"][error_type] = 1
        logger.info("chat_structured_output_invalid error_type=%s", error_type)
        return result(
            AnswerStatus.TECHNICAL_ERROR,
            prompts.TECHNICAL_ERROR_RESPONSE,
            error_type=error_type,
        )

    def _compose_compare_answer(
        self,
        result,
        state: Dict[str, Any],
        chunks: List[RetrievedChunk],
        bundle: Any,
        question: str,
        deadline: Optional[Deadline] = None,
    ) -> RagResult:
        """
        Comparison answer as validated JSON rendered by the backend (fix pack J1).

        The model writes no heading, no citation and no count. Invalid JSON or a
        truncated completion is a technical error with no retry (quota), and a
        payload with nothing traceable to a document is NO_SOURCE.
        """
        compare_by = state.get("compare_by")
        labels = [
            group["label"]
            for group in config.COMPARISON_GROUPS.get(compare_by, ())
        ]
        if not labels:
            return self._structured_technical_error(
                result, state, "invalid_structured_output"
            )

        alias_by_report = {
            report_id: alias for alias, report_id in bundle.alias_map.items()
        }
        group_aliases: Dict[str, List[str]] = {label: [] for label in labels}
        report_group: Dict[str, str] = {}
        for chunk in chunks:
            label = getattr(chunk, "group_label", None)
            if label is None or label not in group_aliases:
                continue
            report_group.setdefault(chunk.report_id, label)
            alias = alias_by_report.get(chunk.report_id)
            if alias and alias not in group_aliases[label]:
                group_aliases[label].append(alias)

        user_prompt = prompts.COMPARE_USER_TEMPLATE.format(
            context=bundle.text,
            hint=prompts.build_group_hint(state.get("groups") or [], compare_by),
            group_aliases=prompts.build_compare_group_aliases(group_aliases),
            question=question,
        )
        completion = self.llm.complete(
            prompts.SYSTEM_PROMPT_COMPARE,
            user_prompt,
            temperature=config.LLM_TEMPERATURE_CHAT,
            max_tokens=config.LLM_MAX_OUTPUT_TOKENS_COMPARE,
            reasoning_effort=self.answer_reasoning_effort,
            json_mode=True,
            response_format=compare_answer.response_format(labels),
            deadline=deadline,
        )
        state["calls"].append(completion.to_metrics("answer_compare"))
        state["raw_answers"].append(self._raw_answer(1, completion))

        if completion.finish_reason == "length":
            state["incidents"]["answer_truncated"] = 1
            return self._structured_technical_error(
                result, state, "invalid_structured_output"
            )

        try:
            payload = compare_answer.parse_payload(completion.content)
            validation = compare_answer.validate_payload(
                payload,
                labels=labels,
                alias_map=bundle.alias_map,
                report_group=report_group,
                retrieved=list(bundle.report_metadata),
            )
        except compare_answer.CompareFormatError:
            return self._structured_technical_error(
                result, state, "invalid_structured_output"
            )

        for key, value in validation.incidents.items():
            state["incidents"][key] = state["incidents"].get(key, 0) + value

        if validation.is_empty:
            return result(AnswerStatus.NO_SOURCE, prompts.NO_SOURCE_RESPONSE)

        answer_text = compare_answer.render_answer(validation)
        state["style"] = citation_style_metrics(answer_text)
        state["negation"] = {
            **negation_sentences(answer_text),
            "per_point": point_negation_counts(
                [point.text for section in validation.sections for point in section.points]
                + [point.text for point in validation.differences]
            ),
        }

        coverage_line = self._coverage_line(
            state, bundle, chunks, validation.cited_report_ids
        )
        state["coverage"] = coverage_line
        if coverage_line:
            answer_text = f"{answer_text}{config.COVERAGE_BLOCK_SEPARATOR}{coverage_line}"

        source_block = build_source_block(
            validation.cited_report_ids, bundle.report_metadata
        )
        final_answer = compose_answer(answer_text, source_block)

        return result(
            AnswerStatus.ANSWERED,
            final_answer,
            cited=validation.cited_report_ids,
        )

    def _compose_answer(
        self,
        result,
        state: Dict[str, Any],
        chunks: List[RetrievedChunk],
        question: str,
        mode: QueryMode,
        catalog: Optional[List[Dict[str, Any]]],
        deadline: Optional[Deadline] = None,
        hint_extra: str = "",
        id_route: bool = False,
        max_context_tokens: Optional[int] = None,
    ) -> RagResult:
        if catalog is not None:
            # Fix pack G5: the catalog answer is backend-rendered. The analyzer
            # already ran for scope and filters; no answer-LLM call is made and
            # the list itself is the source, so no Sumber block is attached.
            state["retrieved"] = []
            state["alias_map"] = {}
            answer = render_catalog_answer(catalog)
            state["style"] = citation_style_metrics(answer)
            return result(AnswerStatus.ANSWERED, answer, cited=[])

        bundle, dropped = self.context_builder.build_chunks_within_budget(
            chunks,
            self.token_counter,
            max_context_tokens or self.max_context_tokens,
        )
        state["context_dropped_chunk_ids"] = dropped
        state["alias_map"] = bundle.alias_map
        state["retrieved"] = self._retrieved_debug(chunks)
        state["context_text"] = bundle.text
        retrieved_ids = list(bundle.report_metadata)

        if state.get("compare_by"):
            return self._compose_compare_answer(
                result, state, chunks, bundle, question, deadline=deadline
            )

        if id_route:
            # Fix pack H1: the question names a report id while the model only sees
            # the alias. Rewrite the identifier in the prompt copy of the question
            # (never user-facing) and say which reports are missing.
            question = prompts.build_id_route_question(question, bundle.alias_map)
            hint_extra = prompts.build_id_route_hint(
                bundle.alias_map, state.get("missing_report_ids") or ()
            )

        completion = self._answer_with_llm(
            bundle.text, question, mode, deadline=deadline, hint_extra=hint_extra
        )
        state["calls"].append(completion.to_metrics("answer"))
        state["raw_answers"].append(self._raw_answer(1, completion))
        answer_text = sanitize_answer(completion.content)

        truncated = completion.finish_reason == "length"
        if truncated:
            state["incidents"]["answer_truncated"] = 1
            answer_text = trim_to_last_complete_unit(answer_text)

        if prompts.is_not_found_refusal(answer_text):
            # Fix pack H2: the model declared the information unavailable. That is
            # a decisive outcome, so it is never retried as an uncited answer.
            return self._declared_not_found(result, state, id_route)

        def resolve(raw_answer: str):
            citations = process_citations(
                raw_answer, bundle.alias_map, retrieved_ids
            )
            missing = find_uncited_mentions(
                citations.text,
                citations.cited_report_ids,
                bundle.report_metadata,
            )
            return citations, missing

        citations, uncited = resolve(answer_text)
        if (not citations.cited_report_ids or uncited) and not truncated:
            # One stricter attempt before giving up (Brief 5.6). A truncated
            # answer already spent the completion cap, so it is not retried: the
            # repair is the deterministic trim above.
            retry_prompt = (
                f"{self._answer_user_prompt(bundle.text, question, mode, hint_extra=hint_extra)}"
                f"\n\n{prompts.CITATION_RETRY_INSTRUCTION}"
            )
            retry_completion = self.llm.complete(
                prompts.SYSTEM_PROMPT,
                retry_prompt,
                temperature=config.LLM_TEMPERATURE_CHAT,
                max_tokens=config.LLM_MAX_OUTPUT_TOKENS,
                reasoning_effort=self.answer_reasoning_effort,
                json_mode=False,
                deadline=deadline,
            )
            state["calls"].append(retry_completion.to_metrics("answer_retry"))
            state["raw_answers"].append(self._raw_answer(2, retry_completion))
            answer_text = sanitize_answer(retry_completion.content)
            truncated = retry_completion.finish_reason == "length"
            if truncated:
                state["incidents"]["answer_truncated"] = 1
                answer_text = trim_to_last_complete_unit(answer_text)
            if prompts.is_not_found_refusal(answer_text):
                return self._declared_not_found(result, state, id_route)
            citations, uncited = resolve(answer_text)

        for key, value in citations.incidents.items():
            state["incidents"][key] = state["incidents"].get(key, 0) + value

        if uncited:
            state["incidents"]["uncited_mention"] = len(uncited)
            logger.info("chat_uncited_report_mentions count=%d", len(uncited))

        if not citations.cited_report_ids or uncited:
            return result(AnswerStatus.NO_SOURCE, prompts.NO_SOURCE_RESPONSE)

        guarded = self.output_guard.apply(citations.text)
        for key, value in guarded.incidents.items():
            state["incidents"][key] = state["incidents"].get(key, 0) + value

        if guarded.blocked:
            # A blocked answer carries no source block, so it claims no sources.
            return result(AnswerStatus.SECURITY_REFUSED, guarded.text, cited=[])

        state["style"] = citation_style_metrics(guarded.text)

        answer_text = guarded.text
        if truncated:
            answer_text = f"{answer_text}\n\n{prompts.TRUNCATION_NOTICE}"

        # Fix pack K2: measured on the answer body, before the backend coverage
        # line and Sumber block, which are not model statements.
        state["negation"] = negation_sentences(answer_text)

        coverage_line = self._coverage_line(
            state, bundle, chunks, citations.cited_report_ids
        )
        state["coverage"] = coverage_line
        if coverage_line:
            answer_text = f"{answer_text}{config.COVERAGE_BLOCK_SEPARATOR}{coverage_line}"

        source_block = build_source_block(
            citations.cited_report_ids, bundle.report_metadata
        )
        final_answer = compose_answer(answer_text, source_block)

        return result(
            AnswerStatus.ANSWERED, final_answer, cited=citations.cited_report_ids
        )
