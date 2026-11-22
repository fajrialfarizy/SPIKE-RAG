"""
Offline tests for fix pack 9.

O1: free-text answers through the strict JSON schema.
O2: keyed comparison groups, citation ceilings that drop, concrete dimensions,
    empty-group handling and coverage wording.
O3: one structured repair retry, and the Q1 quota replay on a fake clock.
O4: backend-written lines are not counted as model negation or causation.

FakeLLM and the other doubles come from tests/stubs.py: no network, no API key,
no model weights.
"""

import json

import pytest

import config
from core import compare_answer, prompts, structured_answer
from core.causal import causal_connectors
from core.coverage import build_coverage_line
from core.knowledge_base import ChunkMatch, StoredChunk
from core.llm_client import LLMRateLimitError
from core.pipeline import AnswerStatus, RagPipeline
from core.rate_limiter import MinuteTokenBudgetExhausted, TokenAwareRateLimiter
from tests.stubs import (
    FakeLLM,
    FilteringKnowledgeBase,
    RecordingEmbedder,
    WordCountTokenizer,
)

AWARD_LABEL = "meraih penghargaan"
FINALIST_LABEL = "belum juara (finalis)"


# -- fixtures --------------------------------------------------------------
def _metadata(report_id, level="Juara 1", competition_type="Business Plan"):
    return {
        "report_id": report_id,
        "competition_name": f"Kompetisi {report_id}",
        "competition_type": competition_type,
        "competition_year": 2023,
        "achievement_level": level,
        "section_type": "evaluasi_tim",
        "chunk_id": f"{report_id}:evaluasi_tim:01",
    }


def _match(report_id, level="Juara 1", competition_type="Business Plan"):
    meta = _metadata(report_id, level, competition_type)
    return ChunkMatch(
        chunk_id=meta["chunk_id"],
        report_id=report_id,
        document=f"Isi {report_id}.",
        metadata=meta,
        distance=0.1,
    )


def _analyzer(mode="specific", compare_by=None):
    return json.dumps(
        {
            "in_scope": True,
            "mode": mode,
            "search_query": "pengalaman tim",
            "filters": {},
            "compare_by": compare_by,
        }
    )


def _answer_pipeline(structured, *, matches=None, metadatas=None):
    matches = matches or [_match("RPT9001"), _match("RPT9002")]
    metadatas = metadatas or [item.metadata for item in matches]
    llm = FakeLLM(
        json_responses=[_analyzer()],
        structured_responses=structured,
    )
    pipeline = RagPipeline(
        knowledge_base=FilteringKnowledgeBase(matches=matches, metadatas=metadatas),
        llm=llm,
        embedder=RecordingEmbedder(),
        token_counter=WordCountTokenizer(),
    )
    return pipeline, llm


def _points(*items):
    return [{"text": text, "cite": list(cite)} for text, cite in items]


def _payload(points, found=True):
    items = [
        {"text": value[0], "cite": list(value[1])}
        if isinstance(value, (tuple, list))
        else value
        for value in points
    ]
    return json.dumps({"found": found, "points": items}, ensure_ascii=False)


# -- O1: structured free-text answers -------------------------------------
def test_the_free_text_schema_is_strict():
    schema = structured_answer.build_schema()

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"found", "points"}
    point = schema["properties"]["points"]["items"]
    assert point["additionalProperties"] is False
    assert set(point["required"]) == {"text", "cite"}


def test_a_valid_answer_is_rendered_as_one_bullet_per_point():
    payload = _payload(
        [
            ("Tim memvalidasi masalah lewat wawancara.", ["LAP1"]),
            ("Riset pasar terbatas pada data sekunder.", ["LAP2"]),
        ]
    )
    pipeline, _ = _answer_pipeline([payload])

    result = pipeline.run("Apa pengalaman tim pada business plan?")

    assert result.status is AnswerStatus.ANSWERED
    assert result.answer.splitlines()[:2] == [
        "- Tim memvalidasi masalah lewat wawancara. [RPT9001]",
        "- Riset pasar terbatas pada data sekunder. [RPT9002]",
    ]
    assert "LAP" not in result.answer
    assert "**Sumber**" in result.answer


def test_several_cites_are_rendered_as_separate_brackets():
    payload = _payload(
        [("Tim memvalidasi masalah lewat wawancara.", ["LAP1", "LAP2"])]
    )
    pipeline, _ = _answer_pipeline([payload])

    result = pipeline.run("Apa pengalaman tim pada business plan?")

    assert "- Tim memvalidasi masalah lewat wawancara. [RPT9001] [RPT9002]" in (
        result.answer
    )
    assert result.cited_report_ids == ["RPT9001", "RPT9002"]


def test_an_uncited_point_is_dropped_and_counted():
    payload = _payload(
        [
            ("Klaim tanpa sumber.", []),
            ("Tim memvalidasi masalah lewat wawancara.", ["LAP1"]),
        ]
    )
    pipeline, _ = _answer_pipeline([payload])

    result = pipeline.run("Apa pengalaman tim pada business plan?")

    assert "Klaim tanpa sumber." not in result.answer
    assert result.debug.incidents["point_uncited"] == 1


def test_an_unknown_alias_is_dropped():
    payload = _payload(
        [
            ("Klaim dari alias asing.", ["LAP9"]),
            ("Tim memvalidasi masalah lewat wawancara.", ["LAP1"]),
        ]
    )
    pipeline, _ = _answer_pipeline([payload])

    result = pipeline.run("Apa pengalaman tim pada business plan?")

    assert "Klaim dari alias asing." not in result.answer
    assert result.debug.incidents["unknown_alias"] == 1


def test_the_per_report_cap_and_the_total_cap_apply():
    points = (
        [(f"Poin A{index}.", ["LAP1"]) for index in range(8)]
        + [(f"Poin B{index}.", ["LAP2"]) for index in range(2)]
        + [(f"Poin C{index}.", ["LAP3"]) for index in range(2)]
    )
    matches = [_match("RPT9001"), _match("RPT9002"), _match("RPT9003")]
    pipeline, _ = _answer_pipeline([_payload(points)], matches=matches)

    result = pipeline.run("Apa pengalaman tim pada business plan?")

    assert result.status is AnswerStatus.ANSWERED
    body = result.answer.split("**Sumber**")[0]
    bullets = [line for line in body.splitlines() if line.startswith("- ")]
    assert len(bullets) == 6
    for report_id in ("RPT9001", "RPT9002", "RPT9003"):
        assert sum(1 for line in bullets if report_id in line) == 2
    assert result.debug.dropped_points == 6


def test_an_alias_token_inside_the_text_is_stripped():
    payload = _payload([("Tim juara (LAP1) memvalidasi masalah.", ["LAP1"])])
    pipeline, _ = _answer_pipeline([payload])

    result = pipeline.run("Apa pengalaman tim pada business plan?")

    assert "LAP" not in result.answer
    assert "- Tim juara memvalidasi masalah. [RPT9001]" in result.answer
    assert result.debug.incidents["alias_in_text"] == 1


def test_a_model_declared_not_found_on_the_semantic_route_is_not_found():
    pipeline, llm = _answer_pipeline([_payload([], found=False)])

    result = pipeline.run("Apa pengalaman tim pada kompetisi robotika?")

    assert result.status is AnswerStatus.NOT_FOUND
    assert result.answer == prompts.NOT_FOUND_RESPONSE
    assert llm.structured_call_count == 1
    assert result.debug.incidents.get("model_declared_not_found") == 1


def test_a_declared_not_found_on_the_id_route_is_no_source():
    stored = StoredChunk(
        chunk_id="RPT0002:saran_pengembangan:01",
        report_id="RPT0002",
        document="Isi RPT0002.",
        metadata=_metadata("RPT0002"),
    )
    llm = FakeLLM(structured_responses=[_payload([], found=False)])
    pipeline = RagPipeline(
        knowledge_base=FilteringKnowledgeBase(chunks=[stored], metadatas=[stored.metadata]),
        llm=llm,
        embedder=RecordingEmbedder(),
        token_counter=WordCountTokenizer(),
    )

    result = pipeline.run("Apa saja saran pengembangan pada RPT0002?")

    assert result.status is AnswerStatus.NO_SOURCE
    assert result.debug.incidents["model_declared_not_found_with_context"] == 1


def test_no_valid_point_triggers_one_repair_retry():
    bad = _payload([("Klaim dari alias asing.", ["LAP9"])])
    good = _payload([("Tim memvalidasi masalah lewat wawancara.", ["LAP1"])])
    pipeline, llm = _answer_pipeline([bad, good])

    result = pipeline.run("Apa pengalaman tim pada business plan?")

    assert result.status is AnswerStatus.ANSWERED
    assert llm.structured_call_count == 2
    assert result.debug.incidents["structured_repair"] == 1
    assert "[RPT9001]" in result.answer
    # The retry may wait longer for the minute window than a first call.
    assert llm.calls[2]["queue_max_wait_seconds"] == config.QUEUE_MAX_WAIT_RETRY_SECONDS


class _RateLimitedRepairLLM(FakeLLM):
    """Serves the first structured answer, then refuses the repair retry."""

    def complete(self, *args, **kwargs):
        response_format = kwargs.get("response_format") or {}
        structured = response_format.get("type") == "json_schema"
        if structured and self.structured_call_count >= 1:
            raise LLMRateLimitError("rate limited", retry_after_seconds=52.0)
        return super().complete(*args, **kwargs)


def test_an_unaffordable_repair_is_rate_limited_not_no_source():
    llm = _RateLimitedRepairLLM(
        json_responses=[_analyzer()],
        structured_responses=[_payload([("Klaim dari alias asing.", ["LAP9"])])],
    )
    pipeline = RagPipeline(
        knowledge_base=FilteringKnowledgeBase(
            matches=[_match("RPT9001")], metadatas=[_metadata("RPT9001")]
        ),
        llm=llm,
        embedder=RecordingEmbedder(),
        token_counter=WordCountTokenizer(),
    )

    result = pipeline.run("Apa pengalaman tim pada business plan?")

    assert result.status is AnswerStatus.RATE_LIMITED
    assert result.retry_after_seconds == 52.0


def test_the_sumber_block_lists_only_cited_reports():
    payload = _payload([("Tim memvalidasi masalah lewat wawancara.", ["LAP2"])])
    pipeline, _ = _answer_pipeline([payload])

    result = pipeline.run("Apa pengalaman tim pada business plan?")

    assert "RPT9002" in result.answer
    sumber = result.answer.split("**Sumber**")[1]
    assert "RPT9001" not in sumber


def test_the_text_path_is_still_selectable_by_config(monkeypatch):
    monkeypatch.setattr(config, "ANSWER_FORMAT", "text")
    llm = FakeLLM(
        json_responses=[_analyzer()],
        text_responses=["Tim memvalidasi masalah [LAP1]."],
    )
    pipeline = RagPipeline(
        knowledge_base=FilteringKnowledgeBase(
            matches=[_match("RPT9001")], metadatas=[_metadata("RPT9001")]
        ),
        llm=llm,
        embedder=RecordingEmbedder(),
        token_counter=WordCountTokenizer(),
    )

    result = pipeline.run("Apa pengalaman tim pada business plan?")

    assert result.status is AnswerStatus.ANSWERED
    assert llm.structured_call_count == 0


def test_the_structured_addendum_is_versioned_and_keeps_the_prd_core():
    addendum = prompts.STRUCTURED_FORMAT_ADDENDUM

    assert prompts.STRUCTURED_ADDENDUM_VERSION == "3.0"
    assert "(v3.0)" in addendum
    assert "SATU fakta" in addendum
    assert "kecuali kalimat sumber yang sama menyatakan hubungan itu" in addendum
    assert prompts.PRD_9_1_SYSTEM_PROMPT in prompts.SYSTEM_PROMPT_STRUCTURED


# -- O2: comparison groups, ceilings and coverage --------------------------
GROUP_KEYS = {"g1": AWARD_LABEL, "g2": FINALIST_LABEL}
ALIAS_MAP = {
    "LAP1": "RPT0001",
    "LAP2": "RPT0002",
    "LAP3": "RPT0003",
    "LAP4": "RPT0004",
    "LAP5": "RPT0005",
}
REPORT_GROUP = {
    "RPT0001": AWARD_LABEL,
    "RPT0002": FINALIST_LABEL,
    "RPT0003": FINALIST_LABEL,
    "RPT0004": FINALIST_LABEL,
    "RPT0005": FINALIST_LABEL,
}


def _validate(payload, *, alias_map=None, report_group=None, keys=None):
    return compare_answer.validate_payload(
        payload,
        group_keys=keys or GROUP_KEYS,
        alias_map=alias_map or ALIAS_MAP,
        report_group=report_group or REPORT_GROUP,
        retrieved=list((alias_map or ALIAS_MAP).values()),
    )


def _compare_payload(points_award, points_finalist, contrasts=None):
    return {
        "groups": {"g1": points_award, "g2": points_finalist},
        "contrasts": contrasts or [],
    }


def test_a_point_over_the_group_citation_ceiling_is_dropped_not_cut():
    validation = _validate(
        _compare_payload(
            [],
            [{"text": "Klaim.", "cite": ["LAP2", "LAP3", "LAP4"]}],
        )
    )

    assert validation.sections[1].points == []
    assert validation.incidents["cite_over_limit_dropped"] == 1


def test_a_side_with_three_aliases_is_kept():
    # The real run's finalist side cited LAP4, LAP5, LAP6 and the backend cut it
    # to two. Three are now allowed on a contrast side.
    validation = _validate(
        _compare_payload(
            [{"text": "Poin juara.", "cite": ["LAP1"]}],
            [{"text": "Poin finalis.", "cite": ["LAP2"]}],
            [
                {
                    "dimension": "kesiapan data",
                    "sides": [
                        {"group": "g1", "text": "Data primer.", "cite": ["LAP1"]},
                        {
                            "group": "g2",
                            "text": "Data sekunder.",
                            "cite": ["LAP2", "LAP3"],
                        },
                    ],
                }
            ],
        )
    )

    assert len(validation.contrasts) == 1
    assert validation.contrasts[0].sides[1].report_ids == ["RPT0002", "RPT0003"]


def test_a_side_over_three_aliases_is_dropped_not_cut():
    validation = _validate(
        _compare_payload(
            [{"text": "Poin juara.", "cite": ["LAP1"]}],
            [{"text": "Poin finalis.", "cite": ["LAP2"]}],
            [
                {
                    "dimension": "kesiapan data",
                    "sides": [
                        {"group": "g1", "text": "Data primer.", "cite": ["LAP1"]},
                        {
                            "group": "g2",
                            "text": "Data sekunder.",
                            "cite": ["LAP2", "LAP3", "LAP4", "LAP5"],
                        },
                    ],
                }
            ],
        )
    )

    assert validation.contrasts == []
    assert validation.incidents["cite_over_limit_dropped"] == 1


@pytest.mark.parametrize("dimension", ["Pembelajaran", "kendala", "ASPEK"])
def test_a_generic_dimension_is_rejected(dimension):
    validation = _validate(
        _compare_payload(
            [{"text": "Poin juara.", "cite": ["LAP1"]}],
            [{"text": "Poin finalis.", "cite": ["LAP2"]}],
            [
                {
                    "dimension": dimension,
                    "sides": [
                        {"group": "g1", "text": "A.", "cite": ["LAP1"]},
                        {"group": "g2", "text": "B.", "cite": ["LAP2"]},
                    ],
                }
            ],
        )
    )

    assert validation.contrasts == []
    assert validation.incidents["contrast_invalid"] == 1


def test_a_concrete_dimension_is_accepted():
    validation = _validate(
        _compare_payload(
            [{"text": "Poin juara.", "cite": ["LAP1"]}],
            [{"text": "Poin finalis.", "cite": ["LAP2"]}],
            [
                {
                    "dimension": "Persiapan presentasi",
                    "sides": [
                        {"group": "g1", "text": "A.", "cite": ["LAP1"]},
                        {"group": "g2", "text": "B.", "cite": ["LAP2"]},
                    ],
                }
            ],
        )
    )

    assert len(validation.contrasts) == 1


def test_the_keyed_schema_requires_every_group_key():
    schema = compare_answer.build_schema(["g1", "g2"])

    groups = schema["properties"]["groups"]
    assert groups["properties"].keys() == {"g1", "g2"}
    assert groups["required"] == ["g1", "g2"]
    assert groups["additionalProperties"] is False


def test_the_keyed_payload_renders_both_groups():
    validation = _validate(
        _compare_payload(
            [{"text": "Poin juara.", "cite": ["LAP1"]}],
            [{"text": "Poin finalis.", "cite": ["LAP2"]}],
        )
    )

    rendered = compare_answer.render_answer(validation, "achievement_group")

    assert rendered.startswith("**Tim yang meraih penghargaan")
    assert "**Tim belum juara (finalis)**" in rendered
    assert "- Poin juara. [RPT0001]" in rendered


def _compare_pipeline(payload, *, compare_by="achievement_group", matches=None):
    matches = matches or [_match("RPT9001"), _match("RPT9002", level="Finalis")]
    metadatas = [item.metadata for item in matches]
    llm = FakeLLM(
        json_responses=[_analyzer("cross_report", compare_by)],
        structured_responses=[json.dumps(payload)],
    )
    pipeline = RagPipeline(
        knowledge_base=FilteringKnowledgeBase(matches=matches, metadatas=metadatas),
        llm=llm,
        embedder=RecordingEmbedder(),
        token_counter=WordCountTokenizer(),
    )
    return pipeline, llm


def test_an_empty_group_with_documents_is_repaired_once():
    first = _compare_payload(
        [{"text": "Poin juara.", "cite": ["LAP1"]}], []
    )
    second = _compare_payload(
        [{"text": "Poin juara.", "cite": ["LAP1"]}],
        [{"text": "Poin finalis.", "cite": ["LAP2"]}],
    )
    matches = [_match("RPT9001"), _match("RPT9002", level="Finalis")]
    llm = FakeLLM(
        json_responses=[_analyzer("cross_report", "achievement_group")],
        structured_responses=[json.dumps(first), json.dumps(second)],
    )
    pipeline = RagPipeline(
        knowledge_base=FilteringKnowledgeBase(
            matches=matches, metadatas=[item.metadata for item in matches]
        ),
        llm=llm,
        embedder=RecordingEmbedder(),
        token_counter=WordCountTokenizer(),
    )

    result = pipeline.run("Bandingkan tim juara dan belum juara.")

    assert result.status is AnswerStatus.ANSWERED
    assert llm.structured_call_count == 2
    assert "Poin finalis. [RPT9002]" in result.answer
    assert config.COMPARE_GROUP_EMPTY_RETRY not in result.answer


def test_a_still_empty_group_degrades_and_omits_contrasts(monkeypatch):
    monkeypatch.setattr(config, "STRUCTURED_REPAIR_RETRIES", 0)
    payload = _compare_payload(
        [{"text": "Poin juara.", "cite": ["LAP1"]}], []
    )
    pipeline, _ = _compare_pipeline(payload)

    result = pipeline.run("Bandingkan tim juara dan belum juara.")

    assert result.status is AnswerStatus.ANSWERED
    assert config.COMPARE_GROUP_EMPTY_RETRY in result.answer
    assert config.COMPARE_EMPTY_GROUP not in result.answer
    assert config.COMPARE_CONTRASTS_HEADING not in result.answer
    assert result.debug.incidents["group_empty"] == 1


def test_reports_cited_per_group_is_measured():
    payload = _compare_payload(
        [{"text": "Poin juara.", "cite": ["LAP1"]}],
        [{"text": "Poin finalis.", "cite": ["LAP2"]}],
    )
    pipeline, _ = _compare_pipeline(payload)

    result = pipeline.run("Bandingkan tim juara dan belum juara.")

    assert result.debug.reports_cited_per_group == {
        AWARD_LABEL: 1,
        FINALIST_LABEL: 1,
    }


def test_the_coverage_sentence_for_two_groups_uses_dan():
    totals = [
        {"label": AWARD_LABEL, "coverage_label": AWARD_LABEL, "total": 3},
        {"label": FINALIST_LABEL, "coverage_label": FINALIST_LABEL, "total": 4},
    ]

    line = build_coverage_line(totals, {AWARD_LABEL: 3, FINALIST_LABEL: 1})

    assert line == (
        f"Ringkasan ini didasarkan pada 3 dari 3 laporan {AWARD_LABEL} dan "
        f"1 dari 4 laporan {FINALIST_LABEL}."
    )


def test_the_coverage_sentence_for_three_groups_uses_commas():
    totals = [
        {"label": "Business Plan", "coverage_label": "Business Plan", "total": 3},
        {"label": "Business Case", "coverage_label": "Business Case", "total": 4},
        {"label": "Essay", "coverage_label": "Essay", "total": 2},
    ]

    line = build_coverage_line(totals, {"Business Plan": 3, "Business Case": 1})

    assert line == (
        "Ringkasan ini didasarkan pada 3 dari 3 laporan Business Plan, "
        "1 dari 4 laporan Business Case dan 0 dari 2 laporan Essay."
    )


def test_a_competition_type_comparison_renders_only_the_two_requested_groups():
    matches = [
        _match("RPT9001", competition_type="Business Plan"),
        _match("RPT9002", competition_type="Business Case"),
    ]
    payload = _compare_payload(
        [{"text": "Poin BP.", "cite": ["LAP1"]}],
        [{"text": "Poin BC.", "cite": ["LAP2"]}],
    )
    pipeline, llm = _compare_pipeline(
        payload, compare_by="competition_type", matches=matches
    )

    result = pipeline.run("Bandingkan kendala business plan dan business case.")

    assert result.status is AnswerStatus.ANSWERED
    keys = llm.calls[1]["response_format"]["json_schema"]["schema"]["properties"][
        "groups"
    ]
    assert keys["required"] == ["g1", "g2"]
    assert "Essay" not in result.answer
    assert "Business Case" in result.answer


# -- O3: the repair retry cannot be starved by the queue --------------------
class _FakeClock:
    def __init__(self, start=0.0):
        self.now = float(start)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += float(seconds)


def test_the_real_q1_sequence_can_now_be_served():
    """
    Replay of the supervisor's Q1: the minute window already holds 5,034 tokens,
    the first structured answer returned 12 points with no citation, and the
    repair retry needs about 3,800 tokens. That wait is 52 s, over the 45 s
    first-call queue limit but inside the 60 s retry limit.
    """
    clock = _FakeClock()
    limiter = TokenAwareRateLimiter(clock=clock, sleep=clock.advance)
    limiter.reserve(5034)
    clock.advance(8.0)

    with pytest.raises(MinuteTokenBudgetExhausted):
        limiter.reserve(3800, max_wait_seconds=config.QUEUE_MAX_WAIT_SECONDS)

    # The retry is reserved with the longer retry limit: it waits for the first
    # call's tokens to age out (52 s) and is then served instead of answered
    # with a misleading RATE_LIMITED.
    limiter.reserve(
        3800, max_wait_seconds=config.QUEUE_MAX_WAIT_RETRY_SECONDS
    )

    assert clock.now == pytest.approx(60.0)
    assert limiter.waits == 1


def test_the_repair_retry_estimate_fits_the_retry_window():
    assert config.QUEUE_MAX_WAIT_RETRY_SECONDS > config.QUEUE_MAX_WAIT_SECONDS


# -- O4: backend lines are not model claims --------------------------------
def test_the_empty_group_sentence_is_not_a_model_negation():
    text = (
        "**Tim belum juara (finalis)**\n\n"
        f"{config.COMPARE_GROUP_EMPTY_RETRY}"
    )

    from core.negation import negation_sentences

    assert negation_sentences(text)["count"] == 0
    assert causal_connectors(text)["count"] == 0


def test_the_coverage_line_stays_out_of_the_metrics():
    text = (
        "- Poin tim. [RPT0001]\n\n"
        "Ringkasan ini didasarkan pada 1 dari 2 laporan meraih penghargaan."
    )

    from core.negation import negation_sentences

    assert negation_sentences(text)["count"] == 0
    assert causal_connectors(text)["count"] == 0
