"""Phase 4 fix pack E2: analyzer prompt v2 and the example-copy guard.

The 16 legitimate day-1 questions were rejected as out of scope; the prompt's
scope rule (and the example block) is the cause under test here. The guard is the
second line of defence: an analyzer query copied from an example never becomes the
retrieval query for an unrelated question.
"""

import json

import pytest

from core import prompts
from core.analyzer import Analyzer, guard_example_copy, validate_analysis
from core.pipeline import RagPipeline
from tests.stubs import FakeKnowledgeBase, FakeLLM, RecordingEmbedder, WordCountTokenizer

# The two day-1 questions whose analyzer query was the negative example's query.
Q061 = "What problems did village tourism managers face with online bookings?"
Q066 = (
    "What did the packaging automation team do after the belt slipped under heavy"
    " packages?"
)
COPIED_QUERY = "bantuan menulis kode python"


def analysis_for(question, search_query):
    payload = {
        "in_scope": True,
        "mode": "specific",
        "search_query": search_query,
        "filters": {},
        "compare_by": None,
    }
    return validate_analysis(payload, question)


# -- the prompt -------------------------------------------------------------
def test_prompt_version_is_v2():
    assert prompts.ANALYZER_PROMPT_VERSION == "2.0"


def test_scope_rule_covers_projects_without_the_competition_words():
    body = prompts.ANALYZER_PROMPT_BODY
    rule = body.split("Aturan:")[1].split("2. mode")[0]
    assert "SETIAP pertanyaan" in rule
    assert "tanpa kata" in rule
    assert "kompetisi/lomba/laporan" in rule
    assert "dalam bahasa apa pun" in rule
    assert "belum ada" in rule
    assert "Jika ragu" in rule
    for unrelated in ("politik", "cuaca", "hiburan", "hasil", "data pribadi"):
        assert unrelated in rule


def test_search_query_rule_forbids_reusing_the_examples():
    rule = prompts.ANALYZER_PROMPT_BODY.split("Aturan:")[1].split("4. filters")[0]
    assert "merumuskan ulang PERTANYAAN INI" in rule
    assert "jangan memakai kata, topik, atau teks dari contoh" in rule


def test_new_positive_examples_do_not_use_the_competition_words():
    questions = [question for question, _answer in prompts.ANALYZER_FEW_SHOT_EXAMPLES]
    # The two appended positives are the only Indonesian/English pair without the
    # words; every original positive example contains one of them.
    fresh = [question for question in questions if question in (
        "Kendala apa yang muncul saat pengingat minum obat terlambat?",
        "What broke in the classroom booking app test?",
    )]
    assert len(fresh) == 2
    assert any(question.startswith("Kendala") for question in fresh)
    assert any(question.startswith("What") for question in fresh)
    for question in fresh:
        lowered = question.lower()
        assert "kompetisi" not in lowered and "lomba" not in lowered
        assert "laporan" not in lowered
        index = questions.index(question)
        answer = json.loads(prompts.ANALYZER_FEW_SHOT_EXAMPLES[index][1])
        assert answer["in_scope"] is True
        assert answer["mode"] == "specific"
        assert answer["filters"] == {}


def test_recurring_pattern_example_is_cross_report():
    questions = [question for question, _answer in prompts.ANALYZER_FEW_SHOT_EXAMPLES]
    question = "Kendala apa yang berulang di beberapa tim?"
    answer = json.loads(
        prompts.ANALYZER_FEW_SHOT_EXAMPLES[questions.index(question)][1]
    )
    assert answer["mode"] == "cross_report"
    assert answer["compare_by"] is None


def test_the_negative_examples_are_kept_and_clearly_unrelated():
    questions = [question for question, _answer in prompts.ANALYZER_FEW_SHOT_EXAMPLES]
    assert "Tolong tuliskan fungsi Python untuk mengurutkan list." in questions
    assert "Siapa presiden Indonesia saat ini?" in questions


# -- the guard --------------------------------------------------------------
@pytest.mark.parametrize("question", [Q061, Q066])
def test_a_copied_example_query_becomes_the_users_question(question):
    result = guard_example_copy(analysis_for(question, COPIED_QUERY), question)
    assert result.example_copy_guard is True
    assert result.search_query == question
    assert result.in_scope is True


def test_the_guard_ignores_case_and_punctuation():
    question = Q061
    result = guard_example_copy(
        analysis_for(question, "Bantuan menulis kode Python!"), question
    )
    assert result.example_copy_guard is True
    assert result.search_query == question


def test_the_guard_leaves_an_ordinary_query_alone():
    question = "Kendala apa yang muncul saat pengingat minum obat terlambat?"
    result = guard_example_copy(
        analysis_for(question, "kendala pengingat obat lansia"), question
    )
    assert result.example_copy_guard is False
    assert result.search_query == "kendala pengingat obat lansia"


def test_the_guard_does_not_fire_when_the_user_asks_the_example_itself():
    question = "Tolong tuliskan fungsi Python untuk mengurutkan list."
    result = guard_example_copy(analysis_for(question, COPIED_QUERY), question)
    assert result.example_copy_guard is False
    assert result.search_query == COPIED_QUERY


# -- through the analyzer and the pipeline ----------------------------------
def test_the_analyzer_applies_the_guard_to_its_own_result():
    raw = json.dumps(
        {
            "in_scope": True,
            "mode": "specific",
            "search_query": COPIED_QUERY,
            "filters": {},
        }
    )
    analyzer = Analyzer(FakeLLM(json_responses=[raw]))
    result = analyzer.analyze(Q066)
    assert result.example_copy_guard is True
    assert result.search_query == Q066


def test_the_pipeline_counts_the_incident_and_uses_the_question():
    raw = json.dumps(
        {
            "in_scope": True,
            "mode": "specific",
            "search_query": COPIED_QUERY,
            "filters": {},
        }
    )
    embedder = RecordingEmbedder()
    pipeline = RagPipeline(
        # total=1 so the retriever really embeds the query instead of returning early.
        knowledge_base=FakeKnowledgeBase(total=1),
        llm=FakeLLM(json_responses=[raw]),
        embedder=embedder,
        token_counter=WordCountTokenizer(),
    )

    result = pipeline.run(Q061)

    assert result.debug.incidents["example_copy_guard"] == 1
    # No chunks in the fake, so the run stops at NOT_FOUND; the query was still
    # the user's question, never the example text.
    assert any(Q061 in query for query in embedder.queries)
    assert not any(COPIED_QUERY in query for query in embedder.queries)
