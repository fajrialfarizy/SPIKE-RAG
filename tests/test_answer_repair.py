"""
Offline regression tests for fix pack G1 (exact report-id route) and G2
(length-truncated answers).

Both paths run through RagPipeline with FakeLLM, so no network or model weights
are involved.
"""

import config
from core.knowledge_base import StoredChunk
from core.pipeline import AnswerStatus, RagPipeline
from tests.stubs import FakeKnowledgeBase, FakeLLM, RecordingEmbedder

ID_QUESTION = "Apa saja saran pengembangan yang tercantum pada RPT0002?"


def stored(report_id="RPT0002", section="saran_pengembangan", chunk_id=None):
    return StoredChunk(
        chunk_id=chunk_id or f"{report_id}:{section}:01",
        report_id=report_id,
        document="Validasi pasar dan penguatan model bisnis.",
        metadata={
            "report_id": report_id,
            "competition_name": "Inovasi Bisnis Mahasiswa Nasional (IBMN) 2023",
            "competition_type": "Business Plan",
            "competition_year": 2023,
            "achievement_level": "Finalis",
            "section_type": section,
            "source_file_id": f"DRV_{report_id}.pdf",
            "chunk_id": chunk_id or f"{report_id}:{section}:01",
        },
    )


def build_id(chunks=None, text_responses=None, finish_reasons=None):
    llm = FakeLLM(
        text_responses=text_responses,
        text_finish_reasons=finish_reasons,
    )
    pipeline = RagPipeline(
        knowledge_base=FakeKnowledgeBase(chunks=chunks or [stored()]),
        llm=llm,
        embedder=RecordingEmbedder(),
    )
    return pipeline, llm


# -- G1: the identity route -------------------------------------------------
def test_the_id_route_tells_the_model_which_alias_the_id_refers_to():
    pipeline, llm = build_id(
        text_responses=["Saran utamanya adalah validasi pasar [LAP1]."]
    )

    result = pipeline.run(ID_QUESTION)

    assert result.status is AnswerStatus.ANSWERED
    assert result.cited_report_ids == ["RPT0002"]
    assert result.debug.route == "report_id"
    assert "[RPT0002]" in result.answer
    assert "LAP1" not in result.answer

    prompt = llm.calls[0]["user_prompt"]
    assert "rute identitas laporan" in prompt
    assert "Laporan pada konteks: [LAP1]." in prompt
    assert 'bagian="saran_pengembangan" bagian_label="Saran Pengembangan"' in prompt


def test_a_raw_report_id_citation_is_accepted_when_it_is_the_retrieved_report():
    # The model sometimes writes the id from the question instead of the alias.
    # The backend keeps it only because it is in the retrieved set.
    pipeline, _ = build_id(
        text_responses=["Saran utamanya adalah validasi pasar [RPT0002]."]
    )

    result = pipeline.run(ID_QUESTION)

    assert result.status is AnswerStatus.ANSWERED
    assert result.cited_report_ids == ["RPT0002"]


def test_an_answer_without_any_citation_is_refused_after_the_retry():
    pipeline, llm = build_id(
        text_responses=[
            "Informasi tersebut belum tersedia dalam basis pengetahuan.",
            "Informasi tersebut belum tersedia dalam basis pengetahuan.",
        ]
    )

    result = pipeline.run(ID_QUESTION)

    assert result.status is AnswerStatus.NO_SOURCE
    assert llm.text_call_count == 2
    # Every attempt's raw output is kept for the developer smoke run.
    assert [item["attempt"] for item in result.debug.raw_answers] == [1, 2]
    assert all(item["content"] for item in result.debug.raw_answers)


def test_raw_answer_evidence_never_reaches_the_user_answer():
    pipeline, _ = build_id(
        text_responses=[
            "Informasi belum tersedia.",
            "Informasi belum tersedia.",
        ]
    )

    result = pipeline.run(ID_QUESTION)

    assert result.answer == (
        "Maaf, saya belum dapat menyusun jawaban yang dapat ditelusuri ke dokumen "
        "sumber. Coba ubah pertanyaanmu."
    )


# -- G2: finish_reason == "length" -----------------------------------------
def test_a_length_truncated_answer_is_trimmed_and_flagged_without_a_retry():
    truncated = (
        "Saran utamanya adalah validasi pasar [LAP1]. Rincian berikutnya tentang **"
    )
    pipeline, llm = build_id(
        text_responses=[truncated], finish_reasons=["length"]
    )

    result = pipeline.run(ID_QUESTION)

    assert result.status is AnswerStatus.ANSWERED
    assert llm.text_call_count == 1
    assert "**" not in result.answer.split("**Sumber**")[0]
    assert config.ANSWER_TARGET_WORDS <= 400
    assert result.answer.startswith("Saran utamanya adalah validasi pasar [RPT0002].")
    assert "Jawaban dipersingkat karena batas panjang." in result.answer
    assert result.debug.incidents["answer_truncated"] == 1
    assert result.debug.raw_answers[0]["finish_reason"] == "length"


def test_a_dangling_alias_bracket_is_removed_instead_of_failing_postprocessing():
    truncated = "Validasi pasar menjadi fokus utama [LAP1]. Catatan tambahan [LAP"
    pipeline, _ = build_id(
        text_responses=[truncated], finish_reasons=["length"]
    )

    result = pipeline.run(ID_QUESTION)

    assert result.status is AnswerStatus.ANSWERED
    assert "LAP" not in result.answer
    assert "[RPT0002]" in result.answer


def test_a_dangling_open_parenthesis_is_trimmed_away():
    truncated = "Fokus utama adalah validasi pasar [LAP1]. Rincian (belum selesai"
    pipeline, _ = build_id(
        text_responses=[truncated], finish_reasons=["length"]
    )

    result = pipeline.run(ID_QUESTION)

    assert result.status is AnswerStatus.ANSWERED
    assert "belum selesai" not in result.answer
    assert "Jawaban dipersingkat karena batas panjang." in result.answer


def test_a_normal_finish_reason_keeps_the_whole_answer_and_adds_no_notice():
    pipeline, _ = build_id(
        text_responses=["Saran utamanya adalah validasi pasar [LAP1]."],
        finish_reasons=["stop"],
    )

    result = pipeline.run(ID_QUESTION)

    assert result.status is AnswerStatus.ANSWERED
    assert "Jawaban dipersingkat" not in result.answer
    assert "answer_truncated" not in result.debug.incidents


def test_a_truncated_answer_without_citations_is_refused_without_a_second_call():
    pipeline, llm = build_id(
        text_responses=["Informasi ini belum tersedia di basis pengetahuan."],
        finish_reasons=["length"],
    )

    result = pipeline.run(ID_QUESTION)

    assert result.status is AnswerStatus.NO_SOURCE
    assert llm.text_call_count == 1
    assert result.debug.incidents["answer_truncated"] == 1
