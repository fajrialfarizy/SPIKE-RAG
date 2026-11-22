"""
Offline tests for fix pack 10.

R1: the exact report-id route (and a single-report context) lists every item of
    the requested report, up to the total cap.
R2: a comparison contrast references group points by number; the backend renders
    the validated point text; dimensions are compared exactly and a generic one
    gets the neutral label.
R3: wording metrics (interpretive phrases, strong verbs, lexical support) that
    measure and never block.
R4: reports_cited in the debug object for every mode.
R6: the smoke output numbers the group points so picks can be checked.

FakeLLM and the other doubles come from tests/stubs.py: no network, no API key,
no model weights.
"""

import json

import pytest

import config
from core import compare_answer, prompts, structured_answer, wording
from core.causal import causal_connectors
from core.knowledge_base import ChunkMatch, StoredChunk
from core.pipeline import AnswerStatus, RagPipeline
from tests.stubs import FakeLLM, FilteringKnowledgeBase, RecordingEmbedder, WordCountTokenizer

AWARD_LABEL = "meraih penghargaan"
FINALIST_LABEL = "belum juara (finalis)"
AWARD_HEADING = (
    "Tim yang meraih penghargaan (Juara 1-3, Juara Harapan, Most Aspiring)"
)
FINALIST_HEADING = "Tim belum juara (finalis)"


def _metadata(report_id, level="Juara 1", competition_type="Business Plan", section="evaluasi_tim"):
    return {
        "report_id": report_id,
        "competition_name": f"Kompetisi {report_id}",
        "competition_type": competition_type,
        "competition_year": 2023,
        "achievement_level": level,
        "section_type": section,
        "chunk_id": f"{report_id}:{section}:01",
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


def _stored(report_id, section="saran_pengembangan"):
    meta = _metadata(report_id, section=section)
    return StoredChunk(
        chunk_id=meta["chunk_id"],
        report_id=report_id,
        document=f"Isi {report_id}.",
        metadata=meta,
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


def _payload(points, found=True):
    return json.dumps(
        {
            "found": found,
            "points": [{"text": text, "cite": list(cite)} for text, cite in points],
        },
        ensure_ascii=False,
    )


def _structured_pipeline(structured, *, matches):
    llm = FakeLLM(json_responses=[_analyzer()], structured_responses=structured)
    pipeline = RagPipeline(
        knowledge_base=FilteringKnowledgeBase(
            matches=matches, metadatas=[item.metadata for item in matches]
        ),
        llm=llm,
        embedder=RecordingEmbedder(),
        token_counter=WordCountTokenizer(),
    )
    return pipeline, llm


def _id_route_pipeline(structured, *, report_ids=("RPT0002",)):
    stored = [_stored(report_id) for report_id in report_ids]
    llm = FakeLLM(structured_responses=structured)
    pipeline = RagPipeline(
        knowledge_base=FilteringKnowledgeBase(
            chunks=stored, metadatas=[item.metadata for item in stored]
        ),
        llm=llm,
        embedder=RecordingEmbedder(),
        token_counter=WordCountTokenizer(),
    )
    return pipeline, llm


def _bullets(answer):
    body = answer.split("**Sumber**")[0]
    return [line for line in body.splitlines() if line.startswith("- ")]


# -- R1: every item of one explicit report ---------------------------------
def test_the_id_route_renders_all_three_items_of_the_requested_report():
    pipeline, _ = _id_route_pipeline(
        [
            _payload(
                [
                    ("Saran pertama.", ["LAP1"]),
                    ("Saran kedua.", ["LAP1"]),
                    ("Saran ketiga.", ["LAP1"]),
                ]
            )
        ]
    )

    result = pipeline.run("Apa saja saran pengembangan yang tercantum pada RPT0002?")

    assert result.status is AnswerStatus.ANSWERED
    assert _bullets(result.answer) == [
        "- Saran pertama. [RPT0002]",
        "- Saran kedua. [RPT0002]",
        "- Saran ketiga. [RPT0002]",
    ]
    assert result.debug.dropped_points == 0


def test_the_id_route_still_stops_at_the_total_cap():
    pipeline, _ = _id_route_pipeline(
        [_payload([(f"Saran {index}.", ["LAP1"]) for index in range(1, 10)])]
    )

    result = pipeline.run("Apa saja saran pengembangan yang tercantum pada RPT0002?")

    assert len(_bullets(result.answer)) == config.ANSWER_MAX_BULLETS == 6
    assert result.debug.dropped_points == 3


def test_a_single_report_context_raises_the_per_report_cap():
    pipeline, _ = _structured_pipeline(
        [
            _payload(
                [
                    ("Poin satu.", ["LAP1"]),
                    ("Poin dua.", ["LAP1"]),
                    ("Poin tiga.", ["LAP1"]),
                ]
            )
        ],
        matches=[_match("RPT9001")],
    )

    result = pipeline.run("Apa saja kendala tim pada laporan ini?")

    assert len(_bullets(result.answer)) == 3
    assert result.debug.dropped_points == 0


def test_the_ordinary_breadth_cap_still_applies_with_several_reports():
    pipeline, _ = _structured_pipeline(
        [
            _payload(
                [
                    ("Poin satu.", ["LAP1"]),
                    ("Poin dua.", ["LAP1"]),
                    ("Poin tiga.", ["LAP1"]),
                ]
            )
        ],
        matches=[_match("RPT9001"), _match("RPT9002")],
    )

    result = pipeline.run("Apa kendala tim pada business plan?")

    assert len(_bullets(result.answer)) == config.ANSWER_MAX_BULLETS_PER_REPORT == 2
    assert result.debug.dropped_points == 1


def test_the_structured_addendum_states_the_single_report_exception():
    addendum = prompts.STRUCTURED_FORMAT_ADDENDUM

    assert "SETIAP butir" in addendum
    assert prompts.STRUCTURED_ADDENDUM_VERSION == "3.1"


# -- R2: contrasts by reference --------------------------------------------
GROUP_KEYS = {"g1": AWARD_LABEL, "g2": FINALIST_LABEL}
ALIAS_MAP = {
    "LAP1": "RPT0001",
    "LAP2": "RPT0002",
    "LAP3": "RPT0003",
    "LAP4": "RPT0004",
}
REPORT_GROUP = {
    "RPT0001": AWARD_LABEL,
    "RPT0002": FINALIST_LABEL,
    "RPT0003": FINALIST_LABEL,
    "RPT0004": FINALIST_LABEL,
}


def _validate(payload, *, keys=None, alias_map=None, report_group=None):
    alias_map = alias_map or ALIAS_MAP
    return compare_answer.validate_payload(
        payload,
        group_keys=keys or GROUP_KEYS,
        alias_map=alias_map,
        report_group=report_group or REPORT_GROUP,
        retrieved=list(alias_map.values()),
    )


def _compare_payload(award_points, finalist_points, contrasts=None):
    return {
        "groups": {"g1": award_points, "g2": finalist_points},
        "contrasts": contrasts or [],
    }


def test_the_contrast_schema_picks_a_number_per_group():
    schema = compare_answer.build_schema(["g1", "g2"])

    contrast = schema["properties"]["contrasts"]["items"]
    assert contrast["additionalProperties"] is False
    assert set(contrast["required"]) == {"dimension", "pick"}
    pick = contrast["properties"]["pick"]
    assert pick["required"] == ["g1", "g2"]
    assert pick["additionalProperties"] is False
    assert pick["properties"]["g2"] == {"type": "integer"}
    assert "sides" not in contrast["properties"]


def test_a_contrast_renders_the_picked_points_with_nesting():
    validation = _validate(
        _compare_payload(
            [{"text": "Validasi lewat wawancara.", "cite": ["LAP1"]}],
            [{"text": "Riset terbatas pada data sekunder.", "cite": ["LAP2"]}],
            [{"dimension": "Kedalaman validasi", "pick": {"g1": 1, "g2": 1}}],
        )
    )

    rendered = compare_answer.render_answer(validation, "achievement_group")

    assert rendered.endswith(
        "**Perbedaan utama**\n\n"
        "- Kedalaman validasi\n"
        f"  - {AWARD_HEADING}: Validasi lewat wawancara. [RPT0001]\n"
        f"  - {FINALIST_HEADING}: Riset terbatas pada data sekunder. [RPT0002]"
    )
    assert "LAP" not in rendered


def test_a_pick_on_a_dropped_point_drops_the_contrast_with_a_reason():
    validation = _validate(
        _compare_payload(
            [{"text": "Validasi lewat wawancara.", "cite": ["LAP1"]}],
            [
                {"text": "Tanpa sumber.", "cite": []},
                {"text": "Riset terbatas.", "cite": ["LAP2"]},
            ],
            [{"dimension": "Kedalaman validasi", "pick": {"g1": 1, "g2": 1}}],
        )
    )

    assert validation.contrasts == []
    assert validation.incidents["contrast_invalid_pick_dropped"] == 1
    assert validation.incidents["contrast_invalid"] == 1


def test_an_out_of_range_pick_drops_the_contrast():
    validation = _validate(
        _compare_payload(
            [{"text": "Validasi lewat wawancara.", "cite": ["LAP1"]}],
            [{"text": "Riset terbatas.", "cite": ["LAP2"]}],
            [{"dimension": "Kedalaman validasi", "pick": {"g1": 2, "g2": 1}}],
        )
    )

    assert validation.contrasts == []
    assert validation.incidents["contrast_invalid_pick_range"] == 1


def test_three_groups_pick_one_point_each():
    keys = {"g1": "Business Plan", "g2": "Business Case", "g3": "Essay"}
    alias_map = {
        "LAP1": "RPT0001",
        "LAP2": "RPT0002",
        "LAP3": "RPT0003",
    }
    report_group = {
        "RPT0001": "Business Plan",
        "RPT0002": "Business Case",
        "RPT0003": "Essay",
    }
    validation = _validate(
        {
            "groups": {
                "g1": [{"text": "Kendala bahan baku.", "cite": ["LAP1"]}],
                "g2": [{"text": "Kendala data sekunder.", "cite": ["LAP2"]}],
                "g3": [{"text": "Kendala waktu penulisan.", "cite": ["LAP3"]}],
            },
            "contrasts": [
                {"dimension": "Sumber kendala", "pick": {"g1": 1, "g2": 1, "g3": 1}}
            ],
        },
        keys=keys,
        alias_map=alias_map,
        report_group=report_group,
    )

    assert len(validation.contrasts) == 1
    assert [side.label for side in validation.contrasts[0].sides] == [
        "Business Plan",
        "Business Case",
        "Essay",
    ]


def test_a_generic_dimension_gets_the_neutral_label_and_is_counted():
    validation = _validate(
        _compare_payload(
            [{"text": "Validasi lewat wawancara.", "cite": ["LAP1"]}],
            [{"text": "Riset terbatas.", "cite": ["LAP2"]}],
            [{"dimension": "kendala", "pick": {"g1": 1, "g2": 1}}],
        )
    )

    assert validation.contrasts[0].dimension == "Perbandingan"
    assert validation.incidents["generic_dimension"] == 1
    assert "contrast_invalid" not in validation.incidents


def test_an_empty_dimension_gets_the_neutral_label():
    validation = _validate(
        _compare_payload(
            [{"text": "Validasi lewat wawancara.", "cite": ["LAP1"]}],
            [{"text": "Riset terbatas.", "cite": ["LAP2"]}],
            [{"dimension": " ", "pick": {"g1": 1, "g2": 1}}],
        )
    )

    assert validation.contrasts[0].dimension == config.COMPARE_NEUTRAL_DIMENSION
    assert validation.incidents["generic_dimension"] == 1


@pytest.mark.parametrize(
    "dimension",
    ["Kendala Keuangan", "Kendala Teknis", "Kendala Waktu", "Pembelajaran tim"],
)
def test_a_dimension_is_compared_as_a_whole_string(dimension):
    # The Q4 regression: word-level matching rejected "Kendala Keuangan" because
    # it contains "kendala".
    assert not compare_answer._is_generic_dimension(dimension)

    validation = _validate(
        _compare_payload(
            [{"text": "Validasi lewat wawancara.", "cite": ["LAP1"]}],
            [{"text": "Riset terbatas.", "cite": ["LAP2"]}],
            [{"dimension": dimension, "pick": {"g1": 1, "g2": 1}}],
        )
    )

    assert validation.contrasts[0].dimension == dimension
    assert "generic_dimension" not in validation.incidents


def test_bracketed_spaced_and_lowercase_aliases_are_accepted():
    validation = _validate(
        _compare_payload(
            [{"text": "Validasi lewat wawancara.", "cite": ["[LAP1]", " lap2 "]}],
            [{"text": "Riset terbatas.", "cite": ["[Lap3]"]}],
            [{"dimension": "Kedalaman validasi", "pick": {"g1": 1, "g2": 1}}],
        )
    )

    assert validation.sections[0].points[0].report_ids == ["RPT0001"]
    assert validation.incidents["cross_group_citation"] == 1
    assert validation.sections[1].points[0].report_ids == ["RPT0003"]
    assert validation.contrasts[0].sides[1].report_ids == ["RPT0003"]


def test_the_structured_alias_normalization_ignores_brackets_and_case():
    payload = {
        "found": True,
        "points": [{"text": "Tim memvalidasi masalah.", "cite": ["[lap1]"]}],
    }

    validation = structured_answer.validate_payload(
        payload, alias_map={"LAP1": "RPT0001"}, retrieved=["RPT0001"]
    )

    assert validation.points[0].report_ids == ["RPT0001"]


def test_the_q4_shape_now_renders_all_three_contrasts():
    """The real Q4 payload shape, converted to picks (offline replay)."""
    validation = _validate(
        _compare_payload(
            [
                {"text": "Kelayakan finansial kurang digali.", "cite": ["LAP1"]},
                {"text": "Validasi pasar terbatas.", "cite": ["LAP1"]},
                {"text": "Tim kesulitan mengatur waktu.", "cite": ["LAP1"]},
            ],
            [
                {"text": "Data sekunder dominan.", "cite": ["LAP2"]},
                {"text": "Regulasi OJK belum dipetakan.", "cite": ["LAP3"]},
                {"text": "Anggaran prototipe terbatas.", "cite": ["LAP3"]},
            ],
            [
                {"dimension": "Kendala Keuangan", "pick": {"g1": 1, "g2": 3}},
                {"dimension": "Kendala Teknis", "pick": {"g1": 2, "g2": 1}},
                {"dimension": "Kendala Waktu", "pick": {"g1": 3, "g2": 2}},
            ],
        )
    )

    # All three contrasts are valid; the configured cap of two truncates the
    # third instead of dropping it as invalid.
    assert [item.dimension for item in validation.contrasts] == [
        "Kendala Keuangan",
        "Kendala Teknis",
    ]
    assert validation.incidents["contrast_truncated"] == 1
    assert "contrast_invalid" not in validation.incidents
    rendered = compare_answer.render_answer(validation, "competition_type")
    assert "- Kendala Keuangan" in rendered
    assert "- Kendala Teknis" in rendered
    assert "  - Business Plan: Regulasi OJK belum dipetakan. [RPT0003]" not in rendered


def test_the_group_points_are_reported_numbered_with_their_drop_reason():
    validation = _validate(
        _compare_payload(
            [{"text": "Validasi lewat wawancara.", "cite": ["LAP1"]}],
            [
                {"text": "Tanpa sumber.", "cite": []},
                {"text": "Riset terbatas.", "cite": ["LAP2"]},
            ],
        )
    )

    assert [
        (item["index"], item["kept"], item["reason"])
        for item in validation.points_by_key["g2"]
    ] == [(1, False, "point_uncited"), (2, True, "")]


# -- R3: wording metrics (non-blocking) ------------------------------------
def test_the_causal_metric_counts_mengakibatkan_and_berakibat():
    text = (
        "- Biaya pengeringan tinggi mengakibatkan margin tipis. [RPT0001]\n"
        "- Keterlambatan berakibat pada pengurangan waktu. [RPT0001]\n"
        "- Riset pasar menjadi kendala. [RPT0001]"
    )

    result = causal_connectors(text)

    assert result["count"] == 2


def test_interpretive_phrases_are_measured_and_keep_samples():
    text = (
        "- Pengalaman prototipe mengajarkan tim tentang pentingnya uji ketahanan. "
        "[RPT0001]\n"
        "- Tim kesulitan mengatur waktu. [RPT0001]"
    )

    result = wording.interpretive_phrases(text)

    assert result["count"] == 1
    assert result["samples"][0].startswith("Pengalaman prototipe")


def test_strong_verbs_are_measured():
    result = wording.strong_verbs(
        "- Tim mengabaikan kelayakan finansial. [RPT0001]\n"
        "- Tim kurang menggali kelayakan finansial. [RPT0001]"
    )

    assert result["count"] == 1


def test_support_overlap_flags_a_point_that_the_source_does_not_support():
    points = [
        ("Tim gagal melakukan uji ketahanan prototipe.", ["RPT0001"]),
        ("Margin laba tipis akibat biaya pengeringan buatan.", ["RPT0001"]),
    ]
    report_texts = {
        "RPT0001": "Margin laba tipis akibat tingginya biaya pengeringan buatan."
    }

    result = wording.support_overlap(points, report_texts)

    assert result["per_point"][0]["low_support"] is True
    assert result["per_point"][1]["low_support"] is False
    assert result["low_support_points"] == 1
    assert result["minimum"] == result["per_point"][0]["overlap"]


def test_support_overlap_cannot_catch_a_verb_strength_error():
    # Documented limitation: every content word is in the source, so the lexical
    # metric scores the point as supported. Only a human can rate it.
    points = [("Tim mengabaikan kelayakan finansial.", ["RPT0001"])]
    report_texts = {
        "RPT0001": "Tim kurang menggali aspek kelayakan finansial serta regulasi OJK."
    }

    result = wording.support_overlap(points, report_texts)

    assert result["per_point"][0]["overlap"] >= config.SUPPORT_OVERLAP_MIN
    assert result["low_support_points"] == 0


def test_the_wording_metrics_do_not_change_the_answer():
    pipeline, _ = _structured_pipeline(
        [
            _payload(
                [
                    ("Pengalaman prototipe mengajarkan pentingnya uji ketahanan.", ["LAP1"]),
                    ("Riset pasar terbatas pada data sekunder.", ["LAP2"]),
                ]
            )
        ],
        matches=[_match("RPT9001"), _match("RPT9002")],
    )

    result = pipeline.run("Apa pembelajaran tim pada business plan?")

    assert result.status is AnswerStatus.ANSWERED
    assert "- Pengalaman prototipe mengajarkan pentingnya uji ketahanan. [RPT9001]" in (
        result.answer
    )
    assert result.debug.wording["interpretive"]["count"] == 1
    assert "support" in result.debug.to_dict()


# -- R4: reports_cited -----------------------------------------------------
def test_reports_cited_counts_distinct_reports_in_the_answer():
    pipeline, _ = _structured_pipeline(
        [
            _payload(
                [
                    ("Poin satu.", ["LAP1"]),
                    ("Poin dua.", ["LAP2"]),
                    ("Poin tiga.", ["LAP2"]),
                ]
            )
        ],
        matches=[_match("RPT9001"), _match("RPT9002")],
    )

    result = pipeline.run("Apa kendala tim pada business plan?")

    assert result.debug.reports_cited == 2
    assert result.debug.to_dict()["reports_cited"] == 2


def test_the_structured_addendum_carries_the_breadth_hint():
    addendum = prompts.STRUCTURED_FORMAT_ADDENDUM

    assert "sebar poin ke laporan yang BERBEDA" in addendum


# -- R6: smoke output ------------------------------------------------------
def test_the_smoke_output_numbers_the_group_points(capsys):
    from scripts.smoke_groq import print_compare_points

    print_compare_points(
        {"g1": AWARD_LABEL},
        {
            "g1": [
                {"index": 1, "text": "Poin juara.", "cites": ["RPT0001"], "kept": True, "reason": ""},
                {"index": 2, "text": "Poin gugur.", "cites": [], "kept": False, "reason": "point_uncited"},
            ]
        },
    )

    output = capsys.readouterr().out
    assert f"  g1 = {AWARD_LABEL}" in output
    assert "    1. Poin juara. [RPT0001] [kept]" in output
    assert "    2. Poin gugur.  [dropped:point_uncited]" in output
