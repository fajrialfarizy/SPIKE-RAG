"""
Offline tests for fix pack 12 (T1): the contrast switch.

The real run paired non-comparable points, so CONTRAST="off" is the default: the
compare schema has no contrasts property, the addendum carries no contrast
instruction, the renderer writes no "Perbedaan utama" section, and the contrast
counters stay 0. "pick" and "sides" stay selectable by config only.

FakeLLM and the offline doubles come from tests/stubs.py and
scripts/offline_compare_endtoend.py: no network, no API key.
"""

import importlib.util
import json
from pathlib import Path

import pytest

import config
from core import compare_answer, prompts, schema_check
from core.pipeline import AnswerStatus

AWARD_LABEL = "meraih penghargaan"
FINALIST_LABEL = "belum juara (finalis)"
AWARD_HEADING = (
    "Tim yang meraih penghargaan (Juara 1-3, Juara Harapan, Most Aspiring)"
)
FINALIST_HEADING = "Tim belum juara (finalis)"


def _load_script(name: str):
    path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate(payload, mode):
    return compare_answer.validate_payload(
        payload,
        group_keys={"g1": AWARD_LABEL, "g2": FINALIST_LABEL},
        alias_map={"LAP1": "RPT0001", "LAP2": "RPT0002"},
        report_group={"RPT0001": AWARD_LABEL, "RPT0002": FINALIST_LABEL},
        retrieved=["RPT0001", "RPT0002"],
        mode=mode,
    )


# -- T1a: the switch -------------------------------------------------------
def test_off_is_the_default_and_the_values_stay_explicit():
    assert config.COMPARE_CONTRAST_MODE == "off"
    assert config.COMPARE_CONTRAST_MODE_VALUES == ("off", "pick", "sides")
    assert compare_answer.contrast_mode() == "off"
    assert compare_answer.contrast_mode("pick") == "pick"
    assert compare_answer.contrast_mode("sides") == "sides"


def test_an_unknown_mode_is_still_a_loud_error():
    with pytest.raises(ValueError, match="COMPARE_CONTRAST_MODE"):
        compare_answer.contrast_mode("guess")


# -- T1a: the schema -------------------------------------------------------
@pytest.mark.parametrize("keys", [["g1", "g2"], ["g1", "g2", "g3"]])
def test_the_off_schema_has_no_contrasts_property(keys):
    schema = compare_answer.build_schema(keys, "off")

    assert "contrasts" not in schema["properties"]
    assert schema["required"] == ["groups"]
    assert schema["properties"]["groups"]["required"] == keys
    assert schema_check.check_schema(schema) == []


@pytest.mark.parametrize("keys", [["g1", "g2"], ["g1", "g2", "g3"]])
def test_every_off_response_format_is_within_the_documented_subset(keys):
    response_format = compare_answer.response_format(keys, "off")

    assert schema_check.check_response_format(response_format) == []
    assert "contrast" not in json.dumps(response_format)


def test_pick_and_sides_keep_their_contrasts_property():
    for mode in ("pick", "sides"):
        schema = compare_answer.build_schema(["g1", "g2"], mode)

        assert "contrasts" in schema["properties"]
        assert schema["required"] == ["groups", "contrasts"]


# -- T1a: validation and rendering ----------------------------------------
def test_off_ignores_a_contrasts_key_and_keeps_every_counter_at_zero():
    payload = {
        "groups": {
            "g1": [{"text": "Validasi lewat wawancara.", "cite": ["LAP1"]}],
            "g2": [{"text": "Riset terbatas pada data sekunder.", "cite": ["LAP2"]}],
        },
        "contrasts": [
            {"dimension": "Kedalaman validasi", "pick": {"g1": "1", "g2": "1"}}
        ],
    }

    validation = _validate(payload, "off")

    assert validation.contrasts == []
    assert validation.incidents == {}
    assert validation.cited_report_ids == ["RPT0001", "RPT0002"]
    assert (
        "Kedalaman validasi"
        not in compare_answer.render_answer(validation, "achievement_group")
    )


def test_off_renders_exactly_the_two_group_blocks_and_nothing_else():
    validation = compare_answer.CompareValidation(
        sections=[
            compare_answer.CompareSection(
                label=AWARD_LABEL,
                points=[
                    compare_answer.ComparePoint(
                        text="Validasi lewat wawancara.", report_ids=["RPT0001"]
                    )
                ],
            ),
            compare_answer.CompareSection(
                label=FINALIST_LABEL,
                points=[
                    compare_answer.ComparePoint(
                        text="Riset terbatas pada data sekunder.",
                        report_ids=["RPT0002"],
                    )
                ],
            ),
        ],
        contrasts=[],
    )

    rendered = compare_answer.render_answer(validation, "achievement_group")

    assert rendered == (
        f"**{AWARD_HEADING}**\n\n"
        "- Validasi lewat wawancara. [RPT0001]\n\n"
        f"**{FINALIST_HEADING}**\n\n"
        "- Riset terbatas pada data sekunder. [RPT0002]"
    )
    assert config.COMPARE_CONTRASTS_HEADING not in rendered


# -- T1a: the prompt -------------------------------------------------------
def test_the_off_addendum_carries_no_contrast_instruction():
    addendum = prompts.compare_addendum("off")
    lowered = addendum.casefold()

    assert prompts.COMPARE_ADDENDUM_VERSION == "2.5"
    assert "(v2.5)" in addendum
    assert "contrast" not in lowered
    assert "pick" not in lowered
    assert "dimensi" not in lowered
    assert '"contrasts"' not in addendum
    # The fidelity rules are untouched by the switch.
    assert '"kurang menggali" bukan "mengabaikan"' in addendum
    assert "Jangan menyambung dua fakta" in addendum


def test_the_pick_addendum_still_carries_the_contrast_rules():
    addendum = prompts.compare_addendum("pick")

    assert '"contrasts"' in addendum
    assert "paling banyak 3 poin per kelompok, 2 contrasts" in addendum
    assert "poin yang benar-benar sebanding" in addendum


def test_every_mode_keeps_the_prd_core_verbatim():
    for mode in config.COMPARE_CONTRAST_MODE_VALUES:
        prompt = prompts.compare_system_prompt(mode)

        assert prompt.startswith(prompts.PRD_9_1_SYSTEM_PROMPT)
        assert prompt == (
            f"{prompts.PRD_9_1_SYSTEM_PROMPT}\n\n{prompts.compare_addendum(mode)}"
        )
    assert prompts.SYSTEM_PROMPT_COMPARE == prompts.compare_system_prompt()


# -- T1a: the whole comparison path in off mode ----------------------------
@pytest.mark.parametrize("case", ["juara_vs_belum_juara", "compare_by_type"])
def test_off_answers_both_compare_questions_without_a_contrast_section(case):
    result, llm = _load_script("offline_compare_endtoend").run_case(case, "off")

    assert result.status is AnswerStatus.ANSWERED, result.debug.error_type
    assert llm.schemas_checked == ["compare_answer"]
    assert result.cited_report_ids
    assert config.COMPARE_CONTRASTS_HEADING not in result.answer
    assert "LAP" not in result.answer
