"""
Offline tests for the CLI surface: answer printing, --debug output, and the two
startup failures (no API key, unusable knowledge base).
"""

import json

import pytest

import config
import core.cli as cli
from core.knowledge_base import ChunkMatch
from core.pipeline import RagPipeline
from tests.stubs import FakeKnowledgeBase, FakeLLM, RecordingEmbedder


def fake_pipeline():
    metadata = {
        "report_id": "RPT0002",
        "competition_name": "IBMN 2023",
        "competition_type": "Business Plan",
        "competition_year": 2023,
        "achievement_level": "Finalis",
        "section_type": "evaluasi_tim",
        "chunk_id": "RPT0002:evaluasi_tim:01",
    }
    kb = FakeKnowledgeBase(
        matches=[
            ChunkMatch(
                chunk_id="RPT0002:evaluasi_tim:01",
                report_id="RPT0002",
                document="Isi evaluasi.",
                metadata=metadata,
                distance=0.2,
            )
        ]
    )
    llm = FakeLLM(
        json_responses=[
            json.dumps(
                {
                    "in_scope": True,
                    "mode": "specific",
                    "search_query": "kendala",
                    "filters": {"competition_type": ["Business Plan"]},
                }
            )
        ],
        text_responses=["Kendala logistik [S1]."],
    )
    return RagPipeline(knowledge_base=kb, llm=llm, embedder=RecordingEmbedder())


def test_cli_prints_the_answer_only_by_default(monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_pipeline", lambda **kwargs: fake_pipeline())

    exit_code = cli.main(["Apa kendala tim?"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "[RPT0002]" in out
    assert "**Sumber**" in out
    assert "DEBUG" not in out
    assert "alias_map" not in out


def test_cli_debug_flag_prints_the_backend_object(monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_pipeline", lambda **kwargs: fake_pipeline())

    cli.main(["Apa kendala tim?", "--debug"])

    out = capsys.readouterr().out
    assert "--- DEBUG (backend only) ---" in out
    assert "status: ANSWERED" in out
    assert '"mode": "specific"' in out
    assert '"filters"' in out
    assert '"alias_map"' in out


def test_cli_without_api_key_exits_with_a_clear_message(monkeypatch, capsys):
    monkeypatch.delenv(config.LLM_API_KEY_ENV_VAR, raising=False)

    exit_code = cli.main(["Apa kendala tim?"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert config.LLM_API_KEY_ENV_VAR in captured.err
    assert "DEBUG" not in captured.out


def test_cli_without_a_knowledge_base_exits_with_a_clear_message(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setenv(config.LLM_API_KEY_ENV_VAR, "dummy-key-for-test")

    exit_code = cli.main(
        ["Apa kendala tim?", "--chroma-dir", str(tmp_path / "missing")]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "does not exist" in captured.err
