"""
Offline end-to-end check of the comparison path (fix pack 11, S5).

Runs the two real comparison questions through the FULL pipeline:

- the real ``Retriever`` and ``ContextBuilder`` over an in-memory knowledge base
  built from the dummy corpus' metadata,
- the real ``TokenAwareRateLimiter`` (word-count tokenizer, fake clock), so the
  reservation and the recording path are exercised,
- the real comparison validator, with every schema checked against the
  documented Groq rules by ``core.schema_check`` before it is sent,
- a scripted LLM (no network, no key), with payloads converted from the points
  the earlier real run reported.

Both ``COMPARE_CONTRAST_MODE`` values are exercised. In "pick" mode the model
answers with the fix pack 10 payload and the backend renders the validated point
text; in "sides" mode the same run's points are converted to the fix pack 9 side
shape, which is known to work on the real model.

No model, no network, no API key.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from core import schema_check
from core.knowledge_base import ChunkMatch
from core.llm_client import LLMClient, LLMResult, TokenUsage
from core.pipeline import AnswerStatus, RagPipeline
from core.rate_limiter import TokenAwareRateLimiter
from tests.stubs import WordCountTokenizer


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += max(0.0, float(seconds))


def _matches_where(metadata: Dict[str, Any], where: Optional[Dict[str, Any]]) -> bool:
    """Evaluate the subset of Chroma's where syntax the retriever emits."""
    if not where:
        return True
    if "$and" in where:
        return all(_matches_where(metadata, clause) for clause in where["$and"])
    for field, condition in where.items():
        if not isinstance(condition, dict) or "$in" not in condition:
            raise AssertionError(f"Unsupported where clause: {condition!r}")
        value = metadata.get(field)
        allowed = list(condition["$in"])
        if field == "competition_year":
            value = None if value is None else int(value)
            allowed = [int(item) for item in allowed]
        if value not in allowed:
            return False
    return True


class InMemoryKnowledgeBase:
    """Minimal knowledge base: retrieval, id lookup and the catalog counters."""

    def __init__(self, matches: List[ChunkMatch]) -> None:
        self.matches = list(matches)

    def count_chunks(self) -> int:
        return len(self.matches)

    def query(self, query_embedding, where=None, n_results=1):
        matching = [item for item in self.matches if _matches_where(item.metadata, where)]
        return matching[:n_results]

    def get_by_report_ids(self, report_ids):
        return [item for item in self.matches if item.report_id in report_ids]

    def list_report_metadatas(self, where=None, limit=1000):
        seen: List[Dict[str, Any]] = []
        report_ids = set()
        for item in self.matches:
            if item.report_id in report_ids:
                continue
            report_ids.add(item.report_id)
            seen.append(item.metadata)
        return seen[:limit]


class ScriptedCompareLLM(LLMClient):
    """
    Scripted analyst + comparison answer, through the real token limiter.

    The analyzer returns the validated filters the real run produced; the
    comparison call returns the payload for the requested contrast mode.
    """

    def __init__(
        self,
        limiter: TokenAwareRateLimiter,
        clock: FakeClock,
        *,
        analyzer: Dict[str, Any],
        pick_payload: Dict[str, Any],
        sides_payload: Dict[str, Any],
    ) -> None:
        self.limiter = limiter
        self.clock = clock
        self.analyzer = analyzer
        self.pick_payload = pick_payload
        self.sides_payload = sides_payload
        self.schemas_checked: List[str] = []

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        reasoning_effort: str,
        json_mode: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        deadline: Optional[Any] = None,
        queue_max_wait_seconds: Optional[float] = None,
    ) -> LLMResult:
        static_extra = (
            json.dumps(response_format, ensure_ascii=False, sort_keys=True)
            if response_format
            else ""
        )
        estimate = self.limiter.estimate_tokens(
            system_prompt, user_prompt, min(max_tokens, 400), static_extra=static_extra
        )
        entry = self.limiter.reserve(estimate, max_wait_seconds=queue_max_wait_seconds)
        self.clock.advance(1.0)
        self.limiter.record(entry, 500)

        if response_format is not None:
            schema_check.assert_valid_response_format(response_format)
            self.schemas_checked.append(config.COMPARE_JSON_SCHEMA_NAME)
            payload = (
                self.pick_payload
                if config.COMPARE_CONTRAST_MODE == "pick"
                else self.sides_payload
            )
            content = json.dumps(payload, ensure_ascii=False)
        else:
            content = json.dumps(self.analyzer, ensure_ascii=False)
        return LLMResult(
            content=content,
            usage=TokenUsage(prompt_tokens=estimate, completion_tokens=500),
            finish_reason="stop",
        )


class ScriptedAnalyzer:
    """Analyzer double that returns the filters of the real run."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self.payload = payload

    def analyze(self, question: str, deadline=None):
        from core.analyzer import validate_analysis

        return validate_analysis(dict(self.payload), question)


def _match(report_id: str, *, level: str, competition_type: str, text: str) -> ChunkMatch:
    metadata = {
        "report_id": report_id,
        "competition_name": f"Kompetisi {report_id}",
        "competition_type": competition_type,
        "competition_year": 2023,
        "achievement_level": level,
        "section_type": "evaluasi_tim",
        "chunk_id": f"{report_id}:evaluasi_tim:01",
    }
    return ChunkMatch(
        chunk_id=metadata["chunk_id"],
        report_id=report_id,
        document=text,
        metadata=metadata,
        distance=0.1,
    )


def _sides_from_pick(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a pick payload to the fix pack 9 side shape with the same points."""
    converted = {"groups": payload["groups"], "contrasts": []}
    for contrast in payload["contrasts"]:
        sides = []
        for key, number in contrast["pick"].items():
            point = payload["groups"][key][number - 1]
            sides.append(
                {"group": key, "text": point["text"], "cite": list(point["cite"])}
            )
        converted["contrasts"].append(
            {"dimension": contrast["dimension"], "sides": sides}
        )
    return converted


CASES: Dict[str, Dict[str, Any]] = {
    "juara_vs_belum_juara": {
        "question": (
            "Apa perbedaan pembelajaran dari tim yang memperoleh juara dan tim "
            "yang belum juara?"
        ),
        "analyzer": {
            "in_scope": True,
            "mode": "cross_report",
            "search_query": "pembelajaran tim juara",
            "filters": {},
            "compare_by": "achievement_group",
        },
        "matches": [
            _match(
                "RPT9001",
                level="Juara 2",
                competition_type="Business Plan",
                text="Tim juara memvalidasi masalah lewat wawancara pengguna.",
            ),
            _match(
                "RPT9002",
                level="Finalis",
                competition_type="Business Plan",
                text="Tim finalis lebih banyak memakai data sekunder.",
            ),
        ],
        "payload": {
            "groups": {
                "g1": [
                    {
                        "text": "Tim juara memvalidasi masalah lewat wawancara pengguna.",
                        "cite": ["LAP1"],
                    }
                ],
                "g2": [
                    {
                        "text": "Tim finalis lebih banyak memakai data sekunder.",
                        "cite": ["LAP2"],
                    }
                ],
            },
            "contrasts": [
                {"dimension": "Pembelajaran", "pick": {"g1": 1, "g2": 1}}
            ],
        },
    },
    "compare_by_type": {
        "question": "Bandingkan kendala yang dialami tim business plan dan business case.",
        "analyzer": {
            "in_scope": True,
            "mode": "cross_report",
            "search_query": "kendala tim business plan business case",
            "filters": {"competition_type": ["Business Case", "Business Plan"]},
            "compare_by": "competition_type",
        },
        "matches": [
            _match(
                "RPT9101",
                level="Finalis",
                competition_type="Business Plan",
                text="Kelayakan finansial kurang digali. Validasi pasar terbatas.",
            ),
            _match(
                "RPT9102",
                level="Juara 2",
                competition_type="Business Case",
                text="Data sekunder dominan. Regulasi OJK belum dipetakan.",
            ),
        ],
        "payload": {
            "groups": {
                "g1": [
                    {"text": "Kelayakan finansial kurang digali.", "cite": ["LAP1"]},
                    {"text": "Validasi pasar terbatas.", "cite": ["LAP1"]},
                ],
                "g2": [
                    {"text": "Data sekunder dominan.", "cite": ["LAP2"]},
                    {"text": "Regulasi OJK belum dipetakan.", "cite": ["LAP2"]},
                ],
            },
            "contrasts": [
                {"dimension": "Kendala Keuangan", "pick": {"g1": 1, "g2": 1}},
                {"dimension": "Kendala Waktu", "pick": {"g1": 2, "g2": 2}},
            ],
        },
    },
}


def run_case(name: str, mode: str):
    """Run one comparison question through the whole pipeline."""
    case = CASES[name]
    previous = config.COMPARE_CONTRAST_MODE
    config.COMPARE_CONTRAST_MODE = mode
    try:
        clock = FakeClock()
        limiter = TokenAwareRateLimiter(
            clock=clock, sleep=clock.advance, token_counter=WordCountTokenizer()
        )
        llm = ScriptedCompareLLM(
            limiter,
            clock,
            analyzer=case["analyzer"],
            pick_payload=case["payload"],
            sides_payload=_sides_from_pick(case["payload"]),
        )
        pipeline = RagPipeline(
            knowledge_base=InMemoryKnowledgeBase(case["matches"]),
            llm=llm,
            analyzer=ScriptedAnalyzer(case["analyzer"]),
            embedder=_HashEmbedder(),
            token_counter=WordCountTokenizer(),
            clock=clock,
        )
        result = pipeline.run(case["question"])
    finally:
        config.COMPARE_CONTRAST_MODE = previous
    return result, llm


class _HashEmbedder:
    """Deterministic unit vector; retrieval is driven by the fake knowledge base."""

    model_name = "offline-hash"

    def embed_query(self, text: str) -> List[float]:
        return [0.0] * config.EMBEDDING_DIMENSION


def main() -> int:
    print(f"compare_contrast_mode default={config.COMPARE_CONTRAST_MODE}")
    for mode in config.COMPARE_CONTRAST_MODE_VALUES:
        print(f"\n=== mode={mode}")
        for name in CASES:
            result, llm = run_case(name, mode)
            schemas = sorted(set(llm.schemas_checked))
            print(f"\n--- {name}: status={result.status.value}")
            print(f"schemas_checked={schemas}")
            print(f"citations={result.cited_report_ids}")
            print(f"incidents={json.dumps(result.debug.incidents, sort_keys=True)}")
            print(f"compare_points={json.dumps(result.debug.compare_points, sort_keys=True)}")
            if result.status is not AnswerStatus.ANSWERED:
                print(f"error_type={result.debug.error_type}")
                print(result.answer)
                continue
            print(result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
