"""
Offline replay of one two-question quota sequence (fix pack 7, M1b).

The supervisor's real run answered one specific question and then failed the
comparison question with TECHNICAL_ERROR after exactly the pipeline budget. This
script rebuilds that sequence without any provider call:

- the real retrieval and context builder (same path as show_context.py),
- the real prompt strings (the pipeline assembles them),
- the real token estimator and the real rolling-window limiter,
- a scripted provider that reports the token usage and the model latency of the
  real run,
- one fake clock shared by the deadline, the limiter and the client, so the
  windows and the queue wait advance without sleeping.

    python scripts/replay_limiter_sequence.py --chroma-dir chroma_db

Dummy corpus only. Never point it at real evaluation documents.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from core import prompts
from core.deadline import Deadline
from core.knowledge_base import KnowledgeBase, KnowledgeBaseError
from core.llm_client import GroqLLMClient, LLMConcurrencyLimiter
from core.pipeline import RagPipeline
from core.rate_limiter import TokenAwareRateLimiter
from core.tokens import E5TokenCounter
from ingestion.embedder import E5Embedder

SPECIFIC_QUESTION = (
    "Apa saja pengalaman tim Lab Prestasi ketika mengikuti kompetisi business plan?"
)
COMPARE_QUESTION = (
    "Apa perbedaan pembelajaran dari tim yang memperoleh juara dan tim yang belum juara?"
)

# The usage the supervisor's run reported, in total tokens.
ANALYZER_TOTAL_Q1 = 1239
ANSWER_TOTAL_Q1 = 3451
ANALYZER_TOTAL_Q2 = 1257
COMPARE_TOTAL_Q2 = 4138


class FakeClock:
    """One monotonic clock for the deadline, the limiter and the client."""

    def __init__(self, start: float = 0.0):
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)

    def jump_to(self, value: float) -> None:
        self.now = float(value)


@dataclass
class Step:
    """One scripted provider answer with the usage and latency of the real run."""

    kind: str  # "analyzer" | "answer" | "compare"
    content: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int = 0
    latency_seconds: float = 1.0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _response(content: str, prompt_tokens: int, completion_tokens: int,
              reasoning_tokens: int) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, reasoning=None),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            completion_tokens_details=SimpleNamespace(
                reasoning_tokens=reasoning_tokens
            ),
            prompt_tokens_details=SimpleNamespace(cached_tokens=None),
        ),
    )


@dataclass
class ScriptedCompletions:
    """Answers calls from a fixed script, spending the fake clock as it goes."""

    clock: FakeClock
    script: List[Step]
    requests: List[Dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def _kind(kwargs: Dict[str, Any]) -> str:
        response_format = kwargs.get("response_format") or {}
        if response_format.get("type") == "json_schema":
            # Fix pack 9: the free-text answer is structured too, so the schema
            # name decides which scripted step it draws.
            name = (response_format.get("json_schema") or {}).get("name")
            return "answer" if name == config.ANSWER_JSON_SCHEMA_NAME else "compare"
        if response_format.get("type") == "json_object":
            return "analyzer"
        return "answer"

    def create(self, **kwargs: Any) -> Any:
        kind = self._kind(kwargs)
        if not self.script or self.script[0].kind != kind:
            expected = self.script[0].kind if self.script else "(empty script)"
            raise AssertionError(f"unexpected {kind} call, expected {expected}")

        step = self.script.pop(0)
        messages = kwargs.get("messages") or []
        content = step.content
        user_text = "\n".join(
            item["content"] for item in messages if item["role"] == "user"
        )
        if content == "__COMPARE__":
            content = compare_payload(kwargs, user_text)
        elif content == "__ANSWER__":
            content = answer_payload(user_text)
        self.requests.append({
            "kind": kind,
            "max_completion_tokens": kwargs.get("max_completion_tokens"),
            "system_chars": sum(
                len(item["content"]) for item in messages if item["role"] == "system"
            ),
            "user_chars": sum(
                len(item["content"]) for item in messages if item["role"] == "user"
            ),
            "at_seconds": round(self.clock.now, 2),
        })
        self.clock.advance(step.latency_seconds)
        return _response(
            content,
            step.prompt_tokens,
            step.completion_tokens,
            step.reasoning_tokens,
        )


@dataclass
class ScriptedGroq:
    completions: ScriptedCompletions

    @property
    def chat(self) -> Any:
        return SimpleNamespace(completions=self.completions)


def analyzer_payload(mode: str, compare_by: Optional[str], query: str) -> str:
    return json.dumps({
        "in_scope": True,
        "mode": mode,
        "search_query": query,
        "filters": {
            "competition_type": [],
            "competition_year": [],
            "achievement_level": [],
        },
        "compare_by": compare_by,
    })


def compare_payload(kwargs: Dict[str, Any], user_text: str) -> str:
    """A valid keyed comparison answer built from the prompt's own aliases."""
    schema = (kwargs.get("response_format") or {}).get("json_schema", {}).get(
        "schema", {}
    )
    keys = schema.get("properties", {}).get("groups", {}).get("required", [])

    alias_groups: Dict[str, List[str]] = {}
    for key in keys:
        match = re.search(
            rf"^- {re.escape(key)}: (.+)$", user_text, re.MULTILINE
        )
        alias_groups[key] = (
            re.findall(r"LAP[0-9]{1,3}", match.group(1)) if match else []
        )

    groups = {}
    for index, key in enumerate(keys):
        aliases = alias_groups[key]
        groups[key] = (
            [
                {
                    "text": f"Tim pada kelompok {index + 1} memperkuat validasi pasar.",
                    "cite": aliases[:1],
                }
            ]
            if aliases
            else []
        )
    return json.dumps({"groups": groups, "contrasts": []})


def answer_payload(user_text: str) -> str:
    """A valid structured free-text answer citing the prompt's first aliases."""
    aliases = list(dict.fromkeys(re.findall(r"LAP[0-9]{1,3}", user_text)))
    points = [
        {
            "text": "Tim menghadapi kendala validasi pasar dan waktu terbatas.",
            "cite": aliases[:1],
        }
    ] if aliases else []
    return json.dumps({"found": bool(points), "points": points})


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the two-question quota sequence with a fake clock."
    )
    parser.add_argument("--chroma-dir", default=str(config.CHROMA_PERSIST_DIR))
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--gap",
        type=float,
        default=33.0,
        help="Fake seconds between the end of Q1 and the start of Q2 (default 33).",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Trace every deadline and limiter decision (diagnosis output).",
    )
    return parser.parse_args(argv)


def install_estimate_recorder(limiter: TokenAwareRateLimiter) -> List[Dict[str, Any]]:
    """Record every estimator input so the components can be inspected."""
    records: List[Dict[str, Any]] = []
    original = limiter.estimate_tokens

    def wrapped(system_prompt, user_prompt, max_completion_tokens, *args, **kwargs):
        value = original(
            system_prompt, user_prompt, max_completion_tokens, *args, **kwargs
        )
        records.append({
            "system_chars": len(system_prompt or ""),
            "user_chars": len(user_prompt or ""),
            "completion_cap": int(max_completion_tokens),
            "estimate": value,
        })
        print(
            f"    [estimate] system_chars={len(system_prompt or '')} "
            f"user_chars={len(user_prompt or '')} "
            f"cap={int(max_completion_tokens)} -> {value}"
        )
        return value

    limiter.estimate_tokens = wrapped
    return records


def install_trace() -> None:
    """Print each deadline/limiter decision and the exception it raises."""
    original_sleep = Deadline.sleep
    original_require = Deadline.require_remaining
    original_reserve = TokenAwareRateLimiter.reserve
    original_slot = LLMConcurrencyLimiter.slot

    def traced_sleep(self, seconds, sleep_fn=None):
        print(f"    [trace] Deadline.sleep({seconds:.2f}) "
              f"remaining={self.remaining():.2f}")
        try:
            result = original_sleep(self, seconds, sleep_fn or (lambda _: None))
        except BaseException as exc:
            print(f"    [trace] Deadline.sleep raised {type(exc).__name__}")
            raise
        print(f"    [trace] Deadline.sleep done remaining={self.remaining():.2f}")
        return result

    def traced_require(self):
        try:
            return original_require(self)
        except BaseException as exc:
            print(f"    [trace] require_remaining raised {type(exc).__name__} "
                  f"remaining={self.remaining():.2f}")
            raise

    def traced_reserve(self, tokens, max_wait_seconds=None):
        print(f"    [trace] reserve({tokens}) used={self.snapshot()['tokens_used_this_minute']} "
              f"max_wait={max_wait_seconds}")
        try:
            entry = original_reserve(self, tokens, max_wait_seconds)
        except BaseException as exc:
            print(f"    [trace] reserve raised {type(exc).__name__}")
            raise
        print(f"    [trace] reserve granted used={self.snapshot()['tokens_used_this_minute']}")
        return entry

    class _TracedSlot:
        def __init__(self, owner):
            self.owner = owner

        def __enter__(self):
            self._cm = original_slot(self.owner, self.deadline)
            return self._cm.__enter__()

        def __exit__(self, *exc_info):
            return self._cm.__exit__(*exc_info)

    def traced_slot(self, deadline=None):
        print(f"    [trace] slot(remaining={None if deadline is None else round(deadline.remaining(), 2)})")
        traced = _TracedSlot(self)
        traced.deadline = deadline
        return traced

    Deadline.sleep = traced_sleep
    Deadline.require_remaining = traced_require
    TokenAwareRateLimiter.reserve = traced_reserve
    LLMConcurrencyLimiter.slot = traced_slot


def print_report(label: str, result: Any, requests: List[Dict[str, Any]],
                 limiter: TokenAwareRateLimiter, clock: FakeClock) -> None:
    print(f"\n=== {label}")
    print(f"status={result.status.value} error_type={result.debug.error_type} "
          f"latency_ms={result.debug.latency_ms} fake_clock={clock.now:.2f}")
    print(f"mode={result.debug.mode} route={result.debug.route} "
          f"compare_by={result.debug.compare_by}")
    for call in result.debug.calls:
        print(f"  - {call.get('name')} queue_wait_ms={call.get('queue_wait_ms')} "
              f"model_latency_ms={call.get('model_latency_ms')} "
              f"usage={json.dumps(call.get('usage') or {})}")
        if call.get("reservation"):
            print(f"    limiter={json.dumps(call['reservation'])}")
    print(f"provider_requests={json.dumps(requests)}")
    print(f"limiter_now={json.dumps(limiter.snapshot())}")


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    if args.trace:
        install_trace()

    try:
        knowledge_base = KnowledgeBase(
            persist_dir=Path(args.chroma_dir),
            manifest_path=Path(args.manifest) if args.manifest else None,
        )
    except KnowledgeBaseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    clock = FakeClock()
    limiter = TokenAwareRateLimiter(clock=clock, sleep=clock.advance)
    estimates = install_estimate_recorder(limiter)
    completions = ScriptedCompletions(clock=clock, script=[
        Step("analyzer",
             analyzer_payload("specific", None, "pengalaman tim kompetisi business plan"),
             prompt_tokens=1177, completion_tokens=ANALYZER_TOTAL_Q1 - 1177,
             latency_seconds=1.0),
        Step("answer", "__ANSWER__",
             prompt_tokens=2922, completion_tokens=ANSWER_TOTAL_Q1 - 2922,
             latency_seconds=5.0),
        Step("analyzer",
             analyzer_payload("cross_report", "achievement_group",
                              "perbandingan pembelajaran tim juara dan finalis"),
             prompt_tokens=1181, completion_tokens=ANALYZER_TOTAL_Q2 - 1181,
             latency_seconds=1.5),
        Step("compare", "__COMPARE__", prompt_tokens=3600,
             completion_tokens=COMPARE_TOTAL_Q2 - 3600, latency_seconds=4.0),
    ])
    client = GroqLLMClient(
        api_key="offline-replay",
        client=ScriptedGroq(completions),
        token_limiter=limiter,
        limiter=LLMConcurrencyLimiter(config.LLM_MAX_CONCURRENT_CALLS),
        sleep_fn=clock.advance,
        clock=clock,
    )
    counter = E5TokenCounter()
    pipeline = RagPipeline(
        knowledge_base=knowledge_base,
        llm=client,
        embedder=E5Embedder(),
        token_counter=counter,
        clock=clock,
    )

    print(f"budget_seconds={pipeline.deadline_budget_seconds} "
          f"tpm={limiter.tokens_per_minute} tpd={limiter.tokens_per_day} "
          f"gap={args.gap}s")

    request_mark = len(completions.requests)
    result_q1 = pipeline.run(SPECIFIC_QUESTION)
    print_report("Q1 specific", result_q1, completions.requests[request_mark:], limiter, clock)

    # The comparison question starts --gap seconds after Q1 finished.
    clock.jump_to(clock.now + args.gap)
    request_mark = len(completions.requests)
    try:
        result_q2 = pipeline.run(COMPARE_QUESTION)
    except BaseException:
        traceback.print_exc()
        return 1
    print_report("Q2 comparison", result_q2, completions.requests[request_mark:], limiter, clock)

    context_chars = len(result_q2.debug.context_text)
    print(f"\ncontext_chars={context_chars} "
          f"context_tokens_e5={counter.count_tokens(result_q2.debug.context_text)}")
    print(f"estimate_components: system_chars={len(prompts.SYSTEM_PROMPT_COMPARE)} "
          f"addendum_chars={len(prompts.COMPARE_CITATION_ADDENDUM)} "
          f"completion_cap={config.LLM_MAX_OUTPUT_TOKENS_COMPARE}")
    print(f"waits={limiter.waits} reserved_total={limiter.total_reserved} "
          f"recorded_total={limiter.total_recorded}")
    print("\n=== estimate components")
    for item in estimates:
        print(f"  - {json.dumps(item)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
