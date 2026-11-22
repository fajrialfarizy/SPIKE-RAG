"""
Measure the schema token delta of fix pack 11 (offline, no model, no key).

The comparison call's static cost is the system prompt plus the JSON schema,
which the limiter counts with the real tokenizer. The pick schema changed from
an open integer to a string with a generated enum, and the sides shape is
available again, so this prints all three against the fix pack 10 baseline.

The baseline is loaded from the commit that introduced the pick schema
(``57e9d9b``), because the current module no longer builds it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import compare_answer, schema_check
from core.tokens import E5TokenCounter

BASELINE_COMMIT = "57e9d9b"


def _tokens(counter: E5TokenCounter, value) -> int:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return int(counter.count_tokens(text))


def _baseline_schema() -> dict:
    """The fix pack 10 pick schema, rebuilt from the committed module."""
    source = subprocess.run(
        ["git", "show", f"{BASELINE_COMMIT}:core/compare_answer.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    # The module must be registered before execution: its dataclasses look up
    # their own module in sys.modules while being created.
    module = types.ModuleType("fp10_compare")
    module.__package__ = "core"
    sys.modules["fp10_compare"] = module
    exec(compile(source, "fp10_compare_answer.py", "exec"), module.__dict__)
    return module.build_schema(["g1", "g2"])


def main() -> int:
    counter = E5TokenCounter()
    baseline = _baseline_schema()
    pick = compare_answer.build_schema(["g1", "g2"], "pick")
    sides = compare_answer.build_schema(["g1", "g2"], "sides")

    print("schema strict-subset check:")
    for name, schema in (("pick", pick), ("sides", sides)):
        print(f"  {name}: {schema_check.check_schema(schema) or 'ok'}")

    baseline_tokens = _tokens(counter, baseline)
    print("\nschema tokens (E5 tokenizer, [\"g1\", \"g2\"]):")
    for name, schema in (("fp10 pick (integer)", baseline), ("fp11 pick (string enum)", pick), ("fp11 sides", sides)):
        tokens = _tokens(counter, schema)
        print(f"  {name}: {tokens} ({tokens - baseline_tokens:+d} vs fp10)")

    print(
        "\nstatic compare overhead (system prompt + addendum + schema):"
    )
    from core import prompts

    for name, schema in (("pick", pick), ("sides", sides)):
        static = _tokens(
            counter,
            prompts.SYSTEM_PROMPT_COMPARE + json.dumps(schema, sort_keys=True),
        )
        print(f"  {name}: {static} tokens")

    print(
        "\nThe fix pack 10 real run measured about 5.7k tokens for "
        "juara_vs_belum_juara and 6.2k for compare_by_type (analyzer + compare). "
        "The delta above is the only change to that cost."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
