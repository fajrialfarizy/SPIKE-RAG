"""
Token delta of the fix pack 10 prompt/schema change (offline, exact tokenizer).

Compares the fix pack 9 addenda and comparison schema (commit 46656e1) with the
current ones. The old prompts module is read from git and executed with the
current config, so both sides are counted by the same tokenizer.

    python scripts/measure_fixpack10_tokens.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from core import compare_answer, prompts, structured_answer
from core.tokens import E5TokenCounter

BASE_COMMIT = "46656e1"


def load_old_prompts() -> ModuleType:
    source = subprocess.check_output(
        ["git", "show", f"{BASE_COMMIT}:core/prompts.py"], text=True
    )
    # The old addendum formats with constants fix pack 10 removed from config.
    if not hasattr(config, "COMPARE_CONTRAST_TEXT_MAX_WORDS"):
        config.COMPARE_CONTRAST_TEXT_MAX_WORDS = 35
    module = ModuleType("old_prompts")
    module.__dict__["__name__"] = "old_prompts"
    # Registered before exec, because a dataclass resolves its module by name.
    sys.modules[module.__name__] = module
    exec(compile(source, "old_prompts.py", "exec"), module.__dict__)
    return module


def load_old_compare() -> ModuleType:
    """The fix pack 8/9 comparison module, for its real schema."""
    source = subprocess.check_output(
        ["git", "show", f"{BASE_COMMIT}:core/compare_answer.py"], text=True
    )
    module = ModuleType("old_compare_answer")
    module.__dict__["__name__"] = "old_compare_answer"
    sys.modules[module.__name__] = module
    exec(compile(source, "old_compare_answer.py", "exec"), module.__dict__)
    return module


def main() -> int:
    counter = E5TokenCounter()
    old = load_old_prompts()
    old_compare = load_old_compare()

    rows = [
        (
            "structured addendum",
            old.STRUCTURED_FORMAT_ADDENDUM,
            prompts.STRUCTURED_FORMAT_ADDENDUM,
        ),
        (
            "compare addendum",
            old.COMPARE_CITATION_ADDENDUM,
            prompts.COMPARE_CITATION_ADDENDUM,
        ),
        (
            "structured schema",
            json.dumps(structured_answer.build_schema(), sort_keys=True),
            json.dumps(structured_answer.build_schema(), sort_keys=True),
        ),
    ]

    rows.append(
        (
            "compare schema (g1,g2)",
            json.dumps(old_compare.build_schema(["g1", "g2"]), sort_keys=True),
            json.dumps(compare_answer.build_schema(["g1", "g2"]), sort_keys=True),
        )
    )

    print("item\told_tokens\tnew_tokens\tdelta")
    for name, before, after in rows:
        old_count = counter.count_tokens(before)
        new_count = counter.count_tokens(after)
        print(f"{name}\t{old_count}\t{new_count}\t{new_count - old_count:+d}")

    print()
    print(f"structured_addendum_version={prompts.STRUCTURED_ADDENDUM_VERSION}")
    print(f"compare_addendum_version={prompts.COMPARE_ADDENDUM_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
