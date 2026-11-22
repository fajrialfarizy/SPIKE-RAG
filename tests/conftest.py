"""
Shared test setup.

Fix pack 7 made the process-wide token limiter count the static prompt parts with
the real e5 tokenizer. Loading that tokenizer costs up to a minute on a cold cache
(and several seconds even warm), which has nothing to do with what most tests
check. This fixture gives the global limiter a cheap deterministic counter for the
default suite; the real tokenizer is covered by the slow calibration test in
tests/test_fixpack7.py and by scripts/replay_limiter_sequence.py.
"""

import pytest

import config
from core.rate_limiter import get_token_limiter
from tests.stubs import WordCountTokenizer


@pytest.fixture(autouse=True)
def fast_prompt_estimator():
    limiter = get_token_limiter()
    if limiter._token_counter is None:
        limiter._token_counter = WordCountTokenizer()
    yield


@pytest.fixture
def legacy_answer_contract(monkeypatch):
    """
    The pre-fix-pack-9 contract for the older module suites.

    Those tests cover the free-prose path and the comparison validation rules as
    they were, with exactly one structured call. Fix pack 9's structured
    free-text path and repair retry are covered by tests/test_fixpack9.py.
    """
    monkeypatch.setattr(config, "ANSWER_FORMAT", "text")
    monkeypatch.setattr(config, "STRUCTURED_REPAIR_RETRIES", 0)
    yield
