"""ContextCompactor tests with a fake tokenizer (no model)."""

import pytest

from cognition_engine.compaction import ContextCompactor


class FakeTokenizer:
    def encode(self, text):
        return text.split()


def make(max_tokens=50):
    c = ContextCompactor(model=None, tokenizer=FakeTokenizer(), max_context_tokens=max_tokens)
    c.set_anchor("build a thing")
    return c


def fill(c, n):
    for i in range(n):
        c.add_turn("user", f"user step {i}")
        c.add_turn("assistant", f"assistant step {i}")


def test_needs_compaction_on_length_or_low_chi():
    c = make(max_tokens=10)
    fill(c, 10)
    assert c.needs_compaction()
    c2 = make(max_tokens=10_000)
    fill(c2, 2)
    assert not c2.needs_compaction()
    assert not c2.needs_compaction(0.9)
    assert c2.needs_compaction(0.5)


def test_trim_keeps_anchor_and_last_turns():
    c = make()
    fill(c, 10)
    c.compact("trim")
    assert len(c.conversation_turns) == 5
    assert c.conversation_turns[-1]["content"] == "assistant step 9"
    assert "build a thing" in c._build_context()
    assert c.compaction_count == 1
    assert c.last_compaction_strategy == "trim"


def test_anchor_reset_keeps_two_turns():
    c = make()
    fill(c, 10)
    c.compact("anchor_reset")
    assert len(c.conversation_turns) == 2


def test_auto_is_noop_when_under_limit():
    c = make(max_tokens=10_000)
    fill(c, 3)
    c.compact("auto")
    assert c.compaction_count == 0
    assert len(c.conversation_turns) == 6


def test_auto_shrinks_when_over_limit():
    c = make(max_tokens=40)
    fill(c, 30)
    c.compact("auto")
    assert c.compaction_count == 1
    assert len(c.conversation_turns) <= 5


def test_summarize_failure_falls_back_to_trim():
    c = make()  # model=None; generate is a stub that is not callable
    fill(c, 10)
    c.compact("summarize")
    assert len(c.conversation_turns) == 5


def test_raw_turns_start_with_anchor():
    c = make()
    c.add_turn("user", "hi")
    turns = c.get_raw_turns()
    assert turns[0] == ("system", "build a thing")
    assert turns[1] == ("user", "hi")
