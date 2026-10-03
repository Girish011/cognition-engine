"""Pure helper tests for cognition_engine.agent (mlx is stubbed off Apple Silicon)."""

import pytest

from cognition_engine.agent import (
    CognitionAgent,
    _command_looks_like_test_run,
    _parse_tool_call_prefer_exec_over_done,
    _tool_call_signature,
    _truncate_at_stop_strings,
)
from cognition_engine.escalation import build_escalation_prompt, build_escalator


@pytest.mark.parametrize(
    "cmd,expected",
    [
        ("python -m pytest -q", True),
        ("cd proj && pytest", True),
        ("python test_calc.py", True),
        ("python -m unittest", True),
        ("pip install pytest", False),
        ("python -m pip install pytest", False),
        ("cat pytest.ini", False),
        ("", False),
    ],
)
def test_command_looks_like_test_run(cmd, expected):
    assert _command_looks_like_test_run(cmd) is expected


def test_exec_is_recovered_when_packed_with_done():
    parsed = _parse_tool_call_prefer_exec_over_done("[EXEC] python -m pytest\n[DONE]")
    assert parsed == {"tool": "execute", "command": "python -m pytest"}


def test_plain_done_stays_done():
    assert _parse_tool_call_prefer_exec_over_done("[DONE]") == {"tool": "done"}


def test_edit_with_plain_code_block_is_treated_as_write():
    resp = "[EDIT] a.py\n```python\nx = 1\n```\n[DONE]"
    parsed = _parse_tool_call_prefer_exec_over_done(resp)
    assert parsed["tool"] == "write" and parsed["content"] == "x = 1"


def test_truncate_at_stop_strings():
    assert _truncate_at_stop_strings("abc<|im_end|>junk", ["<|im_end|>"]) == "abc"
    assert _truncate_at_stop_strings("abc", ["<|im_end|>"]) == "abc"
    assert _truncate_at_stop_strings("a END b STOP c", ["STOP", "END"]) == "a "


def test_tool_call_signature_distinguishes_arguments():
    a = _tool_call_signature({"tool": "write", "path": "a", "content": "1"})
    b = _tool_call_signature({"tool": "write", "path": "a", "content": "2"})
    assert a != b
    assert a == _tool_call_signature({"tool": "write", "path": "a", "content": "1"})


# --- Level 4 escalation -------------------------------------------------------------


class FakeEscalator:
    def __init__(self, advice="try X", error=None):
        self.advice, self.error, self.calls = advice, error, 0

    def advise(self, task, recent_turns, files):
        self.calls += 1
        if self.error:
            raise self.error
        return self.advice


class _Compactor:
    def __init__(self):
        self.conversation_turns = []

    def add_turn(self, role, content):
        self.conversation_turns.append({"role": role, "content": content})


class _Anchors:
    files_modified = ["a.py"]


def make_agent(escalator):
    agent = CognitionAgent.__new__(CognitionAgent)  # skip model-dependent __init__
    agent.escalator = escalator
    agent._last_escalation_turn = None
    agent.turn_count = 10
    agent.task = "task"
    agent.compactor = _Compactor()
    agent.anchor_mgr = _Anchors()
    return agent


def test_escalation_injects_advice_below_threshold():
    esc = FakeEscalator()
    agent = make_agent(esc)
    assert agent._maybe_escalate(0.2) is True
    assert "try X" in agent.compactor.conversation_turns[-1]["content"]


def test_no_escalation_when_healthy_or_unconfigured():
    esc = FakeEscalator()
    agent = make_agent(esc)
    assert agent._maybe_escalate(0.6) is False and esc.calls == 0
    assert make_agent(None)._maybe_escalate(0.1) is False


def test_escalation_respects_cooldown():
    esc = FakeEscalator()
    agent = make_agent(esc)
    assert agent._maybe_escalate(0.1) is True
    agent.turn_count += 1
    assert agent._maybe_escalate(0.1) is False
    agent.turn_count += 10
    assert agent._maybe_escalate(0.1) is True
    assert esc.calls == 2


def test_escalation_failure_does_not_crash_the_loop():
    agent = make_agent(FakeEscalator(error=RuntimeError("network")))
    assert agent._maybe_escalate(0.1) is False


def test_steering_mode_is_validated():
    with pytest.raises(ValueError):
        CognitionAgent(None, None, "t", steering_mode="sometimes")


def test_escalation_prompt_is_bounded_and_includes_task():
    turns = [{"role": "assistant", "content": "x" * 5000} for _ in range(5)]
    prompt = build_escalation_prompt("fix the bug", turns, ["a.py"])
    assert "fix the bug" in prompt and "a.py" in prompt
    assert len(prompt) < 8000


def test_build_escalator_names():
    assert build_escalator(None) is None
    assert build_escalator("none") is None
    with pytest.raises(ValueError):
        build_escalator("gpt")
