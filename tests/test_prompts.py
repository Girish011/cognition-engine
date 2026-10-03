"""Unit tests for the tool-call parser in cognition_engine.prompts."""

from cognition_engine.prompts import parse_tool_call, strip_chat_template_artifacts


def test_empty_response_returns_none():
    assert parse_tool_call("") is None
    assert parse_tool_call("no directive here") is None


def test_done_has_priority():
    assert parse_tool_call("[DONE] all finished") == {"tool": "done"}


def test_read_and_exec():
    assert parse_tool_call("[READ] src/app.py") == {"tool": "read", "path": "src/app.py"}
    assert parse_tool_call("[EXEC] python -m pytest -q") == {
        "tool": "execute",
        "command": "python -m pytest -q",
    }


def test_think():
    call = parse_tool_call("[THINK] plan the next step")
    assert call == {"tool": "think", "thought": "plan the next step"}


def test_write_extracts_code_block_and_drops_language_tag():
    resp = "[WRITE] hello.py\n```python\nprint('hi')\n```"
    assert parse_tool_call(resp) == {
        "tool": "write",
        "path": "hello.py",
        "content": "print('hi')",
    }


def test_write_without_code_block_is_rejected():
    assert parse_tool_call("[WRITE] hello.py\nprint('hi')") is None


def test_directive_is_case_insensitive():
    assert parse_tool_call("[read] a.txt") == {"tool": "read", "path": "a.txt"}


def test_chat_template_tokens_are_stripped():
    assert strip_chat_template_artifacts("x = 1<|im_end|>") == "x = 1"
    call = parse_tool_call("[EXEC] ls<|im_end|>")
    assert call == {"tool": "execute", "command": "ls"}
