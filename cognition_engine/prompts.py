"""
Prompt templates and tool-call parser for the Cognition Engine agent (TASK-024).
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict


SYSTEM_PROMPT = """You are Cognition Engine, an AI coding assistant running locally.
You have access to these tools:

[READ] path — Read a file
[WRITE] path
```
file content here
```
[EDIT] path
```old
text to find
```new
replacement text
```
[EXEC] command — Run a shell command
[THINK] your reasoning — Record a thought (no action taken)
[DONE] — Signal that the task is complete

Rules:
- Use [THINK] before complex decisions
- Use [READ] before editing a file you haven't seen
- One tool call per response. Wait for the result before the next action.
- When the task is fully complete and tested, say [DONE]
- Be precise with file paths. Use relative paths from the project root.
"""


def build_system_prompt(task: str, project_context: str | None = None) -> str:
    """Build the full system prompt for a task."""
    prompt = SYSTEM_PROMPT + f"\n\nYour current task: {task}\n"
    if project_context:
        prompt += f"\nProject context:\n{project_context}\n"
    return prompt


def parse_tool_call(response: str) -> Dict | None:
    """
    Parse a single tool call from model response.

    Returns dict with tool type and parameters, or None if no tool call found.
    """
    if not response:
        return None

    text = response.strip()

    # [DONE] has priority: we treat it as task completion
    if "[DONE]" in text:
        return {"tool": "done"}

    # Normalize to first non-empty line for directive
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    first = lines[0].strip()

    # [READ] path
    if first.startswith("[READ]"):
        path = first[len("[READ]") :].strip()
        if not path:
            return None
        return {"tool": "read", "path": path}

    # [EXEC] command
    if first.startswith("[EXEC]"):
        command = first[len("[EXEC]") :].strip()
        if not command:
            return None
        return {"tool": "execute", "command": command}

    # [THINK] thought
    if first.startswith("[THINK]"):
        thought = first[len("[THINK]") :].strip()
        return {"tool": "think", "thought": thought}

    # [WRITE] path\n```\ncontent\n```
    if first.startswith("[WRITE]"):
        path = first[len("[WRITE]") :].strip()
        if not path:
            return None
        content = _extract_code_block(text)
        if content is None:
            return None
        return {"tool": "write", "path": path, "content": content}

    # [EDIT] path\n```old\n...\n```new\n...\n```
    if first.startswith("[EDIT]"):
        path = first[len("[EDIT]") :].strip()
        if not path:
            return None
        old_text, new_text = _extract_edit_blocks(text)
        if old_text is None:
            return None
        return {"tool": "edit", "path": path, "old": old_text, "new": new_text}

    return None


def _extract_code_block(text: str) -> Optional[str]:
    """Extract content from first ``` code block."""
    start = text.find("```")
    if start == -1:
        return None
    end = text.find("```", start + 3)
    if end == -1:
        return None
    return text[start + 3 : end].strip()


def _extract_edit_blocks(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract old and new text from ```old and ```new blocks."""
    marker_old = "```old"
    marker_new = "```new"

    start_old = text.find(marker_old)
    if start_old == -1:
        return None, None
    start_old_content = text.find("\n", start_old)
    if start_old_content == -1:
        return None, None

    start_new = text.find(marker_new, start_old_content)
    if start_new == -1:
        return None, None

    old_block = text[start_old_content:start_new]

    start_new_content = text.find("\n", start_new)
    if start_new_content == -1:
        return None, None

    end_new = text.find("```", start_new_content)
    if end_new == -1:
        return None, None

    new_block = text[start_new_content:end_new]

    return old_block.strip(), new_block.strip()


if __name__ == "__main__":
    # Quick manual tests for the parser
    assert parse_tool_call("[READ] src/main.py") == {
        "tool": "read",
        "path": "src/main.py",
    }
    assert parse_tool_call("[EXEC] python -m pytest") == {
        "tool": "execute",
        "command": "python -m pytest",
    }
    think = parse_tool_call("[THINK] I need to check the schema")
    assert think and think["tool"] == "think" and think["thought"].startswith(
        "I need"
    )
    assert parse_tool_call("[DONE]") == {"tool": "done"}

    write_resp = "[WRITE] test.py\n```\nprint('hello')\n```"
    parsed = parse_tool_call(write_resp)
    assert parsed and parsed["tool"] == "write"
    assert parsed["path"] == "test.py"
    assert "hello" in parsed["content"]

    edit_resp = "[EDIT] test.py\n```old\nhello\n```new\nworld\n```"
    parsed_edit = parse_tool_call(edit_resp)
    assert parsed_edit and parsed_edit["tool"] == "edit"
    assert parsed_edit["path"] == "test.py"
    assert parsed_edit["old"] == "hello"
    assert parsed_edit["new"] == "world"

    print("prompts.py manual parser tests passed.")

