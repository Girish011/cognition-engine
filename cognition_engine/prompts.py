"""
Prompt templates and tool-call parser for the Cognition Engine agent (TASK-024).
"""

from __future__ import annotations

import re
from typing import Optional, Tuple, Dict

# Qwen / chat models often append template tokens to the end of generations (e.g. <|im_end|>).
_CHAT_TEMPLATE_BRACKET_RE = re.compile(r"<\|[^|]*\|>")


def strip_chat_template_artifacts(s: str) -> str:
    """
    Remove chat-template noise from model output before parsing tool arguments (TASK-045H).
    Strips <|im_end|> and any <|...|> tokens (e.g. <|im_start|>).
    Applied separately to path vs file body: paths keep valid ``subdir/file`` layout;
    code bodies only lose template suffixes, not directory structure.
    """
    if not s:
        return s
    t = s.replace("<|im_end|>", "")
    t = _CHAT_TEMPLATE_BRACKET_RE.sub("", t)
    return t.strip()


SYSTEM_PROMPT = """You are Cognition Engine, an AI coding assistant running locally.
You have access to these tools and MUST use them with the exact bracketed format:

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
- Use [THINK] before complex decisions.
- NEVER guess file contents. Always [READ] a file before you describe, edit, or rely on it.
- Use [READ] before editing a file you haven't seen in this session.
- CRITICAL: ONE tool call per response, nothing more. Stop after that tool. Wait for the result.
- Do NOT combine [READ]+[EDIT]+[EXEC]+[DONE] in a single response. Each goes in its own turn.
- The tool call should be the main content of your response, not buried inside prose.
- When the task is fully complete and tested, say [DONE] on its own line.
- Be precise with file paths. Use relative paths from the project root.

COMPLETION RULE: When you have created all requested files AND the last
test/command succeeded (exit code 0), respond with only [DONE].
Do NOT continue working after tests pass. Do NOT re-run tests that
already passed. Say [DONE] and stop.

IMPORTANT: Only say [DONE] after `python -m pytest` returns exit code 0.

Each example below is ONE complete response. Study the format:

Response 1:
[THINK] I should inspect the file before editing it.

Response 2:
[READ] calculator.py

Response 3:
[WRITE] calculator.py
```
class Calculator:
    def add(self, a, b):
        return a + b
```

Response 4:
[EXEC] python -m pytest

Response 5:
[DONE]
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

    text = strip_chat_template_artifacts(response.strip())

    # [DONE] has priority: we treat it as task completion (case-insensitive)
    if "[DONE]" in text.upper():
        return {"tool": "done"}

    # Normalize to first non-empty line that looks like a directive
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None

    directive_line = None
    directive_index = 0
    for idx, ln in enumerate(lines):
        stripped = ln.strip()
        if stripped.startswith("["):
            directive_line = stripped
            directive_index = idx
            break
    if not directive_line:
        return None

    first = directive_line

    # Parse directive within brackets, tolerating variants like [Read], [READ]:, [READ] - path
    if "]" not in first:
        return None
    end_bracket = first.find("]")
    directive = first[1:end_bracket].strip().upper()
    rest = first[end_bracket + 1 :].lstrip(" :-").strip()

    # [READ] path
    if directive == "READ":
        path = rest
        if not path:
            # Fall back to next line if present
            if directive_index + 1 < len(lines):
                path = lines[directive_index + 1].strip()
        if not path:
            return None
        return {"tool": "read", "path": strip_chat_template_artifacts(path)}

    # [EXEC] command
    if directive == "EXEC":
        command = rest
        if not command and directive_index + 1 < len(lines):
            command = lines[directive_index + 1].strip()
        if not command:
            return None
        return {"tool": "execute", "command": strip_chat_template_artifacts(command)}

    # [THINK] thought
    if directive == "THINK":
        thought = rest
        if not thought and directive_index + 1 < len(lines):
            thought = lines[directive_index + 1].strip()
        thought = strip_chat_template_artifacts(thought) if thought else thought
        return {"tool": "think", "thought": thought}

    # [WRITE] path\n```\ncontent\n```
    if directive == "WRITE":
        path = rest
        if not path and directive_index + 1 < len(lines):
            path = lines[directive_index + 1].strip()
        if not path:
            return None
        path = strip_chat_template_artifacts(path)
        content = _extract_code_block(text)
        if content is None:
            return None
        content = strip_chat_template_artifacts(content)
        return {"tool": "write", "path": path, "content": content}

    # [EDIT] path\n```old\n...\n```new\n...\n```
    if directive == "EDIT":
        path = rest
        if not path and directive_index + 1 < len(lines):
            path = lines[directive_index + 1].strip()
        if not path:
            return None
        path = strip_chat_template_artifacts(path)
        old_text, new_text = _extract_edit_blocks(text)
        if old_text is None:
            return None
        old_text = strip_chat_template_artifacts(old_text)
        new_text = strip_chat_template_artifacts(new_text)
        return {"tool": "edit", "path": path, "old": old_text, "new": new_text}

    return None


def _extract_code_block(text: str) -> Optional[str]:
    """Extract content from first ``` code block, stripping any language tag."""
    start = text.find("```")
    if start == -1:
        return None
    end = text.find("```", start + 3)
    if end == -1:
        return None
    raw = text[start + 3 : end]
    # If the fence has a language tag (e.g. ```python), drop that first line.
    newline = raw.find("\n")
    if newline != -1:
        tag = raw[:newline].strip()
        # A language tag contains only word chars/digits/dashes (no spaces or
        # Python syntax).  If the "tag" looks like code (has spaces, parens, etc.)
        # keep it as content.
        if tag and re.match(r"^[A-Za-z0-9_+\-\.]+$", tag):
            raw = raw[newline + 1 :]
    return strip_chat_template_artifacts(raw.strip())


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
    assert parse_tool_call("[EXEC] python -m pytest<|im_end|>") == {
        "tool": "execute",
        "command": "python -m pytest",
    }
    assert parse_tool_call("[READ] calculator.py<|im_end|>") == {
        "tool": "read",
        "path": "calculator.py",
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

