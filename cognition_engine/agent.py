"""
Agent Loop — The core execution engine (TASK-023).

The agent receives a task, plans steps, executes them using tools,
and monitors its own cognitive health throughout.
"""

from __future__ import annotations

import os
import subprocess
from typing import List, Dict

from mlx_lm import generate

from cognition_engine.chi import CHIMonitor
from cognition_engine.compaction import ContextCompactor
from cognition_engine.anchors import AnchorManager


class CognitionAgent:
    """
    A coding agent with cognitive health monitoring.

    Tools available:
    - read_file(path) — read file contents
    - write_file(path, content) — write/overwrite a file
    - edit_file(path, old_text, new_text) — find and replace in a file
    - execute(command) — run a shell command (requires user approval)
    - think(thought) — record a reasoning step (no action)
    """

    def __init__(
        self,
        model,
        tokenizer,
        task: str,
        working_dir: str = ".",
        auto_approve: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.task = task
        self.working_dir = working_dir
        self.auto_approve = auto_approve

        # Core systems
        self.chi_monitor = CHIMonitor(model, tokenizer, task)
        self.compactor = ContextCompactor(model, tokenizer)
        self.anchor_mgr = AnchorManager(task)

        self.compactor.set_anchor(task)

        # Tool call history
        self.tool_history: List[Dict] = []
        self.turn_count = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, max_turns: int = 50):
        """
        Main agent loop.

        Each turn:
        1. Build context (with anchors if CHI is low)
        2. Generate model response
        3. Parse tool calls from response
        4. Execute tool calls
        5. Update CHI
        6. Check if compaction needed
        7. Repeat until task done or max_turns
        """
        print(f"Starting agent on task: {self.task}")
        print(f"Working directory: {os.path.abspath(self.working_dir)}\n")

        while self.turn_count < max_turns:
            # 1. Check if anchor needed
            if self.chi_monitor.chi_history:
                chi = self.chi_monitor.chi_history[-1]
                anchor = self.anchor_mgr.get_anchor_for_chi(chi)
                if anchor:
                    self.compactor.add_turn("system", anchor)

            # 2. Build context and generate
            context = self.compactor._build_context()

            # Add generation prompt
            context += "\nAssistant: "
            response = generate(
                self.model,
                self.tokenizer,
                prompt=context,
                max_tokens=800,
            )

            # 3. Parse and execute tool calls
            tool_calls = self._parse_tool_calls(response)
            tool_results = []

            for tool_call in tool_calls:
                result = self._execute_tool(tool_call)
                tool_results.append(result)
                self.tool_history.append(tool_call)

            # 4. Add to conversation
            self.compactor.add_turn("assistant", response)
            if tool_results:
                results_text = "\n".join(tool_results)
                self.compactor.add_turn(
                    "system", f"[Tool Results]\n{results_text}"
                )

            # 5. Update CHI
            chi_result = self.chi_monitor.update(response, context + response)
            level = self.chi_monitor.get_intervention_level()

            # 6. Display status
            chi = chi_result["chi_score"]
            self._print_status(chi, level, tool_calls)

            # 7. Check compaction
            if self.compactor.needs_compaction(chi):
                if chi < 0.3:
                    self.compactor.compact("anchor_reset")
                    print("  [COMPACTION: anchor reset]")
                elif chi < 0.5:
                    self.compactor.compact("summarize")
                    print("  [COMPACTION: summarize]")
                else:
                    self.compactor.compact("trim")
                    print("  [COMPACTION: trim]")

            # 8. Check if model says it's done
            if self._is_task_complete(response):
                print("\n Agent reports task complete.")
                break

            self.turn_count += 1

        self._print_summary()

    # ------------------------------------------------------------------
    # Tool parsing and dispatch
    # ------------------------------------------------------------------
    def _parse_tool_calls(self, response: str) -> List[Dict]:
        """
        Parse tool calls from the model's response.

        Expected simple formats (one tool call per response):
        - [READ] path
        - [WRITE] path\\n```\\ncontent\\n```
        - [EDIT] path\\n```old\\n...\\n```new\\n...\\n```
        - [EXEC] command
        - [THINK] free-form reasoning

        Returns a list with 0 or 1 tool call dicts.
        """
        text = response.strip()
        if not text.startswith("["):
            return []

        # Only parse the first bracketed directive; ignore trailing prose.
        first_line = text.splitlines()[0].strip()

        if first_line.startswith("[READ]"):
            path = first_line[len("[READ]") :].strip()
            if not path:
                return []
            return [{"tool": "read", "path": path}]

        if first_line.startswith("[EXEC]"):
            command = first_line[len("[EXEC]") :].strip()
            if not command:
                return []
            return [{"tool": "execute", "command": command}]

        if first_line.startswith("[THINK]"):
            thought = first_line[len("[THINK]") :].strip()
            return [{"tool": "think", "thought": thought}]

        if first_line.startswith("[WRITE]"):
            # [WRITE] path on first line; content between first pair of ``` fences
            lines = text.splitlines()
            path = lines[0][len("[WRITE]") :].strip()
            if not path:
                return []
            content = self._extract_code_block(text)
            if content is None:
                return []
            return [{"tool": "write", "path": path, "content": content}]

        if first_line.startswith("[EDIT]"):
            lines = text.splitlines()
            path = lines[0][len("[EDIT]") :].strip()
            if not path:
                return []
            old_text, new_text = self._extract_edit_blocks(text)
            if old_text is None:
                return []
            return [
                {
                    "tool": "edit",
                    "path": path,
                    "old": old_text,
                    "new": new_text,
                }
            ]

        return []

    def _execute_tool(self, tool_call: Dict) -> str:
        """Execute a single tool call and return the result."""
        tool = tool_call.get("tool")

        if tool == "read":
            return self._tool_read(tool_call["path"])
        if tool == "write":
            return self._tool_write(tool_call["path"], tool_call["content"])
        if tool == "edit":
            return self._tool_edit(
                tool_call["path"],
                tool_call["old"],
                tool_call["new"],
            )
        if tool == "execute":
            return self._tool_execute(tool_call["command"])
        if tool == "think":
            return f"[Thought recorded: {tool_call['thought'][:100]}]"
        return f"[Unknown tool: {tool}]"

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------
    def _tool_read(self, path: str) -> str:
        """Read a file and return its contents."""
        full_path = os.path.join(self.working_dir, path)
        try:
            with open(full_path) as f:
                content = f.read()
            self.anchor_mgr.add_file(path)
            return f"[READ {path}]\n{content[:2000]}"  # Truncate large files
        except Exception as e:  # noqa: BLE001
            return f"[ERROR reading {path}: {e}]"

    def _tool_write(self, path: str, content: str) -> str:
        """Write content to a file."""
        full_path = os.path.join(self.working_dir, path)
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(content)
            self.anchor_mgr.add_file(path)
            return f"[WROTE {path} ({len(content)} chars)]"
        except Exception as e:  # noqa: BLE001
            return f"[ERROR writing {path}: {e}]"

    def _tool_edit(self, path: str, old_text: str, new_text: str) -> str:
        """Find and replace in a file."""
        full_path = os.path.join(self.working_dir, path)
        try:
            with open(full_path) as f:
                content = f.read()
            if old_text not in content:
                return f"[ERROR: '{old_text[:50]}...' not found in {path}]"
            content = content.replace(old_text, new_text, 1)
            with open(full_path, "w") as f:
                f.write(content)
            self.anchor_mgr.add_file(path)
            return f"[EDITED {path}]"
        except Exception as e:  # noqa: BLE001
            return f"[ERROR editing {path}: {e}]"

    def _tool_execute(self, command: str) -> str:
        """Execute a shell command with user approval."""
        if not self.auto_approve:
            approval = input(f"  Execute: `{command}`? [y/N] ")
            if approval.lower() != "y":
                return "[EXEC cancelled by user]"
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.working_dir,
                timeout=30,
            )
            output = result.stdout[:1000]
            if result.returncode != 0:
                output += f"\n[STDERR: {result.stderr[:500]}]"
            return f"[EXEC `{command}` → exit {result.returncode}]\n{output}"
        except subprocess.TimeoutExpired:
            return "[EXEC `{command}` → TIMEOUT (30s)]"
        except Exception as e:  # noqa: BLE001
            return f"[EXEC ERROR: {e}]"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _is_task_complete(self, response: str) -> bool:
        """Check if the model indicates the task is done."""
        markers = [
            "[DONE]",
            "[TASK COMPLETE]",
            "task is complete",
            "all requirements have been met",
        ]
        lower = response.lower()
        return any(m.lower() in lower for m in markers)

    def _print_status(self, chi: float, level: int, tool_calls: List[Dict]):
        """Print turn status."""
        color = (
            "\033[92m" if chi > 0.8 else "\033[93m" if chi > 0.5 else "\033[91m"
        )
        tools_str = (
            ", ".join(tc.get("tool", "?") for tc in tool_calls)
            if tool_calls
            else "none"
        )
        print(
            f"  Turn {self.turn_count:3d} | {color}CHI {chi:.3f} "
            f"L{level}\033[0m | Tools: {tools_str}"
        )

    def _print_summary(self):
        """Print session summary."""
        print("\n--- Session Summary ---")
        print(f"Turns: {self.turn_count}")
        print(f"Compactions: {self.compactor.compaction_count}")
        print(f"Files modified: {self.anchor_mgr.files_modified}")
        if self.chi_monitor.chi_history:
            print(
                f"CHI: min={min(self.chi_monitor.chi_history):.3f} "
                f"final={self.chi_monitor.chi_history[-1]:.3f}"
            )

    @staticmethod
    def _extract_code_block(text: str) -> str | None:
        """Extract content from the first ``` code block."""
        start = text.find("```")
        if start == -1:
            return None
        end = text.find("```", start + 3)
        if end == -1:
            return None
        return text[start + 3 : end].strip()

    @staticmethod
    def _extract_edit_blocks(text: str) -> tuple[str | None, str | None]:
        """Extract old and new text from ```old and ```new blocks."""
        # Expect pattern: ```old ... ```new ... ```
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

        # The final ``` after new block
        end_new = text.find("```", start_new_content)
        if end_new == -1:
            return None, None

        new_block = text[start_new_content:end_new]

        return old_block.strip(), new_block.strip()


if __name__ == "__main__":
    # Manual test for tools (not full agent loop).
    from cognition_engine.inference import load

    model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")

    test_dir = "/tmp/test_agent"
    os.makedirs(test_dir, exist_ok=True)

    agent = CognitionAgent(
        model, tokenizer, "test task", working_dir=test_dir, auto_approve=True
    )

    # Test write
    print(agent._tool_write("test.py", "print('hello')"))

    # Test read
    print(agent._tool_read("test.py"))

    # Test edit
    print(agent._tool_edit("test.py", "hello", "world"))

    # Test read again to verify edit
    print(agent._tool_read("test.py"))

    # Test execute
    print(agent._tool_execute("python test.py"))

