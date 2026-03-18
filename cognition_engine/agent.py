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
from cognition_engine.memory import WorkingMemory
from cognition_engine.prompts import build_system_prompt, parse_tool_call
from cognition_engine.session import SessionManager


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
        self.memory = WorkingMemory(working_dir)
        self.session_mgr = SessionManager(working_dir)
        self.chi_monitor = CHIMonitor(model, tokenizer, task)
        self.compactor = ContextCompactor(model, tokenizer)
        self.anchor_mgr = AnchorManager(task)

        # Seed the context with a full system prompt that teaches the tool format.
        system_anchor = build_system_prompt(task)
        self.compactor.set_anchor(system_anchor)

        # Inject any existing project memory into the context so the agent can
        # build on previous sessions.
        memory_context = self.memory.get_context_for_task(task)
        if memory_context:
            self.compactor.add_turn("system", memory_context)

        # If there is an interrupted session, inject resume context so the model
        # can continue without re-discovering state.
        if self.session_mgr.has_interrupted_session():
            resume_ctx = self.session_mgr.get_resume_context()
            if resume_ctx:
                self.compactor.add_turn("system", resume_ctx)

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

        status = "interrupted"
        completed_explicitly = False
        executed_action_tool = False
        try:
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

                # Treat explicit [DONE] as completion. Do not rely on natural-language
                # "task is complete" phrases, since those can appear prematurely and
                # would incorrectly clear resumable session state.
                parsed_top = parse_tool_call(response)
                if parsed_top and parsed_top.get("tool") == "done":
                    # Guardrail: do not allow immediate [DONE] before any real tool
                    # has been executed. This prevents a resumed session from being
                    # cleared due to a premature completion.
                    if not executed_action_tool:
                        self.compactor.add_turn("assistant", response)
                        self.compactor.add_turn(
                            "system",
                            "[IGNORED DONE] Model emitted [DONE] before executing any "
                            "tool. Continue with the task using exactly one tool call.",
                        )
                        # Do not mark completed; continue loop.
                        self.turn_count += 1
                        continue
                    self.compactor.add_turn("assistant", response)
                    status = "completed"
                    completed_explicitly = True
                    print("\n Agent reports task complete. ([DONE])")
                    break

                # 3. Parse and execute tool calls
                tool_calls = self._parse_tool_calls(response)
                tool_results = []

                for tool_call in tool_calls:
                    result = self._execute_tool(tool_call)
                    tool_results.append(result)
                    self.tool_history.append(tool_call)
                    if tool_call.get("tool") in {"read", "write", "edit", "execute"}:
                        executed_action_tool = True

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

                self.turn_count += 1
        except KeyboardInterrupt:
            print("\n[KeyboardInterrupt] Stopping agent loop.")
            status = "interrupted"
        except EOFError:
            # Can happen when stdin is non-interactive (e.g. piped execution)
            # and a tool requests approval via input().
            print("\n[EOFError] Input stream closed; stopping agent loop.")
            status = "interrupted"
        finally:
            # Persist session state for resume/status inspection.
            self.session_mgr.save_session(
                self.task,
                self.compactor.conversation_turns,
                self.chi_monitor.chi_history,
                self.turn_count,
                self.anchor_mgr.files_modified,
                status=status,
            )
            if status == "completed" and completed_explicitly:
                self.session_mgr.clear_session()

            # On exit (normal or interrupted), write a context note to memory.
            chi_final = (
                self.chi_monitor.chi_history[-1]
                if self.chi_monitor.chi_history
                else None
            )
            context_note = (
                f"Session ended. {self.turn_count} turns. "
                f"Final CHI: {chi_final:.3f}. "
                f"Files: {self.anchor_mgr.files_modified}"
                if chi_final is not None
                else f"Session ended. {self.turn_count} turns. "
                f"Files: {self.anchor_mgr.files_modified}"
            )
            self.memory.add_note(
                "context",
                context_note,
                related_files=self.anchor_mgr.files_modified,
            )

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
        parsed = parse_tool_call(response)
        if not parsed:
            return []
        # The agent loop treats [DONE] specially at a higher level; we don't
        # execute it as a tool call.
        if parsed.get("tool") == "done":
            return []
        # Exactly one tool call per response.
        return [parsed]

    def _execute_tool(self, tool_call: Dict) -> str:
        """Execute a single tool call and return the result."""
        tool = tool_call.get("tool")

        if tool == "read":
            return self._tool_read(tool_call["path"])
        if tool == "write":
            result = self._tool_write(tool_call["path"], tool_call["content"])
            # Record file purpose in working memory.
            preview = tool_call["content"][:100]
            self.memory.add_note(
                "file_purpose",
                f"Created or updated {tool_call['path']}: {preview}",
                related_files=[tool_call["path"]],
            )
            return result
        if tool == "edit":
            return self._tool_edit(
                tool_call["path"],
                tool_call["old"],
                tool_call["new"],
            )
        if tool == "execute":
            return self._tool_execute(tool_call["command"])
        if tool == "think":
            thought = tool_call["thought"]
            self.memory.add_note(
                "decision",
                thought,
                related_files=self.anchor_mgr.files_modified,
            )
            return f"[Thought recorded: {thought[:100]}]"
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

