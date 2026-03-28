"""
Agent Loop — The core execution engine (TASK-023).

The agent receives a task, plans steps, executes them using tools,
and monitors its own cognitive health throughout.
"""

from __future__ import annotations

import os
import subprocess
from typing import List, Dict, Optional

from mlx_lm import generate

from cognition_engine.chi import CHIMonitor
from cognition_engine.compaction import ContextCompactor
from cognition_engine.anchors import AnchorManager
from cognition_engine.memory import WorkingMemory
from cognition_engine.prompts import build_system_prompt, parse_tool_call
from cognition_engine.session import SessionManager

# ------------------------------------------------------------------
# Steering (TASK-041/042)
# ------------------------------------------------------------------
STEERING_VECTOR = "tool_format"
STEERING_LAYER = 16
STEERING_ALPHA = 0.5
STEERING_ACTIVATE_THRESHOLD = 0.7
STEERING_DEACTIVATE_THRESHOLD = 0.8


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
        *,
        enable_steering: bool = False,
        enable_compaction: bool = True,
        turn_log: Optional[List[Dict]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.task = task
        self.working_dir = working_dir
        self.auto_approve = auto_approve
        self.enable_steering = bool(enable_steering)
        self.enable_compaction = bool(enable_compaction)
        self.turn_log = turn_log

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

        # Steering state (optional)
        self.steering = None
        self.steering_active = False
        self.steering_alpha_current = 0.0
        if self.enable_steering:
            try:
                from cognition_engine.steering.injector import SteeringEngine

                self.steering = SteeringEngine(model)
                self.steering.load_vectors(
                    STEERING_VECTOR,
                    layer_idx=STEERING_LAYER,
                    alpha=STEERING_ALPHA,
                    normalize=False,
                )
                # Start inactive; CHI controls activation.
                self.steering.deactivate()
                self.steering_alpha_current = STEERING_ALPHA
                print(
                    f"[STEERING] Loaded {STEERING_VECTOR} at layer {STEERING_LAYER}, "
                    f"alpha={STEERING_ALPHA} (inactive until CHI < {STEERING_ACTIVATE_THRESHOLD})"
                )

                # If resuming an interrupted session, restore steering_active state.
                if self.session_mgr.has_interrupted_session():
                    session = self.session_mgr.load_session()
                    if session and session.steering_active:
                        self.steering.activate()
                        self.steering_active = True
                        print("[STEERING] Restored active steering from interrupted session.")
            except FileNotFoundError:
                print("[STEERING] Warning: vectors not found; steering disabled")
                self.enable_steering = False
                self.steering = None
            except Exception as e:  # noqa: BLE001
                print(f"[STEERING] Warning: failed to initialize steering; disabled. Error: {e}")
                self.enable_steering = False
                self.steering = None

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
                    if self.turn_log is not None:
                        self._append_turn_log(
                            response=response,
                            chi_result=None,
                            level=self.chi_monitor.get_intervention_level(),
                            tool_calls=[{"tool": "done"}],
                            compaction_fired=False,
                            compaction_strategy=None,
                        )
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
                if self.enable_steering and self.steering is not None:
                    from cognition_engine.steering.injector import compute_adaptive_alpha

                    # TASK-042: adaptive proportional control (P-only).
                    alpha = compute_adaptive_alpha(chi, base_alpha=STEERING_ALPHA)
                    self.steering_alpha_current = alpha

                    # Conditional activation (TASK-041): hysteresis band.
                    if chi < STEERING_ACTIVATE_THRESHOLD and not self.steering_active:
                        self.steering.set_alpha(alpha)
                        self.steering.activate()
                        self.steering_active = True
                        print(
                            f"[STEERING] ACTIVATED at turn {self.turn_count} — "
                            f"CHI={chi:.3f}, alpha={alpha:.3f}"
                        )
                    elif chi > STEERING_DEACTIVATE_THRESHOLD and self.steering_active:
                        self.steering.deactivate()
                        self.steering_active = False
                        print(f"[STEERING] Deactivated at turn {self.turn_count} — CHI={chi:.3f}")
                    elif self.steering_active:
                        self.steering.set_alpha(alpha)

                self._print_status(chi, level, tool_calls)

                # 7. Compaction (Level 3) — optional for TASK-044 A/B comparisons
                c_before = self.compactor.compaction_count
                if self.enable_compaction:
                    if chi < 0.5 and self.compactor.needs_compaction(chi):
                        print(
                            f"[COMPACTION] Level 3 triggered at turn {self.turn_count} — "
                            f"CHI={chi:.3f}"
                        )
                        self.compactor.compact("auto")
                    elif self.compactor.needs_compaction(chi):
                        if chi < 0.3:
                            self.compactor.compact("anchor_reset")
                            print("  [COMPACTION: anchor reset]")
                        elif chi < 0.5:
                            self.compactor.compact("summarize")
                            print("  [COMPACTION: summarize]")
                        else:
                            self.compactor.compact("trim")
                            print("  [COMPACTION: trim]")
                    if chi < 0.3:
                        print(
                            "[LEVEL 4] Cloud escalation threshold (CHI < 0.3) — "
                            "would escalate (not implemented)"
                        )

                compaction_fired = self.compactor.compaction_count > c_before
                strat = (
                    self.compactor.last_compaction_strategy
                    if compaction_fired
                    else None
                )

                if self.turn_log is not None:
                    self._append_turn_log(
                        response=response,
                        chi_result=chi_result,
                        level=level,
                        tool_calls=tool_calls,
                        compaction_fired=compaction_fired,
                        compaction_strategy=strat,
                    )

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
            if self.enable_steering and self.steering is not None:
                try:
                    self.steering.cleanup()
                except Exception:
                    pass

            # Persist session state for resume/status inspection.
            self.session_mgr.save_session(
                self.task,
                self.compactor.conversation_turns,
                self.chi_monitor.chi_history,
                self.turn_count,
                self.anchor_mgr.files_modified,
                steering_active=(self.steering_active if self.enable_steering else False),
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

    def _append_turn_log(
        self,
        *,
        response: str,
        chi_result: Optional[Dict],
        level: int,
        tool_calls: List[Dict],
        compaction_fired: bool,
        compaction_strategy: Optional[str],
    ) -> None:
        """Append one TASK-044-style turn record (requires turn_log list in __init__)."""
        if self.turn_log is None:
            return
        if chi_result is not None:
            chi_score = float(chi_result["chi_score"])
            chi_raw = float(chi_result.get("chi_raw", chi_score))
        elif self.chi_monitor.chi_history:
            chi_score = float(self.chi_monitor.chi_history[-1])
            chi_raw = chi_score
        else:
            chi_score = 0.0
            chi_raw = 0.0

        tool_used = tool_calls[0].get("tool", "none") if tool_calls else "none"
        resp_tokens = len(self.tokenizer.encode(response))
        ctx_tokens = self.compactor.get_context_tokens()

        self.turn_log.append(
            {
                "turn": self.turn_count,
                "chi_score": chi_score,
                "chi_raw": chi_raw,
                "intervention_level": level,
                "steering_active": bool(self.steering_active)
                if self.enable_steering
                else False,
                "steering_alpha": float(self.steering_alpha_current)
                if self.enable_steering
                else 0.0,
                "compaction_fired": compaction_fired,
                "compaction_strategy": compaction_strategy,
                "tool_used": tool_used,
                "response_length": resp_tokens,
                "context_length": ctx_tokens,
                "files_modified": list(self.anchor_mgr.files_modified),
                "response_preview": response[:100],
            }
        )

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
        steering_str = ""
        if self.enable_steering and self.steering is not None:
            state = "ON" if self.steering_active else "OFF"
            steering_str = f" | STEERING: {state} (alpha={self.steering_alpha_current:.3f})"

        print(
            f"  Turn {self.turn_count:3d} | {color}CHI {chi:.3f} "
            f"L{level}\033[0m{steering_str} | Tools: {tools_str}"
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

