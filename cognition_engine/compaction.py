"""
Context compaction engine (TASK-021).

When context gets too long or CHI drops below a threshold, we compact the
conversation without losing critical information. This is the Level 3
intervention: context compaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional

from mlx_lm import generate


@dataclass
class Turn:
    role: str
    content: str


class ContextCompactor:
    """Manages conversation context with automatic compaction."""

    def __init__(self, model, tokenizer, max_context_tokens: int = 24_000):
        self.model = model
        self.tokenizer = tokenizer
        # ~75% of a 32K context limit by default
        self.max_context_tokens = max_context_tokens
        self.conversation_turns: List[Dict[str, str]] = []
        self.compaction_count = 0
        self.anchor_text = ""  # Original task description, always kept
        # Set when a compaction actually increments compaction_count (for TASK-044 logging).
        self.last_compaction_strategy: Optional[str] = None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def set_anchor(self, task_description: str) -> None:
        """Set the anchor text — always preserved across compactions."""
        self.anchor_text = task_description.strip()

    def add_turn(self, role: str, content: str) -> None:
        """Add a conversation turn."""
        self.conversation_turns.append({"role": role, "content": content})

    def get_context_tokens(self) -> int:
        """Count total tokens in current context."""
        full_text = self._build_context()
        return len(self.tokenizer.encode(full_text))

    def needs_compaction(self, chi_score: Optional[float] = None) -> bool:
        """Check if compaction is needed based on context length and CHI."""
        token_count = self.get_context_tokens()
        if token_count > self.max_context_tokens:
            return True
        if chi_score is not None and chi_score < 0.5:
            return True
        return False

    def compact(self, strategy: str = "auto") -> str:
        """
        Compact the context. Returns the new context string.

        Strategies:
        - "trim": Drop oldest turns, keep anchor + last N turns
        - "summarize": Use the model to summarize older turns
        - "anchor_reset": Nuclear option — keep only anchor + last 2 turns
        - "auto": Choose strategy based on current context length
        """
        self.last_compaction_strategy = None
        if not self.conversation_turns and not self.anchor_text:
            return ""

        if strategy == "auto":
            # Simple heuristic: try trim first; if still too long, summarize;
            # if still too long or CHI is presumed very low, anchor_reset.
            context_before = self.get_context_tokens()
            if context_before <= self.max_context_tokens:
                return self._build_context()

            context = self._trim()
            if self.get_context_tokens() <= self.max_context_tokens:
                self.compaction_count += 1
                self.last_compaction_strategy = "auto"
                return context

            context = self._summarize()
            if self.get_context_tokens() <= self.max_context_tokens:
                self.compaction_count += 1
                self.last_compaction_strategy = "auto"
                return context

            context = self._anchor_reset()
            self.compaction_count += 1
            self.last_compaction_strategy = "auto"
            return context

        if strategy == "trim":
            context = self._trim()
        elif strategy == "summarize":
            context = self._summarize()
        elif strategy == "anchor_reset":
            context = self._anchor_reset()
        else:
            # Fallback to trim for unknown strategy
            context = self._trim()

        self.compaction_count += 1
        self.last_compaction_strategy = strategy
        return context

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _build_context(self) -> str:
        """Build the full context string from anchor + turns."""
        parts: List[str] = []
        if self.anchor_text:
            parts.append(f"[TASK]: {self.anchor_text}\n")
        for turn in self.conversation_turns:
            role = turn.get("role", "assistant")
            prefix = "User" if role == "user" else "Assistant" if role == "assistant" else role.capitalize()
            parts.append(f"{prefix}: {turn.get('content', '')}")
        return "\n".join(parts)

    def _trim(self, keep_last: int = 5) -> str:
        """Keep anchor + last N turns. Fast, no model call."""
        if keep_last <= 0:
            self.conversation_turns = []
        else:
            self.conversation_turns = self.conversation_turns[-keep_last:]
        return self._build_context()

    def _summarize(self, recent_keep: int = 5) -> str:
        """
        Summarize older turns into a brief recap, keep recent turns verbatim.

        Split conversation into:
        - Old turns (to be summarized)
        - Recent turns (keep as-is, last `recent_keep`)
        """
        if not self.conversation_turns:
            return self._build_context()

        if recent_keep <= 0 or len(self.conversation_turns) <= recent_keep:
            # Not enough history to justify summarization; fall back to trim.
            return self._trim(keep_last=len(self.conversation_turns))

        old_turns = self.conversation_turns[:-recent_keep]
        recent_turns = self.conversation_turns[-recent_keep:]

        old_context_lines = []
        if self.anchor_text:
            old_context_lines.append(f"[TASK]: {self.anchor_text}\n")
        for turn in old_turns:
            role = turn.get("role", "assistant")
            prefix = "User" if role == "user" else "Assistant" if role == "assistant" else role.capitalize()
            old_context_lines.append(f"{prefix}: {turn.get('content', '')}")
        old_context = "\n".join(old_context_lines)

        # Build summarization prompt
        summary_prompt = (
            "[SYSTEM] You are summarizing a coding session.\n"
            "Read the following conversation history and produce a concise summary "
            "of the key decisions, plans, and partial implementations. "
            "Keep it under 200 tokens.\n\n"
            "[HISTORY START]\n"
            f"{old_context}\n"
            "[HISTORY END]\n\n"
            "[SUMMARY]:"
        )

        try:
            summary_text = generate(
                self.model,
                self.tokenizer,
                prompt=summary_prompt,
                max_tokens=220,
            )
        except Exception:
            # If summarization fails, fall back to trimming.
            return self._trim(keep_last=recent_keep)

        # Rebuild conversation: anchor + summary block + recent turns
        self.conversation_turns = [
            {"role": "system", "content": "[CONTEXT SUMMARY]\n" + summary_text.strip()}
        ] + recent_turns
        return self._build_context()

    def _anchor_reset(self) -> str:
        """
        Nuclear option: wipe everything except anchor + last 2 turns.

        Used when CHI is critically low and summarization isn't enough.
        """
        keep_last = 2 if len(self.conversation_turns) >= 2 else len(self.conversation_turns)
        self.conversation_turns = self.conversation_turns[-keep_last:]
        return self._build_context()


if __name__ == "__main__":
    # Lightweight manual test scaffold; real tests should live in tests/.
    from cognition_engine.inference import load

    model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    compactor = ContextCompactor(model, tokenizer, max_context_tokens=2_000)
    compactor.set_anchor("Build a TaskManager class in Python")

    # Add 20 turns of fake conversation
    for i in range(20):
        compactor.add_turn("user", f"Step {i}: Add feature number {i} " * 20)
        compactor.add_turn(
            "assistant",
            f"Here is the implementation for step {i} " * 20,
        )

    before = compactor.get_context_tokens()
    print(f"Before compaction: {before} tokens")
    assert compactor.needs_compaction()

    compactor.compact("trim")
    after_trim = compactor.get_context_tokens()
    print(f"After trim: {after_trim} tokens")

    context = compactor._build_context()
    assert "TaskManager" in context

