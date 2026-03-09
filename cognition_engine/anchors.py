"""
SCAN-style anchor tokens (TASK-022).

AnchorManager generates periodic reminder blocks that can be injected into
the model's context to keep it focused on the current task and decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AnchorManager:
    """Generates and injects anchor tokens based on CHI level."""

    task_description: str
    key_decisions: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)

    def generate_anchor(self, level: str = "ANCHOR") -> str:
        """
        Generate an anchor token string.

        Levels:
        - "FULL": ~300 tokens. Complete task context restoration.
        - "MINI": ~120 tokens. Brief reminder.
        - "ANCHOR": ~20 tokens. One-liner.
        """
        level = level.upper()
        if level == "FULL":
            return self._full_anchor()
        if level == "MINI":
            return self._mini_anchor()
        return self._anchor()

    def _full_anchor(self) -> str:
        parts: List[str] = [
            "[SYSTEM REMINDER — FULL CONTEXT RESTORATION]",
            f"Original task: {self.task_description}",
        ]
        if self.key_decisions:
            parts.append("Key decisions so far:")
            for d in self.key_decisions[-5:]:  # Last 5 decisions
                parts.append(f"  - {d}")
        if self.files_modified:
            parts.append(
                f"Files modified: {', '.join(self.files_modified[-10:])}"
            )
        parts.append("[END REMINDER — Continue with the task above.]")
        return "\n".join(parts)

    def _mini_anchor(self) -> str:
        return (
            f"[REMINDER] Task: {self.task_description[:200]}. "
            f"Decisions: {len(self.key_decisions)}. "
            f"Files: {len(self.files_modified)}. Stay focused."
        )

    def _anchor(self) -> str:
        return f"[FOCUS] {self.task_description[:100]}"

    def add_decision(self, decision: str) -> None:
        """Track a key decision made during the session."""
        self.key_decisions.append(decision)

    def add_file(self, filepath: str) -> None:
        """Track a file the agent has modified."""
        if filepath not in self.files_modified:
            self.files_modified.append(filepath)

    def get_anchor_for_chi(self, chi_score: float) -> Optional[str]:
        """
        Based on CHI score, return the appropriate anchor level.
        Returns None if CHI is healthy (no anchor needed).
        """
        if chi_score > 0.8:
            return None  # Healthy, no anchor needed
        if chi_score > 0.6:
            return self.generate_anchor("ANCHOR")  # ~20 tokens
        if chi_score > 0.4:
            return self.generate_anchor("MINI")  # ~120 tokens
        return self.generate_anchor("FULL")  # ~300 tokens


if __name__ == "__main__":
    # Lightweight manual test scaffold; real tests should live in tests/.
    mgr = AnchorManager("Build a REST API with Flask")
    mgr.add_decision("Use SQLite for database")
    mgr.add_decision("JWT for authentication")
    mgr.add_file("app.py")
    mgr.add_file("models.py")

    # Test each level
    print("FULL ANCHOR:\n", mgr.generate_anchor("FULL"))
    print("\nMINI ANCHOR:\n", mgr.generate_anchor("MINI"))
    print("\nANCHOR:\n", mgr.generate_anchor("ANCHOR"))

    # Test CHI-based selection
    assert mgr.get_anchor_for_chi(0.9) is None  # Healthy
    assert "FOCUS" in mgr.get_anchor_for_chi(0.7)  # Light anchor
    assert "REMINDER" in mgr.get_anchor_for_chi(0.5)  # Mini
    assert "FULL CONTEXT" in mgr.get_anchor_for_chi(0.3)  # Full

    print("\nAnchorManager manual scaffold passed.")

