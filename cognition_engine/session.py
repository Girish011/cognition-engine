"""
Session Management — Start, stop, resume agent sessions.

State is saved to .cognition/session.json so the agent can pick up where it
left off after interruption (Ctrl+C) or review last completion state.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class SessionState:
    task: str
    status: str  # "interrupted", "completed", "failed"
    turn_count: int
    timestamp: str
    chi_history: list[float]
    chi_final: float | None
    files_modified: list[str]
    conversation_turns: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "status": self.status,
            "turn_count": self.turn_count,
            "timestamp": self.timestamp,
            "chi_history": self.chi_history,
            "chi_final": self.chi_final,
            "files_modified": self.files_modified,
            "conversation_turns": self.conversation_turns,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "SessionState":
        chi_hist = d.get("chi_history") or []
        if not isinstance(chi_hist, list):
            chi_hist = []
        # Best-effort cast to floats
        chi_hist_f: list[float] = []
        for x in chi_hist:
            try:
                chi_hist_f.append(float(x))
            except Exception:  # noqa: BLE001
                continue

        chi_final = d.get("chi_final", None)
        try:
            chi_final_f = float(chi_final) if chi_final is not None else None
        except Exception:  # noqa: BLE001
            chi_final_f = None

        files = d.get("files_modified") or []
        if not isinstance(files, list):
            files = []

        turns = d.get("conversation_turns") or []
        if not isinstance(turns, list):
            turns = []

        return SessionState(
            task=str(d.get("task", "")),
            status=str(d.get("status", "interrupted")),
            turn_count=int(d.get("turn_count", 0) or 0),
            timestamp=str(d.get("timestamp", "")),
            chi_history=chi_hist_f,
            chi_final=chi_final_f,
            files_modified=[str(f) for f in files if isinstance(f, (str, int))],
            conversation_turns=[t for t in turns if isinstance(t, dict)],
        )


class SessionManager:
    """Manages agent session state for persistence across runs."""

    def __init__(self, project_dir: str):
        self.session_dir = os.path.join(project_dir, ".cognition")
        os.makedirs(self.session_dir, exist_ok=True)
        self.session_file = os.path.join(self.session_dir, "session.json")

    def save_session(
        self,
        task: str,
        conversation_turns: list[dict[str, Any]],
        chi_history: list[float],
        turn_count: int,
        files_modified: list[str],
        status: str = "interrupted",
    ) -> SessionState:
        """Save current session state to disk."""
        state = SessionState(
            task=task,
            status=status,
            turn_count=turn_count,
            timestamp=datetime.now().isoformat(),
            chi_history=list(chi_history or []),
            chi_final=(chi_history[-1] if chi_history else None),
            files_modified=list(files_modified or []),
            conversation_turns=list((conversation_turns or [])[-10:]),
        )
        with open(self.session_file, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
        return state

    def load_session(self) -> SessionState | None:
        """Load previous session if it exists."""
        if not os.path.exists(self.session_file):
            return None
        try:
            with open(self.session_file) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return None
            return SessionState.from_dict(data)
        except Exception:  # noqa: BLE001
            return None

    def has_interrupted_session(self) -> bool:
        """Check if there's an interrupted session to resume."""
        session = self.load_session()
        return session is not None and session.status == "interrupted"

    def clear_session(self) -> None:
        """Clear session state (after completion or user choice)."""
        if os.path.exists(self.session_file):
            os.remove(self.session_file)

    def get_resume_context(self) -> str | None:
        """
        Build a context string for resuming an interrupted session.
        Returns None if no session to resume.
        """
        session = self.load_session()
        if not session or session.status != "interrupted":
            return None

        lines: list[str] = [
            "[RESUMING PREVIOUS SESSION]",
            f"Original task: {session.task}",
            f"Progress: {session.turn_count} turns completed",
        ]
        if session.chi_final is not None:
            lines.append(f"Last CHI: {session.chi_final:.3f}")
        if session.files_modified:
            lines.append(f"Files modified: {', '.join(session.files_modified)}")

        lines.append("")
        lines.append("Recent conversation:")

        for turn in session.conversation_turns[-5:]:
            role = str(turn.get("role", "assistant")).capitalize()
            content = str(turn.get("content", ""))
            # Some chat templates may leak special tokens; strip common ones to
            # keep the resume context readable and less confusing.
            for tok in ("<|im_end|>", "<|im_start|>"):
                content = content.replace(tok, "")
            content = content.strip()[:200]
            lines.append(f"  {role}: {content}")

        lines.append("[END RESUME CONTEXT — Continue from where you left off]")
        return "\n".join(lines)

