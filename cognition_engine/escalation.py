"""
Level 4 intervention: escalate to a stronger model for advice.

When CHI collapses (< 0.3) the local model is unlikely to recover on its own. An
``Escalator`` is asked for a short piece of guidance based on the task and the
recent transcript; the agent injects it into the context as a system turn. The
local model stays in control of every action -- the escalator only advises.

Escalation is opt-in and sends the task and recent transcript to an external API.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Protocol

DEFAULT_ESCALATION_MODEL = "claude-sonnet-5-5"
MAX_TRANSCRIPT_CHARS = 6000


class Escalator(Protocol):
    def advise(
        self, task: str, recent_turns: List[Dict[str, str]], files: List[str]
    ) -> Optional[str]:
        """Return guidance text for the local model, or None if there is none."""


def build_escalation_prompt(
    task: str, recent_turns: List[Dict[str, str]], files: List[str]
) -> str:
    """Compact, bounded description of the stuck session."""
    lines: List[str] = []
    for turn in recent_turns:
        role = turn.get("role", "assistant")
        lines.append(f"{role}: {turn.get('content', '')}")
    transcript = "\n".join(lines)[-MAX_TRANSCRIPT_CHARS:]
    return (
        "A small local coding model is stuck on a task and its quality has degraded "
        "(looping, repeating itself, or drifting). Read the task and its most recent "
        "transcript, then give it concise, concrete guidance: what is going wrong and "
        "the single next action it should take. Maximum 120 words. Do not write the "
        "full solution.\n\n"
        f"TASK:\n{task}\n\nFILES TOUCHED: {', '.join(files) or '(none)'}\n\n"
        f"RECENT TRANSCRIPT:\n{transcript}"
    )


class AnthropicEscalator:
    """Escalate to a Claude model through the ``anthropic`` SDK (optional dependency)."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 400,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "Escalation needs the 'anthropic' package: pip install 'cognition-engine[escalation]'"
            ) from exc
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Escalation needs ANTHROPIC_API_KEY to be set.")
        self._client = anthropic.Anthropic(api_key=key)
        self.model = model or os.environ.get("COGNITION_ESCALATION_MODEL", DEFAULT_ESCALATION_MODEL)
        self.max_tokens = max_tokens

    def advise(
        self, task: str, recent_turns: List[Dict[str, str]], files: List[str]
    ) -> Optional[str]:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": build_escalation_prompt(task, recent_turns, files)}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        return text or None


def build_escalator(name: Optional[str]) -> Optional[Escalator]:
    """Map a CLI name to an escalator. ``None`` / ``"none"`` disables escalation."""
    if not name or name == "none":
        return None
    if name == "anthropic":
        return AnthropicEscalator()
    raise ValueError(f"Unknown escalator: {name!r} (expected 'none' or 'anthropic')")
