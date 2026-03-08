"""
CHI — Cognitive Health Index (TASK-015)

Combines entropy, repetition, and goal drift into a single
score from 0.0 (completely degraded) to 1.0 (fully healthy).

The score is intentionally simple in v1. We can add more signals
and tune weights later. What matters now is that it WORKS and
DETECTS degradation.

Additions:
- Repetition is gated by response length: if response has < 20 tokens,
  the previous turn's repetition score is carried forward (Jaccard on
  tiny text is unreliable).
- CHI is smoothed with an exponential moving average (0.3 * current + 0.7 * previous)
  over turns to avoid wild single-turn swings.
"""

import sys
from pathlib import Path

# Allow running as script from project root (e.g. python engine/chi.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx

from engine.entropy import compute_token_entropy, get_logits_for_sequence
from engine.repetition import compute_repetition_score
from engine.goal_drift import compute_goal_drift

# Minimum response token count before we compute repetition (otherwise carry forward)
MIN_RESPONSE_TOKENS_FOR_REPETITION = 20

# Exponential moving average: chi_smoothed = ALPHA * chi_current + (1 - ALPHA) * chi_previous
# Lower ALPHA = smoother (more weight on history). 0.3 gives ~3-5 turn effective window.
CHI_SMOOTH_ALPHA = 0.3


class CHIMonitor:
    """
    Tracks cognitive health across a conversation.
    Call update() after each model response to get the current CHI score.
    """

    def __init__(self, model, tokenizer, original_task: str, layer_idx: int = 27):
        self.model = model
        self.tokenizer = tokenizer
        self.original_task = original_task
        self.layer_idx = layer_idx

        # History for tracking trends
        self.entropy_history = []
        self.repetition_history = []
        self.drift_history = []
        self.chi_history = []  # smoothed CHI values (used for intervention level)
        self.previous_response = ""

    def update(self, current_response: str, full_context: str = None) -> dict:
        """
        Compute CHI after a new model response.

        Args:
            current_response: The model's latest response text
            full_context: The full conversation so far (for entropy). If None, uses
                original_task + current_response for logits.

        Returns: dict with chi_score (smoothed), raw signal values, and health components
        """
        # --- Signal 1: Entropy ---
        # Compute entropy on the response tokens. Use full_context if provided so
        # we get logits at the correct positions.
        context_for_entropy = full_context if full_context else (self.original_task + "\n" + current_response)
        logits = get_logits_for_sequence(self.model, self.tokenizer, context_for_entropy)
        entropy = compute_token_entropy(logits)
        mx.eval(entropy)
        response_tokens = len(self.tokenizer.encode(current_response))
        response_entropy = (
            entropy[-response_tokens:] if response_tokens > 0 else entropy
        )
        mean_ent = mx.mean(response_entropy)
        mx.eval(mean_ent)
        entropy_raw = float(mean_ent.item())
        # Normalize to 0-1: 0 bits = 1.0 (healthy), 10+ bits ≈ 0.0 (degraded)
        entropy_health = max(0.0, min(1.0, 1.0 - entropy_raw / 10.0))

        # --- Signal 2: Repetition ---
        # Jaccard similarity vs previous response. Gate by response length: if
        # response is under 20 tokens, carry forward previous turn's repetition
        # instead of computing Jaccard on tiny text.
        if response_tokens < MIN_RESPONSE_TOKENS_FOR_REPETITION and self.repetition_history:
            repetition_raw = self.repetition_history[-1]
        elif self.previous_response:
            repetition_raw = compute_repetition_score(self.previous_response, current_response)
        else:
            repetition_raw = 0.0  # First turn: no previous response
        rep_health = 1.0 - repetition_raw  # 0.0 repetition = 1.0 healthy

        # --- Signal 3: Goal Drift ---
        # compute_goal_drift returns cosine similarity (higher = more on-task)
        try:
            similarity = compute_goal_drift(
                self.model,
                self.tokenizer,
                self.original_task,
                current_response,
                layer_idx=self.layer_idx,
                use_forward_only=True,
            )
            drift_raw = 1.0 - similarity  # drift amount (higher = worse)
            drift_health = similarity  # 0.0 drift = 1.0 healthy
        except Exception:
            drift_raw = 0.5
            drift_health = 0.5

        # --- Combine into raw CHI (before smoothing) ---
        w_entropy = 0.4
        w_repetition = 0.3
        w_drift = 0.3
        chi_current = (
            w_entropy * entropy_health
            + w_repetition * rep_health
            + w_drift * drift_health
        )
        chi_current = max(0.0, min(1.0, chi_current))

        # --- Smooth CHI with exponential moving average ---
        if not self.chi_history:
            chi_smoothed = chi_current
        else:
            chi_previous = self.chi_history[-1]
            chi_smoothed = CHI_SMOOTH_ALPHA * chi_current + (1.0 - CHI_SMOOTH_ALPHA) * chi_previous

        # Update history (append before we set previous_response)
        self.entropy_history.append(entropy_raw)
        self.repetition_history.append(repetition_raw)
        self.drift_history.append(drift_raw)
        self.chi_history.append(chi_smoothed)
        self.previous_response = current_response

        return {
            "chi_score": chi_smoothed,
            "chi_raw": chi_current,
            "entropy_raw": entropy_raw,
            "entropy_health": entropy_health,
            "repetition_raw": repetition_raw,
            "repetition_health": rep_health,
            "drift_raw": drift_raw,
            "drift_health": drift_health,
            "turn": len(self.chi_history) - 1,
        }

    def get_intervention_level(self) -> int:
        """
        Based on current (smoothed) CHI, return the intervention level.
        1 = normal, 2 = steer, 3 = compact, 4 = escalate
        """
        if not self.chi_history:
            return 1
        chi = self.chi_history[-1]
        if chi > 0.8:
            return 1
        elif chi > 0.5:
            return 2  # Would trigger EasySteer in Phase 2
        elif chi > 0.3:
            return 3  # Would trigger context compaction in Phase 1
        else:
            return 4  # Would trigger cloud escalation


if __name__ == "__main__":
    from mlx_lm import load

    model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    monitor = CHIMonitor(model, tokenizer, "Build a TaskManager class")

    # Healthy response
    r1 = monitor.update(
        "class TaskManager:\n    def __init__(self):\n        self.tasks = []",
        full_context="Build a TaskManager class\n\nclass TaskManager:\n    def __init__(self):\n        self.tasks = []",
    )
    print(f"Turn 0 CHI: {r1['chi_score']:.3f} Level: {monitor.get_intervention_level()}")
    # Should be high (0.7+), Level 1

    # Simulate degraded response (repetitive, off-topic)
    r2 = monitor.update(
        "I think we should consider the weather today. The weather is nice."
    )
    print(f"Turn 1 CHI: {r2['chi_score']:.3f} Level: {monitor.get_intervention_level()}")
    # Should be lower, possibly Level 2 or 3

    print("Manual test passed.")
