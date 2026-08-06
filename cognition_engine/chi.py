"""
CHI — Cognitive Health Index (TASK-015 / TASK-047E)

Combines entropy, repetition, and goal drift into a single
score from 0.0 (completely degraded) to 1.0 (fully healthy).

TASK-047E fixes:
- Goal drift uses rolling baseline normalization (first 3 turns calibrate
  the per-task baseline; drift is measured as deviation from that baseline,
  not absolute cosine).
- Cold-start grace: turns 0-1 return CHI 0.85 regardless of raw signals
  (CHI needs ≥2 turns of history to produce meaningful measurements).
- Full raw signal logging on every update for debugging.
"""

import math

import mlx.core as mx

from .entropy import compute_token_entropy, get_logits_for_sequence
from .repetition import compute_repetition_score
from .goal_drift import compute_goal_drift

# Minimum response token count before we compute repetition (otherwise carry forward)
MIN_RESPONSE_TOKENS_FOR_REPETITION = 20

# Exponential moving average: chi_smoothed = ALPHA * chi_current + (1 - ALPHA) * chi_previous
# Lower ALPHA = smoother (more weight on history). 0.3 gives ~3-5 turn effective window.
CHI_SMOOTH_ALPHA = 0.3

# TASK-047E: rolling baseline for goal drift
DRIFT_BASELINE_TURNS = 3          # collect baselines for first N turns
DRIFT_BASELINE_NEUTRAL = 0.8      # health returned during baseline collection
DRIFT_MIN_STD = 0.03              # floor on baseline std to avoid div-by-tiny-number

# TASK-047E: cold-start grace
COLD_START_TURNS = 2              # return neutral CHI for turns 0..(N-1)
COLD_START_CHI = 0.85             # neutral healthy score during cold start


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
        self.entropy_history: list[float] = []
        self.repetition_history: list[float] = []
        self.drift_history: list[float] = []
        self.chi_history: list[float] = []  # smoothed CHI values (used for intervention level)
        self.previous_response: str = ""

        # TASK-047E: rolling baseline for goal drift cosine
        self.drift_cosine_history: list[float] = []
        self.drift_baseline_mean: float | None = None
        self.drift_baseline_std: float | None = None

    @property
    def turn_count(self) -> int:
        return len(self.chi_history)

    def _compute_drift_health(self, raw_cosine: float) -> float:
        """
        TASK-047E: rolling baseline normalization for goal drift.

        Turns 0..DRIFT_BASELINE_TURNS-1: collect cosine values, return neutral health.
        Turn DRIFT_BASELINE_TURNS: compute baseline mean/std from collected values.
        Turn DRIFT_BASELINE_TURNS+: drift_health = how far current is below baseline.
        """
        self.drift_cosine_history.append(raw_cosine)

        if len(self.drift_cosine_history) <= DRIFT_BASELINE_TURNS:
            # Still collecting baseline
            if len(self.drift_cosine_history) == DRIFT_BASELINE_TURNS:
                # Freeze the baseline
                vals = self.drift_cosine_history[:DRIFT_BASELINE_TURNS]
                self.drift_baseline_mean = sum(vals) / len(vals)
                variance = sum((v - self.drift_baseline_mean) ** 2 for v in vals) / len(vals)
                self.drift_baseline_std = max(math.sqrt(variance), DRIFT_MIN_STD)
            return DRIFT_BASELINE_NEUTRAL

        # Post-baseline: measure deviation below the mean
        drop = self.drift_baseline_mean - raw_cosine
        if drop <= 0:
            # Current cosine is at or above baseline — no drift
            return 1.0
        # Scale by std: 1 std below = health 0.5, 2 std = 0.0
        drift_health = max(0.0, min(1.0, 1.0 - drop / (2.0 * self.drift_baseline_std)))
        return drift_health

    def update(self, current_response: str, full_context: str = None) -> dict:
        """
        Compute CHI after a new model response.

        Args:
            current_response: The model's latest response text
            full_context: The full conversation so far (for entropy). If None, uses
                original_task + current_response for logits.

        Returns: dict with chi_score (smoothed), all raw signal values for debugging
        """
        current_turn = self.turn_count

        # --- Signal 1: Entropy ---
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
        entropy_health = max(0.0, min(1.0, 1.0 - entropy_raw / 10.0))

        # --- Signal 2: Repetition ---
        repetition_gated = False
        if response_tokens < MIN_RESPONSE_TOKENS_FOR_REPETITION and self.repetition_history:
            repetition_raw = self.repetition_history[-1]
            repetition_gated = True
        elif self.previous_response:
            repetition_raw = compute_repetition_score(self.previous_response, current_response)
        else:
            repetition_raw = 0.0  # First turn: no previous response
        rep_health = 1.0 - repetition_raw

        # --- Signal 3: Goal Drift (TASK-047E: rolling baseline) ---
        drift_cosine = 0.0
        try:
            drift_cosine = compute_goal_drift(
                self.model,
                self.tokenizer,
                self.original_task,
                current_response,
                layer_idx=self.layer_idx,
                use_forward_only=True,
            )
            drift_health = self._compute_drift_health(drift_cosine)
            drift_raw = 1.0 - drift_cosine
        except Exception:
            drift_cosine = 0.0
            drift_raw = 0.5
            drift_health = 0.5
            self.drift_cosine_history.append(0.5)

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

        # --- TASK-047E: Cold-start grace ---
        if current_turn < COLD_START_TURNS:
            chi_smoothed = COLD_START_CHI
        elif not self.chi_history or self.chi_history[-1] == COLD_START_CHI:
            # First real turn after cold start: seed EMA with current value
            chi_smoothed = chi_current
        else:
            chi_previous = self.chi_history[-1]
            chi_smoothed = CHI_SMOOTH_ALPHA * chi_current + (1.0 - CHI_SMOOTH_ALPHA) * chi_previous

        # Update history
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
            "repetition_gated": repetition_gated,
            "goal_drift_raw": drift_raw,
            "goal_drift_health": drift_health,
            "goal_drift_cosine": drift_cosine,
            "drift_baseline_mean": self.drift_baseline_mean,
            "drift_baseline_std": self.drift_baseline_std,
            "response_token_count": response_tokens,
            "turn_index": current_turn,
            "intervention_level": self.get_intervention_level(),
            "cold_start": current_turn < COLD_START_TURNS,
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
    import json
    from cognition_engine.inference import load

    model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    monitor = CHIMonitor(model, tokenizer, "Build a TaskManager class")

    responses = [
        ("class TaskManager:\n    def __init__(self):\n        self.tasks = []",
         "Build a TaskManager class\n\nclass TaskManager:\n    def __init__(self):\n        self.tasks = []"),
        ("I think we should consider the weather today. The weather is nice.", None),
        ("def add_task(self, task): self.tasks.append(task)", None),
        ("The weather is sunny. Let me think about clouds.", None),
    ]

    for i, (resp, ctx) in enumerate(responses):
        r = monitor.update(resp, ctx)
        print(f"\nTurn {i}:")
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, indent=2))

    print("\nManual test passed.")
