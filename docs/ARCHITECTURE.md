# Architecture

Cognition Engine wraps a local MLX model in a **measure → decide → correct** loop.

```
                 ┌──────────────────────────────────────────────┐
 task ──────────►│ CognitionAgent (agent.py)                    │
                 │   generate → parse tool call → execute tool  │
                 └──────┬───────────────────────────▲───────────┘
                        │ response                  │ intervention
                        ▼                           │
                 ┌─────────────┐   CHI, level   ┌───┴────────────────────────┐
                 │ CHIMonitor  │───────────────►│ Intervention ladder        │
                 │ (chi.py)    │                │ L1 none                    │
                 │ entropy     │                │ L2 activation steering     │
                 │ repetition  │                │ L3 context compaction      │
                 │ goal drift  │                │ L4 escalate (log only)     │
                 └─────────────┘                └────────────────────────────┘
```

## Modules

| Module | Role |
|---|---|
| `inference.py`, `hidden_states.py`, `attention_weights.py` | Thin MLX wrappers; `CaptureLayer` exposes per-layer hidden states because MLX has no forward hooks. |
| `entropy.py` | Shannon entropy of the next-token distribution over the response span. |
| `repetition.py` | Jaccard similarity on word n-grams between consecutive responses (gated for responses under 20 tokens). |
| `goal_drift.py` | Cosine similarity between the task's and the response's hidden state at one layer. |
| `chi.py` | `CHIMonitor`: weighted mix (0.4 / 0.3 / 0.3), per-task drift baseline over the first 3 turns, 2-turn cold-start grace, EMA smoothing (α = 0.3). |
| `steering/` | Contrastive pair sets → activations → per-layer vectors (`compute_vectors.py`) → `SteeringWrapper` / `SteeringEngine` that add `alpha * vector` to one layer's output. |
| `escalation.py` | `Escalator` protocol and `AnthropicEscalator` (opt-in, optional dependency). |
| `analysis.py` | Summaries of recorded runs; flags a CHI trace too narrow (< 0.15) to drive the ladder. |
| `compaction.py`, `anchors.py` | Trim / summarize context and re-inject short task "anchor" reminders. |
| `agent.py`, `prompts.py` | Bracketed tool protocol (`[READ] [WRITE] [EDIT] [EXEC] [THINK] [DONE]`), loop and think-loop guards, completion detection. |
| `memory.py`, `session.py` | `.cognition/` JSON notes and resumable session state. |
| `cli.py` | `chat`, `agent`, `status` commands. |

## CHI and the ladder

`CHI = 0.4·entropy_health + 0.3·repetition_health + 0.3·drift_health`, EMA-smoothed.

| Level | CHI (smoothed) | Action |
|---|---|---|
| 1 | > 0.80 | normal generation |
| 2 | 0.50 – 0.80 | in the agent: steering turns on below 0.85 and off above 0.90 (hysteresis), alpha scaled by CHI |
| 3 | 0.30 – 0.50 | context compaction (agent threshold 0.65, 3-turn grace, forced after 5 steered turns or 2 guard fires) |
| 4 | < 0.30 | optional escalation (`escalation.py`): a stronger model returns short guidance, injected as a system turn; 5-turn cooldown |

The `chat`-mode level mapping and the agent's tuned thresholds differ on purpose: the agent
thresholds were recalibrated against the observed CHI range of tool-using runs (TASK-047E).

## Steering mechanics

1. `contrastive_pairs.py` holds three pair sets (`task_focus`, `tool_format`, `code_coherence`).
2. `extract_activations.py` records every layer's last-token state for both sides of each pair.
3. `compute_vectors.py` takes `mean(positive) − mean(negative)` per layer.
4. `injector.py` wraps one decoder layer and adds `alpha · vector` to its output while active.

Vectors are not committed (derived data); build them with `python -m cognition_engine steering-build`.
The agent supports two policies: `chi` (hysteresis band driven by CHI) and `always_on` (the control condition).
