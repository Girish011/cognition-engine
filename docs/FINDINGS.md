# Findings

The question: **can a local LLM agent be corrected in the middle of execution, using signals
read from the model's own internals?** This is what the experiments (Weeks 1-8, Qwen2.5-Coder-7B-Instruct
4-bit on an M4 Pro) did and did not show. Raw run data is in [`../data`](../data); the
day-by-day lab notebook is [`../.cofounder/SESSION_LOG.md`](../.cofounder/SESSION_LOG.md).

## Summary

| # | Claim | Verdict |
|---|---|---|
| 1 | Internal signals (entropy, repetition, goal drift) rise/fall measurably as a long chat degrades | **Supported** (one 20-turn run; inflection ≈ turn 10, ~2.5k tokens) |
| 2 | Activation steering can be injected into MLX models at negligible cost | **Supported** (no measurable latency change) |
| 3 | A steering vector changes behaviour without destroying output | **Conditionally** — only layer ~16 at alpha ≈ 0.25-0.5; alpha 1.0 killed output at every layer tried; `task_focus` failed at 10 turns, `tool_format` passed |
| 4 | Steering measurably improves CHI | **Weak** (+0.01-0.02) |
| 5 | CHI-triggered correction improves task outcomes | **Not shown** — see below |
| 6 | The original 3-signal CHI is a reliable trigger | **No, until recalibrated** — goal drift was geometrically pinned (~0.39 cosine), so CHI sat in a ~0.02 band and the ladder was always at Level 2 |

## What happened, in order

**Detection works in chat.** Over a scripted 20-turn conversation entropy, repetition and
goal drift worsen together after roughly turn 10 (`data/full_degradation_graph.png`);
the smoothed CHI crosses 0.7 at turn 11 (`data/chi_validation.png`).

**Steering is mechanically sound but narrow.** A 45-cell layer × alpha sweep found layer 16 the
only stable region. Dual-vector steering did not help.

**Most "degradation" in the agent was a bug, not cognition.** Before TASK-045 the agent generated
hundreds of garbage tokens past `<|im_end|>` every turn, poisoning context and leaking template
tokens into shell commands. After fixing stop-token handling, parsing, and completion detection,
the calculator and unit-converter tasks finish in 3-4 turns in **all three modes** (baseline,
steering, full ladder) with identical results: the model completes before it can degrade
(`data/task047_*.json`).

**The one apparent steering win did not survive scrutiny.** On a harder expense-tracker task the
baseline failed (25 turns, think-loop) while the steered run passed (10 turns). The later
analysis (TASK-047D) showed that steering had been on from turn 0 because CHI never exceeded
0.7, so the difference came from an always-on behavioural prior producing different code
(explicit `save`/`load` instead of auto-save in `__init__`), not from a CHI-triggered
correction. Generation at temperature 0 is deterministic, so this is a clean n = 1 A/B, but it
is one task.

**CHI was then repaired (TASK-047E):** per-task drift baseline over the first 3 turns,
cold-start grace, and thresholds recalibrated to the observed 0.82-0.95 healthy range. These
changes are in the code; a before/after replication on a long task was not completed.

## Conclusions

- Reading health from model internals is feasible and cheap; the weakest part was the goal-drift
  signal, which needs per-task normalisation.
- Mid-run correction with a 7B model is hard to evaluate, because well-engineered scaffolding
  (stop tokens, parsing, guards) removes most failures before cognition-level correction matters.
- Steering behaves as a **behavioural prior**, not a repair tool: it can change what gets written,
  but we found no case of it rescuing a run that had already degraded.

## Limitations and next experiments

- Single model, single machine, small task set; no statistical claims.
- No task long enough (15+ turns) to reach the degradation regime *and* complete, so Levels 3-4
  never fired in agent runs.
- Open: validate the recalibrated CHI on a 25-turn task; compare steering switched on only after
  a CHI drop against always-on; try a larger model (14B) and a hybrid-attention model
  (Qwen3.5 / DeltaNet), where steering targets may differ.
