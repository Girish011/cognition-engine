# Cognition Engine

A local AI coding agent that **monitors its own cognitive health** and self-corrects when degrading. Built on MLX for Apple Silicon.

## One-liner

"A local AI that never forgets what it's doing."

## What it does

- **CHI (Cognitive Health Index)** — Combines entropy, repetition, and goal drift into a single score (0–1). High = healthy; low = degraded.
- **Intervention levels** — When CHI drops, the system can trigger steering, context compaction, or escalation (see docs).
- **MLX native** — Uses Apple's MLX for inference and hidden-state access (no Ollama wrapper).

## Quick start

```bash
cd cognition-engine
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .
pip install mlx mlx-lm
```

From project root:

```bash
python -c "from cognition_engine.chi import CHIMonitor; print('OK')"
python scripts/full_degradation_experiment.py --max-turns 2
```

## Project layout

- **cognition_engine/** — Main package: `chi`, `entropy`, `repetition`, `goal_drift`, `hidden_states`, `inference`
- **scripts/** — Experiments and one-off scripts (multi-turn entropy, degradation, validation)
- **data/** — Experiment outputs (graphs, JSON)
- **tests/** — Unit tests

## How CHI works

CHI v1 is a weighted combination of three signals:

1. **Entropy** — Shannon entropy over the next-token distribution (low = confident, high = confused).
2. **Repetition** — Jaccard similarity between consecutive responses (high = stuck repeating).
3. **Goal drift** — Cosine similarity of hidden states (current output vs original task).

The score is smoothed over turns and used to select an intervention level (1 = normal, 2 = steer, 3 = compact, 4 = escalate).

## License

MIT. See [LICENSE](LICENSE).
