# Cognition Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**A local AI coding agent that monitors its own brain.**

Multi-turn coding sessions degrade: models repeat themselves, drift off-task, and lose coherence. Cognition Engine runs on your Mac (MLX), tracks **cognitive health** in real time, and is built to trigger interventions (steering, context compaction, escalation) when quality drops.

---

## The graph

This is what degradation looks like over a 20-turn conversation. Entropy and goal drift rise; repetition spikes. CHI (Cognitive Health Index) combines these into one score so the system knows when to act.

![Degradation over 20 turns: entropy, repetition, goal drift](data/full_degradation_graph.png)

*Generate it: `python scripts/full_degradation_experiment.py` (output in `data/`).*

---

## What it does

- **CHI score (0-1)** - One number from entropy, repetition, and goal drift. High = healthy; low = degraded.
- **Intervention levels** - Level 1 (normal) → 2 (steer) → 3 (compact context) → 4 (escalate). Thresholds drive when to intervene.
- **CLI with live CHI** - Chat with the model from the terminal; see CHI and E/R/D after every reply.

---

## Quick start

```bash
git clone <your-repo>
cd cognition-engine
pip install -e .
pip install mlx mlx-lm
python -m cognition_engine --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
```

Type a task, get a response, see CHI. Ctrl+C to quit.

---

## How it works

CHI is a weighted mix of three signals: **entropy** (next-token uncertainty), **repetition** (Jaccard between consecutive responses), and **goal drift** (hidden-state similarity to the original task). The score is smoothed over turns and mapped to intervention levels. Details: [.cofounder/COFOUNDER_BRAIN.md](.cofounder/COFOUNDER_BRAIN.md) and code in `cognition_engine/chi.py`.

---

## Why this exists

Studies show large multi-turn coding sessions suffer **~39% degradation** in correctness and coherence. Local models are especially prone: long context, no built-in “am I still on task?” check. Cognition Engine is built to detect when the model’s brain is drifting or looping, so a future layer can steer, compact context, or hand off to a stronger model—instead of failing silently.

---

## Roadmap

- **Activation steering** - Use hidden states to nudge the model back on task (EasySteer-style).
- **Context compaction** - Shrink working context when CHI drops while keeping task anchors.
- **LoRA crystallization** - Nightly fine-tuning on session data to improve the local model over time.

---

## License

MIT. See [LICENSE](LICENSE).
