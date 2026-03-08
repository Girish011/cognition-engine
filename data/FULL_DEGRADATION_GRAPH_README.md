# What the full degradation graph shows

**File:** `full_degradation_graph.png`  
**Source data:** `full_degradation.json` (from TASK-013: `python scripts/full_degradation_experiment.py`)

---

## In one sentence

Three signals over conversation turns: **entropy** (how confident/confused the model is), **repetition** (how much it repeats the previous answer), and **goal drift** (how far it’s moved from the original task). When these get worse together, we see **cognitive degradation**.

---

## How to read the 3 panels

| Panel | What it is | How to read it |
|-------|------------|----------------|
| **Top (blue)** | **Response entropy** (bits) | **Up = worse.** Higher entropy = more uncertain/erratic. Lower = more confident, on-topic. |
| **Middle (green)** | **Repetition vs previous** (Jaccard) | **Up = worse.** Compares this turn’s reply to the last. Higher = more copy-paste, less new content. |
| **Bottom (red)** | **Goal drift** (1 − similarity) | **Up = worse.** How far the current response is from the original task in “hidden space.” 0 = on-task, 1 = fully drifted. |

**X-axis:** Turn number (each step in the multi-turn coding conversation). Full run = 20 turns (0–19).

---

## Latest run (20-turn full experiment)

- **Inflection turn:** **10** — degradation likely begins here; CHI would start Level 2 intervention around this turn.
- **Entropy:** Low and stable for turns 0–9; rises from ~turn 10 onward (clear spike).
- **Repetition:** Rises through early turns; later turns can show 0 or 1.0 when responses are short or variable.
- **Goal drift:** Varies by turn (not pegged at 1.0); computed from hidden-state similarity (forward-only, last-token for long responses).

To recompute the inflection from current data: `python scripts/find_inflection.py` (from project root).

---

## What to look for

- **Entropy rising** → model getting confused or random.
- **Repetition rising** → model reusing the same kind of answer.
- **Goal drift rising** → model forgetting the original task.

If two or three rise together, that’s where **CHI would trigger intervention** (e.g. summarise context, restart, or hand off).

---

## Regenerate the graph

From project root:

- **Full experiment** (runs model, saves JSON, then plots):
  ```bash
  python scripts/full_degradation_experiment.py
  ```
- **Quick test** (e.g. 5 turns):
  ```bash
  python scripts/full_degradation_experiment.py --max-turns 5
  ```
- **Plot only** (redraw from existing `full_degradation.json`, no model run):
  ```bash
  python scripts/full_degradation_experiment.py --plot-only
  ```
- **Find inflection turn** (after you have `full_degradation.json` with ≥7 turns):
  ```bash
  python scripts/find_inflection.py
  ```
