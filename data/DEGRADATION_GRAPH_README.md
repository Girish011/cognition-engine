# What the degradation graph shows

**File:** `degradation_graph.png`  
**Source data:** `multiturn_entropy.json` (from TASK-009 multi-turn run)

---

## In one sentence

The graph shows that the model stays coherent for the first ~10 turns, then **response entropy rises** (model gets more “confused” or erratic) even as context length plateaus — i.e. **where cognitive degradation starts**.

---

## How to read it

| Part of the graph | Meaning |
|-------------------|--------|
| **Top plot (blue)** | **Response entropy** (bits). Low = confident, on-topic. High = more random, less coherent. |
| **Green shaded region** | Early turns: entropy low → model stable. |
| **Orange shaded region** | Later turns: entropy can rise → degradation. |
| **Bottom plot (red)** | **Context length** (tokens) — how much conversation the model “sees” at each turn. Grows then plateaus. |

---

## Takeaway

- **Turns 0–10:** Low entropy, model handles the long conversation well.
- **From ~turn 10 onward:** Entropy spikes (e.g. around turn 16) → model’s answers become more unpredictable and less coherent = **degradation**.
- So: **we can see exactly when the model starts to degrade** (entropy going up), which is what CHI will use to trigger interventions.

Regenerate the graph (from project root):

```bash
python scripts/visualize_degradation.py
```
