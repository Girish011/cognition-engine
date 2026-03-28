# WEEK 7 TASKS: Steering Calibration + CHI Integration
# Week 6 proved the pipeline works. Week 7 makes it actually useful.
# The gap: steering changes behavior but kills output at high alpha.
# This week: find the sweet spot and wire it into the agent.

---

## WEEK 7: From "It Works" to "It's Useful"
**Goal:** Find optimal steering parameters that improve coherence WITHOUT
killing output, then wire CHI-triggered steering into the agent loop.

**Key insight from Week 6:** Layer 27 + alpha=2.0 = dead output.
Layer 17 + alpha=1.0 = changed output, non-empty, different from baseline.
The steering vectors are valid. The parameters need calibration.

---

### TASK-039: Systematic Alpha/Layer Sweep
**Status:** ⬜ TODO
**Do:** Create `scripts/steering_sweep.py`

The Week 6 experiment used one setting (layer 27, alpha 2.0) which
over-steered. You need a grid search to find the best combination.

**Sweep grid:**
- Layers: [12, 14, 16, 17, 18, 20, 22, 24, 27]
- Alphas: [0.25, 0.5, 1.0, 1.5, 2.0]
- Vector: task_focus (start with one, expand after finding best params)
- Prompt: single 5-turn conversation (not full 10 — keep sweep fast)

**For each (layer, alpha) pair, measure:**
1. `response_alive`: did the model produce >10 tokens on all turns? (bool)
2. `mean_response_length`: average token count across turns
3. `mean_chi`: average CHI across turns
4. `chi_drop`: CHI(turn 0) - CHI(turn 4) — how much it degraded
5. `response_changed`: is output different from unsteered baseline? (bool)

**Output:** `data/steering_sweep.json` + heatmap visualization
`data/steering_sweep.png` (x=layer, y=alpha, color=mean_chi,
dead cells marked with X where response_alive=False)

**What you're looking for:**
- The "Goldilocks zone": response_alive=True AND response_changed=True
  AND mean_chi > unsteered mean_chi
- Expect: middle layers (14-20) with moderate alpha (0.5-1.5) will
  be the sweet spot. Late layers (24-27) will need very low alpha.

**Success criteria:**
- At least 3 (layer, alpha) pairs where steering improves CHI
  while keeping responses alive and substantive (>50 tokens avg)
- Document the best setting for each of the 3 vector types

**Time:** 3-4 hours (lots of model runs — ~45 grid points × 5 turns)
**Git:** `git commit -m "TASK-039: systematic alpha/layer sweep for steering calibration"`

---

### TASK-040: Re-run Steering Experiment with Calibrated Parameters
**Status:** ⬜ TODO
**Do:** Create `scripts/steering_experiment_v2.py`

Week 6 experiment (TASK-038) used layer 27 + alpha 2.0 — over-steered.
TASK-039 sweep found the Goldilocks zone. Now re-run with real parameters
and push context INTO the degradation zone (past turn 10, past 2000 tokens).

---

#### Calibrated Parameters from TASK-039

| Vector | Layer | Alpha | Rationale |
|--------|-------|-------|-----------|
| task_focus | 16 | 0.5 | Best sweep result: CHI 0.953, +0.124 vs baseline, all responses alive |
| tool_format | 16 | 0.5 | Start with same layer/alpha, adjust if needed |
| code_coherence | 16 | 0.5 | Start with same layer/alpha, adjust if needed |

**Critical:** Use RAW vectors, NOT normalized. Normalized vectors showed
zero differentiation across all configs in TASK-039 (identical CHI to 4
decimal places — the perturbation is too small to affect the model).

---

#### Why This Experiment Is Different from TASK-038

TASK-038 problem: 5 turns, ~688 tokens max context. Degradation doesn't
start until turn 10 (~2000-2500 tokens per Week 2 findings). We were
measuring steering effect when the model didn't need help.

This experiment MUST push past the degradation inflection point.

---

#### Experiment Protocol

```
Model:      mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
Turns:      10 (same EXPERIMENT_TURNS as TASK-038)
max_tokens: 300 (up from 120 — need real responses, not truncated)
Runs:       4 total (1 unsteered baseline + 3 vector types)
```

**EXPERIMENT_TURNS** (reuse from TASK-038):
```python
EXPERIMENT_TURNS = [
    "Create a Calculator class with add and subtract methods",
    "Add multiply and divide methods. Handle division by zero.",
    "Write unit tests for all four operations",
    "Add a history feature that stores the last 10 operations",
    "Refactor to use a Strategy pattern for the operations",
    "Add logging to track which operations are called most",
    "Write a CLI wrapper so the calculator can be used from terminal",
    "Add support for chaining operations like calc.add(1).multiply(3)",
    "Write integration tests for the CLI",
    "Add a square root method and update all tests",
]
```

**For each run (unsteered + 3 vectors), record per turn:**
1. `turn` — turn number (0-9)
2. `chi_score` — smoothed CHI
3. `chi_raw` — unsmoothed CHI (for debugging)
4. `entropy`, `repetition`, `goal_drift` — raw signal values
5. `response_length` — token count (CRITICAL: track this to rule out truncation confound)
6. `context_length` — cumulative context tokens
7. `response_preview` — first 100 chars of response (for qualitative review)

---

#### Thermal Management

The M4 Pro heats up during long runs. Build in cooldown:

```python
import time

COOLDOWN_BETWEEN_RUNS = 120  # 2 minutes between each full 10-turn run

for vector_name in [None, "task_focus", "tool_format", "code_coherence"]:
    run_experiment(vector_name, layer=16, alpha=0.5)
    print(f"Cooldown {COOLDOWN_BETWEEN_RUNS}s...")
    time.sleep(COOLDOWN_BETWEEN_RUNS)
```

Alternatively: run each condition as a separate invocation with a CLI arg:
```bash
python scripts/steering_experiment_v2.py --condition baseline
python scripts/steering_experiment_v2.py --condition task_focus
python scripts/steering_experiment_v2.py --condition tool_format
python scripts/steering_experiment_v2.py --condition code_coherence
python scripts/steering_experiment_v2.py --plot-only  # combine all into one graph
```

This lets you run one at a time and let the machine cool between runs.

---

#### Output Files

- `data/steering_experiment_v2.json` — full data for all 4 conditions
- `data/steering_experiment_v2.png` — THE MONEY GRAPH:
  - 4 CHI curves (unsteered + 3 vectors) on one plot
  - X-axis: turns 0-9
  - Y-axis: CHI score (0.0-1.0)
  - Horizontal line at 0.7 threshold (Level 2 trigger)
  - Legend with mean CHI per condition
  - Secondary info: response length per turn per condition (small subplot or annotation)

---

#### Success Criteria (ALL must pass)

1. **Responses alive:** Steered responses are >20 tokens on ALL 10 turns for ALL 3 vectors.
   If any vector kills output, note it and proceed with the vectors that work.

2. **CHI improvement after turn 5:** Mean CHI for turns 5-9 is higher for
   at least ONE steered condition vs unsteered. This is where degradation
   should be visible in the unsteered run.

3. **Length parity:** Mean response length for steered runs is within 50%
   of unsteered mean. If steered = 50 tokens and unsteered = 300 tokens,
   the CHI improvement is fake (just truncation). Flag this explicitly.

4. **Qualitative check:** Manually review turns 7-9 for unsteered vs best
   steered. Write 2-3 sentences in the session log about whether the
   steered responses are actually more focused/coherent or just shorter.

---

#### If Steering Shows No Improvement at 10 Turns

This is possible. The 5-turn sweep showed CHI was ~0.83 at baseline and
~0.95 steered, but that was pre-degradation. If at 10 turns the unsteered
run is still healthy (CHI > 0.7), then:

- Extend to 15 turns to push further into degradation
- OR reduce max_tokens to 200 to accumulate context faster
- OR use a more challenging task sequence that causes faster drift

Document what you find either way. Negative results are still results.

---

#### What This Tells Us for TASK-041

The best-performing vector from this experiment becomes the DEFAULT for
CHI-triggered steering in the agent loop. If task_focus wins, wire that
in first. If code_coherence is stronger, use that instead. The data
decides, not our assumptions.

---

**Time:** 2-3 hours (4 runs × ~15 min each + plotting + analysis)
**Git:** `git commit -m "TASK-040: calibrated steering experiment — 10 turns, all 3 vectors, response length tracked"`


---

### TASK-041: Wire Steering into the Agent Loop (CHI-Triggered)
**Status:** ⬜ TODO
**Do:** Modify `cognition_engine/agent.py` to integrate `SteeringEngine`

This is the core product integration. When CHI drops below 0.7, the
agent activates steering automatically. When CHI recovers above 0.8,
steering deactivates. This is Level 2 of the intervention ladder.

---

#### Changes from Original TASK-041 (Based on TASK-040 Results)

| Original | Updated | Why |
|----------|---------|-----|
| Use `task_focus` vector | Use `tool_format` vector | task_focus kills output at turn 5. tool_format passed all 10 turns with length parity. |
| Load vectors in the if-block | Load vectors ONCE at `__init__`, toggle `activate()`/`deactivate()` | Avoid repeated disk I/O and layer wrapping mid-session. |
| Hardcoded thresholds | Configurable thresholds via constants | Need to tune activation/deactivation thresholds based on real agent runs. |

---

#### Implementation

```python
# --- Constants (top of agent.py or in a config) ---

STEERING_VECTOR = "tool_format"      # TASK-040 winner: all turns alive, +0.011 CHI
STEERING_LAYER = 16                  # TASK-039 sweep: most stable layer
STEERING_ALPHA = 0.5                 # TASK-039 sweep: safe zone
STEERING_ACTIVATE_THRESHOLD = 0.7    # CHI below this → activate steering
STEERING_DEACTIVATE_THRESHOLD = 0.8  # CHI above this → deactivate (0.1 hysteresis band)


# --- In CognitionAgent.__init__ ---

from cognition_engine.steering.injector import SteeringEngine

self.steering_enabled = enable_steering  # from --enable-steering CLI flag
self.steering_active = False

if self.steering_enabled:
    self.steering = SteeringEngine(model)
    # Load and wrap layer ONCE at init — vector stays in memory
    self.steering.load_vectors(
        STEERING_VECTOR,
        layer_idx=STEERING_LAYER,
        alpha=STEERING_ALPHA
    )
    # Start deactivated — only steer when CHI demands it
    self.steering.deactivate()
    print(f"[STEERING] Loaded {STEERING_VECTOR} at layer {STEERING_LAYER}, "
          f"alpha={STEERING_ALPHA} (inactive until CHI < {STEERING_ACTIVATE_THRESHOLD})")


# --- In the agent loop, AFTER CHI update each turn ---

if self.steering_enabled:
    chi = self.monitor.chi_history[-1]  # latest smoothed CHI

    if chi < STEERING_ACTIVATE_THRESHOLD and not self.steering_active:
        # CHI dropped — activate steering
        self.steering.activate()
        self.steering_active = True
        print(f"[STEERING] ⚡ ACTIVATED at turn {turn} — CHI={chi:.3f}")

    elif chi > STEERING_DEACTIVATE_THRESHOLD and self.steering_active:
        # CHI recovered — deactivate steering
        self.steering.deactivate()
        self.steering_active = False
        print(f"[STEERING] ✓ Deactivated at turn {turn} — CHI={chi:.3f}")


# --- In the agent cleanup (finally block) ---

if self.steering_enabled:
    self.steering.cleanup()  # restore original layers
```

---

#### CLI Integration

Add `--enable-steering` flag to the `agent` subcommand in `cli.py`:

```python
agent_parser.add_argument(
    "--enable-steering", action="store_true",
    help="Enable CHI-triggered activation steering (Level 2 intervention)"
)
```

Pass it through to `CognitionAgent.__init__`:

```python
agent = CognitionAgent(
    model, tokenizer, task=args.task,
    working_dir=args.dir,
    enable_steering=args.enable_steering,
    # ... other args
)
```

---

#### What to Log Each Turn (Steering Status Line)

Extend the existing CHI status line to include steering state:

```
Turn  7 | CHI=0.683 [E:0.04 R:0.89 D:0.23] | STEERING: ON (alpha=0.5) | Tools: write
Turn  8 | CHI=0.742 [E:0.03 R:0.45 D:0.21] | STEERING: ON (alpha=0.5) | Tools: exec
Turn  9 | CHI=0.812 [E:0.05 R:0.12 D:0.19] | STEERING: OFF            | Tools: think
```

---

#### Manual Test

Run `scripts/test_agent.py` with steering enabled:

```bash
caffeinate -dimsu .venv/bin/python3 -u scripts/test_agent.py --enable-steering
```

**Observe and record in session log:**
1. Which turn does steering first activate? (Expected: around turn 7-10
   based on TASK-040 baseline degradation pattern)
2. Does output stay alive (>20 tokens) when steering is active?
   (tool_format kept all turns alive in TASK-040 — should hold here too)
3. Does the agent still follow [READ]/[WRITE]/[EXEC] protocol while steered?
   (tool_format vector was trained on format compliance — should help, not hurt)
4. Does CHI recover after steering activates, even partially?
5. Does the agent complete the task? (Files created, tests pass?)

---

#### Edge Cases to Handle

1. **Steering vectors not found on disk:**
   If `data/steering_vectors/tool_format.npz` doesn't exist, print a
   warning and continue without steering. Don't crash the agent.

   ```python
   if self.steering_enabled:
       try:
           self.steering = SteeringEngine(model)
           self.steering.load_vectors(STEERING_VECTOR, ...)
           self.steering.deactivate()
       except FileNotFoundError:
           print("[STEERING] Warning: vectors not found, steering disabled")
           self.steering_enabled = False
   ```

2. **Steering + compaction interaction:**
   If context compaction triggers (Level 3, CHI < 0.5), steering should
   stay active — compaction and steering work together. Don't deactivate
   steering when compaction fires. The intervention ladder is additive:
   Level 2 (steering) stays on when Level 3 (compaction) activates.

3. **Session save/resume:**
   Save `steering_active` state in `SessionManager` so a resumed session
   knows whether steering was active when interrupted.

---

#### What NOT to Do This Task

- Don't implement adaptive alpha yet (that's TASK-042)
- Don't add multi-vector steering yet (that's TASK-043)
- Don't tune the thresholds yet — use 0.7/0.8 as defaults and collect data
- Don't run code_coherence as a secondary vector — keep it single-vector

---

**Time:** 2-3 hours
**Git:** `git commit -m "TASK-041: CHI-triggered steering in agent loop (tool_format, layer 16, alpha 0.5)"`


---

### TASK-042: Adaptive Alpha — CHI Drives Steering Strength
**Status:** ⬜ TODO
**Do:** Add adaptive alpha to `SteeringEngine` + verify Level 3 compaction fires

---

#### Why This Task Exists

TASK-041 showed fixed alpha 0.5 doesn't arrest decline. CHI fell from
0.67 → 0.53 over 17 turns with steering ON the entire time. Two fixes:
1. Push harder when CHI is low (adaptive alpha — this task)
2. Make sure Level 3 compaction actually fires when steering fails
   (compaction check — also this task, because if Level 3 is broken,
   adaptive alpha alone won't save anything)

---

#### Part A: Adaptive Alpha Formula

```python
def compute_adaptive_alpha(chi_score: float, base_alpha: float = 0.5) -> float:
    """
    Lower CHI = stronger steering. Piecewise linear.

    CRITICAL: base_alpha default is 0.5, NOT 1.0.
    TASK-039 proved alpha >= 1.0 kills output at all tested layers.
    Max effective alpha is base_alpha * 1.5 = 0.75 (untested but below
    the 1.0 cliff). If 0.75 causes dead output, cap at 1.2x (0.60).

    CHI >= 0.8  → alpha = 0.0   (healthy, no steering)
    CHI  = 0.7  → alpha = 0.3   (mild — steering just activated)
    CHI  = 0.5  → alpha = 0.5   (moderate — matching TASK-040 level)
    CHI  = 0.3  → alpha = 0.75  (strong — last resort before Level 3/4)
    CHI  < 0.3  → alpha = 0.75  (capped — escalate to compaction/cloud)

    The key insight: at CHI 0.7 (activation threshold), start GENTLE.
    At CHI 0.5 (where TASK-041 failed), match the proven TASK-040 alpha.
    Only push past 0.5 when we're in deep trouble (CHI < 0.5).
    """
    if chi_score >= 0.8:
        return 0.0
    elif chi_score >= 0.3:
        # Linear from 0.0 at CHI=0.8 to 1.5*base at CHI=0.3
        t = (0.8 - chi_score) / 0.5  # 0.0 at CHI=0.8, 1.0 at CHI=0.3
        return base_alpha * (0.6 + t * 0.9)  # 0.3 at CHI=0.7, 0.75 at CHI=0.3
    else:
        return base_alpha * 1.5  # 0.75 — capped, don't go higher

# Quick reference table (base_alpha=0.5):
# CHI 0.80 → alpha 0.00
# CHI 0.75 → alpha 0.18
# CHI 0.70 → alpha 0.30  ← steering activates here
# CHI 0.65 → alpha 0.39
# CHI 0.60 → alpha 0.48
# CHI 0.55 → alpha 0.57
# CHI 0.50 → alpha 0.50  ← TASK-040 experiment level
# CHI 0.40 → alpha 0.66
# CHI 0.30 → alpha 0.75  ← max before escalation
```

---

#### Part B: Integrate into Agent Loop

```python
# In the agent loop, AFTER CHI update each turn:

if self.steering_enabled:
    chi = self.monitor.chi_history[-1]

    # Compute adaptive alpha regardless of whether steering is active
    alpha = compute_adaptive_alpha(chi, base_alpha=STEERING_ALPHA)

    if chi < STEERING_ACTIVATE_THRESHOLD and not self.steering_active:
        self.steering.activate()
        self.steering_active = True
        self.steering.set_alpha(alpha)
        print(f"[STEERING] ⚡ ACTIVATED at turn {turn} — CHI={chi:.3f}, alpha={alpha:.3f}")

    elif chi > STEERING_DEACTIVATE_THRESHOLD and self.steering_active:
        self.steering.deactivate()
        self.steering_active = False
        print(f"[STEERING] ✓ Deactivated at turn {turn} — CHI={chi:.3f}")

    elif self.steering_active:
        # Steering already active — update alpha based on current CHI
        self.steering.set_alpha(alpha)
```

Status line update:
```
Turn  7 | CHI=0.683 | STEERING: ON (alpha=0.390) | Tools: write
Turn  8 | CHI=0.612 | STEERING: ON (alpha=0.480) | Tools: exec
Turn  9 | CHI=0.812 | STEERING: OFF              | Tools: think
```

---

#### Part C: Verify Level 3 Compaction Actually Fires

TASK-041 showed CHI dropped to 0.526 — well below the Level 3 threshold
(CHI < 0.5 per intervention ladder). Compaction should have fired but
didn't (`Compactions: 0`).

**Check in agent.py:**
1. Is `needs_compaction(chi_score)` being called each turn?
2. Is it checking `chi_score < 0.5`?
3. If yes, does it actually call `compact()` and reset context?

If compaction is NOT wired in the agent loop, wire it now:

```python
# After steering update, check if Level 3 needed:
if chi < 0.5:
    if self.compactor.needs_compaction(chi):
        print(f"[COMPACTION] Level 3 triggered at turn {turn} — CHI={chi:.3f}")
        self.compactor.compact(strategy="auto")
```

**This is critical.** Without Level 3, the intervention ladder is broken.
Steering (Level 2) slows decline but doesn't reverse it. Compaction
(Level 3) resets context which is the actual fix for accumulated garbage.

---

#### Manual Test

Run the same scenario that failed in TASK-041:

```bash
caffeinate -dimsu .venv/bin/python3 -u -m cognition_engine agent \
  --task "Create calculator.py and tests, run pytest" \
  --dir /tmp/task042_test \
  --enable-steering \
  --max-turns 20
```

**Observe and record:**
1. Does alpha increase as CHI drops? (Print the alpha value each turn)
2. Does higher alpha help more than fixed 0.5 did in TASK-041?
3. Does Level 3 compaction fire if CHI drops below 0.5?
4. After compaction fires, does CHI recover?
5. Does the model ever break out of the [THINK] loop?

**Realistic expectation:** Adaptive alpha alone probably won't fix the
[THINK] loop problem. The model gets stuck because accumulated context
is confusing it, not because the steering vector is too weak. Compaction
(clearing old context) is the real fix. Adaptive alpha + compaction
together should produce the recovery that neither achieves alone.

---

#### What NOT to Do

- Don't implement PID (integral/derivative). This is proportional-only.
  Note in code: "P-only controller. I/D components deferred to Week 8
  if proportional response is insufficient."
- Don't change the vector (still tool_format) or layer (still 16)
- Don't implement multi-vector yet (TASK-043)

---

**Time:** 2-3 hours (formula is simple, but verifying compaction fires adds scope)
**Git:** `git commit -m "TASK-042: adaptive alpha + verify Level 3 compaction fires"`


---

### TASK-043: Multi-Vector Steering (Reduced Scope)
**Status:** ⬜ TODO
**Do:** Test dual-vector steering with the two vectors that work

---

#### Why Reduced Scope

TASK-040 results:
- task_focus: **DEAD.** Kills output at turn 5. Do not use.
- tool_format: Passed all 10 turns. Reliable.
- code_coherence: Strong CHI improvement, died on turn 9.

Only two vectors survived TASK-040. Multi-vector means two vectors, not
three. And TASK-041/042 showed that steering alone can't recover a
degenerate model — the urgent need is the full intervention ladder
(TASK-044), not adding more vectors.

**Treat this as a quick experiment, not a deep engineering task.**

---

#### Safe Parameters (From Actual Data)

```python
# WRONG — from original task (will kill output):
engine.load_vectors("task_focus", layer_idx=17, alpha=0.8)    # dead vector, lethal alpha
engine.load_vectors("tool_format", layer_idx=20, alpha=0.5)   # layer 20 untested for tool_format
engine.load_vectors("code_coherence", layer_idx=14, alpha=0.6) # alpha too high

# RIGHT — based on TASK-039/040 results:
engine.load_vectors("tool_format", layer_idx=16, alpha=0.3)
engine.load_vectors("code_coherence", layer_idx=17, alpha=0.25)
```

**Why these values:**
- tool_format at layer 16: proven safe in TASK-040 at alpha 0.5.
  Reduce to 0.3 because a second vector adds cumulative perturbation.
- code_coherence at layer 17: different layer to avoid interference.
  Alpha 0.25 because code_coherence died at turn 9 with alpha 0.5 —
  halving should keep it alive.
- Total perturbation budget: ~0.55 combined, staying well under the
  single-vector ceiling of 0.5 that was stable.

---

#### Test Protocol (Quick — 1 Hour Max)

Run a single 10-turn experiment comparing 3 conditions.
Reuse the TASK-040 experiment framework:

```bash
# Condition 1: tool_format only (already have this data from TASK-040)
# Condition 2: code_coherence only (already have this data from TASK-040)
# Condition 3: both vectors simultaneously (NEW)
python scripts/steering_experiment_v2.py --condition dual_vector
```

**For condition 3, measure:**
1. Are all 10 turns alive (>20 tokens)?
2. Mean CHI turns 5-9 vs baseline vs single-vector runs
3. Response length parity with baseline
4. Does dual-vector beat the best single-vector (code_coherence)?

---

#### Possible Outcomes and Decisions

**Outcome A — Dual beats single:** Use dual-vector as default config.
Update TASK-041 constants to load both vectors.

**Outcome B — Dual matches single:** No benefit. Stick with tool_format
only (simpler, fewer moving parts).

**Outcome C — Dual degrades vs single:** Vectors interfere. Stick with
single vector. Multi-vector needs different layers or lower alphas.

**Outcome D — Dual kills output:** Combined perturbation is too strong.
Reduce both alphas further or abandon multi-vector for this model size.

---

#### If You're Short on Time — Skip to TASK-044

The findings from TASK-041 and TASK-042 make it clear that the
bottleneck is not "steering isn't strong enough" — it's "the full
intervention ladder doesn't fire." TASK-044 tests the complete system:
CHI monitoring → steering activation → compaction → recovery.

That matters more than a second steering vector.

Mark TASK-043 as DEFERRED and move to TASK-044 with a note:
"Multi-vector deferred — single-vector tool_format is sufficient for
integration testing. Revisit Week 8 if single-vector results are
insufficient in production agent runs."

---

**Time:** 1-2 hours (reduced from 2-3 — quick A/B test, not deep engineering)
**Git:** `git commit -m "TASK-043: dual-vector steering test (tool_format + code_coherence)"`


---

### TASK-044: End-to-End Integration Test — Full Intervention Ladder
**Status:** ✅ DONE (harness + agent hooks; run A/B/C locally for data)
**Do:** Create `scripts/test_steering_agent.py`

---

#### What Changed Since This Task Was Written

The original TASK-044 assumed: steering activates → behavior corrects → task completes.
Three runs (TASK-041 run 2, TASK-041 run 3, TASK-042) proved that wrong.
Steering alone does NOT recover a model stuck in a degenerate loop.

What DID work: TASK-041 Run 1 (test_agent.py with calculator task).
That run used the TASK-027 prompt, the model followed tools from the start,
CHI dipped to 0.698 and recovered to 0.811. Files created, tests run.

This task now tests the FULL intervention ladder, not just steering:
- Level 1: Normal inference (CHI > 0.8)
- Level 2: Steering activation (CHI < 0.7) with adaptive alpha
- Level 3: Context compaction (CHI < 0.5)
- Level 4: Cloud escalation (CHI < 0.3) — log only, don't implement

---

#### Task Choice: Use the Calculator Task, Not Flask

The original specified a Flask REST API task. That's too complex.
The ONLY task that has ever completed end-to-end in testing is the
calculator task from test_agent.py. Use what works:

```python
TASK = (
    "Create a Python file called calculator.py with a Calculator class "
    "that has add, subtract, multiply, and divide methods. "
    "Then create test_calculator.py with tests for each method. "
    "Run the tests to verify they pass."
)
```

Once the full ladder works on the calculator task, you can try harder
tasks in Week 8. Don't change two variables at once.

---

#### Test Protocol: 3 Runs, Compared

Run the same task under 3 configurations to isolate what helps:

**Run A — No steering, no compaction (pure baseline):**
```bash
caffeinate -dimsu .venv/bin/python3 -u scripts/test_steering_agent.py \
  --mode baseline --dir /tmp/task044_baseline --max-turns 20
```

**Run B — Steering only (Level 2, no compaction):**
```bash
caffeinate -dimsu .venv/bin/python3 -u scripts/test_steering_agent.py \
  --mode steering --dir /tmp/task044_steering --max-turns 20
```

**Run C — Full ladder (steering + compaction):**
```bash
caffeinate -dimsu .venv/bin/python3 -u scripts/test_steering_agent.py \
  --mode full_ladder --dir /tmp/task044_full --max-turns 20
```

Cooldown 2 minutes between runs for thermal management.

---

#### What Each Mode Enables

| | Baseline | Steering | Full Ladder |
|---|---------|----------|-------------|
| CHI monitoring | ✅ | ✅ | ✅ |
| Level 2: Steering (tool_format, L16, adaptive alpha) | ❌ | ✅ | ✅ |
| Level 3: Compaction (auto strategy when CHI < 0.5) | ❌ | ❌ | ✅ |
| Level 4: Cloud escalation | ❌ | ❌ | Log only |

---

#### Per-Turn Logging (All Runs)

Each turn must record:

```python
{
    "turn": 0,
    "chi_score": 0.785,
    "chi_raw": 0.801,
    "intervention_level": 2,
    "steering_active": false,
    "steering_alpha": 0.0,
    "compaction_fired": false,
    "compaction_strategy": null,
    "tool_used": "write",
    "response_length": 288,
    "context_length": 312,
    "files_modified": ["calculator.py"],
    "response_preview": "class Calculator:..."
}
```

Save to: `data/task044_{mode}.json`

---

#### Success Criteria

**Minimum bar (at least ONE run must achieve all of these):**
1. calculator.py created with working Calculator class
2. test_calculator.py created with tests for all 4 operations
3. pytest executed at least once
4. Agent reached [DONE] or completed within max-turns

**Comparison metrics (record for all 3 runs):**
1. Task completed? (yes/no)
2. Turns to completion (fewer = better)
3. Total tool calls (write + exec, excluding think)
4. CHI min / final
5. Turns spent in [THINK]-only loop (0 = good)
6. Steering activations count
7. Compaction firings count (Run C only)

---

#### What to Watch For

**The [THINK] loop problem:**
TASK-041/042 showed the model gets stuck producing only [THINK] for
20+ turns. If this happens in Run B (steering only), that confirms
steering can't fix it. If Run C (full ladder) breaks out of the loop
after compaction fires, that's the money finding — compaction is the
recovery mechanism.

**Compaction as reset:**
When CHI drops below 0.5 and compaction fires, it should:
1. Clear old conversation turns
2. Keep the anchor (original task + system prompt)
3. Keep recent turns (last 2-5)
4. Reset context length dramatically

After compaction, the model gets essentially a fresh start with the
task still in context. This SHOULD break the [THINK] loop because
the degenerate accumulated context is gone.

**The comparison that matters:**
- If Run A and Run B both fail but Run C succeeds → compaction is the hero
- If Run B succeeds but Run A fails → steering actually helped
- If all three succeed → the task is too easy, need a harder test
- If all three fail → deeper agent/prompt issues, not an intervention problem

---

#### Output Files

- `data/task044_baseline.json` — full turn-by-turn data
- `data/task044_steering.json` — full turn-by-turn data
- `data/task044_full_ladder.json` — full turn-by-turn data
- `data/task044_comparison.png` — 3 CHI curves on one graph
  (like TASK-040 money graph but for the agent loop)
- `data/task044_report.md` — written summary:
  - Which runs completed the task
  - Turn-by-turn narrative of interesting events
  - Which intervention level made the difference
  - Honest assessment of what works and what doesn't

---

#### After the Test: Update COFOUNDER_BRAIN.md

Whatever happens, document it honestly:

**If full ladder works:**
```
- Week 7 capstone: Full intervention ladder proven end-to-end.
  CHI monitoring → steering (Level 2) → compaction (Level 3) → recovery.
  Steering slows decline. Compaction resets context. Together they work.
```

**If nothing works:**
```
- Week 7 capstone: Intervention ladder tested, results mixed.
  Steering slows CHI decline but doesn't recover stuck models.
  Compaction fires but model re-enters degenerate state.
  Root cause: 7B model's instruction following degrades past ~2500 tokens
  regardless of intervention. May need better prompting or larger model.
```

Both outcomes are valuable. The demo video story changes, but the
product thesis (local models need help staying coherent) is validated
either way.

---

#### Realistic Expectations

Based on all data so far:
- TASK-041 Run 1 completed the calculator task in 15 turns WITHOUT
  steering explicitly helping (CHI stayed above 0.7 most of the time).
- Runs 2-3 failed because the model entered [THINK] loops immediately.
- The difference was likely prompt/model randomness, not steering.

So the honest prediction: Run A might complete, Run B might complete,
Run C might complete — or none of them might. The 7B model's tool
following is inconsistent. What we're measuring is whether the
intervention ladder HELPS on average, not whether it guarantees success.

If you run each mode 3 times (9 runs total) you'd get statistical
signal. But that's 3+ hours of compute. One run each is sufficient
for Week 7 — note the variance caveat in the report.

---

**Time:** 2-3 hours (3 runs × ~15 min each + cooldown + analysis + report)
**Git:** `git commit -m "TASK-044: end-to-end integration test — full intervention ladder"`

---

## WEEK 7 CHECKLIST
- [ ] TASK-039: Alpha/layer sweep — find the Goldilocks zone (3-4 hrs)
- [ ] TASK-040: Re-run experiment with calibrated params (2-3 hrs)
- [ ] TASK-041: Wire steering into agent loop, CHI-triggered (2-3 hrs)
- [ ] TASK-042: Adaptive alpha — CHI controls strength (1-2 hrs)
- [ ] TASK-043: Multi-vector steering at different layers (2-3 hrs)
- [x] TASK-044: Full integration test — prove the system works (2-3 hrs)

**Total: ~13-18 hours across 3-4 days**

**Week 7 milestone:** The agent activates steering when CHI drops,
uses adaptive alpha proportional to degradation, and can apply
multiple steering vectors simultaneously. End-to-end integration
test shows the full monitoring → intervention → correction loop.

---

## NOTES

### On the Week 6 Empty Response Problem
The steered model producing 0-length responses is a known failure mode
of over-steering. It happens because the steering vector pushes the
residual stream so far from the model's natural distribution that the
model assigns high probability to EOS or produces degenerate output.

Fix: lower alpha, use middle layers, or normalize the steering vector
before injection (divide by its norm, then multiply by alpha — this
decouples direction from magnitude).

### Qwen3.5 Thinking Mode as Free Level 2 (Investigate)
From Week 0 notes: Qwen3.5 has toggle reasoning on/off. When CHI drops,
enabling thinking mode could be a zero-cost Level 2 intervention that
doesn't require steering vectors at all. This is worth testing in Week 8
as a complementary approach. Steering + thinking mode together could be
more powerful than either alone.

### Looking Ahead — Week 8-9
- Week 8: CAST-style conditional steering (only steer specific failure modes)
- Week 9: Context compaction + steering combined (Level 2+3 working together)
- Week 10: Production hardening, benchmarks, performance optimization

### 📌 BRAIN UPDATE (add to COFOUNDER_BRAIN.md):
```
- WEEK 6: Activation steering foundations (TASKS 033-038) — COMPLETE
  Pipeline working: pairs → extraction → vectors → injection → measurement.
  Proved steering changes behavior. Over-steering at layer 27/alpha 2.0
  kills output. Middle layers + moderate alpha are the sweet spot.
- WEEK 7: Steering calibration + agent integration (TASKS 039-044)
  Alpha/layer sweep, adaptive alpha, CHI-triggered steering in agent loop.
  Multi-vector steering. End-to-end integration test.
```
