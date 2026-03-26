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
**Do:** Add adaptive alpha to `SteeringEngine`

Fixed alpha is a blunt instrument. The field guide warns that steering
strength should vary based on how much correction is needed. This is
where CHI becomes the control signal for a feedback loop.

**Formula:**
```python
def compute_adaptive_alpha(chi_score: float, base_alpha: float = 1.0) -> float:
    """
    Lower CHI = stronger steering. Linear interpolation.
    
    CHI 0.8+ → alpha = 0.0 (no steering needed)
    CHI 0.7  → alpha = base_alpha * 0.5 (mild correction)
    CHI 0.5  → alpha = base_alpha * 1.0 (full correction)
    CHI 0.3  → alpha = base_alpha * 1.5 (strong correction)
    CHI < 0.3 → escalate to Level 3/4 (steering alone won't save it)
    """
    if chi_score >= 0.8:
        return 0.0
    elif chi_score >= 0.3:
        # Linear from 0.0 at CHI=0.8 to 1.5*base at CHI=0.3
        t = (0.8 - chi_score) / 0.5  # 0.0 at CHI=0.8, 1.0 at CHI=0.3
        return base_alpha * (0.5 + t * 1.0)  # 0.5x to 1.5x base
    else:
        return base_alpha * 1.5  # max steering before escalation
```

**Integrate into agent loop:**
```python
# After CHI update each turn:
alpha = compute_adaptive_alpha(chi_score, base_alpha=BEST_ALPHA)
self.steering.set_alpha(alpha)
```

**This is our implementation of the CAST/PID paradigm:**
- CAST = conditional activation (steer only when needed)
- PID-based steering = strength proportional to error signal
- We combine both: CHI triggers activation AND controls magnitude

**Manual test:** Run a 15-turn agent session. Verify:
- Alpha starts at 0 when CHI is healthy
- Alpha increases as CHI degrades
- Alpha prints in the CHI status line so you can see it changing

**Time:** 1-2 hours
**Git:** `git commit -m "TASK-042: adaptive alpha — CHI controls steering strength"`

---

### TASK-043: Multi-Vector Steering (Different Vectors at Different Layers)
**Status:** ⬜ TODO
**Do:** Extend `SteeringEngine` to support multiple simultaneous vectors

Right now: one vector, one layer. But you have 3 vector types that
fix different problems. The field guide recommends injecting different
vectors at DIFFERENT layers to avoid interference.

**Implementation:**
```python
# SteeringEngine already supports this — just call load_vectors multiple times
engine = SteeringEngine(model)
engine.load_vectors("task_focus", layer_idx=17, alpha=0.8)
engine.load_vectors("tool_format", layer_idx=20, alpha=0.5)
engine.load_vectors("code_coherence", layer_idx=14, alpha=0.6)
```

Verify that `_wrap_layer` correctly handles multiple layers by checking
that `self.active_wrappers` and `self.originals` track all three.

**Test:** Run a generation with all 3 vectors active. Compare to:
- No steering
- task_focus only
- All three vectors

**Check for interference:** Does adding more vectors degrade output
quality vs single vector? If yes, the vectors are interfering and
you need to adjust layers/alphas.

**Document which combination works best** — this becomes the default
"full steering" configuration for the agent.

**Time:** 2-3 hours
**Git:** `git commit -m "TASK-043: multi-vector steering at different layers"`

---

### TASK-044: Steering Integration Test + Session Log Validation
**Status:** ⬜ TODO
**Do:** Create `scripts/test_steering_agent.py`

End-to-end test: run the agent on a real coding task with steering
enabled, observe the full intervention ladder in action.

**Test scenario:**
Task: "Build a REST API with Flask — create app.py, add 3 endpoints
(GET /items, POST /items, DELETE /items/:id), write tests, and run them."

This task is long enough (10-15 turns) to trigger CHI degradation
and observe steering kicking in.

**What to observe and log:**
1. Which turn does steering first activate?
2. Does the agent still follow [READ]/[WRITE]/[EXEC] protocol while steered?
3. Does tool_format vector help with protocol compliance?
4. Does CHI recover after steering activates (even partially)?
5. Does the agent complete the task? (Files created, tests pass?)

**Output:** A detailed session report in `data/steering_agent_test.md`:
- Turn-by-turn CHI, alpha, steering state
- Which files were created
- Whether tests passed
- Any anomalies (empty responses, protocol violations, crashes)

**This is the proof that the full system works:**
CHI monitoring → degradation detected → steering activated →
behavior corrected → task completed. That's the product.

**Time:** 2-3 hours (mostly observing and documenting)
**Git:** `git commit -m "TASK-044: end-to-end steering agent integration test"`

---

## WEEK 7 CHECKLIST
- [ ] TASK-039: Alpha/layer sweep — find the Goldilocks zone (3-4 hrs)
- [ ] TASK-040: Re-run experiment with calibrated params (2-3 hrs)
- [ ] TASK-041: Wire steering into agent loop, CHI-triggered (2-3 hrs)
- [ ] TASK-042: Adaptive alpha — CHI controls strength (1-2 hrs)
- [ ] TASK-043: Multi-vector steering at different layers (2-3 hrs)
- [ ] TASK-044: Full integration test — prove the system works (2-3 hrs)

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
