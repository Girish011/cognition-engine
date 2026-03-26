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

### TASK-040: Re-run TASK-038 Experiment with Calibrated Parameters
**Status:** ⬜ TODO
**Do:** Update `scripts/steering_experiment.py` or create `scripts/steering_experiment_v2.py`

Re-run the full 10-turn experiment using the best (layer, alpha) pair
found in TASK-039. This time you need to verify:

1. Steered responses are non-empty on ALL turns (>20 tokens each)
2. Steered CHI is higher than unsteered after turn 5
3. Responses are qualitatively different (manual review of 3-4 turns)

**Also add to the experiment:**
- Run all 3 vector types separately: task_focus, tool_format, code_coherence
- Compare: which vector has the most impact on CHI?
- This tells you which steering target matters most for your agent

**Output:**
- `data/steering_experiment_v2.json` — full data for all 3 vectors
- `data/steering_experiment_v2.png` — 4 CHI curves (unsteered + 3 vectors)

**This is the real money graph.** The Week 6 graph was a proof of concept.
This one needs to show actual improvement with actual responses.

**Time:** 2-3 hours
**Git:** `git commit -m "TASK-040: calibrated steering experiment — all 3 vectors"`

---

### TASK-041: Wire Steering into the Agent Loop (CHI-Triggered)
**Status:** ⬜ TODO
**Do:** Modify `cognition_engine/agent.py` to integrate `SteeringEngine`

This is the core product integration. When CHI drops below 0.7, the
agent activates steering automatically. When CHI recovers above 0.8,
steering deactivates. This is Level 2 of the intervention ladder.

```python
# In CognitionAgent.__init__:
from cognition_engine.steering.injector import SteeringEngine

self.steering = SteeringEngine(model)
self.steering_active = False

# In the agent loop, after CHI update:
chi = monitor.get_smoothed_chi()
intervention = monitor.get_intervention_level()

if intervention >= 2 and not self.steering_active:
    # CHI dropped — activate steering
    self.steering.load_vectors("task_focus", layer_idx=BEST_LAYER, alpha=BEST_ALPHA)
    self.steering.activate()
    self.steering_active = True
    print("[STEERING] Activated — CHI dropped below threshold")

elif intervention < 2 and self.steering_active:
    # CHI recovered — deactivate steering
    self.steering.deactivate()
    self.steering_active = False
    print("[STEERING] Deactivated — CHI recovered")
```

**Key design decisions:**
- Use **only task_focus** vector for now (simplest, most relevant to agent drift)
- Load vectors once at agent init, toggle active/inactive based on CHI
- Log every steering activation/deactivation with turn number and CHI value
- Add `--enable-steering` flag to CLI so it's opt-in during testing

**Manual test:**
Run `scripts/test_agent.py` with steering enabled. Observe:
- Does steering activate when expected (CHI < 0.7)?
- Does output quality visibly change when steering kicks in?
- Does the agent still follow the tool protocol while steered?

**Time:** 2-3 hours
**Git:** `git commit -m "TASK-041: CHI-triggered steering in agent loop"`

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
