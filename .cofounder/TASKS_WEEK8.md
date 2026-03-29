# WEEK 8 TASKS: Production Hardening + Model Upgrade Prep
# Week 7 proved the steering pipeline works mechanically but doesn't
# change task outcomes on Qwen2.5-7B. Week 8 fixes the actual blockers:
# completion detection, compaction tuning, and Qwen3.5 evaluation.

---

## WEEK 8: Make It Actually Complete Tasks
**Goal:** Fix the two real blockers (no [DONE] detection, compaction
never fires), then evaluate Qwen3.5-9B as a model upgrade.

**Key findings from Week 7 driving this week:**
1. Model completes coding work by turn 6-8 but never says [DONE]
2. Steering keeps CHI at ~0.60, preventing compaction (threshold 0.50) from firing
3. Steering provides +0.01-0.02 CHI improvement but doesn't change task outcomes
4. tool_format is the only reliable vector (all turns alive, length parity)

---

### TASK-045: Fix Completion Detection
**Status:** ✅ DONE
**Do:** Add heuristic completion detection + strengthen [DONE] prompt

This is the #1 priority. All three TASK-044 runs created files and ran
pytest by turn 6-8. The remaining 12+ turns were wasted because the
model didn't emit [DONE].

**Part A — Heuristic completion detection (agent.py):**

```python
def _check_task_completion(self) -> bool:
    """
    Heuristic: if the task mentions specific files and tests,
    check if they exist and tests passed.
    """
    # Check if all expected files exist
    expected_files = self._extract_expected_files(self.task)
    files_exist = all(
        os.path.exists(os.path.join(self.working_dir, f))
        for f in expected_files
    )
    
    # Check if last execute tool had exit code 0
    last_exec_success = any(
        t.get("tool_used") == "execute" and t.get("exit_code") == 0
        for t in self.tool_history[-3:]  # last 3 turns
    )
    
    if files_exist and last_exec_success:
        print("[COMPLETION] Heuristic: files exist + tests passed → task complete")
        return True
    return False
```

Add to the agent loop: after each turn, call `_check_task_completion()`.
If true, treat as [DONE] equivalent.

**Part B — Strengthen [DONE] in system prompt (prompts.py):**

Add to SYSTEM_PROMPT:
```
COMPLETION RULE: When you have created all requested files AND the last
test/command succeeded (exit code 0), you MUST respond with [DONE].
Do NOT continue working after tests pass. Do NOT re-run tests that
already passed. Say [DONE] and stop.
```

Add a few-shot example showing the [DONE] response after pytest passes.

**Part C — Anti-loop guard:**

If the model produces the same tool call (same type + same args) for
3 consecutive turns, force-break with a warning injection:
"You are repeating the same action. If the task is complete, say [DONE]."

**Manual test:** Re-run the calculator task from TASK-044 baseline mode.
Verify the agent completes in <10 turns instead of running to max_turns.

**Time:** 2-3 hours
**Git:** `git commit -m "TASK-045: completion detection — heuristic + prompt + anti-loop"`

### TASK-045D: Heuristic only on test commands (addendum)
**Status:** ✅ DONE
**Do:** Filter `_check_task_completion()` so exit-0 `pip install`, `ls`, etc. do not complete the task; require a command that looks like pytest/unittest/`python test_*`.

**Git:** `git commit -m "TASK-045D: filter heuristic to test commands only"`

### TASK-045E: Prompt + [DONE] validation for “tests must pass”
**Status:** ✅ DONE
**Do:** Clarify pip install ≠ tests; reject explicit `[DONE]` when task implies tests but `_check_task_completion`-style evidence is missing.

**Git:** `git commit -m "TASK-045E: validate [DONE] against test evidence + prompt"`

---

### TASK-046: Fix Compaction Threshold + Time-Based Escalation
**Status:** ⬜ TODO
**Do:** Lower compaction threshold + add steering timeout

TASK-044 showed steering keeps CHI at ~0.60, which prevents the 0.50
compaction threshold from triggering. Level 2 blocks Level 3.

**Fix 1 — Lower compaction threshold:**

Change from `CHI < 0.5` to `CHI < 0.65` for compaction consideration.
But don't compact immediately — add a grace period:

```python
COMPACTION_CHI_THRESHOLD = 0.65
COMPACTION_GRACE_TURNS = 3  # must stay below threshold for 3 turns

# Track consecutive turns below threshold
if chi < COMPACTION_CHI_THRESHOLD:
    self._turns_below_compaction += 1
else:
    self._turns_below_compaction = 0

if self._turns_below_compaction >= COMPACTION_GRACE_TURNS:
    print(f"[COMPACTION] CHI below {COMPACTION_CHI_THRESHOLD} for "
          f"{COMPACTION_GRACE_TURNS} turns — compacting")
    self.compactor.compact(strategy="auto")
    self._turns_below_compaction = 0
```

**Fix 2 — Steering timeout escalation:**

If steering has been active for N consecutive turns without CHI
recovering above the deactivation threshold, force compaction:

```python
STEERING_TIMEOUT_TURNS = 8  # if steering active 8+ turns, escalate

if self.steering_active:
    self._steering_active_turns += 1
else:
    self._steering_active_turns = 0

if self._steering_active_turns >= STEERING_TIMEOUT_TURNS:
    print(f"[ESCALATION] Steering active {STEERING_TIMEOUT_TURNS} turns "
          f"without recovery — forcing Level 3 compaction")
    self.compactor.compact(strategy="auto")
    self._steering_active_turns = 0
```

**Manual test:** Run the steered agent with these fixes. Verify:
1. Compaction fires when CHI stays around 0.60 for 3+ turns
2. After compaction, context length drops significantly
3. Model gets a "fresh start" and CHI recovers (even temporarily)

**Time:** 2-3 hours
**Git:** `git commit -m "TASK-046: compaction threshold + steering timeout escalation"`

---

### TASK-047: Re-run TASK-044 with Fixes (The Real Capstone)
**Status:** ⬜ TODO
**Do:** Re-run all 3 modes with TASK-045 + TASK-046 fixes

Same protocol as TASK-044, same calculator task, 3 modes:
- baseline (no steering, no compaction)
- steering (Level 2 only)
- full_ladder (Level 2 + Level 3)

**Expected improvement:** Baseline should complete in <10 turns (heuristic
[DONE] detection). Steering should complete similarly. Full ladder should
show compaction firing when steering fails, with context reset.

**This is the graph you show in the demo video.**

**Output:** Same as TASK-044 — comparison JSON + PNG + report.

**Time:** 2-3 hours (3 runs + analysis)
**Git:** `git commit -m "TASK-047: re-run integration test with completion + compaction fixes"`

---

### TASK-048: Evaluate Qwen3.5-9B on MLX
**Status:** ⬜ TODO
**Do:** Create `scripts/eval_qwen35_v2.py`

Qwen3.5 MLX models are available:
- `mlx-community/Qwen3.5-9B-MLX-4bit`
- `mlx-community/Qwen3.5-4B-MLX-4bit`

**WARNING:** These use `mlx-vlm` for loading, NOT `mlx-lm`. Qwen3.5
is MoE + Gated DeltaNet architecture. Loading pattern may differ.

**Test checklist:**
1. Can you load the model with `mlx_lm.load()` or do you need `mlx_vlm`?
2. Does `model.model.layers` exist with the same structure?
3. Does CaptureLayer wrapper work (hidden state extraction)?
4. What's the token speed? (target: >30 tok/s on M4 Pro)
5. Run the calculator task WITHOUT steering — does it follow tools better?
6. Does it say [DONE] when the task is complete?

**If hidden state extraction works:**
- Verify hidden state shape (may differ from 3584-dim in Qwen2.5)
- Check number of layers (may differ from 28)
- Note which layers are full-attention vs DeltaNet (every 4th block)

**If loading fails:**
- Try `mlx_vlm.load()` instead
- Try self-converting with `mlx_lm.convert --hf-path Qwen/Qwen3.5-9B-Instruct`
- Document the error and file it as a blocker

**Time:** 3-4 hours (mostly debugging loading + architecture differences)
**Git:** `git commit -m "TASK-048: evaluate Qwen3.5-9B on MLX"`

---

### TASK-049: Recompute Steering Vectors for Qwen3.5 (If TASK-048 Passes)
**Status:** ⬜ TODO (blocked by TASK-048)
**Do:** Run TASK-034→036 pipeline on Qwen3.5-9B

If Qwen3.5 loads and hidden states extract correctly:
1. Run contrastive pair extraction (TASK-035 equivalent)
2. Compute steering vectors (TASK-036 equivalent)
3. Run the 5-turn sweep to find safe layer/alpha (TASK-039 equivalent)
4. Run a 10-turn calibrated experiment (TASK-040 equivalent)

**Key difference:** Qwen3.5 has ~40 layers with MoE. Only every 4th
layer uses full attention. Steering should target full-attention layers
specifically (layers 4, 8, 12, 16, 20, 24, 28, 32, 36, 40).

**Skip if TASK-048 shows loading or extraction failures.**

**Time:** 3-4 hours
**Git:** `git commit -m "TASK-049: steering vectors for Qwen3.5-9B"`

---

### TASK-050: Demo Script + Narrative Prep
**Status:** ⬜ TODO
**Do:** Create `scripts/demo.py` — a scripted demo for the Phase 3 video

Regardless of Qwen3.5 status, prepare a demo script that shows:
1. Start the agent on a coding task
2. CHI monitoring running in real-time (color-coded)
3. CHI drops below threshold → steering activates (visual indicator)
4. If CHI keeps dropping → compaction fires (context reset)
5. Agent completes the task

This doesn't need to be perfect — it's the skeleton for the Phase 3
demo video (Weeks 11-12). Get the narrative and script structure right.

**Output:** `scripts/demo.py` + `docs/demo_narrative.md` (talking points)

**Time:** 1-2 hours
**Git:** `git commit -m "TASK-050: demo script skeleton and narrative prep"`

---

## WEEK 8 CHECKLIST
- [x] TASK-045: Fix completion detection — heuristic + prompt + anti-loop (2-3 hrs)
- [ ] TASK-046: Fix compaction threshold + steering timeout (2-3 hrs)
- [ ] TASK-047: Re-run TASK-044 with all fixes (2-3 hrs)
- [ ] TASK-048: Evaluate Qwen3.5-9B on MLX (3-4 hrs)
- [ ] TASK-049: Recompute steering vectors for Qwen3.5 (3-4 hrs, conditional)
- [ ] TASK-050: Demo script + narrative prep (1-2 hrs)

**Total: ~14-19 hours across 4-5 days**

**Week 8 milestone:** Agent completes calculator task reliably (<10 turns).
Compaction fires when steering can't recover CHI. Qwen3.5 evaluated.
Demo script skeleton ready for Phase 3.

---

## NOTES

### The Demo Video Story (Revised)

The original plan: "Steering fixes degradation" → graph of steered vs unsteered.
The honest story from Week 7: steering provides marginal improvement.

**New narrative:** "Cognitive monitoring catches problems that users don't see.
The intervention ladder responds automatically — steering for mild issues,
context reset for severe ones. The framework makes ANY local model more reliable."

Show:
1. The degradation graph from Week 2 (the problem exists)
2. CHI monitoring detecting it in real-time (we can see it)
3. The intervention ladder responding (we can fix it)
4. Before/after task completion rates (the impact)

The story is about the SYSTEM, not about any single intervention.

### Qwen3.5 Architecture Notes
- Hybrid: Gated DeltaNet + Sparse MoE
- 3:1 ratio linear-to-full attention layers
- Only full-attention layers (every 4th) are good steering targets
- MoE routing adds complexity to hidden state interpretation
- Loading may require mlx-vlm instead of mlx-lm

### 📌 BRAIN UPDATE (add to COFOUNDER_BRAIN.md):
```
- WEEK 7 COMPLETE: Steering pipeline working end-to-end on MLX.
  Sweep → calibration → agent integration → adaptive alpha → experiment.
  Steering provides marginal CHI improvement (+0.01-0.02) but doesn't
  change task outcomes on Qwen2.5-7B. tool_format is the only reliable vector.
  Compaction never fired (steering holds CHI above threshold).
  Key insight: value is in the monitoring + full ladder, not steering alone.
- WEEK 8: Production hardening. Fix [DONE] detection, compaction threshold,
  evaluate Qwen3.5-9B, prepare demo narrative.
```
