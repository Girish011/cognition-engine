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

### TASK-045F: Harness pytest + no-tool-output guard
**Status:** ✅ DONE
**Do:** Pre-install `pytest` in `test_steering_agent.py`; in agent, count consecutive turns with no parseable tool and inject recovery after 3.

**Git:** `git commit -m "TASK-045F: pre-install pytest + no-tool-output guard"`

### TASK-045G: Pytest via `python -m pytest` (PATH)
**Status:** ✅ DONE
**Do:** Few-shot uses `[EXEC] python -m pytest`; `_tool_execute` rewrites bare `pytest` / `pytest .` / `pytest …` to `sys.executable -m pytest`.

**Git:** `git commit -m "TASK-045G: fix pytest PATH — use python -m pytest"`

### TASK-045H: Strip chat template tokens from tool arguments
**Status:** ✅ DONE
**Do:** `strip_chat_template_artifacts()` — remove `<|im_end|>` and `<|...|>` from parsed paths/commands/content; apply in `parse_tool_call`, `_tool_execute`, and defensively in tool dispatch.

**Git:** `git commit -m "TASK-045H: strip chat template tokens from tool arguments"`

---

### TASK-046: Fix Compaction Threshold + Time-Based Escalation
Status: ✅ DONE

PART A — Lower compaction threshold with grace period:
- COMPACTION_CHI_THRESHOLD = 0.65  (was 0.50)
- COMPACTION_GRACE_TURNS = 3
- Track _turns_below_compaction, reset when CHI recovers
- Same logic as original spec

PART B — Steering timeout escalation:
- STEERING_TIMEOUT_TURNS = 5  (was 8, lowered because tasks
  complete in 6 turns now)
- Force compaction when steering can't recover CHI

PART C — Guard-triggered compaction (NEW):
- Track _guard_fire_count (incremented by no-tool guard
  and anti-loop guard)
- If _guard_fire_count >= 2 within a session, force compaction
  regardless of CHI value
- Reset counter after compaction fires
- Logic: if the model needed 2 behavioral nudges and still
  isn't producing tool calls, context is poisoned

PART D — Post-compaction steering reset:
- After compaction fires, deactivate steering and reset
  _steering_active_turns to 0
- Let CHI re-evaluate on the fresh context before re-engaging
- Prevents stale steering state from carrying over

TESTING:
- Test 1: Run baseline on calculator task — should complete
  in ~6 turns, compaction should NOT fire (regression check)
- Test 2: Run steering mode on calculator — verify compaction
  fires if steering can't hold CHI
- Test 3: Run a harder task (multi-file or one the 7B model
  struggles with) to exercise the full ladder path
- Test 4: Manually verify guard-triggered compaction by
  checking logs for [NO-TOOL-GUARD] + [COMPACTION] sequence

Time: 2-3 hours
Git: TASK-046: compaction threshold + steering timeout + guard escalation

---

### TASK-047: Re-run TASK-044 with Fixes (The Real Capstone)
Add the duplicate-write guard first. Without it, all three converter runs will produce the same 15-turn rewrite loop. Here's the full plan — execute in order.

**PART 1: Duplicate-write guard (agent.py)**

Add `self._last_write_hash = {}` in `__init__`.

After `_tool_write` succeeds, before recording in tool_history:

```python
import hashlib

def _content_hash(self, content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()

# In the write execution path, after file is written:
content_hash = self._content_hash(content)
if path in self._last_write_hash and self._last_write_hash[path] == content_hash:
    # Identical rewrite — don't record, inject correction
    self.compactor.add_turn("system",
        f"{path} already contains that exact content. "
        f"Do NOT rewrite it. Move to the next step of the task.")
    self._last_write_hash[path] = content_hash
    continue  # skip to next turn
self._last_write_hash[path] = content_hash
```

Also reset `_last_write_hash` after compaction fires (in the compaction block):

```python
self._last_write_hash.clear()
```

**PART 2: Update test_steering_agent.py for TASK-047**

Add `--task` CLI argument if not already present. Add output file naming based on task type:

```python
# Detect task type for output naming
task_key = "calc" if "calculator" in args.task.lower() else "conv"
output_path = f"data/task047_{task_key}_{args.mode}.json"
```

**PART 3: Run all 6 (copy-paste block)**

After implementing parts 1-2:

```bash
cd /Users/girish11/cognition-engine

CALC_TASK="Create a Python file called calculator.py with a Calculator class that has add, subtract, multiply, and divide methods. Then create test_calculator.py with tests for each method. Run the tests to verify they pass."

CONV_TASK="Create a Python file called converter.py with a TemperatureConverter class that has celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin, and kelvin_to_celsius methods. Each method should handle edge cases (absolute zero check for kelvin methods - raise ValueError if result would be below 0K). Then create test_converter.py with at least 8 tests covering normal conversions and edge cases. Run the tests to verify they pass."

echo "=== CALCULATOR RUNS ==="

.venv/bin/python -u scripts/test_steering_agent.py \
  --mode baseline --dir /tmp/task047_calc_base --max-turns 10 \
  --task "$CALC_TASK" --auto-approve
echo "=== calc baseline done ==="

.venv/bin/python -u scripts/test_steering_agent.py \
  --mode steering --dir /tmp/task047_calc_steer --max-turns 10 \
  --task "$CALC_TASK" --auto-approve
echo "=== calc steering done ==="

.venv/bin/python -u scripts/test_steering_agent.py \
  --mode full_ladder --dir /tmp/task047_calc_full --max-turns 10 \
  --task "$CALC_TASK" --auto-approve
echo "=== calc full_ladder done ==="

echo "=== COOLDOWN 120s ==="
sleep 120

echo "=== CONVERTER RUNS ==="

caffeinate -dimsu .venv/bin/python -u scripts/test_steering_agent.py \
  --mode baseline --dir /tmp/task047_conv_base --max-turns 20 \
  --task "$CONV_TASK" --auto-approve
echo "=== conv baseline done ==="

sleep 120

caffeinate -dimsu .venv/bin/python -u scripts/test_steering_agent.py \
  --mode steering --dir /tmp/task047_conv_steer --max-turns 20 \
  --task "$CONV_TASK" --auto-approve
echo "=== conv steering done ==="

sleep 120

caffeinate -dimsu .venv/bin/python -u scripts/test_steering_agent.py \
  --mode full_ladder --dir /tmp/task047_conv_full --max-turns 20 \
  --task "$CONV_TASK" --auto-approve
echo "=== conv full_ladder done ==="
```

**PART 4: What to paste back**

After all 6 complete, paste:
- Terminal output for all 6 runs
- The 6 JSON files (or just the metrics blocks from each)
- Whether converter.py and test_converter.py exist in each converter run dir

I'll analyze the results, generate the comparison graph spec, and write the report.

**Git plan:** Commit the duplicate-write guard as `TASK-047-prep: duplicate-write guard`, then commit results as `TASK-047: three-mode comparison on calculator + converter`. Two commits keeps it clean.


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
