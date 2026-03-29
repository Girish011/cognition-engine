# SESSION_LOG.md

# ✏️ Write entries in REAL-TIME as you work (not at end of day)

# Only paste the LAST 3-5 entries when starting a new Claude session

## 2026-03-01 (Week 0 — Planning Session 1)

- Completed full architecture research with Claude (8 domains, 50+ papers)
- Stress-tested 10 assumptions — most survived, some pivoted
- Finalized 5-layer architecture and 5-level intervention ladder
- Compared Layer 4 approaches: context refresh vs EasySteer → EasySteer wins
- Created 16-week execution plan with weekly milestones
- Set up co-founder continuity protocol (crash-resilient, no graceful shutdown needed)
- Created initial COFOUNDER_[BRAIN.md](http://BRAIN.md)
- DECISION: Build from scratch on MLX, no wrappers
- DECISION: Use Cursor as IDE for development
- DECISION: Apple Silicon first, NVIDIA later

## 2026-03-03 (Week 0 — Planning Session 2)

- Fixed continuity protocol: crash-resilient (Claude drops 📌 checkpoints mid-session)
- Created final file set: COFOUNDER_[BRAIN.md](http://BRAIN.md), SESSION_[LOG.md](http://LOG.md), [TASKS.md](http://TASKS.md), START_[HERE.md](http://HERE.md)
- Evaluated OpenFang — NOT a competitor (agent orchestration, not cognitive monitoring)
- Evaluated Qwen 3.5 small models release:
  - 9B beats prev-gen 30B, uses hybrid DeltaNet attention
  - Good for us: better local models = stronger product thesis
  - Keep Qwen2.5 for learning (pure transformer), switch to Qwen3.5 Week 4+
- Fixed TASK-004: mlx-lm is a SEPARATE repo from mlx
  - mlx = low-level array framework (github.com/ml-explore/mlx)
  - mlx-lm = LLM package with model code (github.com/ml-explore/mlx-lm)
  - Source files: mlx_lm/models/[qwen2.py](http://qwen2.py), mlx_lm/[generate.py](http://generate.py), mlx_lm/[utils.py](http://utils.py)
- Created skills inventory (45 skills across 8 domains)
- TASK-001 — environment setup - Completed
- TASK-002 - Download and Run First Model - `first model inference working`
- `TASK-003: benchmark baseline 53.0 tok/s`
- Local M4 24gb ram hardware baseline - 336 tokens in 6.3s = 53.0 tok/s

## 2026-03-04 (Week 0 — Planning Session 3)

- **How many transformer layers does Qwen2.5‑7B have?**
  - **28 transformer layers** (num_hidden_layers = 28 in the Qwen2.5‑7B config).
- **What is the hidden dimension size?**
  - **3584** (hidden_size = 3584 for Qwen2.5‑7B; in code this is ModelArgs.hidden_size used all through [qwen2.py](http://qwen2.py)).
- **Where does self‑attention happen?**
  - Self‑attention is implemented in the **Attention** class (conceptually the same as a Qwen2Attention), and used inside each transformer layer:
    -   class Attention(nn.Module):
            ...
            self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=True)
            self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
            self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
        ...
        117:120:/Users/girish11/cognition-engine/.venv/lib/python3.13/site-packages/mlx_lm/models/[qwen2.py](http://qwen2.py)
            r = self.self_attn(self.input_layernorm(x), mask, cache)
            h = x + r
- **How is the KV cache passed through layers?**
  - In Qwen2Model.__call__, cache is a **list with one entry per layer**, and each entry is passed into that layer:
    -   if cache is None:
            cache = [None] * len(self.layers)
        ...
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
  - Inside Attention.__call__, that layer‑specific cache is used and updated:
    -     if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)

- **Where are logits computed from hidden states?**
  - In the top‑level Model.__call__, after Qwen2Model returns hidden states, they are mapped to **vocabulary logits** via either the tied embedding matrix or lm_head:
    -   out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_[tokens.as](http://tokens.as)_linear(out)
        else:
            out = self.lm_head(out)
  - That final linear mapping from hidden size 3584 → vocab size is where **logits are computed from hidden states**.

## 2026-03-07 (Week 0 — Engine: hidden states)

- Implemented `extract_hidden_states()` in `engine/hidden_states.py`: runs inference and returns `(generated_text, hidden_state_tensor)` with shape `(1, seq_len, hidden_dim)`.
- MLX has no PyTorch-style `register_forward_hook`: intercept by replacing the layer in the list with a wrapper.
  - Introduced `_CaptureLayer` wrapper: holds the real layer and a `captured` list; `__call__` runs the layer, appends output, returns it. Set `model.model.layers[layer_idx] = _CaptureLayer(original_layer, captured)` and restore in `finally`.
- Flow: (1) generate text with `generate()`, (2) build full sequence = prompt tokens + generated tokens, (3) clear `captured`, (4) one forward `model(full_tokens[None], cache=None)` and `mx.eval(...)` so the forward runs, (5) take `captured[0]` as the hidden state for that layer.
- Bug fix: `mx.eval(x)` evaluates in place and returns `None`. Do not assign: use `mx.eval(hidden)` only; never `hidden = mx.eval(hidden)` or `hidden` becomes `None`.
- Hidden state shape example: `(1, 15, 3584)` = batch 1, sequence length 15 (prompt + generated), hidden size 3584.
- `if __name__ == "__main__"`: loads `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`, runs `extract_hidden_states(..., "Write hello world in Python")`, prints generated text and hidden state shape.

## 2026-03-07 (Week 0 — TASK-006: Verify hidden states)

- TASK-006 — Verify Hidden States Are Meaningful — Completed
- Created `scripts/verify_states.py`: runs two prompts through `extract_hidden_states()` (layer 16, max_tokens 80).
  - Prompt A: "Write a Python sorting algorithm"
  - Prompt B: "Write a poem about the ocean"
- Averages hidden states over sequence length, then computes cosine similarity between the two (3584-d) vectors.
- Result: cosine similarity 0.7379 (< 1 → different prompts produced different representations; capture mechanism working).
- Hidden state shapes observed: A (1, 85, 3584), B (1, 86, 3584); averaged vector shape (3584,).

## 2026-03-07 (Week 0 — TASK-007: Attention weights)

- TASK-007 — Extract Attention Weights (Prep for Week 2) — Completed
- Created `engine/attention_weights.py`: wraps the self-attention module to capture softmax attention weights (same approach as TASK-005: replace module with wrapper, run forward, restore in `finally`).
- `_CaptureAttention`: replicates Attention forward (Q/K/V, RoPE, GQA repeat), computes scores → mask → softmax, appends weights to list, then output = weights @ V and o_proj.
- `extract_attention_weights(model, tokenizer, prompt, layer_idx=16, max_tokens=50)` returns `(generated_text, attention_weights)`; weights shape `(1, n_heads, seq_len, seq_len)`.
- Manual test: `python engine/attention_weights.py` — output: generated text + **Attention weights shape: (1, 28, 46, 46)** (28 heads, 46 tokens).

## 2026-03-07 (Week 2 — TASK-008: Token-level entropy)

- TASK-008 — Token-Level Entropy from Logits — Completed
- Created `engine/entropy.py`: Shannon entropy H = -Σ p(x) log2(p(x)) over next-token distribution.
  - `compute_token_entropy(logits)`: softmax → p*log2(p) with 0*log(0)=0 for stability → sum over vocab, negate → returns (seq_len,).
  - `get_logits_for_sequence(model, tokenizer, text)`: single forward pass (no sampling), returns (1, seq_len, vocab_size).
- Manual test (`python engine/entropy.py`):
  - 'The' mean entropy: **7.3047** (high, 6–10+ as required)
  - 'def fibonacci(n):' mean entropy: **3.3906** (lower)
  - Code prompt has meaningfully lower average entropy than open-ended prompt ✓
- Git: `TASK-008: token-level entropy computation`

## 2026-03-07 (Week 2 — TASK-009: Multi-turn entropy)

- TASK-009 — Entropy Over a Multi-Turn Conversation — Completed
- Created `data/` and `scripts/multiturn_entropy.py`: multi-turn coding conversation with entropy at each turn.
  - Uses TASK_SEQUENCE (20 turns) building a TaskManager class step by step; each turn appends User/Assistant to conversation, generates with `generate()`, then gets logits for full text and computes `response_entropy` (last N tokens, N = response token count).
  - Results: turn, prompt snippet, response_length, context_length, response_entropy_mean/max/min; prints per-turn stats; saves `data/multiturn_entropy.json`; stops if context_length > 28K.
- MLX: use `mx.eval()` on entropy then `mean_ent.item()` / `max_ent.item()` / `min_ent.item()` for JSON-serializable scalars.
- Quick test (3 turns): script runs, entropy printed per turn, result dicts have expected keys. Full 20-turn run: run `python scripts/multiturn_entropy.py` (takes ~10+ min).
- **data/multiturn_entropy.json check:** 20 turns, valid structure. Context grows 334→3814 tokens (under 28K). Entropy: low early (mean ~0.02–0.20), higher/variable later (e.g. 0.44–1.36); turns 14/16/18 show clear rise. Caveat: from turn 11 onward many responses are 2 tokens (short replies), so entropy there is over few tokens; when model does generate longer (turns 16, 18), mean entropy ~1.0+.

## 2026-03-07 (Week 2 — TASK-010: Repetition detection)

- TASK-010 — Repetition Detection (Second CHI Signal) — Completed
- Created `engine/repetition.py`: Jaccard similarity on word n-grams (default trigrams).
  - `compute_repetition_score(text_a, text_b, n=3)`: word split → trigrams → Jaccard = |A∩B|/|A∪B|; 0.0–1.0.
  - `compute_self_repetition(text, window_size=100)`: consecutive word-windows, average pairwise Jaccard (high = stuck in loop).
  - `track_repetition_over_turns(responses)`: Jaccard between consecutive responses; list of length len(responses)-1.
- Manual test (`python engine/repetition.py`): identical → 1.0; different texts → 0.0; similar code → ~0.18; self-repetition on repeated "word" → 1.0; track over turns smoke test OK.
- Git: `TASK-010: repetition detection (Jaccard similarity)`

## 2026-03-07 (Week 2 — TASK-011: Goal drift detection)

- TASK-011 — Goal Drift Detection (Third CHI Signal) — Completed
- Created `engine/goal_drift.py`: drift = 1 − cosine similarity of hidden states (task vs current output).
  - Uses `extract_hidden_states(..., max_tokens=1)` for task and current_output; mean over (0,1) to get one vector per text; cosine similarity with scale-then-normalize to avoid overflow (hidden state norms can be huge).
  - Default `layer_idx=27` (last layer) so on-task > off-task in manual test; layer 16 gave near-identical scores.
- Manual test: on-task 0.89, off-task 0.82; assert score_on > score_off passes. Path fix for `python engine/goal_drift.py` (project root on sys.path).
- Git: `TASK-011: goal drift detection via hidden state similarity`

## 2026-03-07 (Week 2 — TASK-012: Degradation visualization)

- TASK-012 — Visualize Degradation (The Money Graph) — Completed
- Created `scripts/visualize_degradation.py`: reads `data/multiturn_entropy.json`, plots 2 panels (entropy mean + fill to max over turns; context length over turns), saves `data/degradation_graph.png`. Uses matplotlib Agg backend. `pip install matplotlib` required.
- Manual test: script runs from project root, outputs "Saved: data/degradation_graph.png". Graph shows entropy and context growth over 20 turns.
- Git: `TASK-012: degradation visualization`

## 2026-03-08 (Week 2 — TASK-013: Full 3-signal degradation experiment)

- TASK-013 — Enhanced Experiment with All Three Signals — Completed
- Created `scripts/full_degradation_experiment.py`: combines entropy (TASK-008), repetition (TASK-010), and goal drift (TASK-011) in one multi-turn run.
  - Reuses `TASK_SEQUENCE` and `SYSTEM_PROMPT` from `scripts.multiturn_entropy`; for each turn: generate response, compute response entropy, Jaccard vs previous response (turn 0 → 0.0), and goal drift = 1 − cosine similarity vs original task (first user message).
  - Saves `data/full_degradation.json`; generates `data/full_degradation_graph.png` with 3 panels (entropy, repetition, goal drift). CLI: `--max-turns N` for quick test, `--plot-only` to redraw from existing JSON.
- Created `data/FULL_DEGRADATION_GRAPH_README.md`: how to read the 3 panels (↑ worse for each), what to look for, regenerate commands.
- In-graph annotations: each panel has “↑ worse: …” hint; bottom text box explains shared x-axis and points to README.
- Manual test: `python scripts/full_degradation_experiment.py --max-turns 2` — completes, JSON and 3-panel PNG saved. Full run: `python scripts/full_degradation_experiment.py`.
- Git: `TASK-013: full 3-signal degradation experiment`

## 2026-03-08 (Week 2 — TASK-014: Find inflection point)

- TASK-014 — Find the Inflection Point — Completed
- Created `scripts/find_inflection.py`: analyzes `data/full_degradation.json` to find the turn where degradation begins (when CHI would trigger Level 2 intervention).
  - Method 1 — Moving average crossover: short MA (window 3) vs long MA (window 7); first turn where short_ma > long_ma × 1.05 is the crossover turn.
  - Method 2 — Maximum rate of change: turn index where entropy increase (diffs[i] = entropy[i+1] − entropy[i]) is largest; reported as “largest jump into turn N”.
  - Requires at least 7 turns; otherwise prints “Need at least 7 turns” and suggests running a longer experiment. CLI: `--data <path>` for alternate JSON.
- Manual test: with 2-turn data → “Not enough data points (2). Need at least 7 turns.”
- **Synthetic test finding** (script logic verified on 10-turn synthetic data; file was removed after test):
  - Command: `cd /Users/girish11/cognition-engine && python scripts/find_inflection.py --data data/full_degradation_synthetic_test.json`
  - Output: Total turns 10; entropy range 0.0000 — 0.3400; moving average crossover at turn **7**; maximum entropy increase at turn **4** (largest jump into turn 5); recommendation: “Degradation likely begins around turn 7 — this is where CHI would start triggering Level 2 intervention”.
  - Confirms both methods run and produce a concrete inflection turn when enough data is present.
- **Inflection turn:** For production: run full TASK-013 then `python scripts/find_inflection.py`; write the printed turn number here — it drives Phase 1 (when to trigger interventions). Current repo has only 2-turn real run; full run needed for real inflection number.
- Git: `TASK-014: inflection point detection`

## 2026-03-08 (Goal drift fix + full 3-signal experiment)

- **Bug:** In the full experiment, goal drift was pegged at 1.0 (cosine similarity 0.0). Standalone TASK-011 was correct (0.89); inside the experiment we were comparing hidden states wrongly for long responses.
- **Root cause:** `compute_goal_drift` used `extract_hidden_states(..., max_tokens=1)`, which for a long response (300+ tokens) did a forward over prompt+1 token and averaged hidden state over the full sequence. That average over 300+ positions produced degenerate/near-zero similarity (numerical/scale issue).
- **Fix 1 — Forward-only, no generation:** Added `extract_hidden_states_forward_only(model, tokenizer, text, layer_idx)` in `engine/hidden_states.py`: single forward pass on tokenized text, no generation. Used in `compute_goal_drift(..., use_forward_only=True)` so both task and response are compared as plain text representations.
- **Fix 2 — Long sequences:** For `current_output` with >128 tokens, use **last-token hidden state only** instead of mean over sequence (causal LM last position encodes the sequence). Prevents degenerate 0 similarity when averaging 300+ positions. Short sequences still use mean.
- **Verification:** Short task vs short response → 0.98; short vs long on-task response (358 tokens) → 0.77 (was 0.0 before fix). 3-turn experiment: drift = 0.31, 0.19, 0.36 (no longer pegged at 1.0).
- **Experiment script:** Added `--debug` to print `original_task`, `current_response` snippet, and raw cosine each turn. Confirmed we pass only the current turn’s **response** (not full conversation) to `compute_goal_drift`.
- **Full 20-turn run:** Started `python scripts/full_degradation_experiment.py` (no `--max-turns`). Takes ~15–30 min. When it finishes:
  1. Regenerate graph: already saved as `data/full_degradation_graph.png` (x-axis will be 0–19).
  2. Run `python scripts/find_inflection.py` and **write the printed inflection turn below**.
  3. Open `data/full_degradation_graph.png` and add a one-line description of the 3 panels below.
- **Real inflection turn (from full run):** **Turn 10.** `python scripts/find_inflection.py` on 20-turn data: moving average crossover at turn 10; maximum entropy increase at turn 13 (largest jump into turn 14). Entropy range 0.0182 — 1.3564. → **CHI would start Level 2 intervention around turn 10.**
- **3-signal graph description (from full run):** X-axis 0–19 (20 turns). **Entropy:** low and stable for turns 0–9 (~0.02–0.07), then rises from turn 11 (0.08, 0.44, 0.46, 0.97, 0.90, 1.36, …). **Repetition:** rises through turn 9 (up to ~0.88), then turns 11–17 show 0 or 1.0 (short/variable responses). **Goal drift:** varies per turn (0.15–0.55), not pegged at 1.0. Clear degradation: entropy spike from ~turn 10 onward; inflection at turn 10.

## 2026-03-08 (Week 3 — TASK-015: CHI score implementation)

- TASK-015 — Implement CHI Score (Combining 3 Signals) — Completed
- Created `engine/chi.py`: CHIMonitor class combining entropy (0.4), repetition (0.3), and goal drift (0.3) into a single CHI score 0.0–1.0. `update(current_response, full_context=None)` returns dict with chi_score, raw signals, and health components; `get_intervention_level()` returns 1–4 from smoothed CHI.
- **Addition 1 — Gate repetition by response length:** If response has under 20 tokens, the previous turn’s repetition score is carried forward instead of computing Jaccard on tiny text (avoids noisy repetition panel swings on short replies).
- **Addition 2 — Smooth CHI with exponential moving average:** `chi_smoothed = 0.3 * chi_current + 0.7 * chi_previous` (CHI_SMOOTH_ALPHA = 0.3). First turn uses raw CHI. Prevents wild single-turn swings from whipsawing the overall score; effective window ~3–5 turns.
- Constants: `MIN_RESPONSE_TOKENS_FOR_REPETITION = 20`, `CHI_SMOOTH_ALPHA = 0.3`. Return dict includes `chi_raw` (unsmoothed) and `chi_score` (smoothed); history and intervention level use smoothed value.
- Manual test: `python engine/chi.py` — Turn 0 CHI 0.934 Level 1, Turn 1 CHI 0.897 Level 1; passed.
- Git: `TASK-015: CHI score combining 3 signals`

## 2026-03-08 (Week 3 — TASK-016: Validate CHI against Week 2 data)

- TASK-016 — Validate CHI Against Your Week 2 Data — Completed
- Created `scripts/validate_chi.py`: loads `data/full_degradation.json`, applies same CHI formula (and repetition gating + smoothing) to pre-computed signals (no model load; response text not stored in JSON). Computes CHI per turn and intervention level.
- Output: `data/chi_validation.png` — top panel: CHI score over turns with threshold lines at 0.8, 0.5, 0.3; bottom panel: entropy, repetition, goal drift for comparison.
- Verification: CHI starts high (>0.7) in early turns; drops below 0.7 around inflection (first turn < 0.7 is turn 11; TASK-014 inflection turn 10). Level 1 for turns 0–4, Level 2 from turn 5 onward. Intervention levels make sense.
- Manual test: `python scripts/validate_chi.py` — graph saved, summary and per-turn levels printed.
- Git: `TASK-016: CHI validated against real degradation data`

## 2026-03-08 (Week 3 — TASK-017: Project structure cleanup)

- TASK-017 — Project Structure Cleanup — Completed
- Renamed `engine/` → `cognition_engine/` (Python package convention). Created `cognition_engine/` with: `__init__.py` (`__version__ = "0.1.0"`), `inference.py` (re-export of mlx_lm load/generate), `entropy.py`, `repetition.py`, `hidden_states.py`, `attention_weights.py`, `goal_drift.py`, `chi.py`. Internal imports use relative (e.g. `from .entropy import ...`); removed sys.path hacks from package code.
- Updated all script imports to `cognition_engine`: `full_degradation_experiment.py`, `multiturn_entropy.py`, `verify_states.py`. Added sys.path.insert in `multiturn_entropy.py` for run-from-root.
- Added `pyproject.toml` (name=cognition-engine, version 0.1.0, deps mlx/mlx-lm, setuptools), `.gitignore` (data/*.json, data/*.png, kept data/.gitkeep and data/*README*), `data/.gitkeep`, `tests/__init__.py`, `tests/test_entropy.py`, `tests/test_repetition.py`, `README.md` (draft), `LICENSE` (MIT).
- Deleted `engine/` (all modules moved). Manual tests from project root: `python -c "from cognition_engine.chi import CHIMonitor; print('OK')"`, `python -c "from cognition_engine.entropy import compute_token_entropy; print('OK')"`, `python scripts/full_degradation_experiment.py --max-turns 2` — all passed. `pytest tests/` — 8 passed.
- Git: `TASK-017: restructure as proper Python package`

## 2026-03-08 (Week 3 — TASK-018: Basic CLI)

- TASK-018 — Basic CLI — Completed
- Created `cognition_engine/cli.py`: interactive chat loop with `python -m cognition_engine` (or `python -m cognition_engine.cli`). Loads model via `--model` (default Qwen2.5-Coder-7B-Instruct-4bit), `--max-tokens` (default 500). First user message becomes `original_task` for CHIMonitor; each turn: generate response, `monitor.update(response, conversation)`, color-coded CHI line (green >0.8, yellow >0.5, red otherwise) with E/R/D health components. Ctrl+C exits with summary: turns, final CHI, min CHI.
- Created `cognition_engine/__main__.py`: `from cognition_engine.cli import main; main()` so `python -m cognition_engine` runs the CLI.
- Manual test: `python -m cognition_engine --help` runs (with venv). Full interactive test: `python -m cognition_engine --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` — model loads, type task, model responds, CHI shown; Ctrl+C for summary.
- Git: `TASK-018: basic CLI with real-time CHI display`

## 2026-03-08 (CLI 16-turn run — CHI results)

- **Run:** `python -m cognition_engine` with Qwen2.5-Coder-7B-Instruct-4bit; 16 turns, Ctrl+C exit.
- **CHI:** Min 0.715, Final 0.777. Mostly Level 2 (15/16 turns); one turn Level 1 (CHI 0.813, short “spell orange” reply).
- **Topics:** Tamil Nadu history → sea-level/sinking → formula → research paper → Tamil politics → context window → spelling → long email-server/dual-NIC/PBR thread → foobar rhyming script.
- **Observations:** Model drifted: after topic changes (e.g. dual-NIC routing, then foobar), responses often repeated “VPN / cloud email / DNS” or generic PBR text; foobar answer was standard (no rhyming). Repetition (R) and drift (D) dipped when answers were off-topic or recycled (e.g. R=0.22, 0.57). CLI and CHI display behaved as intended; summary on exit: Turns 16, Final CHI 0.777, Min CHI 0.715.

## 2026-03-08 (Week 3 — TASK-019: Basic unit tests)

- TASK-019 — Write Basic Tests — Completed
- Added TASK-019–spec tests to existing `tests/test_entropy.py` and `tests/test_repetition.py` (no model load). Entropy: `test_uniform_distribution_high_entropy` (vocab 1000, zeros → entropy > 9), `test_peaked_distribution_low_entropy` (one dominant logit → entropy < 1), `test_entropy_shape` (1,5,100 → shape (5,)). Repetition: `test_identical_texts` (Jaccard 1.0), `test_completely_different` (< 0.1), `test_self_repetition_repeated_text` (repeated phrase, window 20 → score > 0.8).
- `python -m pytest tests/ -v`: 14 passed (6 entropy, 8 repetition).
- Git: `TASK-019: basic unit tests for entropy and repetition`

## 2026-03-08 (Week 3 — TASK-020: Draft README)

- TASK-020 — Draft README — Completed
- Rewrote `README.md` per task spec: one-liner + MIT badge; “The graph” section with `data/full_degradation_graph.png` (3-signal degradation); 3 bullet points (CHI, intervention levels, CLI); 4-line quick start (clone, pip install, run `python -m cognition_engine`); short “How it works” (CHI = entropy + repetition + goal drift, link to COFOUNDER_BRAIN); “Why this exists” paragraph (39% multi-turn degradation, local models, detect-and-intervene); roadmap (activation steering, context compaction, LoRA crystallization); MIT license.
- Allowed `data/full_degradation_graph.png` in `.gitignore` so the graph can be committed and shown in the README after running `scripts/full_degradation_experiment.py`.
- Git: `TASK-020: README with degradation graph and quick start`

## 2026-03-09 (Week 4 — TASK-021: Context compaction engine)

- TASK-021 — Context Compaction Engine — Completed
- Created `cognition_engine/compaction.py` with `ContextCompactor` managing conversation turns, anchor text, and automatic compaction when context length or CHI demands it.
- Implemented three strategies: `trim` (keep anchor + last N turns, no model call), `summarize` (model-generated `[CONTEXT SUMMARY]` of older turns, keep recent 5 turns verbatim), and `anchor_reset` (anchor + last 2 turns only).
- `compact(strategy="auto")` tries trim → summarize → anchor_reset until context falls under `max_context_tokens` (~75% of 32K by default) and increments `compaction_count`. `needs_compaction(chi_score)` triggers on token limit or CHI < 0.5.
- Manual scaffold in `__main__`: simulates a 20-turn conversation; `python -m cognition_engine.compaction` from the project venv printed **"Before compaction: 7531 tokens"** and **"After trim: 1006 tokens"**, confirming that `needs_compaction()` triggers on long context, `compact("trim")` dramatically reduces token count, and the rebuilt context still contains the TaskManager anchor text.

## 2026-03-09 (Week 4 — TASK-022: SCAN-style anchor tokens)

- TASK-022 — SCAN-Style Anchor Tokens — Completed
- Created `cognition_engine/anchors.py` with `AnchorManager` generating three reminder levels: `FULL` (full-context restoration block with original task, last 5 key decisions, and last 10 modified files), `MINI` (short `[REMINDER]` line with task snippet, counts of decisions/files), and `ANCHOR` (one-line `[FOCUS]` reminder).
- `add_decision()` and `add_file()` track evolving context so anchors stay grounded in actual work; `get_anchor_for_chi(chi_score)` maps CHI ranges to levels: `None` (>0.8), `ANCHOR` (0.6–0.8), `MINI` (0.4–0.6), `FULL` (≤0.4).
- Manual scaffold in `__main__`: constructs an `AnchorManager` for "Build a REST API with Flask", adds two decisions and two files, prints each anchor level, and asserts CHI-based selection returns `None`, a `[FOCUS]` line, a `[REMINDER]` line, and a `FULL CONTEXT` block respectively. From venv, `python -m cognition_engine.anchors` printed the FULL/MINI/ANCHOR blocks as expected and ended with "AnchorManager manual scaffold passed."

## 2026-03-09 (Week 4 — TASK-023: Tool-calling agent loop)

- TASK-023 — Tool-Calling Agent Loop — Completed
- Created `cognition_engine/agent.py` with `CognitionAgent` wiring together `CHIMonitor`, `ContextCompactor`, and `AnchorManager` plus file and shell tools. `run(max_turns)` builds context (injecting anchors when CHI < 0.8), calls `mlx_lm.generate`, parses a single bracketed tool call per response (`[READ]`, `[WRITE]`, `[EDIT]`, `[EXEC]`, `[THINK]`), executes it, appends tool results to context, updates CHI, triggers compaction (trim/summarize/anchor_reset) as needed, and stops when the model signals completion (`[DONE]`/“task is complete” markers) or hits `max_turns`.
- `_tool_read`, `_tool_write`, `_tool_edit`, and `_tool_execute` use `working_dir`-relative paths and update `AnchorManager.files_modified` so anchors reflect touched files. Execute prompts for approval unless `auto_approve=True`, runs with 30s timeout, and returns exit code + truncated stdout/stderr.
- `_parse_tool_calls()` now delegates to the shared `parse_tool_call()` helper (see TASK-024) so the bracket protocol is defined in one place. Returns 0 or 1 tool-call dict; `_execute_tool()` dispatches and records each call in `tool_history`.
- Manual scaffold in `__main__`: loads Qwen2.5-Coder-7B via `cognition_engine.inference.load`, creates an agent pointing at `/tmp/test_agent` with `auto_approve=True`, then runs the per-task manual tests: `[WRITE]`-equivalent `_tool_write("test.py", "print('hello')")`, `_tool_read("test.py")`, `_tool_edit("test.py", "hello", "world")`, `_tool_read("test.py")` again, and `_tool_execute("python test.py")` to print `world`. From venv, `python -m cognition_engine.agent` printed:
  - `[WROTE test.py (14 chars)]`
  - `[READ test.py]` followed by `print('hello')`
  - `[EDITED test.py]`
  - `[READ test.py]` followed by `print('world')`
  - `[EXEC \`python test.py\` → exit 0]` with `world` on stdout, confirming the tools work end-to-end in the test directory.

## 2026-03-09 (Week 4 — TASK-024: Tool-call prompt format + parser)

- TASK-024 — Tool-Call Prompt Format — Completed
- Created `cognition_engine/prompts.py` with `SYSTEM_PROMPT` describing the bracketed tool protocol, `build_system_prompt(task, project_context=None)` to inject the current task and optional project context, and `parse_tool_call(response)` that turns model outputs into structured tool dicts (`read`, `write`, `edit`, `execute`, `think`, `done`).
- Parser helpers `_extract_code_block()` and `_extract_edit_blocks()` implement the exact `WRITE`/`EDIT` fencing contract (```…``` and ```old / ```new blocks). `CognitionAgent._parse_tool_calls()` now delegates to `parse_tool_call()` so the loop and prompt share a single source of truth for tool syntax.
- Manual scaffold in `__main__`: runs in-process assertions that `[READ]`, `[EXEC]`, `[THINK]`, `[DONE]`, a `[WRITE]` with fenced code, and an `[EDIT]` with ```old/```new blocks all parse into the expected dicts. From venv, `python -m cognition_engine.prompts` printed "prompts.py manual parser tests passed.", confirming the parser behaves as expected for the sample formats.

## 2026-03-09 (Week 4 — TASK-025: Integration test — agent + CHI + compaction)

- TASK-025 — Integration — Agent + CHI + Compaction Working Together — Completed
- Created `scripts/test_agent.py`: sets up a clean `/tmp/cognition_test` directory, loads Qwen2.5-Coder-7B via `mlx_lm.load`, instantiates `CognitionAgent` with a calculator task (implement `calculator.py` + `test_calculator.py` and run tests), and calls `agent.run(max_turns=15)` with `auto_approve=False` so shell commands (e.g. pytest) require confirmation.
- After the run, the script asserts that `calculator.py` exists and prints the files in the test directory. This is the end-to-end harness to observe CHI lines per turn, anchor injections, and context compaction decisions while the agent tries to complete a real coding task.
- Note: Success of the *behavior* (tool format adherence, tests actually passing, [DONE] usage) depends on the local 7B model's instruction following; script is designed so that any issues can be inspected and then logged here for future prompt/agent tuning.
- First real run (`python -m scripts.test_agent` from venv): agent completed 15 turns with **CHI staying in Level 1–2 range (min 0.683, final 0.693)** but **never emitted any tool calls** (`Tools: none` each turn), `Compactions: 0`, `Files modified: []`. Final assertion `calculator.py not created` failed, confirming that with the current SYSTEM_PROMPT + small model, the agent loop runs and reports CHI but the model does not yet follow the `[READ]/[WRITE]/[EDIT]/[EXEC]/[DONE]` protocol in practice.

## 2026-03-09 (Week 4 — TASK-026: Working memory (.cognition/ directory))

- TASK-026 — Working Memory — Completed
- Created `cognition_engine/memory.py` with `WorkingMemory` managing a persistent `.cognition/notes.json` per project directory. `add_note(category, content, related_files)` appends a timestamped note with an incremental `id` and saves immediately; `get_context_for_task(task_description, max_notes=10)` returns a `[PROJECT MEMORY — Previous session notes]` block with the last N notes’ categories and first 200 chars of content; `get_session_summary()` aggregates counts per category and unique `files_referenced`.
- Manual scaffold in `__main__`: writes three sample notes (decision, pattern, bug) against `/tmp/test_project`, verifies `get_context_for_task` includes “SQLite” and “API routes”, checks that `get_session_summary()["total_notes"] == 3`, and confirms persistence by instantiating a second `WorkingMemory` pointing at the same directory and asserting it sees all three notes. From venv, `python -m cognition_engine.memory` printed the expected memory block and `{'total_notes': 3, 'categories': {'decision': 1, 'pattern': 1, 'bug': 1}, 'files_referenced': ['app.py', 'db.py', 'routes.py']}` followed by “WorkingMemory manual scaffold passed.”

## 2026-03-13 (Week 5 — TASK-027: Agent loop fixes from Week 4 integration)

- Strengthened `SYSTEM_PROMPT` in `cognition_engine/prompts.py` with explicit, copyable few-shot examples for `[THINK]`, `[READ]`, `[WRITE]`, and `[EXEC]`, plus a hard rule: **NEVER guess file contents; always [READ] before describing or editing a file.**
- Made `parse_tool_call()` more tolerant: case-insensitive directives (`[Read]`, `[READ]`), flexible separators (`[READ]: path`, `[READ] - path`), and support for putting the path/command on the next line. Also improved directive detection so it skips narrative text and locks onto the first bracketed tool line.
- Wired the tool-format system prompt into the agent loop by seeding `ContextCompactor`’s anchor with `build_system_prompt(task)` in `CognitionAgent.__init__`, so every turn now includes the full tool protocol and task in context.
- Updated `_parse_tool_calls()` in `CognitionAgent` to ignore `[DONE]` as an executable tool and to enforce exactly one structured tool call per response (still delegating parsing to `parse_tool_call()`).
- **Manual test (venv, pre-memory integration):** `python -m scripts.test_agent` — agent completed 15 turns with tool calls (think, write, execute) throughout; `calculator.py` and `test_calculator.py` created; pytest approved and run; CHI min 0.698 (turn 3, Level 2), final 0.811; assertion passed. Output files live in a **tmp folder** (not in repo): `scripts/test_agent.py` uses `TEST_DIR = "/tmp/cognition_test"`. To inspect after a run: `ls -la /tmp/cognition_test/`

## 2026-03-13 (Week 5 — TASK-028: Working memory integration into agent loop)

- Wired `WorkingMemory` into `CognitionAgent` in `cognition_engine/agent.py`: `self.memory = WorkingMemory(working_dir)` plus initial context injection via `memory_context = self.memory.get_context_for_task(task)` added as a `system` turn so each new run sees prior project notes (when they exist).
- Hooked memory writes to tool usage: `[THINK]` calls now create `"decision"` notes (content = thought, `related_files` = current `anchor_mgr.files_modified`), and `[WRITE]` calls create `"file_purpose"` notes with a preview of the written content and `related_files=[path]`.
- Added a `try/except KeyboardInterrupt` around the main loop with a `finally` block that always records a `"context"` note when the agent exits (normal completion or Ctrl+C), including turn count, final CHI (when available), and the list of modified files; notes are stored in `.cognition/notes.json` under the agent's `working_dir` (e.g., `/tmp/cognition_test/.cognition/notes.json` for the calculator test).
- Updated `WorkingMemory.get_context_for_task()` to return an empty string when there are no notes (so we don't inject a confusing `[No previous notes for this project]` block) and to prepend a clarifying line: “Use these as background only. You still must complete the current task using the tool protocol.”
- Smoke test after memory integration & tweak: `python -m scripts.test_agent` — `calculator.py` and `test_calculator.py` created under `/tmp/cognition_test`, `.cognition/notes.json` populated with both `"decision"` and `"file_purpose"` notes (for the calculator class and tests), and `pytest test_calculator.py -v` passes (4/4 tests). Tool behavior remains healthy (think/write/execute across turns) while working memory now records the session.

## 2026-03-18 (Week 5 — TASK-029: Evaluate Qwen3.5/Qwen3 on MLX)

- Added `scripts/eval_qwen35.py`: benchmarks candidate Qwen3.5 models (load/run, tok/s, hidden state extraction via `cognition_engine.hidden_states`, CHI smoke via `CHIMonitor`, and a strict tool-format prompt).
- Attempted Qwen3.5 model IDs:
  - `mlx-community/Qwen3.5-4B-Instruct-4bit` → load failed (HF Hub 401 / repo not found).
  - `mlx-community/Qwen3.5-9B-Instruct-4bit` → load failed (HF Hub 401 / repo not found).
  - Interpretation: Qwen3.5 quantized repos likely not available on `mlx-community` yet (or names differ). Don’t block on conversion per Week 5 note.
- Tested a practical “updated model” fallback that *is* available on MLX:
  - `mlx-community/Qwen3-4B-Instruct-2507-4bit`:
    - Speed: **86.9 tok/s** for a 300-token generation (faster than the recorded Qwen2.5-Coder-7B ~53 tok/s baseline).
    - Hidden states: works; observed shape `(1, 56, 2560)` at `layer_idx=16`.
    - CHI monitoring: works; smoke run produced `chi=0.883 level=1`.
    - Tool-format prompt: contains tool markers, but often collapses formatting (e.g. puts fences inline) and is less reliably “one tool call only.”
- Integration test reality check: running the calculator agent loop with Qwen3-4B can prematurely stop after creating only `calculator.py` (missing tests), so for now **keep Qwen2.5-Coder-7B as the default agent model**; treat Qwen3-4B as a fast chat/experimentation option until tool-following is verified.

## 2026-03-18 (Week 5 — TASK-030: Session management (save/resume))

- Added `cognition_engine/session.py` with `SessionManager`:
  - Persists session state to `.cognition/session.json` (task, status, turn count, timestamp, CHI history/final, files modified, last 10 conversation turns).
  - Detects interrupted sessions (`status == "interrupted"`) and builds a `[RESUMING PREVIOUS SESSION]` context block (last CHI, files, last few turns).
  - Strips common chat-template artifacts (`<|im_start|>`, `<|im_end|>`) from the resume context for readability.
- Wired into `CognitionAgent`:
  - Creates `self.session_mgr = SessionManager(working_dir)`.
  - On startup: if an interrupted session exists, injects resume context as a `system` turn.
  - On exit: saves session state in `finally` and clears it if status is `"completed"`; on Ctrl+C it saves with `"interrupted"`.
- Smoke test (auto-approved, short run): ran a 2-turn agent session in `/tmp/session_smoke` and verified `/tmp/session_smoke/.cognition/session.json` exists with `status: interrupted`, and `get_resume_context()` returns the expected resume block.
- Follow-up fix: tightened completion semantics so the agent only treats **explicit** `[DONE]` as completion (and only then clears `session.json`). This prevents premature natural-language “task complete” outputs from wiping resumable session state.
- Added an additional guardrail: ignore `[DONE]` if emitted before any real tool action (`read/write/edit/execute`) has run, to prevent resumed sessions from getting cleared by an immediate premature `[DONE]`.

## 2026-03-18 (Week 5 — TASK-031: CLI with agent mode, resume, and status)

- Upgraded `cognition_engine/cli.py` to a subcommand-based CLI (`chat`, `agent`, `status`) and kept backward compatibility (no subcommand defaults to chat).
- `chat`: existing interactive CHI loop moved into `run_chat`.
- `agent`: runs `CognitionAgent` with `--task/--dir/--model/--max-turns/--resume/--auto-approve`.
  - `--resume` uses `SessionManager(dir)` to load an interrupted session and reuses the stored task.
- `status`: prints `.cognition/session.json` summary (task/status/turns/CHI/files) and `WorkingMemory` summary (note counts/categories/files referenced).
- Manual checks:
  - `python -m cognition_engine --help` and `python -m cognition_engine agent --help` OK.
  - `python -m cognition_engine agent --task "Create hello.py" --dir /tmp/cli_task031 --auto-approve` created `hello.py` and completed via `[DONE]`.
  - `python -m cognition_engine status --dir /tmp/cli_task031` prints memory summary and no active session.
  - Resume path exercised on `/tmp/cli_task031_resume` where a non-interactive run produced `EOFError` during approval; session was still written to `.cognition/session.json` and `python -m cognition_engine agent --dir /tmp/cli_task031_resume --resume` loads and starts from resume context.
- Follow-up robustness: `CognitionAgent.run` now catches `EOFError` (non-interactive stdin) and exits as `interrupted` instead of crashing.

## 2026-03-18 (Week 5 — TASK-032: MLX community engagement (Path A))

- Prep: identified two high-signal engagement targets (hidden state extraction / hooks pattern + Qwen3.5 model support).
- Suggested MLX core thread to reference/comment on (intermediate outputs / “hooks” equivalent): `https://github.com/ml-explore/mlx/issues/1056`
- Suggested mlx-lm Qwen3.5-related thread to reference/comment on (model support / conversion pain): `https://github.com/ml-explore/mlx-lm/pull/957`
- Draft Discussion post (paste into `ml-explore/mlx-lm` Discussions):
  - **Title:** "Hidden state extraction patterns for runtime monitoring"
  - **Body:**
    - "I’m building a cognitive health monitoring loop on MLX (entropy + repetition + goal-drift) and needed hidden states during/after generation."
    - "Since MLX doesn’t have PyTorch-style forward hooks, the cleanest pattern I’ve found is to wrap a single layer (or module) with a small `DebugWrapper` / capture wrapper, run a forward pass, then restore it in `finally`."
    - "This is lightweight, doesn’t require changes to MLX internals, and works well for monitoring use cases."
    - "Question: has anyone considered adding an optional hook/callback interface to `mlx.nn.Module` (even if limited), or would the project prefer the wrapper pattern as the canonical approach?"
    - "Happy to share a minimal snippet if useful."
- Next step (manual): star/watch `ml-explore/mlx-lm`, post the above as a Discussion (not Issue), and optionally add a short comment to issue #1056 linking the wrapper pattern and the monitoring use case. After posting, record the actual URLs and any maintainer replies here.
- Posted Discussion #3285 on ml-explore/mlx: "Patterns for capturing intermediate layer outputs (forward hooks equivalent)"
  - URL: https://github.com/ml-explore/mlx/discussions/3285
  - Shared CaptureWrapper pattern, mx.eval() gotcha, referenced #1056 and mlx-examples#987
  - No moat revealed: no CHI, no steering, no product details
- Starred + watched ml-explore/mlx and ml-explore/mlx-lm
- Followed awni on GitHub
- TASK-032: COMPLETE
- Next: check for maintainer replies in 24-48 hours
  
**Date:** 2026-03-23 (Week 6 — TASK-033: Activation steering and intervention in Large Language Models (LLMs))
- TASK-033 — Study Steering Math + Architecture Plan — Completed
### 1. The Core Concept of Activation Engineering
Activation engineering aims to elicit or control specific LLM behaviors by modifying the model's internal representations during inference, bypassing the need for computationally expensive weight fine-tuning or reinforcement learning. Two foundational techniques drive this approach:
*   **Activation Addition (ActAdd):** This method extracts a steering vector by calculating the difference in activations between a contrastive pair of prompts (e.g., "Love" versus "Hate"). By adding this vector to the forward pass at inference time, ActAdd can reliably steer high-level output properties like sentiment, tone, and topic. 
*   **Inference-Time Intervention (ITI):** Designed specifically to enhance model "truthfulness," ITI identifies a sparse subset of attention heads that accurately separate true statements from false ones using linear probes. During text generation, ITI shifts the model's internal activations along these specific truth-correlated directions, improving performance on benchmarks like TruthfulQA.

### 2. Practical Capabilities and Limitations
Field testing of activation steering has revealed clear boundaries on what can and cannot be effectively controlled:
*   **Reliably Steerable Behaviors:** Steering vectors are highly effective for altering high-level semantic registers, such as refusal/compliance, formality, sentiment, conciseness, and the expression of uncertainty. 
*   **Effectively Unsteerable Behaviors:** Activation steering completely fails at injecting specific facts, improving factual accuracy, or aiding in complex, multi-step reasoning.
*   **Layer Selection:** Behavioral steering (like refusal or formality) is most effective when applied to the late layers (the last 25%) of the model, as these encode high-level intent and style. Conversely, intervening in the early layers often degrades output quality by corrupting low-level language representations.
*   **Strength Degradation:** The effectiveness of a steering vector (controlled by the intervention strength parameter, $\alpha$) degrades over long generations. After approximately 300 to 500 tokens, the model's autoregressive conditioning gradually pulls its behavior back to its default state. 

### 3. The Advancement of Conditional Activation Steering (CAST)
A major limitation of traditional activation steering is that it is "always on". For example, injecting a refusal vector causes the model to indiscriminately refuse all prompts, including harmless ones, severely limiting the model's utility. 

**Conditional Activation Steering (CAST)** resolves this by adding programmatic control. It works through the following mechanism:
*   **Condition Vectors:** CAST introduces a "condition vector" that acts as a trigger by capturing the activation patterns induced by specific prompt categories (e.g., adult content or hate speech). 
*   **Selective Application:** During inference, the model calculates the similarity between its current hidden state and the condition vector. The behavior vector (e.g., refusal) is only applied if this similarity exceeds a specific threshold.
*   **Logical Composition:** This allows practitioners to build complex, context-dependent rules without weight optimization, such as "if the input is about hate speech or crime, then add refusal". It can also be inverted to constrain a model to a specific domain, refusing all prompts that do *not* match the target condition.

## 2026-03-23 (Week 6 — TASK-034: Contrastive pair generator)

- TASK-034 — Contrastive Pair Generator — Completed
- Created `cognition_engine/steering/contrastive_pairs.py` and `cognition_engine/steering/__init__.py`.
- Implemented three steering pair sets with clear behavioral contrast:
  - `TASK_FOCUS_PAIRS` (anti-drift / stay on task): 10 pairs
  - `TOOL_FORMAT_PAIRS` (anti-hallucination / tool protocol compliance): 10 pairs
  - `CODE_COHERENCE_PAIRS` (anti-degeneration / coherent implementation quality): 10 pairs
- Added `get_all_pairs()` returning all pair sets and `format_pair_for_model(pair, tokenizer)` that formats positive/negative examples with chat templates so hidden-state extraction is comparable.
- Manual verification run:
  - Pair counts: `{'task_focus': 10, 'tool_format': 10, 'code_coherence': 10}`
  - Structural checks passed for every pair (`system`, `positive`, `negative` present)
- Notes:
  - Pairs were written to emphasize behavior-level contrast (focused execution vs drift, concrete tool use vs guessing/description, working code vs stubs/subtle bugs) instead of superficial wording differences.

## 2026-03-23 (Week 6 — TASK-035: Extract activations from contrastive pairs)

- TASK-035 — Extract Activations from Contrastive Pairs — Completed
- Created `cognition_engine/steering/extract_activations.py`.
- Implemented:
  - `CaptureWrapper` for per-layer output capture during forward pass.
  - `extract_all_layer_states(model, tokenizer, text, num_layers=None)`:
    - wraps all transformer layers,
    - runs a single forward pass,
    - extracts **last-token** hidden state per layer,
    - returns `np.ndarray` of shape `(num_layers, hidden_dim)`,
    - always restores original layers in `finally`.
  - `extract_contrastive_activations(model, tokenizer, pairs, pair_name)`:
    - formats positive/negative prompts using chat template,
    - extracts both sides per pair across all layers,
    - returns stacked arrays:
      - `positive`: `(num_pairs, num_layers, hidden_dim)`
      - `negative`: `(num_pairs, num_layers, hidden_dim)`.
- Added runnable `__main__` scaffold:
  - loads `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`,
  - runs extraction for `task_focus`,
  - prints shapes,
  - saves `data/activations_task_focus.npz` (ensures `data/` exists).
- Verification:
  - Import/wiring sanity check passed in project venv (`.venv/bin/python`), confirming module imports and pair integration.
  - Note: full extraction run is model-heavy and should be run as a manual step when ready.
  - Manual extraction run (2 `task_focus` pairs) completed successfully:
    - `positive` shape: `(2, 28, 3584)`
    - `negative` shape: `(2, 28, 3584)`
    - Cosine similarity at layer 16 (pair 1): `0.636230` (`< 1.0`, so positive/negative representations are meaningfully different).

## 2026-03-23 (Week 6 — TASK-036: Compute steering vectors)

- TASK-036 — Compute Steering Vectors — Completed
- Created `cognition_engine/steering/compute_vectors.py`.
- Implemented:
  - `compute_steering_vectors(activations)`:
    - computes per-layer vectors as `mean(positive, axis=0) - mean(negative, axis=0)`,
    - expects activation shape `(num_pairs, num_layers, hidden_dim)`,
    - validates required keys, shape compatibility, and rank.
  - `find_best_layer(steering_vectors)`:
    - computes vector norm per layer,
    - prints all layer norms with middle-layer marker (11-17),
    - returns highest-norm layer index and reports best middle-layer candidate.
  - `save_steering_vectors(vectors, name, output_dir)` + `load_steering_vectors(...)`.
  - `__main__` runner:
    - loads `data/activations_{pair_set}.npz` for `task_focus`, `tool_format`, `code_coherence`,
    - computes vectors, finds best layer, saves to `data/steering_vectors/{pair_set}.npz`,
    - prints best-layer stats (norm/mean/std),
    - skips missing activation files with clear message.
- Manual verification:
  - Synthetic sanity run with mock activations `(4, 28, 16)` passed.
  - Output vector shape confirmed `(28, 16)`.
  - Best-layer selection, save/load round-trip (`np.allclose`) passed.
  - Temporary sanity artifact (`data/steering_vectors/task036_sanity.npz`) removed after test.
  - Full end-to-end run with real data completed (TASK-035 -> TASK-036):
    - Generated activation files for all pair sets:
      - `data/activations_task_focus.npz`
      - `data/activations_tool_format.npz`
      - `data/activations_code_coherence.npz`
      - each with `positive` and `negative` shapes `(10, 28, 3584)`.
    - Ran `python -m cognition_engine.steering.compute_vectors` successfully for all 3 sets and saved:
      - `data/steering_vectors/task_focus.npz`
      - `data/steering_vectors/tool_format.npz`
      - `data/steering_vectors/code_coherence.npz`
    - Lookout checks (from TASK-036) on real output:
      - Norms are clearly non-uniform across layers (strong separation growth by depth) ✅
      - Middle layers (11-17) are stronger than early layers, but final layers dominate in this run (`best layer = 27` for all three sets; `best middle layer = 17`) ⚠️
      - Norms are far from near-zero across sets, so contrastive signal is present (not a weak-pair failure) ✅

## 2026-03-23 (Week 6 — TASK-037: Steering injection wrapper + engine)

- TASK-037 — Build the Steering Injection Wrapper — Completed
- Created `cognition_engine/steering/injector.py`.
- Implemented:
  - `SteeringWrapper(nn.Module)`:
    - wraps a transformer layer,
    - injects `alpha * steering_vector` into hidden states on forward pass,
    - handles both tensor outputs and tuple outputs `(hidden, cache, ...)`,
    - supports runtime toggle with `active` flag.
  - `SteeringEngine`:
    - `load_vectors(name, layer_idx=None, alpha=1.0, vectors_dir=...)` (auto-select highest-norm layer if not provided),
    - `activate()`, `deactivate()`, `set_alpha(alpha)`,
    - `cleanup()` to restore original layers safely,
    - internal wrap/unwrap with original-layer bookkeeping.
  - Runnable `__main__` example:
    - compares generation without steering vs with steering and prints whether outputs differ.
- Manual test results (real vectors from TASK-036):
  - Baseline (no steering): normal calculator-class response (`len=838`).
  - Auto-selected `layer=27`, `alpha=2.0`: response changed but collapsed to near-empty output (`len=1`) — too strong / unstable setting.
  - Follow-up parameter sweep:
    - `layer=17, alpha=1.0` -> `len=836`, output non-empty, `identical=False`
    - `layer=17, alpha=0.5` -> `len=759`, output non-empty, `identical=False`
    - `layer=16, alpha=1.0` -> `len=681`, output non-empty, `identical=False`
    - `layer=27, alpha=0.5` -> `len=828`, output non-empty, `identical=False`
  - Conclusion: steering injection works (responses change), but aggressive top-layer steering (`layer 27`, `alpha=2.0`) can over-steer and degrade output. Middle-layer/lower-alpha settings are more stable for next experiments.
  - Timing benchmark (latency overhead check):
    - Config: Qwen2.5-Coder-7B-4bit, `max_tokens=120`, steering `task_focus` at `layer=17`, `alpha=1.0`.
    - Benchmark 1 (5-run average, separate blocks):
      - Baseline: `2.2747s`, Steered: `2.2512s` (difference `-23.467ms`, within run-to-run noise)
    - Benchmark 2 (interleaved A/B, 6 pairs for tighter comparison):
      - Baseline avg: `2.2181s`
      - Steered avg: `2.2158s`
      - Overhead: `-2.364ms / response` (effectively zero; noise-dominated)
      - Approx overhead per output token: `-0.0197ms/token` (noise floor)
    - Interpretation: No measurable slowdown from single-layer steering injection in this setup; practical overhead appears negligible and consistent with the "<1ms per layer per token" expectation.

## 2026-03-24 (Week 6 — TASK-038: First steering experiment, CHI comparison)

- TASK-038 — First steering experiment (`scripts/steering_experiment.py`) — Completed run on local hardware.
- **Setup:** `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`, `task_focus` vectors, auto layer **27**, **alpha=2.0**, 10 turns (calculator build-out), `max_tokens=300`.
- **Automated success criteria (from saved summary): both satisfied**
  - **Mean CHI after turn 5** (turns 5–9): steered **0.832** vs unsteered **0.799** → steered higher ✅
  - **First turn CHI below 0.7:** both **null** (neither run dropped below threshold) → script marks “steered crosses threshold later” as true (steered never crossed) ✅
- **Per-turn pattern (terminal):**
  - Unsteered: CHI mostly high early, then slides **~0.874 → 0.800 → 0.750 → 0.714** on turns 6–9 as context grows (~2.2k–3.2k tokens).
  - Steered: CHI stays **~0.826–0.906** across all turns; final context ~**400** tokens vs ~**3157** unsteered (steered outputs much shorter — model often stopped well before `max_tokens`).
- **Interpretation:** Matches the Week 6 “money graph” intent: steered curve stays flatter and higher while unsteered degrades toward the end of the scripted task. **Caveat:** the two runs can differ in response length (same `max_tokens` cap, different early stopping); CHI is not length-matched across arms.
- **Artifacts:** `data/steering_experiment.json`, `data/steering_experiment.png`.
- **Note:** A brief experiment with per-turn caps + matched-length CHI replay was **reverted**; it did not fix steering stability and produced degenerate empty steered turns under strong steering. The script again uses the simple protocol: same global `max_tokens` for both conditions, live CHI only.

## 2026-03-25 (Week 7 — TASK-039: Alpha/Layer Sweep updates)

- Implemented TASK-039 sweep runner as `scripts/steering_sweep.py` (9 layers × 5 alphas = 45 runs, 5-turn conversation).
- **Baseline variance control:** script runs the **unsteered baseline first** and caches it to `data/steering_sweep_baseline.json`; every steered run computes `response_changed` against this cached baseline (rerun baseline only with `--recompute-baseline`).
- **Normalized second pass (requested change):** added a follow-up pass that re-runs the **top 5** (layer, alpha) pairs using **L2-normalized steering vectors** (vector / ||vector||, then injection scales by alpha).
  - Added `normalize` flag to `SteeringEngine.load_vectors(..., normalize=True/False)` in `cognition_engine/steering/injector.py`.
- Outputs:
  - `data/steering_sweep.json` (all 45 results + `top5_raw` + `normalized_followup`)
  - `data/steering_sweep.png` heatmap (x=layer, y=alpha, color=mean_chi; dead cells marked with X)
- Run:
  - `python3 scripts/steering_sweep.py`
  - Optional: `python3 scripts/steering_sweep.py --vector-name task_focus --max-tokens 250`

## 2026-03-25 (Week 7 — TASK-040: Updated implementation scaffold)

- Implemented `scripts/steering_experiment_v2.py` per updated TASK-040 instructions.
- Supports 4 conditions with calibrated defaults (raw vectors only):
  - `baseline`
  - `task_focus` @ layer 16, alpha 0.5
  - `tool_format` @ layer 16, alpha 0.5
  - `code_coherence` @ layer 16, alpha 0.5
- Tracks required per-turn fields for each condition:
  - `turn`, `chi_score`, `chi_raw`
  - `entropy`, `repetition`, `goal_drift`
  - `response_length`, `context_length`, `response_preview` (100 chars)
- Added CLI modes for thermal management and staged execution:
  - `--condition baseline|task_focus|tool_format|code_coherence|all`
  - `--cooldown-between-runs` (default 120s when running `--condition all`)
  - `--plot-only` to regenerate combined graph from saved JSON without rerunning model inference.
- Outputs:
  - `data/steering_experiment_v2.json` (all condition data + computed summary)
  - `data/steering_experiment_v2.png` with:
    - top subplot: 4 CHI curves + threshold line (0.7)
    - bottom subplot: response length per turn for each condition
- Summary now computes the updated success checks:
  - responses alive (>20 tokens for all 10 turns on steered runs),
  - at least one vector improving mean CHI on turns 5–9 vs baseline,
  - per-vector length parity within 50% of baseline mean response length.

## 2026-03-26 (Week 7 — TASK-042: Adaptive alpha in agent loop)

- (Reverted) This entry was superseded: we rolled back adaptive alpha (TASK-042) in favor of completing TASK-041 first.

## 2026-03-26 (Week 7 — TASK-041: CHI-triggered steering in agent loop)

- Implemented CHI-triggered steering (fixed alpha) in `cognition_engine/agent.py`:
  - `--enable-steering` flag enables steering (default off).
  - Loads `tool_format` vectors once at init via `SteeringEngine.load_vectors(...)` (layer 16, alpha 0.5), starts inactive.
  - After each CHI update: activates steering when CHI < 0.7, deactivates when CHI > 0.8 (hysteresis).
  - Status line prints steering state: `STEERING: ON/OFF (alpha=0.500)`.
  - Cleanup restores original layers on exit.
- Session save/resume: `SessionManager` now persists `steering_active` in `.cognition/session.json` and restores active steering state on resume.
- CLI help text updated to describe Level 2 intervention (no adaptive alpha).

## 2026-03-26 (Week 7 — TASK-042: Adaptive alpha + Level 3 compaction check)

- Re-introduced adaptive alpha in `cognition_engine/steering/injector.py` with `compute_adaptive_alpha(chi_score, base_alpha=0.5)`:
  - CHI >= 0.8 -> alpha 0.0
  - CHI in [0.3, 0.8) -> piecewise-linear ramp to `1.5 * base_alpha`
  - CHI < 0.3 -> capped at `1.5 * base_alpha`
- Integrated into `cognition_engine/agent.py` steering loop:
  - computes adaptive alpha after each CHI update,
  - applies alpha when steering activates and updates alpha every turn while steering stays active,
  - activation/deactivation thresholds remain 0.7 / 0.8 (hysteresis),
  - status line continues to print current alpha.
- Added explicit Level-3 compaction trigger path:
  - when `chi < 0.5` and `needs_compaction(chi)` is true, logs
    `[COMPACTION] Level 3 triggered ...` and runs `compact("auto")`.
- Quick sanity check for adaptive alpha mapping (base=0.5): `CHI 0.8 -> 0.0`, `0.7 -> 0.39`, `0.5 -> 0.57`, `0.3 -> 0.75`, `0.2 -> 0.75`.

## 2026-03-27 (Week 7 — TASK-043: Dual-vector steering, reduced scope)

- Implemented reduced-scope TASK-043 wiring in `scripts/steering_experiment_v2.py`:
  - Added new condition: `dual_vector`.
  - `dual_vector` now loads two steering engines simultaneously with safe params from revised task:
    - `tool_format` @ layer 16, alpha 0.3
    - `code_coherence` @ layer 17, alpha 0.25
  - Both engines are activated/deactivated each turn around generation, then cleaned up safely.
- Updated experiment reporting:
  - Added `dual_vector_vs_single` block in summary with:
    - best single-vector mean CHI (turns 5-9),
    - dual-vector mean CHI (turns 5-9),
    - boolean flag for whether dual beats best single.
  - Added `dual_vector` plot color so CHI/length charts include the new arm.
- Validation:
  - Syntax/compile check passed: `python3 -m py_compile scripts/steering_experiment_v2.py`.
- Run command for the actual TASK-043 measurement:
  - `python scripts/steering_experiment_v2.py --condition dual_vector`
  - Compare output JSON summary against existing `tool_format` and `code_coherence` single-vector runs.
- Actual dual-vector run completed (10 turns) and saved:
  - `data/steering_experiment_v2.json`
  - `data/steering_experiment_v2.png`
- Key outcomes from this run:
  - **Outcome C (degrades vs best single):** dual-vector mean CHI turns 5-9 = `0.813997` vs best single (`code_coherence`) = `0.818518` -> dual does **not** beat best single.
  - **Alive-turn failure:** turn 8 response length = `4` tokens, so `responses_alive_gt_20_all_turns = false` for dual-vector.
  - Dual still beats baseline on mean CHI turns 5-9 (`0.813997` vs `0.798707`) and keeps length parity within 50%, but reliability is worse than `tool_format`-only.
- Decision for now: keep single-vector default (`tool_format`) for integration work; revisit multi-vector only after full ladder testing (TASK-044+) or lower-alpha retune.

## 2026-03-28 (Week 7 — TASK-044: Full intervention ladder integration harness)

- Added `scripts/test_steering_agent.py` (TASK-044): three modes on the calculator task — `baseline` (no steering, no compaction), `steering` (Level 2 + adaptive alpha, compaction off), `full_ladder` (steering + compaction + Level 4 log line when CHI < 0.3).
- Extended `CognitionAgent`: `enable_compaction` (default True for CLI), optional `turn_log` list for per-turn TASK-044 JSON rows; compaction block skipped when compaction disabled.
- Extended `ContextCompactor.last_compaction_strategy` set whenever `compact()` increments `compaction_count`.
- Outputs: `data/task044_{mode}.json`; `--plot-comparison` or `--plot-only` builds `data/task044_comparison.png` and `data/task044_report.md`.
- Example runs (use `--auto-approve` for unattended pytest):
  - `caffeinate -dimsu .venv/bin/python3 -u scripts/test_steering_agent.py --mode baseline --dir /tmp/task044_baseline --max-turns 20 --auto-approve`
  - same with `--mode steering --dir /tmp/task044_steering`
  - same with `--mode full_ladder --dir /tmp/task044_full`
- `python3 -m py_compile` passed on `agent.py`, `compaction.py`, `test_steering_agent.py`.

## 2026-03-29 (Week 8 — TASK-045: Completion detection)

- **Heuristic completion** in `cognition_engine/agent.py`: `_extract_expected_files()` scans the task for relative paths with common extensions (`.py`, `.md`, etc.); `_check_task_completion()` requires all of those files to exist under `working_dir` and at least one `[EXEC]` in the last three `tool_history` entries with `exit_code == 0`. After each turn (post-CHI/compaction), if the guard passes and an action tool has run, the loop stops with the same session semantics as explicit `[DONE]` (`completed_explicitly`, session cleared).
- **`tool_history` records `exit_code`** for execute tools (via `_execute_tool` → `_tool_execute` returning `(message, code)`); cancelled or failed subprocess paths omit a successful zero code so the heuristic does not misfire.
- **Anti-loop guard (TASK-045 C):** if the last three history entries share the same tool signature (`_tool_call_signature`), inject a system turn: *You are repeating the same action. If the task is complete, say [DONE].*
- **`cognition_engine/prompts.py`:** added COMPLETION RULE text to `SYSTEM_PROMPT` and a few-shot block showing `[DONE]` as the follow-up after a successful `pytest`.
- Repo checklist: `.cofounder/TASKS_WEEK8.md` — TASK-045 marked done.
- **Manual verify (recommended):** `python scripts/test_steering_agent.py --mode baseline --dir /tmp/task045_calc --max-turns 20 --auto-approve` — expect exit before max turns when files + pytest 0 align with heuristic or model emits `[DONE]`.
- Git: `TASK-045: completion detection — heuristic + prompt + anti-loop`

## 2026-03-29 (Week 8 — TASK-045D: test-only heuristic)

- **False-positive fix:** `_check_task_completion()` no longer treats every exit-0 `[EXEC]` as completion. Added module helper `_command_looks_like_test_run()` — matches `python -m pytest`, `python -m unittest`, `python test_…`, or `pytest` invoked as a shell command (with path segments), and explicitly **excludes** `pip`/`pip3 install` and `python -m pip install` so `pip install pytest` does not satisfy the heuristic.
- **`tool_history`:** execute entries still come from `dict(tool_call)` and include `command` (unchanged); matching uses `tool` + `command` + `exit_code` (not `tool_used`).
- Git: `TASK-045D: filter heuristic to test commands only`

## 2026-03-29 (Week 8 — TASK-045E: [DONE] must match test evidence)

- **`cognition_engine/prompts.py`:** under COMPLETION RULE, added IMPORTANT text: installing packages is not running tests; must re-run pytest (or equivalent) and only say `[DONE]` after the test command exits 0 with passing results.
- **`cognition_engine/agent.py`:** `_recent_successful_test_command()` factored from `_check_task_completion()`; `_task_expects_tests()` detects tasks that mention pytest/tests/verify; `_explicit_done_allowed()` — if the task implies tests, `[DONE]` is accepted only when evidence matches the heuristic (expected files on disk + successful test command in recent history), or if no filenames were extracted from the task text, at least a recent successful test run. Otherwise the loop injects a system correction and continues (no session clear).
- Manual verify: `python scripts/test_steering_agent.py --mode baseline --dir /tmp/task045_v2 --max-turns 20 --auto-approve`
- Git: `TASK-045E: validate [DONE] against test evidence + prompt`

## 2026-03-29 (Week 8 — TASK-045F: pytest in harness + no-tool guard)

- **`scripts/test_steering_agent.py`:** before `CognitionAgent` / `run()`, run `pip install -q pytest` with `sys.executable` so TASK-044 calculator runs are not blocked by a missing pytest in the environment (TASK-045F).
- **`cognition_engine/agent.py`:** `self._consecutive_no_tool` initialized to 0; after `_parse_tool_calls`, increment if no tools else reset; when count reaches ≥ 3, log `[NO-TOOL-GUARD]`, inject system message requiring `[WRITE]`, `[READ]`, `[EXEC]`, or `[DONE]`, then reset counter (covers degenerate empty / special-token loops with `tool_used: none`).
- Manual verify: `python scripts/test_steering_agent.py --mode baseline --dir /tmp/task045_v3 --max-turns 20 --auto-approve`
- Git: `TASK-045F: pre-install pytest + no-tool-output guard`

## 2026-03-29 (Week 8 — TASK-045G: pytest PATH / `python -m pytest`)

- **`cognition_engine/prompts.py`:** few-shot example now uses `[EXEC] python -m pytest`; COMPLETION hint prefers `python -m pytest` for venv correctness.
- **`cognition_engine/agent.py`:** `_tool_execute` normalizes bare `pytest`, `pytest .`, and commands starting with `pytest ` to `sys.executable -m pytest` before approval/subprocess; output messages use the resolved command.
- Git: `TASK-045G: fix pytest PATH — use python -m pytest`

## 2026-03-29 (Week 8 — TASK-045H: chat template tokens in tool args)

- **Root cause:** Qwen-style outputs append `<|im_end|>` (and similar `<|...|>` tokens) to lines; a command like `python -m pytest<|im_end|>` was passed to the shell and failed with a syntax error.
- **`cognition_engine/prompts.py`:** `strip_chat_template_artifacts()` removes `<|im_end|>` and regex-strips `<|[^|]*|>`; applied to full response before parsing and to each tool field (read/exec/think/write/edit); `_extract_code_block` applies strip to extracted block text.
- **`cognition_engine/agent.py`:** import strip; `_tool_execute` strips first (then 045G pytest rewrite); `_execute_tool` defensively strips paths/content/command/thought for read/write/edit/think. Removed temporary `[DEBUG _tool_execute]` print.
- Git: `TASK-045H: strip chat template tokens from tool arguments`