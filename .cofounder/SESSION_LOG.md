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