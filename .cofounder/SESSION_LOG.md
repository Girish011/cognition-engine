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