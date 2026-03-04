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

