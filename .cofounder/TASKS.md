# TASKS.md
# Implement → Test manually → Check off → Next task. Don't skip.
# Status: ⬜ TODO | 🔨 IN PROGRESS | ✅ DONE | ❌ BLOCKED

---

## WEEK 1: MLX Inference + Hidden State Extraction
**Goal:** Run a local model and prove you can see inside its brain.

---

### TASK-001: Environment Setup
**Status:** ⬜ TODO
**Do:**
```bash
cd cognition-engine
python3 -m venv .venv
source .venv/bin/activate
pip install mlx mlx-lm
```
**Manual test:** Run these — both should print without error:
```bash
python -c "import mlx; print(mlx.__version__)"
python -c "import mlx_lm; print('mlx_lm OK')"
```
**Time:** 15 min
**Git:** `git commit -m "TASK-001: environment setup"`

---

### TASK-002: Download and Run First Model
**Status:** ⬜ TODO
**Do:** Create `scripts/first_inference.py`:
```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")

prompt = "Write a Python function that computes the nth fibonacci number"
response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
print(response)
```
**Manual test:** Run it. You should see a coherent Python function.
**Time:** 30 min (mostly model download)
**Git:** `git commit -m "TASK-002: first model inference working"`

---

### TASK-003: Measure Inference Speed
**Status:** ⬜ TODO
**Do:** Create `scripts/benchmark.py`:
```python
import time
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")

prompt = "Write a Python class implementing a binary search tree with insert, search, and delete"

start = time.time()
response = generate(model, tokenizer, prompt=prompt, max_tokens=500)
elapsed = time.time() - start

tokens = len(tokenizer.encode(response))
tps = tokens / elapsed
print(f"{tokens} tokens in {elapsed:.1f}s = {tps:.1f} tok/s")
```
**Manual test:** You see a tok/s number. Expected: 25-40 on M4 Pro for 7B Q4.
**Log it:** Write the actual number in SESSION_LOG — this is your hardware baseline.
**Time:** 15 min
**Git:** `git commit -m "TASK-003: benchmark baseline [X] tok/s"`

---

### TASK-004: Read mlx-lm Source Code (Learning Task — No Code to Write)
**Status:** ⬜ TODO
**Do:** Find and read these source files:
```bash
python -c "import mlx_lm; print(mlx_lm.__file__)"
# Then open and read:
# 1. mlx_lm/models/qwen2.py → model architecture
# 2. mlx_lm/generate.py → how tokens are generated step by step
# 3. mlx_lm/utils.py → how models are loaded
```
**What to understand:**
- Where are transformer layers? (likely `model.model.layers`)
- What does each layer's forward() do?
- Where does attention happen?
- How is the KV cache managed?
- What is the hidden dimension size for Qwen2.5-7B?

**Manual test:** You can answer these questions from memory:
- [ ] How many transformer layers does Qwen2.5-7B have?
- [ ] What is the hidden dimension?
- [ ] What module handles self-attention?
**Time:** 2-3 hours (this is deep learning education, take your time)

---

### TASK-005: Extract Hidden States During Inference ⚠️ HARDEST TASK
**Status:** ⬜ TODO
**Do:** Create `engine/hidden_states.py`

The goal: intercept a transformer layer's output during generation.

**Approach hints** (figure out the details yourself — that's the learning):
- MLX may NOT have PyTorch-style `register_forward_hook()`
- You may need to monkey-patch a layer's `__call__` method
- Or wrap the layer in a custom class that captures output
- The hidden state shape should be something like `(1, seq_len, hidden_dim)`

**Skeleton to start from:**
```python
import mlx.core as mx
from mlx_lm import load, generate

def extract_hidden_states(model, tokenizer, prompt, layer_idx=16, max_tokens=100):
    """
    Run inference and capture the hidden state at a specific layer.
    Returns: (generated_text, hidden_state_tensor)
    """
    # TODO: Your job to figure out how to intercept the layer output
    # Hint: look at how model.model.layers[layer_idx] is called
    pass

if __name__ == "__main__":
    model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    text, states = extract_hidden_states(model, tokenizer, "Write hello world in Python")
    print(f"Generated: {text[:100]}...")
    print(f"Hidden state shape: {states.shape}")
```

**Manual test:** Running this prints a shape like `(1, N, 3584)` where 3584 is the hidden dim.

**If stuck after 1-2 hours:** This is the right time to ask Claude. Open new chat, paste COFOUNDER_BRAIN.md, show your code + error.

**Time:** 4-6 hours (expect to struggle — that's normal)
**Git:** `git commit -m "TASK-005: hidden state extraction working"`

---

### TASK-006: Verify Hidden States Are Meaningful
**Status:** ⬜ TODO
**Do:** Create `scripts/verify_states.py`

Run two very different prompts through TASK-005's code:
- Prompt A: "Write a Python sorting algorithm"
- Prompt B: "Write a poem about the ocean"

Capture hidden states from both at the same layer. Compute cosine similarity between them (average across sequence length first).

**Manual test:** Cosine similarity should be < 0.95. If it's exactly 1.0, your capture mechanism from TASK-005 is broken.
**Time:** 1 hour
**Git:** `git commit -m "TASK-006: hidden states verified meaningful"`

---

### TASK-007: Extract Attention Weights (Prep for Week 2)
**Status:** ⬜ TODO
**Do:** Extend your hidden state approach to also capture attention weights from the self-attention module. Print their shape.

**Manual test:** You can print attention weight shapes for at least one layer.
**Time:** 2-3 hours
**Git:** `git commit -m "TASK-007: attention weight extraction working"`

---

## WEEK 1 CHECKLIST
- [ ] TASK-001: Environment setup (15 min)
- [ ] TASK-002: First inference (30 min)
- [ ] TASK-003: Benchmark speed (15 min)
- [ ] TASK-004: Read source code (2-3 hrs)
- [ ] TASK-005: Hidden state extraction (4-6 hrs) ← HARDEST
- [ ] TASK-006: Verify states (1 hr)
- [ ] TASK-007: Attention weights (2-3 hrs)

**Total: ~12-16 hours across 3-4 days**

---

## WEEK 2: Entropy + Degradation Visualization
(Detailed tasks added after Week 1 is complete)

- TASK-008: Token-level entropy computation
- TASK-009: Multi-turn entropy tracking (100+ turns)
- TASK-010: Degradation visualization (matplotlib graph)
- TASK-011: Find the degradation inflection point
- TASK-012: Output repetition detection (Jaccard similarity)
- TASK-013: First blog post draft with entropy graphs

## WEEK 3: CHI Score + CLI Skeleton
(Detailed tasks added after Week 2 is complete)
