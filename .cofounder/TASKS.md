# [TASKS.md](http://TASKS.md)

# Implement → Test manually → Check off → Next task. Don't skip.

# Status: ⬜ TODO | 🔨 IN PROGRESS | ✅ DONE | ❌ BLOCKED

---

## WEEK 1: MLX Inference + Hidden State Extraction

**Goal:** Run a local model and prove you can see inside its brain.

---

### TASK-001: Environment Setup

**Status:** ⬜ TODO **Do:**

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

**Time:** 15 min **Git:** `git commit -m "TASK-001: environment setup"`

---

### TASK-002: Download and Run First Model

**Status:** ⬜ TODO **Do:** Create `scripts/first_inference.py`:

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")

prompt = "Write a Python function that computes the nth fibonacci number"
response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
print(response)
```

**Manual test:** Run it. You should see a coherent Python function. **Time:** 30 min (mostly model download) **Git:** `git commit -m "TASK-002: first model inference working"`

---

### TASK-003: Measure Inference Speed

**Status:** ⬜ TODO **Do:** Create `scripts/benchmark.py`:

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

**Manual test:** You see a tok/s number. Expected: 25-40 on M4 Pro for 7B Q4. **Log it:** Write the actual number in SESSION_LOG — this is your hardware baseline. **Time:** 15 min **Git:** `git commit -m "TASK-003: benchmark baseline [X] tok/s"`

---

### TASK-004: Read mlx-lm Source Code (Learning Task — No Code to Write)

**Status:** ⬜ TODO

**⚠️ IMPORTANT: There are TWO separate repos. Don't confuse them:**

- `mlx` (github.com/ml-explore/mlx) = low-level array framework. Like NumPy for Apple Silicon. You do NOT need to read this now.
- `mlx-lm` (github.com/ml-explore/mlx-lm) = LLM inference package built ON mlx. **THIS is where the model code lives.**

**Do:** Find the mlx-lm source code on your machine:

```bash
# Find where mlx-lm is installed
python -c "import mlx_lm; print(mlx_lm.__file__)"
# This will print something like:
# /path/to/.venv/lib/python3.xx/site-packages/mlx_lm/__init__.py

# Navigate to that directory and explore
ls $(python -c "import os, mlx_lm; print(os.path.dirname(mlx_lm.__file__))")
ls $(python -c "import os, mlx_lm; print(os.path.dirname(mlx_lm.__file__))")/models/
```

**Read these files IN ORDER:**

1. `mlx_lm/models/qwen2.py` → The Qwen2 model architecture
  - Online: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen2.py
  - Look for: `Qwen2Model`, `Qwen2DecoderLayer`, `Qwen2Attention`
  - Note how `self.model.layers` is a list of transformer decoder layers
  - Note the hidden_size, num_heads, num_layers in the ModelArgs
2. `mlx_lm/generate.py` → How tokens are generated step by step
  - Online: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py
  - Look for: `generate_step()` — this is the core generation loop
  - Note how it calls `model(inputs, cache=cache)` each step
  - Note how logits are converted to token probabilities via sampling
3. `mlx_lm/utils.py` → How models are loaded from HuggingFace
  - Online: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/utils.py
  - Look for: `load()` function — this is what you call in your scripts
  - Note how it downloads weights, loads config, builds the model object

**What to understand (write answers in your session log):**

- [ ] How many transformer layers does Qwen2.5-7B have? (look at config)
- [ ] What is the hidden dimension size? (look at ModelArgs.hidden_size)
- [ ] Where does self-attention happen? (look at Qwen2Attention class)
- [ ] How is the KV cache passed through layers? (look at the cache parameter)
- [ ] Where are logits computed from hidden states? (look at the lm_head)

**Tip:** Open these files in Cursor. Use Cursor's AI to help explain specific functions, but make sure YOU understand the flow, not just the AI.

**Time:** 2-3 hours (this is your ML education, take notes)

---

### TASK-005: Extract Hidden States During Inference ⚠️ HARDEST TASK

**Status:** ⬜ TODO **Do:** Create `engine/hidden_states.py`

The goal: intercept a transformer layer's output during generation.

**Approach hints** (figure out the details yourself — that's the learning):

- MLX may NOT have PyTorch-style `register_forward_hook()`
- You may need to monkey-patch a layer's `__call_`_ method
- Or wrap the layer in a custom class that captures output
- Look at how `model.model.layers[N]` is structured in qwen2.py
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
    # Hint: the layer's __call__ returns a tuple or tensor
    pass

if __name__ == "__main__":
    model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    text, states = extract_hidden_states(model, tokenizer, "Write hello world in Python")
    print(f"Generated: {text[:100]}...")
    print(f"Hidden state shape: {states.shape}")
```

**Manual test:** Running this prints a shape like `(1, N, 3584)` where 3584 is the hidden dim.

**If stuck after 1-2 hours:** This is the right time to ask Claude. Open new chat, paste COFOUNDER_BRAIN.md, show your code + error + what you tried.

**Time:** 4-6 hours (expect to struggle — that's normal and where real learning happens) **Git:** `git commit -m "TASK-005: hidden state extraction working"`

---

### TASK-006: Verify Hidden States Are Meaningful

**Status:** ⬜ TODO **Do:** Create `scripts/verify_states.py`

Run two very different prompts through TASK-005's code:

- Prompt A: "Write a Python sorting algorithm"
- Prompt B: "Write a poem about the ocean"

Capture hidden states from both at the same layer. Compute cosine similarity between them (average across sequence length first).

**Manual test:** Cosine similarity should be < 0.95. If it's exactly 1.0, your capture mechanism from TASK-005 is broken. **Time:** 1 hour **Git:** `git commit -m "TASK-006: hidden states verified meaningful"`

---

### TASK-007: Extract Attention Weights (Prep for Week 2)

**Status:** ⬜ TODO **Do:** Extend your hidden state approach to also capture attention weights from the self-attention module (`Qwen2Attention` in qwen2.py). Print their shape.

**Hint:** Look at how `Qwen2Attention.__call__()` computes attention. The attention weights are computed before the output projection. You may need to modify or wrap the attention module similarly to TASK-005.

**Manual test:** You can print attention weight shapes for at least one layer. **Time:** 2-3 hours **Git:** `git commit -m "TASK-007: attention weight extraction working"`

---

## WEEK 1 CHECKLIST

- [ ] TASK-001: Environment setup (15 min)
- [ ] TASK-002: First inference (30 min)
- [ ] TASK-003: Benchmark speed (15 min)
- [ ] TASK-004: Read mlx-lm source code (2-3 hrs)
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