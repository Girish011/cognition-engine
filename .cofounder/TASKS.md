# [TASKS.md](http://TASKS.md)

# Implement → Test manually → Check off → Next task. Don't skip.

# Status: ⬜ TODO | 🔨 IN PROGRESS | ✅ DONE | ❌ BLOCKED

---

## WEEK 1: MLX Inference + Hidden State Extraction

**Goal:** Run a local model and prove you can see inside its brain.

---

### TASK-001: Environment Setup

**Status:** ✅ DONE  **Do:**

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

**Status:** ✅ DONE  **Do:** Create `scripts/first_inference.py`:

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

**Status:** ✅ DONE  **Do:** Create `scripts/benchmark.py`:

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

**Status:** ✅ DONE 

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
  - Online: [https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen2.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen2.py)
  - Look for: `Qwen2Model`, `Qwen2DecoderLayer`, `Qwen2Attention`
  - Note how `self.model.layers` is a list of transformer decoder layers
  - Note the hidden_size, num_heads, num_layers in the ModelArgs
2. `mlx_lm/generate.py` → How tokens are generated step by step
  - Online: [https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py)
  - Look for: `generate_step()` — this is the core generation loop
  - Note how it calls `model(inputs, cache=cache)` each step
  - Note how logits are converted to token probabilities via sampling
3. `mlx_lm/utils.py` → How models are loaded from HuggingFace
  - Online: [https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/utils.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/utils.py)
  - Look for: `load()` function — this is what you call in your scripts
  - Note how it downloads weights, loads config, builds the model object

**What to understand (write answers in your session log):**

- How many transformer layers does Qwen2.5-7B have? (look at config)
- What is the hidden dimension size? (look at ModelArgs.hidden_size)
- Where does self-attention happen? (look at Qwen2Attention class)
- How is the KV cache passed through layers? (look at the cache parameter)
- Where are logits computed from hidden states? (look at the lm_head)

**Tip:** Open these files in Cursor. Use Cursor's AI to help explain specific functions, but make sure YOU understand the flow, not just the AI.

**Time:** 2-3 hours (this is your ML education, take notes)

---

### TASK-005: Extract Hidden States During Inference ⚠️ HARDEST TASK

**Status:** ✅ DONE  **Do:** Create `engine/hidden_states.py`

The goal: intercept a transformer layer's output during generation.

**Approach hints** (figure out the details yourself — that's the learning):

- MLX may NOT have PyTorch-style `register_forward_hook()`
- You may need to monkey-patch a layer's `__call`__ method
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

**Status:** ✅ DONE **Do:** Create `scripts/verify_states.py`

Run two very different prompts through TASK-005's code:

- Prompt A: "Write a Python sorting algorithm"
- Prompt B: "Write a poem about the ocean"

Capture hidden states from both at the same layer. Compute cosine similarity between them (average across sequence length first).

**Manual test:** Cosine similarity should be < 0.95. If it's exactly 1.0, your capture mechanism from TASK-005 is broken. **Time:** 1 hour **Git:** `git commit -m "TASK-006: hidden states verified meaningful"`

---

### TASK-007: Extract Attention Weights (Prep for Week 2)

**Status:** ✅ DONE **Do:** Extend your hidden state approach to also capture attention weights from the self-attention module (`Qwen2Attention` in qwen2.py). Print their shape.

**Hint:** Look at how `Qwen2Attention.__call__()` computes attention. The attention weights are computed before the output projection. You may need to modify or wrap the attention module similarly to TASK-005.

**Manual test:** You can print attention weight shapes for at least one layer. **Time:** 2-3 hours **Git:** `git commit -m "TASK-007: attention weight extraction working"`

---

## WEEK 1 CHECKLIST

- TASK-001: Environment setup (15 min)
- TASK-002: First inference (30 min)
- TASK-003: Benchmark speed (15 min)
- TASK-004: Read mlx-lm source code (2-3 hrs)
- TASK-005: Hidden state extraction (4-6 hrs) ← HARDEST
- TASK-006: Verify states (1 hr)
- TASK-007: Attention weights (2-3 hrs)

**Total: ~12-16 hours across 3-4 days**

---

## WEEK 2: Entropy Computation + Degradation Detection

**Goal:** : Build the measurement system that becomes the foundation of CHI.
By end of Week 2, you'll have a graph showing EXACTLY where a model degrades.

### TASK-008: Token-Level Entropy from Logits
**Status:** ✅ DONE
**Do:** Create `engine/entropy.py`

Compute Shannon entropy over the model's next-token probability distribution.
This tells you how "certain" the model is at each token. High entropy = uncertain/confused.
Low entropy = confident.

```python
import mlx.core as mx

def compute_token_entropy(logits: mx.array) -> mx.array:
    """
    Compute Shannon entropy of the next-token distribution.
    
    Args:
        logits: shape (1, seq_len, vocab_size) — raw model output
    
    Returns:
        entropy: shape (seq_len,) — entropy per position
        
    Formula: H = -Σ p(x) * log2(p(x))
    """
    # Step 1: Convert logits to probabilities via softmax
    # Step 2: Compute p * log2(p) for each vocab item
    # Step 3: Sum across vocab dimension, negate
    # Step 4: Handle numerical stability (log of 0 = -inf)
    #
    # YOUR IMPLEMENTATION HERE
    pass
```

To get logits, you need a DIFFERENT approach than generate().
You need to run a single forward pass and capture the raw output:

```python
def get_logits_for_sequence(model, tokenizer, text):
    """
    Run a single forward pass and return raw logits.
    Unlike generate(), this doesn't sample — just returns
    the model's probability distribution at every position.
    """
    tokens = mx.array(tokenizer.encode(text))[None]  # (1, seq_len)
    logits = model(tokens)  # (1, seq_len, vocab_size)
    mx.eval(logits)
    return logits
```

**Manual test:** 
```python
# High entropy expected (many valid next tokens):
e1 = compute_token_entropy(get_logits_for_sequence(model, tokenizer, "The"))
print(f"'The' entropy: {e1.mean().item():.4f}")  # Should be high (6-10+)

# Lower entropy expected (very predictable continuation):
e2 = compute_token_entropy(get_logits_for_sequence(model, tokenizer, "def fibonacci(n):"))
print(f"'def fibonacci(n):' entropy: {e2.mean().item():.4f}")  # Should be lower
```

The code prompt should have meaningfully lower average entropy than the open-ended prompt.

**Time:** 2-3 hours
**Git:** `git commit -m "TASK-008: token-level entropy computation"`

---

### TASK-009: Entropy Over a Multi-Turn Conversation
**Status:** ✅ DONE
**Do:** Create `scripts/multiturn_entropy.py`

This is the KEY experiment. You're going to:
1. Give the model a coding task
2. Have a multi-turn conversation (50-100+ turns)
3. Track entropy at each turn
4. Find where entropy starts rising (= model getting confused)

```python
"""
Multi-turn entropy tracking experiment.

The model will be given a coding project and asked to build it
step by step over many turns. We track entropy at each turn
to find where degradation begins.
"""
from engine.entropy import compute_token_entropy, get_logits_for_sequence
from mlx_lm import load, generate
import json

# Strategy: Build a conversation context manually.
# After each model response, append it to the context
# and ask the next question. This simulates real multi-turn use.

SYSTEM_PROMPT = "You are a coding assistant. Follow instructions precisely."

TASK_SEQUENCE = [
    "Create a Python class called TaskManager with an __init__ method that stores a list of tasks.",
    "Add a method add_task(title, priority) that adds a task dict to the list.",
    "Add a method get_tasks_by_priority(priority) that filters and returns matching tasks.",
    "Add a method complete_task(title) that marks a task as completed.",
    "Add a method get_statistics() that returns count of total, completed, and pending tasks.",
    "Now add due dates to tasks. Modify add_task to accept an optional due_date parameter.",
    "Add a method get_overdue_tasks() that returns tasks past their due date.",
    "Add a method export_to_json(filepath) that saves all tasks to a JSON file.",
    "Add a method import_from_json(filepath) that loads tasks from a JSON file.",
    "Add input validation to add_task: title must be non-empty string, priority must be 1-5.",
    "Add a method sort_tasks(by='priority', reverse=False) that sorts the task list.",
    "Add a method search_tasks(keyword) that searches task titles and returns matches.",
    "Refactor the class to use a dataclass for individual tasks instead of dicts.",
    "Add unit tests for add_task, complete_task, and get_statistics methods.",
    "Add error handling: raise custom TaskNotFoundError when completing non-existent task.",
    # Add more turns to push the model further...
    # You can extend this to 50+ turns by asking increasingly specific questions
    # about the code it's been generating.
    "Show me the complete, final version of the TaskManager class with all features.",
    "Now review your implementation. Are there any bugs? Fix them.",
    "Add type hints to every method in the class.",
    "Write a comprehensive docstring for the class and every public method.",
    "Create a CLI interface using argparse that wraps the TaskManager.",
]

def run_multiturn_experiment(model, tokenizer, turns=None):
    """
    Run a multi-turn conversation and track entropy at each turn.
    Returns: list of dicts with turn number, entropy stats, response length, etc.
    """
    if turns is None:
        turns = TASK_SEQUENCE
    
    conversation = SYSTEM_PROMPT + "\n"
    results = []
    
    for i, user_msg in enumerate(turns):
        # Build the full prompt (all history + new question)
        conversation += f"\nUser: {user_msg}\nAssistant: "
        
        # Generate response
        response = generate(model, tokenizer, prompt=conversation, max_tokens=300)
        
        # Compute entropy on the FULL conversation so far (including response)
        full_text = conversation + response
        logits = get_logits_for_sequence(model, tokenizer, full_text)
        entropy = compute_token_entropy(logits)
        
        # Track entropy of just the RESPONSE portion
        # (last N tokens, where N = number of tokens in response)
        response_tokens = len(tokenizer.encode(response))
        response_entropy = entropy[-response_tokens:]
        
        result = {
            "turn": i,
            "prompt": user_msg[:80],
            "response_length": response_tokens,
            "context_length": len(tokenizer.encode(full_text)),
            "response_entropy_mean": response_entropy.mean().item(),
            "response_entropy_max": response_entropy.max().item(),
            "response_entropy_min": response_entropy.min().item(),
        }
        results.append(result)
        
        print(f"Turn {i:3d} | ctx {result['context_length']:5d} tokens | "
              f"entropy mean={result['response_entropy_mean']:.4f} "
              f"max={result['response_entropy_max']:.4f}")
        
        # Append response to conversation for next turn
        conversation += response + "\n"
        
        # Safety: stop if we're close to the model's context limit
        if result['context_length'] > 28000:  # ~28K, leave buffer for 32K limit
            print(f"Stopping at turn {i}: approaching context limit")
            break
    
    return results

if __name__ == "__main__":
    model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    results = run_multiturn_experiment(model, tokenizer)
    
    # Save raw data
    with open("data/multiturn_entropy.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nCompleted {len(results)} turns. Data saved.")
```

**IMPORTANT:** Create `data/` directory first: `mkdir -p data`

**Manual test:**
- Script runs to completion (or stops at context limit)
- Entropy numbers are printed for each turn
- JSON file is saved with all data
- Look at the numbers: does entropy trend upward in later turns?

**Time:** 3-4 hours (model generation takes time across 20+ turns)
**Git:** `git commit -m "TASK-009: multi-turn entropy tracking experiment"`

---

### TASK-010: Repetition Detection (Second CHI Signal)
**Status:** ✅ DONE
**Do:** Create `engine/repetition.py`

Entropy alone isn't enough. A model can be "confident" but repetitive
(low entropy, but saying the same thing over and over). We need a second signal.

```python
def compute_repetition_score(text_a: str, text_b: str) -> float:
    """
    Compute Jaccard similarity between two text outputs.
    
    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    where A and B are sets of n-grams (e.g., trigrams).
    
    Returns: 0.0 (completely different) to 1.0 (identical)
    """
    # Step 1: Tokenize both texts into trigrams (3-word chunks)
    # Step 2: Compute intersection and union of trigram sets
    # Step 3: Return |intersection| / |union|
    pass

def compute_self_repetition(text: str, window_size: int = 100) -> float:
    """
    Check if the model is repeating itself WITHIN a single response.
    
    Split the text into windows and check Jaccard similarity
    between consecutive windows.
    
    High self-repetition = the model is stuck in a loop.
    Returns: average pairwise Jaccard between consecutive windows.
    """
    pass

def track_repetition_over_turns(responses: list[str]) -> list[float]:
    """
    Given a list of model responses (one per turn),
    compute Jaccard similarity between consecutive responses.
    
    Returns: list of similarity scores.
    Rising similarity = model converging to repetitive output.
    """
    pass
```

**Manual test:**
```python
# Identical texts should give 1.0
assert compute_repetition_score("hello world foo bar", "hello world foo bar") == 1.0

# Completely different texts should give 0.0 (or very low)
score = compute_repetition_score(
    "The quick brown fox jumps over the lazy dog",
    "Python is a programming language for data science"
)
print(f"Different texts: {score:.4f}")  # Should be near 0

# Slightly similar texts should give mid-range
score = compute_repetition_score(
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "def factorial(n): return 1 if n < 2 else n * factorial(n-1)"
)
print(f"Similar structure: {score:.4f}")  # Should be moderate (0.2-0.5)
```

**Time:** 1-2 hours
**Git:** `git commit -m "TASK-010: repetition detection (Jaccard similarity)"`

---

### TASK-011: Goal Drift Detection (Third CHI Signal)
**Status:** ✅ DONE
**Do:** Create `engine/goal_drift.py`

The model should stay focused on the original task. If the cosine similarity
between its output's hidden state and the original task's hidden state drops,
the model is drifting.

```python
from engine.hidden_states import extract_hidden_states
import mlx.core as mx

def compute_goal_drift(model, tokenizer, original_task: str, 
                       current_output: str, layer_idx: int = 16) -> float:
    """
    Measure how far the model's current output has drifted
    from the original task.
    
    Returns: cosine similarity (0-1). Lower = more drift.
    
    Uses hidden states from layer_idx, averaged over sequence length.
    """
    # Get hidden state for original task description
    _, task_states = extract_hidden_states(model, tokenizer, original_task,
                                           layer_idx=layer_idx, max_tokens=1)
    
    # Get hidden state for current model output
    _, output_states = extract_hidden_states(model, tokenizer, current_output,
                                              layer_idx=layer_idx, max_tokens=1)
    
    # Average over sequence length to get single vector per text
    task_vec = mx.mean(task_states, axis=1).squeeze()    # (3584,)
    output_vec = mx.mean(output_states, axis=1).squeeze() # (3584,)
    
    # Cosine similarity
    cos_sim = mx.sum(task_vec * output_vec) / (
        mx.sqrt(mx.sum(task_vec ** 2)) * mx.sqrt(mx.sum(output_vec ** 2))
    )
    mx.eval(cos_sim)
    return cos_sim.item()
```

**Manual test:**
```python
# On-task response should have high similarity to task
task = "Build a Python TaskManager class"
on_task = "class TaskManager:\n    def __init__(self):\n        self.tasks = []"
off_task = "The weather in Paris is beautiful this time of year."

score_on = compute_goal_drift(model, tokenizer, task, on_task)
score_off = compute_goal_drift(model, tokenizer, task, off_task)

print(f"On-task drift: {score_on:.4f}")   # Should be high (0.7+)
print(f"Off-task drift: {score_off:.4f}") # Should be lower
assert score_on > score_off, "On-task should be closer to original task"
```

**Time:** 1-2 hours (mostly reusing TASK-005 code)
**Git:** `git commit -m "TASK-011: goal drift detection via hidden state similarity"`

---

### TASK-012: Visualize Degradation (The Money Graph)
**Status:** ✅ DONE
**Do:** Create `scripts/visualize_degradation.py`

This is the graph that goes in your README, your blog post, your pitch.
It shows entropy, repetition, and goal drift over a multi-turn conversation.

```python
"""
Generate the "degradation graph" — the visual proof that models lose coherence
and that we can detect exactly when it happens.
"""
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend

def plot_degradation(data_path: str = "data/multiturn_entropy.json",
                     output_path: str = "data/degradation_graph.png"):
    
    with open(data_path) as f:
        results = json.load(f)
    
    turns = [r['turn'] for r in results]
    entropy_mean = [r['response_entropy_mean'] for r in results]
    entropy_max = [r['response_entropy_max'] for r in results]
    context_len = [r['context_length'] for r in results]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Top plot: Entropy over turns
    ax1 = axes[0]
    ax1.plot(turns, entropy_mean, 'b-', linewidth=2, label='Mean Entropy')
    ax1.fill_between(turns, entropy_mean, entropy_max, alpha=0.2, color='blue',
                     label='Max Entropy')
    ax1.set_ylabel('Response Entropy (bits)')
    ax1.set_title('Cognitive Degradation Over Multi-Turn Conversation\n'
                  'Qwen2.5-Coder-7B on Apple M4 Pro')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mark potential degradation point (where entropy starts trending up)
    # You'll refine this after seeing real data
    
    # Bottom plot: Context length over turns
    ax2 = axes[1]
    ax2.plot(turns, context_len, 'r-', linewidth=2)
    ax2.set_ylabel('Context Length (tokens)')
    ax2.set_xlabel('Turn Number')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    plot_degradation()
```

**IMPORTANT:** Install matplotlib first: `pip install matplotlib`

**Manual test:**
- Run TASK-009 first to generate the data
- Run this script to generate the graph
- Open the PNG and look at it — does entropy trend upward?
- If the model hit context limits early, the graph may be short — that's fine

**Time:** 1-2 hours
**Git:** `git commit -m "TASK-012: degradation visualization"`

---

### TASK-013: Enhanced Experiment with All Three Signals
**Status:** ⬜ TODO
**Do:** Create `scripts/full_degradation_experiment.py`

Combine all three signals (entropy + repetition + goal drift) into
a single experiment. This is the proto-CHI.

Run the same multi-turn conversation but track ALL signals:
- Entropy per turn (from TASK-008)
- Repetition between consecutive responses (from TASK-010)
- Goal drift from original task (from TASK-011)

Save all data. Generate a combined 3-panel graph.

This script should:
1. Load model
2. Define a multi-turn coding task (reuse TASK-009's sequence)
3. For each turn: generate response, compute entropy, compute repetition
   vs previous response, compute goal drift vs original task
4. Save all metrics to `data/full_degradation.json`
5. Generate `data/full_degradation_graph.png` with 3 subplots

**Manual test:**
- Graph has 3 panels: entropy, repetition, goal drift
- All three lines are visible and change over turns
- Look for correlations: does repetition rise when entropy does?
  Does goal drift increase as context fills up?

**Time:** 2-3 hours
**Git:** `git commit -m "TASK-013: full 3-signal degradation experiment"`

---

### TASK-014: Find the Inflection Point
**Status:** ⬜ TODO
**Do:** Create `scripts/find_inflection.py`

Analyze the data from TASK-013 to find the EXACT turn where degradation begins.

```python
"""
Find where cognitive degradation begins.
This is the turn number where intervention should start.
"""
import json
import numpy as np

def find_inflection(data_path: str = "data/full_degradation.json"):
    with open(data_path) as f:
        results = json.load(f)
    
    entropy = [r['response_entropy_mean'] for r in results]
    
    # Method 1: Moving average crossover
    # If short-term average exceeds long-term average, degradation has begun
    window_short = 3
    window_long = 7
    
    if len(entropy) < window_long:
        print("Not enough data points. Need at least 7 turns.")
        return
    
    short_ma = [sum(entropy[max(0,i-window_short):i+1]) / min(i+1, window_short)
                for i in range(len(entropy))]
    long_ma = [sum(entropy[max(0,i-window_long):i+1]) / min(i+1, window_long) 
               for i in range(len(entropy))]
    
    # Find first crossover after initial stabilization
    crossover_turn = None
    for i in range(window_long, len(entropy)):
        if short_ma[i] > long_ma[i] * 1.05:  # 5% threshold
            crossover_turn = i
            break
    
    # Method 2: Maximum rate of change
    if len(entropy) >= 3:
        diffs = [entropy[i+1] - entropy[i] for i in range(len(entropy)-1)]
        max_increase_turn = max(range(len(diffs)), key=lambda i: diffs[i])
    else:
        max_increase_turn = None
    
    print(f"Total turns analyzed: {len(entropy)}")
    print(f"Entropy range: {min(entropy):.4f} — {max(entropy):.4f}")
    print(f"Moving average crossover at turn: {crossover_turn}")
    print(f"Maximum entropy increase at turn: {max_increase_turn}")
    
    if crossover_turn:
        print(f"\n→ Degradation likely begins around turn {crossover_turn}")
        print(f"  This is where CHI would start triggering Level 2 intervention")
    else:
        print(f"\n→ No clear inflection found. Model may not have degraded")
        print(f"  within the number of turns tested. Try longer conversations.")

if __name__ == "__main__":
    find_inflection()
```

**Manual test:**
- Script identifies a turn number where degradation begins
- OR reports no inflection (which is also useful data — model was fine)
- Write the inflection turn in your SESSION_LOG — this number drives
  everything in Phase 1 (when to trigger interventions)

**Time:** 1-2 hours
**Git:** `git commit -m "TASK-014: inflection point detection"`

---

## WEEK 2 CHECKLIST
- [x] TASK-008: Token-level entropy computation (2-3 hrs)
- [x] TASK-009: Multi-turn entropy experiment (3-4 hrs)
- [x] TASK-010: Repetition detection — Jaccard (1-2 hrs)
- [x] TASK-011: Goal drift detection (1-2 hrs)
- [x] TASK-012: Degradation visualization (1-2 hrs)
- [ ] TASK-013: Full 3-signal experiment (2-3 hrs)
- [ ] TASK-014: Find inflection point (1-2 hrs)

---

## WEEK 3: CHI Score + CLI Skeleton

(Detailed tasks added after Week 2 is complete)