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
**Status:** ✅ DONE
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
**Status:** ✅ DONE
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
- [x] TASK-013: Full 3-signal experiment (2-3 hrs)
- [x] TASK-014: Find inflection point (1-2 hrs)

---

## WEEK 3: CHI Score + Project Skeleton + CLI
**Goal:** By end of Week 3 you have a working CLI tool that monitors
cognitive health in real-time as a model generates text. This is the
first thing someone can `pip install` and use.

---

### TASK-015: Implement CHI Score (Combining 3 Signals)
**Status:** ✅ DONE
**Do:** Create `engine/chi.py`

CHI (Cognitive Health Index) combines your three signals into
one number between 0 and 1. Higher = healthier.

```python
"""
CHI — Cognitive Health Index

Combines entropy, repetition, and goal drift into a single
score from 0.0 (completely degraded) to 1.0 (fully healthy).

The score is intentionally simple in v1. We can add more signals
and tune weights later. What matters now is that it WORKS and
DETECTS degradation.
"""
import mlx.core as mx
from engine.entropy import compute_token_entropy, get_logits_for_sequence
from engine.repetition import compute_repetition_score
from engine.goal_drift import compute_goal_drift

class CHIMonitor:
    """
    Tracks cognitive health across a conversation.
    Call update() after each model response to get the current CHI score.
    """
    
    def __init__(self, model, tokenizer, original_task: str,
                 layer_idx: int = 27):
        self.model = model
        self.tokenizer = tokenizer
        self.original_task = original_task
        self.layer_idx = layer_idx
        
        # History for tracking trends
        self.entropy_history = []
        self.repetition_history = []
        self.drift_history = []
        self.chi_history = []
        self.previous_response = ""
    
    def update(self, current_response: str, full_context: str = None) -> dict:
        """
        Compute CHI after a new model response.
        
        Args:
            current_response: The model's latest response text
            full_context: The full conversation so far (for entropy)
        
        Returns: dict with chi_score and individual signal values
        """
        # --- Signal 1: Entropy ---
        # Compute entropy on the response tokens
        # Normalize to 0-1 range: entropy of 0 bits = 1.0 (healthy),
        # entropy of 10+ bits = 0.0 (degraded)
        # YOUR IMPLEMENTATION
        
        # --- Signal 2: Repetition ---
        # Jaccard similarity vs previous response
        # 0.0 repetition = 1.0 (healthy), 1.0 repetition = 0.0 (degraded)
        # YOUR IMPLEMENTATION
        
        # --- Signal 3: Goal Drift ---
        # 0.0 drift = 1.0 (healthy), 1.0 drift = 0.0 (degraded)
        # YOUR IMPLEMENTATION
        
        # --- Combine into CHI ---
        # Simple weighted average for v1
        # Weights: entropy 0.4, repetition 0.3, goal_drift 0.3
        # You can tune these after seeing real data
        w_entropy = 0.4
        w_repetition = 0.3
        w_drift = 0.3
        
        # chi_score = w_entropy * entropy_health + w_repetition * rep_health + w_drift * drift_health
        # YOUR IMPLEMENTATION
        
        # Update history
        self.previous_response = current_response
        
        # Return detailed result
        return {
            "chi_score": chi_score,
            "entropy_raw": entropy_raw,
            "entropy_health": entropy_health,
            "repetition_raw": repetition_raw,
            "repetition_health": rep_health,
            "drift_raw": drift_raw,
            "drift_health": drift_health,
            "turn": len(self.chi_history),
        }
    
    def get_intervention_level(self) -> int:
        """
        Based on current CHI, return the intervention level.
        1 = normal, 2 = steer, 3 = compact, 4 = escalate
        """
        if not self.chi_history:
            return 1
        chi = self.chi_history[-1]
        if chi > 0.8:
            return 1
        elif chi > 0.5:
            return 2  # Would trigger EasySteer in Phase 2
        elif chi > 0.3:
            return 3  # Would trigger context compaction in Phase 1
        else:
            return 4  # Would trigger cloud escalation
```

**Manual test:**
```python
monitor = CHIMonitor(model, tokenizer, "Build a TaskManager class")

# Healthy response
r1 = monitor.update("class TaskManager:\n    def __init__(self):\n        self.tasks = []")
print(f"Turn 0 CHI: {r1['chi_score']:.3f} Level: {monitor.get_intervention_level()}")
# Should be high (0.7+), Level 1

# Simulate degraded response (repetitive, off-topic)
r2 = monitor.update("I think we should consider the weather today. The weather is nice.")
print(f"Turn 1 CHI: {r2['chi_score']:.3f} Level: {monitor.get_intervention_level()}")
# Should be lower, possibly Level 2 or 3
```

**Time:** 3-4 hours
**Git:** `git commit -m "TASK-015: CHI score combining 3 signals"`

---

### TASK-016: Validate CHI Against Your Week 2 Data
**Status:** ✅ DONE
**Do:** Create `scripts/validate_chi.py`

Run CHI retroactively on your 20-turn multi-turn data.
This validates that CHI actually detects the degradation
you already proved exists in Week 2.

```python
"""
Replay the 20-turn experiment through CHI and verify it detects
the degradation inflection point found in TASK-014.
"""
# Load the saved responses from full_degradation.json
# Create a CHIMonitor
# Feed each response through update()
# Plot CHI over turns alongside the raw signals
# Verify: CHI drops below 0.7 around the inflection turn
```

Generate a graph: `data/chi_validation.png`
- Top panel: CHI score over turns (with threshold lines at 0.8, 0.5, 0.3)
- Bottom panel: The three raw signals for comparison

**Manual test:**
- CHI starts high (>0.7) in early turns
- CHI drops in later turns, roughly around the inflection point from TASK-014
- Intervention levels printed make sense (Level 1 early, Level 2-3 later)

**Time:** 2-3 hours
**Git:** `git commit -m "TASK-016: CHI validated against real degradation data"`

---

### TASK-017: Project Structure Cleanup
**Status:** ✅ DONE
**Do:** Reorganize the project into a proper Python package.

```
cognition-engine/
├── .cofounder/           # Co-founder protocol files
│   ├── COFOUNDER_BRAIN.md
│   ├── SESSION_LOG.md
│   └── TASKS.md
├── cognition_engine/     # Main package (rename from engine/)
│   ├── __init__.py       # Version, package-level imports
│   ├── chi.py            # CHI monitor
│   ├── entropy.py        # Entropy computation
│   ├── hidden_states.py  # Hidden state extraction
│   ├── attention_weights.py  # Attention weight capture
│   ├── repetition.py     # Repetition detection
│   ├── goal_drift.py     # Goal drift measurement
│   └── inference.py      # Model loading + basic generation
├── scripts/              # Experiments and one-off scripts
│   ├── benchmark.py
│   ├── multiturn_entropy.py
│   ├── full_degradation_experiment.py
│   ├── find_inflection.py
│   ├── validate_chi.py
│   └── visualize_degradation.py
├── data/                 # Experiment outputs (gitignored except READMEs)
│   ├── .gitkeep
│   └── *.json / *.png
├── tests/                # Unit tests
│   ├── __init__.py
│   ├── test_entropy.py
│   └── test_repetition.py
├── pyproject.toml        # Package config
├── README.md             # THE readme (start drafting)
├── .gitignore
└── LICENSE               # MIT
```

Key changes:
- Rename `engine/` → `cognition_engine/` (Python convention: underscores)
- Update all imports across scripts
- Create `pyproject.toml` with basic metadata
- Create `.gitignore` (exclude data/*.json, data/*.png, .venv/, __pycache__/)
- Create `cognition_engine/__init__.py` with `__version__ = "0.1.0"`

**Manual test:** From project root, these should all work:
```bash
python -c "from cognition_engine.chi import CHIMonitor; print('OK')"
python -c "from cognition_engine.entropy import compute_token_entropy; print('OK')"
python scripts/full_degradation_experiment.py --max-turns 2
```

**Time:** 2-3 hours (mostly updating imports and testing)
**Git:** `git commit -m "TASK-017: restructure as proper Python package"`

---

### TASK-018: Basic CLI
**Status:** ✅ DONE
**Do:** Create `cognition_engine/cli.py`

A simple CLI that lets someone run Cognition Engine from the terminal.
v1 is minimal: load model, chat, show CHI score after each response.

```python
"""
cognition-engine CLI — v0.1

Usage:
    python -m cognition_engine.cli --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
    
    Then type your task. The model responds. CHI score shown after each turn.
"""
import argparse
from mlx_lm import load, generate
from cognition_engine.chi import CHIMonitor

def main():
    parser = argparse.ArgumentParser(description="Cognition Engine — AI that monitors its own brain")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
                        help="Model path or HuggingFace repo")
    parser.add_argument("--max-tokens", type=int, default=500,
                        help="Max tokens per response")
    args = parser.parse_args()
    
    print("Loading model...")
    model, tokenizer = load(args.model)
    print(f"Model loaded. Type your task. Ctrl+C to quit.\n")
    
    monitor = None
    conversation = ""
    
    while True:
        try:
            user_input = input("\033[94mYou:\033[0m ")  # Blue prompt
            if not user_input.strip():
                continue
            
            # Initialize CHI monitor on first input
            if monitor is None:
                monitor = CHIMonitor(model, tokenizer, user_input)
            
            # Build conversation
            conversation += f"\nUser: {user_input}\nAssistant: "
            
            # Generate
            response = generate(model, tokenizer, prompt=conversation,
                              max_tokens=args.max_tokens)
            conversation += response
            
            # Show response
            print(f"\033[92mAssistant:\033[0m {response}")  # Green
            
            # Compute CHI
            result = monitor.update(response, conversation)
            level = monitor.get_intervention_level()
            
            # Color-coded CHI display
            chi = result['chi_score']
            if chi > 0.8:
                color = "\033[92m"  # Green
            elif chi > 0.5:
                color = "\033[93m"  # Yellow
            else:
                color = "\033[91m"  # Red
            
            print(f"\n{color}[CHI: {chi:.3f} | Level {level}]\033[0m "
                  f"E={result['entropy_health']:.2f} "
                  f"R={result['repetition_health']:.2f} "
                  f"D={result['drift_health']:.2f}\n")
            
        except KeyboardInterrupt:
            print("\n\nSession ended.")
            if monitor and monitor.chi_history:
                print(f"Turns: {len(monitor.chi_history)}")
                print(f"Final CHI: {monitor.chi_history[-1]:.3f}")
                print(f"Min CHI: {min(monitor.chi_history):.3f}")
            break

if __name__ == "__main__":
    main()
```

Also create `cognition_engine/__main__.py`:
```python
from cognition_engine.cli import main
main()
```

**Manual test:**
```bash
python -m cognition_engine --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
```
- Model loads
- You can type a task
- Model responds
- CHI score shown in color (green/yellow/red) after each response
- Ctrl+C exits cleanly with summary stats

**Time:** 2-3 hours
**Git:** `git commit -m "TASK-018: basic CLI with real-time CHI display"`

---

### TASK-019: Write Basic Tests
**Status:** ✅ DONE
**Do:** Create `tests/test_entropy.py` and `tests/test_repetition.py`

These are quick sanity tests — NOT model-dependent (they run without loading a model).

```python
# tests/test_entropy.py
import mlx.core as mx
from cognition_engine.entropy import compute_token_entropy

def test_uniform_distribution_high_entropy():
    """Uniform distribution over vocab should give high entropy."""
    vocab_size = 1000
    logits = mx.zeros((1, 1, vocab_size))  # All equal = uniform after softmax
    entropy = compute_token_entropy(logits)
    mx.eval(entropy)
    # log2(1000) ≈ 9.97, so entropy should be close to ~10
    assert entropy.item() > 9.0, f"Uniform entropy should be high, got {entropy.item()}"

def test_peaked_distribution_low_entropy():
    """One dominant logit should give low entropy."""
    vocab_size = 1000
    logits = mx.zeros((1, 1, vocab_size))
    logits = logits.at[0, 0, 0].add(100.0)  # One huge logit
    entropy = compute_token_entropy(logits)
    mx.eval(entropy)
    assert entropy.item() < 1.0, f"Peaked entropy should be low, got {entropy.item()}"

def test_entropy_shape():
    """Entropy should have shape (seq_len,)."""
    logits = mx.zeros((1, 5, 100))
    entropy = compute_token_entropy(logits)
    mx.eval(entropy)
    assert entropy.shape == (5,), f"Expected (5,), got {entropy.shape}"
```

```python
# tests/test_repetition.py
from cognition_engine.repetition import compute_repetition_score, compute_self_repetition

def test_identical_texts():
    score = compute_repetition_score("the quick brown fox jumps", "the quick brown fox jumps")
    assert score == 1.0

def test_completely_different():
    score = compute_repetition_score(
        "the quick brown fox jumps over the lazy dog today",
        "python is a great programming language for beginners"
    )
    assert score < 0.1

def test_self_repetition_repeated_text():
    repeated = " ".join(["hello world foo bar"] * 20)
    score = compute_self_repetition(repeated, window_size=20)
    assert score > 0.8, f"Highly repeated text should score high, got {score}"
```

Run with: `python -m pytest tests/ -v`
(Install pytest first: `pip install pytest`)

**Manual test:** All tests pass.
**Time:** 1-2 hours
**Git:** `git commit -m "TASK-019: basic unit tests for entropy and repetition"`

---

### TASK-020: Draft README
**Status:** ✅ DONE
**Do:** Create `README.md` in project root.

This is your marketing page. Structure:

1. **One-liner + badge** — "A local AI coding agent that monitors its own brain"
2. **The graph** — your degradation_graph.png showing where the model breaks
3. **What it does** — 3 bullet points max
4. **Quick start** — 4 lines to install and run
5. **How it works** — brief explanation of CHI (link to docs for detail)
6. **Why this exists** — 1 paragraph on the problem (39% multi-turn degradation)
7. **Roadmap** — what's coming (activation steering, LoRA)
8. **License** — MIT

Keep it SHORT. People decide in 10 seconds whether to star a repo.
The graph is the hook. The quick start is the conversion.

Put your best degradation graph in the repo root or `assets/` folder
and reference it in the README.

**Manual test:** Read it as if you've never heard of Cognition Engine.
Does it make you want to try it in under 30 seconds?

**Time:** 2-3 hours (writing good README is hard — take time on this)
**Git:** `git commit -m "TASK-020: README with degradation graph and quick start"`

---

## WEEK 4: Context Compaction + Agent Loop
**Goal:** Build a coding agent that can read/write files, execute commands,
and automatically manage its own context when CHI drops. This is the
Level 3 intervention (context compaction) and the agent skeleton that
all future features build on.

---

### TASK-021: Context Compaction Engine
**Status:** ✅ DONE
**Do:** Create `cognition_engine/compaction.py`

When context gets too long or CHI drops below 0.5, we need to compress
the conversation without losing critical information. This is Level 3
of the intervention ladder.

Three compaction strategies, applied in order of severity:

```python
"""
Context Compaction — Level 3 Intervention

When CHI drops or context is approaching the model's limit,
compress the conversation to restore coherence.
"""
import mlx.core as mx
from mlx_lm import load, generate

class ContextCompactor:
    """Manages conversation context with automatic compaction."""
    
    def __init__(self, model, tokenizer, max_context_tokens: int = 24000):
        self.model = model
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens  # ~75% of 32K limit
        self.conversation_turns = []  # list of {"role": "user"|"assistant", "content": str}
        self.compaction_count = 0
        self.anchor_text = ""  # Original task description, always kept
    
    def set_anchor(self, task_description: str):
        """Set the anchor text — always preserved across compactions."""
        self.anchor_text = task_description
    
    def add_turn(self, role: str, content: str):
        """Add a conversation turn."""
        self.conversation_turns.append({"role": role, "content": content})
    
    def get_context_tokens(self) -> int:
        """Count total tokens in current context."""
        full_text = self._build_context()
        return len(self.tokenizer.encode(full_text))
    
    def needs_compaction(self, chi_score: float = None) -> bool:
        """Check if compaction is needed."""
        token_count = self.get_context_tokens()
        # Compact if context is too long OR CHI is critically low
        if token_count > self.max_context_tokens:
            return True
        if chi_score is not None and chi_score < 0.5:
            return True
        return False
    
    def compact(self, strategy: str = "auto") -> str:
        """
        Compact the context. Returns the new context string.
        
        Strategies:
        - "trim": Drop oldest turns, keep anchor + last N turns
        - "summarize": Use the model to summarize older turns
        - "anchor_reset": Nuclear option — keep only anchor + last 2 turns
        """
        # YOUR IMPLEMENTATION FOR EACH STRATEGY
        pass
    
    def _build_context(self) -> str:
        """Build the full context string from anchor + turns."""
        parts = []
        if self.anchor_text:
            parts.append(f"[TASK]: {self.anchor_text}\n")
        for turn in self.conversation_turns:
            prefix = "User" if turn["role"] == "user" else "Assistant"
            parts.append(f"{prefix}: {turn['content']}")
        return "\n".join(parts)
    
    def _trim(self, keep_last: int = 5) -> str:
        """Keep anchor + last N turns. Fast, no model call."""
        # Keep the anchor text
        # Keep the last `keep_last` turns
        # Drop everything in between
        # Return rebuilt context
        pass
    
    def _summarize(self) -> str:
        """
        Summarize older turns into a brief recap, keep recent turns verbatim.
        
        Split conversation into:
        - Old turns (to be summarized)
        - Recent turns (keep as-is, last 5)
        
        Use the model itself to generate a summary of old turns.
        Inject summary as a "[CONTEXT SUMMARY]" block at the start.
        """
        # Split turns: old (to summarize) and recent (to keep)
        # Build a summarization prompt
        # Generate summary using the model
        # Rebuild context: anchor + summary + recent turns
        pass
    
    def _anchor_reset(self) -> str:
        """
        Nuclear option: wipe everything except anchor + last 2 turns.
        Used when CHI is critically low and summarization isn't enough.
        """
        pass
```

**Manual test:**
```python
# Simulate a conversation that fills up context
compactor = ContextCompactor(model, tokenizer, max_context_tokens=2000)
compactor.set_anchor("Build a TaskManager class in Python")

# Add 20 turns of fake conversation
for i in range(20):
    compactor.add_turn("user", f"Step {i}: Add feature number {i} " * 20)
    compactor.add_turn("assistant", f"Here is the implementation for step {i} " * 20)

print(f"Before compaction: {compactor.get_context_tokens()} tokens")
assert compactor.needs_compaction()

# Test trim
compactor.compact("trim")
print(f"After trim: {compactor.get_context_tokens()} tokens")
assert compactor.get_context_tokens() < 2000

# Test that anchor is preserved
context = compactor._build_context()
assert "TaskManager" in context
```

**Time:** 3-4 hours
**Git:** `git commit -m "TASK-021: context compaction engine (trim, summarize, anchor reset)"`

---

### TASK-022: SCAN-Style Anchor Tokens
**Status:** ✅ DONE
**Do:** Create `cognition_engine/anchors.py`

SCAN (System Compliance Awareness Notes) are periodic reminders
injected into the context to keep the model focused. Three levels:

```python
"""
SCAN Anchor Tokens — Periodic reminders to maintain focus.

Injected at different frequencies based on CHI:
- FULL (300 tokens): Complete task restatement + key decisions
- MINI (120 tokens): Brief task reminder + current step
- ANCHOR (20 tokens): One-line "Stay focused on: {task}"
"""

class AnchorManager:
    """Generates and injects anchor tokens based on CHI level."""
    
    def __init__(self, task_description: str, key_decisions: list[str] = None):
        self.task_description = task_description
        self.key_decisions = key_decisions or []
        self.files_modified = []  # Track what files the agent has touched
    
    def generate_anchor(self, level: str = "ANCHOR") -> str:
        """
        Generate an anchor token string.
        
        Levels:
        - "FULL": ~300 tokens. Complete task context restoration.
        - "MINI": ~120 tokens. Brief reminder.
        - "ANCHOR": ~20 tokens. One-liner.
        """
        if level == "FULL":
            return self._full_anchor()
        elif level == "MINI":
            return self._mini_anchor()
        else:
            return self._anchor()
    
    def _full_anchor(self) -> str:
        parts = [
            f"[SYSTEM REMINDER — FULL CONTEXT RESTORATION]",
            f"Original task: {self.task_description}",
        ]
        if self.key_decisions:
            parts.append("Key decisions so far:")
            for d in self.key_decisions[-5:]:  # Last 5 decisions
                parts.append(f"  - {d}")
        if self.files_modified:
            parts.append(f"Files modified: {', '.join(self.files_modified[-10:])}")
        parts.append("[END REMINDER — Continue with the task above.]")
        return "\n".join(parts)
    
    def _mini_anchor(self) -> str:
        return (f"[REMINDER] Task: {self.task_description[:200]}. "
                f"Decisions: {len(self.key_decisions)}. "
                f"Files: {len(self.files_modified)}. Stay focused.")
    
    def _anchor(self) -> str:
        return f"[FOCUS] {self.task_description[:100]}"
    
    def add_decision(self, decision: str):
        """Track a key decision made during the session."""
        self.key_decisions.append(decision)
    
    def add_file(self, filepath: str):
        """Track a file the agent has modified."""
        if filepath not in self.files_modified:
            self.files_modified.append(filepath)
    
    def get_anchor_for_chi(self, chi_score: float) -> str | None:
        """
        Based on CHI score, return the appropriate anchor level.
        Returns None if CHI is healthy (no anchor needed).
        """
        if chi_score > 0.8:
            return None  # Healthy, no anchor needed
        elif chi_score > 0.6:
            return self.generate_anchor("ANCHOR")  # 20 tokens
        elif chi_score > 0.4:
            return self.generate_anchor("MINI")    # 120 tokens  
        else:
            return self.generate_anchor("FULL")    # 300 tokens
```

**Manual test:**
```python
anchor_mgr = AnchorManager("Build a REST API with Flask")
anchor_mgr.add_decision("Use SQLite for database")
anchor_mgr.add_decision("JWT for authentication")
anchor_mgr.add_file("app.py")
anchor_mgr.add_file("models.py")

# Test each level
print(anchor_mgr.generate_anchor("FULL"))   # Should include task + decisions + files
print(anchor_mgr.generate_anchor("MINI"))   # Should be brief
print(anchor_mgr.generate_anchor("ANCHOR")) # Should be one line

# Test CHI-based selection
assert anchor_mgr.get_anchor_for_chi(0.9) is None       # Healthy
assert "FOCUS" in anchor_mgr.get_anchor_for_chi(0.7)     # Light anchor
assert "REMINDER" in anchor_mgr.get_anchor_for_chi(0.5)  # Mini
assert "FULL CONTEXT" in anchor_mgr.get_anchor_for_chi(0.3)  # Full
```

**Time:** 1-2 hours
**Git:** `git commit -m "TASK-022: SCAN-style anchor tokens (FULL/MINI/ANCHOR)"`

---

### TASK-023: Tool-Calling Agent Loop
**Status:** ✅ DONE
**Do:** Create `cognition_engine/agent.py`

The agent needs to be able to: read files, write/edit files, 
execute shell commands (with user confirmation), and think step-by-step.

This is the core loop that turns the chat model into a coding agent.

```python
"""
Agent Loop — The core execution engine.

The agent receives a task, plans steps, executes them using tools,
and monitors its own cognitive health throughout.
"""
import os
import subprocess
from cognition_engine.chi import CHIMonitor
from cognition_engine.compaction import ContextCompactor
from cognition_engine.anchors import AnchorManager
from mlx_lm import load, generate

class CognitionAgent:
    """
    A coding agent with cognitive health monitoring.
    
    Tools available:
    - read_file(path) — read file contents
    - write_file(path, content) — write/overwrite a file
    - edit_file(path, old_text, new_text) — find and replace in a file
    - execute(command) — run a shell command (requires user approval)
    - think(thought) — record a reasoning step (no action)
    """
    
    def __init__(self, model, tokenizer, task: str,
                 working_dir: str = ".", auto_approve: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.task = task
        self.working_dir = working_dir
        self.auto_approve = auto_approve
        
        # Core systems
        self.chi_monitor = CHIMonitor(model, tokenizer, task)
        self.compactor = ContextCompactor(model, tokenizer)
        self.anchor_mgr = AnchorManager(task)
        
        self.compactor.set_anchor(task)
        
        # Tool call history
        self.tool_history = []
        self.turn_count = 0
    
    def run(self, max_turns: int = 50):
        """
        Main agent loop.
        
        Each turn:
        1. Build context (with anchors if CHI is low)
        2. Generate model response
        3. Parse tool calls from response
        4. Execute tool calls
        5. Update CHI
        6. Check if compaction needed
        7. Repeat until task done or max_turns
        """
        print(f"Starting agent on task: {self.task}")
        print(f"Working directory: {os.path.abspath(self.working_dir)}\n")
        
        while self.turn_count < max_turns:
            # 1. Check if anchor needed
            if self.chi_monitor.chi_history:
                chi = self.chi_monitor.chi_history[-1]
                anchor = self.anchor_mgr.get_anchor_for_chi(chi)
                if anchor:
                    self.compactor.add_turn("system", anchor)
            
            # 2. Build context and generate
            context = self.compactor._build_context()
            
            # Add generation prompt
            context += "\nAssistant: "
            response = generate(self.model, self.tokenizer,
                              prompt=context, max_tokens=800)
            
            # 3. Parse and execute tool calls
            tool_calls = self._parse_tool_calls(response)
            tool_results = []
            
            for tool_call in tool_calls:
                result = self._execute_tool(tool_call)
                tool_results.append(result)
            
            # 4. Add to conversation
            self.compactor.add_turn("assistant", response)
            if tool_results:
                results_text = "\n".join(tool_results)
                self.compactor.add_turn("system", f"[Tool Results]\n{results_text}")
            
            # 5. Update CHI
            chi_result = self.chi_monitor.update(response)
            level = self.chi_monitor.get_intervention_level()
            
            # 6. Display status
            chi = chi_result['chi_score']
            self._print_status(chi, level, tool_calls)
            
            # 7. Check compaction
            if self.compactor.needs_compaction(chi):
                if chi < 0.3:
                    self.compactor.compact("anchor_reset")
                    print("  [COMPACTION: anchor reset]")
                elif chi < 0.5:
                    self.compactor.compact("summarize")
                    print("  [COMPACTION: summarize]")
                else:
                    self.compactor.compact("trim")
                    print("  [COMPACTION: trim]")
            
            # 8. Check if model says it's done
            if self._is_task_complete(response):
                print("\n Agent reports task complete.")
                break
            
            self.turn_count += 1
        
        self._print_summary()
    
    def _parse_tool_calls(self, response: str) -> list[dict]:
        """
        Parse tool calls from the model's response.
        
        Expected format (you decide the exact format):
        <tool>read_file</tool>
        <path>src/main.py</path>
        
        OR simpler:
        [READ] src/main.py
        [WRITE] src/main.py
        ```content here```
        [EXEC] python -m pytest tests/
        [THINK] I need to check the database schema first.
        
        Start with a simple format. You can make it more sophisticated later.
        """
        # YOUR IMPLEMENTATION
        # Parse the response text for tool call patterns
        # Return list of {"tool": "read"|"write"|"edit"|"execute"|"think", ...}
        pass
    
    def _execute_tool(self, tool_call: dict) -> str:
        """Execute a single tool call and return the result."""
        tool = tool_call.get("tool")
        
        if tool == "read":
            return self._tool_read(tool_call["path"])
        elif tool == "write":
            return self._tool_write(tool_call["path"], tool_call["content"])
        elif tool == "edit":
            return self._tool_edit(tool_call["path"], 
                                   tool_call["old"], tool_call["new"])
        elif tool == "execute":
            return self._tool_execute(tool_call["command"])
        elif tool == "think":
            return f"[Thought recorded: {tool_call['thought'][:100]}]"
        else:
            return f"[Unknown tool: {tool}]"
    
    def _tool_read(self, path: str) -> str:
        """Read a file and return its contents."""
        full_path = os.path.join(self.working_dir, path)
        try:
            with open(full_path) as f:
                content = f.read()
            self.anchor_mgr.add_file(path)
            return f"[READ {path}]\n{content[:2000]}"  # Truncate large files
        except Exception as e:
            return f"[ERROR reading {path}: {e}]"
    
    def _tool_write(self, path: str, content: str) -> str:
        """Write content to a file."""
        full_path = os.path.join(self.working_dir, path)
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)
            self.anchor_mgr.add_file(path)
            return f"[WROTE {path} ({len(content)} chars)]"
        except Exception as e:
            return f"[ERROR writing {path}: {e}]"
    
    def _tool_edit(self, path: str, old_text: str, new_text: str) -> str:
        """Find and replace in a file."""
        full_path = os.path.join(self.working_dir, path)
        try:
            with open(full_path) as f:
                content = f.read()
            if old_text not in content:
                return f"[ERROR: '{old_text[:50]}...' not found in {path}]"
            content = content.replace(old_text, new_text, 1)
            with open(full_path, 'w') as f:
                f.write(content)
            self.anchor_mgr.add_file(path)
            return f"[EDITED {path}]"
        except Exception as e:
            return f"[ERROR editing {path}: {e}]"
    
    def _tool_execute(self, command: str) -> str:
        """Execute a shell command with user approval."""
        if not self.auto_approve:
            approval = input(f"  Execute: `{command}`? [y/N] ")
            if approval.lower() != 'y':
                return "[EXEC cancelled by user]"
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                cwd=self.working_dir, timeout=30
            )
            output = result.stdout[:1000]
            if result.returncode != 0:
                output += f"\n[STDERR: {result.stderr[:500]}]"
            return f"[EXEC `{command}` → exit {result.returncode}]\n{output}"
        except subprocess.TimeoutExpired:
            return f"[EXEC `{command}` → TIMEOUT (30s)]"
        except Exception as e:
            return f"[EXEC ERROR: {e}]"
    
    def _is_task_complete(self, response: str) -> bool:
        """Check if the model indicates the task is done."""
        markers = ["[DONE]", "[TASK COMPLETE]", "task is complete",
                   "all requirements have been met"]
        return any(m.lower() in response.lower() for m in markers)
    
    def _print_status(self, chi: float, level: int, tool_calls: list):
        """Print turn status."""
        color = "\033[92m" if chi > 0.8 else "\033[93m" if chi > 0.5 else "\033[91m"
        tools_str = ", ".join(tc.get("tool", "?") for tc in tool_calls) if tool_calls else "none"
        print(f"  Turn {self.turn_count:3d} | {color}CHI {chi:.3f} L{level}\033[0m | "
              f"Tools: {tools_str}")
    
    def _print_summary(self):
        """Print session summary."""
        print(f"\n--- Session Summary ---")
        print(f"Turns: {self.turn_count}")
        print(f"Compactions: {self.compactor.compaction_count}")
        print(f"Files modified: {self.anchor_mgr.files_modified}")
        if self.chi_monitor.chi_history:
            print(f"CHI: min={min(self.chi_monitor.chi_history):.3f} "
                  f"final={self.chi_monitor.chi_history[-1]:.3f}")
```

**Manual test:**
You won't test the full agent loop end-to-end yet (model output format
needs tuning). Instead, test each tool individually:

```python
agent = CognitionAgent(model, tokenizer, "test task", working_dir="/tmp/test_agent")
os.makedirs("/tmp/test_agent", exist_ok=True)

# Test write
print(agent._tool_write("test.py", "print('hello')"))

# Test read
print(agent._tool_read("test.py"))

# Test edit
print(agent._tool_edit("test.py", "hello", "world"))

# Test read again to verify edit
print(agent._tool_read("test.py"))

# Test execute
print(agent._tool_execute("python /tmp/test_agent/test.py"))
```

**Time:** 4-5 hours
**Git:** `git commit -m "TASK-023: tool-calling agent loop with CHI integration"`

---

### TASK-024: Tool-Call Prompt Format
**Status:** ✅ DONE
**Do:** Create `cognition_engine/prompts.py`

The model needs to know HOW to format tool calls. This is the system
prompt that teaches it the tool-calling protocol.

```python
"""
Prompt templates for the Cognition Engine agent.
"""

SYSTEM_PROMPT = """You are Cognition Engine, an AI coding assistant running locally.
You have access to these tools:

[READ] path — Read a file
[WRITE] path
```
file content here
```
[EDIT] path
```old
text to find
```new
replacement text
```
[EXEC] command — Run a shell command
[THINK] your reasoning — Record a thought (no action taken)
[DONE] — Signal that the task is complete

Rules:
- Use [THINK] before complex decisions
- Use [READ] before editing a file you haven't seen
- One tool call per response. Wait for the result before the next action.
- When the task is fully complete and tested, say [DONE]
- Be precise with file paths. Use relative paths from the project root.
"""

def build_system_prompt(task: str, project_context: str = None) -> str:
    """Build the full system prompt for a task."""
    prompt = SYSTEM_PROMPT + f"\n\nYour current task: {task}\n"
    if project_context:
        prompt += f"\nProject context:\n{project_context}\n"
    return prompt

def parse_tool_call(response: str) -> dict | None:
    """
    Parse a single tool call from model response.
    Returns dict with tool type and parameters, or None if no tool call found.
    """
    response = response.strip()
    
    # [READ] path
    if response.startswith("[READ]"):
        path = response[6:].strip().split("\n")[0].strip()
        return {"tool": "read", "path": path}
    
    # [WRITE] path\n```\ncontent\n```
    if response.startswith("[WRITE]"):
        lines = response.split("\n")
        path = lines[0][7:].strip()
        # Extract content between ``` markers
        content = _extract_code_block(response)
        if content is not None:
            return {"tool": "write", "path": path, "content": content}
    
    # [EDIT] path\n```old\n...\n```new\n...\n```
    if response.startswith("[EDIT]"):
        lines = response.split("\n")
        path = lines[0][6:].strip()
        old_text, new_text = _extract_edit_blocks(response)
        if old_text is not None:
            return {"tool": "edit", "path": path, "old": old_text, "new": new_text}
    
    # [EXEC] command
    if response.startswith("[EXEC]"):
        command = response[6:].strip().split("\n")[0].strip()
        return {"tool": "execute", "command": command}
    
    # [THINK] thought
    if response.startswith("[THINK]"):
        thought = response[7:].strip()
        return {"tool": "think", "thought": thought}
    
    # [DONE]
    if "[DONE]" in response:
        return {"tool": "done"}
    
    return None

def _extract_code_block(text: str) -> str | None:
    """Extract content from first ``` code block."""
    # YOUR IMPLEMENTATION
    pass

def _extract_edit_blocks(text: str) -> tuple[str | None, str | None]:
    """Extract old and new text from ```old and ```new blocks."""
    # YOUR IMPLEMENTATION
    pass
```

**Manual test:**
```python
from cognition_engine.prompts import parse_tool_call

assert parse_tool_call("[READ] src/main.py")["path"] == "src/main.py"
assert parse_tool_call("[EXEC] python -m pytest")["command"] == "python -m pytest"
assert parse_tool_call("[THINK] I need to check the schema")["thought"].startswith("I need")
assert parse_tool_call("[DONE]")["tool"] == "done"

# Test WRITE parsing
write_resp = '[WRITE] test.py\n```\nprint("hello")\n```'
parsed = parse_tool_call(write_resp)
assert parsed["tool"] == "write"
assert parsed["path"] == "test.py"
assert "hello" in parsed["content"]
```

**Time:** 2-3 hours
**Git:** `git commit -m "TASK-024: tool-call prompt format and parser"`

---

### TASK-025: Integration — Agent + CHI + Compaction Working Together
**Status:** ✅ DONE
**Do:** Create `scripts/test_agent.py`

Wire everything together and test the agent on a real (small) coding task.
This is an integration test — does the full system work end-to-end?

```python
"""
Integration test: Run the agent on a real coding task.
Verify CHI monitors, anchors inject, and compaction triggers.
"""
from mlx_lm import load
from cognition_engine.agent import CognitionAgent
import os
import shutil

# Set up a clean test directory
TEST_DIR = "/tmp/cognition_test"
if os.path.exists(TEST_DIR):
    shutil.rmtree(TEST_DIR)
os.makedirs(TEST_DIR)

# Load model
model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")

# Create agent with a simple task
agent = CognitionAgent(
    model, tokenizer,
    task="Create a Python file called calculator.py with a Calculator class "
         "that has add, subtract, multiply, and divide methods. "
         "Then create test_calculator.py with tests for each method. "
         "Run the tests to verify they pass.",
    working_dir=TEST_DIR,
    auto_approve=False  # You'll manually approve shell commands
)

# Run the agent (limit to 15 turns for testing)
agent.run(max_turns=15)

# Verify the agent created files
assert os.path.exists(os.path.join(TEST_DIR, "calculator.py")), "calculator.py not created"
print("\nFiles in test directory:")
for f in os.listdir(TEST_DIR):
    print(f"  {f}")
```

**Expected behavior:**
- Agent uses [THINK] to plan
- Agent uses [WRITE] to create calculator.py
- Agent uses [WRITE] to create test_calculator.py
- Agent uses [EXEC] to run pytest (you approve)
- CHI is displayed each turn
- If CHI drops, anchors are injected
- Agent says [DONE] when tests pass

**Reality check:** The 7B model may not follow the tool format perfectly.
That's fine — you'll see what works and what doesn't. Log the issues
in your SESSION_LOG so we can fix the prompt format in TASK-024.

**Time:** 3-4 hours (includes debugging model output parsing)
**Git:** `git commit -m "TASK-025: integration test — agent + CHI + compaction"`

---

### TASK-026: Working Memory (.cognition/ directory)
**Status:** ✅ DONE
**Do:** Create `cognition_engine/memory.py`

The agent should automatically save notes about the project it's
working on. These notes persist between sessions and get loaded
at the start of the next session.

```python
"""
Working Memory — Zettelkasten-style project notes.

The agent automatically writes notes as it works:
- Architectural decisions
- File purposes
- Key patterns used
- Bugs encountered and fixed

Notes are stored in .cognition/ in the project directory.
"""
import os
import json
from datetime import datetime

class WorkingMemory:
    """Manages persistent project memory in .cognition/ directory."""
    
    def __init__(self, project_dir: str):
        self.memory_dir = os.path.join(project_dir, ".cognition")
        os.makedirs(self.memory_dir, exist_ok=True)
        self.notes_file = os.path.join(self.memory_dir, "notes.json")
        self.notes = self._load_notes()
    
    def _load_notes(self) -> list[dict]:
        """Load existing notes from disk."""
        if os.path.exists(self.notes_file):
            with open(self.notes_file) as f:
                return json.load(f)
        return []
    
    def add_note(self, category: str, content: str, related_files: list[str] = None):
        """
        Add a note to working memory.
        
        Categories: "decision", "pattern", "bug", "file_purpose", "context"
        """
        note = {
            "id": len(self.notes),
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "content": content,
            "related_files": related_files or [],
        }
        self.notes.append(note)
        self._save()
        return note
    
    def get_context_for_task(self, task_description: str, max_notes: int = 10) -> str:
        """
        Get relevant notes for a task. Returns formatted string
        that can be injected into the agent's context.
        
        For v1: just return the last N notes.
        For v2 (later): use embeddings to find relevant notes.
        """
        recent = self.notes[-max_notes:]
        if not recent:
            return "[No previous notes for this project]"
        
        lines = ["[PROJECT MEMORY — Previous session notes]"]
        for note in recent:
            lines.append(f"  [{note['category']}] {note['content'][:200]}")
        lines.append("[END PROJECT MEMORY]\n")
        return "\n".join(lines)
    
    def get_session_summary(self) -> dict:
        """Get a summary of the current memory state."""
        categories = {}
        for note in self.notes:
            cat = note["category"]
            categories[cat] = categories.get(cat, 0) + 1
        return {
            "total_notes": len(self.notes),
            "categories": categories,
            "files_referenced": list(set(
                f for n in self.notes for f in n.get("related_files", [])
            )),
        }
    
    def _save(self):
        """Save notes to disk."""
        with open(self.notes_file, 'w') as f:
            json.dump(self.notes, f, indent=2)
```

**Manual test:**
```python
memory = WorkingMemory("/tmp/test_project")

memory.add_note("decision", "Using SQLite instead of PostgreSQL for simplicity",
                related_files=["db.py"])
memory.add_note("pattern", "All API routes follow /api/v1/{resource} convention",
                related_files=["routes.py", "app.py"])
memory.add_note("bug", "Fixed: forgot to close DB connection in get_users()",
                related_files=["db.py"])

context = memory.get_context_for_task("Add a new API endpoint")
print(context)
assert "SQLite" in context
assert "API routes" in context

summary = memory.get_session_summary()
print(summary)
assert summary["total_notes"] == 3

# Verify persistence — create new instance, should load existing notes
memory2 = WorkingMemory("/tmp/test_project")
assert len(memory2.notes) == 3
```

**Time:** 1-2 hours
**Git:** `git commit -m "TASK-026: working memory (.cognition/ directory)"`

---

## WEEK 4 CHECKLIST
- [x] TASK-021: Context compaction engine (3-4 hrs)
- [x] TASK-022: SCAN-style anchor tokens (1-2 hrs)
- [x] TASK-023: Tool-calling agent loop (4-5 hrs)
- [x] TASK-024: Tool-call prompt format + parser (2-3 hrs)
- [x] TASK-025: Integration test (3-4 hrs)
- [x] TASK-026: Working memory (1-2 hrs)

**Total: ~15-20 hours across 4-5 days**

**Week 4 milestone:** You have a working coding agent that:
1. Takes a task and executes it using tools (read/write/exec)
2. Monitors CHI in real-time
3. Injects anchor tokens when CHI drops
4. Compacts context when it gets too long
5. Saves notes to .cognition/ for session persistence

This is the foundation that activation steering (Phase 2) plugs into.

---

## NOTES

### Don't over-polish the agent loop yet.
The 7B model WILL produce messy tool calls. It will sometimes
forget the format, mix plain text with tool calls, or hallucinate
file paths. That's expected. Log the issues, move on. The agent
loop will get much better when we:
1. Switch to Qwen3.5-9B (better instruction following)
2. Add activation steering to keep it focused

### The tool format is intentionally simple.
XML-style (<tool>) or JSON tool calls would be cleaner but
local 7B models struggle with structured output. The bracket
format [READ] [WRITE] [EXEC] is easier for small models to learn.
You can upgrade to structured output when model quality improves.

### Context compaction is Level 3. We're building it now because
the agent needs it to function, but the real magic (Level 2:
activation steering) comes in Phase 2. Think of compaction as
the safety net that catches what steering doesn't fix.