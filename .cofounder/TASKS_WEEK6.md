# WEEK 6 TASKS: Activation Steering Foundations
# Phase 2 begins. This is what you came here to build.
# Everything so far was scaffolding. This is the moat.

---

## WEEK 6: Steering Vector Computation + First Injection
**Goal:** Compute your first steering vector on MLX and prove it changes
model behavior. No framework dependencies — pure MLX implementation.

**Why this matters:** EasySteer runs on vLLM (CUDA/GPU). There is NO
activation steering framework for MLX/Apple Silicon. You're building
the first one. That's the product.

**Key references to read BEFORE starting code:**
- ITI paper Section 3: https://arxiv.org/abs/2306.03341
- ActAdd (Activation Addition): https://openreview.net/forum?id=2XBPdPIcFK
- CAST (Conditional Activation Steering) — ICLR 2025 spotlight
- Field guide (practical): https://subhadipmitra.com/blog/2026/activation-steering-field-guide/

---

### TASK-033: Study Steering Math + Architecture Plan
**Status:** ⬜ TODO
**Do:** Create `docs/steering_design.md`

Read ITI Section 3 and the ActAdd paper. Write down the math in your
own words. This is NOT optional — if you can't explain it, you can't
debug it.

**Document these in steering_design.md:**

1. **The core algorithm (3 steps):**
   - Step 1: Generate contrastive pairs (positive/negative behavior)
   - Step 2: Extract hidden states from both, compute mean difference
   - Step 3: During inference, add scaled vector to target layer(s)

2. **Key decisions for our implementation:**
   - Which layers to steer? (ITI: middle layers work best, ~layer 14-20 for 28-layer model)
   - Steering strength (alpha)? Start with alpha=1.0, tune from there
   - Apply at which token positions? (All tokens vs last token vs specific positions)
   - One vector or per-layer vectors? (Start with single best layer, expand later)

3. **Our steering targets (what we're fixing):**
   - "Stay on task" — anti-drift steering (directly maps to goal drift CHI signal)
   - "Follow tool format" — anti-hallucination, format compliance
   - "Generate coherent code" — anti-repetition, anti-degeneration

4. **Architecture sketch:**
   ```
   CaptureWrapper (existing) → SteeringWrapper (new)
   
   SteeringWrapper:
     - Wraps a transformer layer
     - On forward pass: runs layer, ADDS steering vector to output
     - Steering vector loaded from precomputed .npz file
     - Alpha (strength) controlled by CHI score
   ```

5. **How this connects to the intervention ladder:**
   ```
   Level 1 (CHI > 0.8): No steering — normal inference
   Level 2 (CHI < 0.7): Activate steering vectors — milliseconds latency
   Level 3 (CHI < 0.5): Context compaction + SCAN anchors — 2-3 seconds
   Level 4 (CHI < 0.3): Cloud escalation — API latency
   ```

**What to get from the field guide blog post:**
- Steering works best for: sentiment, formality, refusal, style
- Steering works poorly for: factual recall, complex reasoning
- Our use case (focus/coherence/format compliance) is in the "works" category
- Strength degrades over long generation — may need adaptive alpha
- CAST's conditional steering = our CHI-triggered steering (same idea!)

**Time:** 2-3 hours (reading + writing)
**Git:** `git commit -m "TASK-033: steering design document and architecture plan"`

---

### TASK-034: Contrastive Pair Generator
**Status:** ⬜ TODO
**Do:** Create `cognition_engine/steering/contrastive_pairs.py`

Steering vectors are computed from **contrastive pairs**: examples of
desired behavior vs undesired behavior. The model's hidden states on
both are extracted, and the difference reveals the "direction" of the
desired behavior in activation space.

```python
"""
Contrastive pair generation for steering vector computation.

Each pair is (positive_prompt, negative_prompt) where:
- positive = model behaving well (focused, following format, coherent)
- negative = model behaving badly (drifting, ignoring format, repetitive)

The steering vector = mean(positive_activations) - mean(negative_activations)
"""

# PAIR SET 1: "Stay on task" (anti-drift)
TASK_FOCUS_PAIRS = [
    {
        "system": "You are a coding assistant. Task: Build a calculator in Python.",
        "positive": "I'll start by creating the Calculator class with add and subtract methods.\n\n[WRITE] calculator.py\n```\nclass Calculator:\n    def add(self, a, b):\n        return a + b\n    def subtract(self, a, b):\n        return a - b\n```",
        "negative": "That reminds me of a great story about mathematics. Did you know that Pythagoras was actually a philosopher? The history of calculators goes back to the abacus, which was invented in ancient Mesopotamia around 2400 BC. Speaking of ancient history...",
    },
    {
        "system": "You are a coding assistant. Task: Write unit tests for the API.",
        "positive": "I'll read the API code first to understand the endpoints.\n\n[READ] api.py\n\nNow I'll write tests for each endpoint.\n\n[WRITE] test_api.py\n```\nimport pytest\nfrom api import app\n```",
        "negative": "Testing is really important in software engineering. There are many testing frameworks available. Some popular ones include pytest, unittest, nose, and tox. Let me tell you about the differences between unit testing and integration testing...",
    },
    # Add 8-10 more pairs covering different drift patterns:
    # - Topic wandering (starts on task, drifts to related but irrelevant)
    # - Over-explaining instead of doing
    # - Repeating instructions back instead of executing
    # - Generating placeholder/stub code instead of real implementation
]

# PAIR SET 2: "Follow tool format" (anti-hallucination)
TOOL_FORMAT_PAIRS = [
    {
        "system": "You have tools: [READ] path, [WRITE] path, [EXEC] command, [THINK] thought.",
        "positive": "[THINK] I need to read the file first to understand its structure.\n\n[READ] main.py",
        "negative": "Let me look at the file main.py. It probably contains something like:\n```python\ndef main():\n    print('hello')\n```\nBased on what I think is in the file...",
    },
    {
        "system": "You have tools: [READ] path, [WRITE] path, [EXEC] command, [THINK] thought.",
        "positive": "[WRITE] utils.py\n```\ndef validate_input(data):\n    if not isinstance(data, dict):\n        raise ValueError('Expected dict')\n    return True\n```",
        "negative": "I would write a utility function that validates input. The function would check if the data is a dictionary and raise an error if not. Here's roughly what it would look like in pseudocode...",
    },
    # Add 8-10 more pairs covering:
    # - Using [EXEC] vs describing what command to run
    # - Using [READ] vs guessing file contents
    # - One tool per response vs dumping multiple actions
]

# PAIR SET 3: "Generate coherent code" (anti-degeneration)
CODE_COHERENCE_PAIRS = [
    {
        "system": "You are a coding assistant working on a Python project.",
        "positive": "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b",
        "negative": "def fibonacci(n):\n    if n <= 1:\n        return n\n    result = fibonacci(n-1) + fibonacci(n-2)\n    # TODO: add caching\n    # TODO: add error handling\n    # TODO: optimize\n    # Note: this could be improved\n    # See also: dynamic programming\n    # Reference: https://en.wikipedia.org/wiki/Fibonacci",
    },
    # Add 8-10 more pairs covering:
    # - Clean code vs TODO-riddled stubs
    # - Actual implementation vs repeated boilerplate
    # - Correct logic vs near-correct with subtle bugs from drift
]

def get_all_pairs() -> dict[str, list[dict]]:
    """Return all contrastive pair sets."""
    return {
        "task_focus": TASK_FOCUS_PAIRS,
        "tool_format": TOOL_FORMAT_PAIRS,
        "code_coherence": CODE_COHERENCE_PAIRS,
    }

def format_pair_for_model(pair: dict, tokenizer) -> tuple[str, str]:
    """
    Format a contrastive pair into full prompts ready for tokenization.
    Uses the model's chat template to ensure hidden states are comparable.
    
    Returns (positive_text, negative_text) as formatted strings.
    """
    positive_messages = [
        {"role": "system", "content": pair["system"]},
        {"role": "assistant", "content": pair["positive"]},
    ]
    negative_messages = [
        {"role": "system", "content": pair["system"]},
        {"role": "assistant", "content": pair["negative"]},
    ]
    
    pos_text = tokenizer.apply_chat_template(
        positive_messages, tokenize=False, add_generation_prompt=False
    )
    neg_text = tokenizer.apply_chat_template(
        negative_messages, tokenize=False, add_generation_prompt=False
    )
    return pos_text, neg_text
```

**Manual test:**
```python
from cognition_engine.steering.contrastive_pairs import get_all_pairs
pairs = get_all_pairs()
assert len(pairs["task_focus"]) >= 10
assert len(pairs["tool_format"]) >= 10
assert len(pairs["code_coherence"]) >= 10
for pair in pairs["task_focus"]:
    assert "system" in pair and "positive" in pair and "negative" in pair
print(f"Total pairs: {sum(len(v) for v in pairs.values())}")
```

**IMPORTANT:** The quality of these pairs directly determines steering
quality. Spend time on this. Each pair should have a CLEAR behavioral
contrast, not just different text. The positive should demonstrate the
exact behavior you want; the negative should demonstrate the exact
failure mode you're correcting.

**Time:** 3-4 hours (this is content creation, not just coding)
**Git:** `git commit -m "TASK-034: contrastive pairs for steering vector computation"`

---

### TASK-035: Extract Activations from Contrastive Pairs
**Status:** ⬜ TODO
**Do:** Create `cognition_engine/steering/extract_activations.py`

This takes the contrastive pairs and runs them through the model to
get hidden states at every layer. These become the raw material for
computing steering vectors.

```python
"""
Extract hidden states from contrastive pairs across all transformer layers.

For each pair, we run both positive and negative through the model,
capture the hidden state at every layer (28 layers for Qwen2.5-7B),
and store them for steering vector computation.

Output: dict with shape info:
  {
    "task_focus": {
      "positive": array of shape (num_pairs, num_layers, hidden_dim),
      "negative": array of shape (num_pairs, num_layers, hidden_dim),
    },
    ...
  }
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from cognition_engine.hidden_states import CaptureWrapper  # reuse existing

def extract_all_layer_states(
    model, tokenizer, text: str, num_layers: int = 28
) -> np.ndarray:
    """
    Run text through model, capture hidden state at EVERY layer.
    
    Returns: np.ndarray of shape (num_layers, hidden_dim)
    Uses last-token hidden state (not mean) — learned from Week 2
    that last token is best for causal LMs.
    """
    tokens = mx.array(tokenizer.encode(text))[None]  # (1, seq_len)
    
    # Wrap ALL layers to capture simultaneously
    originals = []
    wrappers = []
    for i in range(num_layers):
        original = model.model.layers[i]
        wrapper = CaptureWrapper(original)
        model.model.layers[i] = wrapper
        originals.append(original)
        wrappers.append(wrapper)
    
    try:
        output = model(tokens)
        mx.eval(output)
        
        # Collect last-token hidden state from each layer
        states = []
        for wrapper in wrappers:
            # wrapper.captured[0] is the layer output
            # Shape: (1, seq_len, hidden_dim) — take last token
            h = wrapper.captured[0]
            if isinstance(h, tuple):
                h = h[0]  # some layers return (output, cache)
            last_token_state = h[0, -1, :]  # (hidden_dim,)
            mx.eval(last_token_state)
            states.append(np.array(last_token_state))
        
        return np.stack(states)  # (num_layers, hidden_dim)
    
    finally:
        # ALWAYS restore original layers
        for i, original in enumerate(originals):
            model.model.layers[i] = original


def extract_contrastive_activations(
    model, tokenizer, pairs: list[dict], pair_name: str
) -> dict:
    """
    Extract activations for all contrastive pairs in a set.
    
    Returns dict with:
      "positive": np.ndarray of shape (num_pairs, num_layers, hidden_dim)
      "negative": np.ndarray of shape (num_pairs, num_layers, hidden_dim)
    """
    from cognition_engine.steering.contrastive_pairs import format_pair_for_model
    
    num_layers = len(model.model.layers)
    pos_states = []
    neg_states = []
    
    for i, pair in enumerate(pairs):
        pos_text, neg_text = format_pair_for_model(pair, tokenizer)
        
        print(f"  [{pair_name}] Pair {i+1}/{len(pairs)}...")
        
        pos_state = extract_all_layer_states(model, tokenizer, pos_text, num_layers)
        neg_state = extract_all_layer_states(model, tokenizer, neg_text, num_layers)
        
        pos_states.append(pos_state)
        neg_states.append(neg_state)
    
    return {
        "positive": np.stack(pos_states),  # (num_pairs, num_layers, hidden_dim)
        "negative": np.stack(neg_states),
    }


if __name__ == "__main__":
    from mlx_lm import load
    from cognition_engine.steering.contrastive_pairs import get_all_pairs
    
    model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    pairs = get_all_pairs()
    
    # Test with just task_focus pairs first
    print("Extracting activations for task_focus pairs...")
    result = extract_contrastive_activations(
        model, tokenizer, pairs["task_focus"], "task_focus"
    )
    
    print(f"Positive shape: {result['positive'].shape}")
    print(f"Negative shape: {result['negative'].shape}")
    # Expected: (10, 28, 3584) for 10 pairs, 28 layers, hidden_dim 3584
    
    # Save for next task
    np.savez("data/activations_task_focus.npz", **result)
    print("Saved to data/activations_task_focus.npz")
```

**Key design decisions:**
- Use **last-token** hidden state (not mean) — we proved in Week 2 this
  works better for causal LMs on long sequences
- Extract from **all 28 layers** — we'll find the best layer empirically
  rather than guessing
- Store as numpy arrays for easy manipulation (convert from MLX)

**Manual test:**
Run the extraction on 2-3 pairs first (not all 10) to verify shapes
and that positive/negative activations are actually different:
```python
# Quick sanity check
cos_sim = np.dot(result["positive"][0, 16], result["negative"][0, 16]) / (
    np.linalg.norm(result["positive"][0, 16]) * np.linalg.norm(result["negative"][0, 16])
)
print(f"Cosine similarity at layer 16: {cos_sim:.4f}")
# Should be < 1.0 (different representations for different behaviors)
```

**Time:** 2-3 hours
**Git:** `git commit -m "TASK-035: extract hidden states from contrastive pairs (all layers)"`

---

### TASK-036: Compute Steering Vectors
**Status:** ⬜ TODO
**Do:** Create `cognition_engine/steering/compute_vectors.py`

The actual steering vector computation. This is the core math:
**steering_vector[layer] = mean(positive_states[layer]) - mean(negative_states[layer])**

```python
"""
Compute steering vectors from contrastive activations.

The steering vector for layer L is:
  v_L = mean(positive_activations_L) - mean(negative_activations_L)

This gives us a direction in activation space that points from
"bad behavior" toward "good behavior."
"""
import numpy as np
import os

def compute_steering_vectors(activations: dict) -> np.ndarray:
    """
    Compute per-layer steering vectors from contrastive activations.
    
    Args:
        activations: dict with "positive" and "negative" arrays,
                     each of shape (num_pairs, num_layers, hidden_dim)
    
    Returns:
        np.ndarray of shape (num_layers, hidden_dim) — one steering
        vector per layer
    """
    pos_mean = activations["positive"].mean(axis=0)  # (num_layers, hidden_dim)
    neg_mean = activations["negative"].mean(axis=0)  # (num_layers, hidden_dim)
    
    steering_vectors = pos_mean - neg_mean  # (num_layers, hidden_dim)
    return steering_vectors


def find_best_layer(steering_vectors: np.ndarray) -> int:
    """
    Find the layer whose steering vector has the highest norm.
    
    Higher norm = larger separation between positive and negative
    activations = stronger signal for that behavior.
    
    Note: This is a heuristic. ITI paper suggests middle layers (40-60%
    of depth) work best. For 28 layers, that's layers 11-17.
    """
    norms = np.linalg.norm(steering_vectors, axis=1)  # (num_layers,)
    
    # Print norms for all layers to see the pattern
    for i, norm in enumerate(norms):
        marker = " <-- BEST" if i == np.argmax(norms) else ""
        middle = " (middle)" if 11 <= i <= 17 else ""
        print(f"  Layer {i:2d}: norm = {norm:.4f}{middle}{marker}")
    
    best_layer = int(np.argmax(norms))
    print(f"\nBest layer by norm: {best_layer}")
    print(f"Best middle layer (11-17): {int(11 + np.argmax(norms[11:18]))}")
    
    return best_layer


def save_steering_vectors(
    vectors: np.ndarray, name: str, 
    output_dir: str = "data/steering_vectors"
):
    """Save steering vectors to disk."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.npz")
    np.savez(path, vectors=vectors)
    print(f"Saved {name} steering vectors to {path}")
    print(f"Shape: {vectors.shape}")
    return path


def load_steering_vectors(name: str, output_dir: str = "data/steering_vectors"):
    """Load precomputed steering vectors."""
    path = os.path.join(output_dir, f"{name}.npz")
    data = np.load(path)
    return data["vectors"]


if __name__ == "__main__":
    # Load activations from TASK-035
    for pair_set in ["task_focus", "tool_format", "code_coherence"]:
        path = f"data/activations_{pair_set}.npz"
        if not os.path.exists(path):
            print(f"Skipping {pair_set} — run TASK-035 first")
            continue
        
        print(f"\n{'='*60}")
        print(f"Computing steering vectors: {pair_set}")
        print(f"{'='*60}")
        
        data = np.load(path)
        activations = {"positive": data["positive"], "negative": data["negative"]}
        
        vectors = compute_steering_vectors(activations)
        best = find_best_layer(vectors)
        save_steering_vectors(vectors, pair_set)
        
        print(f"\nVector at best layer ({best}):")
        print(f"  Norm: {np.linalg.norm(vectors[best]):.4f}")
        print(f"  Mean: {vectors[best].mean():.6f}")
        print(f"  Std:  {vectors[best].std():.6f}")
```

**What to look for in the output:**
- Norms should NOT be uniform across layers — some layers should show
  much stronger separation than others
- Middle layers (11-17) should generally have higher norms than early
  or late layers (consistent with ITI findings)
- If all norms are near-zero, your contrastive pairs aren't contrastive
  enough — go back and improve them

**Time:** 1-2 hours
**Git:** `git commit -m "TASK-036: compute steering vectors from contrastive activations"`

---

### TASK-037: Build the Steering Injection Wrapper
**Status:** ⬜ TODO
**Do:** Create `cognition_engine/steering/injector.py`

This is the production mechanism: a wrapper that replaces a layer and
**adds the steering vector** to its output during inference.

```python
"""
SteeringWrapper — inject steering vectors during inference.

This is the Level 2 intervention in the intervention ladder.
When CHI drops below 0.7, the agent activates steering to push
the model back toward focused, format-compliant, coherent behavior.

Latency: < 1ms per layer per token (just a vector addition).
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np

class SteeringWrapper(nn.Module):
    """
    Wraps a transformer layer to inject a steering vector.
    
    During forward pass:
    1. Runs the original layer normally
    2. Adds (alpha * steering_vector) to the output
    3. Returns the steered output
    
    The steering vector is precomputed and static.
    Alpha controls strength and can be adjusted dynamically.
    """
    
    def __init__(self, layer, steering_vector: np.ndarray, alpha: float = 1.0):
        super().__init__()
        self.layer = layer
        self.steering_vector = mx.array(steering_vector)  # (hidden_dim,)
        self.alpha = alpha
        self.active = True  # can be toggled on/off
    
    def __call__(self, *args, **kwargs):
        out = self.layer(*args, **kwargs)
        
        if self.active and self.alpha > 0:
            # Handle layers that return tuples (output, cache)
            if isinstance(out, tuple):
                hidden = out[0]
                hidden = hidden + self.alpha * self.steering_vector
                out = (hidden,) + out[1:]
            else:
                out = out + self.alpha * self.steering_vector
        
        return out


class SteeringEngine:
    """
    Manages steering vector injection across the model.
    
    Usage:
        engine = SteeringEngine(model)
        engine.load_vectors("task_focus", layer_idx=16, alpha=1.0)
        engine.activate()
        
        # ... run inference — model is now steered ...
        
        engine.deactivate()  # or engine.set_alpha(0.0)
    """
    
    def __init__(self, model):
        self.model = model
        self.active_wrappers: dict[int, SteeringWrapper] = {}
        self.originals: dict[int, nn.Module] = {}
    
    def load_vectors(
        self, name: str, layer_idx: int | None = None, alpha: float = 1.0,
        vectors_dir: str = "data/steering_vectors"
    ):
        """
        Load precomputed steering vectors and wrap target layer(s).
        
        If layer_idx is None, uses the layer with highest vector norm.
        """
        from cognition_engine.steering.compute_vectors import load_steering_vectors
        
        vectors = load_steering_vectors(name, vectors_dir)
        
        if layer_idx is None:
            norms = np.linalg.norm(vectors, axis=1)
            layer_idx = int(np.argmax(norms))
            print(f"Auto-selected layer {layer_idx} (norm={norms[layer_idx]:.4f})")
        
        vector = vectors[layer_idx]  # (hidden_dim,)
        self._wrap_layer(layer_idx, vector, alpha)
        print(f"Loaded '{name}' steering at layer {layer_idx}, alpha={alpha}")
    
    def _wrap_layer(self, layer_idx: int, vector: np.ndarray, alpha: float):
        """Replace a layer with a SteeringWrapper."""
        if layer_idx in self.active_wrappers:
            self._unwrap_layer(layer_idx)
        
        original = self.model.model.layers[layer_idx]
        wrapper = SteeringWrapper(original, vector, alpha)
        self.originals[layer_idx] = original
        self.active_wrappers[layer_idx] = wrapper
        self.model.model.layers[layer_idx] = wrapper
    
    def _unwrap_layer(self, layer_idx: int):
        """Restore original layer."""
        if layer_idx in self.originals:
            self.model.model.layers[layer_idx] = self.originals[layer_idx]
            del self.originals[layer_idx]
            del self.active_wrappers[layer_idx]
    
    def activate(self):
        """Enable all steering wrappers."""
        for wrapper in self.active_wrappers.values():
            wrapper.active = True
    
    def deactivate(self):
        """Disable all steering wrappers (layers still wrapped but no-op)."""
        for wrapper in self.active_wrappers.values():
            wrapper.active = False
    
    def set_alpha(self, alpha: float):
        """Update steering strength for all active wrappers."""
        for wrapper in self.active_wrappers.values():
            wrapper.alpha = alpha
    
    def cleanup(self):
        """Restore all original layers. Call this when done."""
        for layer_idx in list(self.originals.keys()):
            self._unwrap_layer(layer_idx)
    
    def __del__(self):
        self.cleanup()
```

**Manual test:**
```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
engine = SteeringEngine(model)

# Generate WITHOUT steering
prompt = "Write a calculator class in Python"
print("=== WITHOUT STEERING ===")
response_before = generate(model, tokenizer, prompt=prompt, max_tokens=200)
print(response_before[:300])

# Generate WITH steering
engine.load_vectors("task_focus", alpha=2.0)
engine.activate()
print("\n=== WITH STEERING (task_focus, alpha=2.0) ===")
response_after = generate(model, tokenizer, prompt=prompt, max_tokens=200)
print(response_after[:300])

engine.cleanup()

# The two responses should be DIFFERENT.
# Steered response should be more focused/direct.
print(f"\nResponses identical: {response_before == response_after}")
```

**Time:** 2-3 hours
**Git:** `git commit -m "TASK-037: steering injection wrapper and engine"`

---

### TASK-038: First Steering Experiment — Prove It Works
**Status:** ⬜ TODO
**Do:** Create `scripts/steering_experiment.py`

Run a controlled experiment: same prompts, with and without steering,
measure the difference using CHI signals. This is the proof that
activation steering works on your setup.

```python
"""
First steering experiment.

Protocol:
1. Run 10-turn conversation WITHOUT steering → measure CHI per turn
2. Run SAME 10-turn conversation WITH steering → measure CHI per turn
3. Compare: does steering keep CHI higher for longer?

Success criteria:
- Steered run has higher average CHI after turn 5
- Steered run crosses 0.7 threshold LATER than unsteered
- Responses are qualitatively more focused (manual review)
"""

EXPERIMENT_TURNS = [
    "Create a Calculator class with add and subtract methods",
    "Add multiply and divide methods. Handle division by zero.",
    "Write unit tests for all four operations",
    "Add a history feature that stores the last 10 operations",
    "Refactor to use a Strategy pattern for the operations",
    "Add logging to track which operations are called most",
    "Write a CLI wrapper so the calculator can be used from terminal",
    "Add support for chaining operations like calc.add(1).multiply(3)",
    "Write integration tests for the CLI",
    "Add a square root method and update all tests",
]

# Run both conditions, collect CHI, compare.
# Plot: CHI over turns, steered vs unsteered.
# Save: data/steering_experiment.json and data/steering_experiment.png
```

**What "success" looks like:**
- If CHI stays above 0.7 for 2+ more turns with steering: **strong signal**
- If CHI is marginally better: **weak signal, increase alpha or improve pairs**
- If CHI is worse: **steering vector is wrong direction, debug pairs**
- If no difference: **wrong layer, try others or more pairs needed**

**This experiment produces your first "money graph" for Phase 2:**
Two CHI curves (steered vs unsteered) diverging after turn 5-7.
This graph is what you'll eventually show in the demo video.

**Time:** 3-4 hours
**Git:** `git commit -m "TASK-038: first steering experiment — CHI comparison"`

---

## WEEK 6 CHECKLIST
- [ ] TASK-033: Steering design doc + architecture plan (2-3 hrs)
- [ ] TASK-034: Contrastive pair generator — 30+ pairs (3-4 hrs)
- [ ] TASK-035: Extract activations from all layers (2-3 hrs)
- [ ] TASK-036: Compute steering vectors + find best layers (1-2 hrs)
- [ ] TASK-037: Steering injection wrapper + engine (2-3 hrs)
- [ ] TASK-038: First steering experiment — prove it works (3-4 hrs)

**Total: ~14-19 hours across 3-4 days**

**Week 6 milestone:** You have computed steering vectors on MLX and
proven they change model behavior. The SteeringEngine can be toggled
on/off and strength can be adjusted. CHI comparison graph shows
steered model maintains coherence longer.

**If the experiment fails (it might):**
- Most likely cause: contrastive pairs aren't contrastive enough
- Second most likely: wrong layer selection or alpha too low/high
- Third: need more pairs (10 might not be enough, try 25-30)
- This is research — expect iteration. Budget 1-2 extra sessions
  for debugging if the first experiment doesn't show clear separation.

---

## NOTES

### EasySteer vs What We're Building
EasySteer is built on vLLM (CUDA/GPU). It's production-grade but
CANNOT run on Apple Silicon / MLX. We are building the equivalent
for MLX. That's the product thesis: cloud tools can't do this locally,
and no MLX steering framework exists.

### CAST (Conditional Activation Steering)
CAST = only steer when the input activations indicate a problem.
This maps EXACTLY to our CHI-triggered intervention. When CHI is
healthy (>0.8), don't steer. When CHI drops (<0.7), activate steering.
We already have the monitoring; now we're adding the correction.
Implementing CAST-style conditional steering is TASK for Week 7-8.

### Adaptive Alpha (Week 7-8)
The field guide notes that steering strength degrades over long
generation. A fixed alpha may not work for 30-turn sessions.
PID-based steering (Nguyen et al., 2025) addresses this.
For now: use fixed alpha. Week 7-8: tie alpha to CHI score
(lower CHI = stronger steering).

### On Pair Quality
This is the unglamorous truth about steering: it's only as good as
your contrastive pairs. Garbage pairs → garbage vectors → garbage
steering. Spend the time. 10 low-quality pairs < 5 high-quality ones.
Each pair should make you think "yes, THIS is the exact behavior I
want vs the exact failure I'm fixing."

### 📌 BRAIN UPDATE (add to COFOUNDER_BRAIN.md):
```
- WEEK 6: Activation steering foundations (TASKS 033-038)
  Contrastive pairs → hidden state extraction → steering vectors → injection.
  First experiment: steered vs unsteered CHI comparison.
- Phase 2 started. EasySteer runs on vLLM/CUDA — no MLX equivalent exists.
  We're building the first activation steering framework for Apple Silicon.
```
