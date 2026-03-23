The documentation for `steering_design.md` file, incorporating proposed structure and integrating the practical insights from the 2026 Practitioner's Field Guide and the CAST paper.

***

# steering_design.md

### 1. The core algorithm (3 steps)
Activation steering operates on the residual stream of the model without requiring any weight fine-tuning. The core algorithm consists of three straightforward steps:
*   **Step 1:** Generate contrastive pairs. Create two sets of prompts: one representing the positive/desired behavior, and one representing the negative/undesired behavior. For cleaner vectors, use template prompts that differ on exactly one behavioral dimension.
*   **Step 2:** Extract hidden states. Run both sets through the model and collect the intermediate layer activations. Compute the steering direction vector by calculating the mean difference: `mean(positive_activations) - mean(negative_activations)`.
*   **Step 3:** During inference, add the scaled steering vector to the target layer(s) during the forward pass.

### 2. Key decisions for our implementation
*   **Which layers to steer?** While the Inference-Time Intervention (ITI) paper found success in the early-to-middle layers for truthfulness, practical behavioral steering generally works best in the **late layers (last 25%)**, which encode high-level semantic properties like intent and style. Middle layers (40-60%) can work for reasoning-adjacent behaviors like uncertainty expression. For a typical 28-layer model (like Qwen-2.5), starting around layers 20–24 is recommended.
*   **Steering strength ($\alpha$)?** Start with $\alpha=1.0$ and use a binary search to tune it (e.g., doubling if no effect, halving if output degrades). **Crucial consideration:** Steering strength degrades over long generations, fading after ~300-500 tokens as the model's autoregressive conditioning pulls it back to default behavior. To counter this, we may need an adaptive $\alpha$ or dynamic re-injection.
*   **Apply at which token positions?** (All tokens vs last token vs specific positions). Steering effects are prompt-dependent, and long conversations experience drift. We will need to determine if re-applying the vector at multiple sequence positions is necessary for persistent behavioral modification.
*   **One vector or per-layer vectors?** Start with a single best layer and expand later. Naively stacking multiple vectors at the same layer is fragile and often fails due to vector interference. If multi-property steering is needed later, it is more robust to inject different vectors at different layers (e.g., anti-drift at Layer 24, format compliance at Layer 28).

### 3. Our steering targets (what we're fixing)
According to recent field evaluations, activation steering **works best for** high-level properties like sentiment, formality, refusal, style, and conciseness. It **works poorly for** injecting specific facts, factual recall, and complex multi-step reasoning. 

Fortunately, our specific use cases fall squarely into the "reliably steerable" category, as they deal with behavioral registers and format adherence rather than complex logic:
*   **"Stay on task"** — anti-drift steering (directly maps to our goal drift CHI signal).
*   **"Follow tool format"** — anti-hallucination and format compliance.
*   **"Generate coherent code"** — anti-repetition and anti-degeneration.

### 4. Architecture sketch
```text
CaptureWrapper (existing) → SteeringWrapper (new)

SteeringWrapper:
  - Wraps a transformer layer
  - On forward pass: runs layer, ADDS steering vector to output
  - Steering vector loaded from precomputed .npz file
  - Alpha (strength) controlled by CHI score
```
*Note: Modulating the $\alpha$ strength dynamically based on the CHI score is functionally identical to the 2025 advancement known as **CAST (Conditional Activation Steering)**. CAST solves the "blunt instrument" problem of always-on steering by analyzing activation patterns during inference to determine whether (and how strongly) to apply steering. Our CHI-triggered steering implements this exact conditional paradigm.*

### 5. How this connects to the intervention ladder
Because steering vectors operate at inference time by simply adding a precomputed vector to the residual stream, they introduce near-zero computational overhead. This makes them perfectly suited for our low-latency Level 2 intervention.

```text
Level 1 (CHI > 0.8): No steering — normal inference
Level 2 (CHI < 0.7): Activate steering vectors — milliseconds latency
Level 3 (CHI < 0.5): Context compaction + SCAN anchors — 2-3 seconds
Level 4 (CHI < 0.3): Cloud escalation — API latency
```