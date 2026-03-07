# COFOUNDER_BRAIN.md

# Last updated: 2026-03-03

# ⚡ Paste this at the START of every new Claude conversation

## PRODUCT

- **Name:** Cognition Engine
- **What:** Local AI coding agent that monitors its own cognitive health and self-corrects when degrading
- **One-liner:** "A local AI coding agent that never forgets what it's doing"
- **For:** Developers on Apple Silicon frustrated that local models lose coherence on long tasks
- **Pricing:** Free OSS (monitoring + context mgmt) → $29/month Pro (activation steering + nightly LoRA)

## FOUNDER

- Solo, bootstrapping with salary, no VC funding planned
- Generalist coder learning ML by building. Claude is technical co-founder.
- Hardware: Apple M4 Pro 24GB RAM
- IDE: Cursor
- Availability: 8-10 hrs/day, 6 days/week

## ARCHITECTURE — 5 LAYERS

1. **Hardware Profiler** — detect chip/RAM/thermal → select model+quantization
2. **Intelligent Router** — classify task difficulty → route local vs cloud
3. **CHI Monitor** — 5 signals → cognitive health index score (0-1)
4. **Intervention Engine** — 5-level ladder triggered by CHI thresholds
5. **Memory** — 3-tier: active context / working memory / persistent disk

## CHI SCORE — 5 SIGNALS

1. Attention entropy (Shannon entropy over next-token distribution)
2. Key-token perplexity (LongPPL-style on important tokens)
3. Output repetition (Jaccard similarity between recent outputs)
4. Goal drift (cosine similarity between current output and original task)
5. Response coherence (structural quality of generated code)

## 5-LEVEL INTERVENTION LADDER


| Level | Trigger     | Action                                  | Latency         |
| ----- | ----------- | --------------------------------------- | --------------- |
| 1     | CHI > 0.8   | Normal inference                        | Zero            |
| 2     | CHI < 0.7   | EasySteer activation steering           | Milliseconds    |
| 3     | CHI < 0.5   | Context compaction + SCAN anchor tokens | 2-3 seconds     |
| 4     | CHI < 0.3   | Cloud escalation (Claude/GPT-4 API)     | API latency     |
| 5     | Session end | Nightly LoRA crystallization            | Async overnight |


## LOCKED DECISIONS (do not reverse without good reason)

- MLX native, NOT Ollama — need hidden state access for steering
- EasySteer activation steering is the hero product — cloud tools can't replicate
- Nightly LoRA crystallization, NOT real-time — training takes 15-30 min minimum
- Dual diagnosis — behavioral drift vs knowledge overload need different fixes
- Apple Silicon first — expand to NVIDIA later
- Build from scratch — no PEFT/Unsloth wrappers; native MLX implementation
- Open source core, Pro features gated behind license key

## TECH STACK

- Inference: MLX + mlx-lm (Apple Silicon native)
- Dev model (Weeks 1-3): Qwen2.5-Coder-7B-Instruct-4bit (pure transformer, good for learning)
- Prod model (Week 4+): Evaluate Qwen3.5-9B or Qwen3.5-4B (hybrid DeltaNet attention)
- Language: Python 3.11+
- Cloud fallback: Claude API (Level 4 escalation)
- Memory: Markdown files in .cognition/ directory

## KEY REPOS

- MLX (low-level array framework): [https://github.com/ml-explore/mlx](https://github.com/ml-explore/mlx)
- mlx-lm (LLM inference — THIS is where model code lives): [https://github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)
- Source files to study:
  - mlx_lm/models/[qwen2.py](http://qwen2.py) → model architecture
  - mlx_lm/[generate.py](http://generate.py) → token generation loop
  - mlx_lm/[utils.py](http://utils.py) → model loading
  - Find locally: python -c "import mlx_lm; print(mlx_lm.**file**)"

## COMPETITIVE MOAT

- vs Cursor/Copilot/Claude Code: they're cloud-only, can't do activation steering
- vs Ollama: wrapper with no cognitive monitoring
- vs GSD/SWE-AF: no real-time degradation detection
- vs Letta/MemGPT: memory only, no neural intervention
- vs OpenFang: agent orchestration/routing, no cognitive health monitoring at all
- vs Ruflo (18K stars): multi-agent orchestration for Claude Code (cloud API). Governs agent behavior through rule-based gates. Cannot do activation steering or cognitive monitoring — agents are API black boxes. Complementary, not competitive.

## EXECUTION PLAN — 16 WEEKS

- **Phase 0 (Weeks 1-3):** Foundations — MLX inference, hidden states, entropy, CHI score
- **Phase 1 (Weeks 4-6):** Free Product — context mgmt, agent loop, open source launch
- **Phase 2 (Weeks 7-11):** The Moat — activation steering (EasySteer), Pro tier launch
- **Phase 3 (Weeks 12-16):** Crystallization — nightly LoRA, full integration, Product Hunt

## CURRENT STATUS

- **Week:** 2 of 16
- **Phase:** Phase 0 — Foundations
- **Current task:** TASK-008
- **Blockers:** None

## WHAT'S BEEN BUILT

- Architecture research + adversarial critique (50+ papers)
- 16-week execution plan
- Co-founder continuity protocol
- WEEK 1: MLX inference + hidden state extraction
- WEEK 2: Entropy computation + degradation visualization
- WEEK 3: CHI score + project skeleton + CLI
- - WEEK 1: MLX inference + hidden state extraction
- 53.0 tok/s baseline on M4 Pro 24GB (better than expected)
- CaptureLayer wrapper for hidden states (no PyTorch hooks in MLX)
- Attention weight extraction via replicated forward pass
- Cosine similarity 0.7379 between code vs poetry (capture verified)
- Hidden state shape: (1, seq_len, 3584), Attention: (1, 28, seq_len, seq_len)

## KEY LEARNINGS

- 2026-03-01: Original ATLAS was overengineered → simplified to 5 layers
- 2026-03-01: Real-time LoRA physically impossible → nightly crystallization
- 2026-03-01: Context refresh is commodity → EasySteer is the real moat
- 2026-03-01: 39% avg multi-turn degradation (Stanford) validates product thesis
- 2026-03-01: Apple is best acquirer match (MLX + privacy philosophy)
- 2026-03-03: Qwen3.5 small models released (0.8B/2B/4B/9B). 9B beats prev-gen 30B. Uses hybrid Gated DeltaNet attention (3:1 linear-to-full ratio) — NOT pure transformer. Steering may need to target full-attention layers specifically (every 4th block). Start learning on Qwen2.5 (pure transformer), switch to Qwen3.5 Week 4+.
- 2026-03-03: Model commoditization accelerating. Validates thesis: value is in making models work reliably, not in the model itself.
- 2026-03-03: Qwen3.5 has "toggle reasoning on/off" — potential free Level 2 intervention (enable thinking mode when CHI drops). Investigate Week 7.
- 2026-03-03: OpenFang is agent orchestration (40 channels, 27 providers), NOT cognitive monitoring. Not a competitor. Different product category entirely.
- 2026-03-04: Turkey problem analysis. Our moat (neural-level intervention) is more durable than wrapper/orchestration layers, but not immune. Defensive strategy: publish research + create CHI benchmark standard + build LoRA/steering artifacts that outlive the product. Don't build models from scratch — build the knowledge that makes any model better.
- 2026-03-07: MLX has no forward hooks. Solution: replace layer in model.model.layers[] with CaptureLayer wrapper, restore in finally block.
- 2026-03-07: mx.eval() returns None — never assign its result. Call mx.eval(tensor)
  only for side effect.
- 2026-03-07: 53 tok/s on M4 Pro for 7B Q4 — significantly above 25-40 estimate.
  Qwen3.5-9B should still be fast enough for production use.
  