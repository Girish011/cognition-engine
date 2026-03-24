# COFOUNDER_BRAIN.md

# Last updated: 2026-03-12

# ⚡ This file is in Claude Project Knowledge — Claude reads it automatically

## PRODUCT

- **Name:** Cognition Engine
- **What:** Local AI coding agent that monitors its own cognitive health and self-corrects when degrading
- **One-liner:** "A local AI coding agent that never forgets what it's doing"
- **For:** Developers on Apple Silicon frustrated that local models lose coherence on long tasks
- **Pricing:** Proprietary product. May release CHI monitoring as limited OSS for visibility.

## FOUNDER

- Solo, bootstrapping with salary, no VC funding planned
- Generalist coder learning ML by building. Claude is technical co-founder.
- Hardware: Apple M4 Pro 24GB RAM
- IDE: Cursor
- Availability: 8-10 hrs/day, 6 days/week

## STRATEGY (REVISED 2026-03-09)

- **Build privately.** No full open-source release of core technology.
- **Proprietary:** Steering engine, LoRA crystallization, full intervention ladder
- **Limited OSS (for visibility):** CHI monitoring only (entropy + repetition + goal drift)
- **Reach Apple via:** (1) MLX team engagement on GitHub, (2) demo video, (3) one strategic blog post about the PROBLEM not the solution
- **Target:** Apple IP acquisition ($50-200M) via MLX team → dev tools integration
- **Do NOT:** Publish papers with full implementation details, open-source steering vectors

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

## CHI SCORE — 3 SIGNALS (IMPLEMENTED)

1. Attention entropy (Shannon entropy over next-token distribution, weight 0.4)
2. Output repetition (Jaccard similarity, weight 0.3, gated at <20 tokens)
3. Goal drift (hidden state cosine similarity vs original task, weight 0.3)

- EMA smoothing: alpha=0.3, effective window ~3-5 turns
- Validated: crosses 0.7 threshold at turn 11, matching inflection point (turn 10)

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
- Proprietary core — only CHI monitoring released as OSS for visibility

## TECH STACK

- Inference: MLX + mlx-lm (Apple Silicon native)
- Dev model: Qwen2.5-Coder-7B-Instruct-4bit (pure transformer, current)
- Prod model: Self-convert Qwen3.5-9B-Instruct via mlx_lm.convert (TASK-029)  
NOTE: Qwen3.5 uses Gated DeltaNet hybrid attention — steering may need adaptation  
Convert command: mlx_lm.convert --hf-path Qwen/Qwen3.5-9B-Instruct --mlx-path ./models/Qwen3.5-9B-Instruct-4bit -q
- Language: Python 3.11+
- Package: cognition_engine/ (pip installable, pyproject.toml)
- Cloud fallback: Claude API (Level 4 escalation)
- Memory: JSON files in .cognition/ directory

## KEY REPOS

- MLX: [https://github.com/ml-explore/mlx](https://github.com/ml-explore/mlx)
- mlx-lm (model code HERE): [https://github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)
- Key source files: mlx_lm/models/qwen2.py, mlx_lm/generate.py, mlx_lm/utils.py

## COMPETITIVE LANDSCAPE

- vs Cursor/Copilot/Claude Code: cloud-only, can't do activation steering
- vs Ollama: wrapper, no cognitive monitoring
- vs OpenFang: agent orchestration (40 channels), no cognitive monitoring
- vs Ruflo (18K stars): multi-agent orchestration, rule-based gates, no neural access
- vs EvoSkill: optimizes instructions around frozen model. We modify model's neural state.
- vs Hindsight: cross-session memory system. Potential integration partner for Phase 4.

## EXECUTION PLAN — 16 WEEKS (REVISED)

- **Phase 0 (Weeks 1-3):** Foundations — COMPLETE
- **Phase 1 (Weeks 4-5):** Agent Foundation — context mgmt, agent loop, working memory
- **Phase 2 (Weeks 6-10):** The Moat — activation steering (EasySteer), the hero feature
- **Phase 3 (Week 11-12):** Strategic Launch — demo video, blog post, limited OSS release
- **Phase 4 (Weeks 13-16):** LoRA Crystallization + Apple outreach

## CURRENT STATUS

- **Week:** 4 of 16
- **Phase:** Phase 1 — Agent Foundation
- **Current task:** TASK-021 (Context Compaction Engine)
- **Blockers:** None

## WHAT'S BEEN BUILT

- Phase 0 COMPLETE: Architecture research (50+ papers), execution plan, continuity protocol
- WEEK 1: MLX inference + hidden state extraction  
53 tok/s baseline, CaptureLayer wrapper, attention weights (1,28,seq,seq)  
28 layers, hidden dim 3584, GQA with 28 heads
- WEEK 2: Entropy + repetition + goal drift + degradation graphs  
Inflection at turn 10 (~2500 tokens). Three signals confirm degradation.  
Goal drift fix: last-token hidden state for long sequences.
- WEEK 3: CHI score + CLI + project structure + README  
CHI validated: crosses 0.7 at turn 11. EMA smoothing alpha=0.3.  
Repetition gated <20 tokens. 14 tests passing.  
CLI: python -m cognition_engine (chat mode with color-coded CHI)  
Package restructured as cognition_engine/ with pyproject.toml
- WEEK 4: Context compaction + agent loop + working memory (TASKS 021-026)
- WEEK 5: Agent polish + Qwen3.5 eval + MLX community engagement (TASKS 027-032)
- WEEK 6: Activation steering foundations (TASKS 033-038)
  Contrastive pairs → hidden state extraction → steering vectors → injection.
  First experiment: steered vs unsteered CHI comparison.
- Phase 2 started. EasySteer runs on vLLM/CUDA — no MLX equivalent exists.
  We're building the first activation steering framework for Apple Silicon.

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
- 2026-03-08: Full 20-turn 3-signal experiment completed. Inflection at turn 10
(~2000-2500 tokens context). Entropy is strongest signal (clean regime change).
Goal drift fixed: use last-token hidden state for long sequences, not mean.
Repetition noisy for short responses — gate by response length in CHI.
- 2026-03-08: For CHI design: use exponential moving average (3-5 turn window)
to smooth single-turn noise. Gate repetition signal when response < 20 tokens.
- 2026-03-08: CHI validated against real data. Crosses 0.7 threshold at turn 11,
matching TASK-014 inflection (turn 10). EMA smoothing (α=0.3) handles noisy
repetition signal. 16-turn interactive CLI test confirmed real-world behavior.
- 2026-03-08: Phase 0 complete. Foundation: hidden states, entropy, repetition,
goal drift, CHI, CLI, tests, README. All working end-to-end.
- 2026-03-08: Strategy shift. Don't open-source core IP. Build privately,
launch with full story (monitoring + steering together). Release only
CHI monitoring as OSS. Steering + LoRA stays proprietary. Reach Apple
through MLX team (they watch what people build on MLX), demo video,
and one strategic blog post. Paper publishing risks giving blueprint
to well-funded teams who can replicate in 2-4 weeks.
- 2026-03-09: Path A initiated. Plan: open GitHub Discussion on mlx-lm
about hidden state extraction patterns. Share CaptureLayer approach
(not moat). Goal: become known to MLX team as "the person building
cognitive monitoring on MLX." No mention of acquisition/selling.

## KEY LEARNINGS (condensed)

- MLX: no forward hooks → CaptureLayer wrapper. mx.eval() returns None.
- 53 tok/s on M4 Pro for 7B Q4. Inflection at turn 10 (~2500 tokens).
- Entropy is strongest degradation signal. Repetition noisy for short responses.
- Goal drift: use last-token hidden state for long sequences, not mean.
- EMA smoothing (alpha=0.3) handles noise well. Gate repetition at <20 tokens.
- Turkey problem: neural intervention more durable than wrappers but not immune.
- Qwen3.5: self-convert via mlx_lm.convert. DeltaNet hybrid attention needs testing.
- Strategy: build privately, reach Apple through MLX team + demo video.

