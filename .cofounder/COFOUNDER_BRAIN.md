# COFOUNDER_BRAIN.md
# Last updated: 2026-03-01
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
| Level | Trigger | Action | Latency |
|-------|---------|--------|---------|
| 1 | CHI > 0.8 | Normal inference | Zero |
| 2 | CHI < 0.7 | EasySteer activation steering | Milliseconds |
| 3 | CHI < 0.5 | Context compaction + SCAN anchor tokens | 2-3 seconds |
| 4 | CHI < 0.3 | Cloud escalation (Claude/GPT-4 API) | API latency |
| 5 | Session end | Nightly LoRA crystallization | Async overnight |

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
- Dev model: Qwen2.5-Coder-7B-Instruct-4bit (fast iteration)
- Prod model: Qwen2.5-Coder-14B Q4_K_M (fits 24GB)
- Language: Python 3.11+
- Cloud fallback: Claude API (Level 4 escalation)
- Memory: Markdown files in .cognition/ directory

## COMPETITIVE MOAT
- vs Cursor/Copilot/Claude Code: they're cloud-only, can't do activation steering
- vs Ollama: wrapper with no cognitive monitoring
- vs GSD/SWE-AF: no real-time degradation detection
- vs Letta/MemGPT: memory only, no neural intervention

## EXECUTION PLAN — 16 WEEKS
- **Phase 0 (Weeks 1-3):** Foundations — MLX inference, hidden states, entropy, CHI score
- **Phase 1 (Weeks 4-6):** Free Product — context mgmt, agent loop, open source launch
- **Phase 2 (Weeks 7-11):** The Moat — activation steering (EasySteer), Pro tier launch
- **Phase 3 (Weeks 12-16):** Crystallization — nightly LoRA, full integration, Product Hunt

## CURRENT STATUS
- **Week:** 1 of 16
- **Phase:** Phase 0 — Foundations
- **Current task:** TASK-001
- **Blockers:** None

## WHAT'S BEEN BUILT
- [x] Architecture research + adversarial critique (50+ papers)
- [x] 16-week execution plan
- [x] Co-founder continuity protocol
- [ ] WEEK 1: MLX inference + hidden state extraction
- [ ] WEEK 2: Entropy computation + degradation visualization
- [ ] WEEK 3: CHI score + project skeleton + CLI

## KEY LEARNINGS
- 2026-03-01: Original ATLAS was overengineered → simplified to 5 layers
- 2026-03-01: Real-time LoRA physically impossible → nightly crystallization
- 2026-03-01: Context refresh is commodity → EasySteer is the real moat
- 2026-03-01: 39% avg multi-turn degradation (Stanford) validates product thesis
- 2026-03-01: Apple is best acquirer match (MLX + privacy philosophy)
