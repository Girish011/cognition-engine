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

- **Week:** 7 of 16
- **Phase:** Phase 2 — Activation Steering
- **Current task:** TASK-039
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
- WEEK 6: Activation steering pipeline complete. Contrastive pairs (30), steering vectors computed for 3 behavior types, SteeringEngine with injection/cleanup. First experiment: steering changes behavior, over-steering at layer 27/alpha 2.0 kills output. Layer 17/alpha 1.0 is promising.
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
- WEEK 6: Activation steering foundations (TASKS 033-038) — COMPLETE
  Pipeline working: pairs → extraction → vectors → injection → measurement.
  Proved steering changes behavior. Over-steering at layer 27/alpha 2.0
  kills output. Middle layers + moderate alpha are the sweet spot.
- WEEK 7: Steering calibration + agent integration (TASKS 039-044)
  Alpha/layer sweep, adaptive alpha, CHI-triggered steering in agent loop.
  Multi-vector steering. End-to-end integration test.
- 2026-03-28: Apple-Google Gemini distillation deal confirmed. Apple has full 
  Gemini access in own data centers, can distill on-device models, has access 
  to chain-of-thought reasoning traces. WWDC June 8 = new Siri reveal.
  Impact on us: validates on-device AI thesis strongly. Distilled models still 
  degrade at runtime — CHI + steering remains the unsolved layer. New angle: 
  position as "reliability infrastructure for any on-device model" including 
  Apple's distilled Gemini variants. Time Phase 3 launch for post-WWDC window 
  (late June/July) when on-device limitations become visible.
  No strategy reversal needed. Stay the course on Phase 2 steering.
- TASK-043 COMPLETE. Dual-vector (tool_format L16/α0.3 + code_coherence L17/α0.25)
  scored 0.814 mean CHI turns 5-9 — worse than code_coherence alone (0.819),
  died at turn 8. Vectors don't complement; they dilute.
- DECISION: Single-vector tool_format (L16, α0.5) remains the default.
  Only reliable vector across all 10 turns with full response length.
- Final steering vector ranking:
  1. tool_format: reliable, all turns alive, +0.011 CHI
  2. code_coherence: strongest CHI (+0.020) but dies turn 9
  3. dual_vector: no benefit over single
  4. task_focus: dead, do not use
- TASK-044 COMPLETE. End-to-end integration test: 3 modes compared.
  All three created files + ran pytest. None reached [DONE].
  Steering and full_ladder produced IDENTICAL results (compaction never triggered).
- CRITICAL FINDING: Steering prevents compaction from firing.
  Adaptive alpha keeps CHI at ~0.60, above the 0.50 compaction threshold.
  Level 2 blocks Level 3. Fix: lower compaction threshold to 0.65, or add
  time-based escalation ("steering active 5+ turns without recovery → compact").
- CRITICAL FINDING: The model completes the coding task by turn 6-8 but
  never emits [DONE]. Remaining turns are wasted. This is a prompt/detection
  issue, not an intervention issue. Week 8 priority: better completion detection.
- Week 7 overall: Steering pipeline works mechanically. Adaptive alpha works.
  Multi-vector doesn't help. The intervention ladder's value is in CHI monitoring
  + compaction, not in steering alone. Steering provides marginal CHI improvement
  (+0.01-0.02) but doesn't change task outcomes on a 7B model.
- 2026-03-29: TurboQuant (Google, ICLR 2026) — KV cache compression to 3-bit, 
  ~5x memory reduction, zero accuracy loss. MLX implementation already exists.
  NOT a threat: compresses KV cache (stored attention keys/values), we monitor 
  hidden states and logits (different tensors). Zero interference with CHI or 
  steering. Actually beneficial: extends context runway, making degradation 
  monitoring MORE valuable (model runs longer but still degrades cognitively).
  TODO (Week 8-9): test CHI accuracy with TurboQuant-compressed KV cache.
  Potential demo story: "TurboQuant extends the conversation. We keep it coherent."
- TASK-045 root cause: <|im_end|> not set as stop token in mlx_lm.generate().
  Every response generated hundreds of garbage tokens past the real end,
  poisoning context and leaking chat template tokens into subprocess commands.
  Fix: add eos_token as stop string. CHI jumped 0.790→0.952, agent completes
  calculator task in 6 turns instead of 20. All previous TASK-045 sub-issues
  (PATH, heuristic false-positive, degenerate loops) were symptoms of this.
- TASK-045 COMPLETE (9 iterations). Root causes found and fixed:
  1. <|im_end|> not stripped from tool arguments → shell syntax errors
  2. No EOS stop in generate → 800 tokens of garbage per turn
  3. Heuristic false-positive on pip install → filtered to test commands only
  4. [DONE] accepted without test evidence → validation guard added
  Agent now completes calculator task in 6 turns (was 20+ or never).
  stream_generate + EOS break partially works; truncation as backup.
  Anti-loop and no-tool guards catch degenerate model behavior.
- TASK-045 FINAL FIX (12 iterations total). Three root causes, all fixed:
  1. Generation: apply_chat_template for tokenization + stream_generate with
     manual <|im_end|> (151645) break. eos_token_ids={151643} is wrong token.
  2. Tool parsing: model emits [THINK]+[WRITE] in one response. Parser now
     extracts action tool when think+action both present.
  3. Subprocess: bare 'python' not on PATH. Regex-replace python→sys.executable.
  Agent now completes calculator task in 3 turns reliably.
  CHI stays above 0.86. No interventions needed on successful runs.
- 2026-04-02: Dupoux/LeCun/Malik paper "Why AI systems don't learn" (arXiv 2603.15381)
  proposes System A (observation) + System B (action) + System M (meta-control) architecture.
  System M monitors epistemic meta-states (prediction error, uncertainty) and switches 
  operating modes. Our CHI monitor + intervention ladder IS a practical System M for 
  coding agents. Paper validates our architecture from cognitive science angle.
  Strategic: use their vocabulary in Phase 3 materials but position as independent work.
  Don't subordinate to their framework — we built working code, they wrote theory.
  Their Evo/Devo bilevel optimization maps to our LoRA crystallization (Phase 4).
- TASK-047 COMPLETE. All 6 runs pass (calculator 3 turns, converter 3-4 turns).
  All three modes identical — steering/compaction never fire because model 
  completes before degradation. Generation fixes (chat template, EOS, 
  think+action, python normalization) eliminated most observed degradation.
  Previous "degradation" was largely garbage token generation, not cognitive 
  decline. Real degradation needs 15+ turn tasks to manifest.
  Demo needs a longer task to exercise the intervention ladder.
- REVISED DEMO NARRATIVE: "Infrastructure makes local models reliable. 
  Simple tasks complete in 3 turns. Complex tasks hit degradation — 
  the ladder catches it." Need a 15+ turn task for the second half.

## KEY LEARNINGS (condensed)

- MLX: no forward hooks → CaptureLayer wrapper. mx.eval() returns None.
- 53 tok/s on M4 Pro for 7B Q4. Inflection at turn 10 (~2500 tokens).
- Entropy is strongest degradation signal. Repetition noisy for short responses.
- Goal drift: use last-token hidden state for long sequences, not mean.
- EMA smoothing (alpha=0.3) handles noise well. Gate repetition at <20 tokens.
- Turkey problem: neural intervention more durable than wrappers but not immune.
- Qwen3.5: self-convert via mlx_lm.convert. DeltaNet hybrid attention needs testing.
- Strategy: build privately, reach Apple through MLX team + demo video.
- Layer 27 + alpha 2.0 = dead output (0 tokens). Middle layers (14-20) + moderate alpha (0.5-1.5) produce actual changed responses. Vector norms are non-uniform across layers — strongest at layer 27 but that doesn't mean best for steering.
- TASK-039 sweep results: Layer 16 is most stable for task_focus vectors.
  Best settings: L16/α0.5 (CHI 0.953) and L16/α0.25 (CHI 0.952).
  Alpha 1.0 kills output at ALL tested layers. Normalized vectors too weak.
  Raw vectors + low alpha is the path forward.
  CAVEAT: 5-turn sweep doesn't reach degradation zone. Need 10+ turn validation.
- TASK-040 results (10-turn calibrated experiment, layer 16, alpha 0.5):
  - task_focus: FAILED. Kills output at turn 5. Do not use.
  - tool_format: PASSED. All turns alive, +0.011 CHI improvement, length parity.
    Best reliable vector. Default for agent integration.
  - code_coherence: STRONG but fragile. +0.020 CHI improvement, dies turn 9.
    Try alpha 0.25 to stabilize. Highest potential if over-steering is fixed.
  - Baseline degrades 0.87→0.71 (turns 6-9) via repetition collapse.
    tool_format resists this collapse. code_coherence resists it longer.
- TASK-041 COMPLETE. CHI-triggered steering wired into agent loop.
  tool_format at layer 16, alpha 0.5. Activation at CHI<0.7, deactivation at CHI>0.8.
  Session save/resume preserves steering state.
- FINDING: Steering alone (Level 2) does NOT reverse degradation in agent loop.
  CHI continues to decline (0.67→0.53) even with steering active.
  Cause 1: alpha 0.5 too weak to overcome compounding degenerate context.
  Cause 2: model enters [THINK] loop — never uses tools — steering can't fix this.
- IMPLICATION: Level 3 (compaction) must fire when steering fails.
  Current compaction threshold (CHI<0.5) should trigger but didn't in tests.
  Check: is compaction wired to actually fire in agent.py? If not, that's TASK-044.
- Run 1 (test_agent.py) worked well: CHI recovered 0.698→0.811, files created,
  tests run. The existing prompt + tool protocol from TASK-027 matters more than
  steering for basic task completion.
- TASK-042 COMPLETE. Adaptive alpha working: alpha ramps 0.30→0.75 as CHI drops.
  Level 3 compaction path wired but untested (CHI bottomed at 0.540, threshold 0.5).
- CONFIRMED: Steering alone (fixed or adaptive) cannot recover a model stuck in
  degenerate [THINK] loop. 3 consecutive runs show same pattern.
- Steering's real value: preventing degradation when model is still functional,
  not recovering from deep failure. Level 3 (compaction) is the recovery mechanism.
- CHI calibration issue: shorter/emptier responses inflate CHI artificially.
  CHI "recovery" from 0.540→0.632 was model producing less, not producing better.
- TASK-044 (Week 7 capstone harness): `scripts/test_steering_agent.py` compares three
  modes on the calculator task — `baseline` (no steering, no compaction),
  `steering` (Level 2 + adaptive alpha, no compaction), `full_ladder` (steering +
  Level 3 compaction + Level 4 log-only line). Per-turn JSON logs go to
  `data/task044_{mode}.json`; after all three runs, use `--plot-only` or
  `--plot-comparison` to build `data/task044_comparison.png` and
  `data/task044_report.md`. **Outcome:** run the three commands and record results
  in the report — the harness is the deliverable; measured success is model-dependent.
- 2026-04-04: Previous "degradation" was mostly garbage token generation
  (no EOS stop), not genuine cognitive decline. With proper chat template
  + EOS stopping, 7B model completes 2-file tasks in 3-4 turns with no
  degradation. Real degradation needs 15+ turn tasks or test-debug cycles.
- 2026-04-04: eos_token_ids={151643} (<|endoftext|>) but model emits
  151645 (<|im_end|>). Must check both in stream_generate loop.
- TASK-047B: Full expense tracker (4 files) exceeds 7B debugging capability.
  Model creates files correctly but can't debug failing tests.
  Simplified to 2-file version (models.py + test_models.py, no CLI, no JSON I/O).
  Need to verify pytest error output reaches model context (plumbing check).
  If simplified task also fails → add structured error-feedback injection.
  Demo story: "Simple tasks = 3 turns. Complex tasks hit degradation = ladder activates."
  Real full-ladder demo likely needs Qwen3.5-9B (TASK-048).
- TASK-047B diagnostic: THREE bugs found, all fixed.
  1. stdout[:1000] truncated 83% of pytest output → raised to 3000 + --tb=short
  2. [DONE-REJECTED] dropped assistant turn → context corruption → 10x DONE loop
  3. Actual test bug: shared expenses.json between tests (model must debug this)
  Previous "model can't debug" conclusion was INVALID — plumbing failures masked
  model capability. Re-running full expense tracker before simplifying.
- TASK-047B v9: All plumbing fixes working. Model reasons correctly about 
  shared-file test contamination but gets stuck in 6+ consecutive [THINK] turns.
  NEW FAILURE MODE: think-loop paralysis, not context degradation.
  CHI bottoms at 0.675 (never reaches compaction threshold 0.65).
  Model CAN debug — won't commit to writing the fix.
- IMPLICATION: Our CHI thresholds were tuned for degradation scenarios that 
  7B coding model doesn't actually exhibit. Real failure mode is behavioral, 
  not cognitive. Think-loop happens in CHI 0.65-0.75 range.
- Part C is now more interesting: tests whether tool_format steering can 
  break think-loop paralysis (categorical behavioral change), not whether 
  it prevents gradual CHI decline (marginal improvement). Better demo story.
- ADD: [THINK]-loop specific guard (3 consecutive think calls → system 
  injection "write the fix now"). Run Part C before deciding on 14B pivot.