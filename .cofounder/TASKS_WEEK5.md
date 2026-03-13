# WEEK 5 TASKS: Agent Polish + Qwen3.5 Evaluation + Steering Prep
# Final week of Phase 1. After this, 100% focus on activation steering.
# Also: evaluate Qwen3.5-9B and start engaging with MLX community (Path A).

---

## WEEK 5: Agent Polish + Model Upgrade + Steering Prep
**Goal:** A robust agent that can serve as the test bed for activation steering.
Switch to Qwen3.5 if it's ready. Begin Path A (MLX community visibility).

---

### TASK-027: Fix Agent Issues from Week 4
**Status:** ⬜ TODO
**Do:** Update `cognition_engine/agent.py` and `cognition_engine/prompts.py`

Based on whatever broke in TASK-025, fix the most critical issues.
Common problems you'll likely see:

1. **Model doesn't follow tool format:** It generates plain text instead
   of [READ]/[WRITE]/[EXEC] markers. Fix: improve system prompt with
   few-shot examples. Add 2-3 example tool calls in the system prompt
   so the model sees the format before generating.

2. **Model puts multiple tool calls in one response:** Fix: after parsing
   the first tool call, truncate the response. Tell the model "one tool
   per response" more firmly in the prompt.

3. **Model hallucinates file contents instead of reading:** Fix: add to
   system prompt: "NEVER guess file contents. Always [READ] first."

4. **Parsing is fragile:** Fix: make the parser more tolerant. Handle
   variations like `[Read]`, `[READ]:`, `[READ] - path`, etc.

The goal isn't perfection — it's "works well enough for a 10-turn demo."

**Manual test:** Run the calculator task from TASK-025 again. The agent
should complete it in fewer turns with fewer parsing failures.

**Time:** 3-4 hours
**Git:** `git commit -m "TASK-027: agent loop fixes from integration testing"`

---

### TASK-028: Integrate Working Memory into Agent
**Status:** ⬜ TODO
**Do:** Wire `WorkingMemory` into `CognitionAgent`

The agent should:
1. Load existing notes at session start (inject into context)
2. Auto-save notes when it makes decisions (after [THINK] tool calls)
3. Auto-save file purposes when it creates files
4. Save a session summary when it exits

```python
# In CognitionAgent.__init__, add:
self.memory = WorkingMemory(working_dir)

# At session start, load memory into context:
memory_context = self.memory.get_context_for_task(task)
self.compactor.add_turn("system", memory_context)

# After a [THINK] tool call:
self.memory.add_note("decision", thought, related_files=[...])

# After a [WRITE] tool call:
self.memory.add_note("file_purpose", f"Created {path}: {content[:100]}",
                     related_files=[path])

# On exit:
self.memory.add_note("context", f"Session ended. {self.turn_count} turns. "
                     f"Final CHI: {self.chi_monitor.chi_history[-1]:.3f}. "
                     f"Files: {self.anchor_mgr.files_modified}")
```

**Manual test:**
1. Run agent on calculator task → it creates .cognition/notes.json
2. Kill the agent mid-task (Ctrl+C)
3. Run the agent again on "Add a square root method to Calculator"
4. Verify: the agent's context includes notes from the previous session
   (it should know calculator.py already exists)

**Time:** 2-3 hours
**Git:** `git commit -m "TASK-028: working memory integrated into agent loop"`

---

### TASK-029: Evaluate Qwen3.5 on MLX
**Status:** ⬜ TODO
**Do:** Create `scripts/eval_qwen35.py`

Time to test whether Qwen3.5 models work on your hardware and with
your codebase. This was planned for Week 4+ in the original roadmap.

```python
"""
Evaluate Qwen3.5 models on M4 Pro 24GB.

Test: 
1. Can we load and run Qwen3.5-9B and Qwen3.5-4B on MLX?
2. What's the tok/s for each?
3. Does hidden state extraction still work? (Different architecture)
4. Does CHI monitoring still work?
5. Does the agent follow tool format better than 7B?
"""
import time
from mlx_lm import load, generate

MODELS_TO_TEST = [
    # Check mlx-community for available quantized versions
    # Search: https://huggingface.co/mlx-community?search=qwen3.5
    "mlx-community/Qwen3.5-4B-Instruct-4bit",    # if available
    "mlx-community/Qwen3.5-9B-Instruct-4bit",    # if available
]

def benchmark_model(model_path: str):
    print(f"\n{'='*60}")
    print(f"Testing: {model_path}")
    print(f"{'='*60}")
    
    try:
        model, tokenizer = load(model_path)
    except Exception as e:
        print(f"FAILED to load: {e}")
        print("Model may not be available on mlx-community yet.")
        print("Search: https://huggingface.co/mlx-community")
        return None
    
    # 1. Basic generation + speed
    prompt = "Write a Python function that computes fibonacci numbers"
    start = time.time()
    response = generate(model, tokenizer, prompt=prompt, max_tokens=300)
    elapsed = time.time() - start
    tokens = len(tokenizer.encode(response))
    tps = tokens / elapsed
    print(f"Speed: {tokens} tokens in {elapsed:.1f}s = {tps:.1f} tok/s")
    print(f"Response preview: {response[:200]}...")
    
    # 2. Hidden state extraction
    try:
        from cognition_engine.hidden_states import extract_hidden_states
        text, states = extract_hidden_states(model, tokenizer,
                                              "Write hello world", 
                                              layer_idx=16, max_tokens=50)
        print(f"Hidden states: {states.shape}")
        # NOTE: Qwen3.5 has different architecture (Gated DeltaNet)
        # Layer 16 might be a DeltaNet layer, not full attention
        # Check which layers are full attention (every 4th)
    except Exception as e:
        print(f"Hidden state extraction FAILED: {e}")
        print("May need architecture-specific adaptation")
    
    # 3. Tool format compliance
    tool_prompt = """You are a coding assistant with these tools:
[READ] path — Read a file
[WRITE] path
```
content
```
[EXEC] command — Run a command
[THINK] thought — Record reasoning

Task: Create a hello.py file that prints "Hello World".
Respond with exactly one tool call."""
    
    tool_response = generate(model, tokenizer, prompt=tool_prompt, max_tokens=200)
    print(f"\nTool format test:")
    print(f"Response: {tool_response[:300]}")
    
    has_tool = any(marker in tool_response for marker in 
                   ["[READ]", "[WRITE]", "[EXEC]", "[THINK]"])
    print(f"Contains tool marker: {has_tool}")
    
    return {"model": model_path, "tps": tps, "hidden_states": True,
            "tool_format": has_tool}

if __name__ == "__main__":
    results = []
    for model_path in MODELS_TO_TEST:
        result = benchmark_model(model_path)
        if result:
            results.append(result)
    
    # Compare with baseline
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"Qwen2.5-Coder-7B baseline: 53.0 tok/s")
    for r in results:
        print(f"{r['model']}: {r['tps']:.1f} tok/s | "
              f"hidden_states: {r['hidden_states']} | "
              f"tool_format: {r['tool_format']}")
```

**What to look for:**
- If Qwen3.5-9B runs at 20+ tok/s, it's viable as production model
- If hidden state extraction breaks, note which layer types work
  (full attention every 4th block vs DeltaNet layers)
- If tool format compliance is better than 7B, switch immediately

**IMPORTANT:** The Qwen3.5 models may not be on mlx-community yet.
Search HuggingFace first. If not available, try:
```bash
# Convert yourself
pip install mlx-lm
mlx_lm.convert --model Qwen/Qwen3.5-9B-Instruct -q --upload-repo your-username/Qwen3.5-9B-Instruct-4bit
```
Or just skip this task and revisit next week. Don't block on it.

**Time:** 2-3 hours
**Git:** `git commit -m "TASK-029: Qwen3.5 evaluation on MLX"`

---

### TASK-030: Session Management (Start/Stop/Resume)
**Status:** ⬜ TODO
**Do:** Create `cognition_engine/session.py`

The agent needs to gracefully handle: starting a new session, stopping
mid-task (Ctrl+C), and resuming where it left off next time.

```python
"""
Session Management — Start, stop, resume agent sessions.

State is saved to .cognition/session.json so the agent can
pick up where it left off.
"""
import os
import json
from datetime import datetime

class SessionManager:
    """Manages agent session state for persistence across runs."""
    
    def __init__(self, project_dir: str):
        self.session_dir = os.path.join(project_dir, ".cognition")
        os.makedirs(self.session_dir, exist_ok=True)
        self.session_file = os.path.join(self.session_dir, "session.json")
    
    def save_session(self, task: str, conversation_turns: list[dict],
                     chi_history: list[float], turn_count: int,
                     files_modified: list[str], status: str = "interrupted"):
        """Save current session state to disk."""
        state = {
            "task": task,
            "status": status,  # "interrupted", "completed", "failed"
            "turn_count": turn_count,
            "timestamp": datetime.now().isoformat(),
            "chi_history": chi_history,
            "chi_final": chi_history[-1] if chi_history else None,
            "files_modified": files_modified,
            "conversation_turns": conversation_turns[-10:],  # Last 10 only
        }
        with open(self.session_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_session(self) -> dict | None:
        """Load previous session if it exists."""
        if not os.path.exists(self.session_file):
            return None
        with open(self.session_file) as f:
            return json.load(f)
    
    def has_interrupted_session(self) -> bool:
        """Check if there's an interrupted session to resume."""
        session = self.load_session()
        return session is not None and session.get("status") == "interrupted"
    
    def clear_session(self):
        """Clear session state (after completion or user choice)."""
        if os.path.exists(self.session_file):
            os.remove(self.session_file)
    
    def get_resume_context(self) -> str | None:
        """
        Build a context string for resuming an interrupted session.
        Returns None if no session to resume.
        """
        session = self.load_session()
        if not session or session["status"] != "interrupted":
            return None
        
        lines = [
            "[RESUMING PREVIOUS SESSION]",
            f"Original task: {session['task']}",
            f"Progress: {session['turn_count']} turns completed",
            f"Last CHI: {session['chi_final']:.3f}" if session['chi_final'] else "",
            f"Files modified: {', '.join(session['files_modified'])}",
            "",
            "Recent conversation:",
        ]
        for turn in session["conversation_turns"][-5:]:
            role = turn["role"].capitalize()
            content = turn["content"][:200]
            lines.append(f"  {role}: {content}")
        
        lines.append("[END RESUME CONTEXT — Continue from where you left off]")
        return "\n".join(lines)
```

Wire into the agent:
```python
# In CognitionAgent.__init__:
self.session_mgr = SessionManager(working_dir)

# Check for interrupted session
if self.session_mgr.has_interrupted_session():
    resume_ctx = self.session_mgr.get_resume_context()
    if resume_ctx:
        # Ask user if they want to resume
        # If yes, inject resume_ctx into compactor
        pass

# On Ctrl+C (in the except KeyboardInterrupt block):
self.session_mgr.save_session(
    self.task, self.compactor.conversation_turns,
    self.chi_monitor.chi_history, self.turn_count,
    self.anchor_mgr.files_modified, "interrupted"
)

# On task completion:
self.session_mgr.save_session(..., status="completed")
self.session_mgr.clear_session()
```

**Manual test:**
1. Start agent on a task
2. Let it run 3-4 turns
3. Ctrl+C to interrupt
4. Check .cognition/session.json exists with saved state
5. Start agent again on same directory
6. Verify it detects the interrupted session and offers to resume
7. Resume — verify it has context from previous run

**Time:** 2-3 hours
**Git:** `git commit -m "TASK-030: session management (save/resume interrupted sessions)"`

---

### TASK-031: CLI Upgrade — Agent Mode
**Status:** ⬜ TODO
**Do:** Update `cognition_engine/cli.py`

Add an `agent` subcommand that runs the full agent (not just chat).

```bash
# Chat mode (existing)
python -m cognition_engine chat --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit

# Agent mode (new)
python -m cognition_engine agent \
  --task "Build a REST API with Flask" \
  --dir ./my-project \
  --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
  --max-turns 30

# Resume interrupted session
python -m cognition_engine agent --dir ./my-project --resume

# Show session status
python -m cognition_engine status --dir ./my-project
```

Implementation:
```python
import argparse

def main():
    parser = argparse.ArgumentParser(description="Cognition Engine")
    subparsers = parser.add_subparsers(dest="command")
    
    # Chat mode
    chat_parser = subparsers.add_parser("chat", help="Interactive chat with CHI monitoring")
    chat_parser.add_argument("--model", default="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    chat_parser.add_argument("--max-tokens", type=int, default=500)
    
    # Agent mode
    agent_parser = subparsers.add_parser("agent", help="Run coding agent on a task")
    agent_parser.add_argument("--task", required=False, help="Task description")
    agent_parser.add_argument("--dir", default=".", help="Working directory")
    agent_parser.add_argument("--model", default="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    agent_parser.add_argument("--max-turns", type=int, default=50)
    agent_parser.add_argument("--resume", action="store_true", help="Resume interrupted session")
    agent_parser.add_argument("--auto-approve", action="store_true",
                              help="Auto-approve shell commands (dangerous)")
    
    # Status
    status_parser = subparsers.add_parser("status", help="Show project status")
    status_parser.add_argument("--dir", default=".")
    
    args = parser.parse_args()
    
    if args.command == "chat":
        run_chat(args)
    elif args.command == "agent":
        run_agent(args)
    elif args.command == "status":
        show_status(args)
    else:
        # Default: if no subcommand, run chat mode (backward compatible)
        parser.print_help()

def run_agent(args):
    from mlx_lm import load
    from cognition_engine.agent import CognitionAgent
    from cognition_engine.session import SessionManager
    
    model, tokenizer = load(args.model)
    
    if args.resume:
        session_mgr = SessionManager(args.dir)
        if session_mgr.has_interrupted_session():
            session = session_mgr.load_session()
            print(f"Resuming: {session['task'][:80]}...")
            print(f"Previous progress: {session['turn_count']} turns, "
                  f"CHI {session['chi_final']:.3f}")
            task = session['task']
        else:
            print("No interrupted session found.")
            return
    else:
        task = args.task
        if not task:
            task = input("Enter task: ")
    
    agent = CognitionAgent(model, tokenizer, task,
                           working_dir=args.dir,
                           auto_approve=args.auto_approve)
    agent.run(max_turns=args.max_turns)

def show_status(args):
    from cognition_engine.session import SessionManager
    from cognition_engine.memory import WorkingMemory
    
    session_mgr = SessionManager(args.dir)
    memory = WorkingMemory(args.dir)
    
    session = session_mgr.load_session()
    if session:
        print(f"Task: {session['task'][:80]}")
        print(f"Status: {session['status']}")
        print(f"Turns: {session['turn_count']}")
        print(f"Last CHI: {session.get('chi_final', 'N/A')}")
        print(f"Files: {', '.join(session.get('files_modified', []))}")
    else:
        print("No active session.")
    
    summary = memory.get_session_summary()
    if summary['total_notes'] > 0:
        print(f"\nMemory: {summary['total_notes']} notes")
        print(f"Categories: {summary['categories']}")
        print(f"Files referenced: {', '.join(summary['files_referenced'][:10])}")

def run_chat(args):
    # Move existing chat code here
    pass
```

**Manual test:**
```bash
# Help works
python -m cognition_engine --help
python -m cognition_engine agent --help

# Agent mode works
python -m cognition_engine agent --task "Create hello.py" --dir /tmp/test

# Status works
python -m cognition_engine status --dir /tmp/test

# Resume works after Ctrl+C
python -m cognition_engine agent --dir /tmp/test --resume
```

**Time:** 2-3 hours
**Git:** `git commit -m "TASK-031: CLI with agent mode, resume, and status"`

---

### TASK-032: Path A — First MLX Community Engagement
**Status:** ⬜ TODO
**Do:** This is NOT a code task. This is strategic visibility.

The MLX team (Awni Hannun and contributors) watches what people build.
Your goal: become a known name in the mlx-lm community before you
need anything from them.

**Actions (do all of these this week):**

1. **Star and watch** the mlx-lm repo if you haven't already.
   https://github.com/ml-explore/mlx-lm

2. **Read open issues.** Find one you can meaningfully comment on
   or contribute to. Even a small bug fix or documentation improvement
   gets your name in front of the maintainers.
   
   Good candidates:
   - Issues about hidden state access or hooks (directly relevant)
   - Issues about model support for Qwen3.5 (you'll have tested it)
   - Performance issues on Apple Silicon (you have real benchmarks)

3. **Open a Discussion** (not an Issue) on the mlx-lm repo:
   
   Title: "Hidden state extraction patterns for runtime monitoring"
   
   Content: Brief, technical, helpful. Something like:
   "I've been building a cognitive health monitoring system on MLX
   and found that the cleanest way to extract hidden states during
   inference is to wrap layers in the model.model.layers[] list with
   a capture class, since MLX doesn't have PyTorch-style forward hooks.
   
   Has anyone explored adding optional hook support to mlx.nn.Module?
   Would be useful for interpretability and monitoring use cases.
   
   Here's the pattern I'm using: [brief code snippet]"
   
   This does three things:
   - Shows you're building something real on MLX
   - Provides value (the pattern is useful to others)
   - Opens a conversation with maintainers about hooks/monitoring

4. **Follow Awni Hannun** on GitHub and X/Twitter. Engage with
   MLX-related posts. Not spammy — just genuine technical engagement.

**DO NOT:**
- Mention acquisition or selling to Apple
- Open-source your full codebase
- Share the steering implementation
- Make it sound like marketing

**DO:**
- Be genuinely helpful and technical
- Share the CaptureLayer pattern (it's not your moat)
- Ask smart questions about MLX internals
- Build a reputation as "the person building cognitive monitoring on MLX"

**Time:** 1-2 hours
**Log in SESSION_LOG:** What you posted, any responses you got.
**Git:** No code commit for this task.

---

## WEEK 5 CHECKLIST
- [ ] TASK-027: Fix agent issues from Week 4 (3-4 hrs)
- [ ] TASK-028: Working memory integration (2-3 hrs)
- [ ] TASK-029: Qwen3.5 evaluation (2-3 hrs, skip if models not on MLX yet)
- [ ] TASK-030: Session management (2-3 hrs)
- [ ] TASK-031: CLI with agent mode (2-3 hrs)
- [ ] TASK-032: MLX community engagement — Path A (1-2 hrs)

**Total: ~13-18 hours across 3-4 days**

**Week 5 milestone:** A working coding agent that:
1. Takes tasks, executes them with tools, monitors CHI
2. Persists memory and session state across runs
3. Can be interrupted and resumed
4. Has a proper CLI with subcommands
5. (If Qwen3.5 available) Benchmarked on newer model

Plus: your name appears in the mlx-lm community for the first time.

**After Week 5: Phase 2 begins. 100% activation steering.**
This is what you came here to build. Everything so far was scaffolding.

---

## NOTES

### On TASK-029 (Qwen3.5):
If the models aren't on mlx-community yet, DON'T spend hours
converting them yourself. It's not worth the time right now.
The 7B model works fine for building the agent. You can switch
models any time — the architecture is model-agnostic. Just skip
TASK-029 and come back to it when the community uploads them.

### On TASK-032 (Path A):
This feels small but it's one of the highest-leverage things
you'll do this month. Apple's MLX team is ~10 people. They
personally read their GitHub discussions. One thoughtful post
from someone building something interesting on their framework
is worth more than 1000 cold emails to Apple's M&A inbox.

### Preparing for Phase 2:
While building this week, start reading these in background:
- ITI paper (Inference-Time Intervention): https://arxiv.org/abs/2306.03341
  Focus on Section 3 — how they compute steering vectors
- EasySteer/CAST papers if you haven't already
- The engine/hidden_states.py code you built — you'll modify
  this heavily for steering vector injection

Phase 2 starts next week. You'll need to understand contrastive
pair generation and activation difference computation. The reading
now saves debugging time later.
