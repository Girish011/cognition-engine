# START HERE
# Follow these steps in order. Don't skip.

---

## STEP 1: Create the Project (5 min)

```bash
mkdir cognition-engine
cd cognition-engine
git init
mkdir -p .cofounder engine scripts tests
touch engine/__init__.py tests/__init__.py
```

## STEP 2: Add the Three Files (2 min)

Put these three files into `.cofounder/`:
- `COFOUNDER_BRAIN.md` — Claude's memory (downloaded from this chat)
- `SESSION_LOG.md` — your work diary (downloaded from this chat)
- `TASKS.md` — your task list (downloaded from this chat)

Your project should look like:
```
cognition-engine/
├── .cofounder/
│   ├── COFOUNDER_BRAIN.md
│   ├── SESSION_LOG.md
│   └── TASKS.md
├── engine/
│   └── __init__.py
├── scripts/
└── tests/
    └── __init__.py
```

## STEP 3: Open in Cursor (1 min)

```bash
cursor .
```

Arrange your workspace:
- **Left sidebar:** file tree visible
- **One tab:** `.cofounder/TASKS.md` (your current task)
- **One tab:** `.cofounder/SESSION_LOG.md` (write here as you work)
- **Main editor:** whatever code file you're working on

## STEP 4: Start TASK-001

Open TASKS.md. Do TASK-001. Follow the instructions exactly.
When the test passes, check it off, write a line in SESSION_LOG, commit, move on.

---

## DAILY WORKFLOW

```
1. Open Cursor
2. Open TASKS.md → find your current task
3. Implement it (use Cursor AI for syntax help)
4. Run the manual test described in the task
5. Pass? → check off, write a line in SESSION_LOG, git commit
6. Stuck? → ask Claude (see below)
7. Next task
```

## HOW TO ASK CLAUDE WHEN STUCK

Open a new Claude conversation. Copy-paste this:

```
I'm building Cognition Engine. You're my technical co-founder.

[Paste entire COFOUNDER_BRAIN.md]

Recent work:
[Paste last 3-5 entries from SESSION_LOG.md]

I'm stuck on TASK-XXX: [task name].
Here's my code:
[Paste the relevant file]

Here's the error:
[Paste error message]

What I've tried:
[What you attempted]
```

## HOW CLAUDE'S MEMORY WORKS (Crash-Resilient)

Claude will drop checkpoints like this during conversations:

> 📌 BRAIN UPDATE:
> Under "KEY LEARNINGS" add:
> - 2026-03-05: MLX hidden states accessed via model.model.layers[N]

**When you see this → immediately copy it into COFOUNDER_BRAIN.md and save.**

Don't wait for the session to end. Sessions can crash without warning.
If you save updates as they come, you lose at most 10 minutes if a session dies.

## CURSOR vs CLAUDE — WHEN TO USE WHICH

| Situation | Use |
|-----------|-----|
| Writing Python code, syntax help | **Cursor AI** |
| "How do I do X in MLX?" | **Cursor AI** |
| "Should we use approach A or B?" | **Claude chat** |
| Stuck on a task for 30+ minutes | **Claude chat** |
| Architecture decisions | **Claude chat** |
| Bug you can't figure out | **Claude chat** (with BRAIN file) |

## RULES

1. **Don't skip tasks.** Each one builds on the previous.
2. **Test before moving on.** Each task has a specific manual test.
3. **Log in real-time.** Write SESSION_LOG as you work, not end of day.
4. **Struggle first, ask second.** Try 30 min before asking Claude.
5. **Commit after each task.** `git add -A && git commit -m "TASK-00X: description"`
6. **Update BRAIN immediately** when Claude drops a 📌 checkpoint.

---

**Ready? Open TASKS.md. Start TASK-001. It takes 15 minutes.**
