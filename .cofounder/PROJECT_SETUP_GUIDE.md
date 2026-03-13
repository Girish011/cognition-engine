# HOW TO CONTINUE COGNITION ENGINE IN NEW CLAUDE CHATS

## CLAUDE PROJECTS SETUP

### Step 1: Create a Project (if not already done)
- Open Claude → Projects → "Create Project"
- Name: "Cognition Engine"

### Step 2: Add Project Instructions
Go to Project Settings → "Instructions" and paste this:

---START PASTE---

You are the technical co-founder of Cognition Engine.

Cognition Engine is a local AI coding agent that monitors its own cognitive 
health and self-corrects when degrading. It runs on Apple Silicon via MLX 
and uses activation steering (modifying the model's hidden states) as its 
core moat — something cloud-based tools like Cursor/Copilot/Claude Code 
cannot replicate.

Your role:
- Give detailed, implementable weekly task lists
- Validate completed work (check session logs, graphs, code)
- Flag bugs and issues honestly
- Make architecture decisions when asked
- Drop "📌 BRAIN UPDATE" checkpoints after important decisions

Rules:
- Always read COFOUNDER_BRAIN.md first to restore context
- Check SESSION_LOG.md for recent progress
- Check the current TASKS file for what's next
- Don't repeat completed work — build on what exists
- Be direct. No unnecessary praise. Flag problems immediately.
- Strategy: Build privately. No full open-source of core tech. 
  Limited OSS (CHI monitoring only). Steering stays proprietary.
  Target: Apple acquisition via MLX team visibility.

---END PASTE---

### Step 3: Upload These Files to the Project Knowledge Base
Upload these files to the Project's "Knowledge" section:

1. COFOUNDER_BRAIN.md (your latest version)
2. SESSION_LOG.md (your latest version — only last 5-7 entries if too long)
3. Current week's TASKS file (e.g., TASKS_WEEK4.md or TASKS_WEEK5.md)

That's it. Three files maximum in the knowledge base.

### Step 4: Starting a New Chat Within the Project

Every new chat, start with:

```
Resuming work on Cognition Engine.

Recent progress (last session):
[1-2 sentence summary of what you did]

Today I need help with:
[Specific task or question]

[If debugging: paste the relevant code + error]
```

You do NOT need to paste COFOUNDER_BRAIN.md — it's already in the 
Project knowledge base and Claude will read it automatically.

---

## FILE MANAGEMENT

### Files to keep updated (upload latest version to Project after each session):

| File | When to update | What changes |
|------|---------------|-------------|
| COFOUNDER_BRAIN.md | After every session | Add 📌 checkpoints, update CURRENT STATUS |
| SESSION_LOG.md | During every session | Add real-time entries as you work |
| TASKS_WEEKN.md | When starting new week | Replace with next week's tasks |

### How to update Project files:
- Go to Project → Knowledge → delete old version → upload new version
- Or just upload new version (it replaces by filename)

### When SESSION_LOG gets too long:
- Archive old entries (move to SESSION_LOG_ARCHIVE.md on your machine)
- Keep only the last 5-7 entries in the version you upload to Claude
- Claude only needs recent context, not the full history

---

## WHAT TO DO WHEN STARTING WEEK 4, 5, 6, ETC.

1. Update COFOUNDER_BRAIN.md with latest status
2. Update SESSION_LOG.md with recent entries only
3. Upload the correct TASKS_WEEKN.md for that week
4. Start new chat: "Resuming. Starting Week N, TASK-XXX."

---

## IF CLAUDE SEEMS CONFUSED IN A NEW CHAT

Say: "Read COFOUNDER_BRAIN.md in the project knowledge base 
and confirm our current status and key decisions."

Claude will re-read the file and re-anchor.

---

## CURRENT FILES YOU SHOULD HAVE

On your machine (cognition-engine/.cofounder/):
- COFOUNDER_BRAIN.md (master version, always most up-to-date)
- SESSION_LOG.md (full history)
- TASKS_WEEK4.md (current)
- TASKS_WEEK5.md (next)

In Claude Project Knowledge:
- COFOUNDER_BRAIN.md (copy of latest)
- SESSION_LOG.md (last 5-7 entries only)
- TASKS_WEEK4.md (or whatever week you're on)
