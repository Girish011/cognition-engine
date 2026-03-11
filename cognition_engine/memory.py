"""
Working Memory — Zettelkasten-style project notes (TASK-026).

The agent automatically writes notes as it works:
- Architectural decisions
- File purposes
- Key patterns used
- Bugs encountered and fixed

Notes are stored in .cognition/ in the project directory.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import List, Dict, Any


class WorkingMemory:
    """Manages persistent project memory in .cognition/ directory."""

    def __init__(self, project_dir: str):
        self.memory_dir = os.path.join(project_dir, ".cognition")
        os.makedirs(self.memory_dir, exist_ok=True)
        self.notes_file = os.path.join(self.memory_dir, "notes.json")
        self.notes: List[Dict[str, Any]] = self._load_notes()

    def _load_notes(self) -> List[Dict[str, Any]]:
        """Load existing notes from disk."""
        if os.path.exists(self.notes_file):
            try:
                with open(self.notes_file) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
            except Exception:  # noqa: BLE001
                # Corrupt or unexpected format; start fresh but don't crash the agent.
                return []
        return []

    def add_note(
        self,
        category: str,
        content: str,
        related_files: List[str] | None = None,
    ) -> Dict[str, Any]:
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

    def get_context_for_task(
        self, task_description: str, max_notes: int = 10
    ) -> str:
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

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current memory state."""
        categories: Dict[str, int] = {}
        for note in self.notes:
            cat = note.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        files_referenced = {
            f
            for n in self.notes
            for f in n.get("related_files", [])
            if isinstance(f, str)
        }
        return {
            "total_notes": len(self.notes),
            "categories": categories,
            "files_referenced": sorted(files_referenced),
        }

    def _save(self) -> None:
        """Save notes to disk."""
        with open(self.notes_file, "w") as f:
            json.dump(self.notes, f, indent=2)


if __name__ == "__main__":
    # Manual test scaffold mirroring TASK-026 spec.
    memory = WorkingMemory("/tmp/test_project")

    memory.add_note(
        "decision",
        "Using SQLite instead of PostgreSQL for simplicity",
        related_files=["db.py"],
    )
    memory.add_note(
        "pattern",
        "All API routes follow /api/v1/{resource} convention",
        related_files=["routes.py", "app.py"],
    )
    memory.add_note(
        "bug",
        "Fixed: forgot to close DB connection in get_users()",
        related_files=["db.py"],
    )

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

    print("WorkingMemory manual scaffold passed.")

