"""Unit tests for session persistence, working memory, and anchors."""

from cognition_engine.anchors import AnchorManager
from cognition_engine.memory import WorkingMemory
from cognition_engine.session import SessionManager


def test_session_roundtrip_and_resume(tmp_path):
    mgr = SessionManager(str(tmp_path))
    assert mgr.load_session() is None
    assert not mgr.has_interrupted_session()

    mgr.save_session(
        "build a calculator",
        [{"role": "assistant", "content": "[WRITE] calc.py"}],
        [0.9, 0.8],
        turn_count=2,
        files_modified=["calc.py"],
        steering_active=True,
    )
    state = mgr.load_session()
    assert state.task == "build a calculator"
    assert state.chi_final == 0.8
    assert state.steering_active is True
    assert mgr.has_interrupted_session()
    assert "build a calculator" in mgr.get_resume_context()

    mgr.clear_session()
    assert mgr.load_session() is None


def test_completed_session_is_not_resumable(tmp_path):
    mgr = SessionManager(str(tmp_path))
    mgr.save_session("t", [], [], 1, [], status="completed")
    assert not mgr.has_interrupted_session()
    assert mgr.get_resume_context() is None


def test_corrupt_session_file_is_ignored(tmp_path):
    mgr = SessionManager(str(tmp_path))
    (tmp_path / ".cognition" / "session.json").write_text("{not json")
    assert mgr.load_session() is None


def test_memory_persists_across_instances(tmp_path):
    mem = WorkingMemory(str(tmp_path))
    mem.add_note("decision", "use sqlite", related_files=["db.py"])
    again = WorkingMemory(str(tmp_path))
    summary = again.get_session_summary()
    assert summary["total_notes"] == 1
    assert "db.py" in summary["files_referenced"]


def test_anchor_levels_get_shorter():
    a = AnchorManager("build a calculator")
    a.add_decision("use argparse")
    a.add_file("calc.py")
    full, mini, tiny = (a.generate_anchor(x) for x in ("FULL", "MINI", "ANCHOR"))
    assert len(full) > len(mini) > len(tiny)
    assert "build a calculator" in tiny
