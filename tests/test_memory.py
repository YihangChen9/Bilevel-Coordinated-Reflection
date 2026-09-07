from pathlib import Path
from memory.markdown import MarkdownMemory


def test_append_creates_file(tmp_path: Path):
    mem = MarkdownMemory(tmp_path / "m.md")
    mem.append("Notes", "hello")
    text = mem.read()
    assert "## Notes" in text
    assert "hello" in text


def test_append_preserves_order(tmp_path: Path):
    mem = MarkdownMemory(tmp_path / "m.md")
    mem.append("A", "one")
    mem.append("B", "two")
    text = mem.read()
    assert text.index("## A") < text.index("## B")


def test_edit_replaces_section(tmp_path: Path):
    mem = MarkdownMemory(tmp_path / "m.md")
    mem.append("A", "first")
    mem.edit("A", "replaced")
    text = mem.read()
    assert "replaced" in text
    assert "first" not in text


def test_clear_removes_content(tmp_path: Path):
    mem = MarkdownMemory(tmp_path / "m.md")
    mem.append("A", "x")
    mem.clear()
    assert mem.read() == ""


def test_persistent_flag_skips_clear(tmp_path: Path):
    mem = MarkdownMemory(tmp_path / "m.md", persistent=True)
    mem.append("A", "x")
    mem.clear()  # no-op when persistent
    assert "x" in mem.read()
