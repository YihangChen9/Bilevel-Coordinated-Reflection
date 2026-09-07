import re
import threading
from pathlib import Path


class MarkdownMemory:
    """Markdown-backed MemoryPort. Sections are H2 headers (`## <name>`).
    Thread-safe via a per-instance lock."""

    def __init__(self, path: Path, persistent: bool = False):
        self.path = Path(path)
        self.persistent = persistent
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def read(self) -> str:
        with self._lock:
            return self.path.read_text(encoding="utf-8")

    def append(self, section: str, content: str) -> None:
        block = f"## {section}\n{content.rstrip()}\n\n"
        with self._lock:
            existing = self.path.read_text(encoding="utf-8")
            self.path.write_text(existing + block, encoding="utf-8")

    def edit(self, section: str, new_content: str) -> None:
        pattern = re.compile(
            rf"(^## {re.escape(section)}\n)(.*?)(?=^## |\Z)",
            re.DOTALL | re.MULTILINE,
        )
        with self._lock:
            text = self.path.read_text(encoding="utf-8")
            replacement = f"## {section}\n{new_content.rstrip()}\n\n"
            new_text, n = pattern.subn(replacement, text)
            if n == 0:
                # section absent — append
                new_text = text + replacement
            self.path.write_text(new_text, encoding="utf-8")

    def clear(self) -> None:
        if self.persistent:
            return
        with self._lock:
            self.path.write_text("", encoding="utf-8")

    def overwrite(self, text: str) -> None:
        """Replace the entire file contents with `text`. Used for snapshot/restore
        rollback (e.g. gated-ascent write rejection). Bypasses `persistent` since
        callers explicitly provide the text to restore."""
        with self._lock:
            self.path.write_text(text, encoding="utf-8")
