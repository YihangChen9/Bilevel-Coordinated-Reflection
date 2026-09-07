import subprocess
import urllib.request
from pathlib import Path

from env.base import Environment, ShellResult


class LocalEnv:
    """Placeholder LocalEnv for milestone-1 smoke testing.
    To be replaced with user-provided scenario integrations."""

    def __init__(self, root: Path | str = "."):
        self.root = Path(root).resolve()

    def _resolve(self, path: str) -> Path:
        p = (self.root / path).resolve()
        return p

    def fs_read(self, path: str) -> str:
        return self._resolve(path).read_text(encoding="utf-8")

    def fs_write(self, path: str, content: str) -> None:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def run_shell(self, cmd: str, timeout: int = 30) -> ShellResult:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            cwd=self.root, timeout=timeout,
        )
        return ShellResult(stdout=proc.stdout, stderr=proc.stderr, exit_code=proc.returncode)

    def http_get(self, url: str, headers: dict | None = None) -> str:
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
