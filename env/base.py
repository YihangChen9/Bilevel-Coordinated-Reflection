from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ShellResult:
    stdout: str
    stderr: str
    exit_code: int


@runtime_checkable
class Environment(Protocol):
    def fs_read(self, path: str) -> str: ...
    def fs_write(self, path: str, content: str) -> None: ...
    def run_shell(self, cmd: str, timeout: int = 30) -> ShellResult: ...
    def http_get(self, url: str, headers: dict | None = None) -> str: ...
