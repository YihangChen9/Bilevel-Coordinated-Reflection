"""Predefined toolset packages. Callers pass a factory to Worker via the
`tools` parameter. Workboard tools are *not* here — they are injected by
Worker.run() regardless of toolset."""
from typing import Callable

from env.base import Environment
from worker.tool import Tool
from worker.tools.fs import make_fs_tools, make_fs_tools_medium
from worker.tools.shell import make_shell_tools
from worker.tools.web import make_web_tools
from worker.tools.placeholders import make_placeholder_tools


ToolsetFactory = Callable[[Environment], list[Tool]]


def MINIMAL(env: Environment) -> list[Tool]:
    return [
        *make_fs_tools(env),
        *make_shell_tools(env),
        *make_web_tools(env),
    ]


def MEDIUM(env: Environment) -> list[Tool]:
    return [
        *MINIMAL(env),
        *make_fs_tools_medium(env),
    ]


def FULL(env: Environment) -> list[Tool]:
    return [
        *MEDIUM(env),
        *make_placeholder_tools(env),
    ]
