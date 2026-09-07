import shlex

from env.base import Environment
from worker.tool import Tool


def make_fs_tools(env: Environment) -> list[Tool]:
    def _read_file(path: str) -> str:
        return env.fs_read(path)

    def _write_file(path: str, content: str) -> str:
        env.fs_write(path, content)
        return f"wrote {len(content)} chars to {path}"

    return [
        Tool(
            name="read_file",
            description="Read a text file relative to workspace root.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path relative to workspace root."}},
                "required": ["path"],
                "additionalProperties": False,
            },
            func=_read_file,
        ),
        Tool(
            name="write_file",
            description="Overwrite a text file relative to workspace root.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path relative to workspace root."}, "content": {"type": "string", "description": "Full file contents (overwrites if exists)."}},
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            func=_write_file,
        ),
    ]


def make_fs_tools_medium(env: Environment) -> list[Tool]:
    """Extras for MEDIUM: edit_file, list_dir."""

    def _edit_file(path: str, old: str, new: str) -> str:
        text = env.fs_read(path)
        if old not in text:
            raise ValueError(f"substring not found in {path}")
        env.fs_write(path, text.replace(old, new, 1))
        return f"edited {path}"

    def _list_dir(path: str = ".") -> list[str]:
        r = env.run_shell(f"ls -1 {shlex.quote(path)}")
        return r.stdout.splitlines()

    return [
        Tool(
            name="edit_file",
            description="Replace the first occurrence of `old` with `new` in `path`.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace root."},
                    "old": {"type": "string", "description": "Substring to replace (must be present exactly once or more)."},
                    "new": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "old", "new"],
                "additionalProperties": False,
            },
            func=_edit_file,
        ),
        Tool(
            name="list_dir",
            description="List entries in a directory relative to workspace root.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Dir path.", "default": "."}},
                "additionalProperties": False,
            },
            func=_list_dir,
        ),
    ]
