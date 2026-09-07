"""Per-agent scratch tools — the agent's private long-term memory.

Symmetric to `worker.tools.workboard.make_workboard_tools`, but bound to a per-
agent MemoryPort instead of the shared workboard. Workers gain `read_scratch` /
`write_scratch` tools when constructed with `agent_memory` (Task 4 wires this).
Writes use `edit` so the scratch holds one canonical 'Notes' section that the
agent rewrites in place — appending would let the scratch grow unbounded.
"""
from memory.base import MemoryPort
from worker.tool import Tool


_SCRATCH_SECTION = "Notes"


def make_scratch_tools(scratch: MemoryPort) -> list[Tool]:
    def _read_scratch() -> str:
        return scratch.read()

    def _write_scratch(content: str) -> str:
        scratch.edit(_SCRATCH_SECTION, content)
        return f"scratch updated ({len(content)} chars)"

    return [
        Tool(
            name="read_scratch",
            description="Read this agent's private long-term notes (persists across timesteps).",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            func=_read_scratch,
        ),
        Tool(
            name="write_scratch",
            description="Replace this agent's private notes with new content (overwrites prior notes).",
            input_schema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "New full content of the agent's private notes."},
                },
                "required": ["content"],
                "additionalProperties": False,
            },
            func=_write_scratch,
        ),
    ]
