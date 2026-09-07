from memory.base import MemoryPort
from worker.tool import Tool


def make_workboard_tools(workboard: MemoryPort) -> list[Tool]:
    def _read_workboard() -> str:
        return workboard.read()

    def _edit_workboard(section: str, content: str) -> str:
        workboard.edit(section, content)
        return f"edited section '{section}'"

    return [
        Tool(
            name="read_workboard",
            description="Read the full shared workboard markdown.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            func=_read_workboard,
        ),
        Tool(
            name="edit_workboard",
            description="Replace or create an H2 section in the shared workboard.",
            input_schema={
                "type": "object",
                "properties": {
                    "section": {"type": "string", "description": "H2 section name to create or replace."},
                    "content": {"type": "string", "description": "New section body."},
                },
                "required": ["section", "content"],
                "additionalProperties": False,
            },
            func=_edit_workboard,
        ),
    ]
