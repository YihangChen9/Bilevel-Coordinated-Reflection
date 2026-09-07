from env.base import Environment
from worker.tool import Tool


def make_placeholder_tools(env: Environment) -> list[Tool]:
    def _unimplemented(**kwargs):
        raise NotImplementedError("placeholder tool; not implemented in milestone 1")

    return [
        Tool(
            name="uv_install",
            description="(placeholder) Install a Python package via uv pip.",
            input_schema={
                "type": "object",
                "properties": {"package": {"type": "string", "description": "Package spec."}},
                "required": ["package"],
                "additionalProperties": False,
            },
            func=_unimplemented,
        ),
        Tool(
            name="create_subagent",
            description="(placeholder) Spawn a sub-agent for a nested task.",
            input_schema={
                "type": "object",
                "properties": {"task": {"type": "string", "description": "Sub-task description."}},
                "required": ["task"],
                "additionalProperties": False,
            },
            func=_unimplemented,
        ),
    ]
