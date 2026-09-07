from env.base import Environment
from worker.tool import Tool


def make_shell_tools(env: Environment) -> list[Tool]:
    def _run_shell(cmd: str, timeout: int = 30) -> dict:
        r = env.run_shell(cmd, timeout=timeout)
        return {"stdout": r.stdout, "stderr": r.stderr, "exit_code": r.exit_code}

    return [
        Tool(
            name="run_shell",
            description="Run a shell command in the workspace root. Returns stdout, stderr, exit_code.",
            input_schema={
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Shell command to run in workspace root."},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 30).", "default": 30, "minimum": 1, "maximum": 600},
                },
                "required": ["cmd"],
                "additionalProperties": False,
            },
            func=_run_shell,
        )
    ]
