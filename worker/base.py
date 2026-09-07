from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from worker.tool import Tool


@dataclass
class AgentContext:
    """Runtime context handed to an agent. Extensible without breaking signature."""
    role_name: str
    system_prompt: str
    workspace_hint: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    final_answer: str
    steps: list[dict]        # list of {thought, action, observation}
    stopped_reason: str      # "final" | "max_steps" | "error"


@runtime_checkable
class BaseAgent(Protocol):
    def run(self, task: str, tools: list[Tool], ctx: AgentContext) -> AgentResult: ...
