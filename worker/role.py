from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Role:
    name: str
    system_prompt: str
    allowed_tools: list[str] | None = None  # None = all tools permitted


@runtime_checkable
class RoleSelector(Protocol):
    def select(self, task: str, ctx: "AgentContext") -> Role: ...  # noqa: F821


DEFAULT_ROLE = Role(
    name="default",
    system_prompt=(
        "You are a careful autonomous worker. You have a set of tools. "
        "Plan in short steps, call one tool at a time, observe the result, "
        "then continue. When the task is complete, respond with a final answer "
        "and stop calling tools."
    ),
    allowed_tools=None,
)
