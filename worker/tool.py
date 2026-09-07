from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict  # JSON schema-ish; agent uses it to format prompts
    func: Callable[..., Any]

    def call(self, **kwargs) -> Any:
        return self.func(**kwargs)


def filter_tools(tools: list[Tool], allowed: list[str] | None) -> list[Tool]:
    """Return tools filtered by `allowed` names, preserving the order of `tools`.
    `None` means no filter (all allowed)."""
    if allowed is None:
        return list(tools)
    allowed_set = set(allowed)
    return [t for t in tools if t.name in allowed_set]
