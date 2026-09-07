"""Orchestrator package.

`OrchestratorAgent` is loaded lazily so importing lightweight modules
(e.g. ``orchestrator.parallel_blackboard`` from the MCP worker subprocess)
does not require LangChain.
"""

from __future__ import annotations

__all__ = ["OrchestratorAgent"]


def __getattr__(name: str):
    if name == "OrchestratorAgent":
        from .orchestrator_agent import OrchestratorAgent

        return OrchestratorAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
