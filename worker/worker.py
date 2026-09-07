from typing import Callable, Union

from env.base import Environment
from memory.base import MemoryPort
from worker.base import AgentContext, AgentResult, BaseAgent
from worker.role import Role, RoleSelector
from worker.scratch import make_scratch_tools
from worker.tool import Tool, filter_tools
from worker.tools.workboard import make_workboard_tools


ToolsetFactory = Callable[[Environment], list[Tool]]


class Worker:
    def __init__(
        self,
        agent: BaseAgent,
        role: Union[Role, RoleSelector],
        tools: ToolsetFactory,
        env: Environment,
        workboard: MemoryPort,
        agent_memory: MemoryPort | None = None,
    ):
        self.agent = agent
        self.role = role
        self.tools_factory = tools
        self.env = env
        self.workboard = workboard
        self.agent_memory = agent_memory

    def _resolve_role(self, task: str, ctx: AgentContext) -> Role:
        if isinstance(self.role, Role):
            return self.role
        return self.role.select(task, ctx)

    def run(self, task: str) -> AgentResult:
        # Preliminary ctx (role_name filled in after selector resolves)
        provisional_ctx = AgentContext(role_name="?", system_prompt="")
        role = self._resolve_role(task, provisional_ctx)
        ctx = AgentContext(role_name=role.name, system_prompt=role.system_prompt)

        base_tools = list(self.tools_factory(self.env))
        workboard_tools = make_workboard_tools(self.workboard)
        scratch_tools = make_scratch_tools(self.agent_memory) if self.agent_memory is not None else []
        # Workboard tools are ALWAYS present, regardless of the toolset factory.
        all_tools = base_tools + workboard_tools + scratch_tools
        # Role filter applies to every tool — but workboard tools are visible by default
        # unless the role explicitly enumerates allowed_tools AND omits them.
        effective = filter_tools(all_tools, role.allowed_tools)

        return self.agent.run(task, effective, ctx)
