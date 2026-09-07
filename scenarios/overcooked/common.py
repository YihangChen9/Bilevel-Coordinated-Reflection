"""Shared types for the Overcooked scenario. No JAX dependency."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class CellType(IntEnum):
    EMPTY = 0
    WALL = 1
    GOAL = 2
    POT = 3
    PLATE_PILE = 4
    INGREDIENT_PILE = 5


class Action(IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3
    STAY = 4
    INTERACT = 5


DIRECTION_DELTAS = {
    Action.RIGHT: (1, 0),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.UP: (0, -1),
    Action.STAY: (0, 0),
}


@dataclass
class Position:
    x: int
    y: int

    def __eq__(self, other):
        return isinstance(other, Position) and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


@dataclass
class PotState:
    pos: Position
    ingredients: int = 0
    cooking_timer: int = -1
    cooked: bool = False
    cooking: bool = False  # explicit flag from env (don't derive from timer)

    @property
    def is_idle(self) -> bool:
        return self.ingredients == 0 and not self.cooked and not self.cooking

    @property
    def is_cooking(self) -> bool:
        return self.cooking

    @property
    def is_ready(self) -> bool:
        return self.cooked


@dataclass
class AgentState:
    agent_id: int
    pos: Position
    direction: Action = Action.DOWN
    holding: str = ""


@dataclass
class GameState:
    """Adapter-agnostic game state snapshot used by renderer/planner/tools."""
    width: int
    height: int
    grid: list[list[CellType]]
    agents: list[AgentState]
    pots: list[PotState]
    recipe: list[int]
    score: int = 0
    step: int = 0
    max_steps: int = 400
    done: bool = False
    ingredient_positions: list[Position] = field(default_factory=list)
    plate_pile_positions: list[Position] = field(default_factory=list)
    goal_positions: list[Position] = field(default_factory=list)


@dataclass
class MacroCommand:
    """A high-level command issued by a worker LLM."""
    verb: str
    target: str = ""

    @classmethod
    def parse(cls, cmd_str: str) -> "MacroCommand":
        parts = cmd_str.strip().split(maxsplit=1)
        return cls(verb=parts[0], target=parts[1] if len(parts) > 1 else "")


@dataclass
class StepEvents:
    delivery: bool = False
    score_delta: float = 0.0
    stuck_agents: list[int] = field(default_factory=list)
