"""Mock Overcooked adapter for unit tests. No JAX dependency.

5×5 grid with: 1 ingredient pile (2,1), 1 pot (3,1), 1 goal (3,2),
1 plate pile (3,3), 2 agents at (1,2) and (1,3).

Simplified mechanics:
  - No collision physics (agents can overlap)
  - cook_time configurable (default 3 for fast tests)
  - Single ingredient type (ingredient_0)
  - Recipe always [ingredient_0] × recipe_size
"""
from __future__ import annotations

from scenarios.overcooked.common import (
    Action, AgentState, CellType, DIRECTION_DELTAS,
    GameState, MacroCommand, Position, PotState, StepEvents,
)
from scenarios.overcooked.config import OvercookedConfig


_MOCK_WIDTH = 5
_MOCK_HEIGHT = 5

_MOCK_GRID = [
    [CellType.WALL] * _MOCK_WIDTH,
    [CellType.WALL, CellType.EMPTY, CellType.INGREDIENT_PILE, CellType.POT, CellType.WALL],
    [CellType.WALL, CellType.EMPTY, CellType.EMPTY, CellType.GOAL, CellType.WALL],
    [CellType.WALL, CellType.EMPTY, CellType.EMPTY, CellType.PLATE_PILE, CellType.WALL],
    [CellType.WALL] * _MOCK_WIDTH,
]

_INGREDIENT_POS = Position(2, 1)
_POT_POS = Position(3, 1)
_GOAL_POS = Position(3, 2)
_PLATE_PILE_POS = Position(3, 3)
_AGENT_SPAWNS = [Position(1, 2), Position(1, 3)]


class MockAdapter:
    """Deterministic mock for unit tests. Call reset() then step() in a loop."""

    def __init__(self, config: OvercookedConfig | None = None):
        self.config = config or OvercookedConfig(
            max_steps=50, cook_time=3, recipe_size=3, n_agents=2,
        )
        self._agents: list[AgentState] = []
        self._pots: list[PotState] = []
        self._score: int = 0
        self._step: int = 0
        self._done: bool = False
        self._position_history: dict[int, list[Position]] = {}

    def reset(self) -> GameState:
        self._agents = [
            AgentState(agent_id=i, pos=Position(_AGENT_SPAWNS[i].x, _AGENT_SPAWNS[i].y))
            for i in range(self.config.n_agents)
        ]
        self._pots = [PotState(pos=Position(_POT_POS.x, _POT_POS.y))]
        self._score = 0
        self._step = 0
        self._done = False
        self._position_history = {i: [] for i in range(self.config.n_agents)}
        return self.get_state()

    def step(self, actions: dict[int, int]) -> tuple[GameState, StepEvents]:
        """Execute one game step. actions = {agent_id: Action int}."""
        events = StepEvents()

        # Movement
        for aid, action_int in actions.items():
            a = self._agents[aid]
            action = Action(action_int)
            if action in DIRECTION_DELTAS:
                dx, dy = DIRECTION_DELTAS[action]
                nx, ny = a.pos.x + dx, a.pos.y + dy
                # Update facing direction regardless of whether movement succeeds
                if action != Action.STAY:
                    a.direction = action
                # Only move into EMPTY or GOAL cells (pots, piles, walls block movement)
                if 0 <= nx < _MOCK_WIDTH and 0 <= ny < _MOCK_HEIGHT:
                    if _MOCK_GRID[ny][nx] in (CellType.EMPTY, CellType.GOAL):
                        a.pos = Position(nx, ny)

            # Interact
            if action == Action.INTERACT:
                self._process_interact(aid, events)

            self._position_history[aid].append(Position(a.pos.x, a.pos.y))

        # Tick pot timers
        for pot in self._pots:
            if pot.is_cooking:
                pot.cooking_timer -= 1
                if pot.cooking_timer <= 0:
                    pot.cooked = True
                    pot.cooking = False
                    pot.cooking_timer = 0

        self._step += 1
        if self._step >= self.config.max_steps:
            self._done = True

        # Stuck detection
        for aid in range(self.config.n_agents):
            hist = self._position_history[aid]
            if len(hist) >= self.config.stuck_threshold:
                recent = hist[-self.config.stuck_threshold:]
                if len(set((p.x, p.y) for p in recent)) == 1:
                    events.stuck_agents.append(aid)

        return self.get_state(), events

    def _process_interact(self, aid: int, events: StepEvents):
        a = self._agents[aid]
        dx, dy = DIRECTION_DELTAS.get(a.direction, (0, 0))
        target = Position(a.pos.x + dx, a.pos.y + dy)

        # Pickup from ingredient pile
        if target == _INGREDIENT_POS and a.holding == "":
            a.holding = "ingredient_0"
            return

        # Pickup plate
        if target == _PLATE_PILE_POS and a.holding == "":
            a.holding = "plate"
            return

        # Place ingredient in pot
        for pot in self._pots:
            if target == pot.pos and a.holding == "ingredient_0":
                if pot.ingredients < self.config.recipe_size and not pot.cooked:
                    pot.ingredients += 1
                    a.holding = ""
                    if pot.ingredients >= self.config.recipe_size:
                        pot.cooking_timer = self.config.cook_time
                        pot.cooking = True
                return

        # Pickup cooked dish from pot with plate
        for pot in self._pots:
            if target == pot.pos and a.holding == "plate" and pot.is_ready:
                a.holding = "dish"
                pot.ingredients = 0
                pot.cooked = False
                pot.cooking = False
                pot.cooking_timer = -1
                return

        # Deliver
        if target == _GOAL_POS and a.holding == "dish":
            a.holding = ""
            self._score += int(self.config.delivery_reward)
            events.delivery = True
            events.score_delta = self.config.delivery_reward

    def get_state(self) -> GameState:
        return GameState(
            width=_MOCK_WIDTH,
            height=_MOCK_HEIGHT,
            grid=[row[:] for row in _MOCK_GRID],
            agents=[AgentState(a.agent_id, Position(a.pos.x, a.pos.y), a.direction, a.holding)
                    for a in self._agents],
            pots=[PotState(Position(p.pos.x, p.pos.y), p.ingredients, p.cooking_timer, p.cooked, p.cooking)
                  for p in self._pots],
            recipe=[0] * self.config.recipe_size,
            score=self._score,
            step=self._step,
            max_steps=self.config.max_steps,
            done=self._done,
            ingredient_positions=[_INGREDIENT_POS],
            plate_pile_positions=[_PLATE_PILE_POS],
            goal_positions=[_GOAL_POS],
        )
