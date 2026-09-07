"""Real Overcooked adapter wrapping JaxMARL's Overcooked (v1).

Requires: uv sync --group overcooked (installs jax + jaxmarl).

The installed jaxmarl package ships v1 (jaxmarl.environments.overcooked), not v2.
This adapter converts between v1's JAX state and our GameState/Action types.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl.environments.overcooked import Overcooked, overcooked_layouts
from jaxmarl.environments.overcooked.common import OBJECT_TO_INDEX

from scenarios.overcooked.common import (
    Action, AgentState, CellType, GameState, Position, PotState, StepEvents,
)
from scenarios.overcooked.config import OvercookedConfig


# ── Action mapping ───────────────────────────────────────────────────────
# Our Action enum:  RIGHT=0, DOWN=1, LEFT=2, UP=3, STAY=4, INTERACT=5
# v1 encoding:      UP=0,    DOWN=1, RIGHT=2, LEFT=3, STAY=4, INTERACT=5

_OUR_TO_V1 = {
    Action.RIGHT: 2,
    Action.DOWN: 1,
    Action.LEFT: 3,
    Action.UP: 0,
    Action.STAY: 4,
    Action.INTERACT: 5,
}

_V1_DIR_TO_OUR = {
    0: Action.UP,
    1: Action.DOWN,
    2: Action.RIGHT,
    3: Action.LEFT,
}

# Inventory decoding (OBJECT_TO_INDEX values)
_INV_EMPTY = OBJECT_TO_INDEX["empty"]
_INV_ONION = OBJECT_TO_INDEX["onion"]
_INV_PLATE = OBJECT_TO_INDEX["plate"]
_INV_DISH = OBJECT_TO_INDEX["dish"]


def _inv_to_str(val: int) -> str:
    if val == _INV_ONION:
        return "ingredient_0"
    if val == _INV_PLATE:
        return "plate"
    if val == _INV_DISH:
        return "dish"
    return ""


# ── Layout parsing ───────────────────────────────────────────────────────


def _flat_to_pos(flat_idx: int, width: int) -> Position:
    return Position(x=int(flat_idx) % width, y=int(flat_idx) // width)


class OvercookedAdapter:
    """Wraps JaxMARL v1 Overcooked into the GameState interface."""

    def __init__(self, config: OvercookedConfig):
        self.config = config
        layout_name = config.layout
        if layout_name not in overcooked_layouts:
            raise ValueError(f"Unknown layout: {layout_name}. Available: {list(overcooked_layouts)}")
        self._layout = overcooked_layouts[layout_name]
        self._env = Overcooked(layout=self._layout, max_steps=config.max_steps)
        self._key = jax.random.PRNGKey(42)
        self._state = None
        self._cumulative_score: float = 0.0
        self._position_history: dict[int, list[Position]] = {}

        # Parse fixed positions from layout (must be before pot tracking init)
        w = int(self._layout["width"])
        self._width = w
        self._height = int(self._layout["height"])
        self._pot_positions = [_flat_to_pos(int(i), w) for i in self._layout["pot_idx"]]
        self._ingredient_positions = [_flat_to_pos(int(i), w) for i in self._layout.get("onion_pile_idx", [])]
        self._plate_positions = [_flat_to_pos(int(i), w) for i in self._layout.get("plate_pile_idx", [])]
        self._goal_positions = [_flat_to_pos(int(i), w) for i in self._layout["goal_idx"]]

        # Build a walkability grid from wall_idx
        wall_set = set(int(i) for i in self._layout["wall_idx"])
        self._grid: list[list[CellType]] = []
        for y in range(self._height):
            row = []
            for x in range(self._width):
                flat = y * w + x
                if flat in wall_set:
                    # Check if it's a known object type
                    pos = Position(x, y)
                    if pos in self._pot_positions:
                        row.append(CellType.POT)
                    elif pos in self._ingredient_positions:
                        row.append(CellType.INGREDIENT_PILE)
                    elif pos in self._plate_positions:
                        row.append(CellType.PLATE_PILE)
                    elif pos in self._goal_positions:
                        row.append(CellType.GOAL)
                    else:
                        row.append(CellType.WALL)
                else:
                    row.append(CellType.EMPTY)
            self._grid.append(row)

        # Track pot state ourselves — v1 stores ingredients on adjacent counter cells,
        # not on the pot cell itself, so reading maze_map at pot position is unreliable.
        self._pot_ingredients: list[int] = [0] * len(self._pot_positions)
        self._pot_cooking_timer: list[int] = [-1] * len(self._pot_positions)
        self._pot_cooked: list[bool] = [False] * len(self._pot_positions)

    def reset(self) -> GameState:
        self._key, subkey = jax.random.split(self._key)
        _, self._state = self._env.reset(subkey)
        self._cumulative_score = 0.0
        self._position_history = {i: [] for i in range(self.config.n_agents)}
        self._pot_ingredients = [0] * len(self._pot_positions)
        self._pot_cooking_timer = [-1] * len(self._pot_positions)
        self._pot_cooked = [False] * len(self._pot_positions)
        return self.get_state()

    def step(self, actions: dict[int, int]) -> tuple[GameState, StepEvents]:
        # Snapshot inventories BEFORE step to detect changes
        inv_before = [int(np.array(self._state.agent_inv)[i]) for i in range(self.config.n_agents)]

        self._key, subkey = jax.random.split(self._key)
        v1_actions = {
            f"agent_{i}": jnp.int32(_OUR_TO_V1.get(Action(a), 4))
            for i, a in actions.items()
        }
        _, self._state, rewards, dones, info = self._env.step_env(
            subkey, self._state, v1_actions,
        )

        # Infer pot state from inventory changes
        inv_after = [int(np.array(self._state.agent_inv)[i]) for i in range(self.config.n_agents)]
        for i in range(self.config.n_agents):
            action = Action(actions.get(i, Action.STAY))
            if action == Action.INTERACT:
                pos = self._get_agent_pos(i)

                # Agent dropped onion (inv went from onion → empty) → placed in pot
                if inv_before[i] == _INV_ONION and inv_after[i] == _INV_EMPTY:
                    for pi, pp in enumerate(self._pot_positions):
                        if abs(pos.x - pp.x) + abs(pos.y - pp.y) == 1:
                            if self._pot_ingredients[pi] < self.config.recipe_size and not self._pot_cooked[pi]:
                                self._pot_ingredients[pi] += 1
                                if self._pot_ingredients[pi] >= self.config.recipe_size:
                                    self._pot_cooking_timer[pi] = self.config.cook_time
                                break

                # Agent picked up dish (inv went from plate → dish) → pot emptied
                if inv_before[i] == _INV_PLATE and inv_after[i] == _INV_DISH:
                    for pi, pp in enumerate(self._pot_positions):
                        if abs(pos.x - pp.x) + abs(pos.y - pp.y) == 1:
                            if self._pot_cooked[pi]:
                                self._pot_ingredients[pi] = 0
                                self._pot_cooked[pi] = False
                                self._pot_cooking_timer[pi] = -1
                                break

        # Tick cooking timers
        for pi in range(len(self._pot_positions)):
            if self._pot_cooking_timer[pi] > 0:
                self._pot_cooking_timer[pi] -= 1
                if self._pot_cooking_timer[pi] <= 0:
                    self._pot_cooked[pi] = True
                    self._pot_cooking_timer[pi] = 0

        events = StepEvents()
        total_reward = sum(float(r) for r in rewards.values())
        if total_reward > 0:
            events.delivery = True
            events.score_delta = total_reward / self.config.n_agents
            self._cumulative_score += events.score_delta

        # Track positions
        for i in range(self.config.n_agents):
            pos = self._get_agent_pos(i)
            self._position_history[i].append(pos)

        # Stuck detection
        for i in range(self.config.n_agents):
            hist = self._position_history[i]
            if len(hist) >= self.config.stuck_threshold:
                recent = hist[-self.config.stuck_threshold:]
                if len(set((p.x, p.y) for p in recent)) == 1:
                    events.stuck_agents.append(i)

        return self.get_state(), events

    def _get_agent_pos(self, agent_id: int) -> Position:
        pos = np.array(self._state.agent_pos)
        return Position(x=int(pos[agent_id, 0]), y=int(pos[agent_id, 1]))

    def get_state(self) -> GameState:
        agents = []
        for i in range(self.config.n_agents):
            pos = self._get_agent_pos(i)
            dir_idx = int(np.array(self._state.agent_dir_idx)[i])
            direction = _V1_DIR_TO_OUR.get(dir_idx, Action.DOWN)
            inv_val = int(np.array(self._state.agent_inv)[i])
            holding = _inv_to_str(inv_val)
            agents.append(AgentState(agent_id=i, pos=pos, direction=direction, holding=holding))

        # Pot state tracked by us (inferred from inventory changes in step())
        pots = [
            PotState(
                pos=self._pot_positions[pi],
                ingredients=self._pot_ingredients[pi],
                cooking_timer=self._pot_cooking_timer[pi],
                cooked=self._pot_cooked[pi],
            )
            for pi in range(len(self._pot_positions))
        ]

        return GameState(
            width=self._width,
            height=self._height,
            grid=[row[:] for row in self._grid],
            agents=agents,
            pots=pots,
            recipe=[0] * self.config.recipe_size,
            score=int(self._cumulative_score),
            step=int(np.array(self._state.time)),
            max_steps=self.config.max_steps,
            done=bool(np.array(self._state.terminal)),
            ingredient_positions=list(self._ingredient_positions),
            plate_pile_positions=list(self._plate_positions),
            goal_positions=list(self._goal_positions),
        )


class JaxPlanner:
    """Delegates to MockPlanner's BFS — reads from OvercookedAdapter's GameState."""

    def __init__(self, adapter: OvercookedAdapter):
        self._adapter = adapter
        from scenarios.overcooked.planner import MockPlanner
        self._inner = MockPlanner(adapter)

    def plan(self, agent_id: int, cmd: str) -> list[int]:
        return self._inner.plan(agent_id, cmd)
