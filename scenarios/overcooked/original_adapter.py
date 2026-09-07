"""Adapter wrapping the original overcooked_ai Python environment.

Much simpler than the JaxMARL adapter: pot states are directly readable,
coordinates match our GameState, and the GreedyHumanModel planner is
available for oracle baselines.

Requires: pip install overcooked_ai
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

from scenarios.overcooked.common import (
    Action, AgentState, CellType, DIRECTION_DELTAS,
    GameState, Position, PotState, StepEvents,
)
from scenarios.overcooked.config import OvercookedConfig

# Suppress the gym deprecation warning
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import (
    NO_COUNTERS_PARAMS, MediumLevelActionManager, MotionPlanner,
)
from overcooked_ai_py.agents.agent import GreedyHumanModel


# ── Action mapping ───────────────────────────────────────────────────────
# Our Action: RIGHT=0, DOWN=1, LEFT=2, UP=3, STAY=4, INTERACT=5
# overcooked_ai uses direction tuples: (1,0)=E, (0,1)=S, (-1,0)=W, (0,-1)=N, (0,0)=STAY, 'interact'

_OUR_TO_ORIG = {
    Action.RIGHT: (1, 0),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.UP: (0, -1),
    Action.STAY: (0, 0),
    Action.INTERACT: "interact",
}

_ORIG_DIR_TO_OUR = {
    (1, 0): Action.RIGHT,
    (0, 1): Action.DOWN,
    (-1, 0): Action.LEFT,
    (0, -1): Action.UP,
    (0, 0): Action.STAY,
}


def _held_to_str(held_obj) -> str:
    if held_obj is None:
        return ""
    name = held_obj.name
    if name == "onion":
        return "ingredient_0"
    if name == "dish":
        return "plate"
    if name == "soup":
        return "dish"
    return name


class OriginalAdapter:
    """Wraps overcooked_ai's OvercookedEnv into our GameState interface.

    Advantages over JaxMARL adapter:
    - Pot states directly readable (no inference needed)
    - GreedyHumanModel available as oracle baseline
    - Pure Python, no JAX dependency
    """

    def __init__(self, config: OvercookedConfig):
        self.config = config
        self._mdp = OvercookedGridworld.from_layout_name(config.layout)
        self._env = OvercookedEnv.from_mdp(self._mdp, horizon=config.max_steps)
        self._score: float = 0.0
        self._position_history: dict[int, list[Position]] = {}

        # Layout info
        self._width = self._mdp.width
        self._height = self._mdp.height
        self._pot_positions = [Position(x, y) for x, y in self._mdp.get_pot_locations()]
        self._ingredient_positions = [Position(x, y) for x, y in self._mdp.get_onion_dispenser_locations()]
        self._plate_positions = [Position(x, y) for x, y in self._mdp.get_dish_dispenser_locations()]
        self._goal_positions = [Position(x, y) for x, y in self._mdp.get_serving_locations()]

        # Build grid
        terrain = self._mdp.terrain_mtx  # list of lists of chars
        self._grid: list[list[CellType]] = []
        _TERRAIN_MAP = {
            " ": CellType.EMPTY,
            "X": CellType.WALL,
            "P": CellType.POT,
            "O": CellType.INGREDIENT_PILE,
            "D": CellType.PLATE_PILE,
            "S": CellType.GOAL,
        }
        for y in range(self._height):
            row = []
            for x in range(self._width):
                ch = terrain[y][x] if y < len(terrain) and x < len(terrain[y]) else "X"
                row.append(_TERRAIN_MAP.get(ch, CellType.WALL))
            self._grid.append(row)

        # Planner (lazy — only built if needed)
        self._mlam = None
        self._oracle_agents = None

    def reset(self) -> GameState:
        self._env.reset()
        self._score = 0.0
        self._position_history = {i: [] for i in range(self.config.n_agents)}
        return self.get_state()

    def step(self, actions: dict[int, int]) -> tuple[GameState, StepEvents]:
        # Convert our actions to overcooked_ai format
        a0 = _OUR_TO_ORIG.get(Action(actions.get(0, Action.STAY)), (0, 0))
        a1 = _OUR_TO_ORIG.get(Action(actions.get(1, Action.STAY)), (0, 0))

        state, reward, done, info = self._env.step((a0, a1))

        events = StepEvents()
        if reward > 0:
            events.delivery = True
            events.score_delta = reward
        self._score += reward

        # Track positions
        for i in range(self.config.n_agents):
            p = state.players[i]
            self._position_history[i].append(Position(p.position[0], p.position[1]))

        # Stuck detection
        for i in range(self.config.n_agents):
            hist = self._position_history[i]
            if len(hist) >= self.config.stuck_threshold:
                recent = hist[-self.config.stuck_threshold:]
                if len(set((p.x, p.y) for p in recent)) == 1:
                    events.stuck_agents.append(i)

        return self.get_state(), events

    def get_state(self) -> GameState:
        state = self._env.state

        # Agents
        agents = []
        for i, p in enumerate(state.players):
            pos = Position(p.position[0], p.position[1])
            direction = _ORIG_DIR_TO_OUR.get(p.orientation, Action.DOWN)
            holding = _held_to_str(p.held_object)
            agents.append(AgentState(agent_id=i, pos=pos, direction=direction, holding=holding))

        # Pots — read directly from SoupState object at pot location
        pots = []
        for pp in self._pot_positions:
            loc = (pp.x, pp.y)
            ingredients = 0
            cooking_timer = -1
            cooked = False

            cooking = False
            soup = state.get_object(loc) if state.has_object(loc) else None
            if soup and hasattr(soup, "ingredients"):
                ingredients = len(soup.ingredients)
                if soup.is_ready:
                    cooked = True
                    cooking_timer = 0
                elif soup.is_cooking:
                    cooking = True
                    cooking_timer = getattr(soup, "cook_time_remaining", 0)

            pots.append(PotState(pos=pp, ingredients=ingredients,
                                 cooking_timer=cooking_timer, cooked=cooked,
                                 cooking=cooking))

        return GameState(
            width=self._width,
            height=self._height,
            grid=[row[:] for row in self._grid],
            agents=agents,
            pots=pots,
            recipe=[0] * self.config.recipe_size,
            score=int(self._score),
            step=self._env.state.timestep,
            max_steps=self.config.max_steps,
            done=self._env.is_done(),
            ingredient_positions=list(self._ingredient_positions),
            plate_pile_positions=list(self._plate_positions),
            goal_positions=list(self._goal_positions),
        )

    # ── Oracle / Planner ─────────────────────────────────────────────────

    def _ensure_planner(self):
        if self._mlam is None:
            self._mlam = MediumLevelActionManager.from_pickle_or_compute(
                self._mdp, NO_COUNTERS_PARAMS, force_compute=False,
            )

    def get_oracle_action(self, agent_id: int) -> int:
        """Get the optimal action for an agent using GreedyHumanModel."""
        self._ensure_planner()
        if self._oracle_agents is None:
            self._oracle_agents = [GreedyHumanModel(self._mlam) for _ in range(self.config.n_agents)]
            for i, ag in enumerate(self._oracle_agents):
                ag.set_agent_index(i)
                ag.set_mdp(self._mdp)

        result = self._oracle_agents[agent_id].action(self._env.state)
        orig_action = result[0] if isinstance(result, tuple) and isinstance(result[1], dict) else result

        # Convert back to our Action enum
        for our, orig in _OUR_TO_ORIG.items():
            if orig == orig_action:
                return int(our)
        return int(Action.STAY)

    def load_ground_truth_table(self, path: str | Path) -> int:
        """Load a precomputed `state_key → min_steps_to_delivery` lookup table
        (produced by `scenarios/overcooked/ground_truth.py`).  Returns the
        number of entries loaded.  Idempotent: calling again replaces the
        previously loaded table.
        """
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._gt_table = data["V"]
        return len(self._gt_table)

    def gt_steps_to_delivery(self) -> Optional[int]:
        """O(1) ground-truth lookup, requires `load_ground_truth_table()` first.

        Returns None if either (a) no table is loaded, or (b) the current
        state is outside the precomputed reachable set (e.g. arrived via a
        path the BFS did not enumerate).
        """
        if not hasattr(self, "_gt_table"):
            return None
        from scenarios.overcooked.ground_truth import state_key
        return self._gt_table.get(state_key(self._env.state))

    def astar_steps_to_delivery(self, max_expansions: int = 1_000_000,
                                max_depth: int | None = None) -> dict:
        """Ground-truth steps to next delivery via joint-state A* search.

        Deterministic and exact (modulo budget).  Returns a dict:
          {"steps": int | None, "reason": str, "expansions": int,
           "elapsed_sec": float}

        Unlike `min_steps_to_delivery()` (which calls a stochastic
        GreedyHumanModel rollout), this is reproducible across calls.
        """
        import time
        from scenarios.overcooked.joint_astar import astar_steps_to_delivery
        t0 = time.time()
        r = astar_steps_to_delivery(
            self._env.state, self._mdp,
            n_agents=self.config.n_agents,
            recipe_size=self.config.recipe_size,
            cook_time=self.config.cook_time,
            max_expansions=max_expansions,
            max_depth=max_depth,
        )
        return {
            "steps": r.steps, "reason": r.reason,
            "expansions": r.expansions, "closed_size": r.closed_size,
            "open_peak": r.open_peak,
            "elapsed_sec": time.time() - t0,
        }

    def min_steps_to_delivery(self) -> int:
        """Compute exact minimum steps to next delivery by simulating the oracle
        (GreedyHumanModel) from the current state until it scores."""
        import copy
        self._ensure_planner()
        if self._oracle_agents is None:
            self._oracle_agents = [GreedyHumanModel(self._mlam) for _ in range(self.config.n_agents)]
            for i, ag in enumerate(self._oracle_agents):
                ag.set_agent_index(i)
                ag.set_mdp(self._mdp)

        # Deep copy the env to simulate without affecting the real state
        sim_env = copy.deepcopy(self._env)
        for step in range(self.config.max_steps):
            for ag in self._oracle_agents:
                ag.set_mdp(self._mdp)
            a0 = self._oracle_agents[0].action(sim_env.state)
            a1 = self._oracle_agents[1].action(sim_env.state)
            a0 = a0[0] if isinstance(a0, tuple) and isinstance(a0[1], dict) else a0
            a1 = a1[0] if isinstance(a1, tuple) and isinstance(a1[1], dict) else a1
            _, reward, done, _ = sim_env.step((a0, a1))
            if reward > 0:
                return step + 1
            if done:
                break
        return -1  # can't deliver within remaining steps

    def render_state_image(self) -> bytes:
        """Render the current state as a PNG image using overcooked_ai's visualizer.
        Returns PNG bytes suitable for st.image() or saving to file."""
        import io
        import pygame
        from PIL import Image
        from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

        if not hasattr(self, "_visualizer"):
            self._visualizer = StateVisualizer()

        surface = self._visualizer.render_state(
            self._env.state, grid=self._mdp.terrain_mtx
        )
        raw = pygame.image.tostring(surface, "RGBA")
        w, h = surface.get_size()
        img = Image.frombytes("RGBA", (w, h), raw)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def run_oracle_episode(self) -> dict:
        """Run a full episode with GreedyHumanModel agents. Returns {score, deliveries, steps}."""
        self.reset()
        self._ensure_planner()
        if self._oracle_agents is None:
            self._oracle_agents = [GreedyHumanModel(self._mlam) for _ in range(self.config.n_agents)]
            for i, ag in enumerate(self._oracle_agents):
                ag.set_agent_index(i)
                ag.set_mdp(self._mdp)

        deliveries = 0
        for step in range(self.config.max_steps):
            actions = {}
            for i in range(self.config.n_agents):
                actions[i] = self.get_oracle_action(i)
            _, events = self.step(actions)
            if events.delivery:
                deliveries += 1
            if self.get_state().done:
                break

        return {
            "score": int(self._score),
            "deliveries": deliveries,
            "steps": self._env.state.timestep,
        }
