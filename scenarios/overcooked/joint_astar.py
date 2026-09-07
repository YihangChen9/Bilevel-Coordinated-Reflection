"""Joint-state A* over OvercookedState — ground-truth `steps_to_delivery`.

Why this exists: `OriginalAdapter.min_steps_to_delivery()` simulates the
GreedyHumanModel oracle, which is a stochastic, stateful heuristic — repeated
calls on the same state can return values spread across ±20 steps (see
diagnosis in experiments/LOG.md).  This module computes a deterministic exact
minimum by joint-action A* search using `mdp.get_state_transition` as the
transition function (no side-effects, fully reproducible).

Action space per agent: 6 = {RIGHT, DOWN, LEFT, UP, STAY, INTERACT}.
Joint branching factor: 6^n_agents (36 for 2 agents).

Heuristic: admissible lower bound = max(
    ceil(remaining_required_interacts / n_agents),
    cook_time_remaining_if_cooking,
).  Both terms are independently admissible (every interact is a separate
timestep; soup can't be delivered before cooking finishes), and max of two
admissible heuristics is admissible.

Returns `None` (not -1) when the search budget is exhausted without finding
a delivery — sentinel-free by design.
"""
from __future__ import annotations

import heapq
import itertools
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional


# overcooked_ai joint-action atoms
_AGENT_ACTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1), (0, 0), "interact")
_INTERACT = "interact"


# ── Layout BFS distance helpers ───────────────────────────────────────────
#
# We precompute, for one mdp, the BFS distance from every walkable cell to
# the nearest cell *adjacent to* each facility type (pot / ingredient pile /
# plate pile / serving counter).  Adjacency matters because the agent must
# stand next to (not on) the facility to INTERACT.
#
# This is purely a function of layout, so we cache it on the mdp by id.

_BFS_CACHE: dict[int, dict] = {}


def _walkable(mdp, x: int, y: int) -> bool:
    if x < 0 or y < 0 or x >= mdp.width or y >= mdp.height:
        return False
    return mdp.terrain_mtx[y][x] == " "


def _bfs_from_targets(mdp, target_cells: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    """BFS over walkable cells starting from cells *adjacent to* target cells
    (the target cells themselves are walls — pot, pile, serving counter —
    so we seed from their walkable 4-neighbours)."""
    dist: dict[tuple[int, int], int] = {}
    q = deque()
    for tx, ty in target_cells:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = tx + dx, ty + dy
            if _walkable(mdp, nx, ny) and (nx, ny) not in dist:
                dist[(nx, ny)] = 0
                q.append((nx, ny))
    while q:
        x, y = q.popleft()
        d = dist[(x, y)]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if _walkable(mdp, nx, ny) and (nx, ny) not in dist:
                dist[(nx, ny)] = d + 1
                q.append((nx, ny))
    return dist


def _layout_distances(mdp) -> dict:
    """Returns dict with keys 'pot', 'ingredient', 'plate', 'goal'; each
    value is a {(x,y) → BFS distance to nearest adjacent walkable cell}."""
    key = id(mdp)
    cached = _BFS_CACHE.get(key)
    if cached is not None:
        return cached
    out = {
        "pot": _bfs_from_targets(mdp, list(mdp.get_pot_locations())),
        "ingredient": _bfs_from_targets(mdp, list(mdp.get_onion_dispenser_locations())),
        "plate": _bfs_from_targets(mdp, list(mdp.get_dish_dispenser_locations())),
        "goal": _bfs_from_targets(mdp, list(mdp.get_serving_locations())),
    }
    _BFS_CACHE[key] = out
    return out


def _d(table: dict, pos: tuple[int, int]) -> int:
    """Distance lookup with a large fallback for unreachable cells."""
    return table.get(tuple(pos), 10**6)


def _required_interacts(state, recipe_size: int) -> int:
    """Lower bound on remaining INTERACT actions needed to score one delivery.

    Counts: onion-pickups + onion-placings + plate-pickup + soup-pickup + deliver.
    Each is a separate INTERACT, so the sum is a valid lower bound on the
    number of timesteps any single agent must spend (or, divided by n_agents,
    a lower bound on parallel time).
    """
    # Inventory across players
    held_onion = 0
    held_plate = 0
    held_soup = 0
    for p in state.players:
        obj = p.held_object
        if obj is None:
            continue
        if obj.name == "onion":
            held_onion += 1
        elif obj.name == "dish":
            held_plate += 1
        elif obj.name == "soup":
            held_soup += 1

    # Soup-in-pot inventory (at most one pot supported in this lower bound;
    # for multi-pot layouts this is still admissible because more pots only
    # reduces work, never increases it).
    in_pot = 0
    cooking_or_ready = False
    for pos, obj in state.objects.items():
        if obj.name == "soup":
            in_pot = len(obj.ingredients)
            cooking_or_ready = True
            break

    if held_soup >= 1:
        # Just need to deliver: 1 INTERACT
        return 1

    onions_still_needed_in_pot = max(0, recipe_size - in_pot)
    # We must place (onions_still_needed_in_pot) onions; for each we need a
    # pickup unless already carried.
    onion_pickups = max(0, onions_still_needed_in_pot - held_onion)
    onion_places = onions_still_needed_in_pot
    plate_pickup = 0 if held_plate >= 1 else 1
    soup_pickup = 1   # someone must pick up the cooked soup
    deliver = 1
    return onion_pickups + onion_places + plate_pickup + soup_pickup + deliver


def _cook_wait(state) -> int:
    """Cooking soup blocks delivery until cook_time_remaining reaches 0.

    `cook_time_remaining` raises before cooking starts (recipe undefined), so
    we only read it when `is_cooking` is true.
    """
    wait = 0
    for _, obj in state.objects.items():
        if obj.name != "soup":
            continue
        if getattr(obj, "is_ready", False):
            return 0
        if getattr(obj, "is_cooking", False):
            try:
                wait = max(wait, obj.cook_time_remaining)
            except Exception:
                pass
    return wait


def _movement_lower_bound(state, mdp, n_agents: int, recipe_size: int) -> int:
    """A movement-only admissible lower bound.

    Idea: the *soup token* must traverse pot → goal eventually, and to be
    delivered some agent must be standing next to the pot when it's ready
    (to pick up) AND next to the goal at some later point.  We can ignore
    everything but the agent who actually does the delivery.

    Cheapest plan: the agent currently nearest the pot picks up the cooked
    soup (already at pot adjacency = 0) and walks to the nearest goal.
    But if no one is holding soup yet, that agent first has to reach a pot
    (carrying a plate) — so the bound is:

        min over agents of (dist_agent_to_pot + dist_pot_to_goal + 1)   if no soup held
        min over agents of (dist_agent_to_goal + 1)                     if soup held

    The +1 accounts for the deliver INTERACT.  Plate / ingredient
    acquisition movement could be added but it gets non-admissible quickly
    when multiple agents share work, so we omit it.
    """
    distances = _layout_distances(mdp)
    holds_soup = any(
        p.held_object is not None and p.held_object.name == "soup"
        for p in state.players
    )
    if holds_soup:
        # the soup-holder must reach a goal
        best = 10**6
        for p in state.players:
            if p.held_object is None or p.held_object.name != "soup":
                continue
            d = _d(distances["goal"], p.position)
            if d < best:
                best = d
        return best + 1  # deliver interact

    # No one holds soup yet.  Someone must end up at pot (to fetch soup),
    # then at goal.  Find the cheapest agent for "go to pot, then go to goal".
    # Distance pot → goal is layout-only, take min over pots.
    pot_to_goal: list[int] = []
    for pos in mdp.get_pot_locations():
        # walk OUT of pot to an adjacent cell, then BFS to goal.  Cheapest
        # adjacent cell's goal-distance is the right bound.
        cells = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = pos[0] + dx, pos[1] + dy
            if _walkable(mdp, nx, ny):
                cells.append(_d(distances["goal"], (nx, ny)))
        if cells:
            pot_to_goal.append(min(cells))
    if not pot_to_goal:
        return 0
    p2g = min(pot_to_goal)

    best_agent = 10**6
    for p in state.players:
        d = _d(distances["pot"], p.position) + p2g
        if d < best_agent:
            best_agent = d
    return best_agent + 2  # +1 soup-pickup interact, +1 deliver interact


def _pipeline_lower_bound(state, n_agents: int, recipe_size: int, cook_time: int) -> int:
    """Lower bound from the *serial* structure of the cooking pipeline.

    Phases are strictly ordered: fill pot → cook → soup-pickup + deliver.
    The agents can parallelise work *within* a phase, but the phases
    themselves cannot overlap (you can't pick up cooked soup until cooking
    finishes; you can't start cooking until the pot is full).

    Returns: ceil(fill_interacts / n_agents) + remaining_cook_time + 2,
    where the 2 accounts for the soup-pickup + deliver interacts.  These
    last two could overlap with cooking via movement, but the INTERACTs
    themselves can only happen after cooking ends, so +2 is admissible.
    """
    # Inventory
    held_soup = sum(1 for p in state.players
                    if p.held_object is not None and p.held_object.name == "soup")
    if held_soup >= 1:
        return 1  # already holding soup: just deliver

    held_onion = sum(1 for p in state.players
                     if p.held_object is not None and p.held_object.name == "onion")
    held_plate = sum(1 for p in state.players
                     if p.held_object is not None and p.held_object.name == "dish")

    soup_in_pot = None
    for _, obj in state.objects.items():
        if obj.name == "soup":
            soup_in_pot = obj
            break

    if soup_in_pot is not None and getattr(soup_in_pot, "is_ready", False):
        # cooked soup sitting in pot: pickup + deliver = 2 interacts min
        return 2

    if soup_in_pot is not None and getattr(soup_in_pot, "is_cooking", False):
        try:
            return int(soup_in_pot.cook_time_remaining) + 2
        except Exception:
            return cook_time + 2

    # Phase 1: still filling.  Count interacts strictly required before
    # cooking can start.  in_pot is the count of onions already in the pot
    # (soup object exists but not cooking) or 0.
    in_pot = len(soup_in_pot.ingredients) if soup_in_pot is not None else 0
    onions_needed = max(0, recipe_size - in_pot)
    onion_pickups = max(0, onions_needed - held_onion)
    onion_places = onions_needed
    start_cook = 1 if in_pot + held_onion < recipe_size or in_pot < recipe_size else 1
    # start_cook is always 1: once full, an INTERACT is needed to start cooking
    fill_interacts = onion_pickups + onion_places + 1
    fill_parallel = math.ceil(fill_interacts / n_agents)
    return fill_parallel + cook_time + 2


def heuristic(state, mdp, n_agents: int, recipe_size: int, cook_time: int = 20) -> int:
    """Admissible lower bound on steps to next delivery.

    Takes the max of three independent admissible lower bounds:
      1. INTERACTs needed, parallelised across agents.
      2. The serial fill→cook→deliver pipeline bound (`_pipeline_lower_bound`).
      3. Movement distance for the agent doing the delivery.
    """
    interacts = _required_interacts(state, recipe_size)
    parallel = math.ceil(interacts / n_agents)
    pipeline = _pipeline_lower_bound(state, n_agents, recipe_size, cook_time)
    movement = _movement_lower_bound(state, mdp, n_agents, recipe_size)
    return max(parallel, pipeline, movement)


@dataclass
class AstarResult:
    steps: Optional[int]          # None = budget exhausted
    expansions: int
    closed_size: int
    open_peak: int
    reason: str                   # "found" | "budget" | "unreachable"


def astar_steps_to_delivery(
    state,
    mdp,
    n_agents: int,
    recipe_size: int,
    cook_time: int = 20,
    max_expansions: int = 200_000,
    max_depth: int | None = None,
) -> AstarResult:
    """Joint-action A* from `state` to the first delivery.

    Returns AstarResult.steps = exact minimum step count, or None if budget
    exceeded.  `mdp.get_state_transition(s, joint_action)` is the transition
    model — pure, no side effects on the real env.
    """
    h0 = heuristic(state, mdp, n_agents, recipe_size, cook_time)
    # OvercookedState supports __hash__ and __eq__ by value.
    start_key = state
    open_heap: list[tuple[int, int, object, int]] = []  # (f, counter, state, g)
    counter = itertools.count()
    heapq.heappush(open_heap, (h0, next(counter), state, 0))
    best_g: dict = {start_key: 0}
    expansions = 0
    open_peak = 1
    joint_action_space = tuple(itertools.product(_AGENT_ACTIONS, repeat=n_agents))

    while open_heap:
        f, _, s, g = heapq.heappop(open_heap)
        # Stale entry?
        if g > best_g.get(s, g):
            continue
        if max_depth is not None and g >= max_depth:
            continue
        expansions += 1
        if expansions > max_expansions:
            return AstarResult(steps=None, expansions=expansions,
                               closed_size=len(best_g), open_peak=open_peak,
                               reason="budget")

        for joint_action in joint_action_space:
            try:
                ns, info = mdp.get_state_transition(s, joint_action)
            except Exception:
                continue
            reward = sum(info.get("sparse_reward_by_agent", (0,) * n_agents))
            ng = g + 1
            if reward > 0:
                return AstarResult(steps=ng, expansions=expansions,
                                   closed_size=len(best_g), open_peak=open_peak,
                                   reason="found")
            prev = best_g.get(ns)
            if prev is not None and prev <= ng:
                continue
            best_g[ns] = ng
            nh = heuristic(ns, mdp, n_agents, recipe_size)
            heapq.heappush(open_heap, (ng + nh, next(counter), ns, ng))
            if len(open_heap) > open_peak:
                open_peak = len(open_heap)

    return AstarResult(steps=None, expansions=expansions,
                       closed_size=len(best_g), open_peak=open_peak,
                       reason="unreachable")
