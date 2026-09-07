"""Offline ground-truth `min_steps_to_delivery` via reachable-state enumeration.

Two-phase algorithm:

1. **Forward BFS** from the start state — explores all states reachable from
   the start via any joint action.  Records the full transition graph
   (state, action) → (next_state, reward).  Edges that yield reward > 0 mark
   their source state as a "pre-delivery" state.

2. **Backward BFS** from the set of pre-delivery states over the reverse
   transition graph.  Assigns each reachable state s a value V(s) =
   minimum steps until next delivery.

Result: a `{state_key → V(s)}` dictionary that can be pickled and reused.

This is exact and deterministic.  Counter-drops are NOT pruned a priori —
the BFS explores them too — but in single-recipe layouts those branches
quickly hit pre-delivery states with the same or worse cost, so they don't
inflate `V(s)` for the start.

Memory budget: each state object is ~1-2 KB in Python; 100K states ≈ 100-
200 MB.  For layouts where reachable_count > some threshold, the BFS is
aborted gracefully.
"""
from __future__ import annotations

import itertools
import time
from collections import deque
from typing import Optional


_AGENT_ACTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1), (0, 0), "interact")


def state_key(state) -> tuple:
    """Canonical, hashable, value-based key for an OvercookedState.

    Excludes timestep and order_list (assumed constant for this run).
    """
    players = tuple(
        (
            tuple(p.position),
            tuple(p.orientation),
            None if p.held_object is None else (
                p.held_object.name,
                tuple(sorted((ing if isinstance(ing, str) else ing.name)
                             for ing in (getattr(p.held_object, "ingredients", []) or [])))
                    if hasattr(p.held_object, "ingredients") else (),
                getattr(p.held_object, "_cooking_tick", -1)
                    if hasattr(p.held_object, "_cooking_tick") else -1,
            ),
        )
        for p in state.players
    )
    objects = tuple(sorted(
        (
            tuple(pos),
            obj.name,
            tuple(sorted((ing if isinstance(ing, str) else ing.name) for ing in obj.ingredients))
                if hasattr(obj, "ingredients") else (),
            getattr(obj, "_cooking_tick", -1)
                if hasattr(obj, "_cooking_tick") else -1,
        )
        for pos, obj in state.objects.items()
    ))
    return (players, objects)


def build_ground_truth(
    start_state,
    mdp,
    n_agents: int,
    max_states: int = 2_000_000,
    progress_every: int = 5_000,
    prune_counter_drops: bool = True,
    log=print,
) -> dict:
    """Returns dict with:
      states: {state_key → index}
      V: {state_key → min steps to next delivery (or None if unreachable)}
      forward_edges: count of transitions explored
      reachable_count: |states|
      elapsed_sec: wall clock
      reason: "complete" | "budget"
    """
    joint_actions = tuple(itertools.product(_AGENT_ACTIONS, repeat=n_agents))
    pot_locations = frozenset(tuple(p) for p in mdp.get_pot_locations())
    t0 = time.time()

    def has_counter_drop(ns) -> bool:
        """An object key not at a pot position = an item dropped on a counter
        (or already on a counter)."""
        for pos in ns.objects.keys():
            if tuple(pos) not in pot_locations:
                return True
        return False

    # ── Phase 1: forward BFS, build reverse adjacency + pre-delivery set ──
    state_objs: dict[tuple, object] = {state_key(start_state): start_state}
    reverse_adj: dict[tuple, list[tuple]] = {state_key(start_state): []}
    pre_delivery: set[tuple] = set()
    queue: deque = deque([start_state])
    explored = 0

    while queue:
        s = queue.popleft()
        sk = state_key(s)
        for ja in joint_actions:
            try:
                ns, info = mdp.get_state_transition(s, ja)
            except Exception:
                continue
            explored += 1
            reward = sum(info.get("sparse_reward_by_agent", (0,) * n_agents))
            if reward > 0:
                pre_delivery.add(sk)
                # don't traverse past delivery — the post-delivery state is
                # outside the "until-next-delivery" subproblem
                continue
            if prune_counter_drops and has_counter_drop(ns):
                # Counter-drops are provably suboptimal in single-recipe
                # layouts (the dropped item must be picked up again, costing
                # extra INTERACTs).  Pruning preserves optimality.
                continue
            nk = state_key(ns)
            if nk not in state_objs:
                state_objs[nk] = ns
                reverse_adj[nk] = []
                queue.append(ns)
                if len(state_objs) % progress_every == 0:
                    log(f"  forward BFS: |states|={len(state_objs):,}  "
                        f"|queue|={len(queue):,}  pre_delivery={len(pre_delivery):,}  "
                        f"t={time.time()-t0:.1f}s")
                if len(state_objs) > max_states:
                    log(f"  ✗ budget exceeded ({max_states:,} states)")
                    return {
                        "states": state_objs, "V": {}, "pre_delivery": pre_delivery,
                        "forward_edges": explored, "reachable_count": len(state_objs),
                        "elapsed_sec": time.time() - t0, "reason": "budget",
                    }
            reverse_adj[nk].append(sk)

    log(f"  forward BFS done: |states|={len(state_objs):,}  edges={explored:,}  "
        f"pre_delivery={len(pre_delivery):,}  t={time.time()-t0:.1f}s")

    # ── Phase 2: backward BFS from pre-delivery, assigning V(s) ──
    V: dict[tuple, int] = {}
    bq: deque = deque()
    for sk in pre_delivery:
        V[sk] = 1
        bq.append(sk)
    while bq:
        sk = bq.popleft()
        d = V[sk]
        for pk in reverse_adj.get(sk, ()):
            if pk in V:
                continue
            V[pk] = d + 1
            bq.append(pk)

    log(f"  backward BFS done: |V|={len(V):,}  t={time.time()-t0:.1f}s")

    return {
        "states": state_objs, "V": V, "pre_delivery": pre_delivery,
        "forward_edges": explored, "reachable_count": len(state_objs),
        "elapsed_sec": time.time() - t0, "reason": "complete",
    }
