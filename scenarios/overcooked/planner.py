"""MacroPlanner: converts high-level commands to action sequences.

MockPlanner uses simple BFS pathfinding. The real OvercookedAdapter will
use jaxmarl's PathPlanner for JAX-compatible planning.
"""
from __future__ import annotations

from collections import deque
from typing import Protocol

from scenarios.overcooked.common import (
    Action, CellType, DIRECTION_DELTAS, GameState, Position,
)


class Planner(Protocol):
    def plan(self, agent_id: int, cmd: str) -> list[int]:
        """Convert a macro command string to a sequence of Action ints."""
        ...


_TARGET_MAP_KEYS = {
    "ingredient_0": "ingredient",
    "ingredient_1": "ingredient",
    "plate_pile": "plate",
    "plates": "plate",
    "goal": "goal",
}


class MockPlanner:
    """BFS-based planner for MockAdapter. Resolves target names to positions
    from the current GameState, then plans a path + final INTERACT."""

    def __init__(self, adapter):
        self._adapter = adapter

    def _resolve_target(self, cmd: str) -> Position | None:
        state = self._adapter.get_state()
        parts = cmd.strip().split(maxsplit=1)
        verb = parts[0]
        target_name = parts[1] if len(parts) > 1 else ""

        if "ingredient" in target_name and state.ingredient_positions:
            # Pick the NEAREST ingredient pile to this agent
            agent_pos = state.agents[self._current_agent_id].pos if hasattr(self, '_current_agent_id') else None
            if agent_pos and len(state.ingredient_positions) > 1:
                return min(state.ingredient_positions,
                           key=lambda p: abs(p.x - agent_pos.x) + abs(p.y - agent_pos.y))
            return state.ingredient_positions[0]
        if "pot" in target_name and state.pots:
            idx = 0
            for ch in target_name:
                if ch.isdigit():
                    idx = int(ch)
                    break
            if idx < len(state.pots):
                return state.pots[idx].pos
        if "plate" in target_name and state.plate_pile_positions:
            return state.plate_pile_positions[0]
        if verb == "deliver" and state.goal_positions:
            return state.goal_positions[0]
        if "goal" in target_name and state.goal_positions:
            return state.goal_positions[0]
        return None

    def _bfs_path(self, start: Position, target_adjacent: Position, state: GameState) -> list[Position]:
        """BFS from start to a cell adjacent to target. Returns list of positions to visit."""
        # Find all cells adjacent to target that are walkable
        adj_cells = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = target_adjacent.x + dx, target_adjacent.y + dy
            if 0 <= nx < state.width and 0 <= ny < state.height:
                if state.grid[ny][nx] in (CellType.EMPTY, CellType.GOAL):
                    adj_cells.append(Position(nx, ny))

        if not adj_cells:
            return []

        # BFS
        queue = deque([(start, [start])])
        visited = {(start.x, start.y)}
        targets = {(p.x, p.y) for p in adj_cells}

        if (start.x, start.y) in targets:
            return [start]

        while queue:
            pos, path = queue.popleft()
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = pos.x + dx, pos.y + dy
                if (nx, ny) in visited:
                    continue
                if not (0 <= nx < state.width and 0 <= ny < state.height):
                    continue
                if state.grid[ny][nx] not in (CellType.EMPTY, CellType.GOAL):
                    continue
                visited.add((nx, ny))
                new_pos = Position(nx, ny)
                new_path = path + [new_pos]
                if (nx, ny) in targets:
                    return new_path
                queue.append((new_pos, new_path))
        return []

    def _path_to_actions(self, path: list[Position], face_target: Position) -> list[int]:
        """Convert position path to movement actions, ending with face + INTERACT."""
        actions: list[int] = []
        for i in range(1, len(path)):
            dx = path[i].x - path[i - 1].x
            dy = path[i].y - path[i - 1].y
            if dx == 1:
                actions.append(Action.RIGHT)
            elif dx == -1:
                actions.append(Action.LEFT)
            elif dy == 1:
                actions.append(Action.DOWN)
            elif dy == -1:
                actions.append(Action.UP)

        # Face the target
        last = path[-1] if path else None
        if last and face_target:
            dx = face_target.x - last.x
            dy = face_target.y - last.y
            if dx > 0:
                actions.append(Action.RIGHT)
            elif dx < 0:
                actions.append(Action.LEFT)
            elif dy > 0:
                actions.append(Action.DOWN)
            elif dy < 0:
                actions.append(Action.UP)
            # Pop the movement, it was just for facing — re-add as stay + direction
            # Actually in our mock, the facing is set by movement direction.
            # The last move action already faces the right way if target is adjacent.

        actions.append(Action.INTERACT)
        return actions

    def plan(self, agent_id: int, cmd: str) -> list[int]:
        state = self._adapter.get_state()
        parts = cmd.strip().split(maxsplit=1)
        verb = parts[0]

        if verb in ("wait", "wait_near"):
            target = self._resolve_target(cmd)
            if target:
                path = self._bfs_path(state.agents[agent_id].pos, target, state)
                actions = self._path_to_actions(path, target) if len(path) > 1 else []
                # Remove the INTERACT at the end (just navigate near, then STAY)
                if actions and actions[-1] == Action.INTERACT:
                    actions.pop()
                actions.extend([Action.STAY] * 5)
                return actions
            return [Action.STAY] * 5

        target = self._resolve_target(cmd)
        if target is None:
            return [Action.STAY]

        path = self._bfs_path(state.agents[agent_id].pos, target, state)
        if not path:
            return [Action.STAY]

        return self._path_to_actions(path, target)
