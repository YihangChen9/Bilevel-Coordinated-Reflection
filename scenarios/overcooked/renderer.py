"""Render GameState as text for LLM consumption + visual grid for humans."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from scenarios.overcooked.common import Action, CellType, GameState, Position


_DIR_ARROWS = {Action.RIGHT: "→", Action.DOWN: "↓", Action.LEFT: "←", Action.UP: "↑"}

# ── Visual grid for humans ───────────────────────────────────────────────

_CELL_CHARS = {
    CellType.WALL: "██",
    CellType.EMPTY: "  ",
    CellType.GOAL: "🎯",
    CellType.POT: "🍲",
    CellType.PLATE_PILE: "🍽️",
    CellType.INGREDIENT_PILE: "🧅",
}

_AGENT_CHARS = ["🧑", "👩"]
_AGENT_HOLD = {"ingredient_0": "+🧅", "plate": "+🍽️", "dish": "+🍛", "": "  "}


def render_grid_visual(state: GameState) -> str:
    """Render the kitchen as a visual grid string for Streamlit display."""
    # Build grid with cell chars
    rows = []
    agent_map = {(a.pos.x, a.pos.y): a for a in state.agents}

    for y in range(state.height):
        cells = []
        for x in range(state.width):
            if (x, y) in agent_map:
                a = agent_map[(x, y)]
                char = _AGENT_CHARS[a.agent_id % len(_AGENT_CHARS)]
                arrow = _DIR_ARROWS.get(a.direction, " ")
                cells.append(f"{char}{arrow}")
            else:
                ct = state.grid[y][x] if y < len(state.grid) and x < len(state.grid[y]) else CellType.WALL
                cells.append(_CELL_CHARS.get(ct, "??"))
        rows.append("│" + "│".join(cells) + "│")

    top = "┌" + "┬".join(["──"] * state.width) + "┐"
    bot = "└" + "┴".join(["──"] * state.width) + "┘"
    grid_str = "\n".join([top] + rows + [bot])

    # Agent status line
    agent_lines = []
    for a in state.agents:
        hold = a.holding if a.holding else "empty"
        arrow = _DIR_ARROWS.get(a.direction, "?")
        agent_lines.append(f"  {_AGENT_CHARS[a.agent_id % len(_AGENT_CHARS)]} Agent {a.agent_id}: ({a.pos.x},{a.pos.y}){arrow} holding={hold}")

    # Pot status
    pot_lines = []
    for i, p in enumerate(state.pots):
        if p.is_ready:
            pot_lines.append(f"  🍲 Pot {i}: READY (pick up with plate!)")
        elif p.is_cooking:
            pot_lines.append(f"  🍲 Pot {i}: cooking ({p.cooking_timer} steps left)")
        elif p.ingredients > 0:
            pot_lines.append(f"  🍲 Pot {i}: {p.ingredients}/{3} ingredients")
        else:
            pot_lines.append(f"  🍲 Pot {i}: empty")

    return "\n".join([
        grid_str,
        f"Score: {state.score}  Step: {state.step}/{state.max_steps}",
    ] + agent_lines + pot_lines)


# ── Game log (JSONL for Streamlit replay) ────────────────────────────────


def log_game_state(path: Path, state: GameState, event: str = "", worker_actions: dict | None = None, frame_image: str | None = None, optimal_remaining: int | None = None):
    """Append one JSON line per macro round for Streamlit replay.
    Includes full grid layout so the viz can draw the kitchen."""
    record = {
        "step": state.step,
        "score": state.score,
        "done": state.done,
        "event": event,
        "width": state.width,
        "height": state.height,
        "grid": [[int(c) for c in row] for row in state.grid],
        "agents": [
            {"id": a.agent_id, "x": a.pos.x, "y": a.pos.y,
             "dir": _DIR_ARROWS.get(a.direction, "?"), "holding": a.holding}
            for a in state.agents
        ],
        "pots": [
            {"x": p.pos.x, "y": p.pos.y, "ingredients": p.ingredients,
             "cooking_timer": p.cooking_timer, "cooked": p.cooked}
            for p in state.pots
        ],
        "ingredient_positions": [{"x": p.x, "y": p.y} for p in state.ingredient_positions],
        "plate_positions": [{"x": p.x, "y": p.y} for p in state.plate_pile_positions],
        "goal_positions": [{"x": p.x, "y": p.y} for p in state.goal_positions],
        "worker_actions": worker_actions or {},
        "frame_image": frame_image,
        "optimal_remaining": optimal_remaining,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def render_global(state: GameState) -> str:
    """Full kitchen state for orchestrator (~120 tokens)."""
    lines = [
        f"Kitchen {state.width}×{state.height}, step {state.step}/{state.max_steps}, "
        f"score {state.score}.",
        f"Recipe: {state.recipe}.",
    ]
    agents_parts = []
    for a in state.agents:
        arrow = _DIR_ARROWS.get(a.direction, "?")
        hold = a.holding if a.holding else "empty"
        agents_parts.append(f"A{a.agent_id}({a.pos.x},{a.pos.y}){arrow} {hold}")
    lines.append("Agents: " + " | ".join(agents_parts))

    pot_parts = []
    for i, p in enumerate(state.pots):
        if p.is_ready:
            status = "READY"
        elif p.is_cooking:
            status = f"COOKING ({p.cooking_timer} steps left)"
        elif p.ingredients >= 3:
            status = "FULL 3/3 — INTERACT with pot to START COOKING!"
        elif p.ingredients > 0:
            need = 3 - p.ingredients
            status = f"{p.ingredients}/3 onions (NEED {need} MORE)"
        else:
            status = "EMPTY (needs 3 onions)"
        pot_parts.append(f"P{i}({p.pos.x},{p.pos.y}) {status}")
    lines.append("Pots: " + " | ".join(pot_parts))

    if state.ingredient_positions:
        lines.append("Ingredients: " + " ".join(
            f"@({p.x},{p.y})" for p in state.ingredient_positions
        ))
    if state.plate_pile_positions:
        lines.append("Plates: " + " ".join(
            f"@({p.x},{p.y})" for p in state.plate_pile_positions
        ))
    if state.goal_positions:
        lines.append("Goal: " + " ".join(
            f"@({p.x},{p.y})" for p in state.goal_positions
        ))
    return "\n".join(lines)


def _check_blocking(state: GameState, agent_id: int) -> list[str]:
    """Check if this agent is blocking access to a key object (pot, goal, plates).
    Returns warning strings the agent can see in its observation."""
    warnings = []
    a = state.agents[agent_id]
    my_pos = (a.pos.x, a.pos.y)

    # Key objects to check: pots, goal, plate pile
    key_objects = []
    for i, p in enumerate(state.pots):
        key_objects.append((f"pot_{i}", p.pos))
    for p in state.goal_positions:
        key_objects.append(("goal", p))
    for p in state.plate_pile_positions:
        key_objects.append(("plate_pile", p))

    for obj_name, obj_pos in key_objects:
        # Find all walkable cells adjacent to this object
        adj_walkable = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = obj_pos.x + dx, obj_pos.y + dy
            if 0 <= ny < state.height and 0 <= nx < state.width:
                if state.grid[ny][nx] in (CellType.EMPTY, CellType.GOAL):
                    adj_walkable.append((nx, ny))

        # If this agent is on the ONLY walkable cell adjacent to the object
        if len(adj_walkable) == 1 and adj_walkable[0] == my_pos:
            # Check if the agent actually has a reason to be there
            is_pot = obj_name.startswith("pot_")
            if is_pot:
                pot_idx = int(obj_name.split("_")[1])
                pot = state.pots[pot_idx]
                if pot.is_cooking and a.holding != "plate":
                    warnings.append(
                        f"You are BLOCKING {obj_name}! The pot is cooking and you have no plate. "
                        f"MOVE AWAY so your teammate can access it when it's ready."
                    )
                elif pot.is_ready and a.holding != "plate":
                    warnings.append(
                        f"You are BLOCKING {obj_name}! The pot is READY but you have no plate. "
                        f"MOVE AWAY so the deliverer can plate the dish."
                    )
                elif pot.ingredients > 0 and pot.ingredients < 3 and a.holding != "ingredient_0":
                    warnings.append(
                        f"You are BLOCKING {obj_name}! Pot has {pot.ingredients}/3 onions but you "
                        f"have no onion to add. MOVE AWAY, go get an onion first."
                    )
            else:
                warnings.append(
                    f"You are BLOCKING access to {obj_name}! Move away."
                )

    return warnings


def render_agent_view(state: GameState, agent_id: int) -> str:
    """Per-agent view for worker LLM (~80 tokens)."""
    a = state.agents[agent_id]
    arrow = _DIR_ARROWS.get(a.direction, "?")
    hold = a.holding if a.holding else "nothing"
    lines = [
        f"You are Agent {agent_id} at ({a.pos.x},{a.pos.y}) facing {arrow}, holding {hold}.",
    ]
    # Nearby objects
    nearby = []
    for i, p in enumerate(state.pots):
        if p.is_ready:
            nearby.append(f"pot_{i}({p.pos.x},{p.pos.y}) ✅READY — pick up with plate!")
        elif p.is_cooking:
            nearby.append(f"pot_{i}({p.pos.x},{p.pos.y}) COOKING ({p.cooking_timer} steps left)")
        elif p.ingredients >= 3:
            nearby.append(f"pot_{i}({p.pos.x},{p.pos.y}) FULL 3/3 — INTERACT to START COOKING!")
        elif p.ingredients > 0:
            need = 3 - p.ingredients
            nearby.append(f"pot_{i}({p.pos.x},{p.pos.y}) {p.ingredients}/3 onions — NEED {need} MORE!")
        else:
            nearby.append(f"pot_{i}({p.pos.x},{p.pos.y}) EMPTY — needs 3 onions")
    for p in state.ingredient_positions:
        nearby.append(f"ingredient@({p.x},{p.y})")
    for p in state.plate_pile_positions:
        nearby.append(f"plates@({p.x},{p.y})")
    for p in state.goal_positions:
        nearby.append(f"goal@({p.x},{p.y})")
    for oa in state.agents:
        if oa.agent_id != agent_id:
            oa_hold = oa.holding if oa.holding else "empty"
            oa_arrow = _DIR_ARROWS.get(oa.direction, "?")
            nearby.append(f"Agent_{oa.agent_id}({oa.pos.x},{oa.pos.y}){oa_arrow} {oa_hold}")
    lines.append("Nearby: " + ", ".join(nearby))
    lines.append(f"Step {state.step}/{state.max_steps}, score {state.score}.")

    # Blocking warning: check if this agent is adjacent to a pot/goal/pile AND
    # a teammate needs to reach it but this agent is in the only access cell.
    warnings = _check_blocking(state, agent_id)
    if warnings:
        lines.append("")
        for w in warnings:
            lines.append(f"⚠️ {w}")

    return "\n".join(lines)
