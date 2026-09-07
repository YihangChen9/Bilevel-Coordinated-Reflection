"""Overcooked-specific worker tools: observe, do_plan, get_status."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from worker.tool import Tool
from scenarios.overcooked.renderer import render_agent_view, render_global, log_game_state


def make_overcooked_tools(
    adapter, agent_id: int, planner,
    game_log_path: Path | None = None,
) -> list[Tool]:
    """Create tools for one worker in the Overcooked scenario.

    - observe(): returns the agent's local view as text
    - do_plan(cmd): executes a macro command (e.g., "pickup ingredient_0") via planner,
      steps the env, returns the new observation
    - get_status(): returns score, step, pot states as a readable string
    """

    def _observe() -> str:
        state = adapter.get_state()
        return render_agent_view(state, agent_id)

    def _do_plan(cmd: str) -> str:
        """Execute a macro command. The planner converts it to a sequence of low-level
        actions. The adapter steps once per action, with the other agents doing STAY.
        Returns the observation after all actions are executed."""
        from scenarios.overcooked.common import Action
        action_seq = planner.plan(agent_id, cmd)
        n_agents = adapter.config.n_agents
        last_events = None
        for action_int in action_seq:
            actions = {i: Action.STAY for i in range(n_agents)}
            actions[agent_id] = action_int
            _, last_events = adapter.step(actions)
            # Log every game step for per-step replay
            if game_log_path is not None:
                _st = adapter.get_state()
                ev = ""
                if last_events and last_events.delivery:
                    ev = "delivery!"
                # Save rendered frame image if adapter supports it
                img_path = None
                if hasattr(adapter, "render_state_image"):
                    frames_dir = game_log_path.parent / "frames"
                    frames_dir.mkdir(exist_ok=True)
                    img_file = frames_dir / f"step_{_st.step:04d}.png"
                    img_file.write_bytes(adapter.render_state_image())
                    img_path = str(img_file.relative_to(game_log_path.parent))
                opt = adapter.min_steps_to_delivery() if hasattr(adapter, "min_steps_to_delivery") else None
                log_game_state(game_log_path, _st, event=ev,
                               worker_actions={str(agent_id): cmd},
                               frame_image=img_path,
                               optimal_remaining=opt)
            if adapter.get_state().done:
                break
        state = adapter.get_state()
        obs = render_agent_view(state, agent_id)
        event_notes = []
        if last_events and last_events.delivery:
            event_notes.append(f"Delivery! +{last_events.score_delta}")
        if last_events and last_events.stuck_agents:
            event_notes.append(f"Stuck agents: {last_events.stuck_agents}")
        if event_notes:
            obs += "\nEvents: " + "; ".join(event_notes)
        return obs

    def _get_status() -> str:
        state = adapter.get_state()
        return render_global(state)

    return [
        Tool(
            name="observe",
            description="Get your current view of the kitchen.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            func=_observe,
        ),
        Tool(
            name="do_plan",
            description=(
                "Execute a macro command. Valid commands:\n"
                "  pickup <target> — go to target and pick it up (ingredient_0, plate_pile)\n"
                "  place_in <target> — place held item into target (pot_0, pot_1)\n"
                "  deliver — go to goal and deliver held dish\n"
                "  wait_near <target> — go near target then wait\n"
                "After execution, returns your updated view."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Macro command string."},
                },
                "required": ["cmd"],
                "additionalProperties": False,
            },
            func=_do_plan,
        ),
        Tool(
            name="get_status",
            description="Get full kitchen status: score, step, all pot states, recipe.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            func=_get_status,
        ),
    ]
