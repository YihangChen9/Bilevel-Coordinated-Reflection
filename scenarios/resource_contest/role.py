"""Role for Resource Contest players."""
from worker.role import Role


# Minimum-information role for the `worker_blind` ablation: the worker is told
# its action space and the feedback channel, and is asked to maximize the env-
# returned effective_x. It is NOT told (a) the reward formula, (b) that there
# is a hidden cap M_i, (c) the monotonicity-then-flat structure, (d) the
# explicit "commit 10" optimal strategy, (e) any role for tau_i. The intent is
# to test how much the bilevel "learning" story collapses once the worker's
# closed-form answer is removed from its prompt.
RC_PLAYER_BLIND = Role(
    name="rc_player_blind",
    system_prompt=(
        "You are an autonomous agent in a coordination game.\n"
        "Each round you choose an integer x ∈ [0, 10] via the commit_action tool.\n"
        "The env will return a numerical signal `effective_x` (also an integer).\n"
        "Your objective for this round: maximize the env-reported effective_x.\n\n"

        "You have no a-priori model of the env. The relationship between your x\n"
        "and the returned effective_x must be inferred from interaction.\n\n"

        "Tools:\n"
        "  - read_scratch / write_scratch: private notes across rounds (use as needed).\n"
        "  - read_workboard: other agents' commits and reflections this round.\n"
        "  - commit_action(action): submit an integer in [0,10]. Up to 3 attempts\n"
        "    per round; the last commit counts.\n"
        "  - submit_reflection(content): OPTIONALLY publish a short note to the\n"
        "    shared workboard for the orchestrator to read.\n\n"

        "Respond with the effective_x value the env reported as your final answer.\n"
    ),
    allowed_tools=None,
)


RC_PLAYER = Role(
    name="rc_player",
    system_prompt=(
        "You are an autonomous agent in a Resource Contest game.\n"
        "There are several agents. Each has a HIDDEN personal cap Mᵢ (an integer 0..10).\n"
        "You DO NOT know your own Mᵢ — you discover it by trial.\n\n"

        "Each round you receive a real-valued allocation τᵢ ≥ 0 from the orchestrator.\n"
        "You pick an integer xᵢ ∈ [0, 10]. The env silently clips it: effective_x = min(xᵢ, Mᵢ).\n"
        "Your contribution to the shared reward is τᵢ · effective_x — strictly increasing in xᵢ\n"
        "up to Mᵢ, then flat. So you want the LARGEST x you can get without clipping…\n"
        "EXCEPT clipping is fine too, because clip gives you effective_x = Mᵢ which IS the max.\n"
        "In other words: just commit a value ≥ Mᵢ and you're optimal. Easiest: commit 10.\n\n"

        "But the tool tells you when you've been clipped. Use that signal to record Mᵢ in\n"
        "your scratch so future rounds are 1-call instead of 3.\n\n"

        "Tools:\n"
        "  - read_scratch / write_scratch: your private memory across rounds. After you've\n"
        "    discovered Mᵢ, store it (e.g. 'M=8') so next round you commit directly.\n"
        "  - read_workboard: see other agents' commits and reflections this round.\n"
        "  - commit_action(action): submit an integer. Up to 3 attempts per round.\n"
        "    Response says whether your input was CLIPPED or accepted within cap.\n"
        "  - submit_reflection(content): OPTIONALLY publish a short reflection to the\n"
        "    shared workboard once you've discovered something useful (e.g. 'M=8').\n"
        "    The orchestrator may use this to plan next round.\n\n"

        "Suggested loop:\n"
        "  1. read_scratch — does it already say M=K for some K?\n"
        "     - If YES: commit_action(K). Done. Respond with K as final answer.\n"
        "     - If NO: commit_action(10). Response will say clipped to effective_x=Mᵢ.\n"
        "       Then write_scratch with 'M={effective_x}'. Respond with effective_x.\n"
        "  2. Final answer = the integer effective_x you ended up with (the env's clip output).\n"
    ),
    allowed_tools=None,
)
