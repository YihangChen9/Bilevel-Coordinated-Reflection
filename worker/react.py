import json
from typing import Any

from orchestrator.llm import LLM
from worker.base import AgentContext, AgentResult
from worker.hooks import (
    PreStepHook, PostStepHook, StepInfo,
    safe_call_pre_hook, safe_call_post_hook,
)
from worker.tool import Tool


_SYSTEM_TEMPLATE = """\
{role_prompt}

You have these tools:
{tool_list}

Each turn output EXACTLY ONE JSON object, no prose outside it.

To call a tool:
{{"thought": "short reasoning", "action": {{"tool": "<name>", "args": {{...}}}}}}

To finish:
{{"thought": "short reasoning", "final": "your answer"}}

Rules:
- Only one tool call per turn.
- Arguments must match the tool's input_schema.
- When you have enough information, respond with "final".
"""


def _format_tool_list(tools: list[Tool]) -> str:
    lines = []
    for t in tools:
        schema = json.dumps(t.input_schema, ensure_ascii=False)
        lines.append(f"- {t.name}: {t.description}\n  schema: {schema}")
    return "\n".join(lines)


def _safe_json_loads(text: str) -> tuple[Any, str | None]:
    """Try to parse JSON from text, tolerating code fences and leading prose."""
    text = text.strip()

    # 1. Strip ```...``` or ```json...``` code fences.
    if text.startswith("```"):
        # Find the closing fence
        end = text.rfind("```", 3)
        if end != -1:
            # Strip the opening line (e.g. "```json")
            inner = text[3:end]
            newline = inner.find("\n")
            if newline != -1:
                inner = inner[newline + 1:]
            text = inner.strip()
    else:
        # 2. Brace-matching fallback: extract first balanced {...} substring,
        #    skipping content inside string literals so braces inside strings
        #    do not count.
        start = text.find("{")
        if start != -1:
            depth = 0
            in_string = False
            escape_next = False
            end_idx = -1
            for i in range(start, len(text)):
                ch = text[i]
                if escape_next:
                    escape_next = False
                    continue
                if ch == "\\" and in_string:
                    escape_next = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end_idx = i
                        break
            if end_idx != -1:
                text = text[start:end_idx + 1]
            else:
                # depth never returned to 0 → the object is unclosed, almost always
                # because the response hit the token limit mid-emit. Give the model
                # an actionable error instead of a confusing "char 0".
                return None, (
                    "JSON object not closed (response likely truncated at the token "
                    "limit). Emit ONE small, complete action; for large file content "
                    "make several small edit_file calls instead of one big one."
                )

    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"


def _normalize_hook_list(name: str, singular, plural) -> list:
    """Normalize (singular | None, plural | None) into a single list. Either form
    is accepted, never both. Empty list means 'no hooks'."""
    if singular is not None and plural is not None:
        raise ValueError(
            f"{name}_hook and {name}_hooks are mutually exclusive; pass only one"
        )
    if plural is not None:
        return list(plural)
    if singular is not None:
        return [singular]
    return []


class ReactAgent:
    def __init__(
        self,
        llm: LLM,
        max_steps: int = 20,
        pre_step_hook: PreStepHook | None = None,
        post_step_hook: PostStepHook | None = None,
        pre_step_hooks: list[PreStepHook] | None = None,
        post_step_hooks: list[PostStepHook] | None = None,
    ):
        self.llm = llm
        self.max_steps = max_steps
        self._pre_hooks = _normalize_hook_list("pre_step", pre_step_hook, pre_step_hooks)
        self._post_hooks = _normalize_hook_list("post_step", post_step_hook, post_step_hooks)

    def run(self, task: str, tools: list[Tool], ctx: AgentContext) -> AgentResult:
        tool_map = {t.name: t for t in tools}
        system = _SYSTEM_TEMPLATE.format(
            role_prompt=ctx.system_prompt,
            tool_list=_format_tool_list(tools),
        )
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Task: {task}"},
        ]
        steps: list[dict] = []

        for step in range(self.max_steps):
            pre_info = StepInfo(step_idx=step)
            for hook in self._pre_hooks:
                messages = safe_call_pre_hook(hook, messages, pre_info)

            resp = self.llm.complete(messages)
            raw = resp.content.strip()
            parsed, err = _safe_json_loads(raw)

            if err is not None:
                observation = f"error: could not parse JSON ({err}). Re-emit valid JSON."
                steps.append({"thought": None, "action": None, "observation": observation, "raw": raw})
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": observation})
                post_info = StepInfo(step_idx=step, raw_response=raw, parsed_action=None, observation=observation)
                for hook in self._post_hooks:
                    safe_call_post_hook(hook, post_info)
                continue

            # Fix 3: guard against non-dict top-level JSON
            if not isinstance(parsed, dict):
                observation = "error: top-level JSON must be an object, got {}".format(type(parsed).__name__)
                steps.append({"thought": None, "action": None, "observation": observation, "raw": raw})
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": observation})
                post_info = StepInfo(step_idx=step, raw_response=raw, parsed_action=None, observation=observation)
                for hook in self._post_hooks:
                    safe_call_post_hook(hook, post_info)
                continue

            thought = parsed.get("thought")

            if "final" in parsed:
                # Fix 2: include raw on final step
                steps.append({"thought": thought, "action": None, "observation": None, "final": parsed["final"], "raw": raw})
                post_info = StepInfo(step_idx=step, raw_response=raw, parsed_action=None, observation=None)
                for hook in self._post_hooks:
                    safe_call_post_hook(hook, post_info)
                return AgentResult(final_answer=str(parsed["final"]), steps=steps, stopped_reason="final")

            action = parsed.get("action") or {}
            if not isinstance(action, dict):
                observation = ("error: 'action' must be an object like "
                               '{"tool": "<name>", "args": {...}}, got '
                               f"{type(action).__name__}. Re-emit valid JSON.")
                steps.append({"thought": thought, "action": None, "observation": observation, "raw": raw})
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": observation})
                post_info = StepInfo(step_idx=step, raw_response=raw, parsed_action=None, observation=observation)
                for hook in self._post_hooks:
                    safe_call_post_hook(hook, post_info)
                continue
            tool_name = action.get("tool")
            args = action.get("args") or {}
            if not isinstance(args, dict):
                observation = (f"error: 'args' must be an object, got {type(args).__name__}. "
                               "Re-emit with args as a JSON object.")
                steps.append({"thought": thought, "action": action, "observation": observation, "raw": raw})
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": observation})
                post_info = StepInfo(step_idx=step, raw_response=raw, parsed_action=action, observation=observation)
                for hook in self._post_hooks:
                    safe_call_post_hook(hook, post_info)
                continue

            if tool_name not in tool_map:
                observation = f"error: unknown tool '{tool_name}'. Available: {list(tool_map)}"
            else:
                try:
                    result = tool_map[tool_name].call(**args)
                    observation = json.dumps(result, ensure_ascii=False, default=str)
                except Exception as e:
                    observation = f"error: tool '{tool_name}' raised {type(e).__name__}: {e}"

            # Fix 2: include raw on tool step
            steps.append({"thought": thought, "action": action, "observation": observation, "raw": raw})
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
            post_info = StepInfo(step_idx=step, raw_response=raw, parsed_action=action, observation=observation)
            for hook in self._post_hooks:
                safe_call_post_hook(hook, post_info)

        return AgentResult(
            final_answer="",
            steps=steps,
            stopped_reason="max_steps",
        )
