"""Hook protocol + reference implementations for ReactAgent.

Hooks let users splice arbitrary read/write logic around each LLM call without
modifying the agent core. Pre-hooks see (messages, step_info) and return possibly-
modified messages. Post-hooks see a fully-populated step_info and return None
(side-effect only).
"""
import sys
from dataclasses import dataclass
from typing import Any, Callable

from memory.base import MemoryPort


@dataclass
class StepInfo:
    """Snapshot of one ReAct step. Pre-hook receives this with raw_response /
    parsed_action / observation = None. Post-hook receives a populated copy."""
    step_idx: int
    raw_response: str | None = None
    parsed_action: dict | None = None
    observation: str | None = None


PreStepHook = Callable[[list[dict], StepInfo], list[dict]]
PostStepHook = Callable[[StepInfo], None]


def inject_workboard_on_start(workboard: MemoryPort) -> PreStepHook:
    """Pre-hook: on step_idx == 0 only, append a 'Current workboard:' block to the
    system message. Other steps pass through. Cheap, non-invasive context bootstrap."""

    def hook(messages: list[dict], info: StepInfo) -> list[dict]:
        if info.step_idx != 0:
            return messages
        text = workboard.read().strip()
        if not text:
            return messages
        out = [dict(m) for m in messages]
        if out and out[0]["role"] == "system":
            out[0]["content"] = out[0]["content"] + f"\n\nCurrent workboard:\n{text}"
        else:
            out.insert(0, {"role": "system", "content": f"Current workboard:\n{text}"})
        return out

    return hook


_WB_LIVE_MARKER = "\n\nCurrent workboard (live, re-read each step):\n"


def inject_workboard_every_step(workboard: MemoryPort) -> PreStepHook:
    """Pre-hook: every step, refresh the system message with the LATEST workboard.

    Required for genuine parallel coordination — workers running concurrently must
    see each other's in-flight reflections, not just a start-of-round snapshot.

    Idempotent: strips any previously-injected block (split on the marker) before
    re-appending, so the system message does not accumulate a new workboard copy
    every step. Pays token cost continuously by design."""

    def hook(messages: list[dict], info: StepInfo) -> list[dict]:
        text = workboard.read().strip()
        out = [dict(m) for m in messages]
        if not (out and out[0]["role"] == "system"):
            if text:
                out.insert(0, {"role": "system", "content": _WB_LIVE_MARKER.lstrip() + text})
            return out
        base = out[0]["content"].split(_WB_LIVE_MARKER)[0]  # drop prior injection
        out[0]["content"] = base + (_WB_LIVE_MARKER + text if text else "")
        return out

    return hook


def record_step_to_scratch(scratch: MemoryPort) -> PostStepHook:
    """Post-hook: append a one-line summary of every step to the worker's scratch.
    Useful for debug/trace, not used by GS by default."""

    def hook(info: StepInfo) -> None:
        tool = (info.parsed_action or {}).get("tool")
        obs = (info.observation or "").splitlines()[0][:120] if info.observation else ""
        scratch.append(f"step {info.step_idx}", f"tool={tool} obs={obs}")

    return hook


def commit_pressure(
    max_steps: int,
    edit_tool: str = "edit_file",
    reflect_tool: str = "submit_reflection",
    src_hint: str = "the library source (a .py under the package dir, NOT under tests/)",
    require_edit: bool = True,
) -> tuple[PreStepHook, PostStepHook]:
    """Escalating "stop exploring, commit an edit" pressure.

    Diagnosed failure mode (SWE-bench trace): the worker reads/searches for the
    entire step budget and never calls edit_file — 12/12 worker runs exhausted
    max_steps with ~1 edit attempt and 0 reflections. The role prompt already
    says "edit early"; the model ignores static instructions. This adds a RUNTIME
    forcing function.

    The pre-hook appends a forceful reminder to the most-recent observation at
    1/3, 2/3, and the final stretch of the budget — but ONLY while no source edit
    has landed. The paired post-hook watches tool observations and flips the
    ``edited`` flag on the first successful non-test edit_file, so a worker that
    has already committed is never nagged. Appending to the last user turn (not a
    new turn) keeps the nudge recent and avoids consecutive user messages.

    Returns ``(pre_hook, post_hook)``; register BOTH on the same ReactAgent via
    ``pre_step_hooks=[..., pre]`` and ``post_step_hooks=[post]``.
    """
    state = {"edited": not require_edit, "reflected": False}
    fire_steps = {
        max(1, max_steps // 3),
        max(1, (2 * max_steps) // 3),
        max(1, max_steps - 4),
    }

    def pre(messages: list[dict], info: StepInfo) -> list[dict]:
        # Nag until BOTH a source edit AND a workboard reflection have landed —
        # the reflection is how each agent "puts its result into shared memory",
        # so an edit alone is not enough (edit and reflect are gated separately).
        # require_edit=False (reproducer/locator roles): only the reflection is forced.
        if (state["edited"] and state["reflected"]) or info.step_idx not in fire_steps:
            return messages
        remaining = max_steps - info.step_idx
        tag = "FINAL WARNING" if remaining <= 5 else "REMINDER"
        if not state["edited"]:
            lead = ("you have NOT edited any source file yet. Reading and searching "
                    "do NOT fix the bug")
            ask = (f"call edit_file NOW on {src_hint} to change the buggy behaviour, "
                   f"then submit_reflection naming the file+function you changed")
            fail = "An episode with no source edit FAILS."
        elif require_edit:  # edited but not yet published to the shared workboard
            lead = ("you edited source but have NOT recorded it to the shared "
                    "workboard yet — your teammates cannot build on invisible work")
            ask = ("call submit_reflection NOW with the source file+function you "
                   "changed, the root cause, and what still remains")
            fail = "An episode with no source edit FAILS."
        else:  # reflect-only role (reproducer/locator): findings are the deliverable
            lead = ("you have NOT published your findings to the shared workboard "
                    "yet — your teammates cannot build on invisible work")
            ask = ("call submit_reflection NOW with your concrete findings so far "
                   "(exact commands, tracebacks, file+function candidates)")
            fail = "Unpublished findings are lost when you stop."
        note = (
            f"[{tag}] Step {info.step_idx + 1}/{max_steps} and {lead}. STOP exploring: "
            f"{ask}. {fail}"
        )
        out = [dict(m) for m in messages]
        if out and out[-1].get("role") == "user":
            out[-1]["content"] = out[-1]["content"] + "\n\n" + note
        else:
            out.append({"role": "user", "content": note})
        return out

    def post(info: StepInfo) -> None:
        a = info.parsed_action or {}
        if not isinstance(a, dict):
            return
        tool = a.get("tool")
        obs = info.observation or ""
        if tool == edit_tool and "edited " in obs:
            path = str((a.get("args") or {}).get("path", "")).lower()
            if "test" not in path:  # a source edit, not a test edit
                state["edited"] = True
        elif tool == reflect_tool and "published" in obs:
            state["reflected"] = True

    return pre, post


def safe_call_pre_hook(
    hook: PreStepHook | None, messages: list[dict], info: StepInfo
) -> list[dict]:
    """Call a pre-hook. On exception, log to stderr and return the original messages."""
    if hook is None:
        return messages
    try:
        return hook(messages, info)
    except Exception as e:  # noqa: BLE001
        print(f"warning: pre_step_hook raised {type(e).__name__}: {e}", file=sys.stderr)
        return messages


def safe_call_post_hook(hook: PostStepHook | None, info: StepInfo) -> None:
    """Call a post-hook. On exception, log to stderr and continue."""
    if hook is None:
        return
    try:
        hook(info)
    except Exception as e:  # noqa: BLE001
        print(f"warning: post_step_hook raised {type(e).__name__}: {e}", file=sys.stderr)


# ── Retrieval helpers ────────────────────────────────────────────────────

import re as _re


def _split_md_sections(text: str) -> list[str]:
    """Split a markdown text on H2 headers (`## <name>`), returning each section
    as a single string INCLUDING its header. Empty sections are skipped."""
    parts = _re.split(r"(?m)^(?=## )", text)
    return [p.strip() for p in parts if p.strip()]


def _default_query_fn(messages: list[dict], info: "StepInfo") -> str:
    """Return the content of the last user message, or falling back to the last
    message of any role. Used as the default query for retrieval hooks."""
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    if messages:
        return messages[-1].get("content", "")
    return ""


# ── Retrieval hook factories ─────────────────────────────────────────────

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from worker.retrieval import Retriever


def inject_workboard_top_k(
    workboard: MemoryPort,
    retriever: "Retriever",
    k: int = 5,
    query_fn: Callable[[list[dict], StepInfo], str] | None = None,
) -> PreStepHook:
    """Pre-hook: read workboard, split by H2 section, retrieve top-k most
    relevant to the query, inject as a 'Relevant peer commits:' block into the
    system message. Default query_fn = last user message content."""
    _qfn = query_fn or _default_query_fn

    def hook(messages: list[dict], info: StepInfo) -> list[dict]:
        text = workboard.read().strip()
        if not text:
            return messages
        sections = _split_md_sections(text)
        if not sections:
            return messages
        query = _qfn(messages, info)
        results = retriever.retrieve(query=query, corpus=sections, k=k)
        if not results:
            return messages
        top_sections = [sections[idx] for idx, _ in results]
        block = "Relevant peer commits:\n" + "\n".join(top_sections)
        out = [dict(m) for m in messages]
        if out and out[0]["role"] == "system":
            out[0]["content"] = out[0]["content"] + f"\n\n{block}"
        else:
            out.insert(0, {"role": "system", "content": block})
        return out

    return hook


def inject_scratch_top_k(
    scratch: MemoryPort,
    retriever: "Retriever",
    k: int = 3,
    query_fn: Callable[[list[dict], StepInfo], str] | None = None,
) -> PreStepHook:
    """Pre-hook: read agent scratch, split by H2 section, retrieve top-k most
    relevant to the query, inject as a 'Relevant notes:' block into the system
    message. Same shape as inject_workboard_top_k but targets per-agent memory."""
    _qfn = query_fn or _default_query_fn

    def hook(messages: list[dict], info: StepInfo) -> list[dict]:
        text = scratch.read().strip()
        if not text:
            return messages
        sections = _split_md_sections(text)
        if not sections:
            return messages
        query = _qfn(messages, info)
        results = retriever.retrieve(query=query, corpus=sections, k=k)
        if not results:
            return messages
        top_sections = [sections[idx] for idx, _ in results]
        block = "Relevant notes:\n" + "\n".join(top_sections)
        out = [dict(m) for m in messages]
        if out and out[0]["role"] == "system":
            out[0]["content"] = out[0]["content"] + f"\n\n{block}"
        else:
            out.insert(0, {"role": "system", "content": block})
        return out

    return hook
