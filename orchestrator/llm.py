import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class LLMResponse:
    content: str


@runtime_checkable
class LLM(Protocol):
    def complete(self, messages: list[dict]) -> LLMResponse: ...


_KNOWN_PROVIDER_PREFIXES = (
    "openai/", "anthropic/", "openrouter/", "azure/", "gemini/",
    "mistral/", "groq/", "together_ai/", "bedrock/", "vertex_ai/",
)


class LiteLLM:
    """Thin wrapper over litellm chat completions, configured via env vars.

    Reads (in priority order):
      * model:   LLM_MODEL | OPENROUTER_MODEL
      * api_key: OPENAI_API_KEY | OPENROUTER_API_KEY | LLM_API
      * base_url: OPENAI_API_BASE | OPENROUTER_BASE_URL
      * optional: LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_TIMEOUT

    For litellm routing, the model name is auto-prefixed with `openai/` when it
    has no provider prefix — this is correct for any OpenAI-compatible endpoint
    (OpenRouter, Azure, local proxies) once `api_base` is set.
    """

    def __init__(self, model: str | None = None, trace_path: str | Path | None = None, tag: str = ""):
        import litellm  # lazy
        # Silence litellm's "Provider List:" banner and debug-info spam.
        # This is a module-level setting; doing it idempotently on every init is fine.
        try:
            litellm.suppress_debug_info = True
        except Exception:  # noqa: BLE001 — attribute may move in future litellm versions
            pass
        self._litellm = litellm

        raw_model = model or os.getenv("LLM_MODEL") or os.getenv("OPENROUTER_MODEL") or "anthropic/claude-sonnet-4-5"
        self.model = raw_model if raw_model.startswith(_KNOWN_PROVIDER_PREFIXES) else f"openai/{raw_model}"

        self.api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("LLM_API")
            or ""
        )
        self.base_url = (
            os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENROUTER_BASE_URL")
            or "https://openrouter.ai/api/v1"
        )

        self.max_tokens = _int_env("LLM_MAX_TOKENS")
        self.temperature = _float_env("LLM_TEMPERATURE")
        self.timeout = _float_env("LLM_TIMEOUT")

        # Pass-through for vLLM / provider-specific kwargs. Common use:
        #   LLM_EXTRA_BODY='{"chat_template_kwargs":{"thinking":false}}'
        # to disable thinking mode on Kimi / MiniMax / other reasoning models.
        self.extra_body = _json_env("LLM_EXTRA_BODY")
        self.extra_headers = _json_env("LLM_EXTRA_HEADERS")

        # I/O trace: one JSON object per call appended to this file.
        # Set via constructor or LLM_TRACE_LOG env var. None = no logging.
        trace = trace_path if trace_path is not None else os.getenv("LLM_TRACE_LOG")
        self.trace_path = Path(trace) if trace else None
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.tag = tag

    def complete(self, messages: list[dict]) -> LLMResponse:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "api_key": self.api_key,
            "api_base": self.base_url,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.extra_body is not None:
            kwargs["extra_body"] = self.extra_body
        if self.extra_headers is not None:
            kwargs["extra_headers"] = self.extra_headers

        # Retry on transient errors (504, timeout, empty-response).
        # Up to 3 attempts total with exponential backoff (2s, 4s).
        # Tunable via LLM_MAX_RETRIES env var (default 2 extra retries).
        max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
        attempts = 0
        last_error: Exception | None = None
        while True:
            t0 = time.perf_counter()
            error: str | None = None
            content = ""
            try:
                resp = self._litellm.completion(**kwargs)
                content = resp["choices"][0]["message"]["content"] or ""
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {e}"
                last_error = e
            finally:
                if self.trace_path is not None:
                    _trace_append(self.trace_path, {
                        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                        "tag": self.tag,
                        "model": self.model,
                        "latency_ms": int((time.perf_counter() - t0) * 1000),
                        "messages": messages,
                        "response": content,
                        "error": error,
                        "attempt": attempts,
                    })

            # Success path: returned non-empty content.
            if error is None and content.strip():
                return LLMResponse(content=content)

            # Decide whether to retry.
            attempts += 1
            if attempts > max_retries:
                if last_error is not None:
                    raise last_error
                # Exhausted retries on empty content → return the empty string;
                # caller (strategy) already has uniform fallback.
                return LLMResponse(content=content)

            backoff = min(2.0 * (2 ** (attempts - 1)), 30.0)  # 2,4,8,16,30,30... capped
            print(
                f"LLM retry {attempts}/{max_retries} after {error or 'empty response'} (sleep {backoff}s)",
                file=__import__("sys").stderr,
            )
            time.sleep(backoff)


_TRACE_LOCK = threading.Lock()


def _trace_append(path: Path, record: dict) -> None:
    """Append one JSON record per line. Process-thread-safe via a module lock
    (file appends across processes still rely on OS atomicity for short writes)."""
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _TRACE_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _int_env(name: str) -> int | None:
    v = os.getenv(name)
    return int(v) if v not in (None, "") else None


def _float_env(name: str) -> float | None:
    v = os.getenv(name)
    return float(v) if v not in (None, "") else None


def _json_env(name: str) -> dict | None:
    """Parse a JSON-formatted env var. Returns None if unset; raises ValueError on
    malformed JSON (better to fail loudly than silently drop per-provider kwargs)."""
    v = os.getenv(name)
    if v in (None, ""):
        return None
    return json.loads(v)


class FakeLLM:
    """Scripted LLM for tests. Each `complete()` call pops the next scripted
    response. Raises if the script is exhausted."""

    def __init__(self, script: list[str]):
        self._script = list(script)
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict]) -> LLMResponse:
        self.calls.append(messages)
        if not self._script:
            raise AssertionError("FakeLLM script exhausted")
        return LLMResponse(content=self._script.pop(0))
