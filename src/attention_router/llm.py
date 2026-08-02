"""Anthropic client construction and JSON-constrained completion.

Shared by the vision pass and the router because both need the same three
things: a client built from an environment variable, a completion that is
required to return parseable JSON, and bounded retries when it does not.

Nothing here knows about routing. Prompt content lives in `prompts/`.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

from attention_router import config


class LLMError(RuntimeError):
    """Raised when a call cannot be completed or parsed within the retry budget."""


class Usage:
    """Process-wide token tally so a run can report what it cost.

    Every retry and every vote counts, which is the point: the headline
    number should be what was actually spent, not what one successful call
    would have cost.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

    def estimated_usd(self) -> float:
        rate_in, rate_out = config.price_per_mtok(config.ROUTER_MODEL)
        return (
            self.input_tokens * rate_in + self.output_tokens * rate_out
        ) / 1_000_000

    def summary(self) -> str:
        return (
            f"{self.calls} model calls | "
            f"{self.input_tokens:,} in + {self.output_tokens:,} out tokens | "
            f"~${self.estimated_usd():.3f} on {config.ROUTER_MODEL}"
        )


USAGE = Usage()


def api_key() -> str:
    key = os.environ.get(config.ANTHROPIC_API_KEY_ENV, "").strip()
    if not key:
        raise LLMError(
            f"{config.ANTHROPIC_API_KEY_ENV} is not set. Export it or put it in a "
            f".env file at the repo root. Never commit the key."
        )
    return key


def build_client(provider: str | None = None):
    """Create a client for the given provider (defaults to the router's)."""
    provider = provider or config.ROUTER_PROVIDER
    if provider == "groq":
        return GroqClient()
    try:
        from anthropic import Anthropic
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise LLMError(
            "The 'anthropic' package is required. Run: "
            "pip install -r code/requirements.txt"
        ) from exc
    return Anthropic(api_key=api_key())


class GroqClient:
    """Minimal OpenAI-compatible chat client over the standard library.

    Groq speaks the OpenAI chat-completions shape, and the router only ever
    sends text, so a couple of dozen lines of urllib avoids pulling in a
    second SDK for one endpoint. Exposes the same `messages.create(...)`
    surface the Anthropic client does, so `complete_json` does not branch.
    """

    def __init__(self, model_env: str = "GROQ_API_KEY"):
        self.key = os.environ.get(model_env, "").strip()
        if not self.key:
            raise LLMError(
                f"{model_env} is not set. Export it or put it in a .env at the "
                f"repo root, or set ROUTER_PROVIDER=anthropic."
            )

    @property
    def messages(self):
        return self

    def create(self, *, model, max_tokens, system, messages, **ignored):
        """Send a chat completion and return an Anthropic-shaped response."""
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0,  # accepted here, unlike current Claude models
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}]
            + [
                {
                    "role": m["role"],
                    "content": m["content"]
                    if isinstance(m["content"], str)
                    else json.dumps(m["content"]),
                }
                for m in messages
            ],
        }
        request = urllib.request.Request(
            config.GROQ_CHAT_ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
                "User-Agent": "message-notification-router/1.0",
            },
        )
        body = self._send(request)
        text = body["choices"][0]["message"]["content"] or ""
        return _TextResponse(text, body.get("usage") or {})

    def _send(self, request, attempts: int = 6) -> dict:
        """POST with rate-limit backoff.

        Rate limiting is handled here rather than in `complete_json` on
        purpose: a 429 is a "wait and repeat", not a malformed answer, and
        letting it consume the JSON-repair budget is what silently pushed
        whole runs onto the heuristic fallback.
        """
        for attempt in range(1, attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable or attempt == attempts:
                    detail = exc.read().decode("utf-8", "replace")[:300]
                    raise LLMError(
                        f"Groq request failed ({exc.code}): {detail}"
                    ) from exc
                # Honour the server's own pacing when it offers one.
                header = exc.headers.get("retry-after") if exc.headers else None
                try:
                    delay = float(header) if header else 0.0
                except ValueError:
                    delay = 0.0
                time.sleep(max(delay, min(2**attempt, 30)))
            except urllib.error.URLError as exc:
                if attempt == attempts:
                    raise LLMError(f"Groq request failed: {exc}") from exc
                time.sleep(min(2**attempt, 30))
        raise LLMError("Groq request failed: retries exhausted")


class _TextBlock:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class _Usage:
    def __init__(self, raw: dict):
        self.input_tokens = raw.get("prompt_tokens", 0)
        self.output_tokens = raw.get("completion_tokens", 0)


class _TextResponse:
    """Adapts a chat-completions reply to the block shape callers expect."""

    def __init__(self, text: str, usage: dict | None = None):
        self.content = [_TextBlock(text)]
        self.usage = _Usage(usage or {})


# A model that wraps JSON in prose or a fenced block is still usable; a model
# that emits no object at all is not. Strip the former, fail on the latter.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> dict:
    """Parse the first JSON object in a completion."""
    candidate = (text or "").strip()
    fenced = _FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    for attempt in (candidate, _outermost_object(candidate)):
        if attempt is None:
            continue
        try:
            parsed = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        raise LLMError(f"Expected a JSON object, got {type(parsed).__name__}.")

    raise LLMError("Model returned no JSON object.")


def _outermost_object(text: str) -> str | None:
    """The span from the first '{' to the last '}', if both are present."""
    start = text.find("{")
    end = text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else None


def complete_json(
    client,
    *,
    system: str,
    content,
    model: str,
    max_tokens: int,
    validate=None,
    max_attempts: int = config.MAX_ROUTER_ATTEMPTS,
) -> dict:
    """Call the model until it returns JSON that `validate` accepts.

    `validate` raises ValueError to reject a payload; its message is fed back
    to the model as a repair instruction, so a retry is an informed correction
    rather than a blind resample.

    No sampling parameters are sent: Sonnet 5 and Opus 5 reject `temperature`,
    `top_p`, and `top_k` with a 400. Behaviour is steered by the prompt and by
    `effort` instead.
    """
    messages = [{"role": "user", "content": content}]
    last_error: Exception | None = None
    text = "{}"

    for attempt in range(1, max_attempts + 1):
        try:
            extra = (
                {}
                if isinstance(client, GroqClient)
                else {
                    "thinking": config.ROUTER_THINKING,
                    "output_config": {"effort": config.ROUTER_EFFORT},
                }
            )
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                **extra,
            )
            usage = getattr(response, "usage", None)
            if usage is not None:
                USAGE.record(
                    model,
                    getattr(usage, "input_tokens", 0) or 0,
                    getattr(usage, "output_tokens", 0) or 0,
                )
            text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            payload = extract_json(text)
            if validate is not None:
                validate(payload)
            return payload
        except (LLMError, ValueError) as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            messages = messages + [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": (
                        f"That response was rejected: {exc}\n"
                        f"Return only a single valid JSON object that fixes this. "
                        f"No prose, no code fences."
                    ),
                },
            ]
        except Exception as exc:  # transient API/network failure
            last_error = exc
            if attempt == max_attempts:
                break
            time.sleep(min(2**attempt, 8))

    raise LLMError(f"Failed after {max_attempts} attempts: {last_error}")
