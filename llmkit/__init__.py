"""Small client for an OpenAI-compatible LiteLLM gateway.

Gateway URL and default model come from ``LITELLM_URL`` and ``LITELLM_MODEL``.
They are read on every call, so apps and tests can set them late.
"""
import json
import os
import re
from typing import Any, Callable

import requests

__all__ = ["LLMError", "chat", "chat_json"]

# An unclosed block (length limit hit while thinking) runs to the end of the text.
_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


class LLMError(RuntimeError):
    """The gateway gave no usable answer."""


def _settings(model: str | None) -> tuple[str, str]:
    """Gateway URL and model, from the arguments or the environment."""
    url = os.environ.get("LITELLM_URL", "").rstrip("/")
    model = model or os.environ.get("LITELLM_MODEL")
    if not url or not model:
        raise LLMError("LITELLM_URL and LITELLM_MODEL (or model=) must be set")
    return url, model


def _messages(prompt: str, system: str | None, images: list[str] | None) -> list[dict]:
    """Chat messages for one prompt; images are base64 JPEGs sent as content parts."""
    content: str | list = prompt
    if images:
        content = [{"type": "text", "text": prompt}]
        content += [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b}}
                    for b in images]
    head = [{"role": "system", "content": system}] if system else []
    return head + [{"role": "user", "content": content}]


def _complete(messages: list[dict], model: str | None, json_mode: bool,
              temperature: float, timeout: float, max_tokens: int | None) -> tuple[str, str]:
    """One completion; returns (answer, reasoning) with thinking blocks removed."""
    url, model = _settings(model)
    body = {"model": model, "messages": messages, "temperature": temperature}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if max_tokens:
        body["max_tokens"] = max_tokens
    try:
        resp = requests.post(f"{url}/v1/chat/completions", json=body, timeout=timeout)
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
    except (requests.RequestException, ValueError, LookupError, TypeError) as e:
        raise LLMError(f"gateway call failed: {e}") from e
    answer = _THINK.sub("", msg.get("content") or "").strip()
    # Providers name the thinking channel differently (Groq: reasoning).
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return answer, reasoning


def _repair(text: str) -> str:
    """Fix the slips models make most: trailing commas and empty values."""
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return re.sub(r":\s*(?=[,}])", ": null", text)


def _loads(text: str):
    """json.loads with NaN and Infinity read as null instead of floats."""
    return json.loads(text, parse_constant=lambda _: None)


def _parse_object(text: str) -> dict:
    """The JSON object in a reply, tolerating code fences, talk around it and small slips."""
    text = _FENCE.sub("", _THINK.sub("", text).strip())
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    try:
        value = _loads(text)
    except json.JSONDecodeError as e:
        # The repair only runs on text that is broken anyway, so it cannot
        # alter a valid reply. If it does not help, report the original error.
        try:
            value = _loads(_repair(text))
        except json.JSONDecodeError:
            raise e from None
    if not isinstance(value, dict):
        raise json.JSONDecodeError("expected a JSON object", text, 0)
    return value


def chat(prompt: str, system: str | None = None, *, model: str | None = None,
         json_mode: bool = False, images: list[str] | None = None,
         temperature: float = 0.2, timeout: float = 90, max_tokens: int | None = None) -> str:
    """Send one prompt and return the answer text."""
    answer, _ = _complete(_messages(prompt, system, images), model, json_mode,
                          temperature, timeout, max_tokens)
    if not answer:
        raise LLMError("empty response")
    return answer


def chat_json(prompt: str, system: str | None = None, *, model: str | None = None,
              images: list[str] | None = None, temperature: float = 0.2,
              timeout: float = 90, max_tokens: int | None = None, retries: int = 1,
              check: Callable[[dict], Any] | None = None) -> Any:
    """Send one prompt in JSON mode and return the parsed object.

    ``check`` gets the parsed object and returns what the caller receives
    instead (e.g. a Pydantic model's ``model_validate``). An unparsable reply
    or a ``ValueError`` from ``check`` is answered with the error and asked
    again, up to ``retries`` times. A thinking model that leaves the answer
    empty and puts the JSON into its reasoning channel is read from there.
    """
    messages = _messages(prompt, system, images)
    for attempt in range(retries + 1):
        answer, reasoning = _complete(messages, model, True, temperature, timeout, max_tokens)
        reply = answer or reasoning
        try:
            value = _parse_object(reply)
            return check(value) if check else value
        except ValueError as e:  # JSONDecodeError and Pydantic's ValidationError included
            error = e
        messages = messages + [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": f"Your reply was not usable ({error}). "
                                        "Reply with the JSON object only."},
        ]
    raise LLMError(f"no usable JSON after {retries + 1} attempts: {error}")
