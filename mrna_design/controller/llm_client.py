"""
LLM Client — A lightweight abstraction over OpenAI, Anthropic, and Gemini SDKs.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from mrna_design.logging_utils import get_logger

log = get_logger("controller.llm_client")


class LLMError(Exception):
    """Base exception for LLM API errors."""

    pass


class LLMUnavailable(LLMError):
    """
    The provider cannot be reached at all: SDK not installed, no API key, or
    an unrecognised model name.

    Distinguished from a transient API error because retrying will not help,
    and because a benchmark should degrade to "this controller proposed
    nothing" rather than aborting a multi-hour sweep.
    """


@dataclass
class ChatResponse:
    """An LLM reply plus the token accounting needed to price the run."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# Optional imports — we only load the SDK required for the chosen model.
_OPENAI_CLIENT = None
_ANTHROPIC_CLIENT = None
_GEMINI_CLIENT = None


def _get_openai_client() -> Any:
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        import openai

        _OPENAI_CLIENT = openai.Client()
    return _OPENAI_CLIENT


def _get_anthropic_client() -> Any:
    global _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT is None:
        import anthropic

        _ANTHROPIC_CLIENT = anthropic.Anthropic()
    return _ANTHROPIC_CLIENT


def _get_gemini_client() -> Any:
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        import google.generativeai as genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required.")
        genai.configure(api_key=api_key)
        _GEMINI_CLIENT = genai
    return _GEMINI_CLIENT


# Exception types to retry on
def _get_retry_exceptions() -> tuple:
    exceptions = []
    try:
        import openai

        exceptions.extend(
            [openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError]
        )
    except ImportError:
        pass

    try:
        import anthropic

        exceptions.extend(
            [anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError]
        )
    except ImportError:
        pass

    try:
        from google.api_core import exceptions as google_exceptions

        exceptions.extend(
            [
                google_exceptions.ResourceExhausted,
                google_exceptions.ServiceUnavailable,
                google_exceptions.InternalServerError,
            ]
        )
    except ImportError:
        pass

    return tuple(exceptions) if exceptions else (Exception,)


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(_get_retry_exceptions()),
    reraise=True,
)
def chat_with_usage(
    prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0.1,
    json_mode: bool = True,
) -> ChatResponse:
    """
    Send a prompt to the specified LLM and return the reply with token usage.

    Token counts matter here beyond cost: the research question compares
    controllers "under the same tool-call budget", and an LLM controller that
    reaches a better front while consuming an order of magnitude more tokens has
    not won the comparison it claims to. Usage is recorded per call and
    aggregated into the run manifest.

    Raises
    ------
    LLMUnavailable
        The provider cannot be used at all (SDK missing, no credentials,
        unknown model). Not retried — retrying cannot fix it.
    """
    log.event("llm_chat_request", model=model, temperature=temperature)

    try:
        if model.startswith("gpt-") or model.startswith("o1-") or model.startswith("o3-"):
            return _chat_openai(prompt, model, temperature, json_mode)
        if model.startswith("claude-"):
            return _chat_anthropic(prompt, model, temperature, json_mode)
        if model.startswith("gemini-"):
            return _chat_gemini(prompt, model, temperature, json_mode)
    except ImportError as exc:
        raise LLMUnavailable(
            f"SDK for model {model!r} is not installed: {exc}. "
            f"Install with: pip install -e '.[llm]'"
        ) from exc
    except Exception as exc:
        # Authentication and configuration failures are permanent for this run.
        name = type(exc).__name__
        if any(k in name for k in ("Authentication", "PermissionDenied", "NotFound")):
            raise LLMUnavailable(f"{name} calling {model!r}: {exc}") from exc
        raise

    raise LLMUnavailable(
        f"Unsupported model: {model!r}. Expected a gpt-*, o1-*, o3-*, claude-* "
        f"or gemini-* model name."
    )


def chat(
    prompt: str, model: str = "gpt-4o", temperature: float = 0.1, json_mode: bool = True
) -> str:
    """Backwards-compatible wrapper returning just the reply text."""
    return chat_with_usage(prompt, model, temperature, json_mode).text


def _chat_openai(prompt: str, model: str, temperature: float, json_mode: bool) -> ChatResponse:
    client = _get_openai_client()
    kwargs = {
        "model": model,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }

    # Optional strict JSON mode (not supported by all OpenAI models, e.g., o1)
    if json_mode and not model.startswith("o1-"):
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    usage = getattr(response, "usage", None)
    return ChatResponse(
        text=content or "",
        model=model,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )


def _chat_anthropic(prompt: str, model: str, temperature: float, json_mode: bool) -> ChatResponse:
    client = _get_anthropic_client()

    # Anthropic expects '{"' prefix to enforce JSON structure in pre-fill,
    # or just normal prompting. We use a system prompt for JSON constraint if needed.
    system_prompt = (
        "You are an expert mRNA sequence designer. Always output valid JSON." if json_mode else ""
    )

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
    )

    # Anthropic returns a list of ContentBlock objects
    text_blocks = [block.text for block in response.content if hasattr(block, "text")]
    usage = getattr(response, "usage", None)
    return ChatResponse(
        text="".join(text_blocks),
        model=model,
        prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
        completion_tokens=getattr(usage, "output_tokens", 0) or 0,
    )


def _chat_gemini(prompt: str, model: str, temperature: float, json_mode: bool) -> ChatResponse:
    genai = _get_gemini_client()

    # Gemini 1.5 Pro and Flash support JSON schema, but we'll use standard text generation
    # with strong prompt instructions for simplicity unless we specifically pass the schema.
    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        response_mime_type="application/json" if json_mode else "text/plain",
    )

    gemini_model = genai.GenerativeModel(model)
    response = gemini_model.generate_content(prompt, generation_config=generation_config)
    usage = getattr(response, "usage_metadata", None)
    return ChatResponse(
        text=response.text,
        model=model,
        prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
        completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
    )


def extract_json_from_response(response_text: str) -> list | dict:
    """
    Attempt to parse JSON from the LLM response.
    Handles markdown code blocks (e.g., ```json ... ```).
    """
    text = response_text.strip()

    # Remove markdown code blocks if present
    match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.error("json_decode_error", error=str(e), text=response_text)
        raise LLMError(f"Failed to parse JSON from response: {e}") from e
