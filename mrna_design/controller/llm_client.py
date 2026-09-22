"""
LLM Client — A lightweight abstraction over OpenAI, Anthropic, and Gemini SDKs.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from mrna_design.logging_utils import get_logger

log = get_logger("controller.llm_client")


class LLMError(Exception):
    """Base exception for LLM API errors."""
    pass


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
        exceptions.extend([openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError])
    except ImportError:
        pass

    try:
        import anthropic
        exceptions.extend([anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError])
    except ImportError:
        pass

    try:
        from google.api_core import exceptions as google_exceptions
        exceptions.extend([google_exceptions.ResourceExhausted, google_exceptions.ServiceUnavailable, google_exceptions.InternalServerError])
    except ImportError:
        pass
    
    return tuple(exceptions) if exceptions else (Exception,)


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(_get_retry_exceptions()),
    reraise=True,
)
def chat(prompt: str, model: str = "gpt-4o", temperature: float = 0.1, json_mode: bool = True) -> str:
    """
    Send a prompt to the specified LLM and return the string response.
    Supports retries on rate limits and API errors.
    """
    log.event("llm_chat_request", model=model, temperature=temperature)

    if model.startswith("gpt-") or model.startswith("o1-"):
        return _chat_openai(prompt, model, temperature, json_mode)
    elif model.startswith("claude-"):
        return _chat_anthropic(prompt, model, temperature, json_mode)
    elif model.startswith("gemini-"):
        return _chat_gemini(prompt, model, temperature, json_mode)
    else:
        raise ValueError(f"Unsupported model: {model}")


def _chat_openai(prompt: str, model: str, temperature: float, json_mode: bool) -> str:
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
    return content or ""


def _chat_anthropic(prompt: str, model: str, temperature: float, json_mode: bool) -> str:
    client = _get_anthropic_client()
    
    # Anthropic expects '{"' prefix to enforce JSON structure in pre-fill, 
    # or just normal prompting. We use a system prompt for JSON constraint if needed.
    system_prompt = "You are an expert mRNA sequence designer. Always output valid JSON." if json_mode else ""
    
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Anthropic returns a list of ContentBlock objects
    text_blocks = [block.text for block in response.content if hasattr(block, "text")]
    return "".join(text_blocks)


def _chat_gemini(prompt: str, model: str, temperature: float, json_mode: bool) -> str:
    genai = _get_gemini_client()
    
    # Gemini 1.5 Pro and Flash support JSON schema, but we'll use standard text generation 
    # with strong prompt instructions for simplicity unless we specifically pass the schema.
    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        response_mime_type="application/json" if json_mode else "text/plain",
    )
    
    gemini_model = genai.GenerativeModel(model)
    response = gemini_model.generate_content(prompt, generation_config=generation_config)
    return response.text


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
        raise LLMError(f"Failed to parse JSON from response: {str(e)}")
