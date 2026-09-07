import json
import os
import re
from typing import Any

from anthropic import Anthropic

try:
    from groq import Groq
except ImportError:  # pragma: no cover - handled at runtime if Groq is not installed yet
    Groq = None


def safe_json_loads(text: str, default_fallback: Any = None) -> Any:
    if not text:
        return default_fallback if default_fallback is not None else {}

    cleaned = text.strip()

    # Remove markdown code block wrappers if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start_brace = cleaned.find("{")
        end_brace = cleaned.rfind("}")
        start_bracket = cleaned.find("[")
        end_bracket = cleaned.rfind("]")

        has_brace = start_brace != -1 and end_brace != -1 and end_brace > start_brace
        has_bracket = start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket

        if has_brace and (not has_bracket or start_brace < start_bracket):
            try:
                return json.loads(cleaned[start_brace:end_brace+1])
            except json.JSONDecodeError:
                pass
            if has_bracket:
                try:
                    return json.loads(cleaned[start_bracket:end_bracket+1])
                except json.JSONDecodeError:
                    pass
        elif has_bracket:
            try:
                return json.loads(cleaned[start_bracket:end_bracket+1])
            except json.JSONDecodeError:
                pass
            if has_brace:
                try:
                    return json.loads(cleaned[start_brace:end_brace+1])
                except json.JSONDecodeError:
                    pass

        return default_fallback if default_fallback is not None else {}



GROQ_MODEL_ALIASES = {
    "llama-3.1-8b-instant": "openai/gpt-oss-120b",
    "llama-3.1-70b-versatile": "openai/gpt-oss-120b",
    "llama3-70b-8192": "openai/gpt-oss-120b",
    "llama3-8b-8192": "openai/gpt-oss-20b",
    "llama-3.1-8b": "openai/gpt-oss-20b",
    "llama-3.1-70b": "openai/gpt-oss-120b",
    "llama-3.3-70b": "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile": "openai/gpt-oss-120b",
    "mixtral-8x7b-32768": "openai/gpt-oss-120b",
}

ANTHROPIC_MODEL_ALIASES = {
    "claude-sonnet-4-20250514": "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-haiku": "claude-3-5-haiku-20241022",
}


def _provider_name() -> str:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider:
        return provider
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    raise RuntimeError("Set LLM_PROVIDER and the matching API key, or provide GROQ_API_KEY/ANTHROPIC_API_KEY.")


def _default_model(provider: str) -> str:
    if provider == "groq":
        raw = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
        return GROQ_MODEL_ALIASES.get(raw, raw)
    if provider == "anthropic":
        raw = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip()
        return ANTHROPIC_MODEL_ALIASES.get(raw, raw)
    raise ValueError(f"Unsupported LLM provider: {provider}")


def generate_text(system_prompt: str, user_prompt: str, *, model: str | None = None, max_tokens: int = 1000, temperature: float = 0.2, json_mode: bool = False) -> str:
    provider = _provider_name()
    raw_model = model or _default_model(provider)

    if provider == "groq":
        if Groq is None:
            raise RuntimeError("Groq provider selected but the groq package is not installed.")
        
        model_name = GROQ_MODEL_ALIASES.get(raw_model, raw_model)
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        try:
            response = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"} if json_mode else None,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # If model is not found or decommissioned, dynamically query active models on the user's Groq account!
            err_str = str(e)
            if "model_not_found" in err_str or "404" in err_str or "does not exist" in err_str:
                try:
                    models_resp = client.models.list()
                    active_models = [m.id for m in models_resp.data if not any(x in m.id.lower() for x in ["whisper", "guard", "embed", "tts", "stt"])]
                    for fallback_model in active_models:
                        try:
                            response = client.chat.completions.create(
                                model=fallback_model,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                response_format={"type": "json_object"} if json_mode else None,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt},
                                ],
                            )
                            return response.choices[0].message.content.strip()
                        except Exception:
                            continue
                except Exception:
                    pass
            raise RuntimeError(f"Groq API error ({model_name}): {e}") from e


    if provider == "anthropic":
        model_name = ANTHROPIC_MODEL_ALIASES.get(raw_model, raw_model)
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        try:
            message = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return message.content[0].text.strip()
        except Exception as e:
            raise RuntimeError(f"Anthropic API error ({model_name}): {e}") from e

    raise ValueError(f"Unsupported LLM provider: {provider}")

